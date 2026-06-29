"""Windowed dataset for temporal accident detection.

A *window* is `num_frames` consecutive frames of a pre-extracted strip (a causal
1s clip). Its label depends on where the window's END time falls relative to
`time_of_event` (see config `window`). Sliding windows over a strip = the
temporal scan the model performs at inference time.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ----------------------------------------------------------------------
# Model input assembly (RGB, optionally + temporal-difference channels)
# ----------------------------------------------------------------------
# Motion mode feeds the per-frame 2D backbone explicit inter-frame motion (the
# Rockchip models can't use 3D convs to learn it). Each lag k adds 3 channels:
# (RGB_t - RGB_{t-k}), all referencing the current frame so each frame stays
# independently computable (deploy splits per-frame onto the NPU). With lags
# [1,2,4] the input is 3 (RGB) + 3*3 = 12 channels of multi-scale motion.
MOTION_SCALE = 0.5                       # std-ish scale for the diff channels


def motion_enabled(cfg) -> bool:
    return bool(cfg.input.get("motion", False))


def motion_lags(cfg) -> list[int]:
    """Temporal-difference lags (frames). [] when motion is off; default [1]."""
    if not motion_enabled(cfg):
        return []
    return [int(k) for k in cfg.input.get("motion_lags", [1])]


def in_channels(cfg) -> int:
    return 3 + 3 * len(motion_lags(cfg))


def assemble_input(rgb01, mean, std, lags):
    """[3,L,H,W] float 0-1 -> [3 + 3*len(lags), L,H,W] normalised model input."""
    norm = (rgb01 - mean) / std
    if not lags:
        return norm
    chans = [norm]
    for k in lags:
        diff = torch.zeros_like(rgb01)
        diff[:, k:] = (rgb01[:, k:] - rgb01[:, :-k]) / MOTION_SCALE   # causal: diff[<k]=0
        chans.append(diff)
    return torch.cat(chans, dim=0)


# ----------------------------------------------------------------------
# Window labeling
# ----------------------------------------------------------------------
def window_end_time(start_idx, L, s0, fps):
    """Source time of the last frame of a window starting at strip index start_idx."""
    return s0 + (start_idx + L - 1) / fps


def window_label(end_t, event_t, w):
    """Return 1 (accident), 0 (normal), or -1 (ignore) for a window end time."""
    if event_t is None or (isinstance(event_t, float) and np.isnan(event_t)):
        return 0
    if event_t - w["pos_pre"] <= end_t <= event_t + w["pos_post"]:
        return 1
    if event_t + w["pos_post"] < end_t <= event_t + w["ignore_post"]:
        return -1
    return 0


def enumerate_windows(meta_row, cfg, drop_ignore=True):
    """All candidate windows for one strip -> list of dicts."""
    L = cfg.input.num_frames
    N = int(meta_row["n_frames"])
    s0 = float(meta_row["strip_start_time"])
    fps = float(meta_row["target_fps"])
    event_t = meta_row.get("event_time", np.nan)
    w = cfg.window
    rows = []
    for st in range(0, N - L + 1, int(w.stride)):
        et = window_end_time(st, L, s0, fps)
        lab = window_label(et, event_t, w)
        if lab == -1 and drop_ignore:
            continue
        rows.append({"id": meta_row["id"], "strip_path": meta_row["strip_path"],
                     "start_idx": st, "end_time": round(et, 4), "label": lab,
                     "event_time": event_t})
    return rows


def _load_manifest_negatives(cfg, video_ids, filename):
    """Load a negatives manifest (full-timeline or mined hard negs) for given videos."""
    path = Path(cfg.paths.clips_dir) / filename
    if not path.exists():
        return pd.DataFrame(columns=["id", "strip_path", "start_idx", "label"])
    nm = pd.read_csv(path, dtype={"id": str})
    nm["id"] = nm["id"].str.zfill(5)
    nm = nm[nm["id"].isin(set(video_ids))].copy()
    nm["end_time"] = np.nan
    nm["event_time"] = np.nan
    return nm[["id", "strip_path", "start_idx", "label", "end_time", "event_time"]]


def _load_fulltime_negatives(cfg, video_ids):
    return _load_manifest_negatives(cfg, video_ids, "neg_manifest.csv")


def build_window_manifest(meta_df, cfg, train: bool, seed: int):
    rows = []
    for _, r in meta_df.iterrows():
        rows.extend(enumerate_windows(r, cfg, drop_ignore=True))
    wdf = pd.DataFrame(rows)
    wdf = wdf[wdf["label"] >= 0].reset_index(drop=True)
    pos = wdf[wdf["label"] == 1]
    neg = wdf[wdf["label"] == 0]
    # pool: near-event negatives + full-timeline hard negatives (both splits, so the
    # monitored val metric reflects real false-alarm behaviour on full streams)
    ft_neg = _load_fulltime_negatives(cfg, set(meta_df["id"]))
    neg = pd.concat([neg, ft_neg], ignore_index=True)
    if train:                                   # subsample negatives for balance
        # mined hard negatives (oversampled) — emphasise the windows the previous
        # model false-fired on. Only used when hardneg_manifest.csv exists.
        over = int(cfg.window.get("hardneg_oversample", 0))
        if over > 0:
            hard = _load_manifest_negatives(cfg, set(meta_df["id"]), "hardneg_manifest.csv")
            if len(hard):
                neg = pd.concat([neg] + [hard] * over, ignore_index=True)
        k = min(len(neg), int(cfg.window.neg_ratio) * max(1, len(pos)))
        neg = neg.sample(n=k, random_state=seed)
    # val: keep ALL negatives (representative of deployment imbalance)
    wdf = pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return wdf


def video_level_split(cfg):
    """Stratified split of VIDEOS (not windows) to prevent leakage."""
    meta = pd.read_csv(Path(cfg.paths.clips_dir) / "train_meta.csv", dtype={"id": str})
    meta["id"] = meta["id"].str.zfill(5)
    from sklearn.model_selection import train_test_split
    tr, va = train_test_split(meta, test_size=cfg.train.val_fraction,
                              stratify=meta["target"], random_state=cfg.train.seed)
    return tr.reset_index(drop=True), va.reset_index(drop=True)


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class WindowDataset(Dataset):
    def __init__(self, window_df, cfg, train: bool):
        self.df = window_df.reset_index(drop=True)
        self.cfg = cfg
        self.train = train
        self.L = cfg.input.num_frames
        self.crop = cfg.input.crop_size
        self.mean = torch.tensor(cfg.input.mean).view(3, 1, 1, 1)
        self.std = torch.tensor(cfg.input.std).view(3, 1, 1, 1)
        self.lags = motion_lags(cfg)
        self.rng = np.random.default_rng(cfg.train.seed + (0 if train else 1))
        self._cache = {}

    def __len__(self):
        return len(self.df)

    def _strip(self, path):
        m = self._cache.get(path)
        if m is None:
            m = np.load(path, mmap_mode="r")
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[path] = m
        return m

    def _spatial(self, clip):
        S, c = clip.shape[1], self.crop
        if self.train:
            lo, hi = self.cfg.aug.random_resized_crop_scale
            cs = min(S, max(c, int(round(S * float(self.rng.uniform(lo, hi))))))
            top = int(self.rng.integers(0, S - cs + 1))
            left = int(self.rng.integers(0, S - cs + 1))
            clip = clip[:, top:top + cs, left:left + cs, :]
            if cs != c:
                import cv2
                clip = np.stack([cv2.resize(f, (c, c)) for f in clip])
            if self.rng.random() < self.cfg.aug.hflip_prob:
                clip = clip[:, :, ::-1, :]
        else:
            top = (S - c) // 2
            clip = clip[:, top:top + c, top:top + c, :]
        return np.ascontiguousarray(clip)

    def _color_jitter(self, x):
        if not self.train or self.cfg.aug.color_jitter <= 0:
            return x
        j = self.cfg.aug.color_jitter
        x = x * (1 + float(self.rng.uniform(-j, j)))
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * (1 + float(self.rng.uniform(-j, j))) + mean
        return x.clamp(0, 1)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        strip = self._strip(row["strip_path"])
        N = strip.shape[0]
        st = int(row["start_idx"])
        if self.train and self.cfg.window.temporal_jitter > 0:
            j = int(self.cfg.window.temporal_jitter)
            st = int(np.clip(st + self.rng.integers(-j, j + 1), 0, N - self.L))
        clip = np.asarray(strip[st:st + self.L])           # [L,S,S,3] uint8
        clip = self._spatial(clip)
        x = torch.from_numpy(clip).float().div_(255.0).permute(3, 0, 1, 2).contiguous()
        x = self._color_jitter(x)
        x = assemble_input(x, self.mean, self.std, self.lags)
        y = torch.tensor(float(row["label"]))
        return x, y, row["id"]


def strip_to_windows_tensor(meta_row, cfg):
    """Build a [W,3,L,H,W] tensor of ALL center-cropped windows of one strip,
    plus their end-times. Used for per-video temporal evaluation/inference."""
    L = cfg.input.num_frames
    c = cfg.input.crop_size
    strip = np.load(meta_row["strip_path"], mmap_mode="r")
    N = strip.shape[0]
    s0 = float(meta_row["strip_start_time"]); fps = float(meta_row["target_fps"])
    mean = torch.tensor(cfg.input.mean).view(3, 1, 1, 1)
    std = torch.tensor(cfg.input.std).view(3, 1, 1, 1)
    lags = motion_lags(cfg)
    S = strip.shape[1]; top = (S - c) // 2
    xs, ets = [], []
    for st in range(0, N - L + 1, int(cfg.window.stride)):
        clip = np.asarray(strip[st:st + L, top:top + c, top:top + c, :]).astype(np.float32) / 255.0
        rgb01 = torch.from_numpy(clip).permute(3, 0, 1, 2).contiguous()
        xs.append(assemble_input(rgb01, mean, std, lags))
        ets.append(window_end_time(st, L, s0, fps))
    x = torch.stack(xs).float()
    return x, np.array(ets)
