"""
Pre-extract one contiguous frame STRIP per video, resampled to a fixed fps.

Each strip is a [n_frames, S, S, 3] uint8 array sampled uniformly at
`strip.target_fps`. Because the strip fps is uniform, the source time of strip
frame j is simply  strip_start_time + j / target_fps. A causal 1s window is then
just 16 consecutive strip frames, and we can label any window by where its end
time falls relative to `time_of_event`.

Positive videos: strip is positioned so the event sits `pre_event_seconds` into
the strip (giving normal-driving context before + a little aftermath).
Negative videos: a random strip location.

Outputs:
    artifacts/clips/<split>/<id>.npy           # the strip
    artifacts/clips/<split>_meta.csv           # per-video temporal metadata

Usage:
    python -m src.preprocess --config configs/jetson_r2plus1d.yaml --split train
    python -m src.preprocess --config configs/jetson_r2plus1d.yaml --split test
"""
from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import load_config


def _resize_short_side(frame, short):
    h, w = frame.shape[:2]
    if h <= w:
        nh, nw = short, int(round(w * short / h))
    else:
        nh, nw = int(round(h * short / w)), short
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def _center_square(frame, size):
    h, w = frame.shape[:2]
    top, left = max(0, (h - size) // 2), max(0, (w - size) // 2)
    return frame[top:top + size, left:left + size]


def extract_strip(video_path, src_indices, short, store):
    """Read the given source frame indices -> [T, store, store, 3] uint8 (RGB)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (int(src_indices.max()) + 1)
    src_indices = np.clip(src_indices, 0, max(0, total - 1))

    frames, ptr = {}, 0
    wanted = sorted(set(int(i) for i in src_indices))
    next_i = 0
    while next_i < len(wanted) and ptr <= wanted[-1]:
        target = wanted[next_i]
        if target - ptr > 8:                       # large gap -> seek
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ptr = target
        while ptr < target:
            cap.grab(); ptr += 1
        ok, fr = cap.read(); ptr += 1
        if not ok:
            break
        fr = _center_square(_resize_short_side(fr, short), store)
        frames[target] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        next_i += 1
    cap.release()
    if not frames:
        return None
    last = list(frames.values())[-1]
    seq = [frames.get(i, last) for i in src_indices]
    return np.stack(seq).astype(np.uint8)


def build_split(cfg, split):
    ds_root = Path(cfg.paths.dataset_root)
    out_dir = Path(cfg.paths.clips_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    csv = pd.read_csv(ds_root / f"{split}.csv", dtype={"id": str})
    csv["id"] = csv["id"].str.zfill(5)

    tgt_fps = float(cfg.strip.target_fps)
    strip_s = float(cfg.strip.strip_seconds)
    n_frames = int(round(strip_s * tgt_fps))
    pre_s = float(cfg.strip.pre_event_seconds)
    short = int(cfg.strip.short_side)
    fb_fps = float(cfg.strip.fallback_fps)

    rng = np.random.default_rng(123)
    meta = []
    for _, row in tqdm(csv.iterrows(), total=len(csv), desc=f"strip[{split}]"):
        vid = row["id"]
        vpath = ds_root / split / f"{vid}.mp4"
        if not vpath.exists():
            continue
        cap = cv2.VideoCapture(str(vpath))
        fps = cap.get(cv2.CAP_PROP_FPS) or fb_fps
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps <= 1:
            fps = fb_fps
        dur = total / fps

        target = int(row["target"]) if ("target" in row and not pd.isna(row.get("target"))) else -1
        t_event = row.get("time_of_event", None)
        has_event = target == 1 and t_event is not None and not pd.isna(t_event)

        if has_event:
            start_t = float(t_event) - pre_s
        else:
            start_t = float(rng.uniform(0, max(0.1, dur - strip_s)))
        # clamp strip within [0, dur]
        start_t = float(np.clip(start_t, 0.0, max(0.0, dur - strip_s)))

        # uniform sampling at target_fps -> source frame indices
        strip_times = start_t + np.arange(n_frames) / tgt_fps
        src_idx = np.round(strip_times * fps).astype(int)
        arr = extract_strip(vpath, src_idx, short, short)
        if arr is None or arr.shape[0] != n_frames:
            continue

        np.save(out_dir / f"{vid}.npy", arr)
        rec = {"id": vid, "strip_path": str(out_dir / f"{vid}.npy"),
               "strip_start_time": round(start_t, 4), "target_fps": tgt_fps,
               "n_frames": n_frames, "duration": round(dur, 3)}
        if target >= 0:
            rec["target"] = target
        rec["event_time"] = float(t_event) if has_event else np.nan
        meta.append(rec)

    meta_df = pd.DataFrame(meta)
    meta_path = Path(cfg.paths.clips_dir) / f"{split}_meta.csv"
    meta_df.to_csv(meta_path, index=False)
    print(f"[{split}] wrote {len(meta_df)} strips -> {meta_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    build_split(cfg, args.split)


if __name__ == "__main__":
    main()
