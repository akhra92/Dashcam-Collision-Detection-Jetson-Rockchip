"""
Extract full-timeline HARD NEGATIVE window-clips for every training video.

The event-strip extraction (src.preprocess) only covers 8s per video, so the
model never learns most of the normal-driving footage and false-alarms on long
streams. Here we sample `negatives.per_video` one-second windows from across the
WHOLE video (excluding a guard region around the event for positives) and store
them concatenated as a pseudo-strip [K*L, S, S, 3]; window k is frames
[k*L : (k+1)*L]. A manifest lists every clip with label 0.

Usage:
    python -m src.extract_negatives --config configs/jetson_r2plus1d.yaml
"""
from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import load_config
from src.preprocess import _resize_short_side, _center_square


def read_indices(cap, src_idx, short, store):
    """Read sorted source frame indices -> dict{idx: frame(RGB)}."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (int(src_idx.max()) + 1)
    src_idx = np.clip(src_idx, 0, max(0, total - 1))
    wanted = sorted(set(int(i) for i in src_idx))
    frames, ptr, ni = {}, 0, 0
    while ni < len(wanted) and ptr <= wanted[-1]:
        tgt = wanted[ni]
        if tgt - ptr > 8:
            cap.set(cv2.CAP_PROP_POS_FRAMES, tgt); ptr = tgt
        while ptr < tgt:
            cap.grab(); ptr += 1
        ok, fr = cap.read(); ptr += 1
        if not ok:
            break
        frames[tgt] = cv2.cvtColor(_center_square(_resize_short_side(fr, short), store),
                                   cv2.COLOR_BGR2RGB)
        ni += 1
    return frames, src_idx


def build(cfg):
    ds_root = Path(cfg.paths.dataset_root)
    out_dir = Path(cfg.paths.clips_dir) / "neg"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = pd.read_csv(ds_root / "train.csv", dtype={"id": str})
    csv["id"] = csv["id"].str.zfill(5)

    tgt_fps = float(cfg.strip.target_fps)
    L = int(cfg.input.num_frames)
    win_s = L / tgt_fps
    short = int(cfg.strip.short_side)
    fb_fps = float(cfg.strip.fallback_fps)
    K = int(cfg.negatives.per_video)
    guard = float(cfg.negatives.guard_seconds)
    pos_pre = float(cfg.window.pos_pre)
    ignore_post = float(cfg.window.ignore_post)
    rng = np.random.default_rng(2024)

    manifest = []
    for _, row in tqdm(csv.iterrows(), total=len(csv), desc="neg"):
        vid = row["id"]
        vpath = ds_root / "train" / f"{vid}.mp4"
        if not vpath.exists():
            continue
        cap = cv2.VideoCapture(str(vpath))
        fps = cap.get(cv2.CAP_PROP_FPS) or fb_fps
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 1:
            fps = fb_fps
        dur = total / fps
        latest = max(0.0, dur - win_s)

        target = int(row["target"]) if not pd.isna(row.get("target")) else 0
        ev = row.get("time_of_event", np.nan)
        excl = None
        if target == 1 and not pd.isna(ev):
            excl = (float(ev) - pos_pre - guard, float(ev) + ignore_post + guard)

        # sample K valid start times (window END time avoids the excl region)
        starts = []
        tries = 0
        while len(starts) < K and tries < K * 20:
            tries += 1
            s = float(rng.uniform(0, latest)) if latest > 0 else 0.0
            e = s + win_s
            if excl and not (e < excl[0] or s > excl[1]):
                continue
            starts.append(s)
        if not starts:
            cap.release(); continue

        # gather all needed source indices, read in one pass
        all_idx, per_clip = [], []
        for s in starts:
            idx = np.round((s + np.arange(L) / tgt_fps) * fps).astype(int)
            per_clip.append(idx); all_idx.append(idx)
        frames, _ = read_indices(cap, np.concatenate(all_idx), short, short)
        cap.release()
        if not frames:
            continue
        last = list(frames.values())[-1]

        clips = []
        for idx in per_clip:
            idx = np.clip(idx, 0, max(0, total - 1))
            clips.append(np.stack([frames.get(int(i), last) for i in idx]))
        arr = np.concatenate(clips).astype(np.uint8)           # [k*L, S, S, 3]
        np.save(out_dir / f"{vid}.npy", arr)
        for k in range(len(clips)):
            manifest.append({"id": vid, "strip_path": str(out_dir / f"{vid}.npy"),
                             "start_idx": k * L, "label": 0})

    man = pd.DataFrame(manifest)
    out = Path(cfg.paths.clips_dir) / "neg_manifest.csv"
    man.to_csv(out, index=False)
    print(f"wrote {len(man)} negative windows from {man['id'].nunique()} videos -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    build(load_config(args.config))


if __name__ == "__main__":
    main()
