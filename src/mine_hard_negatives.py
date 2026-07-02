"""
Hard-negative mining for the trained detector.

Streams every TRAINING-split video through the current model, finds the
**negative** windows it (wrongly) scores as high-confidence accident, and saves
those exact 16-frame clips as extra hard negatives. Re-training with them
directly attacks the full-video false-alarm floor.

A window is negative-eligible if:
  * the video has no accident (target 0), or
  * the video is an accident video but the window's end time is well OUTSIDE the
    event region [event - pos_pre - guard, event + ignore_post + guard].

Motion configs are supported: windows are scored with real-history temporal
diffs (exactly like the streaming deploy), and each saved clip keeps `max_lag`
context frames in front of its L window frames so re-training also sees real
diffs (the manifest's `ctx` column tells the dataset where the window starts).

Usage:
    python -m src.mine_hard_negatives --config configs/jetson_videomae.yaml \
        --ckpt artifacts/runs/jetson_videomae_base/best.pt \
        --mine-threshold 0.85 --max-per-video 12
Outputs (into the config's clips_dir):
    hardneg/<id>.npy           concatenated mined clips [k*(ctx+L), S, S, 3] uint8
    hardneg_manifest.csv       id, strip_path, start_idx, ctx, label=0, prob
"""
from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config import load_config
from src.dataset import video_level_split, assemble_input, motion_lags
from src.model import build_model
from src.detect_video import preprocess_frame


@torch.no_grad()
def mine_video(model, path, cfg, device, mean, std, event_time,
               mine_thr, max_per_video, stride, guard=0.5):
    """Return (clips[list of [ctx+L,S,S,3] uint8], probs[list]) for hard negatives."""
    L, S = cfg.input.num_frames, cfg.input.crop_size
    lags = motion_lags(cfg)
    max_lag = max(lags) if lags else 0
    tgt_fps = cfg.strip.target_fps
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / tgt_fps)))
    # decode -> uniform 16fps frame array + the source time of each kept frame
    frames, times, fc = [], [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fc % step == 0:
            frames.append(preprocess_frame(fr, S))
            times.append(fc / src_fps)
        fc += 1
    cap.release()
    if len(frames) < L:
        return [], []
    arr = np.stack(frames)                                   # [N,S,S,3] uint8
    times = np.array(times)
    N = arr.shape[0]

    # window end time = time of last frame in window; start at max_lag so every
    # window has full motion context (and every stored clip the same length)
    starts = list(range(max_lag, N - L + 1, stride))
    excl = None
    if event_time is not None and not (isinstance(event_time, float) and np.isnan(event_time)):
        excl = (event_time - cfg.window.pos_pre - guard,
                event_time + cfg.window.ignore_post + guard)

    keep_starts, probs = [], []
    for bstart in range(0, len(starts), 16):
        chunk = starts[bstart:bstart + 16]
        xs = []
        for s in chunk:
            seq = torch.from_numpy(arr[s - max_lag:s + L].astype(np.float32) / 255.0) \
                .permute(3, 0, 1, 2)                         # [3,ctx+L,S,S]
            xs.append(assemble_input(seq, mean, std, lags, ctx=max_lag))
        x = torch.stack(xs).to(device)
        with torch.autocast("cuda", enabled=device == "cuda"):
            p = torch.sigmoid(model(x)).float().cpu().numpy()
        for s, pr in zip(chunk, p):
            end_t = times[s + L - 1]
            negative_eligible = excl is None or (end_t < excl[0] or end_t > excl[1])
            if negative_eligible and pr >= mine_thr:
                keep_starts.append(s); probs.append(float(pr))

    if not keep_starts:
        return [], []
    order = np.argsort(probs)[::-1][:max_per_video]          # hardest first
    clips = [arr[keep_starts[i] - max_lag:keep_starts[i] + L] for i in order]
    probs = [probs[i] for i in order]
    return clips, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--mine-threshold", type=float, default=0.85)
    ap.add_argument("--max-per-video", type=int, default=12)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(cfg.paths.output_dir) / cfg.experiment_name
    ckpt = torch.load(Path(args.ckpt) if args.ckpt else out_dir / "best.pt",
                      map_location=device, weights_only=False)
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    mean = torch.tensor(cfg.input.mean).view(3, 1, 1, 1)
    std = torch.tensor(cfg.input.std).view(3, 1, 1, 1)

    tr_meta, _ = video_level_split(cfg)                      # TRAIN videos only
    src = Path(cfg.paths.dataset_root) / "train"
    hard_dir = Path(cfg.paths.clips_dir) / "hardneg"
    hard_dir.mkdir(parents=True, exist_ok=True)

    L = cfg.input.num_frames
    lags = motion_lags(cfg)
    max_lag = max(lags) if lags else 0
    seg = L + max_lag                     # stored clip = ctx frames + window frames
    manifest, total = [], 0
    for _, r in tqdm(tr_meta.iterrows(), total=len(tr_meta), desc="mine"):
        clips, probs = mine_video(
            model, src / f"{r['id']}.mp4", cfg, device, mean, std,
            r.get("event_time", np.nan), args.mine_threshold,
            args.max_per_video, args.stride)
        if not clips:
            continue
        arr = np.concatenate(clips).astype(np.uint8)
        p = hard_dir / f"{r['id']}.npy"
        np.save(p, arr)
        for k, pr in enumerate(probs):
            manifest.append({"id": r["id"], "strip_path": str(p),
                             "start_idx": k * seg + max_lag, "ctx": max_lag,
                             "label": 0, "prob": round(pr, 4)})
        total += len(clips)

    man = pd.DataFrame(manifest)
    out = Path(cfg.paths.clips_dir) / "hardneg_manifest.csv"
    man.to_csv(out, index=False)
    n_vid = man["id"].nunique() if len(man) else 0
    print(f"mined {total} hard negatives from {n_vid} videos "
          f"(thr={args.mine_threshold}) -> {out}")


if __name__ == "__main__":
    main()
