"""
Run the trained PyTorch model as a streaming temporal detector on a single video.

Useful on the workstation to (a) demo detection and (b) serve as the parity
reference the Jetson TensorRT output is compared against. Decodes the video,
resamples to target_fps, slides the causal window, and reports the detected
accident time + optional probability curve.

Usage:
    python -m src.detect_video --config configs/jetson_r2plus1d.yaml \
        --video dataset/train/00822.mp4 [--ckpt ...] [--dump-curve curve.csv]
"""
from __future__ import annotations
import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from src.config import load_config
from src.dataset import assemble_input, motion_enabled
from src.model import build_model


def preprocess_frame(fr, S):
    h, w = fr.shape[:2]
    if h <= w:
        nh, nw = S, int(round(w * S / h))
    else:
        nh, nw = int(round(h * S / w)), S
    fr = cv2.resize(fr, (nw, nh))
    top, left = (nh - S) // 2, (nw - S) // 2
    return cv2.cvtColor(fr[top:top + S, left:left + S], cv2.COLOR_BGR2RGB)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--dump-curve", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(cfg.paths.output_dir) / cfg.experiment_name
    ckpt_path = Path(args.ckpt) if args.ckpt else out_dir / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])

    thr = ckpt.get("detect_threshold", cfg.detect.threshold)
    consec = cfg.detect.consec
    L = cfg.input.num_frames
    S = cfg.input.crop_size
    tgt_fps = cfg.strip.target_fps
    stride = cfg.window.stride
    mean = torch.tensor(cfg.input.mean).view(3, 1, 1, 1)
    std = torch.tensor(cfg.input.std).view(3, 1, 1, 1)
    motion = motion_enabled(cfg)

    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / tgt_fps)))

    buf = deque(maxlen=L)
    curve, run, fired = [], 0, None
    fcount, scount = 0, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fcount % step == 0:
            buf.append(preprocess_frame(fr, S))
            scount += 1
            if len(buf) == L and scount % stride == 0:
                rgb01 = torch.from_numpy(np.stack(buf)).float().div_(255.0) \
                    .permute(3, 0, 1, 2).contiguous()
                x = assemble_input(rgb01, mean, std, motion).unsqueeze(0).to(device)
                prob = torch.sigmoid(model(x)).item()
                t = fcount / src_fps
                curve.append((t, prob))
                run = run + 1 if prob >= thr else 0
                if run >= consec and fired is None:
                    fired = t
        fcount += 1
    cap.release()

    if fired is not None:
        print(f"ACCIDENT DETECTED at t = {fired:.2f}s (thr={thr:.2f}, consec={consec})")
    else:
        print(f"no accident detected (thr={thr:.2f})")
    if curve:
        peak = max(curve, key=lambda c: c[1])
        print(f"peak prob {peak[1]:.3f} at t={peak[0]:.2f}s")
    if args.dump_curve:
        with open(args.dump_curve, "w") as f:
            f.write("end_time,prob\n")
            for t, p in curve:
                f.write(f"{t:.3f},{p:.4f}\n")
        print(f"curve -> {args.dump_curve}")


if __name__ == "__main__":
    main()
