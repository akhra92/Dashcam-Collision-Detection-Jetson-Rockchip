"""Realistic eval: stream the detector across FULL-LENGTH val videos.

Unlike src.evaluate (which scans only the 8s strip), this decodes the entire
video, so the false-alarm rate reflects real deployment on long streams.
"""
from __future__ import annotations
import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.dataset import video_level_split
from src.model import build_model
from src.detect_video import preprocess_frame


@torch.no_grad()
def stream_video(model, path, cfg, device, mean, std):
    L, S = cfg.input.num_frames, cfg.input.crop_size
    tgt_fps, stride = cfg.strip.target_fps, cfg.window.stride
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / tgt_fps)))
    buf = deque(maxlen=L)
    probs, times = [], []
    fcount, scount = 0, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fcount % step == 0:
            buf.append(preprocess_frame(fr, S))
            scount += 1
            if len(buf) == L and scount % stride == 0:
                clip = torch.from_numpy(np.stack(buf)).float().div_(255.0)
                x = ((clip.permute(3, 0, 1, 2).contiguous() - mean) / std).unsqueeze(0).to(device)
                with torch.autocast("cuda", enabled=device == "cuda"):
                    probs.append(torch.sigmoid(model(x)).item())
                times.append(fcount / src_fps)
        fcount += 1
    cap.release()
    return np.array(probs), np.array(times)


def detect(probs, times, thr, consec):
    run = 0
    for i, p in enumerate(probs):
        run = run + 1 if p >= thr else 0
        if run >= consec:
            return float(times[i])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--limit", type=int, default=0, help="cap videos per class (0=all)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg.paths.output_dir) / cfg.experiment_name
    ckpt = torch.load(Path(args.ckpt) if args.ckpt else out_dir / "best.pt", map_location=device)
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    mean = torch.tensor(cfg.input.mean).view(3, 1, 1, 1)
    std = torch.tensor(cfg.input.std).view(3, 1, 1, 1)

    _, va_meta = video_level_split(cfg)
    src = Path(cfg.paths.dataset_root) / "train"
    consec = cfg.detect.consec
    tol = cfg.detect.tolerance_s

    if args.limit:
        va_meta = pd.concat([va_meta[va_meta.target == 0].head(args.limit),
                             va_meta[va_meta.target == 1].head(args.limit)])

    curves = {}
    for k, (_, r) in enumerate(va_meta.iterrows()):
        p, t = stream_video(model, src / f"{r['id']}.mp4", cfg, device, mean, std)
        curves[r["id"]] = (p, t, int(r["target"]), r.get("event_time", np.nan))
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(va_meta)} videos streamed")

    print("\nthr   det_rate  false_alarm  loc_err  lead")
    best = None
    for thr in np.linspace(0.5, 0.97, 24):
        pos_t = pos_h = neg_t = neg_f = 0
        errs, leads = [], []
        for p, t, tgt, ev in curves.values():
            d = detect(p, t, thr, consec)
            if tgt == 1 and not np.isnan(ev):
                pos_t += 1
                if d is not None and abs(d - ev) <= tol:
                    pos_h += 1; errs.append(abs(d - ev)); leads.append(ev - d)
            elif tgt == 0:
                neg_t += 1; neg_f += int(d is not None)
        dr, far = pos_h / max(1, pos_t), neg_f / max(1, neg_t)
        le = np.mean(errs) if errs else float("nan")
        ld = np.mean(leads) if leads else float("nan")
        print(f"{thr:.2f}   {dr:.3f}     {far:.3f}       {le:.3f}    {ld:+.3f}")
        if best is None or (dr - far) > best[1]:
            best = (thr, dr - far, dr, far, le, ld)
    print(f"\nbest thr={best[0]:.2f}  det_rate={best[2]:.3f}  false_alarm={best[3]:.3f}  "
          f"loc_err={best[4]:.3f}s  lead={best[5]:+.3f}s")


if __name__ == "__main__":
    main()
