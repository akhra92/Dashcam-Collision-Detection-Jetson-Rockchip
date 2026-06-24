"""
Temporal detection evaluation on the held-out validation videos.

For each video we slide the window classifier across its strip to get a
P(accident) curve, then apply causal detection logic (fire when `consec`
consecutive windows exceed the threshold). We report:
  - detection_rate : positives detected within +/- tolerance_s of time_of_event
  - false_alarm_rate: negatives that fire anywhere
  - mean_loc_error : |detected - event| over correct detections
  - mean_lead_time : event - detected (how early it warns; >0 = before impact)
  - window AUC / AP : frame-window level ranking quality
The detection threshold is tuned to maximize (detection_rate - false_alarm_rate).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.dataset import (video_level_split, build_window_manifest,
                         WindowDataset, strip_to_windows_tensor)
from src.model import build_model
from src.utils import compute_metrics
from src.train import evaluate as window_eval


def detect(probs, end_times, thr, consec):
    """First time `consec` consecutive windows are >= thr -> (fired, time)."""
    run = 0
    for i, p in enumerate(probs):
        run = run + 1 if p >= thr else 0
        if run >= consec:
            return True, float(end_times[i])
    return False, None


@torch.no_grad()
def video_curves(model, va_meta, cfg, device):
    """Return {id: (probs, end_times, target, event)} for all val videos."""
    model.eval()
    out = {}
    for _, r in va_meta.iterrows():
        x, ets = strip_to_windows_tensor(r, cfg)
        probs = []
        for j in range(0, len(x), cfg.train.batch_size):
            xb = x[j:j + cfg.train.batch_size].to(device)
            with torch.autocast("cuda", enabled=device == "cuda"):
                probs.append(torch.sigmoid(model(xb)).float().cpu())
        out[r["id"]] = (torch.cat(probs).numpy(), ets,
                        int(r["target"]), r.get("event_time", np.nan))
    return out


def score_videos(curves, thr, consec, tol):
    pos_tot = pos_hit = neg_tot = neg_fa = 0
    loc_err, lead = [], []
    for _vid, (probs, ets, tgt, ev) in curves.items():
        fired, t = detect(probs, ets, thr, consec)
        if tgt == 1 and not np.isnan(ev):
            pos_tot += 1
            if fired and abs(t - ev) <= tol:
                pos_hit += 1
                loc_err.append(abs(t - ev))
                lead.append(ev - t)
        elif tgt == 0:
            neg_tot += 1
            neg_fa += int(fired)
    return {
        "threshold": float(thr),
        "detection_rate": pos_hit / max(1, pos_tot),
        "false_alarm_rate": neg_fa / max(1, neg_tot),
        "mean_loc_error": float(np.mean(loc_err)) if loc_err else None,
        "mean_lead_time": float(np.mean(lead)) if lead else None,
        "n_pos": pos_tot, "n_neg": neg_tot, "n_detected": pos_hit,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(cfg.paths.output_dir) / cfg.experiment_name
    ckpt_path = Path(args.ckpt) if args.ckpt else out_dir / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    _, va_meta = video_level_split(cfg)

    # ---- window-level ranking quality ----
    va_win = build_window_manifest(va_meta, cfg, train=False, seed=cfg.train.seed)
    vl = DataLoader(WindowDataset(va_win, cfg, train=False),
                    batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers)
    yt, yp = window_eval(model, vl, device)
    win_metrics = compute_metrics(yt, yp, 0.5)

    # ---- temporal detection: tune threshold ----
    curves = video_curves(model, va_meta, cfg, device)
    consec, tol = cfg.detect.consec, cfg.detect.tolerance_s
    best = None
    for thr in np.linspace(0.3, 0.95, 27):
        s = score_videos(curves, thr, consec, tol)
        s["objective"] = s["detection_rate"] - s["false_alarm_rate"]
        if best is None or s["objective"] > best["objective"]:
            best = s

    report = {"window_level": win_metrics, "temporal": best,
              "consec": consec, "tolerance_s": tol}
    print(json.dumps(report, indent=2))
    with open(out_dir / "temporal_eval.json", "w") as f:
        json.dump(report, f, indent=2)

    # persist tuned detection threshold for inference
    ckpt["detect_threshold"] = best["threshold"]
    ckpt["temporal_eval"] = best
    torch.save(ckpt, ckpt_path)
    print(f"saved -> {out_dir/'temporal_eval.json'}  (detect_threshold={best['threshold']:.3f})")


if __name__ == "__main__":
    main()
