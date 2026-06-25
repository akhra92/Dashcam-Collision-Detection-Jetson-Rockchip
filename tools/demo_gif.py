"""Build a README demo GIF: dashcam video next to the live P(accident) curve.

Two modes:
  rank  — score candidate accident videos and print the best demo clips
          (fires near the labelled event, high peak prob, low pre-event alarms).
  make  — render the side-by-side GIF for one video, trimmed to a short window
          centred on the collision.

Inference mirrors app.py / the ONNX meta (R(2+1)D, 16x112x112 @16fps, CPU).

Examples
--------
    python -m tools.demo_gif rank --limit 60 --top 10
    python -m tools.demo_gif make --id 00208 --out assets/demo.gif
"""
from __future__ import annotations
import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ONNX = "artifacts/runs/jetson_r2plus1d18/jetson_r2plus1d18.onnx"
META = "artifacts/runs/jetson_r2plus1d18/jetson_r2plus1d18.meta.json"
TRAIN_CSV = "dataset/train.csv"
VIDEO_DIR = "dataset/train"


def load_cfg():
    m = json.loads(Path(META).read_text())
    return {
        "L": m["window_frames"], "S": m["input_shape"][-1],
        "fps": m["target_fps"], "thr": m["detect_threshold"],
        "consec": m["consec"], "mean": np.array(m["mean"], np.float32),
        "std": np.array(m["std"], np.float32),
    }


def _prep(fr, S):
    h, w = fr.shape[:2]
    nh, nw = (S, round(w * S / h)) if h <= w else (round(h * S / w), S)
    fr = cv2.resize(fr, (nw, nh))
    top, left = (nh - S) // 2, (nw - S) // 2
    return cv2.cvtColor(fr[top:top + S, left:left + S], cv2.COLOR_BGR2RGB)


def run_curve(sess, in_name, path, cfg, t0, t1, scan_stride, keep_frames=False):
    """Slide the window over [t0, t1] (seconds). Returns (times, probs, frames)."""
    L, S = cfg["L"], cfg["S"]
    cap = cv2.VideoCapture(path)
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src / cfg["fps"]))
    buf, raw = deque(maxlen=L), deque(maxlen=L)
    times, probs, frames = [], [], []
    fc = scount = 0
    warm = t0 - L / cfg["fps"]            # let the window fill before t0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = fc / src
        if t > t1 + 0.05:
            break
        if fc % step == 0 and t >= warm:
            buf.append(_prep(fr, S))
            raw.append(fr)
            scount += 1
            if len(buf) == L and scount % scan_stride == 0 and t >= t0:
                clip = (np.stack(buf).astype(np.float32) / 255.0 - cfg["mean"]) / cfg["std"]
                x = clip.transpose(3, 0, 1, 2)[None].astype(np.float32)
                logit = sess.run(None, {in_name: x})[0].reshape(-1)[0]
                times.append(t)
                probs.append(float(1.0 / (1.0 + np.exp(-logit))))
                if keep_frames:
                    frames.append(raw[-1].copy())
        fc += 1
    cap.release()
    return np.array(times), np.array(probs), frames


def first_fire(times, probs, thr, consec):
    run = 0
    for i, p in enumerate(probs):
        run = run + 1 if p >= thr else 0
        if run >= consec:
            return times[i]
    return None


def cmd_rank(args):
    import onnxruntime as ort
    cfg = load_cfg()
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    df = pd.read_csv(TRAIN_CSV, dtype={"id": str})
    acc = df[(df.target == 1) & df.time_of_event.notna()]
    acc = acc[(acc.time_of_event >= 7.5)].head(args.limit)

    half = args.window / 2.0
    rows = []
    for _, r in acc.iterrows():
        vid, e = r["id"], float(r["time_of_event"])
        path = f"{VIDEO_DIR}/{vid}.mp4"
        if not Path(path).exists():
            continue
        t0, t1 = max(0.0, e - half), e + half
        times, probs, _ = run_curve(sess, in_name, path, cfg, t0, t1, args.stride)
        if len(probs) < 5:
            continue
        fire = first_fire(times, probs, cfg["thr"], cfg["consec"])
        pre = probs[times < e - 1.0]                 # alarms before the event
        post_peak = probs[times >= e - 1.0].max() if (times >= e - 1.0).any() else 0.0
        fire_err = abs(fire - e) if fire is not None else 99.0
        pre_alarm = float(pre.max()) if len(pre) else 0.0
        # good demo: fires within 1s, high post peak, quiet before
        score = post_peak - 0.6 * pre_alarm - 0.3 * min(fire_err, 3.0)
        rows.append((round(score, 3), vid, round(e, 2),
                     None if fire is None else round(float(fire), 2),
                     round(post_peak, 2), round(pre_alarm, 2)))
        print(f"  scored {vid}: score={score:.2f} event={e:.1f} "
              f"fire={fire} peak={post_peak:.2f} prealarm={pre_alarm:.2f}", flush=True)

    rows.sort(reverse=True)
    print("\n=== TOP DEMO CANDIDATES (score | id | event | fire_t | peak | pre_alarm) ===")
    for row in rows[:args.top]:
        print("  ", row)


def cmd_make(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import onnxruntime as ort

    cfg = load_cfg()
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    df = pd.read_csv(TRAIN_CSV, dtype={"id": str})
    e = float(df.loc[df.id == args.id, "time_of_event"].iloc[0]) if args.event is None else args.event

    half = args.window / 2.0
    t0, t1 = max(0.0, e - half), e + half
    path = f"{VIDEO_DIR}/{args.id}.mp4"
    print(f"running model on {args.id} over [{t0:.1f}, {t1:.1f}]s (event={e:.2f})...")
    times, probs, frames = run_curve(sess, in_name, path, cfg, t0, t1,
                                     args.stride, keep_frames=True)
    print(f"got {len(times)} windows; rendering GIF...")
    fire_t = first_fire(times, probs, cfg["thr"], cfg["consec"])

    panel_w = args.height * 16 // 9          # video panel ~16:9
    pil_frames = []
    for k in range(len(times)):
        # left: video frame (RGB), letterboxed to panel
        fr = cv2.cvtColor(frames[k], cv2.COLOR_BGR2RGB)
        h, w = fr.shape[:2]
        scale = min(panel_w / w, args.height / h)
        fr = cv2.resize(fr, (int(w * scale), int(h * scale)))
        canvas = np.zeros((args.height, panel_w, 3), np.uint8)
        yo, xo = (args.height - fr.shape[0]) // 2, (panel_w - fr.shape[1]) // 2
        canvas[yo:yo + fr.shape[0], xo:xo + fr.shape[1]] = fr
        fired = fire_t is not None and times[k] >= fire_t
        if fired:                            # red banner once detected
            cv2.rectangle(canvas, (0, 0), (panel_w - 1, args.height - 1), (220, 40, 40), 6)
            cv2.putText(canvas, f"COLLISION  t={fire_t:.1f}s", (16, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 80, 80), 2, cv2.LINE_AA)

        # right: probability curve up to current time
        fig, ax = plt.subplots(figsize=(panel_w / 100, args.height / 100), dpi=100)
        ax.plot(times[:k + 1] - t0, probs[:k + 1], color="#1f77b4", lw=2)
        ax.axhline(cfg["thr"], ls="--", color="#888", lw=1)
        ax.text(0.02, cfg["thr"] + 0.02, f"threshold {cfg['thr']:.2f}",
                color="#666", fontsize=8)
        ax.axvline(e - t0, color="#2ca02c", ls=":", lw=1.5)
        ax.text(e - t0, 1.02, " true event", color="#2ca02c", fontsize=8, ha="center")
        if fired:
            ax.scatter([times[k] - t0], [probs[k]], color="#d62728", zorder=5)
        ax.set_xlim(0, t1 - t0)
        ax.set_ylim(0, 1.08)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("P(accident)")
        ax.set_title("R(2+1)D — live collision score")
        fig.tight_layout(pad=0.6)
        fig.canvas.draw()
        plot = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)
        plot = cv2.resize(plot, (panel_w, args.height))

        combo = np.hstack([canvas, plot])
        pil_frames.append(Image.fromarray(combo))

    # hold the last frame a bit longer
    durations = [int(1000 / args.fps)] * len(pil_frames)
    durations[-1] = 1500
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(out, save_all=True, append_images=pil_frames[1:],
                       duration=durations, loop=0, optimize=True)
    mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({len(pil_frames)} frames, {mb:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("rank")
    r.add_argument("--limit", type=int, default=60, help="how many accident videos to scan")
    r.add_argument("--top", type=int, default=10)
    r.add_argument("--window", type=float, default=12.0, help="seconds centred on event")
    r.add_argument("--stride", type=int, default=3)
    r.set_defaults(func=cmd_rank)

    m = sub.add_parser("make")
    m.add_argument("--id", required=True)
    m.add_argument("--event", type=float, default=None, help="override event time (s)")
    m.add_argument("--out", default="assets/demo.gif")
    m.add_argument("--window", type=float, default=12.0, help="clip length (s), event centred")
    m.add_argument("--stride", type=int, default=2, help="scan stride (smaller = smoother)")
    m.add_argument("--height", type=int, default=288, help="panel height px")
    m.add_argument("--fps", type=int, default=8, help="GIF playback fps")
    m.set_defaults(func=cmd_make)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
