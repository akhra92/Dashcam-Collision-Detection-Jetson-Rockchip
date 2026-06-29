"""Streaming temporal accident detection on the Rockchip RK3588.

Runs on the **board** with `rknn-toolkit-lite2` (NPU) + `onnxruntime` (CPU head).

Pipeline, per the deploy split:
  * each sampled frame -> NPU backbone (RKNN)        -> 1 feature vector
  * keep the last T features; every `stride` frames  -> CPU temporal head (ONNX)
    -> logit -> P(accident); fire when `consec` windows exceed the threshold.

Because the heavy 2D backbone runs once per frame (not once per window), this is
efficient on a live camera stream. Mirrors deploy/jetson/infer_trt.py.

Usage (on the RK3588):
    python3 infer_rknn.py --backbone backbone.rknn --head temporal_head.onnx \
        --meta rockchip.meta.json --video clip.mp4 [--dump-curve curve.csv]
"""
from __future__ import annotations
import argparse
import json
import time
from collections import deque

import cv2
import numpy as np


def preprocess_frame(fr, S):
    """BGR frame -> center-cropped [S,S,3] uint8 RGB (raw; RKNN does mean/std)."""
    h, w = fr.shape[:2]
    if h <= w:
        nh, nw = S, int(round(w * S / h))
    else:
        nh, nw = int(round(h * S / w)), S
    fr = cv2.resize(fr, (nw, nh))
    top, left = (nh - S) // 2, (nw - S) // 2
    fr = fr[top:top + S, left:left + S]
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, help="backbone.rknn")
    ap.add_argument("--head", required=True, help="temporal_head.onnx")
    ap.add_argument("--meta", required=True, help="rockchip.meta.json")
    ap.add_argument("--video", required=True)
    ap.add_argument("--dump-curve", default=None)
    args = ap.parse_args()

    meta = json.loads(open(args.meta).read())
    L = meta["window_frames"]
    S = meta["frame_shape"][-1]
    C = meta["feat_dim"]
    tgt_fps = meta["target_fps"]
    thr = meta["detect_threshold"]
    consec = meta["consec"]
    stride = meta.get("stride", 3)
    motion = meta.get("motion", False)
    lags = meta.get("motion_lags", []) if motion else []
    mean = np.array(meta["mean"], np.float32).reshape(3, 1, 1)
    std = np.array(meta["std"], np.float32).reshape(3, 1, 1)
    mscale = meta.get("motion_scale", 0.5)
    max_lag = max(lags) if lags else 0

    # NPU backbone
    from rknnlite.api import RKNNLite
    rk = RKNNLite()
    assert rk.load_rknn(args.backbone) == 0, "load_rknn failed"
    assert rk.init_runtime() == 0, "init_runtime failed"

    # CPU temporal head
    import onnxruntime as ort
    head = ort.InferenceSession(args.head, providers=["CPUExecutionProvider"])
    head_in = head.get_inputs()[0].name

    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / tgt_fps)))

    feat_buf = deque(maxlen=L)
    rgb_hist = deque(maxlen=max_lag + 1)                     # recent frames for motion diffs
    curve, run, fired_time = [], 0, None
    fcount, scount, bb_ms, head_ms = 0, 0, [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fcount % step == 0:
            frame = preprocess_frame(fr, S)                  # [S,S,3] uint8 RGB
            if lags:
                # build the (3+3*len(lags))ch normalised input on CPU (RKNN passes through)
                rgb01 = frame.astype(np.float32).transpose(2, 0, 1) / 255.0   # [3,S,S]
                rgb_hist.append(rgb01)
                chans = [(rgb01 - mean) / std]
                for k in lags:
                    prev = rgb_hist[-1 - k] if len(rgb_hist) > k else rgb01     # zero diff early
                    chans.append((rgb01 - prev) / mscale)
                inp = np.concatenate(chans, 0)[None].astype(np.float32)
            else:
                inp = frame[None]                            # raw NHWC; RKNN normalises
            t0 = time.time()
            feat = rk.inference(inputs=[inp])[0].reshape(-1)  # -> [C]
            bb_ms.append((time.time() - t0) * 1000)
            feat_buf.append(feat)
            scount += 1
            if len(feat_buf) == L and (scount % stride == 0):
                feats = np.stack(feat_buf)[None].astype(np.float32)   # [1,T,C]
                t1 = time.time()
                logit = head.run(None, {head_in: feats})[0].reshape(-1)[0]
                head_ms.append((time.time() - t1) * 1000)
                prob = 1.0 / (1.0 + np.exp(-logit))
                end_time = fcount / src_fps
                curve.append((end_time, float(prob)))
                run = run + 1 if prob >= thr else 0
                if run >= consec and fired_time is None:
                    fired_time = end_time
        fcount += 1
    cap.release()
    rk.release()

    if fired_time is not None:
        print(f"ACCIDENT DETECTED at t = {fired_time:.2f}s "
              f"(threshold={thr:.2f}, consec={consec})")
    else:
        print(f"no accident detected (threshold={thr:.2f})")
    if bb_ms:
        print(f"NPU backbone: {np.mean(bb_ms):.2f} ms/frame "
              f"({1000/np.mean(bb_ms):.0f} fps) | CPU head: {np.mean(head_ms):.2f} ms/window")
    if args.dump_curve:
        with open(args.dump_curve, "w") as f:
            f.write("end_time,prob\n")
            for t, p in curve:
                f.write(f"{t:.3f},{p:.4f}\n")
        print(f"curve -> {args.dump_curve}")


if __name__ == "__main__":
    main()
