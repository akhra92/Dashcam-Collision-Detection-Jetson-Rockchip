"""
Streaming temporal accident detection with TensorRT on the Jetson Orin.

Slides a causal 1s window (16 frames @ target_fps) across a dashcam video,
runs the TensorRT engine per window to get a P(accident) curve, and fires a
detection when `consec` consecutive windows exceed the tuned threshold. Prints
the detected accident time + a latency benchmark, and can dump the curve.

This mirrors exactly how the model would run live on a camera stream.

Usage:
    python3 infer_trt.py --engine model_fp16.engine --video clip.mp4 \
        --meta model.meta.json [--dump-curve curve.csv]
"""
from __future__ import annotations
import argparse
import json
import time
from collections import deque

import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda


class TRTModel:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.in_name = self.engine.get_tensor_name(0)
        self.out_name = self.engine.get_tensor_name(1)
        self.d_in = self.d_out = None

    def infer(self, x):
        x = np.ascontiguousarray(x.astype(np.float32))
        self.ctx.set_input_shape(self.in_name, x.shape)
        out = np.empty(self.ctx.get_tensor_shape(self.out_name), dtype=np.float32)
        if self.d_in is None:
            self.d_in = cuda.mem_alloc(x.nbytes)
            self.d_out = cuda.mem_alloc(out.nbytes)
            self.ctx.set_tensor_address(self.in_name, int(self.d_in))
            self.ctx.set_tensor_address(self.out_name, int(self.d_out))
        cuda.memcpy_htod_async(self.d_in, x, self.stream)
        self.ctx.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(out, self.d_out, self.stream)
        self.stream.synchronize()
        return out.reshape(-1)


def preprocess_frame(fr, S):
    h, w = fr.shape[:2]
    short = S
    if h <= w:
        nh, nw = short, int(round(w * short / h))
    else:
        nh, nw = int(round(h * short / w)), short
    fr = cv2.resize(fr, (nw, nh))
    top, left = (nh - S) // 2, (nw - S) // 2
    fr = fr[top:top + S, left:left + S]
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--dump-curve", default=None)
    args = ap.parse_args()

    meta = json.loads(open(args.meta).read())
    L = meta["window_frames"]
    S = meta["input_shape"][3]
    tgt_fps = meta["target_fps"]
    thr = meta["detect_threshold"]
    consec = meta["consec"]
    stride = meta.get("stride", 2)
    mean = np.array(meta["mean"]); std = np.array(meta["std"])

    model = TRTModel(args.engine)

    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / tgt_fps)))   # source frames per strip frame

    buf = deque(maxlen=L)                            # rolling window of preprocessed frames
    curve, run, fired_time = [], 0, None
    fcount, scount, latencies = 0, 0, []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fcount % step == 0:                       # subsample to target_fps
            buf.append(preprocess_frame(fr, S))
            scount += 1
            if len(buf) == L and (scount % stride == 0):
                clip = (np.stack(buf).astype(np.float32) / 255.0 - mean) / std
                x = clip.transpose(3, 0, 1, 2)[None]  # [1,3,L,S,S]
                t0 = time.time()
                logit = model.infer(x)[0]
                latencies.append(time.time() - t0)
                prob = 1.0 / (1.0 + np.exp(-logit))
                end_time = fcount / src_fps
                curve.append((end_time, float(prob)))
                run = run + 1 if prob >= thr else 0
                if run >= consec and fired_time is None:
                    fired_time = end_time
        fcount += 1
    cap.release()

    if fired_time is not None:
        print(f"ACCIDENT DETECTED at t = {fired_time:.2f}s "
              f"(threshold={thr:.2f}, consec={consec})")
    else:
        print(f"no accident detected (threshold={thr:.2f})")
    if latencies:
        ms = np.mean(latencies) * 1000
        print(f"per-window latency: {ms:.2f} ms  ({1000/ms:.1f} windows/s)")
    if args.dump_curve:
        with open(args.dump_curve, "w") as f:
            f.write("end_time,prob\n")
            for t, p in curve:
                f.write(f"{t:.3f},{p:.4f}\n")
        print(f"curve -> {args.dump_curve}")


if __name__ == "__main__":
    main()
