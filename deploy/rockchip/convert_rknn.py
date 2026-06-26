"""Convert the per-frame backbone ONNX -> INT8 RKNN for the RK3588 NPU.

Runs on an **x86 Linux host** with `rknn-toolkit2` installed (NOT on the board).
Only the 2D backbone is quantized/compiled for the NPU; the tiny temporal head
stays in ONNX and runs on the device CPU (see infer_rknn.py).

INT8 needs a calibration set representative of real frames. This script samples
frames from the pre-extracted strips (artifacts/clips/train/*.npy), writes them as
RGB PNGs + a dataset.txt, then quantizes. rknn applies (x - mean)/std internally,
so we pass the SAME normalization the model was trained with (from the meta file).

Usage (on the x86 host):
    pip install rknn-toolkit2            # from Rockchip's wheel/repo
    python convert_rknn.py --backbone backbone.onnx --meta rockchip.meta.json \
        --strips ../../artifacts/clips/train --out backbone.rknn --num-calib 300
"""
from __future__ import annotations
import argparse
import glob
import json
import os

import cv2
import numpy as np


def _crop(fr, S):
    s = fr.shape[0]
    top = (s - S) // 2
    return fr[top:top + S, top:top + S, :]


def build_calibration(strips_dir, out_dir, S, n_imgs, meta, seed=0):
    """Dump a calibration set + dataset.txt.

    3-channel (RGB): PNGs (rknn normalises raw 0-255 via mean/std config).
    6-channel (motion): pre-normalised [Cin,S,S] float .npy (RGB norm + temporal
    diff), since rknn can't form the diff or load 6-ch images — rknn passes through."""
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(strips_dir, "*.npy")))
    if not files:
        raise SystemExit(f"no .npy strips in {strips_dir}")
    rng = np.random.default_rng(seed)
    listing = os.path.join(out_dir, "dataset.txt")
    motion = meta.get("motion", False)
    mean = np.array(meta["mean"], np.float32).reshape(3, 1, 1)
    std = np.array(meta["std"], np.float32).reshape(3, 1, 1)
    scale = meta.get("motion_scale", 0.5)
    written = 0
    with open(listing, "w") as f:
        while written < n_imgs:
            arr = np.load(rng.choice(files), mmap_mode="r")     # [N,Sx,Sx,3] uint8 RGB
            if not motion:
                fr = _crop(np.asarray(arr[int(rng.integers(0, arr.shape[0]))]), S)
                p = os.path.abspath(os.path.join(out_dir, f"calib_{written:05d}.png"))
                cv2.imwrite(p, cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            else:
                j = int(rng.integers(1, arr.shape[0]))          # need a previous frame
                cur = _crop(np.asarray(arr[j]), S).astype(np.float32) / 255.0
                prev = _crop(np.asarray(arr[j - 1]), S).astype(np.float32) / 255.0
                cur = cur.transpose(2, 0, 1); prev = prev.transpose(2, 0, 1)  # [3,S,S]
                six = np.concatenate([(cur - mean) / std, (cur - prev) / scale], 0)
                p = os.path.abspath(os.path.join(out_dir, f"calib_{written:05d}.npy"))
                np.save(p, six.astype(np.float32))
            f.write(p + "\n")
            written += 1
    print(f"calibration: {written} samples ({'6ch npy' if motion else 'RGB png'}) -> {listing}")
    return listing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, help="backbone.onnx")
    ap.add_argument("--meta", required=True, help="rockchip.meta.json")
    ap.add_argument("--strips", required=True, help="dir of .npy strips for calibration")
    ap.add_argument("--out", default="backbone.rknn")
    ap.add_argument("--num-calib", type=int, default=300)
    ap.add_argument("--target", default="rk3588")
    ap.add_argument("--no-quant", action="store_true", help="export fp16 instead of INT8")
    args = ap.parse_args()

    meta = json.loads(open(args.meta).read())
    S = meta["frame_shape"][-1]
    mean = meta["rknn_mean_values"]                 # 3ch: scaled 0-255 ; 6ch: passthrough
    std = meta["rknn_std_values"]

    dataset = None if args.no_quant else build_calibration(
        args.strips, "rknn_calib", S, args.num_calib, meta)

    from rknn.api import RKNN
    rknn = RKNN(verbose=True)
    # RGB frames in, model trained on RGB -> no RGB2BGR swap. For the 6ch motion
    # model the input is already normalised (passthrough mean/std from the meta).
    rknn.config(mean_values=[mean], std_values=[std],
                target_platform=args.target, quant_img_RGB2BGR=False)

    assert rknn.load_onnx(model=args.backbone) == 0, "load_onnx failed"
    assert rknn.build(do_quantization=not args.no_quant, dataset=dataset) == 0, "build failed"
    assert rknn.export_rknn(args.out) == 0, "export_rknn failed"
    print(f"RKNN backbone -> {args.out} "
          f"({'fp16' if args.no_quant else 'INT8'}, target={args.target})")
    rknn.release()


if __name__ == "__main__":
    main()
