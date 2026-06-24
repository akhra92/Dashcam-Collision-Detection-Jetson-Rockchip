"""Export the trained PyTorch checkpoint to ONNX and verify numerical parity.

The ONNX graph is the device-agnostic hand-off artifact. On the Jetson it is
compiled into a TensorRT engine (see deploy/jetson/). We verify the ONNX output
matches PyTorch within tolerance before shipping.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    out_dir = Path(cfg.paths.output_dir) / cfg.experiment_name
    ckpt_path = Path(args.ckpt) if args.ckpt else out_dir / "best.pt"
    onnx_path = Path(args.out) if args.out else out_dir / f"{cfg.experiment_name}.onnx"

    model = build_model(cfg).eval()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    T, S = cfg.input.num_frames, cfg.input.crop_size
    dummy = torch.randn(1, 3, T, S, S)

    dynamic_axes = {"input": {0: "batch"}, "logit": {0: "batch"}} \
        if cfg.export.dynamic_batch else None

    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["logit"],
        opset_version=cfg.export.opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    print(f"exported -> {onnx_path}")

    if cfg.export.simplify:
        try:
            import onnx
            from onnxsim import simplify
            m = onnx.load(str(onnx_path))
            m_simp, ok = simplify(m)
            if ok:
                onnx.save(m_simp, str(onnx_path))
                print("onnx-sim: simplified graph saved")
        except Exception as e:
            print(f"onnx-sim skipped: {e}")

    # ---- parity check vs ONNXRuntime ----
    import onnxruntime as ort
    with torch.no_grad():
        torch_out = model(dummy).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(["logit"], {"input": dummy.numpy()})[0]
    diff = np.abs(torch_out.reshape(-1) - ort_out.reshape(-1)).max()
    print(f"max |torch - onnx| = {diff:.3e}")
    assert diff < 1e-3, "ONNX parity check failed!"
    print("parity OK")

    meta = {
        "input_shape": [1, 3, T, S, S],
        "detect_threshold": ckpt.get("detect_threshold", cfg.detect.threshold),
        "consec": cfg.detect.consec,
        "target_fps": cfg.strip.target_fps,
        "window_frames": T,
        "stride": cfg.window.stride,
        "tolerance_s": cfg.detect.tolerance_s,
        "mean": cfg.input.mean, "std": cfg.input.std,
        "arch": cfg.model.arch,
    }
    (onnx_path.with_suffix(".meta.json")).write_text(__import__("json").dumps(meta, indent=2))
    print(f"meta -> {onnx_path.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
