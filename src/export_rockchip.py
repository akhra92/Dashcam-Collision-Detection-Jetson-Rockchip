"""Export the Rockchip (2D-CNN + temporal head) model to TWO static ONNX graphs.

The RK3588 NPU wants 4D tensors and no 3D convs, so we split the model exactly
where it deploys:

  * backbone.onnx     [1,3,S,S]      -> [1,C]   — the per-frame 2D CNN. This is
                                                  what rknn-toolkit2 quantizes to
                                                  INT8 and runs on the NPU.
  * temporal_head.onnx[1,T,C]        -> [1]     — aggregates T frame features into
                                                  one logit. Tiny; runs on CPU.

At inference (deploy/rockchip/infer_rknn.py) each new frame is pushed through the
NPU backbone once; the last T features are fed to the head every `stride` frames.

We verify the split pipeline (backbone per-frame + head) matches the full PyTorch
model within tolerance before shipping.

Usage:
    python -m src.export_rockchip --config configs/rockchip_mobilenet.yaml
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.config import load_config
from src.model import build_model


class _HeadWrap(nn.Module):
    """Expose the temporal head as [1,T,C] -> [1] for a clean ONNX signature."""

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, feats):                 # [B,T,C]
        return self.head(feats).reshape(-1)   # [B]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    if not cfg.model.arch.endswith("_temporal") and cfg.model.arch not in (
            "mnv3s_temporal", "mnv3l_temporal", "resnet18_temporal"):
        raise SystemExit(f"{cfg.model.arch} is not a Rockchip 2D+temporal model; "
                         f"use src.export_onnx for the 3D/ViT models.")

    out_dir = Path(cfg.paths.output_dir) / cfg.experiment_name
    ckpt_path = Path(args.ckpt) if args.ckpt else out_dir / "best.pt"
    exp_dir = Path(args.outdir) if args.outdir else out_dir
    exp_dir.mkdir(parents=True, exist_ok=True)
    bb_path = exp_dir / "backbone.onnx"
    head_path = exp_dir / "temporal_head.onnx"

    model = build_model(cfg).eval()
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"loaded weights from {ckpt_path}")
    else:
        ckpt = {}
        print(f"WARNING: {ckpt_path} not found — exporting an untrained model "
              f"(graph/parity check only).")

    f2t = model.backbone                       # Frame2DTemporal
    backbone = f2t.backbone.eval()             # [N,Cin,S,S] -> [N,C]
    head = _HeadWrap(f2t.temporal).eval()      # [1,T,C]   -> [1] (eval: dropout off)

    T, S, C = cfg.input.num_frames, cfg.input.crop_size, f2t.c_out
    Cin = f2t.in_chans                         # 3 (RGB) or 6 (RGB + motion diff)
    opset = cfg.export.opset

    # ---- export backbone (per-frame, static [1,Cin,S,S]) ----
    torch.onnx.export(
        backbone, torch.randn(1, Cin, S, S), str(bb_path),
        input_names=["frame"], output_names=["feat"],
        opset_version=opset, do_constant_folding=True,
    )
    print(f"backbone -> {bb_path}  ([1,{Cin},{S},{S}] -> [1,{C}])")

    # ---- export temporal head (static [1,T,C]) ----
    torch.onnx.export(
        head, torch.randn(1, T, C), str(head_path),
        input_names=["feats"], output_names=["logit"],
        opset_version=opset, do_constant_folding=True,
    )
    print(f"head     -> {head_path}  ([1,{T},{C}] -> [1])")

    if cfg.export.simplify:
        try:
            import onnx
            from onnxsim import simplify
            for p in (bb_path, head_path):
                m, ok = simplify(onnx.load(str(p)))
                if ok:
                    onnx.save(m, str(p))
            print("onnx-sim: simplified both graphs")
        except Exception as e:
            print(f"onnx-sim skipped: {e}")

    # ---- parity: split (backbone per-frame + head) vs full PyTorch model ----
    import onnxruntime as ort
    dummy = torch.randn(1, Cin, T, S, S)
    with torch.no_grad():
        ref = model(dummy).numpy().reshape(-1)

    bb_sess = ort.InferenceSession(str(bb_path), providers=["CPUExecutionProvider"])
    head_sess = ort.InferenceSession(str(head_path), providers=["CPUExecutionProvider"])
    frames = dummy[0].permute(1, 0, 2, 3).numpy()           # [T,3,S,S]
    feats = np.stack([bb_sess.run(["feat"], {"frame": frames[i:i + 1]})[0][0]
                      for i in range(T)])                    # [T,C]
    logit = head_sess.run(["logit"], {"feats": feats[None].astype(np.float32)})[0].reshape(-1)
    diff = float(np.abs(ref - logit).max())
    print(f"max |torch - onnx(split)| = {diff:.3e}")
    assert diff < 1e-3, "Rockchip split parity check failed!"
    print("parity OK")

    from src.dataset import MOTION_SCALE, motion_enabled, motion_lags
    motion = motion_enabled(cfg)
    meta = {
        "frame_shape": [1, Cin, S, S],
        "in_channels": Cin,
        "motion": motion,
        "motion_lags": motion_lags(cfg),
        "motion_scale": MOTION_SCALE,
        "feat_dim": C,
        "window_frames": T,
        "stride": cfg.window.stride,
        "target_fps": cfg.strip.target_fps,
        "detect_threshold": ckpt.get("detect_threshold", cfg.detect.threshold),
        "consec": cfg.detect.consec,
        "tolerance_s": cfg.detect.tolerance_s,
        "mean": cfg.input.mean, "std": cfg.input.std,
        "arch": cfg.model.arch,
        "temporal_head": cfg.model.get("temporal_head", "tconv"),
    }
    if motion:
        # 6ch signed input -> infer_rknn pre-normalises on CPU; RKNN passes through.
        meta["rknn_mean_values"] = [0.0] * Cin
        meta["rknn_std_values"] = [1.0] * Cin
    else:
        # rknn-toolkit2 applies (x - mean*255)/(std*255) to raw 0-255 RGB input.
        meta["rknn_mean_values"] = [round(m * 255.0, 4) for m in cfg.input.mean]
        meta["rknn_std_values"] = [round(s * 255.0, 4) for s in cfg.input.std]
    (exp_dir / "rockchip.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"meta     -> {exp_dir / 'rockchip.meta.json'}")


if __name__ == "__main__":
    main()
