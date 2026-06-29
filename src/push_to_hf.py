"""Push an exported ONNX model (+ meta sidecar + model card) to the Hugging Face Hub.

This is the publish step after src/export_onnx.py. The ONNX graph is the
device-agnostic hand-off artifact; hosting it on the Hub lets the Jetson /
Rockchip targets pull a versioned model with `hf_hub_download` instead of
shipping weights through git (our GitHub .gitignore excludes *.onnx on purpose).

Auth: run `huggingface-cli login` once, or set HF_TOKEN in the environment.

Examples
--------
    python -m src.push_to_hf --config configs/jetson_videomae_hnm.yaml \
        --repo-id akhra92/dashcam-collision-detector
    # explicit file + private repo
    python -m src.push_to_hf --onnx artifacts/runs/exp/model.onnx \
        --repo-id akhra92/dashcam-collision-detector --private
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from src.config import load_config


def build_model_card(repo_id: str, meta: dict | None, cfg=None) -> str:
    """Render a README.md model card with YAML front-matter from the meta sidecar."""
    arch = (meta or {}).get("arch") or (cfg.model.arch if cfg else "unknown")
    shape = (meta or {}).get("input_shape", [1, 3, 16, 224, 224])
    fps = (meta or {}).get("target_fps", "?")
    win = (meta or {}).get("window_frames", "?")
    thr = (meta or {}).get("detect_threshold", "?")
    consec = (meta or {}).get("consec", "?")

    meta_block = ""
    if meta:
        meta_block = (
            "\n## Inference metadata\n\n```json\n"
            + json.dumps(meta, indent=2)
            + "\n```\n"
        )

    return f"""---
library_name: onnx
pipeline_tag: video-classification
tags:
  - onnx
  - video-classification
  - accident-detection
  - dashcam
  - jetson
  - rockchip
license: mit
---

# Dashcam Collision Detector ({arch}, ONNX)

Causal sliding-window crash detector exported to ONNX. The model scores a short
temporal window of dashcam frames and a downstream rule (`consec` consecutive
detections above `detect_threshold`) decides *when* a collision occurs.

- **Architecture:** `{arch}`
- **Input:** float32 `{shape}` (NCTHW), RGB, ImageNet mean/std normalized
- **Sampling:** {win}-frame window at {fps} fps
- **Decision rule:** threshold `{thr}`, `{consec}` consecutive windows
{meta_block}
## Usage

```python
from huggingface_hub import hf_hub_download
import onnxruntime as ort, numpy as np

path = hf_hub_download(repo_id="{repo_id}", filename="model.onnx")
sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
x = np.random.randn(*{shape}).astype("float32")
logit = sess.run(["logit"], {{"input": x}})[0]
```

On-device, the ONNX graph is compiled to a TensorRT engine (Jetson) or an
`.rknn` model (Rockchip). See `deploy/` in the source repo.
"""


def build_rockchip_card(repo_id: str, meta: dict) -> str:
    """Model card for the Rockchip split export (backbone + temporal head)."""
    arch = meta.get("arch", "unknown")
    fs = meta.get("frame_shape", [1, 3, 112, 112])
    C = meta.get("feat_dim", "?")
    T = meta.get("window_frames", "?")
    lags = meta.get("motion_lags", [])
    thr = meta.get("detect_threshold", "?")
    consec = meta.get("consec", "?")
    motion = "RGB + temporal diffs at lags %s" % lags if lags else "RGB only"
    return f"""---
library_name: onnx
pipeline_tag: video-classification
tags:
  - onnx
  - video-classification
  - accident-detection
  - dashcam
  - rockchip
  - rk3588
license: mit
---

# Dashcam Collision Detector — Rockchip RK3588 (`{arch}`)

Causal sliding-window crash detector for the **RK3588 NPU**. Because the NPU has no
3D convolutions, this is a per-frame 2D-CNN + a small temporal head, deployed as
**two ONNX graphs** (convert the backbone to INT8 RKNN; run the head on the CPU):

| file | shape | runs on |
|---|---|---|
| `backbone.onnx` | `{fs}` → `[1, {C}]` | NPU (INT8 RKNN) |
| `temporal_head.onnx` | `[1, {T}, {C}]` → `[1]` | CPU |
| `rockchip.meta.json` | — | inference config |

- **Input:** {motion} ({fs[1]} channels), {T}-frame window
- **Decision rule:** threshold `{thr}`, `{consec}` consecutive windows

## Usage

```python
from huggingface_hub import hf_hub_download
import onnxruntime as ort, numpy as np, json

repo = "{repo_id}"
bb   = ort.InferenceSession(hf_hub_download(repo, "backbone.onnx"))
head = ort.InferenceSession(hf_hub_download(repo, "temporal_head.onnx"))
meta = json.load(open(hf_hub_download(repo, "rockchip.meta.json")))

T, C = meta["window_frames"], meta["feat_dim"]
feats = np.zeros((1, T, C), np.float32)               # fill from per-frame backbone
frame = np.random.randn(*meta["frame_shape"]).astype("float32")
feats[0, -1] = bb.run(["feat"], {{"frame": frame}})[0][0]
logit = head.run(["logit"], {{"feats": feats}})[0]
```

See `deploy/rockchip/` (convert_rknn.py, infer_rknn.py) in the source repo for the
INT8 conversion and streaming inference.
"""


def push_rockchip(args, cfg):
    out_dir = Path(cfg.paths.output_dir) / cfg.experiment_name
    files = ["backbone.onnx", "temporal_head.onnx", "rockchip.meta.json"]
    missing = [f for f in files if not (out_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {missing} in {out_dir} "
                                f"(run src.export_rockchip first)")
    meta = json.loads((out_dir / "rockchip.meta.json").read_text())

    from huggingface_hub import HfApi
    api = HfApi(token=args.token)
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    print(f"repo ready -> https://huggingface.co/{args.repo_id}")

    for f in files:
        api.upload_file(path_or_fileobj=str(out_dir / f), path_in_repo=f,
                        repo_id=args.repo_id, repo_type="model")
        print(f"uploaded {f}")
    if not args.no_card:
        api.upload_file(path_or_fileobj=build_rockchip_card(args.repo_id, meta).encode(),
                        path_in_repo="README.md", repo_id=args.repo_id, repo_type="model")
        print("uploaded README.md (model card)")
    print(f"done -> https://huggingface.co/{args.repo_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="Config to resolve default ONNX path (optional if --onnx given).")
    ap.add_argument("--onnx", default=None, help="Path to the .onnx file to upload.")
    ap.add_argument("--repo-id", required=True, help="e.g. akhra92/dashcam-collision-detector")
    ap.add_argument("--filename", default="model.onnx", help="Path of the file in the repo.")
    ap.add_argument("--private", action="store_true", help="Create the repo as private.")
    ap.add_argument("--no-card", action="store_true", help="Skip generating/uploading README.md.")
    ap.add_argument("--rockchip", action="store_true",
                    help="Upload the Rockchip split export (backbone + temporal head + meta).")
    ap.add_argument("--token", default=None, help="HF token (else uses login cache / HF_TOKEN).")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else None

    if args.rockchip:
        if cfg is None:
            ap.error("--rockchip needs --config")
        push_rockchip(args, cfg)
        return

    # ---- resolve the ONNX path the same way export_onnx.py writes it ----
    if args.onnx:
        onnx_path = Path(args.onnx)
    elif cfg is not None:
        out_dir = Path(cfg.paths.output_dir) / cfg.experiment_name
        onnx_path = out_dir / f"{cfg.experiment_name}.onnx"
    else:
        ap.error("provide --onnx or --config to locate the model")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX not found: {onnx_path} (run src.export_onnx first?)")

    meta_path = onnx_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else None

    # Imported here so `--help` works without huggingface_hub installed.
    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    print(f"repo ready -> https://huggingface.co/{args.repo_id}")

    api.upload_file(
        path_or_fileobj=str(onnx_path),
        path_in_repo=args.filename,
        repo_id=args.repo_id,
        repo_type="model",
    )
    print(f"uploaded {onnx_path}  ->  {args.filename}")

    if meta is not None:
        api.upload_file(
            path_or_fileobj=str(meta_path),
            path_in_repo=Path(args.filename).with_suffix(".meta.json").name,
            repo_id=args.repo_id,
            repo_type="model",
        )
        print(f"uploaded {meta_path.name}")

    if not args.no_card:
        card = build_model_card(args.repo_id, meta, cfg)
        api.upload_file(
            path_or_fileobj=card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
        )
        print("uploaded README.md (model card)")

    print(f"done -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
