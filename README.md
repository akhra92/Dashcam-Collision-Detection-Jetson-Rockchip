# Dashcam Accident Detection

Edge-deployable accident/collision detection on the Nexar dashcam dataset.
Two device targets, two architectures:

- **Model #1 — Jetson Orin:** R(2+1)D 3D-CNN → ONNX → TensorRT. *(implemented, deployment default)*
  - Also selectable: **VideoMAE** ViT (`configs/jetson_videomae.yaml`) — higher
    classifier AP but a higher false-alarm floor on full videos; see `RESULTS.md`.
- **Model #2 — Rockchip RK3588:** 2D-CNN + temporal head → ONNX → RKNN. *(planned)*

Docs:
- **[PIPELINE.md](PIPELINE.md)** — *concepts*: how preprocessing, training, testing,
  ONNX, and TensorRT/RKNN conversion work, with the data-shape journey.
- **[WORKFLOW.md](WORKFLOW.md)** — *commands & order*: the exact steps to run, for
  both devices, plus the acceptance checklist.
- **[RESULTS.md](RESULTS.md)** — *measurements*: v1/v2/v3 ablation and the
  R(2+1)D vs VideoMAE comparison.

## Quick start (Model #1)
```bash
conda activate myenv
pip install -r requirements-train.txt
python -m src.preprocess        --config configs/jetson_r2plus1d.yaml --split train
python -m src.extract_negatives --config configs/jetson_r2plus1d.yaml   # full-timeline negs
python -m src.train             --config configs/jetson_r2plus1d.yaml
python -m src.evaluate          --config configs/jetson_r2plus1d.yaml   # 8s-strip eval
python -m src.eval_fullvideo    --config configs/jetson_r2plus1d.yaml   # realistic full-video eval
python -m src.detect_video      --config configs/jetson_r2plus1d.yaml --video dataset/train/00822.mp4
python -m src.export_onnx       --config configs/jetson_r2plus1d.yaml
```
Then build the TensorRT engine on the Jetson — see WORKFLOW.md §J6.

## Current result (Model #1, full-video val)
Best model = **VideoMAE-base + hard-negative mining**: detection **65.2%** within
±1 s at **8.8%** false-alarm rate (or 68.8% @ 13.3%), localization **0.31 s**.
Lightweight fallback = **R(2+1)D-18**: 63.4% @ 14.2%. See `RESULTS.md` for the full
head-to-head (R(2+1)D vs VideoMAE vs VideoMAE+HNM).
