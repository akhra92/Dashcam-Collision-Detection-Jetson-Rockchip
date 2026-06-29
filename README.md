# Dashcam Collision Detection — Edge (Jetson + Rockchip)

Real-time **temporal** accident detection on dashcam video — it detects ***when*** a
collision happens, not just whether a clip contains one. A causal 1-second window
slides across the stream to produce a `P(accident)` curve; a detection fires when the
curve stays high. Trained on the **Nexar Collision Prediction** dataset and deployed to
two edge targets: **PyTorch → ONNX → TensorRT (Jetson Orin)** and **→ RKNN (Rockchip
RK3588)**.

![Dashcam collision detection demo](assets/demo.gif)

*Held-out clip (left) and the live `P(accident)` score (right): flat during normal
driving, spiking the instant the collision unfolds (~19.1 s) and firing within ~0.1 s
of the labelled time. Reproduce: `python -m tools.demo_gif make --id 00168`.*

## Models (Hugging Face Hub)

Each repo holds the ONNX graph(s), an inference-config sidecar, and a model card.

| Model | Target / best for | 🤗 Hub |
|---|---|---|
| **VideoMAE-base + HNM** ⭐ | Jetson — accuracy | [jetson-videomae-hnm](https://huggingface.co/akhra92/dashcam-collision-jetson-videomae-hnm) |
| **R(2+1)D-18** | Jetson — lightweight | [jetson-r2plus1d18](https://huggingface.co/akhra92/dashcam-collision-jetson-r2plus1d18) |
| **ResNet18 + motion ×3** ⭐ | Rockchip — best on NPU | [rockchip-resnet18-motion](https://huggingface.co/akhra92/dashcam-collision-rockchip-resnet18-motion) |
| **MobileNetV3-small** | Rockchip — lightest | [rockchip-mnv3s](https://huggingface.co/akhra92/dashcam-collision-rockchip-mnv3s) |

```python
from huggingface_hub import hf_hub_download
onnx = hf_hub_download("akhra92/dashcam-collision-jetson-r2plus1d18", "model.onnx")
# Rockchip repos are split graphs (backbone.onnx + temporal_head.onnx) — see the card
```

## Live demo

[**dashcam-collision-detection-jetson-rockchip.streamlit.app**](https://dashcam-collision-detection-jetson-rockchip.streamlit.app/)
— upload a clip (or try the bundled sample) and watch the curve + detected time. Runs
the R(2+1)D model on CPU via ONNXRuntime. Locally: `pip install -r requirements.txt &&
streamlit run app.py`.

---

## How it works

The model classifies a **1-second causal window** (16 frames @ 16 fps, ending "now") as
accident/normal. Sliding it over a video gives a probability curve; a detection fires
after `consec=3` consecutive windows above threshold — the exact streaming computation
an edge device runs on a live camera. A training **sample is a window, not a video**.

It is **temporal, not scene-level**: negatives include the normal-driving portions of
accident videos, so the model learns the *event*, not the *scene*. For a window with
end-time `t` in an accident video with event `e`:

| window end-time `t` | label |
|---|---|
| `e − 1.0s ≤ t ≤ e + 0.5s` | **positive** |
| `e + 0.5s < t ≤ e + 2.5s` | ignored (ambiguous aftermath) |
| otherwise (and all non-accident video) | **negative** |

A detection is **correct** if it fires within **±1.0 s** of `time_of_event`.

```
MP4 (~40s) ─[preprocess]→ uint8 strip ─[window]→ [3,16,112,112] ─[model]→ logit
         → sigmoid → P(accident) curve → detect() → "ACCIDENT at t=19.3s"
```
Split is stratified **at the video level** (85/15) so no clip straddles train/val. The
Kaggle `dataset/test/` clips are unlabeled — all metrics use the held-out 15 %.

---

## Results

Held-out **225-video** split, scored by streaming over the **full ~40 s videos**
(`src/eval_fullvideo.py`) — the realistic deployment condition.

**Jetson Orin** (3D-CNN / ViT over the window):

| Model | Detection @ low-FAR | FA floor | Loc. err | Size |
|---|---|---|---|---|
| **VideoMAE-base + HNM** ⭐ | **65.2 %** @ 8.8 % | **8.8 %** | 0.31 s | 86 M ViT |
| R(2+1)D-18 (lightweight) | 63.4 % @ 14.2 % | ~8 % | 0.31 s | 33 M CNN |
| VideoMAE-base (no HNM) | — | 23.9 % | 0.42 s | 86 M ViT |

**Rockchip RK3588** (2D-CNN per frame + temporal head; no 3D convs):

| Model | Detection @ low-FAR | Loc. err | Input |
|---|---|---|---|
| **resnet18 + motion ×3** ⭐ | **35.7 %** @ 11.5 % | 0.30 s | RGB + diffs @ lags 1,2,4 (12 ch) |
| resnet18 + motion (1 lag) | 30 % @ 13 % | 0.38 s | RGB + diff (6 ch) |
| resnet18 (RGB only) | 18 % @ 12 % | 0.34 s | RGB (3 ch) |
| mnv3s (RGB only) | 22 % @ 14 % | 0.36 s | RGB (3 ch) |

The RK3588 models trail the 3D models (~36 % vs ~63 %): a per-frame 2D CNN sees
*appearance* but not *motion*. Feeding **temporal-difference channels** (`motion_lags`)
was the decisive lever — multi-scale diffs **doubled** detection over RGB-only
(18 % → 36 %). Motion modelling, not backbone size, is the bottleneck; the residual gap
is the cost of dropping 3D convs to fit the NPU.

**Engineering notes** (baked into the configs):
- **Full-timeline hard negatives** (`extract_negatives.py`) — without them the model
  false-alarms on long streams.
- **Select the checkpoint by val AP including those negatives** (`monitor: val_ap`), not
  near-event AUC.
- **Hard-negative mining** (`mine_hard_negatives.py`) dropped VideoMAE's false-alarm
  floor 23.9 % → 8.8 %.
- Always judge with `eval_fullvideo` (full videos), never the optimistic strip eval.

The tuned `detect_threshold` lives in `best.pt` and the ONNX `.meta.json`; raise it to
trade detection for fewer false alarms.

---

## Architecture

One interface (`src/model.py`), single-logit output per window:

| `model.arch` | Type | Params | Notes |
|---|---|---|---|
| `r2plus1d_18` (default) | 3D-CNN | 33 M | cleanest TensorRT export |
| `s3d`, `mc3_18`, `r3d_18` | 3D-CNN | 8–33 M | lighter 3D CNNs |
| `videomae_base` / `_large` | ViT | 86 / 300 M | highest accuracy; `transformers==4.46.3` |
| `mnv3s_temporal` / `mnv3l` / `resnet18_temporal` | 2D-CNN + temporal head | 1.6–12 M | **no 3D convs** → RK3588 NPU |

The 2D-CNN family takes optional `input.motion: true` + `motion_lags: [1,2,4]`, feeding
RGB **plus** temporal differences (3 + 3·n_lags channels) for multi-scale motion. Each
diff references the current frame, so the backbone stays per-frame (one NPU call/frame).
Head options: `tconv` / `gru` / `tpool`.

**Why two models:** the RK3588 NPU lacks 3D convolutions, so it runs a 2D-CNN-per-frame
+ temporal head instead of the Jetson's 3D-CNN/ViT. The detection *framing* is identical
— only the per-window classifier and export target (TensorRT vs RKNN) differ.

---

## Setup

```bash
# Training PC (RTX 4060, conda env `myenv`, Python 3.11, torch 2.5.0+cu124)
conda activate myenv && pip install -r requirements-train.txt
```
Jetson uses JetPack's CUDA/TensorRT (`pip install -r deploy/jetson/requirements-jetson.txt`);
Rockchip conversion needs a separate env (see `deploy/rockchip/requirements-rockchip.txt`).
Run all commands from the project root with `python -m …`.

## Reproduce

```bash
# Data prep (once per resolution)
python -m src.preprocess        --config configs/jetson_r2plus1d.yaml --split train
python -m src.extract_negatives --config configs/jetson_r2plus1d.yaml

# Train + evaluate (swap the config for any model)
python -m src.train          --config configs/jetson_r2plus1d.yaml
python -m src.eval_fullvideo --config configs/jetson_r2plus1d.yaml

# VideoMAE accuracy path: train base → mine hard negatives → fine-tune
python -m src.train               --config configs/jetson_videomae.yaml
python -m src.mine_hard_negatives --config configs/jetson_videomae.yaml
python -m src.train               --config configs/jetson_videomae_hnm.yaml
```

## Deployment

**Jetson Orin (TensorRT)** — build on the device (engines are hardware-specific):
```bash
python -m src.export_onnx --config configs/jetson_videomae_hnm.yaml     # on the PC
./deploy/jetson/build_engine.sh model.onnx fp16                         # on the Jetson
python3 deploy/jetson/infer_trt.py --engine model_fp16.engine \
    --video clip.mp4 --meta model.meta.json --dump-curve curve.csv
```

**Rockchip RK3588 (RKNN)** — split deploy: 2D backbone on the NPU (INT8), temporal head
on the CPU.
```bash
python -m src.export_rockchip --config configs/rockchip_resnet18_motion3.yaml   # on the PC
#   -> backbone.onnx [1,12,112,112] + temporal_head.onnx [1,T,C] + rockchip.meta.json

python deploy/rockchip/convert_rknn.py --backbone backbone.onnx \
    --meta rockchip.meta.json --strips artifacts/clips/train --out backbone.rknn   # x86 host

python3 deploy/rockchip/infer_rknn.py --backbone backbone.rknn \
    --head temporal_head.onnx --meta rockchip.meta.json --video clip.mp4           # on the RK3588
```
**Sign-off (both):** compare the device prob curve to the PyTorch reference
(`src/detect_video.py`) on the same clip and confirm per-frame latency.

---

## Repository layout
```
configs/         one YAML per model (jetson_*, rockchip_*)
src/             preprocess, extract_negatives, dataset, model, train, eval_fullvideo,
                 detect_video, mine_hard_negatives, export_onnx, export_rockchip, push_to_hf
deploy/jetson/   build_engine.sh, infer_trt.py, int8_calibrator.py
deploy/rockchip/ convert_rknn.py (x86 INT8), infer_rknn.py (NPU + CPU streaming)
tools/           demo_gif.py
artifacts/, dataset/   data, checkpoints, ONNX (gitignored)
```

## Status
- ✅ Data pipeline, 4 trained models (R(2+1)D, VideoMAE±HNM, Rockchip mnv3s/resnet18-motion),
  full-video evaluation, parity-checked ONNX export, all 4 models published to HF.
- ⏳ **Jetson:** build + benchmark the TensorRT engine on a physical Orin.
- ⏳ **Rockchip:** benchmark the INT8 `backbone.rknn` on a physical RK3588.
- **Accuracy headroom:** Rockchip ~36 % vs Jetson ~63 % detection — next levers are
  optical-flow/two-stream input or an NPU-friendly (2+1)D temporal block.
