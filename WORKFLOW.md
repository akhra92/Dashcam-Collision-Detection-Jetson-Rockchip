# Dashcam Accident Detection — End-to-End Workflow

Single source of truth for **what to do and when**, for both target devices.

- **Model #1 — NVIDIA Jetson Orin** → status: **ACTIVE / being implemented now**.
- **Model #2 — Rockchip RK3588** → status: **PLANNED** (start only after Model #1
  is fully done and the user gives the go-ahead).

---

## 0. Task definition — TEMPORAL detection (not whole-video classification)

The goal is to detect **WHEN** an accident happens in a dashcam video, as a live
stream would.

**How:** a **causal 1-second window** (16 frames @ 16 fps, ending at "now") is
classified accident/normal. Sliding it across a video produces a
**P(accident)-over-time curve**; the accident is *detected* the moment the curve
stays above a threshold for `consec` consecutive windows. This is exactly the
streaming computation the edge device performs on a live camera.

**Detection settings (confirmed):**
- Causal / online window (past + current frames only) → can warn live.
- Positive time region = **`[time_of_event − 1.0s, time_of_event + 0.5s]`**.
- A detection is **correct** if it fires within **±1.0s** of `time_of_event`.

### Dataset — Nexar Collision Prediction (Kaggle)
- `dataset/train/*.mp4` — 1500 clips, balanced 750 / 750.
- `dataset/test/*.mp4` — 1344 clips, **unlabeled**.
- `dataset/train.csv` — `id, time_of_event, time_of_alert, target`.
  - `time_of_event` = collision moment (s), positives only.
- Video specs: 1280×720, ~30 fps, mostly ~40 s.

### Labeling (this is what makes it temporal, not scene-level)
Negatives include the **normal-driving portions of accident videos** (before the
event), not just non-accident videos. So the model must tell "this same road, 3 s
before the crash" (negative) from "the crash unfolding" (positive) — it learns the
*event*, not the *scene*.

For a window whose **end time** is `t`, in an accident video with event `e`:
| condition | label |
|---|---|
| `e−1.0 ≤ t ≤ e+0.5` | **positive** |
| `e+0.5 < t ≤ e+2.5` | ignored (ambiguous aftermath) |
| otherwise | **negative** |
Non-accident videos: every window is negative.

### Why two different models?

| | Jetson Orin (Model #1) | Rockchip RK3588 (Model #2) |
|---|---|---|
| Accelerator | Ampere GPU + TensorRT | NPU (6 TOPS) + RKNN toolkit2 |
| 3D convolutions | ✅ well supported | ❌ poorly / not supported |
| Architecture | **3D-CNN** R(2+1)D over the window | **2D-CNN per frame + temporal head** (GRU / temporal-pool) |
| Export path | PyTorch → ONNX → TensorRT | PyTorch → ONNX → RKNN |

The detection *framing* (sliding causal window → curve → fire time) is identical
for both; only the per-window classifier architecture differs.

---

## 1. Environment

- **Training PC:** RTX 4060 8 GB, conda env **`myenv`** (Python 3.11,
  torch 2.5.0+cu124, torchvision 0.20). `pip install -r requirements-train.txt`.
- **Jetson Orin:** JetPack provides CUDA + TensorRT (do **not** pip-install them).
  `pip install -r deploy/jetson/requirements-jetson.txt`.

Run all module commands from the project root with `python -m` after
`conda activate myenv`.

---

## ===================  MODEL #1 — JETSON ORIN  ===================

### Architecture — selectable backbones (set `model.arch` in the config)
Two families, one interface (`src/model.py`), both → single logit per window:

| `model.arch` | Type | Input | Params | Notes |
|---|---|---|---|---|
| **`r2plus1d_18`** (default) | 3D-CNN | 16×112×112 | 33 M | factorized (2+1)D convs; lightest, cleanest TensorRT export — best for Orin Nano |
| `s3d`, `mc3_18`, `r3d_18` | 3D-CNN | 16×112/224 | 8–33 M | lighter/alt CNNs |
| **`videomae_base`** | ViT (VideoMAE) | 16×224×224 | 86 M | self-supervised, higher Kinetics accuracy; heavier — needs `transformers==4.46.3` and the 224px data dir |
| `videomae_large`, `videomaev2_base` | ViT | 16×224×224 | 300 M / 86 M | even heavier / VideoMAE-v2 |

- CNN config: `configs/jetson_r2plus1d.yaml` (112px, `artifacts/clips`).
- VideoMAE config: `configs/jetson_videomae.yaml` (224px, `artifacts/clips224`,
  VideoMAE normalization, grad-checkpointing, batch 8).
- **Common:** one causal window, dropout + single logit, `BCEWithLogitsLoss` with
  `pos_weight`; transfer learning, discriminative LR, freeze-warmup, cosine LR +
  warmup, AMP, label smoothing, spatial + temporal-jitter aug.
- **Best accuracy:** `videomae_base` **+ hard-negative mining**
  (`configs/jetson_videomae_hnm.yaml`) is the most accurate Model #1 — 65.2 %
  detection @ 8.8 % false-alarm vs R(2+1)D's 63.4 % @ 14.2 %. Plain VideoMAE without
  HNM false-alarms badly (~24 % floor); HNM fixes it. See `RESULTS.md`.
- **Hard-negative mining (HNM)** — run after a first model to cut false alarms:
  ```bash
  python -m src.mine_hard_negatives --config configs/jetson_videomae.yaml \
      --mine-threshold 0.85 --max-per-video 12 --stride 4
  python -m src.train --config configs/jetson_videomae_hnm.yaml   # fine-tunes from base
  ```
- **Deployment note:** VideoMAE (ViT, 86 M @224) exports to ONNX and runs on Jetson
  via TensorRT but is far heavier than R(2+1)D (33 M @112) — **verify per-window
  latency on your Orin (Step J7)**; use R(2+1)D as the lightweight fallback.

### Step J1 — Extract frame strips  ✅ once
One contiguous strip per video, resampled to 16 fps, centered on the event for
positives. Lets the loader form windows by slicing 16 consecutive frames.
```bash
python -m src.preprocess --config configs/jetson_r2plus1d.yaml --split train
python -m src.preprocess --config configs/jetson_r2plus1d.yaml --split test   # optional
```
→ `artifacts/clips/train/*.npy` + `artifacts/clips/train_meta.csv`.

### Step J1b — Extract full-timeline hard negatives  ✅ once
Samples negative 1s windows from across the WHOLE video (not just the 8s event
strip). **Critical for low false alarms** on long streams — without this the model
only learns "normal" from 8s and over-triggers elsewhere.
```bash
python -m src.extract_negatives --config configs/jetson_r2plus1d.yaml
```
→ `artifacts/clips/neg/*.npy` + `artifacts/clips/neg_manifest.csv`.

### Step J2 — Train
```bash
python -m src.train --config configs/jetson_r2plus1d.yaml
```
- **Video-level** stratified 85/15 split (windows from a video never straddle the
  split → no leakage).
- Window manifest with temporal labels; train negatives balanced to `neg_ratio`.
- Best checkpoint by **val window-AUC** → `artifacts/runs/jetson_r2plus1d18/best.pt`.

### Step J3 — Temporal evaluation
```bash
python -m src.evaluate      --config configs/jetson_r2plus1d.yaml   # fast: 8s-strip scan
python -m src.eval_fullvideo --config configs/jetson_r2plus1d.yaml  # REAL: full 40s videos
```
- `evaluate.py` scans only the 8s strips → fast but **optimistic** on false alarms.
- `eval_fullvideo.py` streams the **entire** video → the realistic false-alarm rate
  you will see in deployment. **Always trust this one for the operating point.**

Both report, per val video, prob curve → detection logic:
- **detection_rate** (positives caught within ±1 s of the event)
- **false_alarm_rate** (negatives that fire anywhere)
- **mean_loc_error**, **mean_lead_time** (how early it warns)
plus window-level AUC / AP. Set the chosen `detect_threshold` in the checkpoint
from the `eval_fullvideo` curve (it sweeps thresholds and prints the best).

> **Lesson learned (baked into the config):** select the checkpoint by **val AP on a
> validation set that includes the full-timeline negatives** (`monitor: val_ap`).
> Selecting by AUC on near-event windows picks an undertrained early epoch that
> false-alarms badly. See `RESULTS.md` for the v1/v2/v3 ablation.

### Step J4 — Demo / sanity check on real videos (PyTorch)
```bash
python -m src.detect_video --config configs/jetson_r2plus1d.yaml \
    --video dataset/train/00822.mp4 --dump-curve curve.csv
```
Streaming detector in PyTorch — also the **parity reference** for the Jetson.

### Step J5 — Export to ONNX (on the PC)
```bash
python -m src.export_onnx --config configs/jetson_r2plus1d.yaml
```
Produces `…onnx` + `.meta.json` (threshold, consec, fps, normalization),
simplifies the graph, and verifies parity vs ONNXRuntime (`|torch−onnx| < 1e-3`).

### Step J6 — Build the TensorRT engine (**on the Jetson**)
Copy `.onnx` + `.meta.json` to the device (engines are hardware/version specific).
```bash
chmod +x deploy/jetson/build_engine.sh
./deploy/jetson/build_engine.sh model.onnx fp16        # recommended
# optional INT8 (≈2× faster — copy a few hundred strips for calibration):
python3 deploy/jetson/int8_calibrator.py --onnx model.onnx --clips ./calib_clips --out model_int8.engine
```

### Step J7 — Streaming inference + benchmark (on the Jetson)
```bash
python3 deploy/jetson/infer_trt.py --engine model_fp16.engine \
    --video some_clip.mp4 --meta model.meta.json --dump-curve curve.csv
```
Prints the detected accident time and per-window latency (windows/s).

### Step J8 — Parity & sign-off
- Compare PyTorch (J4) vs TensorRT (J7) prob curves on the same clips — agree
  within FP16 tolerance, same detected time.
- Confirm per-window latency meets the streaming budget. **Model #1 done.**

### Model #1 acceptance checklist
- [x] J1 strips extracted + `train_meta.csv`
- [x] J1b full-timeline hard negatives extracted + `neg_manifest.csv`
- [x] J2 training converged, `best.pt` (v3, epoch 14, val AP 0.65)
- [x] J3 temporal metrics — full-video: **det 63.4% / FAR 14.2% / loc 0.31s** @ thr 0.68
- [x] J4 streaming demo sane on real clips (all sample accidents caught within ±1s)
- [x] J5 ONNX exported + parity passed (max |Δ| 8e-7)
- [ ] J6 TensorRT engine built on Jetson  ← **runs on the device**
- [ ] J7 on-device streaming detection + latency benchmark  ← **on the device**
- [ ] J8 PyTorch↔TensorRT parity confirmed  ← **on the device**

---

## ===================  MODEL #2 — ROCKCHIP RK3588  (PLANNED)  ===================

> Do not start until Model #1 is signed off and the user says go.

### Planned architecture
- **2D-CNN backbone** per frame (mobilenetv3 / repvgg / resnet18, all RKNN-friendly)
  → per-frame embeddings.
- **Temporal head** from RKNN-supported ops only (temporal pool / 1D-conv / small
  GRU over frame features) — **no 3D convs** (RKNN NPU doesn't support them well).
- Same causal-window → curve → fire-time detection framing as Model #1, so the
  evaluation and metrics are directly comparable.

### Planned steps (mirror Jetson)
1. **R1** reuse the extracted strips (frames identical).
2. **R2** train `configs/rockchip_*.yaml` (2D backbone + temporal head).
3. **R3** temporal evaluation (same metrics for apples-to-apples comparison).
4. **R4** export ONNX — **static shapes** (RKNN dislikes dynamic axes), verify parity.
5. **R5** convert ONNX → RKNN with `rknn-toolkit2` on an x86 host, **INT8** with a
   calibration set.
6. **R6** deploy `.rknn` on RK3588 via `rknn-toolkit-lite2`; benchmark NPU latency.
7. **R7** parity + latency sign-off.

### To confirm before Model #2
- Exact board / OS image and RKNN toolkit version.
- Per-frame streaming (2D CNN every frame + rolling temporal buffer on CPU) vs
  whole-window inference on the NPU.

---

## Directory map
```
configs/        YAML configs (one per model/device)
src/            config, preprocess, dataset, model, train, evaluate,
                detect_video (streaming demo), export_onnx
deploy/jetson/  build_engine.sh, infer_trt.py (streaming), int8_calibrator.py
artifacts/      clips/ (strips + meta), runs/ (ckpts, onnx, metrics)  [gitignored]
dataset/        provided Nexar data
WORKFLOW.md     this file
```
