# Pipeline Explained — Concepts & Data Flow

This is the **concepts** doc: *how and why* preprocessing, training, testing, and
device conversion work. For the *commands and the order to run them*, see
**[WORKFLOW.md](WORKFLOW.md)**; for *measured results*, see **[RESULTS.md](RESULTS.md)**.

---

## 0. The mental model (read this first)

The model **never sees a whole video at once.** It classifies a **1-second causal
clip** ("window") as *accident* vs *normal*. To detect *when* a crash happens, we
slide that window across the video and read off a **P(accident)-over-time curve**;
the accident is "detected" the moment the curve stays high briefly.

The whole project is one question asked thousands of times:

> *"Given the last 1 second of video, is a collision happening right now?"*

- **Window** = 16 frames @ 16 fps = exactly 1.0 s, ending at "now" (causal — no
  future frames).
- **Detection** = slide window → curve → fire when `consec=3` consecutive windows
  exceed a threshold.

Key consequence: a training **sample is a window, not a video.** One 40 s video
produces many windows.

---

## 1. Preprocessing — preparing the data on disk

Decoding 720p MP4s during training is far too slow, so we decode **once** into small
`.npy` arrays. Two scripts do this.

### 1a. Event strips — `src/preprocess.py`
One contiguous **strip** of frames per video, around the relevant moment:

- Resample to **16 fps** (`strip.target_fps`), take an **8 s** strip
  (`strip.strip_seconds`) → **128 frames**.
- **Positives:** strip positioned so the collision sits 5 s in
  (`strip.pre_event_seconds`) → spans `[event−5s, event+3s]`: normal driving leading
  up to the crash + the crash + a little aftermath.
- **Negatives** (no-accident videos): a random 8 s window.
- Each frame: resize (short side → 128) + **center-crop 128×128**, stored RGB `uint8`.

Output per video: `artifacts/clips/train/<id>.npy` shape **`[128,128,128,3]`**
(frames, H, W, C) + a row in `artifacts/clips/train_meta.csv` (`strip_start_time`,
`target_fps`, `event_time`, `target`, …). Because the strip is uniform 16 fps, the
real-world time of strip-frame *j* is `strip_start_time + j/16` — that is what lets
us label windows by time later.

### 1b. Full-timeline hard negatives — `src/extract_negatives.py`
The 8 s strip only teaches "normal" in a narrow window around each event, so on a
real 40 s stream the model over-triggers. Fix: for **every** video, sample **16
random 1 s clips** (`negatives.per_video`) from *across the whole video*, excluding a
guard band around the event for accident videos
(`[event − pos_pre − guard, event + ignore_post + guard]`).

Concatenated into one pseudo-strip `artifacts/clips/neg/<id>.npy` shape
**`[256,128,128,3]`** (16 clips × 16 frames); `neg_manifest.csv` lists each as a
labeled negative. **This is the single most important fix for false alarms** (see
`RESULTS.md`).

### 1c. Turning strips into labeled windows — `src/dataset.py`
A window = a strip + a **start index**. `enumerate_windows()` slides the start over a
strip in steps of `window.stride=3` frames. For each window it computes the **end
time** and labels it (`window_label()`) relative to the collision time `e`:

| Window end-time falls in… | Label |
|---|---|
| `[e − 1.0s, e + 0.5s]` (`pos_pre`/`pos_post`) | **positive (1)** |
| `(e + 0.5s, e + 2.5s]` (`ignore_post`) | **ignored** (ambiguous aftermath, dropped) |
| anything else | **negative (0)** |
| any window of a non-accident video | **negative (0)** |

`build_window_manifest()` assembles the final training windows:
- all positives + near-event negatives (from event strips),
- **plus** the full-timeline hard negatives (1b),
- **training**: negatives subsampled to `neg_ratio=3` per positive (so loss isn't
  swamped),
- **validation**: *all* negatives kept → realistic ~8 %-positive imbalance.

### 1d. Train/validation split — `video_level_split()`
The 1500 labeled videos split **85 / 15, stratified by class, at the VIDEO level**
(seed 42). Splitting by video (not window) prevents windows from the same clip
leaking across train/val and inflating scores.

> ⚠️ The Kaggle `dataset/test/` folder (1344 clips) is **unlabeled** — it cannot
> measure accuracy. All accuracy numbers come from the held-out 15 % validation
> split of the labeled training set.

---

## 2. Training — `src/train.py` (+ `dataset.py`, `model.py`)

### 2a. How a training sample is produced — `WindowDataset.__getitem__`
1. memory-map the strip `.npy`, slice 16 frames `[start : start+16]` (lazy read).
2. **Temporal jitter** (train): shift start ±1 frame (`window.temporal_jitter`).
3. **Spatial aug** (train): random-resized-crop → **112×112** (scale 0.7–1.0) + hflip.
4. **Color jitter** (train): small brightness/contrast.
5. to float [0,1], **normalize** (Kinetics mean/std), reorder to channels-first →
   **`[3,16,112,112]`** = (C,T,H,W).
6. return `(tensor, label, id)`.

Validation = same dataset with `train=False` → deterministic **center crop**, no aug.

### 2b. The model — `src/model.py` (`AccidentNet`)
- Backbone **`r2plus1d_18`** (R(2+1)D 3D-CNN), Kinetics-400 pretrained; final `fc`
  replaced with `Dropout(0.5) → Linear(→1)` → **one logit per window**; `forward()`
  returns `[B]`.
- Same class also supports **VideoMAE** (ViT) by permuting input to `[B,T,C,H,W]` —
  identical single-logit interface. (See backbone table in WORKFLOW.md.)

### 2c. Loss & optimization
- **Loss:** `BCEWithLogitsLoss`, `pos_weight = #neg/#pos` (counters 3:1 imbalance) +
  label smoothing 0.05.
- **Discriminative LR:** backbone trains at `0.1×` the head LR (`backbone_lr_mult`).
- **Freeze-warmup:** backbone frozen first `freeze_stem_epochs=2` epochs (head only),
  then unfrozen.
- **Schedule:** cosine LR + 2-epoch warmup; **AMP**; grad-clip 1.0; batch 16; AdamW
  (wd 0.05).

### 2d. Per-epoch evaluation & checkpoint selection
After each epoch `evaluate()` computes val AUC/AP/F1/acc. **Best checkpoint = highest
validation AP** (`monitor: val_ap`) — and that val set **includes the full-timeline
hard negatives**, so the chosen checkpoint is the one that actually avoids false
alarms. (Selecting by AUC on near-event windows once picked an undertrained early
epoch — the v2 regression in `RESULTS.md`.) Saves `best.pt` + `last.pt`; early-stop
patience 8.

---

## 3. Testing / Evaluation — three increasingly realistic views

### 3a. Window-level — in `train.py` / `evaluate.py`
Each val window independently (center crop, no aug). Measures **classifier** quality:
AUC, AP, F1. Fast; says nothing about temporal behaviour.

### 3b. Strip-level temporal — `src/evaluate.py`
`strip_to_windows_tensor()` builds **every** window of a strip → a prob curve over the
8 s strip; `detect()` applies the firing rule. **Optimistic** (only 8 s scanned).

### 3c. Full-video (the real metric) — `src/eval_fullvideo.py`
How deployment actually works:
1. decode the **entire ~40 s video**, subsample to 16 fps;
2. rolling buffer of the last 16 frames (`deque(maxlen=16)`) = true **causal sliding
   window**;
3. every `stride` frames → normalize buffer → `[1,3,16,112,112]` → model → sigmoid →
   append `(time, prob)`;
4. **fire** when **3 consecutive** windows (`detect.consec`) ≥ threshold; detected
   time = when that condition is met;
5. **score:** a positive video is caught if a fire lands within **±1.0 s**
   (`detect.tolerance_s`) of `time_of_event`; a negative video is a false alarm if it
   fires anywhere. The script **sweeps thresholds** and prints the best operating
   point, plus localization error and lead time.

`src/detect_video.py` is the single-video PyTorch version — used for demos and as the
**parity reference** for the on-device engine.

> **Training samples = labeled 16-frame windows from strips. Test samples = the same
> 16-frame windows, but generated by sliding over real videos in time order** (all
> windows kept, no subsampling, no augmentation, read as a temporal curve).

---

## 4. ONNX conversion — `src/export_onnx.py`

ONNX is the **framework-neutral handoff format** (same file for both devices).

1. Rebuild `AccidentNet`, load `best.pt`, `.eval()`.
2. Dummy input `[1,3,16,112,112]`.
3. `torch.onnx.export(...)`, **opset 17**, `dynamic_axes` = dynamic **batch**, constant
   folding on.
4. **Simplify** with `onnxsim`.
5. **Parity check:** PyTorch vs **ONNXRuntime**, assert `max|Δ| < 1e-3` (we measured
   `8e-7` for R(2+1)D).
6. Write sidecar **`<model>.meta.json`**: `detect_threshold`, `consec`, `target_fps`,
   `window_frames`, `stride`, `mean`, `std`, `arch` — everything the runtime needs
   that isn't in the graph.

Output: `jetson_r2plus1d18.onnx` (120 MB) + `.meta.json`.

---

## 5. TensorRT conversion & inference (Jetson Orin) — `deploy/jetson/`

TensorRT compiles the ONNX into a hardware-optimized **engine**. **Engines are
specific to the exact GPU + TensorRT version → they must be built ON the Jetson**
(cannot build on the RTX 4060 and copy).

On the device:
1. Copy `*.onnx` + `.meta.json` over (JetPack provides CUDA/cuDNN/TensorRT).
2. **Build engine** — `build_engine.sh` wraps `trtexec`: `--fp16` (default, ~2×
   speed), with an **optimization profile** for the dynamic batch
   (`--minShapes/--optShapes/--maxShapes` on `input:1×3×16×112×112`) → `*.engine`.
3. **(Optional) INT8** — `int8_calibrator.py`: faster still, needs **calibration
   clips** (a few hundred `.npy`) so TensorRT learns activation ranges; falls back to
   FP16 if accuracy drops.
4. **Streaming inference** — `infer_trt.py`: same sliding-window loop as
   `eval_fullvideo`, per-window forward through the TensorRT engine (`pycuda`); reads
   threshold/consec/fps from meta; prints detected time + per-window latency.
5. **Parity & sign-off:** PyTorch (PC) vs engine (Jetson) curves match within FP16
   tolerance; latency meets the streaming budget.

(Checklist steps **J6–J8** in WORKFLOW.md — the only remaining items; need the board.)

---

## 6. RKNN conversion (Rockchip RK3588) — later, Model #2

**Planned, not built**, and a **different model**: the RK3588 NPU does **not** support
3D convolutions well, so the R(2+1)D engine cannot be reused.

1. **Different architecture:** **2D-CNN per frame** (MobileNetV3 / RepVGG / ResNet18,
   all RKNN-friendly) → per-frame embeddings, **plus a lightweight temporal head**
   (temporal pool / 1D-conv / small GRU) from RKNN-supported ops only. The detection
   *framing* (sliding window → curve → fire time) is unchanged, so metrics stay
   comparable.
2. **Train/evaluate** with the same pipeline (extracted strips are reusable; only the
   model changes) via a new `configs/rockchip_*.yaml`.
3. **Export ONNX with STATIC shapes** (RKNN dislikes dynamic axes; batch=1), verify
   parity.
4. **Convert ONNX → RKNN** on an **x86 host** with `rknn-toolkit2`: set normalization
   in the RKNN config, build with **INT8 quantization** using a **calibration set**,
   export `.rknn`.
5. **Deploy on RK3588** with `rknn-toolkit-lite2` / `librknnrt`: same streaming loop
   calls the NPU per window; benchmark latency.
6. **Sign-off:** parity (PyTorch vs RKNN) + latency.

| | Jetson (Model #1) | Rockchip (Model #2) |
|---|---|---|
| Compiler | TensorRT (`trtexec`) | RKNN toolkit2 |
| Build location | **on the Jetson** | on an **x86 host** |
| Shapes | dynamic batch OK | **static** preferred |
| Quantization | FP16 default, INT8 optional | **INT8** typical (NPU) |
| 3D convs | supported | **avoid** → 2D-CNN + temporal head |

---

## 7. The data-shape journey, one line each

```
MP4 (1280×720, 30fps, ~40s)
  └─[preprocess.py]→ strip .npy            [128,128,128,3] uint8   (8s @16fps, 128² center crop)
  └─[extract_negatives.py]→ neg .npy       [256,128,128,3] uint8   (16 negative clips)
     └─[enumerate_windows]→ window          start_idx + label
        └─[WindowDataset]→ training tensor  [3,16,112,112] float    (aug + normalized)
           └─[model]→ logit                 scalar → sigmoid → P(accident)
              └─[slide over video]→ curve   [(t0,p0),(t1,p1),…]
                 └─[detect()]→ fire time    "ACCIDENT at t=19.3s"
   ONNX export: model → .onnx + .meta.json  (device-neutral)
      ├─ Jetson:  .onnx →[trtexec, on device]→ .engine (FP16/INT8)
      └─ Rockchip: 2D-CNN+temporal .onnx →[rknn-toolkit2, x86]→ .rknn (INT8)
```
