# Model #1 (Jetson) — Results & Ablation

Task: **temporal** accident detection — a causal 1 s window (16 frames @ 16 fps) is
classified accident/normal; sliding it over a video yields a P(accident) curve and
a detected event time. A detection is **correct** if it fires within **±1 s** of
`time_of_event`. Backbone: **R(2+1)D-18**, Kinetics-400 pretrained, fine-tuned.

All numbers are on the **same held-out 225-video validation split** (112 accident /
113 normal), evaluated by streaming the detector over the **full ~40 s videos**
(`src.eval_fullvideo`) — the realistic deployment condition.

## Ablation

| Ver | Negatives in training | Checkpoint selection | Detection rate | False-alarm rate | Loc. error | Lead |
|----|----|----|----|----|----|----|
| v1 | event-strip only (8 s) | val AUC, near-event windows | 60.7 % | 13.3 % | 0.33 s | +0.01 s |
| v2 | + full-timeline hard negs | val AUC, near-event windows | 49.1 % | 18.6 % | 0.36 s | 0.00 s |
| **v3** | **+ full-timeline hard negs** | **val AP incl. full-timeline negs** | **63.4 %** | **14.2 %** | **0.31 s** | **+0.10 s** |

### Why v2 regressed (important)
Adding full-timeline negatives *and keeping the old checkpoint metric* (AUC over
near-event windows) made things **worse**, because that metric is blind to false
alarms on normal footage. Its "best" checkpoint landed on **epoch 3** — the first
unfrozen epoch — which is undertrained on negatives and fires everywhere.

### The fix (v3)
1. Put the full-timeline hard negatives into the **validation** set too, so the
   monitored metric reflects real false-alarm behaviour (val is now ~8 % positive).
2. Select the checkpoint by **AP** (`monitor: val_ap`) — sensitive to false
   positives under imbalance. v3's best is **epoch 14** (well-trained).

## v3 operating-point curve (full video)

| Threshold | Detection rate | False-alarm rate | Loc. error |
|----|----|----|----|
| 0.60 | 64.3 % | 15.9 % | 0.32 s |
| **0.68 (default)** | **63.4 %** | **14.2 %** | **0.31 s** |
| 0.83 | 59.8 % | 10.6 % | 0.27 s |
| 0.87 | 58.9 % | 9.7 % | 0.26 s |
| 0.91 | 56.2 % | 8.8 % | 0.27 s |

`detect_threshold` is stored in `best.pt` and exported into the ONNX `.meta.json`.
Raise it to trade detection for fewer false alarms; the on-device script reads it
from the meta file. Default **0.68** maximizes `detection_rate − false_alarm_rate`.

## Window-level (classifier) quality, v3
Val AUC 0.86, **AP 0.65**, accuracy 0.95 on the realistic imbalanced window set.

---

# Backbone comparison: R(2+1)D vs VideoMAE

User asked to try **VideoMAE** (self-supervised ViT) expecting higher accuracy.
Trained `videomae_base` (Kinetics-finetuned, 16×224×224, `transformers==4.46.3`)
with the **identical** data, splits, labeling and detection pipeline.

### Window-level (val, includes full-timeline negatives)
| Backbone | AUC | **AP** | F1 | acc |
|---|---|---|---|---|
| R(2+1)D-18 (112px, 33 M) | 0.86 | 0.65 | 0.68 | 0.95 |
| **VideoMAE-base (224px, 86 M)** | **0.92** | **0.74** | **0.70** | **0.95** |

VideoMAE is clearly the better **classifier** — higher AUC and AP.

### Full-video detection (the deployment metric) — same 225 videos
Detection rate at matched false-alarm levels (and the best single operating point):

| Backbone | @ FAR ≈ 0.13 | @ FAR ≈ 0.09 | best op. point | **Min FAR** | Loc. error |
|---|---|---|---|---|---|
| R(2+1)D-18 | 63.4 % | ~57 % | 63.4 % @ 14.2 % | ~8 % | 0.31 s |
| VideoMAE-base (no HNM) | — (floor 24 %) | — | 69.6 % @ 23.9 % | 23.9 % | 0.42 s |
| **VideoMAE-base + HNM** | **68.8 %** | **65.2 %** | **65.2 % @ 8.8 %** | **8.8 %** | **0.31 s** |

### Verdict
- **Plain VideoMAE** is a better *classifier* (AP 0.74) and localizes the crash
  beautifully (peak ~0.99 at the event), **but is overconfident on normal footage** —
  its full-video false-alarm rate never drops below ~24 %. Better window-ranking did
  not, by itself, translate to fewer full-video false alarms.
- **Hard-negative mining fixes exactly this.** Mining the normal clips VideoMAE
  false-fired on (2 488 clips from 580 of 1 275 train videos, prob ≥ 0.85) and
  fine-tuning with them **3× oversampled** dropped the false-alarm floor from
  **23.9 % → 8.8 %**, *halved* window-level false positives (221 → 109), and even
  **tightened localization (0.42 → 0.31 s)**.
- **Result: `videomae_base + HNM` is the most accurate model** — higher detection at
  every false-alarm level than R(2+1)D. It is the **accuracy-optimal Model #1**.

### Deployment recommendation
- **Accuracy-first (your stated priority):** `videomae_base + HNM`
  (`artifacts/runs/jetson_videomae_hnm/`). **Caveat:** it is an 86 M-param ViT @224 —
  much heavier than R(2+1)D. **Verify per-window latency on the actual Orin** (Step
  J7) before committing; if it can't keep up, fall back to R(2+1)D.
- **Lightweight fallback:** `r2plus1d_18` (`artifacts/runs/jetson_r2plus1d18/`) —
  33 M params @112, far faster on Orin Nano, ~5 pts lower detection at matched FAR.

### How HNM was run (reproducible)
```bash
python -m src.mine_hard_negatives --config configs/jetson_videomae.yaml \
    --mine-threshold 0.85 --max-per-video 12 --stride 4      # -> hardneg_manifest.csv
python -m src.train        --config configs/jetson_videomae_hnm.yaml   # fine-tune (init_from base)
python -m src.eval_fullvideo --config configs/jetson_videomae_hnm.yaml
```

All three models are fully trained and exported (`.onnx` + `.meta.json`):
`jetson_r2plus1d18/`, `jetson_videomae_base/`, `jetson_videomae_hnm/`.

## Notes / further-accuracy ideas (not yet done)
- Localization is already tight (~0.3 s); the headroom is in **recall** (~37 % of
  accidents missed within ±1 s).
- Candidates: larger backbone or higher input res; longer window (32 frames);
  multi-window temporal smoothing of the curve; hard-negative mining of the
  specific normal clips that still fire; class-balanced focal loss; test-time
  augmentation. Each is a 1–3 h experiment on the RTX 4060.
