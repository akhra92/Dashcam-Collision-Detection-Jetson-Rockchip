#!/usr/bin/env bash
# =====================================================================
# Build a TensorRT engine from the exported ONNX *on the Jetson Orin*.
# TensorRT engines are hardware/version specific — they MUST be built on
# the target device, not on the training PC.
#
# Usage:
#   ./build_engine.sh model.onnx [fp16|int8]   (default: fp16)
#   For int8 you must provide a calibration cache or images (see notes).
# =====================================================================
set -euo pipefail

ONNX="${1:?path to .onnx required}"
PRECISION="${2:-fp16}"
ENGINE="${ONNX%.onnx}_${PRECISION}.engine"

# Input T/H/W are read from the sidecar <model>.meta.json (input_shape =
# [N,C,T,H,W]) so this works for any model — R(2+1)D @112 or VideoMAE @224.
# Falls back to 16x112x112 if no meta is found. Override via env: T=.. H=.. W=..
META="${ONNX%.onnx}.meta.json"
if [[ -f "${META}" ]]; then
  read T H W < <(python3 -c "import json;s=json.load(open('${META}'))['input_shape'];print(s[-3],s[-2],s[-1])")
fi
T="${T:-16}"; H="${H:-112}"; W="${W:-112}"
# Batch 1 by default (streaming inference is batch-1). Override via env MAXB=..
MINB="${MINB:-1}"; OPTB="${OPTB:-1}"; MAXB="${MAXB:-1}"
WORKSPACE="${WORKSPACE:-4096}"          # MB; a ViT (VideoMAE) needs more than a CNN
SHAPE="3x${T}x${H}x${W}"
echo "building ${ENGINE}: input Nx${SHAPE} (batch ${MINB}/${OPTB}/${MAXB}), ws=${WORKSPACE}MB"

COMMON=(
  --onnx="${ONNX}"
  --saveEngine="${ENGINE}"
  --minShapes=input:${MINB}x${SHAPE}
  --optShapes=input:${OPTB}x${SHAPE}
  --maxShapes=input:${MAXB}x${SHAPE}
  --memPoolSize=workspace:${WORKSPACE}
  --verbose
)

case "${PRECISION}" in
  fp16)
    trtexec "${COMMON[@]}" --fp16
    ;;
  int8)
    # INT8 needs calibration. Easiest path: build with --fp16 --int8 and let
    # TensorRT do entropy calibration from a cache you generate with the
    # Python calibrator, or fall back to fp16 if accuracy drops too much.
    echo "INT8: ensure calibration cache exists (calib.cache)."
    trtexec "${COMMON[@]}" --int8 --fp16 \
      --calib=calib.cache || {
        echo "INT8 build failed; falling back to FP16"; trtexec "${COMMON[@]}" --fp16; }
    ;;
  *)
    echo "unknown precision: ${PRECISION}"; exit 1;;
esac

echo "engine -> ${ENGINE}"
