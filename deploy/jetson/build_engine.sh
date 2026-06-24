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

# Optimization profile for dynamic batch (input: [N,3,T,H,W]).
# Adjust T/H/W to match your config (default 16x112x112).
T=16; H=112; W=112
MINB=1; OPTB=1; MAXB=4
SHAPE="3x${T}x${H}x${W}"

COMMON=(
  --onnx="${ONNX}"
  --saveEngine="${ENGINE}"
  --minShapes=input:${MINB}x${SHAPE}
  --optShapes=input:${OPTB}x${SHAPE}
  --maxShapes=input:${MAXB}x${SHAPE}
  --memPoolSize=workspace:2048
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
