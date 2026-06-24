"""
Optional INT8 calibration for TensorRT on Jetson.

INT8 ~doubles throughput vs FP16 but needs a calibration set representative of
real inputs. Copy a few hundred pre-extracted .npy clips (artifacts/clips/train/)
to the Jetson, then build an engine with this entropy calibrator.

Usage:
    python3 int8_calibrator.py --onnx model.onnx --clips ./calib_clips \
        --out model_int8.engine
"""
from __future__ import annotations
import argparse
import glob
import os

import numpy as np
import tensorrt as trt
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda

T, S = 16, 112
MEAN = np.array([0.43216, 0.394666, 0.37645])
STD = np.array([0.22803, 0.22145, 0.216989])


def prep(npy_path):
    arr = np.load(npy_path).astype(np.float32) / 255.0     # [N,Sx,Sx,3]
    n = arr.shape[0]
    idx = np.linspace(0, n - 1, T).round().astype(int)
    arr = arr[idx]
    s = arr.shape[1]
    top = (s - S) // 2
    arr = arr[:, top:top + S, top:top + S, :]
    arr = (arr - MEAN) / STD
    return np.ascontiguousarray(arr.transpose(3, 0, 1, 2)[None].astype(np.float32))


class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, clip_dir, cache="calib.cache"):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(clip_dir, "*.npy")))
        self.cache = cache
        self.i = 0
        self.device_input = cuda.mem_alloc(prep(self.files[0]).nbytes)

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.i >= len(self.files):
            return None
        batch = prep(self.files[self.i])
        cuda.memcpy_htod(self.device_input, batch)
        self.i += 1
        return [int(self.device_input)]

    def read_calibration_cache(self):
        return open(self.cache, "rb").read() if os.path.exists(self.cache) else None

    def write_calibration_cache(self, cache):
        open(self.cache, "wb").write(cache)


def build(onnx, clip_dir, out):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx, "rb") as f:
        assert parser.parse(f.read()), "ONNX parse failed"

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)
    config.int8_calibrator = Calibrator(clip_dir)

    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, T, S, S), (1, 3, T, S, S), (4, 3, T, S, S))
    config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    with open(out, "wb") as f:
        f.write(engine)
    print(f"INT8 engine -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", default="model_int8.engine")
    a = ap.parse_args()
    build(a.onnx, a.clips, a.out)
