"""
Streamlit live demo — Dashcam Collision Detection.

Upload a dashcam clip; the app slides a causal 1s window across it (ONNXRuntime,
CPU) to produce a P(accident) curve and reports WHEN a collision is detected.

Runs without PyTorch — only onnxruntime + opencv — so it fits Streamlit Community
Cloud. The R(2+1)D ONNX model is loaded from a local path if present, else
downloaded from the URL in `MODEL_URL` (Streamlit secret or env var).

Run locally:   streamlit run app.py
"""
from __future__ import annotations
import os
import tempfile
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ---- model defaults (R(2+1)D meta; overridable by a sidecar .meta.json) ----
DEFAULTS = {
    "window_frames": 16,
    "crop": 112,
    "target_fps": 16,
    "stride": 4,                # scan stride for the demo (snappier on CPU)
    "consec": 3,
    "detect_threshold": 0.68,
    "mean": [0.43216, 0.394666, 0.37645],
    "std": [0.22803, 0.22145, 0.216989],
}
LOCAL_MODEL = "artifacts/runs/jetson_r2plus1d18/jetson_r2plus1d18.onnx"
CACHE_MODEL = Path(tempfile.gettempdir()) / "accident_model.onnx"

st.set_page_config(page_title="Dashcam Collision Detection",
                   page_icon="🚗", layout="wide")


# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_session():
    import onnxruntime as ort
    path = None
    if Path(LOCAL_MODEL).exists():
        path = LOCAL_MODEL
    else:
        url = os.environ.get("MODEL_URL", "")
        if not url:
            try:                              # st.secrets raises if no secrets file
                url = st.secrets.get("MODEL_URL", "")
            except Exception:
                url = ""
        if not url:
            return None, "no_model"
        if not CACHE_MODEL.exists():
            urllib.request.urlretrieve(url, CACHE_MODEL)
        path = str(CACHE_MODEL)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name


def preprocess_frame(fr, S):
    h, w = fr.shape[:2]
    if h <= w:
        nh, nw = S, int(round(w * S / h))
    else:
        nh, nw = int(round(h * S / w)), S
    fr = cv2.resize(fr, (nw, nh))
    top, left = (nh - S) // 2, (nw - S) // 2
    return cv2.cvtColor(fr[top:top + S, left:left + S], cv2.COLOR_BGR2RGB)


def stream_detect(sess, in_name, video_path, cfg, progress=None):
    L, S = cfg["window_frames"], cfg["crop"]
    mean = np.array(cfg["mean"], np.float32)
    std = np.array(cfg["std"], np.float32)
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step = max(1, int(round(src_fps / cfg["target_fps"])))
    buf, raw_buf = deque(maxlen=L), deque(maxlen=L)
    curve, frames_at = [], []
    fc = scount = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fc % step == 0:
            buf.append(preprocess_frame(fr, S))
            raw_buf.append(fr)
            scount += 1
            if len(buf) == L and scount % cfg["stride"] == 0:
                clip = (np.stack(buf).astype(np.float32) / 255.0 - mean) / std
                x = clip.transpose(3, 0, 1, 2)[None]      # [1,3,L,S,S]
                logit = sess.run(None, {in_name: x.astype(np.float32)})[0].reshape(-1)[0]
                prob = float(1.0 / (1.0 + np.exp(-logit)))
                t = fc / src_fps
                curve.append((t, prob))
                frames_at.append(raw_buf[-1])
        fc += 1
        if progress and total:
            progress.progress(min(1.0, fc / total))
    cap.release()
    return curve, frames_at, src_fps


def detect_time(curve, thr, consec):
    run = 0
    for i, (t, p) in enumerate(curve):
        run = run + 1 if p >= thr else 0
        if run >= consec:
            return t, i
    return None, None


# ----------------------------------------------------------------------
st.title("🚗 Dashcam Collision Detection")
st.caption("Temporal accident detection — detects **when** a collision happens by "
           "sliding a 1-second window across the clip (R(2+1)D, ONNXRuntime/CPU).")

sess_info = load_session()
sess, in_name = sess_info if sess_info[0] is not None else (None, None)

with st.sidebar:
    st.header("Settings")
    thr = st.slider("Detection threshold", 0.30, 0.99, float(DEFAULTS["detect_threshold"]), 0.01,
                    help="Higher = fewer false alarms, lower detection rate.")
    consec = st.slider("Consecutive windows to fire", 1, 5, int(DEFAULTS["consec"]))
    stride = st.select_slider("Scan stride (speed)", options=[2, 3, 4, 6, 8],
                              value=int(DEFAULTS["stride"]),
                              help="Larger = faster, coarser curve.")
    st.markdown("---")
    st.markdown("Model: **R(2+1)D-18** · input 16×112×112 · 16 fps window")

cfg = {**DEFAULTS, "stride": stride}

if sess is None:
    st.error("⚠️ Model not found. Set the `MODEL_URL` secret to a direct link to "
             "`jetson_r2plus1d18.onnx`, or place the file at "
             f"`{LOCAL_MODEL}`. See the README ‘Live demo’ section.")
    st.stop()

SAMPLE = "sample_video/nexar_accident_demo.mp4"

up = st.file_uploader("Upload a dashcam clip (.mp4 / .mov / .avi)",
                      type=["mp4", "mov", "avi", "mkv"])

# decide which video to analyse: an upload, or the bundled sample on request
video_path, source = None, None
if up is not None:
    st.session_state.pop("use_sample", None)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix)
    tmp.write(up.read()); tmp.close()
    video_path, source = tmp.name, up.name
elif Path(SAMPLE).exists():
    if st.button("▶️ Try it with a sample accident clip"):
        st.session_state["use_sample"] = True
    if st.session_state.get("use_sample"):
        video_path, source = SAMPLE, "sample · Nexar dashcam collision"

if video_path is None:
    msg = "👆 Upload a dashcam clip to run detection."
    if Path(SAMPLE).exists():
        msg += " — or click **Try it with a sample accident clip** above."
    st.info(msg + "  \nClips at any resolution work; CPU inference takes a few seconds.")
    st.stop()

# ---- run detection ----
c1, c2 = st.columns([2, 3])
with c1:
    st.video(video_path)
    st.caption(source)
with c2:
    prog = st.progress(0.0, text="Scanning…")
    curve, frames_at, src_fps = stream_detect(sess, in_name, video_path, cfg, prog)
    prog.empty()
    if not curve:
        st.warning("Could not read enough frames from this clip.")
        st.stop()
    t_fire, idx = detect_time(curve, thr, consec)
    peak_i = int(np.argmax([p for _, p in curve]))
    peak_t, peak_p = curve[peak_i]

    if t_fire is not None:
        st.error(f"### 🛑 Collision detected at **t = {t_fire:.2f} s**")
    else:
        st.success("### ✅ No collision detected")
    m1, m2 = st.columns(2)
    m1.metric("Peak P(accident)", f"{peak_p:.2f}", help=f"at t={peak_t:.2f}s")
    m2.metric("Threshold", f"{thr:.2f}")

# probability curve
df = pd.DataFrame(curve, columns=["time_s", "P(accident)"]).set_index("time_s")
df["threshold"] = thr
st.line_chart(df, height=260)

# frame at the detection / peak moment
show_i = idx if idx is not None else peak_i
if 0 <= show_i < len(frames_at):
    st.image(cv2.cvtColor(frames_at[show_i], cv2.COLOR_BGR2RGB),
             caption=f"Frame at {'detection' if idx is not None else 'peak'} "
                     f"(t={curve[show_i][0]:.2f}s)", width=480)
st.download_button("Download probability curve (CSV)",
                   df.to_csv().encode(), "curve.csv", "text/csv")
