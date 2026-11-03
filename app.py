import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best.pt"
IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
VIDEO_TYPES = ["mp4", "avi", "mov"]

st.set_page_config(page_title="Helmet Detection using Custom YOLO", page_icon="🪖", layout="wide")


@st.cache_resource(show_spinner="Loading YOLO model ...")
def load_model(path):
    return YOLO(path)


def to_png_bytes(rgb_array):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("Could not encode the image.")
    return buf.tobytes()


def detections_table(result):
    rows = []
    for b in result.boxes:
        x1, y1, x2, y2 = [round(float(v), 1) for v in b.xyxy[0].tolist()]
        rows.append({"Class": result.names[int(b.cls[0])], "Confidence": round(float(b.conf[0]), 3),
                     "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return pd.DataFrame(rows)


def process_video(model, in_path, out_path, conf, max_frames, progress):
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise IOError("This video cannot be opened (damaged file or unsupported codec). Try converting it to MP4.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0 or fps > 240:
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise IOError("No frames could be read from this video.")
    h, w = frame.shape[:2]
    raw_path = Path(out_path).with_name("raw_" + Path(out_path).name)
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise IOError("Could not create the output video.")
    n, n_det = 0, 0
    target = min(total, max_frames) if total > 0 else max_frames
    try:
        while ok and n < max_frames:
            res = model.predict(source=frame, conf=conf, verbose=False)[0]
            n_det += len(res.boxes)
            writer.write(res.plot())
            n += 1
            progress.progress(min(n / max(target, 1), 1.0), text=f"Processing frame {n} of {target}")
            ok, frame = cap.read()
    finally:
        cap.release()
        writer.release()
    final = raw_path
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_path), "-vcodec", "libx264",
               "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-movflags", "+faststart",
               str(out_path)]
        if subprocess.run(cmd, capture_output=True).returncode == 0 and Path(out_path).exists():
            final = Path(out_path)
    return final, n, n_det, final != raw_path


# ----------------------------- page header ---------------------------------
st.title("🪖 Helmet Detection Using Custom YOLO")
st.markdown(
    "Detects **helmets** worn by two-wheeler riders in images and videos using a custom-trained "
    "**YOLOv8** model (transfer learning from `yolov8n.pt`)."
)
st.info(
    "This model has a single class: **Helmet**. It finds helmets and draws boxes around them. "
    "It does **not** classify riders as 'without helmet'; that would need extra labelled data."
)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: `{MODEL_PATH.name}`. Place your trained `best.pt` next to `app.py`.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"The model could not be loaded: {exc}")
    st.stop()

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05,
                     help="Lower = more boxes (more false alarms). Higher = fewer but surer boxes.")
    st.caption("Model classes: " + ", ".join(str(v) for v in model.names.values()))

tab_img, tab_vid = st.tabs(["🖼️ Image detection", "🎬 Video detection"])

# ----------------------------- image tab -----------------------------------
with tab_img:
    up = st.file_uploader("Upload an image", type=IMAGE_TYPES, key="img")
    if up is not None:
        try:
            image = ImageOps.exif_transpose(Image.open(up)).convert("RGB")   # PIL image is RGB
        except Exception as exc:
            st.error(f"This file could not be read as an image: {exc}")
        else:
            with st.spinner("Detecting helmets ..."):
                try:
                    result = model.predict(source=image, conf=conf, verbose=False)[0]
                except Exception as exc:
                    st.error(f"Detection failed: {exc}")
                    st.stop()
            annotated_rgb = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)   # plot() is BGR -> RGB for display
            c1, c2 = st.columns(2)
            c1.subheader("Original")
            c1.image(image)
            c2.subheader("Detected")
            c2.image(annotated_rgb)
            table = detections_table(result)
            st.subheader("Detection summary")
            m1, m2 = st.columns(2)
            m1.metric("Helmets detected", len(table))
            m2.metric("Average confidence", f"{table['Confidence'].mean():.2f}" if len(table) else "n/a")
            if len(table):
                st.dataframe(table)
            else:
                st.warning("No helmets detected at this threshold. Try lowering the confidence slider.")
            st.download_button("Download annotated image", to_png_bytes(annotated_rgb),
                               file_name="helmet_detection.png", mime="image/png")

# ----------------------------- video tab -----------------------------------
with tab_vid:
    max_frames = st.slider("Maximum frames to process", 30, 1000, 300, 30,
                           help="Limits processing time, especially on the free cloud CPU.")
    vup = st.file_uploader("Upload a video (MP4, AVI, MOV)", type=VIDEO_TYPES, key="vid")
    if vup is not None:
        tmp_dir = tempfile.mkdtemp()
        in_path = os.path.join(tmp_dir, "input" + Path(vup.name).suffix.lower())
        out_path = os.path.join(tmp_dir, "helmet_detection_output.mp4")
        with open(in_path, "wb") as f:
            f.write(vup.getbuffer())
        progress = st.progress(0.0, text="Starting ...")
        try:
            t0 = time.time()
            final, n_frames, n_det, playable = process_video(model, in_path, out_path, conf, max_frames, progress)
        except Exception as exc:
            st.error(f"Video processing failed: {exc}")
        else:
            progress.empty()
            st.success(f"Processed {n_frames} frames in {time.time() - t0:.1f} s")
            s1, s2, s3 = st.columns(3)
            s1.metric("Frames processed", n_frames)
            s2.metric("Total detections", n_det)
            s3.metric("Avg detections / frame", f"{n_det / max(n_frames, 1):.2f}")
            data = Path(final).read_bytes()
            if playable:
                st.video(data)
            else:
                st.warning("ffmpeg is not available, so the video cannot be previewed here. Use the download button.")
            st.download_button("Download processed video", data, file_name="helmet_detection_output.mp4",
                               mime="video/mp4")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

st.markdown("---")
st.caption("Built with Ultralytics YOLOv8 and Streamlit | Academic project: Helmet Detection Using Custom YOLO")
