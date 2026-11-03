
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from analytics import load_results, summarize
from pipeline import process_video
ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

DATA.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="AI Occupancy Monitoring",
    page_icon="👥",
    layout="wide"
)

st.title("👥 AI-Based Occupancy Monitoring System")
st.caption(
    "Intelligent video-based people detection "
    "and real-time occupancy analytics"
)

# ---------------- SIDEBAR ----------------

st.sidebar.title("⚙️ Video Processing")

videos = sorted([
    p.name for p in DATA.glob("*")
    if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
])

uploaded = st.sidebar.file_uploader(
    "Upload Mall Video",
    type=["mp4", "avi", "mov", "mkv"]
)

video_path = None

if uploaded:
    video_path = DATA / uploaded.name
    video_path.write_bytes(uploaded.getbuffer())

elif videos:
    selected_video = st.sidebar.selectbox(
        "Select Existing Video", videos
    )
    video_path = DATA / selected_video

conf = st.sidebar.slider(
    "Detection Confidence",
    0.05, 0.90, 0.25, 0.05
)

stride = st.sidebar.number_input(
    "Process Every Nth Frame",
    min_value=1,
    max_value=30,
    value=5
)

weights = st.sidebar.selectbox(
    "YOLO Model",
    ["yolov8n.pt", "yolo11n.pt"]
)

# ---------------- VIDEO PROCESSING ----------------

if st.sidebar.button(
    "▶ Run Person Detection",
    type="primary",
    disabled=video_path is None
):
    with st.spinner("Detecting people in video..."):
        try:
            process_video(
                video_path,
                OUTPUTS,
                weights,
                conf,
                int(stride)
            )
            st.success("Video processing completed!")
            st.rerun()

        except Exception as exc:
            st.error(f"Processing failed: {exc}")

# ---------------- LOAD PREDICTIONS ----------------

prediction_file = OUTPUTS / "predictions.csv"

df = load_results(prediction_file)

if df.empty:
    st.info(
        "Upload a mall video or select an existing video, "
        "then click Run Person Detection."
    )
    st.markdown(
        "**Pipeline:** Video → Frame Extraction → "
        "YOLO Detection → Occupancy Count → Analytics"
    )
    st.stop()

df = df.copy()
df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
df["predicted_count"] = pd.to_numeric(
    df["predicted_count"], errors="coerce"
)
df = df.dropna(subset=["frame", "predicted_count"])
df["frame"] = df["frame"].astype(int)
df["predicted_count"] = df["predicted_count"].astype(int)
df = df.sort_values("frame")

stats = summarize(df)

# ---------------- OCCUPANCY OVERVIEW ----------------

st.header("📊 Occupancy Overview")

a, b, c, d = st.columns(4)

a.metric(
    "Current Occupancy",
    int(df["predicted_count"].iloc[-1])
)

b.metric(
    "Peak Occupancy",
    int(stats["peak"])
)

c.metric(
    "Average Occupancy",
    f'{stats["average"]:.2f}'
)

d.metric(
    "Frames Processed",
    len(df)
)

# ---------------- OCCUPANCY TREND ----------------

st.subheader("📈 Occupancy Trend")

st.line_chart(
    df.set_index("time_seconds")["predicted_count"],
    width="stretch"
)

# ---------------- FRAME COUNTS ----------------

st.subheader("📋 Frame-by-Frame Predictions")

st.dataframe(
    df.tail(100),
    width="stretch",
    hide_index=True
)

# ---------------- ANNOTATED VIDEO ----------------

annotated = OUTPUTS / "annotated_video_browser.mp4"

if annotated.exists():
    st.subheader("🎬 Processed Video")
    video_bytes = annotated.read_bytes()
    st.video(video_bytes, format="video/mp4")
# ---------------- DOWNLOAD PREDICTIONS ----------------

st.download_button(
    "⬇ Download Predictions CSV",
    data=prediction_file.read_bytes(),
    file_name="occupancy_predictions.csv",
    mime="text/csv"
)

# ---------------- GROUND TRUTH EVALUATION ----------------

st.divider()
st.header("🎯 Model Evaluation")

gt_path = DATA / "ground_truth.csv"

if gt_path.exists():

    gt = pd.read_csv(gt_path)

    required = {"frame", "ground_truth_count"}

    if not required.issubset(gt.columns):
        st.error(
            "ground_truth.csv must contain "
            "frame and ground_truth_count columns."
        )

    else:
        gt["frame"] = pd.to_numeric(
            gt["frame"], errors="coerce"
        )
        gt["ground_truth_count"] = pd.to_numeric(
            gt["ground_truth_count"], errors="coerce"
        )

        gt = gt.dropna(
            subset=["frame", "ground_truth_count"]
        )

        gt["frame"] = gt["frame"].astype(int)

        merged = pd.merge(
            df,
            gt,
            on="frame",
            how="inner"
        )

        if not merged.empty:

            mae = np.mean(
                np.abs(
                    merged["predicted_count"]
                    - merged["ground_truth_count"]
                )
            )

            rmse = np.sqrt(
                np.mean(
                    (
                        merged["predicted_count"]
                        - merged["ground_truth_count"]
                    ) ** 2
                )
            )

            st.success(
                "Ground-truth evaluation completed!"
            )

            x, y, z = st.columns(3)

            x.metric("MAE", f"{mae:.3f}")
            y.metric("RMSE", f"{rmse:.3f}")
            z.metric("Matched Frames", len(merged))

            st.caption(
                "MAE and RMSE are measured in people per sampled frame. "
                "Lower values indicate smaller prediction errors."
            )

            # Actual vs Predicted comparison

            st.subheader(
                "📉 Actual vs Predicted Occupancy"
            )

            comparison = merged.set_index("frame")[
                ["predicted_count", "ground_truth_count"]
            ]

            comparison.columns = [
                "YOLO Predicted",
                "Actual Ground Truth"
            ]

            st.line_chart(
                comparison,
                width="stretch"
            )

            # Error analysis

            merged["absolute_error"] = np.abs(
                merged["predicted_count"]
                - merged["ground_truth_count"]
            )

            st.subheader("🔍 Prediction Error Analysis")

            st.dataframe(
                merged[
                    [
                        "frame",
                        "predicted_count",
                        "ground_truth_count",
                        "absolute_error"
                    ]
                ].head(100),
                width="stretch",
                hide_index=True
            )

            # Download evaluation report

            st.download_button(
                "⬇ Download Evaluation Report",
                data=merged.to_csv(index=False),
                file_name="occupancy_evaluation.csv",
                mime="text/csv"
            )

        else:
            st.warning(
                "No matching frame numbers found "
                "between predictions and ground truth."
            )

else:
    st.warning(
        "Ground-truth file not found. "
        "Place ground_truth.csv inside the data folder."
    )

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "AI-Based Occupancy Monitoring System | "
    "YOLO + OpenCV + Streamlit"
)
