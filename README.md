# AI-Based Occupancy Monitoring System

Video-based people detection and occupancy analytics using pretrained YOLO and Streamlit.

## Setup (Windows / VS Code)
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Dataset
Download the Mall Dataset from:
https://personal.ie.cuhk.edu.hk/~ccloy/downloads_mall_dataset.html

Place the video (e.g. mall.avi) in `data/`. Inspect the dataset annotations before preparing evaluation labels.

## Analyze video metadata
```bash
python scripts/analyze_dataset.py
```
This reports frame count, dimensions and FPS, and saves a sample frame.

## Run dashboard
```bash
streamlit run app.py
```
Upload a video or select one in `data/`, then click Run person detection. First run may download the selected YOLO weights.

## Ground-truth evaluation
Add `data/ground_truth.csv` with columns `frame,ground_truth_count`, using actual Mall Dataset annotation counts. Frame values must match original video frame indices in `outputs/predictions.csv`. The dashboard calculates MAE and RMSE for matching frames.

## Limitations
YOLO counts visible detections and may miss distant or occluded people, especially in dense crowds. Current occupancy is the last sampled frame, not a live CCTV feed. The pretrained model is not fine-tuned on the Mall Dataset. Do not report evaluation scores until compared against real annotations.

## Deliverables
Dataset analysis, sample frame, predictions CSV, MAE/RMSE against real annotations, failure-case analysis, dashboard screenshots, and demo video.
