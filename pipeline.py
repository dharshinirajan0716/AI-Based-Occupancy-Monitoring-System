from pathlib import Path
import cv2
import pandas as pd
from ultralytics import YOLO

def process_video(video_path, output_dir, weights="yolov8n.pt", conf=0.25, frame_stride=5):
    video_path, output_dir = Path(video_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("OpenCV cannot open this video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    model = YOLO(weights)
    rows, writer, frame_no = [], None, 0
    out_path = output_dir / "annotated_video.mp4"
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_no % frame_stride == 0:
            result = model.predict(frame, conf=conf, classes=[0], verbose=False)[0]
            count = len(result.boxes) if result.boxes is not None else 0
            if writer is None:
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         max((fps or 25) / frame_stride, 1), (width, height))
            writer.write(result.plot())
            rows.append({"frame": frame_no, "time_seconds": round(frame_no / fps, 3) if fps else 0,
                         "predicted_count": int(count)})
        frame_no += 1
    cap.release()
    if writer:
        writer.release()
    if not rows:
        raise RuntimeError("No frames processed. Check the video file.")
    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_dir / "predictions.csv", index=False)
    return output_dir / "predictions.csv"
