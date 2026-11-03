from pathlib import Path
import pandas as pd

def load_results(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["frame", "time_seconds", "predicted_count"])

def summarize(df):
    return {"peak": int(df.predicted_count.max()), "average": float(df.predicted_count.mean())} if not df.empty else {"peak": 0, "average": 0.0}
