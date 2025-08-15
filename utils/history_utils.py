# utils/history_utils.py
import os
import pandas as pd
from datetime import datetime

MASTER_CSV = os.path.join("resultados", "master_history.csv")

def ensure_master():
    os.makedirs("resultados", exist_ok=True)
    if not os.path.exists(MASTER_CSV):
        df = pd.DataFrame(columns=[
            "timestamp", "image", "overlay_path", "heatmap_puro_path", "csv_path",
            "probabilidad", "centro_x", "centro_y", "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax",
            "tamano_zona_activa", "nivel_urgencia", "nivel_urgencia_label"
        ])
        df.to_csv(MASTER_CSV, index=False)

def append_record(record_dict):
    ensure_master()
    df = pd.read_csv(MASTER_CSV)
    record = record_dict.copy()
    record["timestamp"] = datetime.now().isoformat()
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(MASTER_CSV, index=False)

def read_master():
    ensure_master()
    return pd.read_csv(MASTER_CSV)
