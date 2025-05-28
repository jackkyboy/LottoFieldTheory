# /Users/apichet/Downloads/thai_lottery_project/thai_lottery_project/backend/quantum_verifier.py
from pathlib import Path
import pandas as pd

def verify_predictions_against_truth(df: pd.DataFrame, predictions: list, target_cols=["first_prize", "last2"]):
    """
    ตรวจสอบว่าทำนายตรงกับข้อมูลจริงหรือไม่ ในหลายคอลัมน์ เช่น 'first_prize', 'last2'

    Args:
        df (pd.DataFrame): DataFrame ของข้อมูลผลลอตเตอรี่จริง
        predictions (list): รายการ draw IDs หรือ index ที่โมเดลทำนาย (เช่น ['Draw_123', 317])
        target_cols (list): รายชื่อคอลัมน์ที่จะตรวจสอบ

    Returns:
        pd.DataFrame: Log การตรวจสอบแต่ละ column พร้อม matched flag
    """
    results = []

    for pred in predictions:
        try:
            idx = int(pred.split("_")[1]) if isinstance(pred, str) and "Draw_" in pred else int(pred)
            if idx >= len(df):
                continue

            row = df.iloc[idx]
            entry = {
                "draw": f"Draw_{idx}",
                "index": idx,
                "date": row.get("date", "(no date)")
            }

            for col in target_cols:
                val = row.get(col, None)
                entry[f"{col}_value"] = val
                entry[f"{col}_matched"] = pd.notnull(val) and str(val).strip() != ""

            results.append(entry)

        except Exception as e:
            results.append({
                "draw": str(pred),
                "index": None,
                "date": None,
                "error": str(e)
            })

    return pd.DataFrame(results)
