# context_extractor.py

import re
from difflib import SequenceMatcher
from typing import List, Dict


def extract_info_signal_for_draw(draw_id: str, trending_keywords: List[str]) -> float:
    """
    ตรวจสอบว่า draw_id (เช่น 'Draw_337') มีส่วนสัมพันธ์กับเลขหรือคำที่เป็นกระแสหรือไม่
    """
    draw_number = draw_id.split('_')[-1]
    score = 0.0

    for keyword in trending_keywords:
        # ตรงตัวเลขตรงๆ เช่น "337"
        if re.search(r'\b' + re.escape(draw_number) + r'\b', keyword):
            score += 1.0
        else:
            # คล้ายกัน เช่น "33", "3379", ใช้ string similarity
            ratio = SequenceMatcher(None, draw_number, keyword).ratio()
            if ratio > 0.7:
                score += ratio * 0.5

    return round(score, 4)


def build_info_context_vector(draw_ids: List[str], trending_keywords: List[str]) -> Dict[str, Dict[str, float]]:
    """
    สร้าง context vector ที่มี info_score สำหรับ draw แต่ละตัว
    """
    context_scores = {}
    for draw_id in draw_ids:
        info_score = extract_info_signal_for_draw(draw_id, trending_keywords)
        context_scores[draw_id] = {"info_score": info_score}
    return context_scores
