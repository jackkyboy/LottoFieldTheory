"""
quantum_utilities.py

ชุดเครื่องมือเชิงควอนตัมสำหรับวิเคราะห์ระบบสุ่มแบบข้อมูล:
- Quantum Believability Score (QBS)
- ตรวจจับ Quantum Paradox
- Mutual Information & Entanglement Graph
- Entropy Test เทียบกับการสุ่ม

ผู้พัฒนา: Apichet & Quantum Mechanics GPT
"""

import pandas as pd
import numpy as np
import networkx as nx
import random
from typing import List, Tuple, Dict
from numpy import log2
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import entropy

__all__ = [
    "rank_draws_with_qbs",
    "detect_paradox",
    "compute_mutual_information",
    "build_entanglement_graph",
    "calculate_draw_entropy",
    "entropy_test_vs_random",
    "find_shadow_draws"
]

# ------------------------------
# 🔗 Quantum Believability Score
# ------------------------------

def rank_draws_with_qbs(
    df_multiverse: pd.DataFrame,
    entangled_list: List[str],
    entropy_threshold: float = 6.0
) -> pd.DataFrame:
    """
    คำนวณ Quantum Believability Score (QBS) ให้ draw แต่ละรายการ

    QBS = (frequency / log2(1 + entropy)) * entangle_boost / (1 + paradox_penalty)
    """
    scores = []
    counts = df_multiverse["collapsed"].value_counts()
    entropies = df_multiverse.groupby("collapsed")["entropy"].mean()

    for draw in counts.index:
        freq = counts[draw]
        entropy_val = entropies[draw]

        # ตรวจจับ paradox
        paradox_score = 0
        if entropy_val > entropy_threshold and freq >= 3:
            paradox_score += 1
        if draw in entangled_list and draw not in df_multiverse["collapsed"].values:
            paradox_score += 2

        # คำนวณ QBS
        entangle_boost = 1.5 if draw in entangled_list else 1.0
        entropy_factor = log2(1 + entropy_val)
        qbs = (freq / entropy_factor) * entangle_boost / (1 + paradox_score)

        scores.append({
            "draw": draw,
            "frequency": freq,
            "avg_entropy": round(entropy_val, 5),
            "entangled": draw in entangled_list,
            "paradox_score": paradox_score,
            "QBS": round(qbs, 5)
        })

    return pd.DataFrame(scores).sort_values(by="QBS", ascending=False).reset_index(drop=True)


# ------------------------------
# 🌀 Quantum Paradox Detector
# ------------------------------

def detect_paradox(
    df_multiverse: pd.DataFrame,
    entangled_list: List[str],
    frequency_threshold: int = 1,
    entropy_threshold: float = 8.5
) -> List[Tuple[str, str]]:
    """
    ตรวจจับ draw ที่มีพฤติกรรมขัดแย้งในระบบควอนตัมจำลอง:
    - พัวพันแต่ไม่เคยปรากฏ
    - ปรากฏบ่อยแต่ entropy สูงผิดปกติ
    """
    paradoxes = []

    freq_map = df_multiverse["collapsed"].value_counts().to_dict()
    entropy_map = df_multiverse.groupby("collapsed")["entropy"].mean().to_dict()

    for draw in entangled_list:
        freq = freq_map.get(draw, 0)
        entropy_val = entropy_map.get(draw, None)

        if freq == 0:
            paradoxes.append((draw, "🌀 Entangled but never appeared in multiverse"))
        elif freq >= frequency_threshold and entropy_val is not None and entropy_val >= entropy_threshold:
            paradoxes.append((draw, f"⚠️ High frequency ({freq}) but high entropy ({entropy_val:.2f})"))

    return paradoxes


# ------------------------------
# 📊 Entanglement Analyzer
# ------------------------------

def compute_mutual_information(draw_matrix: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    คำนวณ Mutual Information ระหว่าง draw แต่ละคู่ (หลังแปลงค่าเป็นกลุ่มแบบจำแนก)

    Args:
        draw_matrix: เวกเตอร์ข้อมูลต่อเนื่อง [n_draws x features]
        n_bins: จำนวน bin สำหรับ discretization

    Returns:
        mutual_info_matrix: เมทริกซ์ mutual information แบบสมมาตร [n x n]
    """
    n = draw_matrix.shape[0]
    mi_matrix = np.zeros((n, n))

    # แปลงค่าต่อเนื่องเป็น discrete labels
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    discretized = est.fit_transform(draw_matrix)

    for i in range(n):
        for j in range(i, n):
            mi = mutual_info_score(discretized[i], discretized[j])
            mi_matrix[i, j] = mi_matrix[j, i] = mi

    return mi_matrix


def build_entanglement_graph(mi_matrix: np.ndarray, threshold: float = 0.05) -> nx.Graph:
    """
    สร้างกราฟพัวพันโดยใช้ mutual information เป็นน้ำหนัก edge
    """
    n = mi_matrix.shape[0]
    G = nx.Graph()

    for i in range(n):
        G.add_node(i)

    for i in range(n):
        for j in range(i + 1, n):
            if mi_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=mi_matrix[i, j])

    return G


# ------------------------------
# 🔐 Entropy Analyzer
# ------------------------------

def calculate_draw_entropy(draw: np.ndarray) -> float:
    """
    คำนวณ Shannon entropy สำหรับ draw เดียว
    """
    values, counts = np.unique(draw, return_counts=True)
    probs = counts / len(draw)
    return entropy(probs, base=2)


def entropy_test_vs_random(draws_real: List[np.ndarray], num_sim: int = 1000) -> Dict[str, float]:
    """
    เปรียบเทียบ entropy ของ draw จริงกับชุดสุ่มจำลอง

    Returns:
        dict: {
            'real_entropy': ค่าเฉลี่ย entropy จริง,
            'simulated_entropy': ค่าเฉลี่ย entropy จากสุ่ม,
            'difference': ส่วนต่าง
        }
    """
    real_entropies = [calculate_draw_entropy(draw) for draw in draws_real]
    simulated_entropies = []

    for _ in range(num_sim):
        sim_draws = [np.random.randint(0, 10, len(draw)) for draw in draws_real]
        sim_ent = [calculate_draw_entropy(draw) for draw in sim_draws]
        simulated_entropies.append(np.mean(sim_ent))

    return {
        'real_entropy': round(np.mean(real_entropies), 5),
        'simulated_entropy': round(np.mean(simulated_entropies), 5),
        'difference': round(np.mean(real_entropies) - np.mean(simulated_entropies), 5)
    }




def find_shadow_draws(entangled_set, multiverse_df):
    collapsed_set = set(multiverse_df["collapsed"])
    return [draw for draw in entangled_set if draw not in collapsed_set]
