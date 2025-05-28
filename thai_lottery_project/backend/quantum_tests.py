"""
quantum_tests.py

โมดูลทดสอบหลักการควอนตัมเชิงข้อมูล:
- สร้าง Entanglement Graph จาก MI Matrix
- ตรวจสอบการละเมิด Bell-like Inequality
- วัด KL-Divergence ระหว่าง Entropy Distribution
- ใช้งานร่วมกับ quantum_utilities.py

ผู้พัฒนา: Apichet & Quantum Mechanics GPT
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import entropy as scipy_entropy

from quantum_utilities import build_entanglement_graph


# ------------------------------
# 📊 Visualize Entanglement Network
# ------------------------------

def plot_entanglement_network(mi_matrix: np.ndarray, threshold: float = 0.1) -> nx.Graph:
    """
    วาดกราฟความพัวพัน (Entanglement Graph) จาก Mutual Information Matrix

    Args:
        mi_matrix (np.ndarray): เมทริกซ์ mutual information [n x n]
        threshold (float): ค่าขั้นต่ำในการเชื่อมโยง node

    Returns:
        networkx.Graph: กราฟความพัวพัน
    """
    G = build_entanglement_graph(mi_matrix, threshold)
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", edge_color="gray")
    plt.title(f"Quantum Entanglement Graph (Threshold = {threshold})")
    plt.tight_layout()
    plt.show()

    return G


# ------------------------------
# 🔬 Bell-like Inequality Test
# ------------------------------

def test_bell_violation(mi_matrix: np.ndarray, threshold: float = 0.1) -> int:
    """
    ทดสอบการละเมิด Bell-like Inequality แบบพื้นฐาน:
    ถ้า MI(A,B) + MI(B,C) < MI(A,C) - threshold → อาจมี non-local correlation

    Args:
        mi_matrix (np.ndarray): Mutual information matrix
        threshold (float): ค่าความคลาดเคลื่อนเล็กน้อยที่ยอมรับได้

    Returns:
        int: จำนวนครั้งที่พบว่าละเมิด
    """
    violations = 0
    n = mi_matrix.shape[0]

    for a in range(n):
        for b in range(n):
            for c in range(n):
                if a != b and b != c and a != c:
                    lhs = mi_matrix[a, b] + mi_matrix[b, c]
                    rhs = mi_matrix[a, c]
                    if lhs < rhs - threshold:
                        violations += 1

    print(f"🔔 Bell-like violations detected: {violations}")
    return violations


# ------------------------------
# 📉 KL Divergence between distributions
# ------------------------------

def kl_divergence(p_real: np.ndarray, p_sim: np.ndarray) -> float:
    """
    คำนวณ KL Divergence ระหว่าง 2 probability distributions

    Args:
        p_real (np.ndarray): Distribution จากข้อมูลจริง
        p_sim (np.ndarray): Distribution จาก RNG

    Returns:
        float: ค่า KL divergence
    """
    p_real = np.asarray(p_real) + 1e-9  # ป้องกัน log(0)
    p_sim = np.asarray(p_sim) + 1e-9
    kl = scipy_entropy(p_real, p_sim)
    print(f"📏 KL-Divergence = {kl:.5f}")
    return kl


# ------------------------------
# 📈 Optional: Compare Distributions
# ------------------------------

def plot_entropy_distributions(real_entropies, sim_entropies, bins=30):
    """
    วาด histogram เปรียบเทียบ entropy ของ draw จริง vs สุ่ม

    Args:
        real_entropies (list): ค่า entropy ของ draw จริง
        sim_entropies (list): ค่า entropy ของ draw จาก RNG
    """
    plt.figure(figsize=(10, 5))
    plt.hist(real_entropies, bins=bins, alpha=0.6, label="Real Draws", color='blue')
    plt.hist(sim_entropies, bins=bins, alpha=0.6, label="Pure RNG", color='orange')
    plt.axvline(np.mean(real_entropies), color='blue', linestyle='--', label='Mean Real Entropy')
    plt.axvline(np.mean(sim_entropies), color='orange', linestyle='--', label='Mean RNG Entropy')
    plt.xlabel("Entropy")
    plt.ylabel("Frequency")
    plt.title("Entropy Distribution: Real vs RNG")
    plt.legend()
    plt.tight_layout()
    plt.show()
