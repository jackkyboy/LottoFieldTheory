#  /Users/apichet/Downloads/thai_lottery_project/thai_lottery_project/backend/quantum_multiverse_simulator.py
# /Users/apichet/Downloads/thai_lottery_project/thai_lottery_project/backend/quantum_multiverse_simulator.py
# /Users/apichet/Downloads/thai_lottery_project/thai_lottery_project/backend/quantum_multiverse_simulator.py

import numpy as np
import pandas as pd
from collections import Counter
from quantum_core import simulate_wavefunction_collapse  # ✅ FIXED
from quantum_entanglement_core import compute_entanglement_entropy

__all__ = [
    "simulate_multiverse",
    "compute_shannon_entropy",
    "analyze_multiverse_draws"
]


def simulate_multiverse(state_ids, models, n_worlds=50, seed_base=42, return_df=True):
    """
    Simulate quantum collapses across multiple models and universes.

    Args:
        state_ids (list): List of draw identifiers (e.g., ["Draw_0", ..., "Draw_N"])
        models (list): List of model functions (each returns a superposition dict)
        n_worlds (int): Number of universes per model
        seed_base (int): Base seed to ensure reproducibility
        return_df (bool): If True, return results as DataFrame

    Returns:
        list[dict] or pd.DataFrame: Simulation results across the multiverse
    """
    results = []

    for model_id, model_fn in enumerate(models):
        for i in range(n_worlds):
            seed = seed_base + model_id * 1000 + i
            superposition = model_fn(state_ids, seed=seed)
            collapsed = simulate_wavefunction_collapse(superposition, seed=seed)
            entropy = compute_shannon_entropy(superposition)

            results.append({
                "universe_id": f"Model{model_id}_Run{i}",
                "model": f"Model_{model_id}",
                "collapsed": collapsed,
                "entropy": entropy,
                "seed": seed
            })

    results.sort(key=lambda r: r["entropy"], reverse=True)
    return pd.DataFrame(results) if return_df else results


def compute_shannon_entropy(prob_dist):
    """
    Compute Shannon entropy from a probability distribution.

    Args:
        prob_dist (dict): e.g., {"Draw_0": 0.1, "Draw_1": 0.2, ...}

    Returns:
        float: entropy in bits
    """
    probs = np.array(list(prob_dist.values()))
    probs = np.clip(probs, 1e-12, 1)  # Avoid log(0)
    return -np.sum(probs * np.log2(probs))


def analyze_multiverse_draws(df_results, entropy_threshold=6.0, top_k=5):
    """
    Analyze multiverse simulation results to find top draw candidates.

    Args:
        df_results (pd.DataFrame): Output from simulate_multiverse()
        entropy_threshold (float): Threshold for "low entropy" universes
        top_k (int): Number of top draws to return

    Returns:
        pd.DataFrame: Summary of top draw recommendations
    """
    counts = Counter(df_results["collapsed"])
    summary = []

    for draw_id, freq in counts.items():
        subset = df_results[df_results["collapsed"] == draw_id]
        avg_entropy = subset["entropy"].mean()
        min_entropy = subset["entropy"].min()
        reliable = avg_entropy <= entropy_threshold

        summary.append({
            "draw": draw_id,
            "frequency": freq,
            "avg_entropy": avg_entropy,
            "min_entropy": min_entropy,
            "reliable": reliable
        })

    df_summary = pd.DataFrame(summary)
    df_summary = df_summary.sort_values(
        by=["reliable", "frequency", "avg_entropy"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    return df_summary.head(top_k)



def map_qbs_to_lotto_info(df: pd.DataFrame, df_qbs: pd.DataFrame, top_k=5):
    """
    Combine QBS results with actual lottery draw info from df.
    
    Args:
        df: Cleaned lotto dataframe
        df_qbs: DataFrame with 'draw' like 'Draw_123', 'QBS', etc.
        top_k: Number of top draws to include
    
    Returns:
        DataFrame: Combined info with real draw results + QBS
    """
    result = []

    for _, row in df_qbs.head(top_k).iterrows():
        draw_idx = int(row["draw"].split("_")[1])
        lotto_row = df.iloc[draw_idx]

        result.append({
            "draw_id": row["draw"],
            "QBS": row["QBS"],
            "frequency": row["frequency"],
            "avg_entropy": row["avg_entropy"],
            "entangled": row["entangled"],
            "first_prize": lotto_row["first_prize"],
            "front3_1": lotto_row["front3_1"],
            "front3_2": lotto_row["front3_2"],
            "last3_1": lotto_row["last3_1"],
            "last3_2": lotto_row["last3_2"],
            "last3_3": lotto_row["last3_3"],
            "last2": lotto_row["last2"],
            "date": lotto_row.get("date", "(no date)")
        })

    return pd.DataFrame(result)
