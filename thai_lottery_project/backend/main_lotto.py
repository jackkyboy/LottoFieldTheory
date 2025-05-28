# ===== 📁 SYSTEM SETUP /Users/apichet/Downloads/thai_lottery_project/thai_lottery_project/backend/main_lotto.py =====
# ===== 🔧 Imports =====
# ===== 🔧 Standard Library =====
from pathlib import Path

# ===== 📚 External Libraries =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA

# ===== 🧠 Core Modules (Local) =====
from engine_lotto_field import generate_lotto_field, compute_similarity
from quantum_core import generate_schrodinger_superposition, simulate_wavefunction_collapse
from quantum_entanglement_core import (
    build_feature_8d_field,
    simulate_collapse_from_8d,
    run_quantum_prediction_pipeline,
    compute_entanglement_entropy
)
from quantum_ensemble_core import quantum_ensemble_picker
from quantum_multiverse_simulator import simulate_multiverse, analyze_multiverse_draws
from quantum_model_bank import MODEL_LIST
from quantum_utilities import (
    rank_draws_with_qbs,
    detect_paradox,
    compute_mutual_information,
    build_entanglement_graph,
    entropy_test_vs_random,
    calculate_draw_entropy,
    find_shadow_draws
)
from quantum_tests import (
    plot_entanglement_network,
    test_bell_violation,
    kl_divergence,
    plot_entropy_distributions
)
from quantum_verifier import verify_predictions_against_truth
from context_extractor import build_info_context_vector
from quantum_context_core import weight_superposition_with_context  # <- ต้องไม่ลืม import นี้
from lotto_embedding_core import DigitEmbedding, string_to_digits, generate_latent_space, boltzmann_sample

# ===== 📁 Load Cleaned Lotto Data =====
# ===== 📁 Load Cleaned Lotto Data =====
DATA_FILE = Path("/Users/apichet/Downloads/lotto_110year_cleaned.csv")
df = pd.read_csv(DATA_FILE)

# ✅ เตรียม draws สำหรับ QuantumLottoPredictor
draws = df["first_prize"].dropna().astype(str).tolist()

# ===== 🎲 STEP 1: Generate Quantum Field (1000D) =====
field = generate_lotto_field(df, dimension=1000)

# ===== 📉 STEP 1.5: Entropy Test vs Pure Random =====
entropy_result = entropy_test_vs_random(field)
print("\n📉 Entropy Test vs Pure Random:")
print(entropy_result)

# ===== 🔗 STEP 1.6: Compute Mutual Information Matrix (with discretization) =====
mi_matrix = compute_mutual_information(field, n_bins=10)

# ===== 🌀 STEP 2: Reduce to 2D for Visualization (PCA) =====
pca = PCA(n_components=2)
pca_result = pca.fit_transform(field)

# ===== 🔮 STEP 3: Standard Superposition & Collapse =====
state_ids = [f"Draw_{i}" for i in range(len(pca_result))]
random_seed = int(pd.Timestamp.now().timestamp()) % (2**32 - 1)
superposition_full = generate_schrodinger_superposition(state_ids, seed=random_seed)
collapsed_draw_std = simulate_wavefunction_collapse(superposition_full, seed=random_seed)
collapsed_index_std = int(collapsed_draw_std.split("_")[1])
collapsed_point_std = pca_result[collapsed_index_std]

# ===== 🔗 STEP 4: Quantum Entangled Collapse =====
collapsed_draw_ent, entangled_set = run_quantum_prediction_pipeline(df, top_k=5, collapse_seed=12345)

# ===== 🧬 STEP 5: Compute Entanglement Entropy =====
entropy = compute_entanglement_entropy(field, base_index=0)
print(f"\n🌀 Entanglement Entropy (Draw_0) = {entropy:.4f}")

# ===== 🌊 STEP 6: Collapse from 8D Quantum Field =====
field_8d = build_feature_8d_field(df)
collapsed_index_8d, collapse_probabilities = simulate_collapse_from_8d(field_8d, seed=42)

# ===== 🎛 STEP 7: Interference-Based Collapse =====
superpositions = [
    superposition_full,
    generate_schrodinger_superposition(state_ids, seed=101),
    generate_schrodinger_superposition(state_ids, seed=202),
]
interfered_sp = quantum_ensemble_picker(superpositions)
collapsed_draw_ensemble = simulate_wavefunction_collapse(interfered_sp, seed=123)
collapsed_index_ensemble = int(collapsed_draw_ensemble.split("_")[1])

# ===== 🧪 DEMO: Simulated Superposition Sample =====
states_sample = [f"Draw_{i}" for i in range(5)]
superposition_sample = generate_schrodinger_superposition(states_sample, temperature=0.5, seed=42)
print("\n🔬 Simulated Superposition (Sample Draws):")
for state, prob in superposition_sample.items():
    print(f" - {state}: {prob:.4f}")

# ===== 🌌 STEP 8: Simulate Multiverse Outcomes =====
df_multiverse = simulate_multiverse(state_ids, models=MODEL_LIST, n_worlds=100)
top_recommend = analyze_multiverse_draws(df_multiverse, entropy_threshold=6.0, top_k=5)
print("\n🌌 Top Recommended Draws from Multiverse:")
print(top_recommend)

# ===== 🤖 STEP 9: Quantum Believability Scoring (QBS) =====
df_qbs = rank_draws_with_qbs(df_multiverse, entangled_list=entangled_set)
print("\n🎯 Recommended Draws by QBS:")
print(df_qbs.head())

# ===== 🧩 STEP 10: Detect Paradoxes in Draws =====
paradox_list = detect_paradox(df_multiverse, entangled_list=entangled_set)
print("\n🧩 Quantum Paradoxes Detected:")
for draw, reason in paradox_list:
    print(f" - {draw}: {reason}")

# ===== 🔗 STEP 11: Entanglement Graph & Tests =====
print("\n📡 Mutual Information & Bell Test Analysis:")
plot_entanglement_network(mi_matrix, threshold=0.1)
violations = test_bell_violation(mi_matrix)
print(f"🔔 Bell Violations Detected: {violations}")

real_entropies = [calculate_draw_entropy(d) for d in field]
simulated = [np.random.randint(0, 10, len(field[0])) for _ in range(len(field))]
sim_entropies = [calculate_draw_entropy(d) for d in simulated]
kl = kl_divergence(real_entropies, sim_entropies)
plot_entropy_distributions(real_entropies, sim_entropies)

# ===== ✅ Verification & Buying =====
cols_to_check = ['first_prize', 'last2']
log_df = verify_predictions_against_truth(df, predictions=["Draw_113", "Draw_317", "Draw_697"], target_cols=cols_to_check)
print(log_df)

buy_candidates = df_qbs[df_qbs["paradox_score"] == 0].head(5)
print("\n📌 Top 5 Buy Candidates (QBS valid, no paradox):")
print(buy_candidates[["draw", "QBS"]])

shadow_candidates = find_shadow_draws(entangled_set, df_multiverse)
print("\n👻 Shadow Draws (Entangled but never collapsed):")
print(shadow_candidates)

# ===== 🧾 UTIL: Print BUY GUIDE for Specific Draws =====
def print_buy_guide_top_k(df: pd.DataFrame, indices: list, title="BUY GUIDE"):
    print(f"\n🧾 {title}")
    for rank, idx in enumerate(indices, 1):
        if idx >= len(df):
            print(f"⚠️  Draw_{idx} not found (index out of range)")
            continue
        row = df.iloc[idx]
        print(f"\n🔗 Draw_{idx}  |  Rank #{rank}")
        print("📅 Date:", row.get("date", "(no date)"))
        print(f"🏆 First Prize: {row['first_prize']}")
        print(f"🎯 Front 3 Digits: {row['front3_1']}, {row['front3_2']}")
        print(f"🎯 Last 3 Digits: {row['last3_1']}, {row['last3_2']}, {row['last3_3']}")
        print(f"🔚 Last 2 Digits: {row['last2']}")

# ===== 📌 Summary of Key Collapses =====
print("\n🌌 Quantum Simulation Results")
print(f"🔮 Standard Collapse → {collapsed_draw_std} (index={collapsed_index_std})")
print(f"🔗 Entangled Collapse → {collapsed_draw_ent}")
print(f"🌊 8D Collapse Index → Draw_{collapsed_index_8d}")
print(f"🌈 Interfered Collapse → {collapsed_draw_ensemble} (index={collapsed_index_ensemble})")
print(f"🌀 Entanglement Entropy (Draw_0) = {entropy:.4f}")

# ===== 📉 Visualize Superposition Sample =====
draws = list(superposition_sample.keys())    # ✅ FIXED: add draws
probs = list(superposition_sample.values())  # ✅ FIXED: add probs

plt.figure(figsize=(8, 4))
plt.bar(draws, probs, color='orange')
plt.title("Quantum Superposition Over Sample Draws")
plt.ylabel("Probability")
plt.xlabel("Draw State")
plt.grid(True)
plt.tight_layout()
plt.show()

# ===== 📈 Visualize PCA Collapse Point =====
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6, label="All Draws")
plt.scatter(collapsed_point_std[0], collapsed_point_std[1], color='red', s=100, label="Standard Collapse")
plt.title("Thai Lottery - Quantum Field Collapse (PCA View)")
plt.xlabel("Latent Dim 1")
plt.ylabel("Latent Dim 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# ===== 💾 Save Field =====
BASE_DIR = Path(__file__).resolve().parent
np.savetxt(BASE_DIR / "lotto_1000d_field.csv", field, delimiter=',')

# ===== 🔗 Similarity Matrix (optional) =====
similarity_matrix = compute_similarity(field)

# ===== 📌 BUY GUIDES =====
top_k_indices = [collapsed_index_std] + [int(e.split("_")[1]) for e in entangled_set[:4]]
print_buy_guide_top_k(df, top_k_indices, title="BUY GUIDE from Standard + Entangled")
print_buy_guide_top_k(df, [collapsed_index_8d], title="BUY GUIDE from 8D Collapse")
print_buy_guide_top_k(df, [collapsed_index_ensemble], title="BUY GUIDE from Interfered Collapse")

# ===== 🔢 Load Draws from Real Data =====
from lotto_embedding_core import DigitEmbedding, generate_latent_space, boltzmann_sample, QuantumLottoPredictor

draws = df["first_prize"].dropna().astype(str).tolist()

# ===== 🔬 Low-level Sampling from DigitEmbedding =====
print("\n🔬 Sampling โดยตรงจาก DigitEmbedding (Low-level Boltzmann Sampling):")

model = DigitEmbedding(embedding_dim=16)
model.fit(draws)  # ✅ placeholder — ไม่ error

latent_space = generate_latent_space(draws, model)
index, prob, energy = boltzmann_sample(latent_space, gmm=None, temperature=1.0)

print("🧬 Boltzmann Sampled Draw:")
print(f"🔥 Selected Draw: {draws[index]}")
print(f"   Probability: {prob:.5f}, Energy: {energy:.5f}")

# ===== 🤖 High-level QuantumLottoPredictor =====
print("\n🤖 Sampling ด้วย QuantumLottoPredictor:")

predictor = QuantumLottoPredictor(embedding_dim=16, n_clusters=5)
predictor.fit(draws)

draw, prob, energy = predictor.sample_draw(temperature=1.0)
print(f"🎯 Suggested Draw: {draw} | Prob: {prob:.5f} | Energy: {energy:.4f}")

# ===== 🧠 Explain Specific Draw =====
target_draw = draws[-1]  # วิเคราะห์งวดล่าสุดในฐานข้อมูล
print(f"\n🧠 Explanation of Draw '{target_draw}':")
info = predictor.explain_draw(target_draw)
for k, v in info.items():
    print(f"   {k}: {v}")
