# /Users/apichet/Downloads/thai_lottery_project/thai_lottery_project/backend/quantum_field_12d.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from pathlib import Path
import json
import warnings

warnings.filterwarnings("ignore")

# ====== 🧠 QUANTUM MODULES ======
def generate_schrodinger_superposition(possible_states, temperature=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    amplitudes = np.random.rand(len(possible_states))
    probabilities = np.exp(amplitudes / temperature)
    probabilities /= probabilities.sum()
    return dict(zip(possible_states, probabilities))

def simulate_wavefunction_collapse(prob_distribution, seed=None):
    if seed is not None:
        np.random.seed(seed + 42)  # Offset to separate randomness
    states = list(prob_distribution.keys())
    probs = list(prob_distribution.values())
    collapsed_state = np.random.choice(states, p=probs)
    return collapsed_state

# ====== 📁 FILE SETUP ======
BASE_DIR = Path("/Users/apichet/Downloads")
INPUT_FILE = BASE_DIR / "lotto_110year.csv"
STATE_FILE = BASE_DIR / "quantum_lotto_state.json"

# ====== 📊 LOAD DATA ======
df = pd.read_csv(INPUT_FILE)
features = ['first_prize', 'last2', 'front3_1', 'front3_2', 'last3_1', 'last3_2', 'last3_3']
df[features] = df[features].apply(pd.to_numeric, errors='coerce')
latest_draw_date = df['date'].iloc[0] if 'date' in df.columns else None

# ====== 🔁 CHECK IF COLLAPSE ALREADY DONE ======
if STATE_FILE.exists():
    with open(STATE_FILE, 'r') as f:
        previous_state = json.load(f)
    if previous_state.get("latest_draw_date") == latest_draw_date:
        print("✅ Quantum collapse already performed for this universe (draw date).")
        print(f"🧬 Collapsed to: {previous_state['collapsed_draw_id']}")
        exit(0)

# ====== ⚙️ PROCESS FIELD ======
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# ====== 🔮 SIMULATE UNIVERSE ======
state_ids = [f"Draw_{i}" for i in range(len(pca_result))]
random_seed = int(pd.Timestamp.now().timestamp()) % (2**32 - 1)

superposition = generate_schrodinger_superposition(state_ids, seed=random_seed)
collapsed_draw = simulate_wavefunction_collapse(superposition, seed=random_seed)
collapsed_index = int(collapsed_draw.split("_")[1])
collapsed_point = pca_result[collapsed_index]

# ====== 💾 SAVE UNIVERSE STATE ======
with open(STATE_FILE, 'w') as f:
    json.dump({
        "latest_draw_date": latest_draw_date,
        "collapsed_draw_id": collapsed_draw,
        "seed": random_seed
    }, f)

print("🌌 Quantum collapse complete.")
print(f"🧬 Collapsed to: {collapsed_draw} (seed={random_seed})")

# ====== 📈 VISUALIZE ======
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5, label="Past Draws")
plt.scatter(collapsed_point[0], collapsed_point[1], color='red', label="Quantum-Collapsed Draw", s=100)
plt.title("Thai Lottery Projection in 2D PCA Space with Quantum Collapse")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.show()
