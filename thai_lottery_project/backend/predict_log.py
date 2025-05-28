from pathlib import Path
import pandas as pd
import json
from datetime import datetime

from engine_lotto_field import generate_lotto_field
from quantum_core import generate_schrodinger_superposition, simulate_wavefunction_collapse

# ===== 🔁 PURE QUANTUM COLLAPSE (No Context Injection) =====

# Load Lotto Data
DATA_FILE = Path("/Users/apichet/Downloads/lotto_110year_cleaned.csv")
df = pd.read_csv(DATA_FILE)

# Generate Quantum Field
field = generate_lotto_field(df, dimension=1000)
state_ids = [f"Draw_{i}" for i in range(len(field))]

print(f"📊 Total Draw States: {len(state_ids)}")

# Generate Superposition
superposition_full = generate_schrodinger_superposition(state_ids, seed=42)
print("\n🎲 Preview of Superposition Probabilities (first 5):")
for k in list(superposition_full.keys())[:5]:
    print(f" - {k}: {superposition_full[k]:.6f}")

# Collapse
prediction = simulate_wavefunction_collapse(superposition_full, seed=999)
print(f"\n🔮 Prediction: {prediction}")
pred_prob = superposition_full.get(prediction, 0.0)
print(f"📈 Prediction Probability: {pred_prob:.6f}")

# Log Prediction
today = datetime.today().strftime("%Y-%m-%d")
log_file = Path("prediction_logs.json")
logs = []

if log_file.exists():
    logs = json.loads(log_file.read_text(encoding="utf-8"))

logs.append({
    "date": today,
    "prediction": prediction,
    "probability": round(pred_prob, 6),
    "note": "pure quantum collapse, no context"
})


draw_index = int(prediction.split("_")[1])
draw_row = df.iloc[draw_index]

print("\n📌 BUY GUIDE")
print(f"📅 Date: {draw_row.get('date', '(no date)')}")
print(f"🏆 First Prize: {draw_row['first_prize']}")
print(f"🎯 Front 3 Digits: {draw_row['front3_1']}, {draw_row['front3_2']}")
print(f"🎯 Last 3 Digits: {draw_row['last3_1']}, {draw_row['last3_2']}, {draw_row['last3_3']}")
print(f"🔚 Last 2 Digits: {draw_row['last2']}")

log_file.write_text(json.dumps(logs, indent=4, ensure_ascii=False), encoding="utf-8")
print(f"\n📁 Logged to {log_file.name}")
