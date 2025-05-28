from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from thai_lottery_project.backend.achive.quantum_paradox_lottery import quantum_paradox_lottery, quantum_paradox_lottery_6digit_fullworld

# 🔗 DATA SOURCE
CLEANED_FILE = Path("/Users/apichet/Downloads/lotto_110year_cleaned.csv")
SEED = 778

# 🕝 FORMAT PER COLUMN
FORMAT_RULES = {
    'first_prize': 6,
    'last2': 2,
    'front3_1': 3,
    'front3_2': 3,
    'last3_1': 3,
    'last3_2': 3,
    'last3_3': 3,
}

def lock_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    print(f"🔐 Locked random seed to: {seed}")

def extract_by_column(filepath):
    df = pd.read_csv(filepath)
    clean = {col: [] for col in FORMAT_RULES}
    for _, row in df.iterrows():
        for col, width in FORMAT_RULES.items():
            val = row.get(col)
            if pd.isnull(val):
                continue
            s = str(val).strip()
            if s.isdigit():
                padded = s.zfill(width)
                if len(padded) == width:
                    clean[col].append(padded)
    return clean

def build_quantum_states(grouped_numbers):
    psi_dict = {}
    for col, numbers in grouped_numbers.items():
        freq = pd.Series(numbers).value_counts()
        total = freq.sum()
        psi = {num: np.sqrt(count / total) for num, count in freq.items()}
        psi_dict[col] = psi
        print(f"⚛️ {col}: {len(psi)} states")
    return psi_dict

def simulate_measurements(psi_dict, filepath):
    df = pd.read_csv(filepath)
    mapping = {}
    for _, row in df.iterrows():
        for col in FORMAT_RULES:
            val = row.get(col)
            if pd.isnull(val): continue
            s = str(val).strip().zfill(FORMAT_RULES[col])
            if s not in mapping:
                mapping[s] = []
            mapping[s].append((row['date'], col))

    print("\n🌀 Quantum Collapse (One per reward):")
    results = {}
    for col, psi in psi_dict.items():
        keys = list(psi.keys())
        probs = np.array([np.abs(psi[k])**2 for k in keys])
        pick = np.random.choice(keys, p=probs)
        origin = mapping.get(pick, [(None, None)])[0]
        print(f"   🎯 {col}: {pick} @ {origin[0]}")
        results[col] = pick
    return results

def plot_all_probabilities(psi_dict, top_n=15):
    for col, psi in psi_dict.items():
        keys = list(psi.keys())
        probs = np.array([np.abs(psi[k])**2 for k in keys])
        idx = np.argsort(probs)[::-1]
        top_keys = np.array(keys)[idx][:top_n]
        top_probs = probs[idx][:top_n]

        plt.figure(figsize=(12, 4))
        plt.bar(top_keys, top_probs)
        plt.title(f"📊 Top {top_n} Probabilities for {col}")
        plt.xlabel("Number")
        plt.ylabel("Probability")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

def print_probabilities(psi_dict, top_n=15):
    print("\n📋 Ranked Probability Table (|cᵒ|²):")
    for col, psi in psi_dict.items():
        keys = list(psi.keys())
        probs = np.array([np.abs(psi[k])**2 for k in keys])
        idx = np.argsort(probs)[::-1]
        top_keys = np.array(keys)[idx][:top_n]
        top_probs = probs[idx][:top_n]

        print(f"\n🔹 {col.upper()} (Top {top_n}):")
        for k, p in zip(top_keys, top_probs):
            print(f"   {k}: {p:.5f}")

def build_full_worlds(filepath):
    df = pd.read_csv(filepath)
    worlds = []
    for _, row in df.iterrows():
        date = pd.to_datetime(row.get("date"), errors="coerce")
        if pd.isnull(date):
            continue
        world = {
            'date': date,
            'first_prize': str(row.get('first_prize')).zfill(6) if pd.notnull(row.get('first_prize')) else None,
            'last2': str(row.get('last2')).zfill(2) if pd.notnull(row.get('last2')) else None,
            'front3': [
                str(row.get('front3_1')).zfill(3) if pd.notnull(row.get('front3_1')) else None,
                str(row.get('front3_2')).zfill(3) if pd.notnull(row.get('front3_2')) else None,
            ],
            'last3': [
                str(row.get('last3_1')).zfill(3) if pd.notnull(row.get('last3_1')) else None,
                str(row.get('last3_2')).zfill(3) if pd.notnull(row.get('last3_2')) else None,
                str(row.get('last3_3')).zfill(3) if pd.notnull(row.get('last3_3')) else None,
            ]
        }
        world['front3'] = [x for x in world['front3'] if x]
        world['last3'] = [x for x in world['last3'] if x]
        worlds.append(world)
    return worlds

def predict_next_draw_from_fullworld(full_worlds, top_n=5):
    latest_date = max(w['date'] for w in full_worlds)
    past_worlds = [w for w in full_worlds if w['date'] < latest_date]
    return quantum_paradox_lottery_6digit_fullworld(
        full_worlds=past_worlds,
        user_seed=None,
        top_n=top_n
    )

def trace_energy_breakdown_for(predicted_numbers, full_worlds):
    print("\n🔍 Tracing energy breakdown per predicted number:")
    for number, prob in predicted_numbers:
        for world in reversed(full_worlds):
            if world['first_prize'] == number:
                print(f"\n🗕 Date: {world['date'].strftime('%d %b %Y')} → 🎯 {number} (p={prob:.6f})")
                print(f"🔹 front3: {world['front3']}, last3: {world['last3']}, last2: {world['last2']}")
                subpatterns = [number[:3], number[1:4], number[3:], number[-2:], number[-3:]]
                matched = []
                for pattern in subpatterns:
                    if pattern in world['front3']:
                        matched.append(f"✅ match front3: {pattern}")
                    if pattern in world['last3']:
                        matched.append(f"✅ match last3: {pattern}")
                    if pattern == world['last2']:
                        matched.append(f"✅ match last2: {pattern}")
                if matched:
                    print("🧪 Pattern resonance:")
                    for m in matched:
                        print(f"   {m}")
                else:
                    print("🧪 Pattern resonance: ❌ None")
                break

if __name__ == "__main__":
    lock_seed()
    print("\n--- CLASSIC QUANTUM FIELD ---")
    numbers_by_column = extract_by_column(CLEANED_FILE)
    psi_by_column = build_quantum_states(numbers_by_column)
    simulate_measurements(psi_by_column, CLEANED_FILE)
    print_probabilities(psi_by_column, top_n=10)
    plot_all_probabilities(psi_by_column)

    print("\n--- QUANTUM FULL COLLAPSE WORLD (6 DIGIT) ---")
    full_worlds = build_full_worlds(CLEANED_FILE)
    for w in full_worlds[-3:]:
        print(f"🧹 {w['date'].strftime('%d %b %Y')} | first_prize={w['first_prize']} | front3={w['front3']} | last3={w['last3']} | last2={w['last2']}")

    print("\n🌟 Quantum Paradox Prediction for FIRST PRIZE (6-digit)")
    results = predict_next_draw_from_fullworld(
        full_worlds,
        top_n=5
    )
    for num, prob in results:
        print(f"   {num} → {prob:.6f}")

    trace_energy_breakdown_for(results, full_worlds)


results_df = pd.DataFrame(results, columns=["number", "probability"])
results_df.to_csv("predicted_results.csv", index=False)
print("📁 Saved results to predicted_results.csv")
