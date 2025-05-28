# quantum_prediction_core.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime

### 🧠 1. Superposition

def generate_schrodinger_superposition(number_pool, weights=None):
    """
    Generate a superposition state over a number pool using optional amplitude weights.
    """
    if weights is None:
        weights = np.ones(len(number_pool)) / len(number_pool)
    else:
        weights = np.array(weights)
        weights /= weights.sum()

    amplitudes = np.sqrt(weights)
    return dict(zip(number_pool, amplitudes))


### 🔗 2. Entanglement

ENTANGLED_DIGITS = {'3', '6', '8'}


def entanglement_boost(number):
    """
    Boost amplitude if number contains entangled digits.
    """
    boost = sum(1 for d in str(number) if d in ENTANGLED_DIGITS)
    return 1.0 + 0.1 * boost


### 🧭 5. Field Mapping

def build_feature_8d_field(numbers):
    """
    Map each number to an 8-dimensional feature vector.
    """
    field = {}
    for num in numbers:
        s = str(num).zfill(6)
        vec = np.array([
            int(s[0]), int(s[1]), int(s[2]),
            int(s[3]), int(s[4]), int(s[5]),
            sum(int(d) for d in s) % 10,
            int(s[-1]) % 2
        ])
        field[num] = vec
    return field


### 🌊 3. Collapse from 8D

def simulate_collapse_from_8d(field, amplitudes):
    """
    Collapse a wavefunction based on amplitude over 8D field.
    """
    probs = np.array([np.abs(amplitudes[n])**2 for n in field])
    probs /= probs.sum()
    keys = list(field.keys())
    chosen = np.random.choice(keys, p=probs)
    return chosen


### 🔄 4. Stochastic Process

def stochastic_optimal_prediction(history, top_n=5):
    """
    Sample most frequent numbers as optimal guesses with noise.
    """
    freq = Counter(history)
    top = freq.most_common(top_n * 2)
    keys = [k for k, _ in top]
    weights = np.array([v for _, v in top], dtype=float)
    weights += np.random.rand(len(weights))
    weights /= weights.sum()

    picks = np.random.choice(keys, size=top_n, replace=False, p=weights)
    return picks.tolist()


### 🎲 6. Probabilistic Interference

def quantum_ensemble_picker(predictions_list):
    """
    Combine multiple prediction lists using interference-like merging.
    """
    combined = Counter()
    for preds in predictions_list:
        for i, val in enumerate(preds):
            combined[val] += (len(preds) - i)

    total = sum(combined.values())
    final = [(k, v / total) for k, v in combined.items()]
    final.sort(key=lambda x: x[1], reverse=True)
    return final[:5]


### 🎛️ Master Pipeline

def run_quantum_prediction_pipeline(history_numbers):
    pool = sorted(set(history_numbers))

    # Superposition base
    superpos = generate_schrodinger_superposition(pool)

    # Apply entanglement boost
    for k in superpos:
        superpos[k] *= entanglement_boost(k)

    # Build 8D field
    field = build_feature_8d_field(pool)

    # Collapse
    pick = simulate_collapse_from_8d(field, superpos)

    # Ensemble
    ensemble = [stochastic_optimal_prediction(history_numbers)] * 3
    ensemble.append([pick])

    final = quantum_ensemble_picker(ensemble)
    return final
