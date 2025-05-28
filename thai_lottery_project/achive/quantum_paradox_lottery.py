# quantum_paradox_lottery.py

import numpy as np
import math
from datetime import datetime
from collections import Counter
from thai_lottery_project.backend.achive.quantum_lattice_rnn import CollapseHistoryManager, QuantumLatticeRNN, QuantumLatticeField

# ------------------------
# Utility Functions
# ------------------------

def get_lunar_phase(date):
    diff_days = (date - datetime(2000, 1, 6)).days
    lunar_cycle = 29.530588
    return (diff_days % lunar_cycle) / lunar_cycle

def get_solar_cycle_influence(date):
    offset = date.year - 2000
    return 1 + 0.1 * math.sin(offset * math.pi / 11)

def seed_to_vector(seed):
    if isinstance(seed, str):
        return np.array([sum(ord(c) for c in seed) % 1000, len(seed) % 100])
    elif isinstance(seed, (int, float)):
        return np.array([seed % 1000, (seed // 1000) % 1000])
    elif isinstance(seed, list) and seed:
        return np.array([sum(seed), np.var(seed)])
    return None

def compute_alignment_factor(seed_vector):
    if seed_vector is None or np.linalg.norm(seed_vector) == 0:
        return 1.0
    return 1.0 + (np.linalg.norm(seed_vector) / 1000.0)

# ------------------------
# CLASSIC MODEL (3-digit)
# ------------------------

def quantum_paradox_lottery(history_data, user_seed=None, top_n=5):
    counter = Counter([rec['number'] for rec in history_data])
    total = sum(counter.values())
    base_probs = {k: v / total for k, v in counter.items()}
    now = datetime.now()
    seed_vector = seed_to_vector(user_seed)
    seed_align = compute_alignment_factor(seed_vector)

    adjusted = {}
    for number, base_p in base_probs.items():
        boost = 1.0
        for rec in history_data:
            if rec['number'] == number:
                dt = (now - rec['date']).days
                decay = math.exp(-0.01 * dt)
                lunar = get_lunar_phase(rec['date'])
                solar = get_solar_cycle_influence(rec['date'])
                lunar_boost = 1.2 if 0.0 <= lunar <= 0.1 or 0.45 <= lunar <= 0.55 else 1.0
                boost += decay * lunar_boost * solar
        adjusted[number] = base_p * boost * seed_align

    total_adj = sum(adjusted.values())
    final = {k: v / total_adj for k, v in adjusted.items()}
    return sorted(final.items(), key=lambda x: x[1], reverse=True)[:top_n]

# ------------------------
# RNN-LATTICE MODEL (6-digit)
# ------------------------

def quantum_paradox_lottery_6digit_fullworld(full_worlds, user_seed=None, top_n=5):
    history_data = []
    for w in full_worlds:
        if w.get('first_prize'):
            history_data.append({
                'number': w['first_prize'],
                'date': w['date'],
                'type': 'first_prize'
            })

    manager = CollapseHistoryManager(history_data)
    rnn = QuantumLatticeRNN(manager, time_steps=5, feature_dim=8)
    rnn.train_rnn(epochs=5)

    lattice = QuantumLatticeField(manager, user_seed=user_seed, rnn_predictor=rnn)
    field = lattice.generate_field()
    return lattice.collapse_field(field, top_n=top_n)
