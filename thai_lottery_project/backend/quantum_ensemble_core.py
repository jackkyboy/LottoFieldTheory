from collections import defaultdict
import numpy as np

def quantum_ensemble_picker(superpositions, method="average"):
    combined = defaultdict(list)
    for sp in superpositions:
        for state, prob in sp.items():
            combined[state].append(prob)
    interfered = {}
    for state, probs in combined.items():
        if method == "average":
            interfered[state] = np.mean(probs)
        elif method == "product":
            interfered[state] = np.prod(probs)
        elif method == "max":
            interfered[state] = np.max(probs)
    total = sum(interfered.values())
    for state in interfered:
        interfered[state] /= total
    return interfered
