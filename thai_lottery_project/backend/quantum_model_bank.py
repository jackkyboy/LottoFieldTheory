from quantum_core import generate_schrodinger_superposition

def model_standard(state_ids, seed):
    return generate_schrodinger_superposition(state_ids, seed=seed)

def model_soft(state_ids, seed):
    return generate_schrodinger_superposition(state_ids, seed=seed, temperature=2.0)

def model_focus(state_ids, seed):
    return generate_schrodinger_superposition(state_ids, seed=seed, temperature=0.3)

MODEL_LIST = [model_standard, model_soft, model_focus]
