import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ==== 🔗 Generate multiple entangled wave states ====
def generate_multi_wave_top_k(field_vectors: np.ndarray, base_index: int, top_k=3):
    """
    Return indices of top-k most entangled (similar) states with the base draw.
    """
    similarities = cosine_similarity([field_vectors[base_index]], field_vectors)[0]
    top_indices = np.argsort(similarities)[::-1][1:top_k+1]
    return top_indices, similarities[top_indices]

# ==== 🌌 Build 8D field with embedded relationships ====
def build_feature_8d_field(df: pd.DataFrame) -> np.ndarray:
    """
    Build an 8D quantum field vector for each draw, combining features.
    """
    selected = ['first_prize', 'last2', 'front3_1', 'front3_2', 'last3_1', 'last3_2', 'last3_3']
    df[selected] = df[selected].apply(pd.to_numeric, errors='coerce')

    field = []
    for _, row in df.iterrows():
        v = np.zeros(8)
        v[0] = int(row['first_prize']) % 1000000 / 1e6
        v[1] = int(row['last2']) / 100
        v[2] = int(row['front3_1']) / 999
        v[3] = int(row['front3_2']) / 999
        v[4] = int(row['last3_1']) / 999
        v[5] = int(row['last3_2']) / 999
        v[6] = int(row['last3_3']) / 999
        v[7] = np.mean(v[:7])  # add entangled latent mean
        field.append(v)
    
    return np.array(field)

# ==== 🔮 Full Entanglement-Based Prediction Pipeline ====
def run_quantum_prediction_pipeline(df: pd.DataFrame, top_k=3, collapse_seed=None):
    """
    Run the full pipeline: build field, find entangled states, and collapse.
    """
    field_8d = build_feature_8d_field(df)
    base_index = 0  # most recent draw
    top_indices, scores = generate_multi_wave_top_k(field_8d, base_index, top_k=top_k)

    print(f"🔗 Top-{top_k} entangled draws to Draw_0:")
    for i, score in zip(top_indices, scores):
        print(f" - Draw_{i} (similarity={score:.4f})")

    # Create superposition of entangled draws only
    entangled_state_ids = [f"Draw_{i}" for i in top_indices]
    from quantum_core import generate_schrodinger_superposition, simulate_wavefunction_collapse

    superposition = generate_schrodinger_superposition(entangled_state_ids, seed=collapse_seed)
    collapsed = simulate_wavefunction_collapse(superposition, seed=collapse_seed)

    return collapsed, entangled_state_ids



def compute_entanglement_entropy(field_vectors: np.ndarray, base_index: int) -> float:
    """
    Compute Shannon entropy of entanglement based on cosine similarity to all other draws.
    """
    similarities = cosine_similarity([field_vectors[base_index]], field_vectors)[0]
    
    # Normalize similarities to probabilities
    probs = similarities / similarities.sum()
    probs = np.clip(probs, 1e-12, 1)  # Avoid log(0)

    entropy = -np.sum(probs * np.log2(probs))
    return entropy



import numpy as np

def simulate_collapse_from_8d(field_vectors: np.ndarray, seed=None):
    """
    Simulate wavefunction collapse from 8D field vectors.
    
    Args:
        field_vectors (np.ndarray): Shape (n_samples, 8)
        seed (int): Optional random seed for reproducibility
    
    Returns:
        collapsed_index (int): Index of selected (collapsed) draw
        probabilities (np.ndarray): Probability distribution over all draws
    """
    if seed is not None:
        np.random.seed(seed)

    # Step 1: Compute |ψ| = norm of each 8D vector
    amplitudes = np.linalg.norm(field_vectors, axis=1)

    # Step 2: Convert to probabilities (Born rule)
    probabilities = amplitudes**2
    probabilities /= probabilities.sum()

    # Step 3: Collapse to one outcome
    collapsed_index = np.random.choice(len(probabilities), p=probabilities)
    return collapsed_index, probabilities



import numpy as np

def generate_schrodinger_superposition(possible_states, temperature=1.0, seed=None):
    """
    Create a probability distribution over possible draw states simulating quantum superposition.

    Args:
        possible_states (list): List of draw identifiers (e.g., ['Draw_0', 'Draw_1', ...])
        temperature (float): Controls sharpness of distribution (higher = more uniform)
        seed (int, optional): Seed for reproducibility

    Returns:
        dict: Mapping of draw ID → probability
    """
    if seed is not None:
        np.random.seed(seed)

    amplitudes = np.random.rand(len(possible_states))  # Random wave amplitudes
    probabilities = np.exp(amplitudes / temperature)   # Softmax-like
    probabilities /= probabilities.sum()               # Normalize

    return dict(zip(possible_states, probabilities))



