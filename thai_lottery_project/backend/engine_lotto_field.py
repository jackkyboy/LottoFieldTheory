import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ====== CONSTANTS ======
FEATURES = ['first_prize', 'last2', 'front3_1', 'front3_2', 'last3_1', 'last3_2', 'last3_3']

# ====== FUNCTION: Embed each lottery draw into high-dimensional vector ======
def embed_draw_to_vector(row, dimension=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    vec = np.zeros(dimension)
    indices = [int(x) % dimension for x in row[FEATURES]]
    for i, idx in enumerate(indices):
        vec[idx] += np.exp(-(i + 1) / 2)  # decay weighting
    return vec

# ====== MAIN ENGINE FUNCTIONS ======

def generate_lotto_field(df: pd.DataFrame, dimension=1000, seed=None) -> np.ndarray:
    """
    Create a field of high-dimensional vectors from lottery draws.
    """
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors='coerce')
    field_vectors = np.vstack(df.apply(embed_draw_to_vector, axis=1, args=(dimension, seed)))
    return field_vectors

def compute_similarity(field_vectors: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix between all draw vectors.
    """
    return cosine_similarity(field_vectors)

def visualize_field(field_vectors: np.ndarray, title="Lottery Field Projection"):
    """
    Reduce the high-dimensional vectors to 2D using PCA and visualize.
    """
    pca = PCA(n_components=2)
    projected = pca.fit_transform(field_vectors)

    plt.figure(figsize=(10, 6))
    plt.scatter(projected[:, 0], projected[:, 1], alpha=0.6, c='blue')
    plt.title(title)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.grid(True)
    plt.show()
