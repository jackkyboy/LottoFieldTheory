# /Users/apichet/Downloads/thai_lottery_project/thai_lottery_project/backend/lotto_embedding_core.py
# lotto_embedding_core.py

import torch
import torch.nn as nn
import numpy as np
from sklearn.mixture import GaussianMixture

# ===== 🎯 Digit Embedding Model =====
class DigitEmbedding(nn.Module):
    def __init__(self, embedding_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(10, embedding_dim)

    def forward(self, digits):
        tensor = torch.tensor(digits, dtype=torch.long)
        embedded = self.embedding(tensor)
        return embedded.mean(dim=0)

    def embed_draw(self, draw_str):
        digits = string_to_digits(draw_str)
        return self.forward(digits)

    def fit(self, draws):
        pass  # Placeholder for future training logic

# ===== 🔢 Utility: Convert draw string to digits =====
def string_to_digits(draw_str):
    return [int(c) for c in str(draw_str).zfill(6)]

# ===== 🔧 Latent Vector & Clustering Functions =====
def generate_latent_space(draw_list, model):
    latent_vectors = []
    for draw in draw_list:
        vec = model.embed_draw(draw).detach().numpy()
        latent_vectors.append(vec)
    return np.vstack(latent_vectors)

def cluster_latent_space(latent_space, n_components=5):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(latent_space)
    return gmm

def compute_energy(vec, gmm):
    probs = gmm.predict_proba(vec.reshape(1, -1))[0]
    energy = -np.log(np.max(probs) + 1e-9)
    return energy

def boltzmann_sample(latent_space, gmm=None, temperature=1.0):
    if gmm:
        energies = np.array([compute_energy(vec, gmm) for vec in latent_space])
    else:
        centroid = latent_space.mean(axis=0)
        energies = np.linalg.norm(latent_space - centroid, axis=1)
        
    probs = np.exp(-energies / temperature)
    probs /= probs.sum()
    idx = np.random.choice(len(latent_space), p=probs)
    return idx, probs[idx], energies[idx]


# ===== 🤖 Predictor Class =====
class QuantumLottoPredictor:
    def __init__(self, embedding_dim=16, n_clusters=5):
        self.model = DigitEmbedding(embedding_dim)
        self.gmm = None
        self.latent_space = None
        self.draws = []
        self.n_clusters = n_clusters

    def fit(self, draws):
        self.draws = draws
        self.latent_space = generate_latent_space(draws, self.model)
        self.gmm = cluster_latent_space(self.latent_space, self.n_clusters)

    def compute_energy(self, draw_str):
        vec = self.model.embed_draw(draw_str).detach().numpy()
        return compute_energy(vec, self.gmm)

    def sample_draw(self, temperature=1.0):
        idx, prob, energy = boltzmann_sample(self.latent_space, self.gmm, temperature)
        return self.draws[idx], prob, energy

    def explain_draw(self, draw_str):
        vec = self.model.embed_draw(draw_str).detach().numpy()
        probs = self.gmm.predict_proba(vec.reshape(1, -1))[0]
        closest_cluster = np.argmax(probs)
        return {
            "draw": draw_str,
            "cluster_probs": probs.tolist(),
            "closest_cluster": closest_cluster,
            "energy": compute_energy(vec, self.gmm)
        }
