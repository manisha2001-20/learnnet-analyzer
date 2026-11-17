"""Text clustering of videos with K-means; k chosen by silhouette score."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


@dataclass
class ClusterResult:
    labels: np.ndarray
    k: int
    silhouette: float
    centroids: np.ndarray
    coords_2d: np.ndarray
    silhouette_by_k: dict[int, float]


def cluster_embeddings(emb: np.ndarray, k_min: int = 2, k_max: int = 10, seed: int = 0) -> ClusterResult:
    n = len(emb)
    if n < 3:
        raise ValueError("Need at least 3 videos to cluster.")
    k_max = min(k_max, n - 1)
    scores, models = {}, {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(emb)
        if len(set(km.labels_)) < 2:
            continue
        scores[k] = float(silhouette_score(emb, km.labels_, metric="cosine"))
        models[k] = km
    best = max(scores, key=scores.get)
    coords = PCA(n_components=2, random_state=seed).fit_transform(emb) if emb.shape[1] >= 2 else np.c_[emb, np.zeros(n)]
    return ClusterResult(models[best].labels_, best, scores[best], models[best].cluster_centers_, coords, scores)
