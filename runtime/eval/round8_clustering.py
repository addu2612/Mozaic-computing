"""Incident-level representations and dev-selected density clustering."""

import numpy as np
from sklearn.cluster import KMeans, HDBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from mozaic.correlation import components


def component_vectors(vectors, membership):
    groups = [sorted(g) for g in components(membership)]
    return np.asarray([vectors[g].mean(axis=0) for g in groups]), groups


def incident_vectors(rows, vectors, membership=None):
    if membership is None:
        ids = sorted({r["incident_id"] for r in rows if r.get("incident_id")})
        groups = [
            [i for i, r in enumerate(rows) if r.get("incident_id") == key]
            for key in ids
        ]
    else:
        x, groups = component_vectors(vectors, membership)
    if membership is None:
        x = np.asarray([vectors[g].mean(axis=0) for g in groups])
    # Representations and groups are fixed before scoring. Every member votes,
    # including normal background; absent a strict majority the group is mixed.
    from eval.incident_scoring import group_truth

    truth = group_truth(rows, groups)
    return x, truth, groups


def silhouette_kmeans(x):
    candidates = []
    for k in range(2, min(8, len(x) - 1) + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=1411).fit_predict(x)
        if 1 < len(set(labels)) < len(x):
            candidates.append((float(silhouette_score(x, labels)), -k, labels))
    return (
        max(candidates, key=lambda t: t[:2])[2]
        if candidates
        else np.zeros(len(x), dtype=int)
    )


def density(x, settings):
    from umap import UMAP

    if len(x) < 4:
        return np.full(len(x), -1)
    projected = UMAP(
        n_neighbors=min(settings["n_neighbors"], len(x) - 1),
        n_components=settings["n_components"],
        min_dist=0.0,
        metric="euclidean",
        init="random",
        random_state=1411,
        n_jobs=1,
    ).fit_transform(x)
    return HDBSCAN(
        min_cluster_size=settings["min_cluster_size"], min_samples=2
    ).fit_predict(projected)


def comparators(x, settings):
    return {
        "MOZAIC": HDBSCAN(min_cluster_size=10, min_samples=5).fit_predict(x),
        "UMAP-HDBSCAN": density(x, settings),
        "KMeans silhouette": silhouette_kmeans(x),
        "KMeans oracle-k": KMeans(
            n_clusters=3, n_init=10, random_state=1411
        ).fit_predict(x),
        "Agglomerative oracle-k": AgglomerativeClustering(
            n_clusters=3, linkage="average"
        ).fit_predict(x),
    }
