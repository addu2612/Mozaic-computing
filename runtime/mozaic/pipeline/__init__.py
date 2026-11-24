"""Seven stages: validation, normalization, context, embedding, clustering, labeling, reporting."""

import hashlib
import os
import threading
from pathlib import Path
import re
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from mozaic.schema import project
from mozaic.cache import digest, embedding_identity, embedding_libraries

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Immutable published MiniLM snapshot; download and hash model files for each run.
REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


def normalize(text):
    text = text.lower()
    text = re.sub(r"blk_-?\d+", "<block>", text)
    text = re.sub(r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", "<id>", text)
    text = re.sub(r"\b[0-9a-f]{12,}\b", "<id>", text)
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<ip>", text)
    text = re.sub(r"\b\d{4}-\d\d-\d\d[t ][\d:.,]+", "<time>", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", text)
    return re.sub(r"\s+", " ", text).strip()


def text(record, include_query=False, message=True, metric=True):
    parts = [normalize(record.message)] if message else []
    if metric and record.metric:
        # Retain numeric measurement magnitude in text, normalized only in message channel.
        parts.extend(
            [
                record.metric.lower(),
                f"value {record.value:.3g}" if record.value is not None else "",
            ]
        )
    if include_query:
        parts.append(normalize(record.query))
    return " ".join(p for p in parts if p) or "telemetry"


def contexts(records, window, **kwargs):
    """Call separately for each already-created partition, never across a fold boundary."""
    result = [""] * len(records)
    history = {}
    for i in sorted(
        range(len(records)), key=lambda j: (records[j].timestamp, records[j].record_id)
    ):
        r = records[i]
        key = (r.source, r.service)
        current = text(r, **kwargs)
        past = history.setdefault(key, [])
        result[i] = (
            " [context] ".join([current] + list(reversed(past[-window:])))
            if window
            else current
        )
        past.append(current)
    return result


def context_texts(complete, window, selected=None, **kwargs):
    """Build context once on a complete fold, then align the selected records."""
    identifiers = [r["record_id"] for r in complete]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Context partition contains duplicate record IDs")
    strings = contexts([project(r) for r in complete], window, **kwargs)
    if selected is None:
        return strings
    lookup = dict(zip(identifiers, strings))
    return [lookup[r["record_id"]] for r in selected]


class Embedder:
    def __init__(self, cache="data/embeddings"):
        from sentence_transformers import SentenceTransformer
        import torch

        torch.set_num_threads(2)
        self.model = SentenceTransformer(MODEL, revision=REVISION, device="cpu")
        self.normalize_embeddings = True
        self.libraries = embedding_libraries()
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)

    def encode(self, texts):
        normalize = self.normalize_embeddings
        identity = digest(
            embedding_identity(
                MODEL,
                REVISION,
                normalize_embeddings=normalize,
                max_seq_length=self.model.max_seq_length,
                libraries=self.libraries,
            )
        )
        # Deduplicate repeated templates without changing row alignment.
        keys = [hashlib.sha256((identity + t).encode()).hexdigest() for t in texts]
        pending = {
            k: t for k, t in zip(keys, texts) if not (self.cache / f"{k}.npy").exists()
        }
        if pending:
            vectors = self.model.encode(
                list(pending.values()),
                batch_size=64,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )
            for k, v in zip(pending, vectors):
                temporary = (
                    self.cache / f".{k}.{os.getpid()}.{threading.get_ident()}.tmp"
                )
                with temporary.open("wb") as handle:
                    np.save(handle, v)
                os.replace(temporary, self.cache / f"{k}.npy")
        return np.stack([np.load(self.cache / f"{k}.npy") for k in keys])


def cluster(vectors):
    return HDBSCAN(min_cluster_size=10, min_samples=5, metric="euclidean").fit_predict(
        vectors
    )


def keywords(texts, labels):
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    names = vectorizer.get_feature_names_out()
    return {
        int(label): names[
            np.asarray(matrix[labels == label].mean(axis=0))
            .ravel()
            .argsort()[-8:][::-1]
        ].tolist()
        for label in sorted(set(labels))
        if label != -1
    }
