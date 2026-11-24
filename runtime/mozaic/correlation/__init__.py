"""Association requires an explicit identifier or time, service and semantic evidence."""

import numpy as np


def associate(records, vectors, strategy="combined", window=300, similarity=0.65):
    n = len(records)
    pred = np.zeros((n, n), dtype=bool)
    scores = vectors @ vectors.T
    for i, a in enumerate(records):
        for j in range(i + 1, n):
            b = records[j]
            if a.source == b.source:
                continue
            by_id = bool(a.correlation_id and a.correlation_id == b.correlation_id)
            temporal = abs((a.timestamp - b.timestamp).total_seconds()) <= window
            inferred = (
                temporal and a.service == b.service and scores[i, j] >= similarity
            )
            ok = (
                by_id
                if strategy == "id"
                else temporal
                if strategy == "temporal"
                else inferred
                if strategy == "semantic"
                else by_id or inferred
            )
            pred[i, j] = pred[j, i] = ok
    return pred


def components(matrix):
    unseen = set(range(len(matrix)))
    groups = []
    while unseen:
        root = unseen.pop()
        group = {root}
        stack = [root]
        while stack:
            new = set(np.flatnonzero(matrix[stack.pop()])) & unseen
            unseen -= new
            group |= new
            stack.extend(new)
        groups.append(group)
    return groups


def identifier_evidence(records):
    """Only call an ID shared when every evidence record supplies that same ID."""
    from collections import Counter

    ids = [r.correlation_id for r in records if r.correlation_id]
    counts = Counter(ids)
    total = len(records)
    return dict(
        shared_identifier=bool(total) and len(ids) == total and len(counts) == 1,
        identifier_count=len(counts),
        records_with_identifier=len(ids),
        records_without_identifier=total - len(ids),
        identifier_coverage=len(ids) / total if total else 0.0,
        identifier_consistency=max(counts.values()) / len(ids) if ids else 0.0,
    )
