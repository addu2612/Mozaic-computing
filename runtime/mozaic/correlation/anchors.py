"""Causal incident anchors, with a correlation-ID-free semantic fallback."""

import numpy as np

WINDOW_SECONDS = 300


def anchor_membership(
    records,
    vectors,
    threshold,
    mode="embedding",
    tie_rule="cosine_then_time",
    window=WINDOW_SECONDS,
):
    if mode not in ("embedding", "hybrid", "id_time", "time"):
        raise ValueError("Unknown anchor method")
    order = sorted(
        range(len(records)), key=lambda i: (records[i].timestamp, records[i].record_id)
    )
    rank = {i: j for j, i in enumerate(order)}
    labels = np.full(len(records), -1, dtype=int)
    anchors = []
    for i in order:
        eligible = [
            a
            for a in anchors
            if abs((records[i].timestamp - records[a].timestamp).total_seconds())
            <= window
        ]
        # The embedding and time branches do not inspect identifiers, even for tie breaking.
        if mode in ("hybrid", "id_time"):
            identifier = records[i].correlation_id
            matching = [
                a
                for a in eligible
                if identifier and identifier == records[a].correlation_id
            ]
        else:
            matching = []
        if mode == "id_time" or (mode == "hybrid" and records[i].correlation_id):
            candidates = matching
        elif mode == "time":
            candidates = eligible
        else:
            candidates = [
                a for a in eligible if float(vectors[i] @ vectors[a]) >= threshold
            ]
        if candidates:

            def ranking(a):
                time_distance = abs(
                    (records[i].timestamp - records[a].timestamp).total_seconds()
                )
                if mode in ("id_time", "time") or (
                    mode == "hybrid" and records[i].correlation_id
                ):
                    return (-time_distance, -rank[a])
                cosine = float(vectors[i] @ vectors[a])
                return (
                    (-time_distance, cosine, -rank[a])
                    if tie_rule == "time_then_cosine"
                    else (cosine, -time_distance, -rank[a])
                )

            labels[i] = labels[max(candidates, key=ranking)]
        else:
            labels[i] = i
            anchors.append(i)
    membership = labels[:, None] == labels[None, :]
    np.fill_diagonal(membership, False)
    return membership
