"""Detector-gated causal fallback, preserving identifier links."""

import numpy as np


def gated_membership(
    records,
    vectors,
    anomalous,
    threshold=0.4,
    entities=None,
    window=300,
    return_trace=False,
):
    if len(records) != len(vectors) or len(records) != len(anomalous):
        raise ValueError("Rows, vectors and detector flags must align")
    order = sorted(
        range(len(records)), key=lambda i: (records[i].timestamp, records[i].record_id)
    )
    rank = {i: j for j, i in enumerate(order)}
    labels = np.arange(len(records))
    anchors = []
    trace = []
    for i in order:
        eligible = [
            a
            for a in anchors
            if abs((records[i].timestamp - records[a].timestamp).total_seconds())
            <= window
        ]
        identifier = records[i].correlation_id
        matching = [
            a
            for a in eligible
            if identifier and identifier == records[a].correlation_id
        ]
        if matching:
            candidates = matching
        elif identifier or not anomalous[i]:
            candidates = []
        else:
            candidates = [
                a
                for a in eligible
                if anomalous[a]
                and (
                    float(vectors[i] @ vectors[a]) >= threshold
                    or (
                        entities is not None
                        and bool(set(entities[i]) & set(entities[a]))
                    )
                )
            ]
        if candidates:
            chosen = max(
                candidates,
                key=lambda a: (
                    -abs((records[i].timestamp - records[a].timestamp).total_seconds()),
                    float(vectors[i] @ vectors[a]) if not matching else 0.0,
                    -rank[a],
                ),
            )
            labels[i] = labels[chosen]
            if return_trace:
                trace.append(
                    dict(
                        record=i,
                        anchor=chosen,
                        reason="identifier" if matching else "detector_gated_fallback",
                        record_flag=bool(anomalous[i]),
                        anchor_flag=bool(anomalous[chosen]),
                        seconds=abs(
                            (
                                records[i].timestamp - records[chosen].timestamp
                            ).total_seconds()
                        ),
                        cosine=float(vectors[i] @ vectors[chosen]),
                    )
                )
        else:
            anchors.append(i)
    matrix = labels[:, None] == labels[None, :]
    np.fill_diagonal(matrix, False)
    return (matrix, trace) if return_trace else matrix
