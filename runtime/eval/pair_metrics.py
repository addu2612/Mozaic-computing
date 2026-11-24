"""Cross-source pair and exact-incident association metrics.

Scoring lives outside eval/common.py so that metric definitions are not part of
the runtime dependency inventory that binds saved system measurements.
"""

import numpy as np


def pair_metrics(rows, pred):
    from mozaic.correlation import components

    upper = [
        (i, j)
        for i in range(len(rows))
        for j in range(i + 1, len(rows))
        if rows[i]["source"] != rows[j]["source"]
    ]
    # Background records have an empty or missing incident ID; they never
    # form a true incident and never make a pair a true association.
    incident = [r["incident_id"] or "" for r in rows]
    truth = np.array(
        [bool(incident[i] and incident[i] == incident[j]) for i, j in upper]
    )
    guess = np.array([pred[i, j] for i, j in upper])
    tp = int(np.sum(truth & guess))
    fp = int(np.sum(~truth & guess))
    fn = int(np.sum(truth & ~guess))
    tn = int(np.sum(~truth & ~guess))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    true_groups = {
        frozenset(i for i, value in enumerate(incident) if value == g)
        for g in set(incident) - {""}
    }
    true_groups = {g for g in true_groups if len({rows[i]["source"] for i in g}) > 1}
    groups = [g for g in components(pred) if len({rows[i]["source"] for i in g}) > 1]
    exact = sum(frozenset(g) in true_groups for g in groups)
    ip = exact / len(groups) if groups else 0.0
    ir = exact / len(true_groups) if true_groups else 0.0
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        false_positive_association_rate=fp / (fp + tn) if fp + tn else 0.0,
        incident_precision=ip,
        incident_recall=ir,
        incident_f1=2 * ip * ir / (ip + ir) if ip + ir else 0.0,
        # Impure: members do not all share one real incident, which includes
        # components that are only background or mix in background records.
        impure_component_fraction=sum(
            len({incident[i] for i in g}) > 1 or not incident[next(iter(g))]
            for g in groups
        )
        / len(groups)
        if groups
        else 0.0,
        predicted_incidents=len(groups),
        true_incidents=len(true_groups),
    )
