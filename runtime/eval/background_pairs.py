"""Background-inclusive scorer extracted from eval/prevalence_correlation.py."""

import numpy as np


def pair_counts(rows, predictions):
    source = np.array([r["source"] for r in rows])
    ids = np.array([r.get("incident_id") for r in rows], dtype=object)
    upper = np.triu(source[:, None] != source[None, :], 1)
    truth = (ids[:, None] == ids[None, :]) & np.array([bool(k) for k in ids])[:, None]
    tp = int(np.sum(upper & truth & predictions))
    fp = int(np.sum(upper & ~truth & predictions))
    fn = int(np.sum(upper & truth & ~predictions))
    tn = int(np.sum(upper & ~truth & ~predictions))
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        false_positive_association_rate=fp / (fp + tn) if fp + tn else 0.0,
    )
