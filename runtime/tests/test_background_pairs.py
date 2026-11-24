"""The all-pairs scorer must count background records as eligible negatives."""

import numpy as np

from eval.background_pairs import pair_counts


def test_cross_source_background_is_negative_not_truth():
    rows = [
        {"source": "kubernetes", "incident_id": "one"},
        {"source": "sentry", "incident_id": "one"},
        {"source": "cloudwatch", "incident_id": None},
        {"source": "grafana", "incident_id": None},
    ]
    membership = np.zeros((4, 4), dtype=bool)
    for i, j in [(0, 1), (2, 3)]:
        membership[i, j] = membership[j, i] = True
    got = pair_counts(rows, membership)
    assert {k: got[k] for k in ("tp", "fp", "fn", "tn")} == {
        "tp": 1,
        "fp": 1,
        "fn": 0,
        "tn": 4,
    }
    assert got["false_positive_association_rate"] == 0.2
