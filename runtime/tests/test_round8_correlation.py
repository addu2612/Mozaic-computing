from dataclasses import replace
from datetime import datetime, timezone
import numpy as np
from mozaic.schema import TelemetryRecord
from eval.round8_correlation import gated_membership


def records():
    r = TelemetryRecord(
        "a", datetime(2026, 1, 1, tzinfo=timezone.utc), "cloudwatch", "svc", "", ""
    )
    return [r, replace(r, record_id="b", source="grafana")]


def test_fallback_requires_both_detector_flags():
    x = np.array([[1.0, 0.0], [1.0, 0.0]])
    for flags in ([False, False], [True, False], [False, True]):
        assert not gated_membership(records(), x, flags).any()
    assert gated_membership(records(), x, [True, True])[0, 1]


def test_identifier_link_survives_gate_and_conflicting_ids_do_not_link():
    rows = [replace(r, correlation_id="shared") for r in records()]
    x = np.eye(2)
    assert gated_membership(rows, x, [False, False])[0, 1]
    rows[1] = replace(rows[1], correlation_id="different")
    assert not gated_membership(rows, np.ones((2, 2)), [True, True]).any()
