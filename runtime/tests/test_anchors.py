from datetime import datetime, timedelta, timezone
import numpy as np
from mozaic.correlation.anchors import anchor_membership


class HiddenID:
    def __init__(self, seconds, record_id):
        self.timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(
            seconds=seconds
        )
        self.record_id = record_id

    @property
    def correlation_id(self):
        raise AssertionError("ID-free matcher inspected a correlation ID")


def test_id_free_anchor_never_reads_identifiers_and_obeys_anchor_window():
    records = [
        HiddenID(0, "a"),
        HiddenID(100, "b"),
        HiddenID(120, "c"),
        HiddenID(450, "d"),
    ]
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    matrix = anchor_membership(records, vectors, 0.7)
    assert matrix[0, 1] and not matrix[0, 2] and not matrix[0, 3]
    assert matrix.sum() == 2


def test_time_control_reads_neither_ids_nor_embedding_values():
    records = [
        HiddenID(0, "a"),
        HiddenID(100, "b"),
        HiddenID(120, "c"),
        HiddenID(450, "d"),
    ]
    matrix = anchor_membership(records, None, 0.7, mode="time")
    assert matrix.sum() == 6 and not matrix[0, 3]
