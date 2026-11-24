from eval.round8_baseline import quiet_baseline
from eval.common import partition


def row(i, anomaly=False, source="cloudwatch"):
    return dict(
        record_id=str(i),
        timestamp=f"2026-01-01T{i // 60:02d}:{i % 60:02d}:00Z",
        source=source,
        is_anomaly=anomaly,
        incident_id="incident" if anomaly else None,
    )


def test_global_quiet_boundary_and_no_late_normal_selection():
    rows = [row(i) for i in range(180)] + [row(89, True, "sentry")]
    history, selected, audit = quiet_baseline(rows, "cloudwatch")
    assert not audit["fallback"]
    assert audit["interval_records"] == 89
    assert all(int(r["record_id"]) < 89 and partition(r) == "fit" for r in history)
    assert selected and set(r["record_id"] for r in selected) <= set(
        r["record_id"] for r in history
    )


def test_no_prefix_uses_declared_normal_only_fallback():
    rows = [row(0, True)] + [row(i) for i in range(1, 180)]
    history, selected, audit = quiet_baseline(rows, "cloudwatch")
    assert audit["fallback"]
    assert all(not r["is_anomaly"] and partition(r) == "fit" for r in history)
    assert selected


def test_quiet_selection_is_order_invariant_and_capped():
    rows = [row(i) for i in range(180)]
    first = quiet_baseline(rows, "cloudwatch", cap=20)
    second = quiet_baseline(rows[::-1], "cloudwatch", cap=20)
    assert first == second
    assert len(first[1]) == 20
