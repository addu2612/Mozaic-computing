"""Dev-only time-based quiet baseline selection for round eight."""

from datetime import datetime
from eval.common import partition, sample


def timestamp(row):
    return datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))


def quiet_baseline(rows, source, cap=2000, seed=1411):
    """Use the global prefix before the first incident, without scattered filtering.

    Labels certify the dev interval boundary only. Deployment requires an operator
    certified quiet period. Calibration and validation rows inside that interval
    are excluded from the fitting targets to preserve the existing split.
    """
    ordered = sorted(rows, key=lambda r: (timestamp(r), r["record_id"]))
    if not ordered:
        raise ValueError("Cannot select a baseline from empty dev data")
    first_incident = next((timestamp(r) for r in ordered if r["is_anomaly"]), None)
    quiet = [
        r for r in ordered if first_incident is None or timestamp(r) < first_incident
    ]
    source_quiet = [r for r in quiet if r["source"] == source]
    fallback = not source_quiet
    if fallback:
        history = [
            r
            for r in ordered
            if r["source"] == source and partition(r) == "fit" and not r["is_anomaly"]
        ]
    else:
        history = [r for r in source_quiet if partition(r) == "fit"]
    if not history:
        raise ValueError("Quiet interval contains no fit-partition records")
    selected = sample(history, cap, seed)
    return (
        history,
        selected,
        dict(
            source=source,
            design="normal-only fallback"
            if fallback
            else "known-quiet baseline period",
            fallback=fallback,
            interval_start=ordered[0]["timestamp"],
            interval_end_exclusive=first_incident.isoformat()
            if first_incident
            else None,
            interval_records=len(source_quiet),
            fit_history_records=len(history),
            fit_records=len(selected),
            record_ids=[r["record_id"] for r in selected],
            deployment_assumption="operator-certified quiet baseline period",
        ),
    )
