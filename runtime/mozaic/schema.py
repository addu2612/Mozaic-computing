"""Deny-by-default projection from source payload to canonical model input."""

from dataclasses import dataclass
from datetime import datetime
import json
import math
import re
from typing import Any

SOURCES = ("kubernetes", "sentry", "cloudwatch", "grafana")
PUBLIC_SOURCES = ("rca_logs", "rca_metrics", "rca_traces")
SCENARIOS = ("memory_leak", "deployment_failure", "database_connection_pool_exhaustion")


@dataclass(frozen=True)
class TelemetryRecord:
    record_id: str
    timestamp: datetime
    source: str
    service: str
    namespace: str
    message: str
    metric: str = ""
    value: float | None = None
    query: str = ""
    correlation_id: str | None = None


def _message(payload):
    value = payload.get("message", payload.get("log", ""))
    if isinstance(value, dict):
        return str(value.get("formatted", value.get("message", "")))
    value = str(value)
    if value.startswith("{"):
        try:
            return str(json.loads(value).get("message", ""))
        except json.JSONDecodeError:
            pass
    match = re.search(r'msg="([^"]*)"', value)
    if match:
        return match[1]
    return re.sub(
        r"^\d{4}-\d\d-\d\d[T ][\d:.,]+(?:Z|\+00:00)?\s+(?:INFO|DEBUG|ERROR|WARNING|CRITICAL|FATAL)\s*(?:\[[^]]*\])?\s*",
        "",
        value,
    )


def project(raw: dict[str, Any]) -> TelemetryRecord:
    if raw.get("source") not in SOURCES and raw.get("source") not in (
        "hdfs",
        "bgl",
        *PUBLIC_SOURCES,
    ):
        raise ValueError("unsupported telemetry source")
    stamp = datetime.fromisoformat(raw["timestamp"].replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("timestamp must specify timezone")
    if not isinstance(raw.get("record_id"), str) or not raw["record_id"]:
        raise ValueError("record_id must be a nonempty string")
    p = raw.get("payload", {})
    if not isinstance(p, dict):
        raise ValueError("payload must be an object")
    metric = str(p.get("MetricName", p.get("panel", {}).get("title", "")))
    value = p.get("Value")
    if p.get("datapoints"):
        value = p["datapoints"][0][0]
    if value is not None and not math.isfinite(float(value)):
        raise ValueError("metric value must be finite")
    msg = _message(p)
    # Annotations embed scenario names and IDs directly, so omit annotation text entirely.
    if p.get("type") == "annotation":
        msg = ""
    for name in SCENARIOS:
        msg = msg.replace(name, "scenario")
    msg = re.sub(r"\b(?:inc|corr|req)_[\w-]+\b", "<id>", msg)
    return TelemetryRecord(
        str(raw["record_id"]),
        stamp,
        str(raw["source"]),
        str(raw.get("service", "")),
        str(raw.get("namespace", "")),
        msg,
        metric,
        float(value) if value is not None else None,
        str(p.get("target", {}).get("expr", "")),
        raw.get("correlation_id"),
    )
