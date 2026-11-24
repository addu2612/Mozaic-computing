"""Exercise source ingestion and association using new, invented records."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from eval.common import group
from eval.round8_correlation import gated_membership
from mcp_servers.server import CLASSES
from mozaic.schema import SOURCES, project


def main():
    start = "2026-01-01T00:00:00Z"
    end = "2026-01-01T00:02:00Z"
    rows = []
    with TemporaryDirectory() as tmp:
        for i, source in enumerate(SOURCES):
            raw = {
                "source": source,
                "record_id": f"toy-{source}",
                "timestamp": f"2026-01-01T00:00:0{i}Z",
                "service": "example-service",
                "namespace": "example",
                "payload": {"message": "Example exception", "MetricName": "example_metric", "Value": float(i)},
                "correlation_id": "corr_toy" if i < 2 else None,
                "incident_id": "toy_incident" if i < 2 else None,
                "is_anomaly": i < 2,
                "anomaly_type": "invented_example",
            }
            Path(tmp, f"{source}.jsonl").write_text(json.dumps(raw) + "\n")
            result = CLASSES[source](tmp).query(start, end)
            assert result["ok"] and result["total"] == 1, result
            public = result["records"][0]
            assert not {"is_anomaly", "anomaly_type", "incident_id"} & set(public)
            rows.append(project(public))
            assert group(raw) == ("toy_incident" if i < 2 else "normal_2026-01-01T00:0")
    # Deliberately identical vectors: negative controls stay separate because
    # normal records without IDs must not enter the semantic fallback.
    vectors = np.ones((len(rows), 1))
    predictions = gated_membership(rows, vectors, [True, True, False, False])
    grouped = np.argwhere(np.triu(predictions, 1)).tolist()
    assert grouped == [[0, 1]], grouped
    print(json.dumps({"result": "passed", "sample": "illustrative records only", "sources": list(SOURCES), "linked_sources": [list(SOURCES[:2])]}))


if __name__ == "__main__":
    main()
