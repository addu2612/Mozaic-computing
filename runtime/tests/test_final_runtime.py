"""The live profile learns only from an explicitly declared observed warm-up."""

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from eval.common import partition
from eval.round8_clustering import component_vectors
from eval.round8_correlation import gated_membership
from mozaic.schema import SOURCES, project
from orchestrator.final_runtime import FinalRuntimeAnalyzer, warmup_partition
from orchestrator.profile_runtime import create_analyzer


class Encoder:
    def encode(self, texts):
        x = np.array(
            [list(hashlib.sha256(t.encode()).digest()[:8]) for t in texts], dtype=float
        )
        return x / np.linalg.norm(x, axis=1)[:, None]


def warmup():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        dict(
            record_id=f"{source}-{i}",
            source=source,
            timestamp=(start + timedelta(minutes=i)).isoformat(),
            service="service",
            payload={"message": f"quiet template {chr(65 + i % 26)}"},
        )
        for source in SOURCES
        for i in range(120)
    ]


def analyzer(tmp_path, rows=None, **kwargs):
    return FinalRuntimeAnalyzer(
        warmup=warmup() if rows is None else rows,
        warmup_start="2026-01-01T00:00:00Z",
        warmup_end="2026-01-01T02:00:00Z",
        encoder=Encoder(),
        cache=tmp_path,
        **kwargs,
    )


def test_warmup_partitions_match_frozen_normal_rule_and_ignore_truth():
    for row in warmup():
        assert warmup_partition(project(row)) == partition(row)
        annotated = dict(
            row, incident_id="hidden", is_anomaly=True, anomaly_type="memory_leak"
        )
        assert warmup_partition(project(annotated)) == partition(row)
    row = warmup()[0]
    shifted = dict(row, timestamp="2026-01-01T05:30:00+05:30")
    assert warmup_partition(project(row)) == warmup_partition(project(shifted))


def test_cache_identity_ignores_hidden_truth_but_binds_observations(tmp_path):
    first = analyzer(tmp_path)
    assert not first.cache_metadata["hit"]
    annotated = [
        dict(r, incident_id="hidden", is_anomaly=True, anomaly_type="memory_leak")
        for r in warmup()
    ]
    second = analyzer(tmp_path, annotated)
    assert second.cache_metadata["hit"]
    assert second.cache_metadata["key"] == first.cache_metadata["key"]
    assert second.cohorts == first.cohorts
    changed = copy.deepcopy(annotated)
    changed[0]["payload"]["message"] = "different observed message"
    third = analyzer(tmp_path, changed)
    assert third.cache_metadata["key"] != first.cache_metadata["key"]
    assert not third.cache_metadata["hit"]


def test_warmup_and_analysis_boundaries_are_enforced(tmp_path):
    with pytest.raises(ValueError, match="four sources"):
        analyzer(tmp_path, [r for r in warmup() if r["source"] != "sentry"])
    with pytest.raises(ValueError, match="calibration rows"):
        analyzer(
            tmp_path, [r for r in warmup() if warmup_partition(project(r)) == "fit"]
        )
    with pytest.raises(ValueError, match="unique"):
        analyzer(tmp_path, warmup() + [warmup()[0]])
    instance = analyzer(tmp_path)
    with pytest.raises(ValueError, match="precede"):
        instance.analyze(warmup()[:1])
    row = dict(warmup()[0], timestamp="2026-01-01T02:00:00Z")
    with pytest.raises(ValueError, match="Duplicate"):
        instance.analyze([row, row])


def test_trace_is_observational_and_requires_both_endpoints():
    raw = [dict(warmup()[0], record_id=str(i), source=SOURCES[i]) for i in range(4)]
    raw[0]["correlation_id"] = raw[1]["correlation_id"] = "shared"
    records = [project(r) for r in raw]
    vectors = np.tile([1.0, 0.0], (4, 1))
    flags = [False, False, True, True]
    ordinary = gated_membership(records, vectors, flags)
    traced, trace = gated_membership(records, vectors, flags, return_trace=True)
    assert np.array_equal(ordinary, traced)
    assert {t["reason"] for t in trace} == {"identifier", "detector_gated_fallback"}
    for edge in trace:
        if edge["reason"] == "detector_gated_fallback":
            assert edge["record_flag"] and edge["anchor_flag"]
    means, groups = component_vectors(vectors, traced)
    assert sorted(groups) == [[0, 1], [2, 3]]
    np.testing.assert_allclose(means, [[1.0, 0.0], [1.0, 0.0]])


def test_profile_selection_preserves_original_default(monkeypatch):
    import orchestrator.runtime

    marker = object()
    monkeypatch.setattr(
        orchestrator.runtime, "RuntimeAnalyzer", lambda **kwargs: marker
    )
    assert create_analyzer() is marker
    assert create_analyzer("original") is marker
    with pytest.raises(ValueError, match="Unknown"):
        create_analyzer("unfrozen")


def test_labels_cannot_change_detection_or_groups(tmp_path):
    called = []

    def label(incident):
        called.append(incident["candidate_id"])
        return {"label": "candidate", "incident_type": "unknown", "summary": "test"}

    instance = analyzer(tmp_path)
    rows = [
        dict(
            warmup()[0],
            timestamp="2026-01-01T02:00:00Z",
            source=s,
            record_id="query-" + s,
            correlation_id="shared",
        )
        for s in SOURCES
    ]
    ordinary = instance.analyze(rows)
    instance.llm_labeler = label
    labeled = instance.analyze(rows)
    assert len(called) == len(ordinary["incidents"]) == 1
    for key in (
        "records",
        "groups",
        "clustering",
        "detected_count",
        "correlation_trace",
    ):
        assert labeled[key] == ordinary[key]
    assert ordinary["incidents"][0]["labeling"]["llm_enabled"] is False
    assert labeled["incidents"][0]["labeling"]["status"] == "accepted"


def test_actual_frozen_detector_parameters_are_applied(tmp_path):
    instance = analyzer(tmp_path)
    config = json.loads(Path("frozen_final_profile.json").read_text())
    for detector in instance.detectors.values():
        assert detector.model.n_clusters == config["detector_centers"]
        assert detector.model.n_init == config["detector_n_init"]
        assert detector.model.random_state == config["sampling_seed"]


def test_local_labeling_is_schema_checked_single_attempt_and_local_only():
    import httpx
    from orchestrator.local_llm import LocalLabeler

    calls = []

    def invalid(request):
        calls.append(request)
        return httpx.Response(200, json={"response": '{"label":"incomplete"}'})

    labeler = LocalLabeler(
        url="http://localhost:11435", transport=httpx.MockTransport(invalid)
    )
    with pytest.raises(ValueError):
        labeler({"evidence": []})
    assert len(calls) == 1
    with pytest.raises(ValueError, match="local Ollama"):
        LocalLabeler(url="https://api.example.com")
