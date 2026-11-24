from dataclasses import replace
from datetime import timedelta
import numpy as np
import pytest
from mozaic.schema import project
from mozaic.pipeline import contexts, text
from mozaic.correlation import associate
from eval.common import group, partition, clustering_metrics
from eval.pair_metrics import pair_metrics


def raw(**kwargs):
    return dict(
        record_id="one",
        timestamp="2026-01-01T00:00:00Z",
        source="sentry",
        service="api",
        namespace="production",
        payload={"message": {"formatted": "database pool exhausted"}},
        **kwargs,
    )


def test_forbidden_fields_do_not_reach_features():
    a = raw(
        is_anomaly=True, incident_id="secret", anomaly_type="foo", correlation_id="abc"
    )
    b = {
        **a,
        "incident_id": "different",
        "is_anomaly": False,
        "anomaly_type": "bar",
        "correlation_id": "def",
    }
    b["payload"] = {
        **b["payload"],
        "level": "fatal",
        "extra": {"incident_id": "hidden"},
        "tags": {"anomaly_type": "nested"},
    }
    assert text(project(a)) == text(project(b)) == "database pool exhausted"


def test_context_is_causal_service_local_and_partition_local():
    a = project(raw())
    b = replace(
        a,
        record_id="two",
        timestamp=a.timestamp + timedelta(seconds=1),
        message="later",
    )
    other = replace(a, service="other", message="unrelated")
    values = contexts([b, other, a], 5)
    assert "later" not in values[2] and "unrelated" not in values[0]
    assert values[0] == "later [context] database pool exhausted"
    assert contexts([b], 5) == ["later"]


def test_incident_split_is_source_independent():
    a = raw(incident_id="same")
    b = {**a, "source": "grafana", "timestamp": "2026-01-01T07:12:00Z"}
    assert group(a) == group(b) and partition(a) == partition(b)


def test_sampled_context_uses_full_fold_once_and_preserves_alignment(monkeypatch):
    import mozaic.pipeline as pipeline

    rows = [
        {**raw(), "record_id": str(i), "timestamp": f"2026-01-01T00:00:{i:02d}Z"}
        for i in range(30)
    ]
    selected = [rows[20], rows[5], rows[20]]
    expected = contexts([project(r) for r in rows], 20)
    calls = []
    original = pipeline.contexts

    def counted(records, window, **kwargs):
        calls.append(len(records))
        return original(records, window, **kwargs)

    monkeypatch.setattr(pipeline, "contexts", counted)
    assert pipeline.context_texts(rows, 20, selected) == [
        expected[i] for i in (20, 5, 20)
    ]
    assert calls == [30]
    assert pipeline.context_texts([rows[20]], 20) != [expected[20]]
    with pytest.raises(ValueError, match="duplicate"):
        pipeline.context_texts([rows[0], rows[0]], 20)
    with pytest.raises(KeyError):
        pipeline.context_texts(rows, 20, [{**raw(), "record_id": "outside-fold"}])


def test_no_null_id_matches_and_temporal_distractor():
    a = project(raw())
    b = replace(a, source="kubernetes", service="other", correlation_id=None)
    assert not associate([a, b], np.array([[1.0, 0], [1.0, 0]]))[0, 1]
    b = replace(b, service=a.service)
    assert associate([a, b], np.array([[1.0, 0], [1.0, 0]]))[0, 1]
    b = replace(b, timestamp=a.timestamp + timedelta(seconds=301))
    assert not associate([a, b], np.array([[1.0, 0], [1.0, 0]]))[0, 1]


def test_pair_metrics_include_false_positives_and_exact_incidents():
    rows = [
        dict(source="a", incident_id="x"),
        dict(source="b", incident_id="x"),
        dict(source="c", incident_id="y"),
    ]
    pred = np.ones((3, 3), dtype=bool)
    m = pair_metrics(rows, pred)
    assert m["tp"] == 1 and m["fp"] == 2 and m["precision"] == pytest.approx(1 / 3)
    assert m["incident_recall"] == 0 and m["false_positive_association_rate"] == 1


def test_undefined_clustering_is_explicit():
    m = clustering_metrics(["a", "b", "b"], [-1, -1, -1], np.eye(3))
    assert m["silhouette"] is None and m["noise_fraction"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"record_id": ""},
        {"timestamp": "2026-01-01T00:00:00"},
        {"source": "unknown"},
        {"payload": []},
        {"payload": {"Value": float("nan")}},
    ],
)
def test_malformed_telemetry_rejected(change):
    with pytest.raises(ValueError):
        project({**raw(), **change})


def test_embedding_cache_is_safe_for_parallel_writers(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from mozaic.pipeline import Embedder

    class Model:
        max_seq_length = 256

        def encode(self, values, **kwargs):
            return np.array([[len(value), 1.0, 2.0] for value in values])

    enc = Embedder.__new__(Embedder)
    enc.model = Model()
    enc.cache = tmp_path
    enc.normalize_embeddings = True
    enc.libraries = {"test": "version"}
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(
            pool.map(lambda _: enc.encode(["same", "other", "same"]), range(20))
        )
    assert all(
        np.array_equal(output, np.array([[4, 1, 2], [5, 1, 2], [4, 1, 2]]))
        for output in outputs
    )
    assert len(list(tmp_path.glob("*.npy"))) == 2 and not list(tmp_path.glob("*.tmp"))


def test_embedding_cache_tracks_effective_encode_settings(tmp_path):
    from mozaic.pipeline import Embedder

    calls = []

    class Model:
        max_seq_length = 256

        def encode(self, values, *, normalize_embeddings, **kwargs):
            calls.append((normalize_embeddings, self.max_seq_length))
            return np.array(
                [[float(normalize_embeddings), self.max_seq_length] for _ in values]
            )

    enc = Embedder.__new__(Embedder)
    enc.model, enc.cache = Model(), tmp_path
    enc.normalize_embeddings = True
    enc.libraries = {"tokenizers": "1"}
    assert enc.encode(["same"])[0].tolist() == [1, 256]
    enc.encode(["same"])
    assert len(calls) == 1
    enc.normalize_embeddings = False
    assert enc.encode(["same"])[0].tolist() == [0, 256]
    enc.model.max_seq_length = 128
    assert enc.encode(["same"])[0].tolist() == [0, 128]
    enc.libraries["tokenizers"] = "2"
    enc.encode(["same"])
    assert calls == [(True, 256), (False, 256), (False, 128), (False, 128)]


@pytest.mark.parametrize(
    "ids,shared,coverage,consistent",
    [
        ([None, None], False, 0.0, 0.0),
        (["a", None, "a"], False, 2 / 3, 1.0),
        (["a", "a"], True, 1.0, 1.0),
        (["a", "b"], False, 1.0, 0.5),
        ([], False, 0.0, 0.0),
    ],
)
def test_identifier_confidence_requires_every_record(ids, shared, coverage, consistent):
    from mozaic.correlation import identifier_evidence

    base = project(raw())
    result = identifier_evidence([replace(base, correlation_id=value) for value in ids])
    assert result["shared_identifier"] is shared
    assert (
        result["identifier_coverage"] == coverage
        and result["identifier_consistency"] == consistent
    )
    assert result["records_with_identifier"] + result[
        "records_without_identifier"
    ] == len(ids)
