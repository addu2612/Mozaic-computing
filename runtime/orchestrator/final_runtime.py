"""Frozen live-test profile with label-free, operator-declared quiet warm-up."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time
import joblib
import numpy as np
from eval.checkpoints import atomic_model
from eval.round8_correlation import gated_membership
from eval.round8_clustering import component_vectors, silhouette_kmeans
from mozaic.cache import digest, environment, file_hash
from mozaic.correlation import identifier_evidence
from mozaic.detection import Detector
from mozaic.labeling import ClusterLabel
from mozaic.pipeline import Embedder, contexts, cluster, keywords, MODEL, REVISION
from mozaic.schema import project, SOURCES


def utc(value):
    stamp = (
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        if isinstance(value, str)
        else value
    )
    if stamp.tzinfo is None:
        raise ValueError("Warm-up and analysis timestamps require timezones")
    return stamp.astimezone(timezone.utc)


def warmup_partition(record):
    stamp = utc(record.timestamp).isoformat()
    bucket = "normal_" + stamp[:14] + str(int(stamp[14:16]) // 10)
    number = int(hashlib.sha256(bucket.encode()).hexdigest()[:8], 16) % 10
    return "fit" if number < 6 else "calibration" if number < 8 else "validation"


def warmup_records(raw, start, end):
    start, end = utc(start), utc(end)
    if start >= end:
        raise ValueError("Warm-up start must precede its end")
    records = sorted(
        (project(r) for r in raw), key=lambda r: (r.timestamp, r.record_id)
    )
    records = [r for r in records if start <= r.timestamp < end]
    if not records or len({r.record_id for r in records}) != len(records):
        raise ValueError("Warm-up must contain nonempty unique observed records")
    if set(r.source for r in records) != set(SOURCES):
        raise ValueError("Quiet warm-up must cover all four sources")
    return records


def load_warmup(directory):
    return [
        json.loads(line)
        for source in SOURCES
        for line in (Path(directory) / (source + ".jsonl")).open()
        if line.strip()
    ]


class FinalRuntimeAnalyzer:
    def __init__(
        self,
        *,
        warmup,
        warmup_start,
        warmup_end,
        cache="data/runtime_final",
        embedding_cache="data/embeddings",
        encoder=None,
        llm_labeler=None,
    ):
        before = time.perf_counter()
        self.config = json.loads(Path("frozen_final_profile.json").read_text())
        self.start, self.end = utc(warmup_start), utc(warmup_end)
        records = warmup_records(warmup, self.start, self.end)
        self.llm_labeler = llm_labeler
        self.embedder = (
            encoder if encoder is not None else Embedder(cache=embedding_cache)
        )
        source_files = [
            *Path("mozaic").rglob("*.py"),
            Path("orchestrator/final_runtime.py"),
            Path("eval/round8_correlation.py"),
            Path("eval/round8_clustering.py"),
            Path("eval/checkpoints.py"),
        ]
        observed = [
            dict(asdict(r), timestamp=utc(r.timestamp).isoformat()) for r in records
        ]
        identity = dict(
            profile=self.config,
            warmup_start=self.start.isoformat(),
            warmup_end=self.end.isoformat(),
            observed_sha256=digest(observed),
            source={str(p): file_hash(p) for p in sorted(source_files)},
            environment=environment(),
            model=MODEL,
            revision=REVISION,
            encoder_implementation=type(self.embedder).__module__
            + "."
            + type(self.embedder).__qualname__,
        )
        key = digest(identity)
        path = Path(cache) / (key + ".joblib")
        self.cache_metadata = dict(key=key, hit=False, identity=identity)
        self.cohorts = {}
        partitions = {}
        for source in SOURCES:
            partitions[source] = {}
            self.cohorts[source] = {}
            for split in ("fit", "calibration"):
                full = [
                    r
                    for r in records
                    if r.source == source and warmup_partition(r) == split
                ]
                if not full:
                    raise ValueError(
                        "Quiet warm-up lacks " + source + " " + split + " rows"
                    )
                chosen = random.Random(self.config["sampling_seed"]).sample(
                    range(len(full)), min(self.config["sample_cap"], len(full))
                )
                partitions[source][split] = (full, chosen)
                self.cohorts[source][split] = dict(
                    history_records=len(full),
                    record_ids=[full[i].record_id for i in chosen],
                )
        if path.exists():
            saved = joblib.load(path)
            if saved["identity"] != identity or set(saved["detectors"]) != set(SOURCES):
                raise ValueError(
                    "Final model cache identity or source inventory differs"
                )
            self.detectors = saved["detectors"]
            self.cache_metadata["hit"] = True
        else:
            self.detectors = {}
            for source in SOURCES:
                vectors = {}
                for split, (full, chosen) in partitions[source].items():
                    strings = contexts(full, self.config["window"])
                    vectors[split] = self.embedder.encode([strings[i] for i in chosen])
                self.detectors[source] = (
                    Detector().fit(vectors["fit"]).calibrate(vectors["calibration"])
                )
            atomic_model(path, dict(identity=identity, detectors=self.detectors))
        self.initialization_seconds = time.perf_counter() - before

    def analyze(self, raw):
        before = time.perf_counter()
        times = {}
        start = time.perf_counter()
        records = [project(r) for r in raw]
        if len({r.record_id for r in records}) != len(records):
            raise ValueError("Duplicate analysis record IDs")
        if any(r.timestamp < self.end for r in records):
            raise ValueError("Analysis records precede quiet warm-up end")
        if any(r.source not in self.detectors for r in records):
            raise ValueError("Source absent from quiet warm-up")
        times["ingestion_validation"] = time.perf_counter() - start
        if not records:
            return dict(
                profile="final",
                record_count=0,
                incidents=[],
                detected_count=0,
                stages_seconds=times,
                analysis_seconds=time.perf_counter() - before,
                clustering_status="not_run_insufficient_groups",
            )
        start = time.perf_counter()
        strings = contexts(records, self.config["window"])
        times["context"] = time.perf_counter() - start
        start = time.perf_counter()
        vectors = self.embedder.encode(strings)
        times["embedding"] = time.perf_counter() - start
        start = time.perf_counter()
        flags = np.zeros(len(records), dtype=bool)
        scores = np.zeros(len(records))
        for source, model in self.detectors.items():
            indices = [i for i, r in enumerate(records) if r.source == source]
            if indices:
                scores[indices] = model.score(vectors[indices])
                flags[indices] = scores[indices] > model.threshold
        times["detection"] = time.perf_counter() - start
        start = time.perf_counter()
        membership, trace = gated_membership(
            records,
            vectors,
            flags,
            threshold=self.config["cosine_threshold"],
            window=self.config["correlation_seconds"],
            return_trace=True,
        )
        grouped, groups = component_vectors(vectors, membership)
        times["correlation"] = time.perf_counter() - start
        start = time.perf_counter()
        labels = cluster(grouped) if len(groups) >= 10 else np.full(len(groups), -1)
        times["clustering_primary"] = time.perf_counter() - start
        start = time.perf_counter()
        comparator = (
            silhouette_kmeans(grouped) if len(groups) >= 3 else np.full(len(groups), -1)
        )
        times["clustering_comparator"] = time.perf_counter() - start
        start = time.perf_counter()
        record_labels = np.full(len(records), -1)
        group_ids = np.zeros(len(records), dtype=int)
        for j, g in enumerate(groups):
            record_labels[g] = labels[j]
            group_ids[g] = j
        terms = keywords(strings, record_labels) if any(record_labels != -1) else {}
        times["keyword_labeling"] = time.perf_counter() - start
        start = time.perf_counter()
        incidents = []
        for j, g in enumerate(groups):
            if len({records[i].source for i in g}) < 2:
                continue
            evidence = [
                dict(
                    record_id=records[i].record_id,
                    source=records[i].source,
                    timestamp=records[i].timestamp.isoformat(),
                    message=records[i].message,
                    metric=records[i].metric,
                    value=records[i].value,
                    detected=bool(flags[i]),
                    detection_score=float(scores[i]),
                )
                for i in g
            ]
            incidents.append(
                dict(
                    candidate_id=hashlib.sha256(
                        "".join(sorted(records[i].record_id for i in g)).encode()
                    ).hexdigest()[:16],
                    sources=sorted({records[i].source for i in g}),
                    evidence=evidence,
                    confidence=dict(
                        kind="heuristic, not calibrated probability",
                        **identifier_evidence([records[i] for i in g]),
                    ),
                    warning="Correlation candidate; not an established causal root cause",
                    remediation=None,
                    group=j,
                    primary_cluster=int(labels[j]),
                    comparator_cluster=int(comparator[j]),
                    labeling=dict(
                        mode="keywords",
                        keywords=terms.get(int(labels[j]), []),
                        llm_enabled=self.llm_labeler is not None,
                    ),
                )
            )
        times["reporting"] = time.perf_counter() - start
        start = time.perf_counter()
        if self.llm_labeler is not None:
            for incident in incidents:
                try:
                    incident["labeling"].update(
                        mode="local_llm",
                        status="accepted",
                        value=ClusterLabel.model_validate(
                            self.llm_labeler(incident)
                        ).model_dump(),
                    )
                except Exception as exc:
                    incident["labeling"].update(
                        mode="local_llm", status="failed", error=type(exc).__name__
                    )
        times["llm_labeling"] = time.perf_counter() - start
        return dict(
            profile="final",
            record_count=len(records),
            detected_count=int(flags.sum()),
            incidents=incidents,
            clustering_status="completed"
            if len(groups) >= 10
            else "not_run_insufficient_groups",
            cluster_count=len(set(labels) - {-1}),
            keywords=terms,
            groups=[[records[i].record_id for i in g] for g in groups],
            clustering=dict(
                unit="predicted component mean",
                primary=labels.tolist(),
                comparator=comparator.tolist(),
                comparator_status="completed"
                if len(groups) >= 3
                else "not_run_insufficient_groups",
            ),
            records=[
                dict(
                    record_id=r.record_id,
                    source=r.source,
                    detected=bool(flags[i]),
                    score=float(scores[i]),
                    group=int(group_ids[i]),
                )
                for i, r in enumerate(records)
            ],
            correlation_trace=[
                dict(
                    t,
                    record_id=records[t["record"]].record_id,
                    anchor_id=records[t["anchor"]].record_id,
                )
                for t in trace
            ],
            stages_seconds=times,
            analysis_seconds=time.perf_counter() - before,
            warmup=dict(
                start=self.start.isoformat(),
                end=self.end.isoformat(),
                declaration="operator-certified fault-free",
                cohorts=self.cohorts,
            ),
        )
