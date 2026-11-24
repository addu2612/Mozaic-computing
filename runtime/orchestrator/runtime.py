"""Live replay analysis; detection scores do not gate ID correlation candidates."""

import hashlib
import json
from pathlib import Path
import time
import joblib
import os
import tempfile
import numpy as np
from eval.common import load_snapshot, partition, sample
from mozaic.schema import SOURCES, project
from mozaic.pipeline import Embedder, contexts, context_texts, text, cluster, keywords
from mozaic.detection import Detector
from mozaic.cache import digest, detector_identity
from mozaic.correlation import associate, components, identifier_evidence


class RuntimeAnalyzer:
    def __init__(self, identity=None, training=None):
        rows, snapshot = load_snapshot("data/dev") if training is None else training
        self.config = json.loads(Path("frozen_config.json").read_text())
        self.embedder = Embedder()
        self.detectors = {}
        derived = detector_identity(snapshot=snapshot) if identity is None else identity
        if derived["dataset"] != dict(
            files=snapshot["files"], manifest_sha256=snapshot["manifest_sha256"]
        ):
            raise ValueError("Detector identity differs from the loaded snapshot")
        identity = derived
        fingerprint = digest(identity)
        cache = Path("data/runtime") / f"{fingerprint}.joblib"
        self.cache_metadata = dict(key=fingerprint, hit=False, identity=identity)
        if cache.exists():
            bundle = joblib.load(cache)
            if bundle.get("identity") == identity and set(
                bundle.get("detectors", {})
            ) == set(SOURCES):
                self.detectors = bundle["detectors"]
                self.cache_metadata["hit"] = True
                return
        for source in SOURCES:
            vectors = {}
            for split in ("fit", "calibration"):
                complete = [
                    r for r in rows if r["source"] == source and partition(r) == split
                ]
                selected = sample(complete, 2000, 1411)
                vectors[split] = self.embedder.encode(
                    context_texts(complete, self.config["window"], selected)
                )
            self.detectors[source] = (
                Detector().fit(vectors["fit"]).calibrate(vectors["calibration"])
            )
        cache.parent.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache.parent, delete=False) as handle:
            temporary = Path(handle.name)
            try:
                joblib.dump(dict(identity=identity, detectors=self.detectors), handle)
                handle.flush()
                os.replace(temporary, cache)
            finally:
                temporary.unlink(missing_ok=True)

    def analyze(self, raw):
        before = time.perf_counter()
        times = {}
        start = time.perf_counter()
        records = [project(r) for r in raw]
        times["ingestion_validation"] = time.perf_counter() - start
        if not records:
            return dict(incidents=[], record_count=0, stages_seconds=times)
        start = time.perf_counter()
        plain = [text(r) for r in records]
        times["preprocessing"] = time.perf_counter() - start
        start = time.perf_counter()
        strings = contexts(records, self.config["window"])
        times["context"] = time.perf_counter() - start
        start = time.perf_counter()
        x = self.embedder.encode(strings)
        times["embedding"] = time.perf_counter() - start
        start = time.perf_counter()
        detected = np.zeros(len(records), dtype=bool)
        scores = np.zeros(len(records))
        for source, model in self.detectors.items():
            idx = [i for i, r in enumerate(records) if r.source == source]
            if idx:
                score = model.score(x[idx])
                scores[idx] = score
                detected[idx] = score > model.threshold
        times["detection"] = time.perf_counter() - start
        start = time.perf_counter()
        labels = cluster(x) if len(x) >= 10 else np.full(len(x), -1)
        times["clustering"] = time.perf_counter() - start
        start = time.perf_counter()
        terms = keywords(strings, labels) if any(labels != -1) else {}
        times["keyword_labeling"] = time.perf_counter() - start
        start = time.perf_counter()
        similarity_vectors = self.embedder.encode(plain)
        links = associate(records, similarity_vectors)
        groups = components(links)
        times["correlation"] = time.perf_counter() - start
        start = time.perf_counter()
        incidents = []
        for g in groups:
            if len({records[i].source for i in g}) < 2:
                continue
            incidents.append(
                dict(
                    candidate_id=hashlib.sha256(
                        "".join(sorted(records[i].record_id for i in g)).encode()
                    ).hexdigest()[:16],
                    sources=sorted({records[i].source for i in g}),
                    evidence=[
                        dict(
                            record_id=records[i].record_id,
                            source=records[i].source,
                            timestamp=records[i].timestamp.isoformat(),
                            message=records[i].message,
                            metric=records[i].metric,
                            value=records[i].value,
                            detected=bool(detected[i]),
                            detection_score=float(scores[i]),
                        )
                        for i in sorted(g)
                    ],
                    confidence=dict(
                        kind="heuristic, not calibrated probability",
                        **identifier_evidence([records[i] for i in sorted(g)]),
                    ),
                    warning="Correlation candidate; detection can be weak and no causal claim is established",
                    labeling=dict(mode="keywords", llm_enabled=False),
                    remediation=None,
                )
            )
        times["reporting"] = time.perf_counter() - start
        return dict(
            incidents=incidents,
            record_count=len(records),
            detected_count=int(detected.sum()),
            cluster_count=len(set(labels) - {-1}),
            clustering_status="completed"
            if len(x) >= 10
            else "not_run_insufficient_records",
            keywords=terms,
            stages_seconds=times,
            analysis_seconds=time.perf_counter() - before,
        )
