import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import numpy as np
from sklearn import metrics


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(
            obj,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=lambda o: o.item() if hasattr(o, "item") else str(o),
        )
        + "\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
            handle.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def load(directory):
    rows = []
    for p in sorted(Path(directory).glob("*.jsonl")):
        rows.extend(json.loads(line) for line in p.read_text().splitlines())
    return rows


def load_snapshot(directory):
    """Hash and parse the same single read of every declared training file."""
    directory = Path(directory)
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    expected = manifest["sha256"]
    if any(Path(name).name != name for name in expected):
        raise ValueError("Training manifest requires plain file names")
    actual_jsonl = {p.name for p in directory.glob("*.jsonl")}
    if actual_jsonl != {name for name in expected if name.endswith(".jsonl")}:
        raise ValueError("Training JSONL inventory differs from its manifest")
    rows, hashes, auxiliary = [], {}, {}
    for name in sorted(expected):
        content = (directory / name).read_bytes()
        hashes[name] = hashlib.sha256(content).hexdigest()
        if hashes[name] != expected[name]:
            raise ValueError("Training data differ from their recorded manifest")
        if name.endswith(".jsonl"):
            rows.extend(json.loads(line) for line in content.splitlines())
        elif name.endswith(".json"):
            auxiliary[name] = json.loads(content)
    return rows, dict(
        files=hashes,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        auxiliary=auxiliary,
    )


def group(row):
    return row.get("incident_id") or "normal_" + row["timestamp"][:14] + str(
        int(row["timestamp"][14:16]) // 10
    )


def partition(row):
    number = int(hashlib.sha256(group(row).encode()).hexdigest()[:8], 16) % 10
    return "fit" if number < 6 else "calibration" if number < 8 else "validation"


def sample(rows, n, seed):
    return random.Random(seed).sample(rows, min(n, len(rows)))


def detection_metrics(y, score, pred):
    out = {
        k: float(f(y, pred, zero_division=0))
        for k, f in [
            ("precision", metrics.precision_score),
            ("recall", metrics.recall_score),
            ("f1", metrics.f1_score),
        ]
    }
    out.update(records=len(y), positives=int(np.sum(y)))
    out["auroc"] = float(metrics.roc_auc_score(y, score)) if len(set(y)) > 1 else None
    out["auprc"] = (
        float(metrics.average_precision_score(y, score)) if np.sum(y) > 0 else None
    )
    out["confusion_matrix"] = metrics.confusion_matrix(y, pred, labels=[0, 1]).tolist()
    return out


def clustering_metrics(y, labels, x):
    labels = np.asarray(labels)
    keep = labels != -1
    unique = set(labels[keep])
    out = dict(
        ari=float(metrics.adjusted_rand_score(y, labels)),
        nmi=float(metrics.normalized_mutual_info_score(y, labels)),
        clusters=len(unique),
        noise_fraction=float(np.mean(~keep)),
        records=len(y),
    )
    for name, fn in [
        ("silhouette", metrics.silhouette_score),
        ("davies_bouldin", metrics.davies_bouldin_score),
        ("calinski_harabasz", metrics.calinski_harabasz_score),
    ]:
        out[name] = (
            float(fn(x[keep], labels[keep]))
            if 1 < len(unique) < int(keep.sum())
            else None
        )
    if out["silhouette"] is None:
        out["undefined_reason"] = (
            "non-noise partition has fewer than two clusters, or one cluster per point"
        )
    return out
