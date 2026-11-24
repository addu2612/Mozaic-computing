"""Content identities for embedding and fitted-detector caches."""

import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def environment():
    return dict(
        python=platform.python_version(),
        machine=platform.machine(),
        libraries={
            d.metadata["Name"].lower(): d.version for d in metadata.distributions()
        },
    )


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def embedding_libraries():
    return {
        name: metadata.version(name)
        for name in ("torch", "sentence-transformers", "transformers", "tokenizers")
    }


def embedding_identity(
    model, revision, *, normalize_embeddings, max_seq_length, libraries=None
):
    return dict(
        model=model,
        revision=revision,
        normalize_embeddings=normalize_embeddings,
        max_seq_length=max_seq_length,
        libraries=embedding_libraries() if libraries is None else libraries,
    )


def detector_identity(root=Path("."), dataset=Path("data/dev"), *, snapshot=None):
    root, dataset = Path(root), Path(dataset)
    if snapshot is None:
        from eval.common import load_snapshot

        _, snapshot = load_snapshot(root / dataset)
    files = sorted((root / "mozaic").rglob("*.py")) + [
        root / "eval/common.py",
        root / "orchestrator/runtime.py",
        root / "requirements.txt",
        root / "requirements.lock.txt",
        root / "frozen_config.json",
    ]
    return dict(
        schema=2,
        source={str(p.relative_to(root)): file_hash(p) for p in files},
        dataset=dict(
            files=snapshot["files"], manifest_sha256=snapshot["manifest_sha256"]
        ),
        environment=environment(),
    )
