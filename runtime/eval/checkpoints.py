"""Atomic model writer extracted from the archived checkpoint module."""

import os
from pathlib import Path
import tempfile

import joblib


def atomic_model(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".model-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            joblib.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
            # Linking a complete file refuses to overwrite a previously published model.
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
