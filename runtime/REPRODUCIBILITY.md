# Reproducibility scope

Run from `runtime/` with Python 3.11 or 3.12:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-smoke.txt
python -m pytest -q
```

The tests use fixtures and a deterministic stand-in encoder. They check
input masking, partition-local causal context, detector initialization,
pair scoring, replay client configuration, and association and clustering
rules. No telemetry account, model download, paid provider, or historical
artifact is needed to execute the tests after installation. The HTTP client
test assumes the process is not configured to use a system outbound proxy.

To also exercise in-process ingestion for all four replay sources with
invented telemetry:

```sh
python -m pip install -r requirements-replay.txt
python -m examples.replay_smoke
```

The replay example checks that the source query response excludes ground
truth and that normal records with missing IDs are not joined solely because
their toy vectors match. It does not start an HTTP server or test the MCP wire
transport. The historical replay server and client are included for inspection.
The replay server entry point uses a container-oriented bind address; do not
expose it to untrusted networks. No production authentication, telemetry
ingestion, or production deployment is provided by this package.

## What comes from the paper

| Area | Code |
| --- | --- |
| Projection and context | `mozaic/schema.py`, `mozaic/pipeline/`, `eval/common.py` |
| Fitted detection | `mozaic/detection/`, `eval/round8_baseline.py`, `eval/round8_numeric.py` |
| Association and scoring | `mozaic/correlation/`, `eval/round8_correlation.py`, `eval/pair_metrics.py`, `eval/background_pairs.py` |
| Clustering and incident truth | `eval/round8_clustering.py`, `eval/incident_scoring.py` |
| Replay services and runtime profiles | `mcp_servers/`, `orchestrator/`, `frozen_final_profile.json` |

`SOURCE_PROVENANCE.json` records the selected byte-identical files, with
SHA-256 and Git blob SHA-1 hashes, and the two deliberate function extractions.
The frozen settings and archived dependency constraints are historical
references, not proof that the published metrics reran in this package.
The supplied releases are separated by protocol: the synthetic Round 8
confirmation and the later runtime warm-up are not the same evaluation.
The archived Round 9 result was obtained by rescoring saved assignments.

A full historical rerun is outside the scope of this extract. It requires
source data, the MORPH-derived generator, the pinned MiniLM encoder, and
experiment-specific checkpoints or row-level artifacts not distributed
here. The toy smoke output does not reproduce a manuscript table, establish
production accuracy, or validate free-text diagnoses.
