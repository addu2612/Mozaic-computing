# Run the runtime checks

Use Python 3.11 or 3.12. Run these commands from the repository root:

```sh
cd runtime
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-smoke.txt
python -m pytest -q
```

The tests use fixtures and a deterministic stand-in encoder. They cover masking, context boundaries, detector initialization, association, clustering and pair scoring. After installation they need no downloaded model, telemetry account or historical data. Run without an outbound proxy configured for the HTTP client test.

## Replay four synthetic sources

From the same environment in `runtime/`:

```sh
python -m pip install -r requirements-replay.txt
python -m examples.replay_smoke
```

This example queries four replay adapters in process. It checks that responses omit reference labels and that identical example vectors do not cause normal records without IDs to be joined. It does not test MCP over HTTP. The archived server entry point uses a container bind address; do not expose it to an untrusted network.

## Read the implementation

| Step | Entry points |
| --- | --- |
| Project and normalize records | [schema](mozaic/schema.py), [context and encoder](mozaic/pipeline/__init__.py) |
| Fit and calibrate detection | [detector](mozaic/detection/__init__.py), [Round 8 baseline](eval/round8_baseline.py) |
| Associate and score pairs | [Round 8 association](eval/round8_correlation.py), [background-inclusive pair counts](eval/background_pairs.py) |
| Cluster and score incidents | [Round 8 clustering](eval/round8_clustering.py), [incident scoring](eval/incident_scoring.py) |
| Retrieve replay pages | [server](mcp_servers/server.py), [client](orchestrator/client.py) |

[Source provenance](SOURCE_PROVENANCE.json) lists the original file hashes and the two function extractions. The frozen settings and `requirements.lock.txt` are archived experiment references. For the commands above, install from the smoke or replay requirements instead.

The original encoder is `sentence-transformers/all-MiniLM-L6-v2`, revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`. To rerun the historical experiments you would also need the original or generated datasets, the generator, and experiment-specific checkpoints or row-level artifacts. These are not included.
