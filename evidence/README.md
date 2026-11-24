# MOZAIC Computing revision: aggregate evidence

This evidence accompanies the 22-page manuscript `computingq1_Updated.pdf`.
The current PDF hash and earlier version provenance are in
`aggregate_manifest.json`. The manuscript and peer review correspondence are
not included.

## Run the offline checks

```sh
python3 verify_aggregates.py
```

Python 3.9 or newer is sufficient. No credentials, network, third-party
libraries, data downloads or model execution are required. The script verifies
the listed file hashes, recomputes aggregate pair precision, recall, F1 and
false-positive rate from shipped counts, and checks the cohort identities and
arithmetic used in the tables. It does not recompute stored bootstrap intervals,
AUROC, average precision, clustering, classifications or historical predictions
from records. Hashes identify files, not the validity of their contents.

## Which result is which?

| File | What readers can verify | What it does not establish |
| --- | --- | --- |
| `round8_background_pair_counts.csv`, `round8_global_gated.json` | All 210 count rows, 240,000 eligible cross-source pairs per seed/method/condition. Five-seed macro scores are computed separately from pooled counts. The gated macro F1 is 0.556741 intact and 0.016000 at full dropout. | Global negative pairs do not equal the selected negative pairs below. Archived detector and pairing results do not validate the current runtime or causal relationships. |
| `round8_overlap_counts.csv` | Retrospective selected-pair diagnostic. In the primary cohort, gated association gave 0/95 false co-memberships intact and 4/95 after full dropout. Counts include indirect co-membership. | Zero of 95 is not zero global false-positive rate. This was a re-score of retained outputs, not a new experiment or an independent row-level replication. |
| `round8_quiet_detection_summary.csv` | Table 5 and Figure 2 source values, including the retained bootstrap endpoints. | The historical model used a label-assisted pre-incident fit interval and a sample of the full development calibration partition. It was not an evaluation of the operator-declared quiet fit/calibration profile selected later. |
| `round8_confirmation_inventory.csv`, `round8_shared_dev_cohorts.csv`, `round8_shared_dev_quiet_fit.csv` | Full-dataset and sampled class counts for 20 held-out seed/source cells and separate shared-development cohort counts. The quiet-fit Kubernetes count is 649, not the 2,000 used for the full-partition comparator. | These are not saved model state, raw memberships or proof of all historical train/test separation. |
| `public_single_source_detection.csv` | Table 4's HDFS/BGL metrics, N/N+, and retained confusion counts for the four displayed methods. | These are single-source component results. The file does not contain every archived baseline or show general superiority, multi-source correlation, root-cause diagnosis or production validity. |
| `rcaeval_exposure_counts.csv` | Nine split/modality counts, 7,200 records total, with before/after fault-injection denominators. | These are temporal-exposure proxy labels, not per-record anomaly or causal labels. No logs, traces, metric series, individual labels or predictions are included. |
| `conditional_dev_context_summary.csv` | The 16 retained development-validation context rows on fixed, label-selected synthetic cohorts. Unweighted mean ARI on this split selected W=20 (smaller W breaks ties). | This does not establish end-to-end, held-out or monotonic context benefit. |
| `historical_taxonomy_counts.csv` | Seven primary runs on the same 41 non-noise clusters from synthetic test seed 2211. Correctness required schema validation and an exact generator-family match, using the most frequent family and lexicographic ties. | These are not parse-only rates or evidence of free-text quality, validated diagnosis, remediation or current provider availability. No prompts, model replies, individual labels, reviewer data or ratings are included. |
| `offline_initialization_counts.csv` | A disposable-environment collection check gave five fit and five calibration observations per source, 40 total. The saved report records successful offline initialization. | The check stopped before analysis; it is not a scored live run or production validation. Repeated observations do not constitute an independent incident population. |

## Provenance and access

The aggregate manifest identifies the retained source report paths and code
commits where known. The audited implementation pins the encoder
`sentence-transformers/all-MiniLM-L6-v2` at revision
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf`. This evidence package does not
include model weights or execute the encoder. Historical development and held-out inventory fields, the
global pair counts and the RCAEval exposure counts were transcribed from
author-supplied private archive audits. The audit reports record hash matches
and, for RCAEval, a replay/label join. The verifier checks the included
aggregates, not the bytes or semantic contents of the original private inputs.
The audit reports and any original file hashes linked to private record
inventories are not shipped.

The upstream notices below apply to their respective sources. No upstream raw
benchmark data or third-party methods are redistributed in this evidence folder.

- HDFS/BGL source: [Loghub](https://github.com/logpai/loghub) and its [data notice](https://github.com/logpai/loghub/blob/master/LICENSE). The license grants research/academic use subject to attribution and citation; no raw dataset is redistributed here.
- [RCAEval code and data](https://github.com/phamquiluan/RCAEval#licensing) and [official dataset card](https://huggingface.co/datasets/phamquiluan/RCAEval). The authors identify their own code and datasets as MIT; separately licensed bundled methods are not automatically covered. No raw benchmark data or third-party methods are included.

This evidence supports verifying aggregate table arithmetic. The historical runs
cannot be independently reproduced from these files alone.
