# Evidence guide

This folder accompanies `computingq1_Updated.pdf`, the 22-page Computing revision. The [manifest](aggregate_manifest.json) identifies the exact PDF and records the sources and hashes of the files below.

Run `python3 evidence/verify_aggregates.py` from the repository root. It checks file hashes, recomputes pair metrics and selected count-based results, and tests cohort arithmetic. Reported bootstrap intervals, ranking scores and clustering scores are archived values, not recalculated from raw records.

| Manuscript | Released evidence | How to read it |
| --- | --- | --- |
| Table 4 | [Public detection summary](data/public_single_source_detection.csv) | HDFS and BGL are sampled single-source tests. The table has retained confusion counts and ranking metrics for the four displayed methods. |
| Table 5 and Figure 2 | [Quiet-baseline detection](data/round8_quiet_detection_summary.csv) | Five synthetic confirmation seeds under the historical quiet-baseline protocol, not the later runtime profile. |
| Table 6 and Figure 3A | [Pair counts](data/round8_background_pair_counts.csv) · [Gated summary](data/round8_global_gated.json) | 210 count rows with 240,000 eligible cross-source pairs per seed, method and condition. Seed-averaged metrics and pooled counts are distinct. |
| Table 7 and Figure 3B | [Selected negative pairs](data/round8_overlap_counts.csv) | False predicted co-memberships among selected pairs from distinct incidents, including indirect links. These are not global false-positive rates. |
| §§4.4, 5 and 6.2 | [Confirmation inventory](data/round8_confirmation_inventory.csv) · [Development cohorts](data/round8_shared_dev_cohorts.csv) · [Quiet-fit counts](data/round8_shared_dev_quiet_fit.csv) | Twenty held-out seed/source cells and separate shared-development counts. These are not saved model states or row-level splits. |
| §§5.1 and 6.4 | [RCAEval exposure counts](data/rcaeval_exposure_counts.csv) | 7,200 records, tabulated by modality and case split. The labels measure exposure to an injected fault, not per-record anomaly or cause. |
| §6.5 | [Context-window results](data/conditional_dev_context_summary.csv) | Sixteen rows on fixed development-validation cohorts. The selected window of 20 is not a held-out result. |
| Table 8 | [Scenario-label counts](data/historical_taxonomy_counts.csv) | Seven archived runs on 41 non-noise clusters from seed 2211. Correctness required valid schema and an exact match to the generator-family reference. No human ratings of free text are included. |
| Table 9 | [Collection and initialization counts](data/offline_initialization_counts.csv) | Five fit and five calibration observations per source, 40 total. No scored live run. |

Some inventories and counts come from author-supplied private archive audits, as noted in the manifest. The public verifier checks the released evidence, not the contents of those private inputs. Round 9 rescored saved assignments rather than refitting or reclustering.

Data sources: [Loghub](https://github.com/logpai/loghub) ([notice](notices/Loghub.txt)) and [RCAEval](https://github.com/phamquiluan/RCAEval) ([notice](notices/RCAEval-MIT.txt)). No raw benchmark datasets are redistributed here.
