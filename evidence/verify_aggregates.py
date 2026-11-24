#!/usr/bin/env python3
"""Verify shipped aggregate integrity and cluster/record/pair arithmetic offline."""
import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCES = {"cloudwatch", "grafana", "kubernetes", "sentry"}
SEEDS = {7711, 8811, 9911, 10111, 12111}
METHODS = {"embedding", "gated", "gated_entities", "hybrid", "id_time", "time"}
CONDITIONS = {(0, 0), (.25, 0), (.5, 0), (.75, 0), (1, 0), (0, .1), (0, .5)}


def require(predicate, message):
    if not predicate:
        raise ValueError(message)


def read_csv(stem):
    with (ROOT / "data" / f"{stem}.csv").open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def counts(row):
    c = tuple(int(row[k]) for k in ("tp", "fp", "fn", "tn"))
    require(min(c) >= 0, "Negative confusion count")
    return c


def fraction(a, b):
    return a / b if b else 0.0


def metrics(c):
    tp, fp, fn, tn = c
    return {
        "precision": fraction(tp, tp+fp),
        "recall": fraction(tp, tp+fn),
        "f1": fraction(2*tp, 2*tp+fp+fn),
        "fpr": fraction(fp, fp+tn),
    }


def close(a, b, atol=1e-12):
    return math.isclose(a, b, rel_tol=0, abs_tol=atol)


def verify():
    manifest = json.loads((ROOT / "aggregate_manifest.json").read_text())
    for name, attrs in manifest["files"].items():
        path = ROOT / name
        require(path.is_file() and path.resolve().is_relative_to(ROOT), f"Invalid path: {name}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == attrs["sha256"], f"Hash: {name}")

    rows = read_csv("round8_background_pair_counts")
    identities = set()
    truth = defaultdict(set)
    grouped = defaultdict(list)
    for row in rows:
        seed, method = int(row["seed"]), row["method"]
        cond = (float(row["id_drop_rate"]), float(row["id_corrupt_rate"]))
        key = (seed, method, cond)
        require(key not in identities and row["panel"] == "background_inclusive", f"Pair key: {key}")
        identities.add(key)
        c = counts(row)
        require(sum(c) == 240_000, f"Pair denominator: {key}")
        truth[seed].add(c[0]+c[2])
        if method == "gated":
            grouped[cond].append(c)
    require(identities == set(itertools.product(SEEDS, METHODS, CONDITIONS)), "Incomplete pair grid")
    require(all(len(x) == 1 for x in truth.values()), "Truth totals varied with method/condition")

    expected = json.loads((ROOT / "data/round8_global_gated.json").read_text())["rows"]
    require(len(expected) == len(CONDITIONS), "Wrong number of retained gated rows")
    for row in expected:
        key = (row["id_drop_rate"], row["id_corrupt_rate"])
        averages = {m: statistics.mean(metrics(c)[m] for c in grouped[key]) for m in metrics((0,0,0,0))}
        for metric in ("precision", "recall", "f1", "fpr"):
            saved = row[f"{metric}_mean_report_display"]
            require(f"{averages[metric]:.{len(saved.partition('.')[2])}f}" == saved,
                    f"Historical displayed macro {metric}: {key}")
        lo, hi = row["fpr_ci95"]
        require(lo <= averages["fpr"] <= hi, f"Saved FPR interval is inconsistent: {key}")
        print(f"drop={key[0]:.2f} corrupt={key[1]:.2f} macro_f1={averages['f1']:.6f} "
              f"mean_fpr_per_million={averages['fpr']*1e6:.2f}")

    for cond, pooled in [((0, 0), (45, 11, 46, 1199898)), ((1, 0), (1, 13, 90, 1199896))]:
        require(tuple(map(sum, zip(*grouped[cond]))) == pooled, f"Pooled reconciliation: {cond}")

    overlap = {}
    for row in read_csv("round8_overlap_counts"):
        key = (row["method"], float(row["id_drop_rate"]), float(row["id_corrupt_rate"]), row["cohort"])
        require(key not in overlap, f"Duplicate selected-pair diagnostic: {key}")
        n = int(row["eligible_negative_pairs"])
        overlap[key] = (int(row["FP"]), n)
        require(int(row["FP"]) + int(row["TN"]) == n, f"Selected-pair denominator: {key}")
    require(len(overlap) == 84, "Incomplete selected-pair grid")
    require(overlap["gated", 0, 0, "primary_overlap"] == (0, 95), "Intact overlap")
    require(overlap["gated", 1, 0, "primary_overlap"] == (4, 95), "Full-drop overlap")

    public = read_csv("public_single_source_detection")
    require(Counter(r["dataset"] for r in public) == {"hdfs": 4, "bgl": 4}, "Public grid")
    for row in public:
        c = counts(row)
        require(sum(c) == int(row["records"]) and c[0]+c[2] == int(row["positives"]), "Public counts")
        for key in ("precision", "recall", "f1"):
            require(close(metrics(c)[key], float(row[key])), f"Public metric: {row['method']}, {key}")

    classes = ("normal", "memory_leak", "deployment_failure", "database_connection_pool_exhaustion")
    heldout = read_csv("round8_confirmation_inventory")
    require({(int(row["seed"]), row["source"].lower()) for row in heldout}
            == set(itertools.product(SEEDS, SOURCES)), "Held-out inventory grid")
    require(len(heldout) == 20, "Held-out count")
    for row in heldout:
        require(sum(int(row[f"full_{c}"]) for c in classes) == int(row["full_records"]), "Full inventory")
        require(sum(int(row[f"detection_{c}"]) for c in classes)
                == int(row["detection_test_records"]) == 2000, "Detection inventory")

    dev = read_csv("round8_shared_dev_cohorts")
    require(len(dev) == 24 and Counter(r["source"] for r in dev) == {s: 6 for s in SOURCES}, "Dev inventory grid")
    for row in dev:
        require(sum(int(row[c]) for c in classes) == int(row["total"]), "Dev inventory")
    quiet = read_csv("round8_shared_dev_quiet_fit")
    require({r["source"]: int(r["quiet_fit_selected"]) for r in quiet}
            == {"cloudwatch":2000,"grafana":2000,"kubernetes":649,"sentry":2000}, "Quiet fit")

    rca = read_csv("rcaeval_exposure_counts")
    require(len(rca) == 9 and sum(int(r["records"]) for r in rca) == 7200, "RCAEval cohort size")
    for row in rca:
        require(int(row["pre_injection"])+int(row["post_injection"]) == int(row["records"]), "RCAEval counts")

    smoke = read_csv("offline_initialization_counts")
    require({r["source"] for r in smoke} == SOURCES and len(smoke) == 4, "Offline smoke sources")
    require(all(int(r["fit_observations"]) == int(r["calibration_observations"]) == 5 for r in smoke),
            "Offline smoke partition counts")

    context = read_csv("conditional_dev_context_summary")
    windows = {0, 5, 10, 20}
    require({(r["source"], int(r["window"])) for r in context} == set(itertools.product(SOURCES, windows)),
            "Context selection grid")
    arb = {w: statistics.mean(float(r["ari"]) for r in context if int(r["window"]) == w) for w in windows}
    require([f"{arb[w]:.4f}" for w in sorted(windows)] == ["0.1760", "0.4398", "0.7547", "0.8421"],
            "Historical development context means")
    require(max(arb, key=lambda w: (arb[w], -w)) == 20, "Development winner")

    labels = read_csv("historical_taxonomy_counts")
    require([int(r["valid_and_correct"]) for r in labels] == [21,8,26,21,28,22,24]
            and all(int(r["eligible_clusters"]) == 41 for r in labels), "Historical taxonomy counts")
    print("PASS: shipped aggregate checks. No experiment, model, provider or raw archive was run.")


if __name__ == "__main__":
    verify()
