"""Unsupervised fitted anomaly detection on all raw telemetry, without flags."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


class Detector:
    def __init__(self, method="MOZAIC", seed=1411):
        self.method = method
        self.seed = seed

    def fit(self, x):
        if self.method == "MOZAIC":
            self.model = KMeans(
                n_clusters=min(32, len(x)), random_state=self.seed, n_init=10
            ).fit(x)
        elif self.method == "IsolationForest":
            self.model = IsolationForest(
                n_estimators=100, random_state=self.seed, n_jobs=2
            ).fit(x)
        elif self.method == "OneClassSVM":
            self.model = OneClassSVM(nu=0.05, gamma="scale").fit(x)
        else:
            raise ValueError("unknown detector")
        return self

    def score(self, x):
        if self.method == "MOZAIC":
            return self.model.transform(x).min(axis=1)
        return -self.model.score_samples(x)

    def calibrate(self, x):
        self.threshold = float(np.quantile(self.score(x), 0.95))
        return self

    def predict(self, x):
        return (self.score(x) > self.threshold).astype(int)


def rule(record):
    msg = record.message.lower()
    metric = record.metric.lower()
    value = record.value
    if any(
        k in msg
        for k in (
            "oom",
            "out of memory",
            "back-off",
            "failed",
            "timeout",
            "exhausted",
            "fatal",
            "cannot allocate",
        )
    ):
        return 1.0
    if value is not None:
        if any(k in metric for k in ("memory", "cpu", "pool")) and value > 85:
            return 1.0
        if (
            any(k in metric for k in ("latency", "duration", "response time"))
            and value > 500
        ):
            return 1.0
        if "5xx" in metric and value > 0:
            return 1.0
    return 0.0
