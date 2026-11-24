"""Quiet-baseline robust magnitude and label-independent calibration rank fusion."""

from collections import defaultdict
import numpy as np
from mozaic.schema import project


class NumericBaseline:
    def fit(self, rows):
        values = defaultdict(list)
        for raw in rows:
            r = project(raw)
            if r.metric and r.value is not None:
                values[(r.source, r.metric)].append(r.value)
        self.parameters = {}
        for key, values_for_metric in values.items():
            median = float(np.median(values_for_metric))
            scale = float(
                1.4826 * np.median(np.abs(np.asarray(values_for_metric) - median))
            )
            self.parameters[key] = (median, max(scale, 1e-6 * max(1.0, abs(median))))
        return self

    def score(self, rows):
        result = []
        for raw in rows:
            r = project(raw)
            parameters = self.parameters.get((r.source, r.metric))
            result.append(
                abs(r.value - parameters[0]) / parameters[1]
                if parameters and r.value is not None
                else 0.0
            )
        return np.asarray(result)


class RankFusion:
    def fit(self, embedding, numeric):
        self.references = [np.sort(x) for x in (embedding, numeric)]
        if any(len(x) == 0 for x in self.references):
            raise ValueError("Calibration cannot be empty")
        return self

    def score(self, embedding, numeric):
        return np.maximum(
            *[
                np.searchsorted(ref, values, side="right") / len(ref)
                for ref, values in zip(self.references, (embedding, numeric))
            ]
        )
