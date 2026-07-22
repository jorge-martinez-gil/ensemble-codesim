"""Ensembles over similarity measures.

Two styles are provided. The unsupervised ensembles (mean, max, weighted)
need no labels and turn a set of measures into a single score with a
threshold. The supervised ensemble learns a combiner from labelled pairs and
exposes per-measure importances, which is what makes an ensemble decision
inspectable rather than opaque.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import feature_matrix, feature_vector, available


@dataclass
class UnsupervisedEnsemble:
    """Combine measures without any training labels.

    Args:
        measures: names to include; None uses every base-install measure.
        rule: "mean", "max", or "weighted".
        weights: per-measure weights for the weighted rule.
        threshold: score at or above which a pair is called a clone.
    """

    measures: list[str] | None = None
    rule: str = "mean"
    weights: np.ndarray | None = None
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.measures is None:
            self.measures = available(include_optional=False)

    def score(self, code1: str, code2: str) -> float:
        vec = feature_vector(code1, code2, self.measures)
        return self._combine(vec.reshape(1, -1))[0]

    def score_matrix(self, features: np.ndarray) -> np.ndarray:
        return self._combine(features)

    def _combine(self, features: np.ndarray) -> np.ndarray:
        if self.rule == "mean":
            return features.mean(axis=1)
        if self.rule == "max":
            return features.max(axis=1)
        if self.rule == "weighted":
            w = self.weights
            if w is None:
                w = np.ones(features.shape[1])
            w = np.asarray(w, dtype=float)
            w = w / (w.sum() or 1.0)
            return features @ w
        raise ValueError(f"unknown rule '{self.rule}'")

    def predict(self, code1: str, code2: str) -> int:
        return int(self.score(code1, code2) >= self.threshold)


@dataclass
class SupervisedEnsemble:
    """Learn to combine measures from labelled clone pairs.

    The default combiner is a random forest, matching the bagging setup in the
    ensemble-codesim study; pass ``model="gradient_boosting"`` for a boosted
    variant. Both expose ``feature_importances_`` after fitting.
    """

    measures: list[str] | None = None
    model: str = "random_forest"
    estimator: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.measures is None:
            self.measures = available(include_optional=False)
        if self.estimator is None:
            self.estimator = self._make_estimator()

    def _make_estimator(self):
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            RandomForestClassifier,
        )

        if self.model == "random_forest":
            return RandomForestClassifier(n_estimators=200, n_jobs=-1)
        if self.model == "gradient_boosting":
            return GradientBoostingClassifier()
        raise ValueError(f"unknown model '{self.model}'")

    def fit_pairs(self, pairs: list[tuple[str, str]], labels: list[int]) -> "SupervisedEnsemble":
        features = feature_matrix(pairs, self.measures)
        return self.fit(features, labels)

    def fit(self, features: np.ndarray, labels) -> "SupervisedEnsemble":
        self.estimator.fit(features, np.asarray(labels))
        return self

    def score(self, code1: str, code2: str) -> float:
        vec = feature_vector(code1, code2, self.measures).reshape(1, -1)
        return float(self.estimator.predict_proba(vec)[0][1])

    def predict(self, code1: str, code2: str) -> int:
        vec = feature_vector(code1, code2, self.measures).reshape(1, -1)
        return int(self.estimator.predict(vec)[0])

    def importances(self) -> dict[str, float]:
        """Map each measure to its learned importance, largest first."""
        weights = getattr(self.estimator, "feature_importances_", None)
        if weights is None:
            return {}
        paired = sorted(
            zip(self.measures, weights), key=lambda kv: kv[1], reverse=True
        )
        return {name: float(w) for name, w in paired}
