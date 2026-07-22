"""Unit tests for the base-install measures, the registry, ensembles, and
the evaluation metrics. These run with numpy and scikit-learn only.
"""

import numpy as np
import pytest

import codesim
from codesim import registry
from codesim.core import available, feature_vector, similarity
from codesim.ensemble import SupervisedEnsemble, UnsupervisedEnsemble
from codesim.evaluation import best_threshold, evaluate

CODE_A = """
public class Sum {
    public static int add(int a, int b) {
        return a + b;
    }
}
"""

CODE_A_RENAMED = """
public class Sum {
    public static int add(int x, int y) {
        return x + y;
    }
}
"""

CODE_B = """
public class Reverse {
    public static String flip(String s) {
        return new StringBuilder(s).reverse().toString();
    }
}
"""


@pytest.mark.parametrize("name", available(include_optional=False))
def test_identity_is_one(name):
    # Every base measure should call a snippet a perfect match with itself.
    assert similarity(CODE_A, CODE_A, measure=name) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("name", available(include_optional=False))
def test_range_is_unit_interval(name):
    for a, b in [(CODE_A, CODE_B), (CODE_A, CODE_A_RENAMED), (CODE_B, CODE_A)]:
        value = similarity(a, b, measure=name)
        assert 0.0 <= value <= 1.0


@pytest.mark.parametrize("name", available(include_optional=False))
def test_clone_scores_above_unrelated(name):
    # A renamed clone should look at least as similar as an unrelated program.
    clone = similarity(CODE_A, CODE_A_RENAMED, measure=name)
    other = similarity(CODE_A, CODE_B, measure=name)
    assert clone >= other - 1e-9


def test_unknown_measure_raises():
    with pytest.raises(KeyError):
        similarity(CODE_A, CODE_B, measure="does-not-exist")


def test_feature_vector_length_matches_available():
    names = available(include_optional=False)
    vec = feature_vector(CODE_A, CODE_B, names)
    assert vec.shape == (len(names),)


def test_registry_metadata_complete():
    for name in registry.names():
        spec = registry.get(name)
        assert spec.family
        assert spec.representation
        assert spec.operator
        assert spec.clone_types


def test_unsupervised_ensemble_rules():
    for rule in ("mean", "max", "weighted"):
        ens = UnsupervisedEnsemble(rule=rule)
        clone = ens.score(CODE_A, CODE_A_RENAMED)
        other = ens.score(CODE_A, CODE_B)
        assert 0.0 <= clone <= 1.0
        assert clone >= other - 1e-9


def test_supervised_ensemble_learns_and_ranks():
    pairs = [
        (CODE_A, CODE_A_RENAMED),
        (CODE_A, CODE_A),
        (CODE_B, CODE_B),
        (CODE_A, CODE_B),
        (CODE_B, CODE_A),
        (CODE_A_RENAMED, CODE_B),
    ]
    labels = [1, 1, 1, 0, 0, 0]
    ens = SupervisedEnsemble(model="random_forest").fit_pairs(pairs, labels)
    imp = ens.importances()
    assert len(imp) == len(ens.measures)
    assert abs(sum(imp.values()) - 1.0) < 1e-6


def test_evaluation_metrics_perfect_separation():
    scores = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]
    labels = [1, 1, 1, 0, 0, 0]
    result = evaluate(scores, labels, threshold=0.5)
    assert result.f1 == pytest.approx(1.0)
    assert result.roc_auc == pytest.approx(1.0)
    assert result.mcc == pytest.approx(1.0)


def test_best_threshold_beats_bad_fixed():
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.uniform(0.6, 1.0, 20), rng.uniform(0.0, 0.4, 20)])
    labels = np.concatenate([np.ones(20), np.zeros(20)])
    tuned = best_threshold(scores, labels)
    assert tuned.f1 >= 0.9


def test_version_string():
    assert isinstance(codesim.__version__, str)
