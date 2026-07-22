"""Core entry points: score one pair with one measure, or build the full
feature vector of every measure at once.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from . import registry


def similarity(code1: str, code2: str, measure: str = "jaccard") -> float:
    """Score a single pair of snippets with a named measure.

    Args:
        code1, code2: the two source-code strings.
        measure: a name registered in :mod:`codesim.registry`.

    Returns:
        A float in [0, 1]. A measure that fails on malformed input returns 0.0
        rather than raising, so batch scoring never aborts midway.
    """
    spec = registry.get(measure)
    try:
        value = spec.func(code1, code2)
    except RuntimeError:
        # A missing optional dependency is a real configuration error.
        raise
    except Exception:
        return 0.0
    if value != value:  # guard against NaN
        return 0.0
    return float(max(0.0, min(1.0, value)))


def available(include_optional: bool = False) -> list[str]:
    """Measure names ready to run.

    With ``include_optional`` off (the default) this lists only measures whose
    dependencies are satisfied in the current environment.
    """
    out = []
    for name in registry.names(include_optional=True):
        spec = registry.get(name)
        if include_optional:
            # Everything whose dependencies are satisfied right now.
            if _deps_present(spec.requires):
                out.append(name)
        else:
            # The default safe set: no optional deps, no side effects.
            if spec.optional or not spec.default:
                continue
            out.append(name)
    return out


def _deps_present(requires: Iterable[str]) -> bool:
    import importlib
    import shutil

    for dep in requires:
        if dep == "javac":
            if shutil.which("javac") is None:
                return False
            continue
        module = {"Pillow": "PIL"}.get(dep, dep)
        try:
            importlib.import_module(module)
        except Exception:
            return False
    return True


def feature_vector(
    code1: str,
    code2: str,
    measures: list[str] | None = None,
) -> np.ndarray:
    """Score a pair with several measures and return the stacked vector.

    This is the representation an ensemble consumes: one clone-pair becomes a
    point in measure space. When ``measures`` is None every base-install
    measure runs, in registry order.
    """
    names = measures if measures is not None else available(include_optional=False)
    return np.array([similarity(code1, code2, m) for m in names], dtype=float)


def feature_matrix(
    pairs: Iterable[tuple[str, str]],
    measures: list[str] | None = None,
) -> np.ndarray:
    """Stack :func:`feature_vector` over many pairs into an (n, k) matrix."""
    names = measures if measures is not None else available(include_optional=False)
    rows = [feature_vector(c1, c2, names) for c1, c2 in pairs]
    if not rows:
        return np.zeros((0, len(names)))
    return np.vstack(rows)
