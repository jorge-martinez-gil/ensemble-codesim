"""CodeSim: a unified toolkit for source code similarity and clone detection.

CodeSim gathers a broad set of unsupervised similarity measures behind one
interface, groups them into a taxonomy, and adds ensembles, evaluation, and
dataset loaders so that results are reproducible from a single install.

Quick start::

    import codesim
    codesim.similarity("int a = 1;", "int b = 1;", measure="jaccard")

    from codesim.ensemble import UnsupervisedEnsemble
    ens = UnsupervisedEnsemble(rule="mean")
    ens.score(code_a, code_b)

The companion study is Martinez-Gil, "Advanced Detection of Source Code
Clones via an Ensemble of Unsupervised Similarity Measures", SWQD 2025.
"""

from __future__ import annotations

from . import registry, evaluation, datasets
from .core import (
    available,
    feature_matrix,
    feature_vector,
    similarity,
)
from .ensemble import SupervisedEnsemble, UnsupervisedEnsemble

__version__ = "0.1.0"

__all__ = [
    "similarity",
    "available",
    "feature_vector",
    "feature_matrix",
    "UnsupervisedEnsemble",
    "SupervisedEnsemble",
    "registry",
    "evaluation",
    "datasets",
    "__version__",
]


def list_measures(include_optional: bool = True) -> list[str]:
    """Convenience wrapper over :func:`codesim.registry.names`."""
    return registry.names(include_optional=include_optional)
