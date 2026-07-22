"""The measure registry: a single place that names every similarity measure
and records how it works.

The registry is what lets the rest of the toolkit treat measures uniformly.
Each entry pairs a callable with metadata used for the taxonomy tables, the
command-line listing, and dependency-aware selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .measures import textual, token, structural, learned


@dataclass(frozen=True)
class MeasureSpec:
    """Description of one similarity measure.

    Attributes:
        name: short identifier used everywhere else in the toolkit.
        func: callable with signature ``(code1, code2) -> float`` in [0, 1].
        family: one of the taxonomy families (textual, token, structural,
            learned, behavioral).
        representation: the program view the measure compares.
        operator: the comparison rule applied to that view.
        clone_types: clone types the measure is designed to catch.
        requires: optional third-party packages; empty means base install.
        cost: rough runtime cost, one of "low", "medium", "high".
    """

    name: str
    func: Callable[[str, str], float]
    family: str
    representation: str
    operator: str
    clone_types: str
    requires: tuple[str, ...] = field(default_factory=tuple)
    cost: str = "low"
    default: bool = True  # part of the default feature set when deps are met

    @property
    def optional(self) -> bool:
        return bool(self.requires)


_SPECS: list[MeasureSpec] = [
    # --- textual / lexical ------------------------------------------------
    MeasureSpec("levenshtein", textual.levenshtein, "textual",
                "character stream", "normalized edit distance", "1, 2", cost="medium"),
    MeasureSpec("lcs", textual.longest_common_substring, "textual",
                "character stream", "longest shared run", "1, 2"),
    MeasureSpec("seqratio", textual.sequence_ratio, "textual",
                "character stream", "Ratcliff/Obershelp ratio", "1, 2"),
    MeasureSpec("fuzzy", textual.fuzzy_token_sort, "textual",
                "sorted word tokens", "order-insensitive ratio", "1, 2, 3"),
    MeasureSpec("linediff", textual.line_diff, "textual",
                "line sequence", "diff coverage", "1, 3"),
    MeasureSpec("ngram", textual.char_ngram, "textual",
                "character n-grams", "set Jaccard", "1, 2"),
    MeasureSpec("winnowing", textual.winnowing, "textual",
                "local fingerprints", "fingerprint overlap", "1, 2, 3"),
    MeasureSpec("rollinghash", textual.rolling_hash_overlap, "textual",
                "fixed-length substrings", "hash-set overlap", "1, 2", cost="medium"),
    MeasureSpec("gst", textual.greedy_string_tiling, "textual",
                "word-token tiles", "greedy string tiling", "1, 2, 3", cost="high"),
    # --- token / vector-space --------------------------------------------
    MeasureSpec("bow", token.bag_of_words, "token",
                "token counts", "cosine", "1, 2"),
    MeasureSpec("tfidf", token.tfidf_cosine, "token",
                "tf-idf weights", "cosine", "1, 2"),
    MeasureSpec("jaccard", token.jaccard_tokens, "token",
                "token multiset", "Jaccard", "1, 2"),
    MeasureSpec("tokentype", token.token_type_overlap, "token",
                "token classes", "class overlap", "1, 2"),
    MeasureSpec("fcall", token.function_call_jaccard, "token",
                "called-name set", "Jaccard", "2, 3"),
    # --- syntactic / structural ------------------------------------------
    MeasureSpec("metrics", structural.software_metrics, "structural",
                "size and control metrics", "cosine", "3, 4"),
    MeasureSpec("graph", structural.adjacency_graph, "structural",
                "token-adjacency edges", "edge overlap", "2, 3"),
    MeasureSpec("pdg", structural.pdg_edit_distance, "structural",
                "program dependence graph", "graph edit distance", "3, 4",
                requires=("javalang", "networkx"), cost="high"),
    # --- learned / behavioral --------------------------------------------
    MeasureSpec("codebert", learned.codebert_cosine, "learned",
                "contextual embedding", "cosine", "3, 4",
                requires=("torch", "transformers"), cost="high"),
    MeasureSpec("doc2vec", learned.doc2vec_cosine, "learned",
                "paragraph vector", "cosine", "3, 4",
                requires=("gensim", "nltk"), cost="high"),
    MeasureSpec("image", learned.perceptual_hash, "learned",
                "rendered image", "perceptual-hash distance", "1, 2",
                requires=("Pillow", "imagehash"), cost="medium"),
    MeasureSpec("execution", learned.execution_output, "behavioral",
                "runtime output", "output similarity", "4",
                requires=("javac",), cost="high", default=False),
]

REGISTRY: dict[str, MeasureSpec] = {spec.name: spec for spec in _SPECS}


def get(name: str) -> MeasureSpec:
    """Return the spec for ``name`` or raise a helpful KeyError."""
    try:
        return REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown measure '{name}'. Known measures: {known}")


def names(include_optional: bool = True) -> list[str]:
    """All registered measure names, base-only when ``include_optional`` is off."""
    return [
        n for n, s in REGISTRY.items()
        if include_optional or not s.optional
    ]


def by_family() -> dict[str, list[str]]:
    """Measure names grouped under their taxonomy family."""
    out: dict[str, list[str]] = {}
    for name, spec in REGISTRY.items():
        out.setdefault(spec.family, []).append(name)
    return out
