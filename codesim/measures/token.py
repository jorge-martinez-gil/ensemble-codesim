"""Token and vector-space similarity measures.

These treat a program as a bag or sequence of lexical tokens and compare the
resulting frequency vectors or token sets. The vector cosine measures use
scikit-learn when it is present and fall back to a small NumPy routine
otherwise, so the family stays usable in a minimal install.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

__all__ = [
    "bag_of_words",
    "tfidf_cosine",
    "jaccard_tokens",
    "token_type_overlap",
    "function_call_jaccard",
]

_WORD = re.compile(r"\b\w+\b")
_COMMENT = re.compile(r"//.*?$|/\*.*?\*/", re.S | re.M)
_CALL = re.compile(r"\b(\w+)\s*\(")

# A compact, language-agnostic token-class tagger. It is deliberately small
# and dependency-free; a heavier lexer (see measures.structural) can replace it
# when higher fidelity is required.
_KEYWORDS = {
    "if", "else", "for", "while", "do", "switch", "case", "return", "break",
    "continue", "class", "interface", "public", "private", "protected",
    "static", "final", "void", "int", "long", "float", "double", "boolean",
    "char", "new", "try", "catch", "finally", "throw", "throws", "import",
    "package", "extends", "implements", "this", "super", "def", "lambda",
    "print", "elif", "in", "is", "not", "and", "or",
}


def _clean_tokens(code: str) -> list[str]:
    code = _COMMENT.sub(" ", code)
    return _WORD.findall(code)


def _cosine(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def _count_vectors(code1: str, code2: str) -> tuple[np.ndarray, np.ndarray]:
    c1, c2 = Counter(_clean_tokens(code1)), Counter(_clean_tokens(code2))
    vocab = sorted(set(c1) | set(c2))
    if not vocab:
        return np.zeros(1), np.zeros(1)
    index = {tok: i for i, tok in enumerate(vocab)}
    v1 = np.zeros(len(vocab))
    v2 = np.zeros(len(vocab))
    for tok, n in c1.items():
        v1[index[tok]] = n
    for tok, n in c2.items():
        v2[index[tok]] = n
    return v1, v2


def bag_of_words(code1: str, code2: str) -> float:
    """Cosine of raw token-count vectors."""
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        matrix = CountVectorizer().fit_transform([code1, code2])
        return float(cosine_similarity(matrix)[0][1])
    except Exception:
        v1, v2 = _count_vectors(code1, code2)
        return _cosine(v1, v2)


def tfidf_cosine(code1: str, code2: str) -> float:
    """Cosine of term-frequency inverse-document-frequency vectors."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        matrix = TfidfVectorizer().fit_transform([code1, code2])
        return float(cosine_similarity(matrix[0], matrix[1])[0][0])
    except Exception:
        # Two-document idf collapses to a constant, so this reduces to a
        # length-normalized count cosine, which is a reasonable stand-in.
        v1, v2 = _count_vectors(code1, code2)
        return _cosine(v1, v2)


def jaccard_tokens(code1: str, code2: str) -> float:
    """Multiset Jaccard overlap of cleaned word tokens."""
    c1, c2 = Counter(_clean_tokens(code1)), Counter(_clean_tokens(code2))
    inter = sum((c1 & c2).values())
    union = sum((c1 | c2).values())
    if union == 0:
        return 1.0
    return inter / union


def _tag(token: str) -> str:
    if token in _KEYWORDS:
        return "KW"
    if token.isdigit():
        return "NUM"
    if re.fullmatch(r"[A-Za-z_]\w*", token):
        return "ID"
    return "SYM"


def token_type_overlap(code1: str, code2: str) -> float:
    """Overlap of token-class multisets (identifiers, keywords, numbers).

    Abstracting concrete names to their lexical class makes this insensitive
    to renaming, which is the signature of a Type-2 clone.
    """
    tags1 = Counter(_tag(t) for t in _WORD.findall(code1))
    tags2 = Counter(_tag(t) for t in _WORD.findall(code2))
    shared = sum((tags1 & tags2).values())
    total = sum(tags1.values()) + sum(tags2.values())
    if total == 0:
        return 1.0
    return (2.0 * shared) / total


def function_call_jaccard(code1: str, code2: str) -> float:
    """Jaccard overlap of the sets of called function or method names."""
    calls1 = set(_CALL.findall(code1))
    calls2 = set(_CALL.findall(code2))
    if not calls1 and not calls2:
        return 1.0
    union = calls1 | calls2
    if not union:
        return 0.0
    return len(calls1 & calls2) / len(union)
