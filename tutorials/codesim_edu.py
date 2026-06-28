# -*- coding: utf-8 -*-
"""
codesim_edu — tiny, transparent code-similarity measures for teaching.

This module is the *educational companion* to the full ``similarity/`` package
in this repository. The production package contains 21 measures and some of
them depend on heavy libraries (PyTorch, Transformers, javalang, networkx,
Pygments, Pillow ...). That is great for research but awkward in a classroom,
where you want every cell to run instantly on any machine.

So here we re-implement a representative *subset* of the measure families using
ONLY the Python standard library and NumPy. Each function:

  * takes two source strings ``a`` and ``b``;
  * returns a similarity in ``[0.0, 1.0]`` (1.0 == identical by that measure);
  * is short enough to read in one sitting.

Families illustrated
--------------------
  token-based        jaccard_tokens, ngram_cosine
  sequence/edit      sequence_ratio, levenshtein_ratio, lcs_ratio
  structure proxy    token_kind_cosine        (a stand-in for AST measures)
  rename-robust      normalized_jaccard       (handles Type-2 clones)
  software metrics   metrics_cosine

Use ``MEASURES`` to iterate over them by name.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher

import numpy as np

# --------------------------------------------------------------------------- #
# Tokenisation
# --------------------------------------------------------------------------- #

# A deliberately simple, language-agnostic tokenizer good enough for Java and
# Python teaching examples: identifiers/keywords, numbers, strings, and
# operators/punctuation.
_TOKEN_RE = re.compile(
    r"""
      [A-Za-z_]\w*           # identifiers and keywords
    | \d+\.?\d*              # numbers
    | "(?:\\.|[^"\\])*"      # double-quoted strings
    | '(?:\\.|[^'\\])*'      # single-quoted strings
    | [^\sA-Za-z0-9_]        # any single punctuation / operator char
    """,
    re.VERBOSE,
)

# Common keywords shared across Java and Python. Used by the rename-robust
# measure to decide which tokens are "structural" vs. user-chosen names.
_KEYWORDS = {
    # control flow / structure (Java + Python)
    "if", "else", "elif", "for", "while", "do", "switch", "case", "default",
    "break", "continue", "return", "try", "catch", "except", "finally",
    "throw", "throws", "raise", "with", "yield", "pass", "lambda",
    # declarations / types
    "class", "interface", "enum", "def", "void", "int", "long", "short",
    "byte", "char", "boolean", "bool", "float", "double", "str", "string",
    "public", "private", "protected", "static", "final", "abstract", "new",
    "import", "package", "from", "as", "extends", "implements", "super",
    "this", "self", "null", "none", "true", "false", "and", "or", "not", "in",
    "is", "global", "nonlocal", "assert", "del",
}

_STRING_RE = re.compile(r'^["\']')
_NUMBER_RE = re.compile(r"^\d")


def tokenize(code):
    """Split source code into a flat list of token strings."""
    return _TOKEN_RE.findall(code)


def token_kind(tok):
    """Classify a token into a coarse 'kind' (a cheap AST-node stand-in)."""
    low = tok.lower()
    if low in _KEYWORDS:
        return "KW:" + low          # keep keyword identity — it is structural
    if _STRING_RE.match(tok):
        return "STR"
    if _NUMBER_RE.match(tok):
        return "NUM"
    if re.match(r"^[A-Za-z_]\w*$", tok):
        return "ID"                 # any user-chosen name collapses to ID
    return "OP:" + tok              # operators/punctuation keep identity


def normalize_identifiers(code):
    """Replace user identifiers with a placeholder, keeping keywords intact.

    This is the classic trick that turns a Type-2 (renamed) clone back into a
    Type-1 match: ``addUp(values)`` and ``sum(a)`` both become ``ID(ID)``.
    """
    out = []
    for tok in tokenize(code):
        if token_kind(tok) == "ID":
            out.append("ID")
        else:
            out.append(tok)
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _cosine(c1, c2):
    """Cosine similarity between two Counters."""
    if not c1 or not c2:
        return 0.0
    common = set(c1) & set(c2)
    dot = sum(c1[k] * c2[k] for k in common)
    n1 = math.sqrt(sum(v * v for v in c1.values()))
    n2 = math.sqrt(sum(v * v for v in c2.values()))
    return dot / (n1 * n2) if n1 and n2 else 0.0


def _jaccard(s1, s2):
    if not s1 and not s2:
        return 1.0
    u = len(s1 | s2)
    return len(s1 & s2) / u if u else 0.0


# --------------------------------------------------------------------------- #
# Measures
# --------------------------------------------------------------------------- #

def jaccard_tokens(a, b):
    """Token-set Jaccard. Strong on Type-1, weak on Type-2 (renaming)."""
    return _jaccard(set(tokenize(a)), set(tokenize(b)))


def ngram_cosine(a, b, n=3):
    """Cosine over token n-gram counts. Captures local ordering."""
    def grams(code):
        t = tokenize(code)
        if len(t) < n:
            return Counter([tuple(t)]) if t else Counter()
        return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))
    return _cosine(grams(a), grams(b))


def sequence_ratio(a, b):
    """difflib ratio on raw characters. Robust to small edits (Type-3)."""
    return SequenceMatcher(None, a, b).ratio()


def levenshtein_ratio(a, b):
    """Edit-distance ratio over tokens, computed with a small DP table."""
    s, t = tokenize(a), tokenize(b)
    m, n = len(s), len(t)
    if m == 0 and n == 0:
        return 1.0
    if m == 0 or n == 0:
        return 0.0
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    dist = prev[n]
    return 1.0 - dist / max(m, n)


def lcs_ratio(a, b):
    """Longest-common-subsequence length over tokens, normalised."""
    s, t = tokenize(a), tokenize(b)
    m, n = len(s), len(t)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return 2.0 * dp[m][n] / (m + n)


def token_kind_cosine(a, b):
    """Cosine over token-KIND multisets — a lightweight structural / AST proxy.

    Because every identifier collapses to 'ID', this measure is naturally
    robust to renaming, so it tends to score Type-2 clones highly.
    """
    ca = Counter(token_kind(t) for t in tokenize(a))
    cb = Counter(token_kind(t) for t in tokenize(b))
    return _cosine(ca, cb)


def normalized_jaccard(a, b):
    """Token Jaccard AFTER identifier normalisation. Designed for Type-2."""
    return _jaccard(set(normalize_identifiers(a)), set(normalize_identifiers(b)))


def metrics_cosine(a, b):
    """Cosine of simple software-metric vectors (size / control-flow shape).

    A pared-down version of ``similarity/metrics.py``: it ignores *what* the
    code says and looks only at its gross shape, so it is easily fooled by
    structurally-similar-but-different code (useful to discuss false positives).
    """
    def vec(code):
        loc = code.count("\n") + 1
        toks = tokenize(code)
        kw = sum(1 for t in toks if t.lower() in _KEYWORDS)
        loops = sum(1 for t in toks if t.lower() in {"for", "while"})
        conds = sum(1 for t in toks if t.lower() in {"if", "else", "elif", "switch"})
        depth, maxd = 0, 0
        for ch in code:
            if ch in "{(":
                depth += 1
                maxd = max(maxd, depth)
            elif ch in "})":
                depth -= 1
        return np.array([loc, kw, loops, conds, len(toks), maxd], dtype=float)

    v1, v2 = vec(a), vec(b)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1 / n1, v2 / n2))


# Registry: name -> function. Notebooks iterate over this.
MEASURES = {
    "jaccard_tokens": jaccard_tokens,
    "ngram_cosine": ngram_cosine,
    "sequence_ratio": sequence_ratio,
    "levenshtein_ratio": levenshtein_ratio,
    "lcs_ratio": lcs_ratio,
    "token_kind_cosine": token_kind_cosine,
    "normalized_jaccard": normalized_jaccard,
    "metrics_cosine": metrics_cosine,
}


def all_scores(a, b):
    """Return ``{measure_name: score}`` for every measure in MEASURES."""
    return {name: float(fn(a, b)) for name, fn in MEASURES.items()}


if __name__ == "__main__":
    # tiny smoke test
    x = "def f(n):\n    return n * 2\n"
    y = "def g(m):\n    return m * 2\n"
    for k, v in all_scores(x, y).items():
        print(f"{k:20s} {v:.3f}")
