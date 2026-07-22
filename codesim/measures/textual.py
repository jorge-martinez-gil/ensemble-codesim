"""Textual and lexical similarity measures.

Every measure in this module works on raw character streams or simple
whitespace/word tokens. Nothing here needs a heavyweight dependency, so the
whole family runs with the Python standard library plus NumPy.

Each public function has the signature ``f(code1: str, code2: str) -> float``
and returns a score in the closed interval [0, 1], where 1 means the two
inputs look identical to that measure and 0 means they share nothing.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter

__all__ = [
    "levenshtein",
    "longest_common_substring",
    "sequence_ratio",
    "fuzzy_token_sort",
    "line_diff",
    "char_ngram",
    "winnowing",
    "rolling_hash_overlap",
    "greedy_string_tiling",
]

_WORD = re.compile(r"\w+")


def _words(text: str) -> list[str]:
    return _WORD.findall(text)


def _edit_distance(a: str, b: str) -> int:
    """Classic dynamic-programming Levenshtein distance with O(min) memory."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            insert = previous[j] + 1
            delete = current[j - 1] + 1
            replace = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def levenshtein(code1: str, code2: str) -> float:
    """One minus the normalized character-level edit distance."""
    longest = max(len(code1), len(code2))
    if longest == 0:
        return 1.0
    return 1.0 - _edit_distance(code1, code2) / longest


def longest_common_substring(code1: str, code2: str) -> float:
    """Twice the longest shared contiguous run over the summed lengths."""
    total = len(code1) + len(code2)
    if total == 0:
        return 1.0
    match = difflib.SequenceMatcher(None, code1, code2).find_longest_match(
        0, len(code1), 0, len(code2)
    )
    return (2.0 * match.size) / total


def sequence_ratio(code1: str, code2: str) -> float:
    """Ratcliff/Obershelp matching ratio over the raw characters."""
    return difflib.SequenceMatcher(None, code1, code2).ratio()


def fuzzy_token_sort(code1: str, code2: str) -> float:
    """Order-insensitive token match: sort the words, then compare.

    This reproduces the behaviour of the popular ``token_sort_ratio`` helper
    with only the standard library, so re-ordered statements still score high.
    """
    t1 = " ".join(sorted(_words(code1.lower())))
    t2 = " ".join(sorted(_words(code2.lower())))
    if not t1 and not t2:
        return 1.0
    return difflib.SequenceMatcher(None, t1, t2).ratio()


def line_diff(code1: str, code2: str) -> float:
    """One minus the fraction of lines that a unified diff adds or removes."""
    lines1 = code1.splitlines()
    lines2 = code2.splitlines()
    denom = len(lines1) + len(lines2)
    if denom == 0:
        return 1.0
    changed = 0
    for line in difflib.ndiff(lines1, lines2):
        if line.startswith("+ ") or line.startswith("- "):
            changed += 1
    return max(0.0, 1.0 - changed / denom)


def char_ngram(code1: str, code2: str, n: int = 12) -> float:
    """Jaccard overlap of character n-gram sets (a fingerprinting proxy)."""
    g1 = {code1[i : i + n] for i in range(len(code1) - n + 1)}
    g2 = {code2[i : i + n] for i in range(len(code2) - n + 1)}
    if not g1 and not g2:
        return 1.0
    union = g1 | g2
    if not union:
        return 0.0
    return len(g1 & g2) / len(union)


def _winnow_fingerprints(text: str, k: int, w: int) -> set[int]:
    grams = [text[i : i + k] for i in range(len(text) - k + 1)]
    if not grams:
        return set()
    hashes = [hash(g) & 0xFFFFFFFF for g in grams]
    prints: set[int] = set()
    for i in range(len(hashes) - w + 1):
        window = hashes[i : i + w]
        prints.add(min(window))
    if not prints:  # text shorter than one window
        prints.add(min(hashes))
    return prints


def winnowing(code1: str, code2: str, k: int = 5, w: int = 4) -> float:
    """Winnowing fingerprint overlap, the local-fingerprint idea from MOSS."""
    f1 = _winnow_fingerprints(code1, k, w)
    f2 = _winnow_fingerprints(code2, k, w)
    if not f1 and not f2:
        return 1.0
    union = f1 | f2
    if not union:
        return 0.0
    return len(f1 & f2) / len(union)


def rolling_hash_overlap(code1: str, code2: str, length: int = 10) -> float:
    """Fraction of shared fixed-length substrings found with a rolling hash."""
    def hashes(text: str) -> set[int]:
        if len(text) < length:
            return set()
        base, mod = 256, 1_000_000_007
        power = pow(base, length - 1, mod)
        value = 0
        out: set[int] = set()
        for i, ch in enumerate(text):
            value = (value * base + ord(ch)) % mod
            if i >= length:
                value = (value - ord(text[i - length]) * base * power) % mod
            if i >= length - 1:
                out.add(value % mod)
        return out

    h1 = hashes(code1)
    h2 = hashes(code2)
    if not h1 and not h2:
        return 1.0
    union = h1 | h2
    if not union:
        return 0.0
    return len(h1 & h2) / len(union)


def greedy_string_tiling(code1: str, code2: str, minimum_tile: int = 3) -> float:
    """Running Karp-Rabin greedy string tiling over word tokens.

    The score is twice the total matched-token length over the summed token
    counts, the standard coverage measure used in plagiarism detectors.
    """
    a = _words(code1)
    b = _words(code2)
    total = len(a) + len(b)
    if total == 0:
        return 1.0
    marked_a = [False] * len(a)
    marked_b = [False] * len(b)
    matched = 0
    while True:
        best = 0
        best_pos = None
        for i in range(len(a)):
            if marked_a[i]:
                continue
            for j in range(len(b)):
                if marked_b[j]:
                    continue
                run = 0
                while (
                    i + run < len(a)
                    and j + run < len(b)
                    and not marked_a[i + run]
                    and not marked_b[j + run]
                    and a[i + run] == b[j + run]
                ):
                    run += 1
                if run > best:
                    best = run
                    best_pos = (i, j)
        if best < minimum_tile or best_pos is None:
            break
        i, j = best_pos
        for r in range(best):
            marked_a[i + r] = True
            marked_b[j + r] = True
        matched += best
    return (2.0 * matched) / total
