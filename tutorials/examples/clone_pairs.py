# -*- coding: utf-8 -*-
"""
Curated, labeled code-clone examples for teaching and quick testing.

These are small, HAND-WRITTEN, ILLUSTRATIVE examples — not benchmark data.
They exist so students and newcomers can *see* what each clone type looks
like and how similarity measures react to them. Each pair is annotated with:

    id          unique short identifier
    language    "java" | "python"
    clone_type  1 | 2 | 3 | 4 | None        (None = not a clone)
    label       1 if the pair is a clone, else 0
    code_a      first snippet
    code_b      second snippet
    explanation why the pair is (or is not) a clone

Clone taxonomy (Bellon et al., 2007; Roy & Cordy, 2007):

    Type-1  Exact clone — identical except whitespace, layout, and comments.
    Type-2  Renamed clone — Type-1 plus renamed identifiers / changed literals
            / changed types.
    Type-3  Near-miss / gapped clone — Type-2 plus added, removed, or modified
            statements.
    Type-4  Semantic clone — functionally equivalent but syntactically
            different (e.g. iterative vs. recursive).

`hard negative` pairs are NOT clones but look superficially similar (they
share tokens or structure). They are included to teach false positives.

Run this file directly to (re)generate ``clone_pairs.json``:

    python clone_pairs.py
"""

from __future__ import annotations

import json
import os

PAIRS = [
    # ---------------------------------------------------------------- TYPE 1
    {
        "id": "java_type1_sum",
        "language": "java",
        "clone_type": 1,
        "label": 1,
        "code_a": (
            "public int sum(int[] a) {\n"
            "    int total = 0;\n"
            "    for (int i = 0; i < a.length; i++) {\n"
            "        total += a[i];\n"
            "    }\n"
            "    return total;\n"
            "}\n"
        ),
        "code_b": (
            "public int sum(int[] a) {\n"
            "    // accumulate the elements\n"
            "    int total = 0;\n"
            "    for (int i = 0; i < a.length; i++) {\n"
            "        total += a[i];   // running sum\n"
            "    }\n"
            "    return total;\n"
            "}\n"
        ),
        "explanation": (
            "Identical code; the only differences are added comments and "
            "whitespace. This is the textbook Type-1 clone."
        ),
    },
    {
        "id": "python_type1_factorial",
        "language": "python",
        "clone_type": 1,
        "label": 1,
        "code_a": (
            "def factorial(n):\n"
            "    result = 1\n"
            "    for i in range(2, n + 1):\n"
            "        result *= i\n"
            "    return result\n"
        ),
        "code_b": (
            "def factorial(n):\n"
            "    result = 1\n"
            "\n"
            "    for i in range(2, n + 1):\n"
            "        result *= i   # multiply in place\n"
            "    return result\n"
        ),
        "explanation": (
            "Same statements, same identifiers; differs only by a blank line "
            "and a trailing comment. Type-1."
        ),
    },
    # ---------------------------------------------------------------- TYPE 2
    {
        "id": "java_type2_sum_renamed",
        "language": "java",
        "clone_type": 2,
        "label": 1,
        "code_a": (
            "public int sum(int[] a) {\n"
            "    int total = 0;\n"
            "    for (int i = 0; i < a.length; i++) {\n"
            "        total += a[i];\n"
            "    }\n"
            "    return total;\n"
            "}\n"
        ),
        "code_b": (
            "public int addUp(int[] values) {\n"
            "    int acc = 0;\n"
            "    for (int k = 0; k < values.length; k++) {\n"
            "        acc += values[k];\n"
            "    }\n"
            "    return acc;\n"
            "}\n"
        ),
        "explanation": (
            "Same structure and logic; method name, parameter, and local "
            "variables are renamed. No statements changed. Type-2."
        ),
    },
    {
        "id": "python_type2_factorial_renamed",
        "language": "python",
        "clone_type": 2,
        "label": 1,
        "code_a": (
            "def factorial(n):\n"
            "    result = 1\n"
            "    for i in range(2, n + 1):\n"
            "        result *= i\n"
            "    return result\n"
        ),
        "code_b": (
            "def fact(m):\n"
            "    acc = 1\n"
            "    for j in range(2, m + 1):\n"
            "        acc *= j\n"
            "    return acc\n"
        ),
        "explanation": (
            "Identical control flow; only identifiers are renamed. Type-2."
        ),
    },
    # ---------------------------------------------------------------- TYPE 3
    {
        "id": "java_type3_sum_gapped",
        "language": "java",
        "clone_type": 3,
        "label": 1,
        "code_a": (
            "public int sum(int[] a) {\n"
            "    int total = 0;\n"
            "    for (int i = 0; i < a.length; i++) {\n"
            "        total += a[i];\n"
            "    }\n"
            "    return total;\n"
            "}\n"
        ),
        "code_b": (
            "public int sumPositive(int[] a) {\n"
            "    int total = 0;\n"
            "    for (int i = 0; i < a.length; i++) {\n"
            "        if (a[i] < 0) {\n"
            "            continue;        // <-- added statement (the 'gap')\n"
            "        }\n"
            "        total += a[i];\n"
            "    }\n"
            "    return total;\n"
            "}\n"
        ),
        "explanation": (
            "Type-2-style renaming PLUS an inserted if/continue block. The "
            "added statements ('gap') make this a near-miss Type-3 clone."
        ),
    },
    {
        "id": "python_type3_factorial_gapped",
        "language": "python",
        "clone_type": 3,
        "label": 1,
        "code_a": (
            "def factorial(n):\n"
            "    result = 1\n"
            "    for i in range(2, n + 1):\n"
            "        result *= i\n"
            "    return result\n"
        ),
        "code_b": (
            "def factorial(n):\n"
            "    if n < 0:\n"
            "        raise ValueError('n must be >= 0')   # added guard\n"
            "    result = 1\n"
            "    for i in range(2, n + 1):\n"
            "        result *= i\n"
            "    return result\n"
        ),
        "explanation": (
            "Same core loop, but a validation guard was inserted. Added "
            "statements => Type-3."
        ),
    },
    # ---------------------------------------------------------------- TYPE 4
    {
        "id": "java_type4_factorial_recursive",
        "language": "java",
        "clone_type": 4,
        "label": 1,
        "code_a": (
            "public int factorial(int n) {\n"
            "    int result = 1;\n"
            "    for (int i = 2; i <= n; i++) {\n"
            "        result *= i;\n"
            "    }\n"
            "    return result;\n"
            "}\n"
        ),
        "code_b": (
            "public int factorial(int n) {\n"
            "    if (n <= 1) {\n"
            "        return 1;\n"
            "    }\n"
            "    return n * factorial(n - 1);\n"
            "}\n"
        ),
        "explanation": (
            "Both compute n!, but one is iterative and the other recursive. "
            "Functionally equivalent, syntactically very different. Type-4 — "
            "the hardest case for token/structure-based measures."
        ),
    },
    {
        "id": "python_type4_sum_formula",
        "language": "python",
        "clone_type": 4,
        "label": 1,
        "code_a": (
            "def sum_to_n(n):\n"
            "    total = 0\n"
            "    for i in range(1, n + 1):\n"
            "        total += i\n"
            "    return total\n"
        ),
        "code_b": (
            "def sum_to_n(n):\n"
            "    return n * (n + 1) // 2\n"
        ),
        "explanation": (
            "Both return 1+2+...+n. One loops, the other uses the closed-form "
            "Gauss formula. Same behaviour, no shared structure. Type-4."
        ),
    },
    # ----------------------------------------------------------- HARD NEGATIVES
    {
        "id": "java_negative_sum_vs_max",
        "language": "java",
        "clone_type": None,
        "label": 0,
        "code_a": (
            "public int sum(int[] a) {\n"
            "    int total = 0;\n"
            "    for (int i = 0; i < a.length; i++) {\n"
            "        total += a[i];\n"
            "    }\n"
            "    return total;\n"
            "}\n"
        ),
        "code_b": (
            "public int max(int[] a) {\n"
            "    int best = a[0];\n"
            "    for (int i = 1; i < a.length; i++) {\n"
            "        if (a[i] > best) {\n"
            "            best = a[i];\n"
            "        }\n"
            "    }\n"
            "    return best;\n"
            "}\n"
        ),
        "explanation": (
            "Both iterate over an int[] and look almost identical token-wise, "
            "but one sums and the other finds the maximum. NOT a clone — a "
            "hard negative that fools shallow token measures."
        ),
    },
    {
        "id": "python_negative_boilerplate",
        "language": "python",
        "clone_type": None,
        "label": 0,
        "code_a": (
            "def load_users(path):\n"
            "    with open(path) as f:\n"
            "        data = json.load(f)\n"
            "    return [u['name'] for u in data]\n"
        ),
        "code_b": (
            "def load_prices(path):\n"
            "    with open(path) as f:\n"
            "        data = json.load(f)\n"
            "    return sum(item['price'] for item in data)\n"
        ),
        "explanation": (
            "Shared file-loading boilerplate (open + json.load) inflates token "
            "overlap, but the functions return different things and serve "
            "different purposes. NOT a clone — near-miss false positive."
        ),
    },
    # --------------------------------------------------- CROSS-LANGUAGE (Type-4)
    {
        "id": "crosslang_type4_gcd",
        "language": "java|python",
        "clone_type": 4,
        "label": 1,
        "code_a": (
            "// Java\n"
            "public int gcd(int a, int b) {\n"
            "    while (b != 0) {\n"
            "        int t = b;\n"
            "        b = a % b;\n"
            "        a = t;\n"
            "    }\n"
            "    return a;\n"
            "}\n"
        ),
        "code_b": (
            "# Python\n"
            "def gcd(a, b):\n"
            "    if b == 0:\n"
            "        return a\n"
            "    return gcd(b, a % b)\n"
        ),
        "explanation": (
            "Euclid's algorithm in two languages — iterative Java vs. recursive "
            "Python. Cross-language semantic (Type-4) clone; out of reach for "
            "most lexical measures and a good stress test for an ensemble."
        ),
    },
]


def load_pairs(only_language=None, only_type="any"):
    """Return the curated clone pairs, optionally filtered.

    Parameters
    ----------
    only_language : str or None
        If given (e.g. "java" or "python"), keep only pairs whose ``language``
        contains it.
    only_type : object
        If "any" (default) keep all. Otherwise keep pairs whose ``clone_type``
        equals this value (use ``None`` to select the hard negatives).
    """
    pairs = PAIRS
    if only_language is not None:
        pairs = [p for p in pairs if only_language in p["language"]]
    if only_type != "any":
        pairs = [p for p in pairs if p["clone_type"] == only_type]
    return pairs


def _dump_json():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "clone_pairs.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(PAIRS, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(PAIRS)} pairs -> {out}")


if __name__ == "__main__":
    _dump_json()
