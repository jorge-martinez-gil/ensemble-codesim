"""Syntactic and structural similarity measures.

The two default members here (software metrics and a token-adjacency graph)
run with NumPy alone. A program-dependence-graph measure is offered as an
optional adapter that activates when the ``javalang`` and ``networkx``
packages are installed.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

__all__ = ["software_metrics", "adjacency_graph", "pdg_edit_distance"]

_JAVA_KEYWORDS = [
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
    "class", "const", "continue", "default", "do", "double", "else", "enum",
    "extends", "final", "finally", "float", "for", "if", "implements",
    "import", "instanceof", "int", "interface", "long", "native", "new",
    "package", "private", "protected", "public", "return", "short", "static",
    "strictfp", "super", "switch", "synchronized", "this", "throw", "throws",
    "transient", "try", "void", "volatile", "while",
]

_METHOD = re.compile(r"\b(public|protected|private|static)\s+[\w<>\[\]]+\s+\w+\s*\(")


def _metric_vector(code: str) -> np.ndarray:
    lines = len(code.split("\n"))
    keywords = sum(code.count(k) for k in _JAVA_KEYWORDS)
    loops = code.count("for") + code.count("while")
    conditionals = code.count("if") + code.count("else") + code.count("switch")
    functions = len(_METHOD.findall(code))
    depth = 0
    max_depth = 0
    for ch in code:
        if ch == "{":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == "}":
            depth -= 1
    return np.array(
        [lines, keywords, loops, conditionals, functions, max_depth],
        dtype=float,
    )


def software_metrics(code1: str, code2: str) -> float:
    """Cosine of six-dimensional size and control-structure metric vectors."""
    v1 = _metric_vector(code1)
    v2 = _metric_vector(code2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1 / n1, v2 / n2))


def adjacency_graph(code1: str, code2: str) -> float:
    """Overlap of directed token-adjacency edges (a lightweight graph view).

    Each program becomes a graph whose edges join consecutive tokens. Scoring
    the shared edge multiset, rather than only shared nodes, keeps local order
    information that a plain bag of tokens discards.
    """
    def edges(code: str) -> Counter:
        toks = code.split()
        return Counter(zip(toks, toks[1:]))

    e1, e2 = edges(code1), edges(code2)
    shared = sum((e1 & e2).values())
    total = sum(e1.values()) + sum(e2.values())
    if total == 0:
        return 1.0
    return (2.0 * shared) / total


def pdg_edit_distance(code1: str, code2: str) -> float:
    """One minus a normalized graph-edit distance over program-dependence
    graphs. Optional: needs ``javalang`` and ``networkx``.
    """
    try:
        import javalang
        import networkx as nx
    except Exception as exc:  # pragma: no cover - optional path
        raise RuntimeError(
            "pdg_edit_distance needs the optional packages 'javalang' and "
            "'networkx' (pip install codesim[structural])"
        ) from exc

    def build(code: str):
        tokens = list(javalang.tokenizer.tokenize(code))
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()
        graph = nx.DiGraph()
        for _, node in tree:
            if isinstance(node, javalang.tree.MethodDeclaration):
                graph.add_node(node.name)
                for param in node.parameters:
                    graph.add_edge(param.name, node.name)
        return graph

    try:
        g1, g2 = build(code1), build(code2)
        denom = max(len(g1.nodes) + len(g2.nodes), 1)
        distance = nx.graph_edit_distance(
            g1, g2, node_match=lambda a, b: a == b
        )
        if distance is None:
            return 0.0
        return max(0.0, min(1.0, 1.0 - distance / denom))
    except Exception:
        return 0.0
