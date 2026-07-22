"""Command-line interface for CodeSim.

Examples::

    codesim list
    codesim compare a.java b.java --measure jaccard
    codesim vector a.java b.java
    codesim evaluate --data data.jsonl --pairs test.txt --measure tfidf
"""

from __future__ import annotations

import argparse
import sys

from . import registry
from .core import available, feature_vector, similarity
from .datasets import load_pairs, load_snippet_index
from .evaluation import best_threshold, evaluate


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _cmd_list(args: argparse.Namespace) -> int:
    ready = set(available(include_optional=False))
    families = registry.by_family()
    for family in sorted(families):
        print(f"[{family}]")
        for name in families[family]:
            spec = registry.get(name)
            mark = "ok " if name in ready or not spec.optional else "opt"
            extra = f" needs {', '.join(spec.requires)}" if spec.requires else ""
            print(f"  {mark}  {name:<12} {spec.representation} -> {spec.operator}"
                  f"  (types {spec.clone_types}){extra}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    score = similarity(_read(args.file1), _read(args.file2), measure=args.measure)
    print(f"{args.measure}\t{score:.4f}")
    return 0


def _cmd_vector(args: argparse.Namespace) -> int:
    names = available(include_optional=False)
    vec = feature_vector(_read(args.file1), _read(args.file2), names)
    for name, value in zip(names, vec):
        print(f"{name}\t{value:.4f}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    index = load_snippet_index(args.data)
    pairs = load_pairs(args.pairs, index)
    if not pairs:
        print("no resolvable pairs found", file=sys.stderr)
        return 1
    scores = [similarity(c1, c2, args.measure) for c1, c2, _ in pairs]
    labels = [lbl for _, _, lbl in pairs]
    result = best_threshold(scores, labels) if args.tune else evaluate(scores, labels, args.threshold)
    for key, value in result.as_dict().items():
        print(f"{key}\t{value:.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codesim", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list registered measures")
    p_list.set_defaults(func=_cmd_list)

    p_cmp = sub.add_parser("compare", help="score two files with one measure")
    p_cmp.add_argument("file1")
    p_cmp.add_argument("file2")
    p_cmp.add_argument("--measure", default="jaccard")
    p_cmp.set_defaults(func=_cmd_compare)

    p_vec = sub.add_parser("vector", help="score two files with every measure")
    p_vec.add_argument("file1")
    p_vec.add_argument("file2")
    p_vec.set_defaults(func=_cmd_vector)

    p_eval = sub.add_parser("evaluate", help="score a benchmark pair file")
    p_eval.add_argument("--data", required=True, help="jsonl snippet index")
    p_eval.add_argument("--pairs", required=True, help="id1 id2 label file")
    p_eval.add_argument("--measure", default="tfidf")
    p_eval.add_argument("--threshold", type=float, default=0.5)
    p_eval.add_argument("--tune", action="store_true", help="pick the best F1 threshold")
    p_eval.set_defaults(func=_cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
