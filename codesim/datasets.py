"""Loaders for the two dataset shapes used across the clone-detection
literature: the BigCloneBench id-pair format and a folder-of-programs format
of the kind used for programming-plagiarism corpora.
"""

from __future__ import annotations

import json
import os
from typing import Iterator


def load_snippet_index(jsonl_path: str) -> dict[str, str]:
    """Read a ``.jsonl`` file of ``{"idx": ..., "func": ...}`` records.

    Returns a mapping from snippet id to source text, matching the layout of
    the BigCloneBench ``data.jsonl`` used in the ensemble-codesim study.
    """
    index: dict[str, str] = {}
    with open(jsonl_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            index[str(record.get("idx"))] = record.get("func", "")
    return index


def load_pairs(pairs_path: str, index: dict[str, str]) -> list[tuple[str, str, int]]:
    """Read a ``id1 id2 label`` pair file and resolve ids against ``index``.

    Lines whose ids are missing from the index are skipped, so a partial index
    still yields a usable subset.
    """
    out: list[tuple[str, str, int]] = []
    with open(pairs_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            id1, id2, label = parts[0], parts[1], parts[2]
            c1, c2 = index.get(id1), index.get(id2)
            if c1 is None or c2 is None:
                continue
            out.append((c1, c2, int(label)))
    return out


def iter_program_folder(root: str, extensions=(".java", ".py", ".c", ".cpp")) -> Iterator[tuple[str, str]]:
    """Yield ``(relative_path, source_text)`` for every program under ``root``.

    Useful for plagiarism-style corpora that ship one file per submission
    rather than an id-indexed pair list.
    """
    for base, _dirs, files in os.walk(root):
        for name in sorted(files):
            if name.endswith(extensions):
                path = os.path.join(base, name)
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as fh:
                        yield os.path.relpath(path, root), fh.read()
                except OSError:
                    continue


def all_program_pairs(root: str, **kwargs) -> list[tuple[str, str, str, str]]:
    """Every unordered pair of programs in a folder as
    ``(name1, code1, name2, code2)`` tuples.
    """
    programs = list(iter_program_folder(root, **kwargs))
    pairs = []
    for i in range(len(programs)):
        for j in range(i + 1, len(programs)):
            n1, c1 = programs[i]
            n2, c2 = programs[j]
            pairs.append((n1, c1, n2, c2))
    return pairs
