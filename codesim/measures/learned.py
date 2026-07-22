"""Learned and behavioral similarity measures (optional adapters).

Each entry here depends on a heavier stack (transformers, gensim, a Java
toolchain, imaging libraries). The functions import their backend lazily and
raise a clear message when it is missing, so the base install stays small
while these remain one ``pip install`` away.
"""

from __future__ import annotations

import re

__all__ = [
    "codebert_cosine",
    "doc2vec_cosine",
    "perceptual_hash",
    "execution_output",
]

_MODEL_CACHE: dict[str, object] = {}


def codebert_cosine(code1: str, code2: str, model_name: str = "microsoft/codebert-base") -> float:
    """Cosine of contextual [CLS] embeddings from a pre-trained code encoder.

    Optional: needs ``torch`` and ``transformers``.
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:  # pragma: no cover - optional path
        raise RuntimeError(
            "codebert_cosine needs 'torch' and 'transformers' "
            "(pip install codesim[learned])"
        ) from exc

    if model_name not in _MODEL_CACHE:
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        _MODEL_CACHE[model_name] = (tok, model)
    tokenizer, model = _MODEL_CACHE[model_name]  # type: ignore[misc]

    def embed(code: str):
        inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            out = model(**inputs)
        cls = out.last_hidden_state[:, 0, :]
        return torch.nn.functional.normalize(cls, p=2, dim=1)

    e1, e2 = embed(code1), embed(code2)
    return float(torch.nn.functional.cosine_similarity(e1, e2).item())


def doc2vec_cosine(code1: str, code2: str, vector_size: int = 30) -> float:
    """Cosine of two paragraph vectors learned on the fly.

    Optional: needs ``gensim`` and ``nltk``.
    """
    try:
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        from nltk.tokenize import word_tokenize
    except Exception as exc:  # pragma: no cover - optional path
        raise RuntimeError(
            "doc2vec_cosine needs 'gensim' and 'nltk' "
            "(pip install codesim[learned])"
        ) from exc

    tokens1 = word_tokenize(code1.lower())
    tokens2 = word_tokenize(code2.lower())
    tagged = [
        TaggedDocument(words=tokens1, tags=["a"]),
        TaggedDocument(words=tokens2, tags=["b"]),
    ]
    model = Doc2Vec(tagged, vector_size=vector_size, window=5, min_count=1, epochs=100)
    return float(model.dv.similarity("a", "b"))


def perceptual_hash(code1: str, code2: str) -> float:
    """Perceptual-hash agreement over rendered images of the two snippets.

    Optional: needs ``Pillow`` and ``imagehash``.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import imagehash
    except Exception as exc:  # pragma: no cover - optional path
        raise RuntimeError(
            "perceptual_hash needs 'Pillow' and 'imagehash' "
            "(pip install codesim[imaging])"
        ) from exc

    def render(code: str):
        img = Image.new("RGB", (300, 100), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), code, fill="black", font=ImageFont.load_default())
        return img

    h1 = imagehash.phash(render(code1))
    h2 = imagehash.phash(render(code2))
    bits = len(h1.hash) ** 2
    return 1.0 - (h1 - h2) / bits


_CLASS = re.compile(r"class\s+(\w+)\s*\{")


def execution_output(code1: str, code2: str, stdin: str = "") -> float:
    """Behavioral score: compile and run both snippets, compare their output.

    Optional: needs a working ``javac``/``java`` toolchain on the path. This
    measure carries real side effects, so it is disabled unless explicitly
    requested.
    """
    import subprocess
    import tempfile
    import os
    from difflib import SequenceMatcher

    def run(code: str) -> str:
        match = _CLASS.search(code)
        if not match:
            return ""
        name = match.group(1)
        with tempfile.TemporaryDirectory() as work:
            src = os.path.join(work, name + ".java")
            with open(src, "w") as fh:
                fh.write(code)
            try:
                subprocess.run(["javac", src], cwd=work, check=True,
                               capture_output=True, timeout=20)
                proc = subprocess.run(["java", name], cwd=work, input=stdin,
                                      capture_output=True, text=True, timeout=20)
                return proc.stdout.strip()
            except Exception:
                return ""

    out1, out2 = run(code1), run(code2)
    if not out1 and not out2:
        return 0.0
    return SequenceMatcher(None, out1, out2).ratio()
