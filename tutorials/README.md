# Tutorials — Learning Code Clone Detection by Doing

A hands-on, classroom-ready introduction to **source code clone detection**,
**code similarity**, and **ensemble learning**, built on top of the
`ensemble-codesim` research repository and its paper:

> Jorge Martinez-Gil. *Advanced Detection of Source Code Clones via an Ensemble
> of Unsupervised Similarity Measures.* SWQD 2025, Springer LNBIP 544, pp. 72–90.
> [DOI](https://doi.org/10.1007/978-3-031-89277-6_5) · [arXiv:2405.02095](https://arxiv.org/abs/2405.02095)

Everything here runs with **Python + NumPy only** — no GPU, no model downloads,
no internet. `matplotlib` and `scikit-learn` are *optional* (used for an extra
plot and an extra ensemble; the notebooks degrade gracefully without them).

---

## Who is this for?

- **Students** meeting code clones, similarity, or ensembles for the first time.
- **Educators** teaching software maintenance, empirical software engineering,
  code intelligence, or applied machine learning, who want a small,
  reproducible, interpretable case study.
- **Researchers** who want a 20-minute, runnable tour of the repository before
  using it as a baseline.

No prior knowledge of clone detection is assumed. Comfort with basic Python and
high-school-level statistics is enough.

---

## Learning path

Work through the notebooks in order (each is self-contained and ~15 minutes):

| # | Notebook | You will learn to … |
|---|----------|---------------------|
| 1 | [`notebooks/01_what_are_code_clones.ipynb`](notebooks/01_what_are_code_clones.ipynb) | Define Type-1/2/3/4 clones, recognise hard negatives, compute a first similarity score |
| 2 | [`notebooks/02_similarity_measures.ipynb`](notebooks/02_similarity_measures.ipynb) | Tour the measure families, find each one's blind spot, quantify complementarity |
| 3 | [`notebooks/03_ensemble_and_interpretability.ipynb`](notebooks/03_ensemble_and_interpretability.ipynb) | Combine measures, evaluate on the repo's **real** data, and **explain** a decision |

Then deepen with:

- **[`lecture_notes.md`](lecture_notes.md)** — a self-contained reading (slide-ready)
  covering the theory behind all three notebooks, with references.
- **[`exercises.md`](exercises.md)** — graded exercises (beginner → mini-project)
  with worked solutions.

---

## Files in this folder

```
tutorials/
├── README.md                 # you are here
├── lecture_notes.md          # lecture-ready reading + references
├── exercises.md              # exercises with solutions
├── codesim_edu.py            # 8 transparent similarity measures (stdlib + NumPy)
├── eval_edu.py               # metrics + ensembles (ROC-AUC, F1, MCC, logreg) in pure NumPy
├── examples/
│   ├── clone_pairs.py        # curated, labelled Type-1..4 + hard-negative pairs
│   └── clone_pairs.json      # the same examples, machine-readable
└── notebooks/
    ├── 01_what_are_code_clones.ipynb
    ├── 02_similarity_measures.ipynb
    └── 03_ensemble_and_interpretability.ipynb
```

`codesim_edu.py` is the **teaching** companion to the repository's full
`similarity/` package (21 measures, some requiring PyTorch / Transformers /
Pygments). The educational subset trades coverage for the ability to read every
line and run anywhere.

---

## How to run

```bash
# from the repository root
pip install numpy            # required
pip install matplotlib scikit-learn jupyter   # optional but recommended

jupyter notebook tutorials/notebooks/
```

You can also execute the example/measure modules directly:

```bash
python tutorials/examples/clone_pairs.py   # (re)generate clone_pairs.json
python tutorials/codesim_edu.py            # smoke-test the measures
python tutorials/eval_edu.py               # smoke-test the metrics
```

No Jupyter? Every notebook is plain Python under the hood — you can copy cells
into a script. The data they read (`outputs/output-karnalim.txt`,
`outputs/output-bigclonebench.txt`) ships with the repository.

---

## A note on honesty

These tutorials **never fabricate results**. Notebook 3 computes its numbers live
from the feature files committed in `outputs/`, which were produced by running
the 21 production measures (`exec-bigclonebench.py`, `exec-karnalim.py`). Where
an ensemble does *not* beat the best single measure, the notebook says so and
explains why. That candor is itself a teaching point about empirical software
engineering.

---

## Citing

If these materials help your teaching or research, please cite the paper (see
the [main README](../README.md#-citation)) and link back to the repository.
