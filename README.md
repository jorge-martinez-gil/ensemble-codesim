# 🔍 Advanced Detection of Source Code Clones via an Ensemble of Unsupervised Similarity Measures

> **Ensemble learning meets code clone detection.** More than twenty unsupervised similarity measures, combined into one inspectable predictor, packaged as the installable **CodeSim** toolkit, and evaluated reproducibly on BigCloneBench.

[![DOI](https://img.shields.io/badge/DOI-10.1007%2F978--3--031--89277--6__5-red.svg)](https://doi.org/10.1007/978-3-031-89277-6_5)
[![arXiv](https://img.shields.io/badge/arXiv-2405.02095-b31b1b.svg)](https://arxiv.org/abs/2405.02095)
[![Zenodo](https://img.shields.io/badge/Zenodo-DOI%20pending-blue.svg)](https://zenodo.org)
[![tests](https://github.com/jorge-martinez-gil/ensemble-codesim/actions/workflows/tests.yml/badge.svg)](https://github.com/jorge-martinez-gil/ensemble-codesim/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](#)

---

## 📌 Cite this work

If this repository helps your research, please cite the paper:

```bibtex
@inproceedings{MartinezGil2025,
  author    = {Jorge Martinez-Gil},
  title     = {Advanced Detection of Source Code Clones via an Ensemble of
               Unsupervised Similarity Measures},
  booktitle = {Software Quality (SWQD 2025)},
  series    = {Lecture Notes in Business Information Processing},
  volume    = {544},
  pages     = {72--90},
  year      = {2025},
  publisher = {Springer},
  doi       = {10.1007/978-3-031-89277-6\_5}
}
```

GitHub also shows a **"Cite this repository"** button (top-right, from
[`CITATION.cff`](CITATION.cff)) for one-click import into your reference manager.
A preprint is on [arXiv:2405.02095](https://arxiv.org/abs/2405.02095).

---

## 📖 Contents

1. [Overview](#-overview)
2. [The CodeSim toolkit](#-the-codesim-toolkit)
3. [Install](#-install)
4. [Quick start](#-quick-start)
5. [Similarity measures](#-similarity-measures)
6. [Reproducible results](#-reproducible-results)
7. [Tutorials](#-tutorials)
8. [Research citing this work](#-research-citing-this-work)
9. [How to cite the software](#-how-to-cite-the-software)
10. [License](#-license)

---

## 🌍 Overview

Judging how similar two pieces of source code are sits at the root of many
software tasks: clone detection, plagiarism checking, vulnerability propagation,
license auditing, and code search. Decades of research have produced a large,
scattered set of similarity measures, from character edit distance to program
dependence graph matching to embeddings from pre-trained models. No single
measure wins across the four clone types, because each rests on a different view
of what a program is, and each has a blind spot.

This project turns that variety into an advantage. It scores a code pair with
many unsupervised measures at once, forming a feature vector in measure space,
then learns to combine those scores into one decision. The result is:

- 📐 **Accurate.** A learned ensemble far outperforms the best single measure.
- ⚡ **Efficient.** The base measures run on standard hardware with no GPU.
- 🔎 **Inspectable.** Feature importances show which measures drove a decision.
- 🌱 **Light.** The core needs only NumPy and scikit-learn.

> **Published at** [SWQD 2025 (17th Int. Conf. on Software Quality)](https://link.springer.com/chapter/10.1007/978-3-031-89277-6_5), Springer LNBIP vol. 544, pp. 72-90.

---

## 🧰 The CodeSim toolkit

`CodeSim` is the reusable core of this project: one interface over every measure,
plus ensembles, evaluation metrics, dataset loaders, and a command-line tool. A
similarity measure is treated as a function that returns a score in `[0, 1]`, so
a Levenshtein distance and a transformer embedding are called the same way.

```
codesim/
├── core.py         # similarity(), feature_vector(), feature_matrix()
├── registry.py     # every measure with its family and metadata
├── ensemble.py     # unsupervised and supervised (inspectable) ensembles
├── evaluation.py   # precision, recall, F1, MCC, ROC-AUC, threshold tuning
├── datasets.py     # BigCloneBench and folder-corpus loaders
├── cli.py          # the `codesim` command
└── measures/       # textual, token, structural, learned, behavioral
```

The original study scripts (`exec-bigclonebench.py`, `exec-karnalim.py`, the
`similarity/` and `ensembles/` modules) remain in place; `CodeSim` gives their
measures a clean, tested, installable home.

---

## ⚙️ Install

```bash
git clone https://github.com/jorge-martinez-gil/ensemble-codesim.git
cd ensemble-codesim

pip install -e .                 # base: 16 measures, ensembles, evaluation
pip install -e ".[structural]"   # adds the program-dependence-graph measure
pip install -e ".[learned]"      # adds transformer and paragraph-vector measures
pip install -e ".[all]"          # everything
```

The base install needs only NumPy and scikit-learn. Heavier measures load their
backend on demand and report a clear message when a dependency is absent.

---

## 🖥️ Quick start

```python
import codesim

# One pair, one measure.
codesim.similarity(code_a, code_b, measure="jaccard")     # -> 0.83

# The full feature vector: one score per base measure.
from codesim.core import feature_vector
feature_vector(code_a, code_b)          # numpy array, length 16

# A supervised ensemble that learns the combiner and reports what it used.
from codesim.ensemble import SupervisedEnsemble
ens = SupervisedEnsemble(model="random_forest").fit_pairs(pairs, labels)
ens.importances()          # {"fcall": 0.16, "jaccard": 0.07, ...}
```

From the command line:

```bash
codesim list                                    # measures grouped per family
codesim compare a.java b.java --measure tfidf   # score two files
codesim vector a.java b.java                     # score with every base measure
codesim evaluate --data data.jsonl --pairs test.txt --measure tfidf --tune
```

---

## 🧩 Similarity measures

Measures are organized on two axes: the program representation and the
comparison operator.

| Family | Representation | Operator | Measures |
|---|---|---|---|
| Textual | characters, lines, fingerprints | edit distance, LCS, string tiling, fingerprint overlap | levenshtein, lcs, seqratio, fuzzy, linediff, ngram, winnowing, rollinghash, gst |
| Token | token bags, sets, tf-idf | set overlap, cosine | bow, tfidf, jaccard, tokentype, fcall |
| Structural | metric vectors, adjacency and dependence graphs | vector cosine, graph edit distance | metrics, graph, pdg |
| Learned | pre-trained or paragraph embeddings | cosine over learned vectors | codebert, doc2vec, image |
| Behavioral | runtime input/output | output agreement | execution |

Run `codesim list` to see each measure with the representation it compares, the
operator it applies, the clone types it targets, and any optional dependency.

---

## 📊 Reproducible results

The numbers below come from `paper/analyze.py`, computed on the committed
BigCloneBench feature file after removing duplicate rows that would otherwise
leak between training and test (14,672 unique pairs: 2,172 clones, 12,500
non-clones). Rerun the script to regenerate every value and figure.

| Method | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Best single measure (function-call overlap) | 0.53 | 0.51 | 0.52 | 0.79 |
| Naive mean over all measures | | | 0.36 | 0.66 |
| Selective mean (top six on ROC-AUC) | | | 0.47 | 0.76 |
| Gradient boosting (default) | 0.86 | 0.50 | 0.63 | 0.91 |
| **Random forest (held-out split)** | **0.99** | **0.90** | **0.94** | **0.99** |
| Random forest (5-fold cross-validation) | | | 0.96 ± 0.01 | 0.997 |

Two findings stand out. A learned ensemble nearly doubles the F1 of the best
single measure. And a naive average of all measures loses, because weak,
near-saturated measures dilute the signal; the value of a learned combiner is
that it can down-weight them. The measures carry a mean pairwise correlation of
only 0.25, which is the quantitative reason their combination pays off.

---

## 🎓 Tutorials

The [`tutorials/`](tutorials/) folder is a runnable, classroom-ready course that
needs only Python and NumPy:

| # | Notebook | Topic |
|---|---|---|
| 1 | `01_what_are_code_clones` | Type-1 to Type-4 clones, hard negatives, a first score |
| 2 | `02_similarity_measures` | measure families, each one's blind spot, complementarity |
| 3 | `03_ensemble_and_interpretability` | combining measures, evaluation on real data, explaining a decision |

Also included: slide-ready lecture notes, graded exercises with solutions, and a
labelled set of Type-1 to Type-4 clone pairs.

---

## 🗂️ Research citing this work

1. **[An Enhanced Transformer-Based Framework for Interpretable Code Clone Detection](https://www.sciencedirect.com/science/article/pii/S0164121225000159)**. M. Nashaat, R. Amin, A.H. Eid, R.F. Abdel-Kader. *Journal of Systems and Software*, Elsevier, 2025.
2. **[A Novel Method for Code Clone Detection Based on Minimally Random Kernel Convolutional Transform](https://ieeexplore.ieee.org/abstract/document/10731684/)**. M. Abdelkader. *IEEE Access*, 2024.
3. **[Improving Source Code Similarity Detection Through GraphCodeBERT and Integration of Additional Features](https://arxiv.org/abs/2408.08903)**. J. Martinez-Gil. *arXiv preprint*, 2024.

---

## 🔗 How to cite the software

Beyond the paper, you can cite the software release itself. Once a Zenodo DOI is
minted for a tagged release (see [`docs/GROWTH_PLAYBOOK.md`](docs/GROWTH_PLAYBOOK.md)),
add it here and swap the Zenodo badge above for the real DOI badge. Until then,
the [`CITATION.cff`](CITATION.cff) file and the paper citation above are the way
to credit this work.

---

## 📄 License

MIT. See [LICENSE](LICENSE).

<p align="center">
<i>If you find this useful, a ⭐ helps others discover it, and a citation helps the research reach a wider audience.</i>
</p>
