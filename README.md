# 🔍 Advanced Detection of Source Code Clones via an Ensemble of Unsupervised Similarity Measures

> **Ensemble learning meets code clone detection** — combining multiple unsupervised similarity measures for robust, interpretable, and efficient source code clone identification.

[![DOI](https://img.shields.io/badge/DOI-10.1007%2F978--3--031--89277--6__5-red.svg)](https://doi.org/10.1007/978-3-031-89277-6_5)
[![arXiv](https://img.shields.io/badge/arXiv-2405.02095-2ebc4f.svg)](https://arxiv.org/abs/2405.02095)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Citations](https://img.shields.io/badge/citations-8-blue)](https://scholar.google.com/citations?view_op=view_citation&hl=en&citation_for_view=X1pRUYcAAAAJ:4pF9x-cDGsoC)
[![Java](https://img.shields.io/badge/Java-85%25-orange.svg)](#)
[![Python](https://img.shields.io/badge/Python-15%25-blue.svg)](#)

---

## 📖 Table of Contents

1. [Abstract](#-abstract)
2. [Key Contributions](#-key-contributions)
3. [Architecture Overview](#-architecture-overview)
4. [Repository Structure](#-repository-structure)
5. [Installation](#-installation)
6. [Usage](#-usage)
7. [Tutorials & Educational Use](#-tutorials--educational-use)
8. [Results](#-results)
9. [Dependencies](#-dependencies)
10. [Citation](#-citation)
11. [Research Citing This Work](#-research-citing-this-work)
12. [Related Work](#-related-work)
13. [License](#-license)

---

## 🌍 Abstract

Accurately determining code similarity is crucial for many software development tasks, such as software maintenance and code duplicate identification. This research introduces an **ensemble learning approach** that combines multiple unsupervised similarity measures to improve the detection of source code clones.

Our method is:
- 📐 **Accurate** — outperforms individual similarity metrics by leveraging their complementary strengths
- ⚡ **Efficient** — designed to perform well on small datasets with reduced computational costs
- 🔎 **Interpretable** — provides transparency into how similarity decisions are made
- 🌱 **Sustainable** — low resource footprint compared to large transformer-based baselines

> **Published at:** [SWQD 2025 – 17th International Conference on Software Quality](https://link.springer.com/chapter/10.1007/978-3-031-89277-6_5), Munich, Germany. Springer LNBIP, Vol. 544, pp. 72–90.

---

## 🚀 Key Contributions

| Contribution | Description |
|---|---|
| **Ensemble Framework** | Novel combination of multiple unsupervised code similarity measures into a single robust predictor |
| **Benchmark Evaluation** | Rigorous evaluation on BigCloneBench and the Karnalim dataset |
| **Lightweight Design** | No pre-training or GPU required — runs on standard hardware |
| **Interpretability** | Each component similarity score is inspectable, enabling explainability |
| **Open Source** | Full implementation, datasets, and results publicly available |

---

## 🏗️ Architecture Overview

```
Source Code Pairs
       │
       ▼
┌─────────────────────────────────────────────┐
│           Similarity Measures               │
│  ┌──────────┐ ┌──────────┐ ┌─────────────┐ │
│  │ Token-   │ │  AST-    │ │  Semantic   │ │
│  │ Based    │ │  Based   │ │  Metrics    │ │
│  └──────────┘ └──────────┘ └─────────────┘ │
└─────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│         Ensemble Aggregation Layer          │
│    (Voting / Weighted Combination)          │
└─────────────────────────────────────────────┘
       │
       ▼
   Clone / Non-Clone Decision
```

---

## 📁 Repository Structure

```
ensemble-codesim/
├── baselines/              # Baseline models for comparison
├── datasets/               # BigCloneBench & Karnalim datasets
│   ├── BigCloneBench/
│   │   ├── data.jsonl      # Code snippets
│   │   ├── train.txt       # Clone pairs (train split)
│   │   ├── valid.txt       # Clone pairs (validation split)
│   │   └── test.txt        # Clone pairs (test split)
├── ensembles/              # Ensemble combination strategies
├── similarity/             # Individual similarity measure implementations
├── utils/                  # Helper utilities
├── outputs/                # Output files
├── results/                # Experimental results
├── exec-bigclonebench.py   # Run experiments on BigCloneBench
├── exec-karnalim.py        # Run experiments on Karnalim dataset
└── README.md
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/jorge-martinez-gil/ensemble-codesim.git
cd ensemble-codesim

# Install Python dependencies
pip install numpy scikit-learn tqdm
```

---

## 🖥️ Usage

### Running on BigCloneBench

```bash
python exec-bigclonebench.py
```

### Running on the Karnalim Dataset

```bash
python exec-karnalim.py
```

### Step-by-Step Workflow

**1. Load Code Snippets**
```python
code_snippets = load_code_snippets('datasets/BigCloneBench/data.jsonl')
```

**2. Prepare Dataset Splits**
```python
train_dataset = prepare_dataset('datasets/BigCloneBench/train.txt', tokenizer, code_snippets)
val_dataset   = prepare_dataset('datasets/BigCloneBench/valid.txt', tokenizer, code_snippets)
test_dataset  = prepare_dataset('datasets/BigCloneBench/test.txt',  tokenizer, code_snippets)
```

**3. Train and Evaluate**
```python
trainer.train()
test_results = trainer.evaluate(test_dataset)

print(f"Precision : {test_results['eval_precision']:.4f}")
print(f"Recall    : {test_results['eval_recall']:.4f}")
print(f"F1 Score  : {test_results['eval_f1']:.4f}")
```

---

## 🎓 Tutorials & Educational Use

New to code clone detection, or want a runnable tour before using this repo as a
baseline? The [`tutorials/`](tutorials/) folder is a self-contained, classroom-ready
course that runs with **Python + NumPy only** (no GPU, no model downloads):

| # | Notebook | Topic |
|---|----------|-------|
| 1 | [`01_what_are_code_clones`](tutorials/notebooks/01_what_are_code_clones.ipynb) | Type-1/2/3/4 clones, hard negatives, your first similarity score |
| 2 | [`02_similarity_measures`](tutorials/notebooks/02_similarity_measures.ipynb) | The measure families, each one's blind spot, measure complementarity |
| 3 | [`03_ensemble_and_interpretability`](tutorials/notebooks/03_ensemble_and_interpretability.ipynb) | Combining measures, evaluation on the repo's **real** data, explaining a decision |

Also included:

- [`lecture_notes.md`](tutorials/lecture_notes.md) — a slide-ready reading covering the theory, with references.
- [`exercises.md`](tutorials/exercises.md) — graded exercises (beginner → capstone) with worked solutions.
- [`examples/clone_pairs.py`](tutorials/examples/clone_pairs.py) — curated, labelled Type-1…4 and hard-negative pairs (Java + Python).
- [`codesim_edu.py`](tutorials/codesim_edu.py) / [`eval_edu.py`](tutorials/eval_edu.py) — transparent, dependency-light similarity measures and metrics (ROC-AUC, F1, MCC, ensembles) you can read end to end.

```bash
pip install numpy            # required
pip install jupyter matplotlib scikit-learn   # optional extras
jupyter notebook tutorials/notebooks/
```

> Notebook 3 computes every number **live** from the committed feature files in
> `outputs/` — nothing is fabricated, and where an ensemble does *not* beat the
> best single measure, the notebook says so and explains why.

Suitable for courses in software maintenance, empirical software engineering,
code intelligence, and applied machine learning. See the
[tutorials README](tutorials/README.md) for the full learning path.

---

## 📊 Results

The ensemble approach consistently outperforms individual similarity measures. Key highlights from the paper:

| Method | Precision | Recall | F1 Score |
|---|---|---|---|
| Single best measure | — | — | — |
| **Ensemble (ours)** | **↑** | **↑** | **↑** |
| GraphCodeBERT baseline | high | high | high (GPU needed) |

> 💡 Our ensemble achieves **competitive F1 scores** against transformer-based models while requiring **no GPU** and being **fully interpretable**. See the full results in the paper or the `results/` directory.

---

## 📦 Dependencies

```
Python  ≥ 3.x
Java    ≥ 8
NumPy
scikit-learn
tqdm
```

---

## 📚 Citation

If this work is useful for your research, please cite:

```bibtex
@inproceedings{MartinezGil2025,
  author    = {Jorge Martinez-Gil},
  editor    = {Jannik Fischbach and
               Rudolf Ramler and
               Dietmar Winkler and
               Johannes Bergsmann},
  title     = {Advanced Detection of Source Code Clones via an Ensemble of Unsupervised
               Similarity Measures},
  booktitle = {Balancing Software Innovation and Regulatory Compliance - 17th International
               Conference on Software Quality, {SWQD} 2025, Munich, Germany,
               May 20--22, 2025, Proceedings},
  series    = {Lecture Notes in Business Information Processing},
  volume    = {544},
  pages     = {72--90},
  year      = {2025},
  publisher = {Springer},
  doi       = {10.1007/978-3-031-89277-6\_5},
  url       = {https://link.springer.com/chapter/10.1007/978-3-031-89277-6\_5}
}
```

> 📄 A preprint is freely available on [arXiv:2405.02095](https://arxiv.org/abs/2405.02095).

---

## 🗂️ Research Citing This Work

This work has already been cited by the following publications:

1. **[An Enhanced Transformer-Based Framework for Interpretable Code Clone Detection](https://www.sciencedirect.com/science/article/pii/S0164121225000159)**
   - **Authors:** M. Nashaat, R. Amin, A.H. Eid, R.F. Abdel-Kader
   - **Venue:** *Journal of Systems and Software*, Elsevier, 2025
   - Proposes an enhanced transformer-based framework for interpretable code clone detection.

2. **[A Novel Method for Code Clone Detection Based on Minimally Random Kernel Convolutional Transform](https://ieeexplore.ieee.org/abstract/document/10731684/)**
   - **Author:** M. Abdelkader
   - **Venue:** *IEEE Access*, 2024
   - Introduces a novel convolution-transform-based approach for code clone detection.

3. **[Improving Source Code Similarity Detection Through GraphCodeBERT and Integration of Additional Features](https://arxiv.org/abs/2408.08903)**
   - **Author:** J. Martinez-Gil
   - **Venue:** *arXiv preprint*, 2024
   - Extends the present work by integrating additional output features with GraphCodeBERT.

---

## 🔗 Related Work

If you are interested in code clone detection and source code similarity, you may also find these resources relevant:

- [BigCloneBench](https://github.com/jeffsvajlenko/BigCloneBench) — the main benchmark dataset used in this work
- [GraphCodeBERT](https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT) — a strong transformer baseline for code understanding
- [arXiv:2405.02095](https://arxiv.org/abs/2405.02095) — preprint version of this paper

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>If you find this repository useful, please consider ⭐ starring it and citing the paper — it helps the research reach a wider audience!</i>
</p>
