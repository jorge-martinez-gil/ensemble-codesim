# Lecture Notes — Interpretable, Lightweight Code Clone Detection

*Companion reading for the `ensemble-codesim` tutorial notebooks. Suitable as a
single 90-minute lecture or two shorter sessions. Each section maps to runnable
code so students can move between theory and practice.*

---

## 1. The problem: why detect code clones?

A **code clone** is a fragment of source code that is identical or similar to
another fragment. Empirical studies of large systems repeatedly report that a
substantial fraction of code — commonly cited figures range from roughly 7% to
over 20% — is cloned, mostly through copy-paste-and-adapt programming.

Clones are not inherently bad, but they matter for several practical tasks:

- **Software maintenance.** A defect discovered in one fragment likely exists in
  its clones. If a maintainer fixes only the copy they happened to find,
  the bug persists elsewhere. Clone detection surfaces the siblings.
- **Refactoring.** Clusters of clones mark opportunities to extract a shared
  abstraction (a method, a class, a template).
- **Plagiarism and academic integrity.** Source-code plagiarism detectors are
  clone detectors with a different label. The `IR-Plag` (Karnalim) dataset in
  this repository comes from exactly this setting.
- **Licensing and provenance.** Copied code can carry license obligations;
  detecting it supports compliance and auditing.
- **Code search and reuse.** "Find code similar to this snippet" is clone
  detection viewed as retrieval.

> **Run it:** `notebooks/01_what_are_code_clones.ipynb`.

---

## 2. A taxonomy of clones

The community taxonomy (Bellon et al. 2007; Roy & Cordy 2007) orders clones by
how far they drift from the original:

| Type | Name | What may differ | Difficulty |
|------|------|-----------------|------------|
| **1** | Exact | Whitespace, layout, comments | Easy |
| **2** | Renamed / parameterised | + identifiers, literals, types | Easy–moderate |
| **3** | Near-miss / gapped | + added / removed / changed statements | Moderate–hard |
| **4** | Semantic | Same behaviour, different implementation | Hard |

The jump from Type-2 to Type-3 introduces *gaps* (edits), which break exact
matching. The jump to Type-4 is qualitative: two snippets may compute the same
function with **no shared surface structure** (iteration vs. recursion; a loop
vs. the closed-form `n*(n+1)/2`). Type-4 generally requires reasoning about
*behaviour*, not text, and is where most detectors lose ground.

A well-designed evaluation reports performance **per clone type**, because an
aggregate score can hide total failure on Type-4 behind easy Type-1 wins.

---

## 3. Similarity measures and their families

A **similarity measure** maps a pair of snippets to a number in `[0, 1]`. The
families differ in *what representation* they compare:

- **Token-based.** Treat code as a bag, set, or sequence of lexical tokens.
  Examples: token Jaccard, n-gram cosine, Winnowing fingerprints. Cheap and
  strong on Type-1/2; blind to deep restructuring.
- **Sequence / edit-based.** Align two token (or character) streams: longest
  common subsequence, edit distance, `difflib` ratio, greedy string tiling
  (Rabin–Karp). Tolerant of small gaps (Type-3).
- **Syntactic / AST-based.** Compare abstract-syntax-tree structure or token
  *kinds* rather than names. Naturally robust to renaming (Type-2).
- **Graph-based.** Compare program-dependence or control-flow graphs. Closer to
  semantics, but expensive (graph matching is costly).
- **Metric-based.** Summarise each snippet as a vector of software metrics
  (size, control-flow counts, nesting) and compare the vectors. Fast, but coarse
  — easily fooled by structurally-similar-but-different code.
- **Embedding-based.** Encode code with a learned model (CodeBERT,
  GraphCodeBERT, UniXcoder, CodeT5) and compare embeddings. Strong, including on
  some Type-4 cases, but heavyweight (GPU, pre-training) and **opaque**.

A central empirical fact, demonstrated in Notebook 2: **every measure has a
blind spot.** Token measures collapse on renamed clones; lexical measures fail
on Type-4; metric measures cannot separate look-alikes; and *non*-clones with
shared boilerplate ("hard negatives") fool overlap measures into false
positives.

> **Run it:** `notebooks/02_similarity_measures.ipynb`. The educational measures
> live in `codesim_edu.py`; the full 21 are in the repository's `similarity/`.

### Normalisation: a small idea with a big payoff

Replacing user identifiers with a placeholder (`addUp(values)` → `ID(ID)`)
turns a Type-2 clone back into a Type-1 match. This single transformation
recovers renamed clones for almost any token measure and is the intuition behind
AST-kind measures. Notebook 1 shows raw vs. normalised Jaccard side by side.

---

## 4. Why ensembles? Complementarity over dominance

If one measure dominated on every clone type and every dataset, we would simply
use it. Two empirical facts rule that out:

1. **No measure is best everywhere.** The strongest single measure differs by
   dataset (Notebook 3 finds different winners on Karnalim vs. BigCloneBench),
   and you do not know the winner in advance.
2. **Measures make *different* mistakes.** Their errors are only weakly
   correlated (Notebook 2 computes the correlation matrix and finds highly
   complementary pairs).

These are exactly the conditions under which **ensemble learning** helps:
combining diverse, individually-imperfect predictors yields a predictor that is
more robust than any member. Three combination strategies, in increasing power:

- **Mean / rank aggregation** — normalise and average the measures. No training
  to combine; one threshold to decide. Fully transparent.
- **Voting** — each measure casts a binary vote at its own threshold; the score
  is the fraction voting "clone".
- **Learned combiners** — fit a model on the measure vector. A linear
  **logistic regression** (Notebook 3, pure NumPy) already learns useful
  weights; **Random Forest** and **XGBoost** (the paper's choice, in
  `ensembles/`) capture *interactions* a linear model cannot.

> **Honesty matters.** On a small, clone-heavy dataset the best single measure
> can beat a naive average. The ensemble's value is **robustness without an
> oracle**: it is consistently competitive-or-better across datasets, and the
> learned versions win where it counts (e.g. the larger, imbalanced
> BigCloneBench). Notebook 3 reports both outcomes rather than cherry-picking.

### When is this preferable to a transformer baseline?

Transformer code models (CodeBERT and friends) are powerful but require
pre-training, a GPU, and yield opaque decisions. An ensemble of lightweight
unsupervised measures is attractive when you need: **no GPU**, **small-data**
operation, **fast** scoring, and — above all — **interpretability** (Section 6).
It is a complement to, not a wholesale replacement for, learned code models.

---

## 5. Evaluating a clone detector properly

Turning scores into a verdict requires a **threshold**; reporting a single
accuracy number is rarely enough.

- **Confusion-matrix metrics.** Precision (purity of flagged pairs), recall
  (coverage of true clones), and their harmonic mean **F1**.
- **MCC** (Matthews correlation) — balanced even under heavy class imbalance,
  where accuracy is misleading (a detector that says "never a clone" can score
  86% accuracy on BigCloneBench while finding nothing).
- **Threshold-free ranking.** **ROC-AUC** and **PR-AUC** summarise quality
  across all thresholds; PR-AUC is the more informative under imbalance.
- **Per-type breakdown.** Report Type-1/2/3/4 separately.
- **Robustness probes.** Measure score stability under formatting changes,
  variable renaming, dead-code insertion, and statement reordering.
- **Cost.** Runtime, memory, training-data requirement, and an energy/carbon
  proxy — central claims when the selling point is being *lightweight*.

Two methodological musts:

1. **Train/test separation.** Select thresholds and fit combiners on a training
   split; report on held-out data. Any threshold chosen on the test set inflates
   results.
2. **Report variance.** Repeat over several random splits or folds and report
   **mean ± standard deviation**, not a single lucky run. Where feasible, add a
   significance test or confidence interval.

> **Run it:** `eval_edu.py` implements ROC-AUC, F1, MCC, and best-threshold
> selection in pure NumPy; Notebook 3 uses them with repeated splits.

---

## 6. Interpretability

An ensemble of explicit measures is **auditable** in a way a single embedding
score is not. For each decision we can surface:

- the **contribution** of every measure (its individual score);
- where the measures **agree** and **disagree**;
- an overall **confidence** (e.g. the mean score) and a **spread** (disagreement);
- an **uncertainty warning** when the measures are split or the score sits near
  the threshold — the cases a human reviewer should examine;
- optionally, **token-level or structural evidence** for why a measure fired.

Notebook 3 builds a compact `explain_pair` that emits such a report and can be
exported as Markdown, JSON, or HTML. This is the practical meaning of
"interpretable clone detection": not just a number, but a defensible rationale.

---

## 7. Reproducibility in software-engineering experiments

The tutorials double as a small case study in reproducible empirical SE:

- **Deterministic data provenance.** The committed `outputs/*.txt` files record
  exactly the per-pair measure vectors used; the scripts that produced them are
  in the repository.
- **One representation, many methods.** Storing the 21 measure scores per pair
  lets anyone re-run *any* downstream ensemble without recomputing features.
- **Fixed seeds, reported variance.** Splits are seeded; results are summarised
  over repetitions.
- **No hidden state.** The educational stack is pure NumPy + standard library,
  so a result does not depend on a particular GPU, driver, or model checkpoint.

A useful classroom exercise is to have students reproduce a number from the
notebook, then perturb one choice (threshold rule, split fraction, measure
subset) and explain the change.

---

## 8. References and further reading

- S. Bellon, R. Koschke, G. Antoniol, J. Krinke, E. Merlo. *Comparison and
  evaluation of clone detection tools.* IEEE TSE, 2007.
- C. K. Roy, J. R. Cordy. *A survey on software clone detection research.*
  Tech. report, Queen's University, 2007.
- J. Svajlenko et al. *Towards a big data curated benchmark of inter-project code
  clones (BigCloneBench).* ICSME, 2014.
- O. Karnalim et al. Source-code plagiarism datasets (the `IR-Plag` collection).
- Z. Feng et al. *CodeBERT: A pre-trained model for programming and natural
  languages.* EMNLP Findings, 2020.
- D. Guo et al. *GraphCodeBERT: Pre-training code representations with data flow.*
  ICLR, 2021.
- **This work:** J. Martinez-Gil. *Advanced Detection of Source Code Clones via
  an Ensemble of Unsupervised Similarity Measures.* SWQD 2025, Springer LNBIP
  544, pp. 72–90. DOI: 10.1007/978-3-031-89277-6_5 · arXiv:2405.02095.

---

*Next: put the theory to work in [`exercises.md`](exercises.md).*
