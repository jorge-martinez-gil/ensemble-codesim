# Exercises — Code Clone Detection

Graded exercises that build on the three tutorial notebooks and the
`codesim_edu` / `eval_edu` modules. Each exercise lists the notebook it follows,
a difficulty, and a collapsible worked solution. Try it before peeking.

All solutions assume you have run the standard setup so the modules import:

```python
import sys; sys.path.insert(0, "tutorials"); sys.path.insert(0, "tutorials/examples")
import numpy as np
from codesim_edu import MEASURES, all_scores
from clone_pairs import load_pairs
import eval_edu as E
```

---

## A. Warm-up (after Notebook 1)

### A1 — Spot the clone type · *beginner*
For each of the following, name the clone type (1–4) or "not a clone":
1. The same method with two variables renamed.
2. A method reimplemented from a `for` loop to recursion.
3. An identical method with an extra logging line inserted.
4. Two methods that both loop over an array but compute different results.

<details><summary>Solution</summary>

1. Type-2 (renamed). 2. Type-4 (semantic). 3. Type-3 (gapped).
4. Not a clone (a hard negative).
</details>

### A2 — Predict before you run · *beginner*
For the `python_type2_factorial_renamed` pair, predict whether `jaccard_tokens`
or `normalized_jaccard` will be higher, then check.

<details><summary>Solution</summary>

`normalized_jaccard` is higher: renaming lowers raw token overlap, but
normalisation collapses both snippets to the same `ID`-placeholder form.

```python
by_id = {p["id"]: p for p in load_pairs()}
p = by_id["python_type2_factorial_renamed"]
from codesim_edu import jaccard_tokens, normalized_jaccard
print(jaccard_tokens(p["code_a"], p["code_b"]),
      normalized_jaccard(p["code_a"], p["code_b"]))
```
</details>

### A3 — Add your own pair · *beginner*
Add a Type-3 clone (in any language) to `examples/clone_pairs.py`, regenerate the
JSON, and confirm it loads.

<details><summary>Solution sketch</summary>

Append a dict with `clone_type: 3`, `label: 1`, and two snippets differing by an
inserted statement; run `python tutorials/examples/clone_pairs.py`; verify with
`len(load_pairs())`.
</details>

---

## B. Measures (after Notebook 2)

### B1 — Build the score table · *beginner*
Print every measure's score for the `crosslang_type4_gcd` pair. Which measures,
if any, recognise this cross-language semantic clone?

<details><summary>Solution</summary>

```python
p = {x["id"]: x for x in load_pairs()}["crosslang_type4_gcd"]
for k, v in sorted(all_scores(p["code_a"], p["code_b"]).items(), key=lambda kv: -kv[1]):
    print(f"{k:20s} {v:.2f}")
```
Most lexical measures score it low; the structural/normalised measures do
best but are still modest — Type-4 across languages is genuinely hard.
</details>

### B2 — Separation power · *intermediate*
For each measure, compute mean score on clones minus mean score on non-clones
across all example pairs. Rank the measures. Which is the best single
discriminator *on these examples*?

<details><summary>Solution</summary>

```python
pairs = load_pairs()
names = list(MEASURES)
M = np.array([[all_scores(p["code_a"], p["code_b"])[m] for m in names] for p in pairs])
y = np.array([p["label"] for p in pairs])
sep = [(m, M[y==1, j].mean() - M[y==0, j].mean()) for j, m in enumerate(names)]
for m, d in sorted(sep, key=lambda r: -r[1]):
    print(f"{m:20s} {d:+.2f}")
```
(Expect a rename-robust or structural measure near the top; `metrics_cosine`
near zero because it cannot separate these tiny snippets.)
</details>

### B3 — Most complementary pair · *intermediate*
Find the two least-correlated measures across the example pairs. Argue why
combining them could help.

<details><summary>Solution</summary>

```python
C = np.corrcoef(M.T)
best = min(((names[a], names[b], C[a, b])
            for a in range(len(names)) for b in range(a+1, len(names))),
           key=lambda r: r[2])
print(best)
```
Low correlation ⇒ the two measures fire on different inputs ⇒ together they
cover more clone types and disagree on different hard negatives.
</details>

---

## C. Ensembles & evaluation (after Notebook 3)

### C1 — Per-measure leaderboard · *intermediate*
On Karnalim, rank the 21 production measures by ROC-AUC. Is the ROC-AUC ranking
the same as the best-F1 ranking? Why might they differ?

<details><summary>Solution</summary>

```python
X, y, names = E.load_feature_file("outputs/output-karnalim.txt")
rows = [(n, E.roc_auc(X[:, i], y), E.best_threshold_f1(X[:, i], y)["f1"])
        for i, n in enumerate(names)]
for n, a, f in sorted(rows, key=lambda r: -r[1]):
    print(f"{n:10s} AUC={a:.3f} F1={f:.3f}")
```
They can differ: ROC-AUC measures *ranking* quality across all thresholds, while
best-F1 rewards one operating point. A measure can rank well yet have no single
threshold that yields high F1 (or vice-versa), especially under class imbalance.
</details>

### C2 — Honest ensemble comparison · *intermediate*
Using a repeated train/test split, compare the best single measure, the mean
ensemble, and the logistic-regression ensemble on Karnalim **and** on a
BigCloneBench subset. State for each dataset which wins, and explain the
difference.

<details><summary>Solution</summary>

Use the `evaluate()` function from Notebook 3. Expect: on Karnalim the best
single measure is hard to beat; on BigCloneBench the learned ensemble wins. The
difference is driven by class balance and which measures happen to be strong:
the ensemble pays off most when no single measure dominates.
</details>

### C3 — Add MCC · *advanced*
Extend the evaluation to also report **MCC** at the F1-optimal threshold. Does
the system ranking change relative to F1? Which metric would you trust more on
BigCloneBench (≈14% positive) and why?

<details><summary>Solution</summary>

`E.metrics_at(scores, labels, thr)` already returns `mcc`. Report it alongside
`f1`. Under heavy imbalance, MCC is more informative because it accounts for all
four confusion-matrix cells; F1 ignores true negatives. Rankings can shift when a
system trades many false positives for a few extra true positives.
</details>

### C4 — Reproduce the paper's ensemble · *advanced*
With scikit-learn installed, train a Random Forest on the 21 measures with
5-fold cross-validation and compare its F1 to the logistic-regression ensemble.
Then inspect feature importances — which measures does the forest rely on?

<details><summary>Solution sketch</summary>

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
print(cross_val_score(clf, X, y, cv=5, scoring="f1").mean())
clf.fit(X, y)
for n, imp in sorted(zip(names, clf.feature_importances_), key=lambda r: -r[1]):
    print(f"{n:10s} {imp:.3f}")
```
Compare with `utils/rf-with-importance.py` in the repository, which does the same
with a hyper-parameter search and an importance plot.
</details>

---

## D. Mini-project · *capstone*

**Build and evaluate your own clone detector.**

1. Implement one new similarity measure in the style of `codesim_edu.py`
   (e.g. a Winnowing fingerprint overlap, or a simple AST-edit distance).
2. Add it to a copy of the measure set and regenerate scores for the example
   pairs and (optionally) for a sample of the BigCloneBench pairs.
3. Build a feature/score file in the repository's JSONL format
   (`[score_1, ..., score_k, label]` per line).
4. Evaluate single measures vs. an ensemble with `eval_edu`, using repeated
   splits; report **F1 and MCC as mean ± std**, plus a **per-clone-type**
   breakdown on the curated examples.
5. Write a one-paragraph, evidence-backed conclusion: did your measure add
   complementary value, or was it redundant with existing ones?

Deliverable: a short notebook or report with reproducible numbers and an
`explain_pair`-style rationale for at least two decisions (one clone, one hard
negative).

> Tip: to plug into the existing evaluation, match the column order in
> `eval_edu.MEASURE_NAMES_21` or simply define your own and keep the label last.
