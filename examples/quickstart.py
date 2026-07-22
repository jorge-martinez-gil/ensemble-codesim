"""A short tour of CodeSim: single measures, the full feature vector, an
unsupervised ensemble, and a small supervised ensemble with importances.

Run with: python examples/quickstart.py
"""

import codesim
from codesim.core import available, feature_vector
from codesim.ensemble import SupervisedEnsemble, UnsupervisedEnsemble

ORIGINAL = """
public int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; i++) result *= i;
    return result;
}
"""

RENAMED = """
public int fact(int m) {
    int acc = 1;
    for (int k = 2; k <= m; k++) acc *= k;
    return acc;
}
"""

UNRELATED = """
public boolean isEmpty(String s) {
    return s == null || s.length() == 0;
}
"""


def main() -> None:
    print("CodeSim", codesim.__version__)
    print("base measures ready:", available(include_optional=False))

    print("\nSingle-measure scores (renamed clone vs unrelated):")
    for measure in ["jaccard", "tfidf", "gst", "metrics", "tokentype"]:
        clone = codesim.similarity(ORIGINAL, RENAMED, measure=measure)
        other = codesim.similarity(ORIGINAL, UNRELATED, measure=measure)
        print(f"  {measure:<10} clone={clone:.3f}  unrelated={other:.3f}")

    names = available(include_optional=False)
    print("\nFeature vector length:", len(feature_vector(ORIGINAL, RENAMED, names)))

    ens = UnsupervisedEnsemble(rule="mean")
    print("\nUnsupervised mean ensemble:")
    print(f"  clone     = {ens.score(ORIGINAL, RENAMED):.3f}")
    print(f"  unrelated = {ens.score(ORIGINAL, UNRELATED):.3f}")

    pairs = [
        (ORIGINAL, RENAMED), (ORIGINAL, ORIGINAL), (RENAMED, RENAMED),
        (ORIGINAL, UNRELATED), (RENAMED, UNRELATED), (UNRELATED, ORIGINAL),
    ]
    labels = [1, 1, 1, 0, 0, 0]
    sup = SupervisedEnsemble(model="random_forest").fit_pairs(pairs, labels)
    print("\nTop measures the supervised ensemble relied on:")
    for name, weight in list(sup.importances().items())[:5]:
        print(f"  {name:<10} {weight:.3f}")


if __name__ == "__main__":
    main()
