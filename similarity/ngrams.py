# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Advanced Detection of Source Code Clones via an Ensemble of Unsupervised Similarity Measures, arXiv preprint arXiv:2405.02095, 2024

@author: Jorge Martinez-Gil
"""

from nltk import ngrams

def similarity (code1, code2):
    n = 15
    ngrams1 = set(ngrams(code1, n))
    ngrams2 = set(ngrams(code2, n))
    similarity_ratio = len(ngrams1.intersection(ngrams2)) / len(ngrams1.union(ngrams2))
    return similarity_ratio