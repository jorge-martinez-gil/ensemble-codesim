# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Advanced Detection of Source Code Clones via an Ensemble of Unsupervised Similarity Measures, arXiv preprint arXiv:2405.02095, 2024

@author: Jorge Martinez-Gil
"""
from fuzzywuzzy import fuzz

def similarity (code1, code2):
    similarity_ratio = fuzz.token_sort_ratio(code1, code2) / 100
    return similarity_ratio
