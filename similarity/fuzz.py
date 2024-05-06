# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Source Code Clone Detection Using an Ensemble of Unsupervised Semantic Similarity Measures, arXiv preprint arXiv:xxxx.xxxx, 2024

@author: Jorge Martinez-Gil
"""
from fuzzywuzzy import fuzz

def similarity (code1, code2):
    similarity_ratio = fuzz.token_sort_ratio(code1, code2) / 100
    return similarity_ratio
