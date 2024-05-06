# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Source Code Clone Detection Using an Ensemble of Unsupervised Semantic Similarity Measures, arXiv preprint arXiv:xxxx.xxxx, 2024

@author: Jorge Martinez-Gil
"""

import Levenshtein

def similarity (code1, code2):
    lev_distance = Levenshtein.distance(code1, code2)
    max_distance = max(len(code1), len(code2))
    similarity_ratio = 1 - (lev_distance / max_distance)
    return similarity_ratio


