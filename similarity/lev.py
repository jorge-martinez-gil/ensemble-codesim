# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Advanced Detection of Source Code Clones via an Ensemble of Unsupervised Similarity Measures, arXiv preprint arXiv:2405.02095, 2024

@author: Jorge Martinez-Gil
"""

import Levenshtein

def similarity (code1, code2):
    lev_distance = Levenshtein.distance(code1, code2)
    max_distance = max(len(code1), len(code2))
    similarity_ratio = 1 - (lev_distance / max_distance)
    return similarity_ratio


