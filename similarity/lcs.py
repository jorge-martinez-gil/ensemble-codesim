# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Source Code Clone Detection Using an Ensemble of Unsupervised Semantic Similarity Measures, arXiv preprint arXiv:xxxx.xxxx, 2024

@author: Jorge Martinez-Gil
"""

import difflib

def similarity (code1, code2):
    lcs = difflib.SequenceMatcher(None, code1, code2).find_longest_match(0, len(code1), 0, len(code2)).size
    similarity_ratio = (2 * lcs) / (len(code1) + len(code2))
    return similarity_ratio

