# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Advanced Detection of Source Code Clones via an Ensemble of Unsupervised Similarity Measures, arXiv preprint arXiv:2405.02095, 2024

@author: Jorge Martinez-Gil
"""

import os
import re
from collections import Counter

# Define a function to tokenize a code snippet
def tokenize(code):
    # Remove comments and whitespace
    code = re.sub(r'//.*?\n|/\*.*?\*/', '', code, flags=re.S)
    code = re.sub(r'\s+', ' ', code)
    # Tokenize the code
    tokens = re.findall(r'\b\w+\b', code)
    # Count the frequency of each token
    token_counts = Counter(tokens)
    return token_counts

def similarity(code1, code2):
    original_tokens = tokenize(code1)
    tokens2 = tokenize(code2)
    intersection = sum((original_tokens & tokens2).values())
    union = sum((original_tokens | tokens2).values())
    similarity_ratio = intersection / union
    return similarity_ratio
