# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Advanced Detection of Source Code Clones via an Ensemble of Unsupervised Similarity Measures, arXiv preprint arXiv:2405.02095, 2024

@author: Jorge Martinez-Gil
"""

import difflib
import re

def extract_method_names(code):
    # Regex to match simple Java method declarations
    method_pattern = r'\bpublic\b\s+(?:[\w\<\>\[\]]+\s+)+?(\w+)\s*\('
    # Finding all matches in the code
    method_names = re.findall(method_pattern, code)
    return method_names

def calculate_similarity_ratio(code1, code2):
    seq_matcher = difflib.SequenceMatcher(None, code1, code2)
    similarity_ratio = seq_matcher.ratio()
    return similarity_ratio

def semantic_clone_detection(code1, code2, threshold=0.8):
    method_names1 = extract_method_names(code1)
    method_names2 = extract_method_names(code2)

    common_method_names = set(method_names1) & set(method_names2)
    similarity_ratio = calculate_similarity_ratio(code1, code2)

    if common_method_names and similarity_ratio >= threshold:
        return True, common_method_names, similarity_ratio
    else:
        return False, None, similarity_ratio

def similarity(code1, code2):
    try:
        is_clone, common_methods, similarity_ratio = semantic_clone_detection(code1, code2)
        return similarity_ratio
    except Exception as e:
        return 0

