# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Advanced Detection of Source Code Clones via an Ensemble of Unsupervised Similarity Measures, arXiv preprint arXiv:2405.02095, 2024

@author: Jorge Martinez-Gil
"""

import re
from collections import Counter
import os

# Function to extract function calls from Java code using regex
def extract_function_calls(code):
    function_calls = []
    # Regular expression to match simple method invocations; might need refinement for complex cases
    pattern = r'\b(\w+)\s*\('
    matches = re.findall(pattern, code)
    for match in matches:
        function_calls.append(match)
    return function_calls

def similarity(code1, code2):
    # Extract function calls from both code snippets
    function_calls1 = extract_function_calls(code1)
    function_calls2 = extract_function_calls(code2)

    # Calculate the similarity based on function calls using Jaccard similarity
    intersection = len(set(function_calls1) & set(function_calls2))
    union = len(set(function_calls1) | set(function_calls2))
    similarity_ratio = intersection / union if union > 0 else 0.0
    return similarity_ratio


