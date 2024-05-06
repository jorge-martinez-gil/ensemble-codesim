# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Source Code Clone Detection Using an Ensemble of Unsupervised Semantic Similarity Measures, arXiv preprint arXiv:xxxx.xxxx, 2024

@author: Jorge Martinez-Gil
"""

import difflib

def semantic_diff(code1, code2):
    lines1 = code1.splitlines()
    lines2 = code2.splitlines()
    
    diff = difflib.ndiff(lines1, lines2)
    
    added_lines = []
    removed_lines = []
    
    for line in diff:
        if line.startswith('+ '):
            added_lines.append(line[2:])
        elif line.startswith('- '):
            removed_lines.append(line[2:])
    
    return added_lines, removed_lines

def similarity (code1, code2):
    added, removed = semantic_diff(code1, code2)
    max_diff = len(code1.split('\n')) + len(code2.split('\n'))
    diff = len(added) + len(removed)
    similarity_ratio = 1 - (diff / max_diff)
    return similarity_ratio
                                    