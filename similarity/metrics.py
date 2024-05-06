# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Advanced Detection of Source Code Clones via an Ensemble of Unsupervised Similarity Measures, arXiv preprint arXiv:2405.02095, 2024

@author: Jorge Martinez-Gil
"""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

def calculate_code_metrics_simple(code):
    lines_of_code = len(code.split('\n'))
    
    # Count occurrences of Java keywords; this list is not exhaustive
    keywords = ['abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum', 'extends', 'final', 'finally', 'float', 'for', 'if', 'implements', 'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new', 'package', 'private', 'protected', 'public', 'return', 'short', 'static', 'strictfp', 'super', 'switch', 'synchronized', 'this', 'throw', 'throws', 'transient', 'try', 'void', 'volatile', 'while']
    num_keywords = sum(code.count(keyword) for keyword in keywords)
    
    # Control flow elements
    num_loops = code.count('for') + code.count('while')
    num_conditionals = code.count('if') + code.count('else') + code.count('switch')
    
    # Function/method declarations - very basic approximation using "void" and some access modifiers
    num_functions = len(re.findall(r'\b(public|protected|private|static)\s+[\w<>\[\]]+\s+\w+\s*\(', code))
    
    # Calculate the depth of nesting using curly braces
    brace_depth = 0
    max_brace_depth = 0
    for char in code:
        if char == "{":
            brace_depth += 1
            max_brace_depth = max(max_brace_depth, brace_depth)
        elif char == "}":
            brace_depth -= 1
    
    return np.array([lines_of_code, num_keywords, num_loops, num_conditionals, num_functions, max_brace_depth])

def similarity(code1, code2):
    try:
        metrics_code1 = calculate_code_metrics_simple(code1)
        metrics_code2 = calculate_code_metrics_simple(code2)
        
        # Normalize the vectors to unit length to account for code size differences
        metrics_code1 = metrics_code1 / np.linalg.norm(metrics_code1)
        metrics_code2 = metrics_code2 / np.linalg.norm(metrics_code2)
        
        similarity_ratio = cosine_similarity([metrics_code1], [metrics_code2])[0][0]
        return similarity_ratio
    except Exception as e:
        return 0
