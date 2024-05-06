# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Source Code Clone Detection Using an Ensemble of Unsupervised Semantic Similarity Measures, arXiv preprint arXiv:xxxx.xxxx, 2024

@author: Jorge Martinez-Gil
"""

from pygments import lex
from pygments.lexers import JavaLexer
from collections import Counter

def tokenize_code(code):
    """
    Tokenizes the Java code into a list of token types.
    """
    lexer = JavaLexer()
    tokens = [token for token, value in lex(code, lexer)]
    return tokens

def calculate_normalized_similarity(tokens1, tokens2):
    """
    Calculates a normalized similarity score between two sets of tokens.
    The score is normalized to be between 0 and 1, where 1 means identical.
    """
    set1, set2 = Counter(tokens1), Counter(tokens2)
    intersection_score = sum((set1 & set2).values())
    total_tokens = len(tokens1) + len(tokens2)
    
    if total_tokens == 0:  # Prevent division by zero if both are empty
        return 1 if intersection_score == 0 else 0
    
    # Normalize the score to be between 0 and 1
    normalized_similarity = (2.0 * intersection_score) / total_tokens
    return normalized_similarity

def similarity(snippet1, snippet2):
    try:
        tokens1 = tokenize_code(snippet1)
        tokens2 = tokenize_code(snippet2)
        normalized_similarity = calculate_normalized_similarity(tokens1, tokens2)
        return normalized_similarity
    except Exception as e:
        return 0