# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Source Code Clone Detection Using an Ensemble of Unsupervised Semantic Similarity Measures, arXiv preprint arXiv:xxxx.xxxx, 2024

@author: Jorge Martinez-Gil
"""

import os
import re

class RollingHash:
    def __init__(self, s, base=256, mod=1000000007):
        self.s = s
        self.base = base
        self.mod = mod
        self.hash_value = 0
        self.power = 1

        for c in s:
            self.hash_value = (self.hash_value * self.base + ord(c)) % self.mod
            self.power = (self.power * self.base) % self.mod

    def update(self, old_char, new_char):
        old_value = ord(old_char)
        new_value = ord(new_char)

        self.hash_value = (self.hash_value * self.base - old_value * self.power + new_value) % self.mod

        if self.hash_value < 0:
            self.hash_value += self.mod

def find_common_substrings(text1, text2, min_length=10):
    common_substrings = []

    for length in range(min_length, min(len(text1), len(text2)) + 1):
        hash_set = set()
        rolling_hash_text1 = RollingHash(text1[:length])
        rolling_hash_text2 = RollingHash(text2[:length])

        for i in range(len(text1) - length + 1):
            rolling_hash_text1.update(text1[i], text1[i + length - 1])
            hash_set.add(rolling_hash_text1.hash_value)

        for i in range(len(text2) - length + 1):
            rolling_hash_text2.update(text2[i], text2[i + length - 1])
            if rolling_hash_text2.hash_value in hash_set:
                common_substrings.append(text2[i:i + length])

    return common_substrings

def calculate_similarity_ratio(java_code1, java_code2):
    tokens1 = tokenize_code(java_code1)
    tokens2 = tokenize_code(java_code2)

    common_substrings = find_common_substrings(java_code1, java_code2)
    similarity_ratio = len(common_substrings) / (len(tokens1) + len(tokens2))

    return similarity_ratio

def tokenize_code(code):
    tokens = re.findall(r'\w+', code)
    return set(tokens)

def normalize_similarity_ratio(similarity_ratio):
    # Define the minimum and maximum possible similarity ratios
    min_similarity = 0
    max_similarity = 1
    
    # Normalize using min-max formula
    normalized_ratio = (similarity_ratio - min_similarity) / (max_similarity - min_similarity)
    return normalized_ratio

def similarity (code1, code2):
    similarity = calculate_similarity_ratio(code1, code2)
    similarity_ratio = normalize_similarity_ratio(similarity)
    return similarity_ratio