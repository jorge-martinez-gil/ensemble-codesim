# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Source Code Clone Detection Using an Ensemble of Unsupervised Semantic Similarity Measures, arXiv preprint arXiv:xxxx.xxxx, 2024

@author: Jorge Martinez-Gil
"""

from collections import Counter

def winnow(text, k, w):
    # Split the text into k-grams
    k_grams = [text[i:i+k] for i in range(len(text)-k+1)]

    # Initialize the window and fingerprint
    window = []
    fingerprint = []

    # Loop through each k-gram
    for i, k_gram in enumerate(k_grams):
        # Add the k-gram to the window
        window.append(k_gram)

        # If the window is full, select the smallest hash value
        if len(window) == w:
            min_hash = float('inf')
            min_index = -1
            for j, k_gram in enumerate(window):
                hash_value = hash(k_gram)
                if hash_value < min_hash:
                    min_hash = hash_value
                    min_index = j
            fingerprint.append(min_hash)

            # Remove the oldest k-gram from the window
            window.pop(min_index)

    return fingerprint

def similarity(code1, code2, k=5, w=10, threshold=0.5):
    # Generate fingerprints for both code snippets
    fingerprint1 = winnow(code1, k, w)
    fingerprint2 = winnow(code2, k, w)

    # Calculate the number of matching fingerprints
    matches = 0
    for fp1 in fingerprint1:
        for fp2 in fingerprint2:
            if fp1 == fp2:
                matches += 1
                break

    # Calculate the similarity score
    similarity_score = matches / min(len(fingerprint1), len(fingerprint2))

    return similarity_score