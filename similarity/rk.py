# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Advanced Detection of Source Code Clones via an Ensemble of Unsupervised Similarity Measures, arXiv preprint arXiv:2405.02095, 2024

@author: Jorge Martinez-Gil
"""

def similarity(code1, code2):
    def tokenize(code):
        # Simplified tokenizer for demonstration; in practice, use a proper tokenizer
        return code.split()

    def rabin_karp_hash(token, prime=101):
        hash_value = 0
        n = len(token)
        for i, char in enumerate(token):
            hash_value += ord(char) * (prime ** (n - i - 1))
        return hash_value

    def match_tiles(tokens1, tokens2):
        hash_to_token = {rabin_karp_hash(token): token for token in set(tokens1 + tokens2)}
        match_matrix = [[0] * (len(tokens2) + 1) for _ in range(len(tokens1) + 1)]

        max_len = 0
        max_pos = (0, 0)
        for i in range(1, len(tokens1) + 1):
            for j in range(1, len(tokens2) + 1):
                if tokens1[i - 1] == tokens2[j - 1]:
                    match_matrix[i][j] = match_matrix[i - 1][j - 1] + 1
                    if match_matrix[i][j] > max_len:
                        max_len = match_matrix[i][j]
                        max_pos = (i, j)

        # Extracting the matched tiles
        total_matched_length = 0
        while max_len > 0:
            i, j = max_pos
            match_length = max_len
            total_matched_length += match_length

            # Zero out the current tile to find next tile
            for di in range(match_length):
                for dj in range(match_length):
                    match_matrix[i - di][j - dj] = 0

            # Find next tile
            max_len = 0
            for i in range(1, len(tokens1) + 1):
                for j in range(1, len(tokens2) + 1):
                    if match_matrix[i][j] > max_len:
                        max_len = match_matrix[i][j]
                        max_pos = (i, j)

        return total_matched_length

    tokens1 = tokenize(code1)
    tokens2 = tokenize(code2)
    total_matched_length = match_tiles(tokens1, tokens2)
    
    total_length = len(tokens1) + len(tokens2)
    similarity_score = (2 * total_matched_length) / total_length if total_length > 0 else 0
    return similarity_score