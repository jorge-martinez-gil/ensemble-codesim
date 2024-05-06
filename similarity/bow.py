# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Source Code Clone Detection Using an Ensemble of Unsupervised Semantic Similarity Measures, arXiv preprint arXiv:xxxx.xxxx, 2024

@author: Jorge Martinez-Gil
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def similarity(code1, code2):
    vectorizer = CountVectorizer().fit_transform([code1, code2])
    similarity_ratio = cosine_similarity(vectorizer)[0][1]
    return similarity_ratio