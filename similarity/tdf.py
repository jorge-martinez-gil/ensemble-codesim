# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Source Code Clone Detection Using an Ensemble of Unsupervised Semantic Similarity Measures, arXiv preprint arXiv:xxxx.xxxx, 2024

@author: Jorge Martinez-Gil
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def similarity (code1, code2):      
    vectorizer = TfidfVectorizer()                        
    tfidf_matrix = vectorizer.fit_transform([code1, code2])
    similarity_ratio = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return similarity_ratio
    

