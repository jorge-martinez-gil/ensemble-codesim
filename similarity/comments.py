# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Advanced Detection of Source Code Clones via an Ensemble of Unsupervised Similarity Measures, arXiv preprint arXiv:2405.02095, 2024

@author: Jorge Martinez-Gil
"""

import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

def similarity(code1, code2):
    tokens1 = word_tokenize(code1.lower())
    tokens2 = word_tokenize(code2.lower())
    tagged_data = [TaggedDocument(words=tokens1, tags=["original"]), TaggedDocument(words=tokens2, tags=["plagiarized"])]
    model = Doc2Vec(tagged_data, vector_size=30, window=5, min_count=1, workers=4, epochs=100)
    similarity_ratio = model.dv.similarity("original", "plagiarized")
    return similarity_ratio
