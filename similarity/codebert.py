# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Source Code Clone Detection Using an Ensemble of Unsupervised Semantic Similarity Measures, arXiv preprint arXiv:xxxx.xxxx, 2024

@author: Jorge Martinez-Gil
"""

import os
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel

# Load the CodeBERT model and tokenizer
model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Define a function to generate embeddings for a given code snippet
def generate_embedding(code):
    # Tokenize the code
    inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True)
    # Generate the model's output
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract the embedding for the [CLS] token
    embedding = outputs.last_hidden_state[:, 0, :]
    # Normalize the embedding
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    return embedding

def similarity(code1, code2):
    embedding1 = generate_embedding(code1)
    embedding2 = generate_embedding(code2)
    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    similarity_ratio = similarity.item()
    return similarity_ratio
                                