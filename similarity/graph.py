# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Advanced Detection of Source Code Clones via an Ensemble of Unsupervised Similarity Measures, arXiv preprint arXiv:2405.02095, 2024

@author: Jorge Martinez-Gil
"""

import os
import networkx as nx
import numpy as np
from sklearn.metrics import jaccard_score

def get_code_graph(code_snippet, num_nodes=None):
    # Create a new directed graph.
    graph = nx.DiGraph()

    # Split the code snippet into tokens.
    tokens = code_snippet.split()

    # Add the tokens as nodes to the graph.
    for token in tokens:
        graph.add_node(token)

    # Add edges between adjacent tokens.
    for i in range(len(tokens) - 1):
        graph.add_edge(tokens[i], tokens[i + 1])

    # Add dummy nodes if necessary.
    if num_nodes is not None and graph.number_of_nodes() < num_nodes:
        while graph.number_of_nodes() < num_nodes:
            dummy_label = f"DUMMY_NODE_LABEL_{graph.number_of_nodes()}"
            graph.add_node(dummy_label)

    # Return the graph.
    return graph

def similarity(code_snippet_1, code_snippet_2):
    # Get the graphs representing the code snippets.
    code_graph_1 = get_code_graph(code_snippet_1, num_nodes=max(len(code_snippet_1.split()), len(code_snippet_2.split())))
    code_graph_2 = get_code_graph(code_snippet_2, num_nodes=max(len(code_snippet_1.split()), len(code_snippet_2.split())))

    # Calculate the Jaccard similarity between the two sets of nodes.
    jaccard_similarity = jaccard_score(np.array(list(code_graph_1.nodes())), np.array(list(code_graph_2.nodes())), average='macro')

    # Return the similarity.
    return jaccard_similarity

