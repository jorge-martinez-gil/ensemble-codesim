# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Advanced Detection of Source Code Clones via an Ensemble of Unsupervised Similarity Measures, arXiv preprint arXiv:2405.02095, 2024

@author: Jorge Martinez-Gil
"""

import javalang
import networkx as nx

# Generate Program Dependence Graph (PDG) for Java code snippets
def generate_pdg(code):
    tokens = list(javalang.tokenizer.tokenize(code))
    parser = javalang.parser.Parser(tokens)
    tree = parser.parse_member_declaration()

    graph = nx.DiGraph()

    for path, node in tree:
        if isinstance(node, javalang.tree.MethodDeclaration):
            method_name = node.name
            graph.add_node(method_name)
            
            # Simulated data dependence
            for param in node.parameters:
                graph.add_edge(param.name, method_name)

            # Simulated control dependence
            if isinstance(node.body, javalang.tree.BlockStatement):
                for stmt in node.body.statements:
                    if isinstance(stmt, javalang.tree.IfStatement):
                        graph.add_edge(stmt.expression.value, method_name)

    return graph

def similarity(code1, code2):
    try:
        pdg_1 = generate_pdg(code1)
        pdg_2 = generate_pdg(code2)
        total_nodes = max(len(pdg_1.nodes) + len(pdg_2.nodes), 1)  # Ensure divisor is not zero
        normalized_edit_distance = nx.graph_edit_distance(pdg_1, pdg_2, node_match=lambda n1, n2: n1 == n2) / total_nodes
        similarity_ratio = 1 - normalized_edit_distance
        similarity_ratio = max(0, min(similarity_ratio, 1))  # Clamp similarity_ratio between 0 and 1
    except Exception as e:
        similarity_ratio = 0
    return similarity_ratio
