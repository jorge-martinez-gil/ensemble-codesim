# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Source Code Clone Detection Using an Ensemble of Unsupervised Semantic Similarity Measures, arXiv preprint arXiv:xxxx.xxxx, 2024

@author: Jorge Martinez-Gil
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(file):
    intermediate_file = file  # Assuming 'file' parameter is used correctly.

    def calculate_metrics_for_similarity(df, similarity_column, threshold):
        """Calculate accuracy, precision, recall, and F1 score for a similarity score."""
        predictions = df[similarity_column].apply(lambda x: 1 if x >= threshold else 0)
        accuracy = accuracy_score(df['Truth'], predictions)
        precision = precision_score(df['Truth'], predictions, zero_division=0)
        recall = recall_score(df['Truth'], predictions, zero_division=0)
        f1 = f1_score(df['Truth'], predictions, zero_division=0)
        return {
            'similarity_metric': similarity_column,
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    with open(intermediate_file, 'r') as file:
        all_results = [json.loads(line) for line in file]

    data = np.array(all_results)
    df = pd.DataFrame(all_results, columns=[f's{i}' for i in range(1, 22)] + ['Truth'])  # Adjust the range based on the number of similarity functions
   
    optimal_thresholds = []
    # For each feature column
    for feature_idx in range(data.shape[1] - 1):  # Exclude the last column (ground truth)
        feature_values = data[:, feature_idx]
        true_labels = data[:, -1]
        
        # Calculate the ROC curve
        fpr, tpr, thresholds = roc_curve(true_labels, feature_values)
        
        # Find the optimal threshold (maximizing TPR - FPR)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        optimal_thresholds.append(optimal_threshold)

    metrics_results = []
    for i, column_name in enumerate(df.columns[:-1]):  # Exclude 'Truth' column
        metric_result = calculate_metrics_for_similarity(df, column_name, optimal_thresholds[i])
        metrics_results.append(metric_result)

    metrics_df = pd.DataFrame(metrics_results)
    print(metrics_df)

def main():
    print ("-----Calculating metrics for Karnalim--------")
    calculate_metrics("outputs/output-karnalim.txt")
    print ("-----Calculating metrics for BigCloneBench--------")
    calculate_metrics("outputs/output-bigclonebench.txt")

if __name__ == "__main__":
    main()
