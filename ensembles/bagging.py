# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Advanced Detection of Source Code Clones via an Ensemble of Unsupervised Similarity Measures, arXiv preprint arXiv:2405.02095, 2024

@author: Jorge Martinez-Gil
"""

import json
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to read data from a .jsonl file
def read_data(filename):
    with open(filename, 'r') as file:
        data = [json.loads(line) for line in file]
    return np.array(data)

# Path to your .jsonl file
filename = 'outputs\output7-bigclonebench.txt'

# Read the data
data = read_data(filename)

# Splitting dataset into features and target
X = data[:, :-1]
y = data[:, -1]

# Splitting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define parameter grid for hyperparameter search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 4, 6]
}

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_jobs=-1)

# Instantiate GridSearchCV for hyperparameter tuning with F1 optimization
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='f1', n_jobs=-1)

# Fit the GridSearchCV object
grid_search.fit(X_train, y_train)

# Get the best model and make predictions
best_rf_classifier = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")
y_pred = best_rf_classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

# Display the metrics
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
