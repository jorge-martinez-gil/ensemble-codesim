# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Source Code Clone Detection Using an Ensemble of Unsupervised Semantic Similarity Measures, arXiv preprint arXiv:xxxx.xxxx, 2024

@author: Jorge Martinez-Gil
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
import matplotlib.pyplot as plt
import json

intermediate_file = 'outputs/output-karnalim.txt'

# Load your data
with open(intermediate_file, 'r') as file:
    all_results = [json.loads(line) for line in file]

# Assuming each item in all_results is a list where the last item is 's0'
data = np.array(all_results)

X = data[:, :-1]  # All rows, all columns except the last one
y = data[:, -1]  # All rows, just the last column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_jobs=-1)

# Define a scoring function
f1_scorer = make_scorer(f1_score, average='binary')

# Setup the grid search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
}
grid_search = GridSearchCV(rf_classifier, param_grid, scoring=f1_scorer, cv=5, n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print(f"Best Training parameters: {grid_search.best_params_}")
print(f"Best Training F1 Score: {grid_search.best_score_:.2f}")

# Use the best estimator to make predictions
y_pred = grid_search.predict(X_test)

print("------------------------")

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

# Get feature importances
importances = grid_search.best_estimator_.feature_importances_

# Sort the feature importances in descending order
indices = np.argsort(importances)[::-1]

# Assuming features are named as 'Feature 1', 'Feature 2', ..., 'Feature N'
feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
sorted_feature_names = [feature_names[i] for i in indices]

# Create the plot
plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], color='b', align='center')
plt.xticks(range(X.shape[1]), sorted_feature_names, rotation=45)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()

