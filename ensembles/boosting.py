# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Advanced Detection of Source Code Clones via an Ensemble of Unsupervised Similarity Measures, arXiv preprint arXiv:2405.02095, 2024

@author: Jorge Martinez-Gil
"""


import xgboost as xgb
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import cudf
import numpy as np
import pandas as pd
import json


filename = 'outputs/output-bigclonebench.txt'

data_list = []
with open(filename, 'r') as f:
    for line in f:
        try:
            data_list.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Error parsing line: {e}")

# Create a pandas DataFrame from the list of dictionaries
data_pd = pd.DataFrame(data_list)

# Convert the pandas DataFrame to a cudf DataFrame
data = cudf.DataFrame.from_pandas(data_pd)

# Preparing the data
X = data.iloc[:, :-1].to_pandas()  # Convert to Pandas DataFrame for compatibility with Scikit-Learn
y = data.iloc[:, -1].astype('int32').to_numpy()

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define a large hyperparameter space to explore
param_distributions = {
    'max_depth': np.arange(3, 16),
    'n_estimators': np.arange(50, 400, 50),
    'learning_rate': np.logspace(-3, 0, 10),
    'subsample': np.linspace(0.5, 1.0, 6),
    'colsample_bytree': np.linspace(0.5, 1.0, 6),
    'gamma': np.linspace(0, 0.5, 6),
    'min_child_weight': np.arange(1, 10),
    'reg_alpha': np.logspace(-4, 1, 6),
    'tree_method': ['hist'],  # Enclose in list
    'device': ['gpu']  # Correct value and enclose in list
}

# Model setup
model = xgb.XGBClassifier(use_label_encoder=False)

# Scorer for optimization
f1_scorer = make_scorer(f1_score, average='binary')

# RandomizedSearchCV setup
search = RandomizedSearchCV(model, param_distributions, n_iter=100, scoring=f1_scorer, cv=3, verbose=2, n_jobs=-1)

# Perform the search
search.fit(X_train, y_train)

# Best parameters and F1 score
print(f"Best F1 Score: {search.best_score_}")
print(f"Best Parameters: {search.best_params_}")

# Predict and evaluate on the test set
y_pred = search.predict(X_test)
f1 = f1_score(y_test, y_pred, average='binary')
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')

print(f"Test Precision: {precision}")
print(f"Test Recall: {recall}")
print(f"Test F1 Score: {f1}")