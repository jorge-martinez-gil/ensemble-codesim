# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Source Code Clone Detection Using an Ensemble of Unsupervised Semantic Similarity Measures, arXiv preprint arXiv:xxxx.xxxx, 2024

@author: Jorge Martinez-Gil
"""


#!pip install transformers[torch] -U

from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import torch
from torch.utils.data import Dataset
import numpy as np  # Added for compute_metrics
from sklearn.metrics import f1_score, precision_score, recall_score  # Added for compute_metrics
import json
import logging
import random  # Import random for sampling

# Disable warnings and logging messages
logging.disable(logging.WARNING)

# Load code snippets from the jsonl file
def load_code_snippets(jsonl_path):
    code_snippets = {}
    with open(jsonl_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            code_snippets[data["idx"]] = data["func"]
    return code_snippets

# Load dataset from the txt file and sample % of the data
def load_dataset(txt_path, code_snippets, sample_percentage=1.0):
    all_pairs = []
    all_labels = []
    with open(txt_path, 'r') as file:
        for line in file:
            id1, id2, label = line.strip().split('\t')
            code1 = code_snippets.get(id1, "")
            code2 = code_snippets.get(id2, "")
            if code1 and code2:
                all_pairs.append((code1, code2))
                all_labels.append(int(label))

    # Sample of the data
    sample_size = int(len(all_labels) * sample_percentage / 100)
    indices = random.sample(range(len(all_labels)), sample_size)
    pairs = [all_pairs[i] for i in indices]
    labels = [all_labels[i] for i in indices]

    return pairs, labels

# Custom dataset class for clone detection
class CloneDetectionDataset(Dataset):
    def __init__(self, tokenizer, pairs, labels):
        self.tokenizer = tokenizer
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.pairs[idx][0], self.pairs[idx][1], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def prepare_dataset(filepath, tokenizer, code_snippets):
    pairs, labels = load_dataset(filepath, code_snippets)
    return CloneDetectionDataset(tokenizer, pairs, labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    f1 = f1_score(labels, predictions, average='binary')
    precision = precision_score(labels, predictions, average='binary')
    recall = recall_score(labels, predictions, average='binary')
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

def main():
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    model = RobertaForSequenceClassification.from_pretrained('microsoft/codebert-base', num_labels=2)

    code_snippets = load_code_snippets('datasets\BigCloneBench\data.jsonl')

    # Load and prepare datasets
    train_dataset = prepare_dataset('datasets\BigCloneBench\train.txt', tokenizer, code_snippets)
    val_dataset = prepare_dataset('datasets\BigCloneBench\valid.txt', tokenizer, code_snippets)
    test_dataset = prepare_dataset('datasets\BigCloneBench\test.txt', tokenizer, code_snippets)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir='results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train the model
    trainer.train()

    # After training, evaluate the model on the test dataset
    test_results = trainer.evaluate(test_dataset)

    # Print the precision, recall, and F1 score
    print(f"Precision: {test_results['eval_precision']:.4f}")
    print(f"Recall: {test_results['eval_recall']:.4f}")
    print(f"F1 Score: {test_results['eval_f1']:.4f}")


if __name__ == "__main__":
    main()