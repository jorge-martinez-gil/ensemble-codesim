# -*- coding: utf-8 -*-
"""
[Martinez-Gil2024b] Source Code Clone Detection Using an Ensemble of Unsupervised Semantic Similarity Measures, arXiv preprint arXiv:xxxx.xxxx, 2024

@author: Jorge Martinez-Gil
"""

#!pip install transformers[torch] -U

from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import json
import logging
import random

# Disable warnings and logging messages
logging.disable(logging.WARNING)

def load_code_snippets(json_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    random.shuffle(data)
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data

def prepare_dataset(code_snippets, tokenizer):
    pairs, labels = [], []
    for snippet in code_snippets:
        code1 = snippet['code1']
        code2 = snippet['code2']
        label = snippet['score']
        pairs.append((code1, code2))
        labels.append(label)
    return CloneDetectionDataset(tokenizer, pairs, labels)

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

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        'f1': f1_score(labels, predictions, average='binary'),
        'precision': precision_score(labels, predictions, average='binary'),
        'recall': recall_score(labels, predictions, average='binary'),
    }

def main():
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    model = RobertaForSequenceClassification.from_pretrained('microsoft/codebert-base', num_labels=2)

    train_data, val_data, test_data = load_code_snippets('datasets\IR-Plag\data.json', 0.70, 0.15, 0.15)
    train_dataset = prepare_dataset(train_data, tokenizer)
    val_dataset = prepare_dataset(val_data, tokenizer)
    test_dataset = prepare_dataset(test_data, tokenizer)

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    test_results = trainer.evaluate(test_dataset)

    print(f"Precision: {test_results['eval_precision']:.4f}")
    print(f"Recall: {test_results['eval_recall']:.4f}")
    print(f"F1 Score: {test_results['eval_f1']:.4f}")

if __name__ == "__main__":
    main()