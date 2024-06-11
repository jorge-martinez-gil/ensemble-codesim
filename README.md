# Advanced Detection of Source Code Clones via an Ensemble of Unsupervised Similarity Measures

This repository contains the implementation for the paper "Advanced Detection of Source Code Clones via an Ensemble of Unsupervised Similarity Measures" by Jorge Martinez-Gil. It focuses on evaluating code similarity using a novel ensemble learning approach, integrating multiple unsupervised similarity measures.

[![arXiv](https://img.shields.io/badge/arXiv-2405.02095-b31b1b.svg)](https://arxiv.org/abs/2405.02095)

## 🌍 Abstract

Accurately determining code similarity is crucial for many software development tasks, such as software maintenance and code duplicate identification. This research introduces an ensemble learning approach for code similarity assessment that combines multiple unsupervised similarity measures. The approach leverages the strengths of diverse similarity measures, mitigating individual weaknesses and improving overall performance. Preliminary results suggest that while Transformers-based CodeBERT excels with abundant training data, our ensemble approach performs comparably on specific small datasets, offering better interpretability and a lower carbon footprint.

## Features

- **Ensemble Similarity Metrics**: Combines various unsupervised methods to assess code similarity.
- **Efficient and Sustainable**: Designed to perform well on small datasets with reduced computational costs.
- **Transparent and Interpretable**: Facilitates understanding of how code similarity decisions are made.

## Installation

Clone this repository using:

```bash
git clone https://github.com/jorge-martinez-gil/ensemble-codesim.git
```

### Baseline Preparation

The dataset should be organized in two parts:
1. **Code Snippets:** Stored in a JSON Lines (.jsonl) file where each line contains a JSON object with the code snippet and its corresponding index.
2. **Clone Pairs:** Stored in a tab-separated values (.txt) file where each line contains a pair of indices and a label indicating whether they are clones.

### Training and Evaluation

The training process involves the following steps:
1. **Load Code Snippets:** Parse the JSONL file to load all code snippets into a dictionary.
2. **Prepare Dataset:** Read the clone pairs from the TXT file and sample the data as needed.
3. **Train Model:** Use the GraphCodeBERT model and the Hugging Face Trainer to train the model on the prepared dataset.
4. **Evaluate Model:** Evaluate the trained model on the test dataset and compute metrics to measure its performance.

### Example Workflow

1. **Loading Code Snippets:**
   ```python
   code_snippets = load_code_snippets('datasets/BigCloneBench/data.jsonl')
   ```

2. **Preparing the Dataset:**
   ```python
   train_dataset = prepare_dataset('datasets/BigCloneBench/train.txt', tokenizer, code_snippets)
   val_dataset = prepare_dataset('datasets/BigCloneBench/valid.txt', tokenizer, code_snippets)
   test_dataset = prepare_dataset('datasets/BigCloneBench/test.txt', tokenizer, code_snippets)
   ```

3. **Training the Model:**
   ```python
   trainer.train()
   ```

4. **Evaluating the Model:**
   ```python
   test_results = trainer.evaluate(test_dataset)
   ```

5. **Printing Results:**
   ```python
   print(f"Precision: {test_results['eval_precision']:.4f}")
   print(f"Recall: {test_results['eval_recall']:.4f}")
   print(f"F1 Score: {test_results['eval_f1']:.4f}")
   ```

## Dependencies
Ensure the following dependencies are installed:

```
Python 3.x
NumPy
scikit-learn
tqdm
```

## 📚 Reference

If you use this work, please cite:

```
@misc{martinezgil2024advanced,
      title={Advanced Detection of Source Code Clones via an Ensemble of Unsupervised Similarity Measures}, 
      author={Jorge Martinez-Gil},
      year={2024},
      eprint={2405.02095},
      archivePrefix={arXiv},
      primaryClass={cs.SE}
}
```

## 📄 License

The project is provided under the MIT License. 
