# Advanced Detection of Source Code Clones via an Ensemble of Unsupervised Similarity Measures

This repository contains the implementation for the paper "Advanced Detection of Source Code Clones via an Ensemble of Unsupervised Similarity Measures" by Jorge Martinez-Gil. It focuses on evaluating code similarity using a novel ensemble learning approach, integrating multiple unsupervised similarity measures.

[![DOI: 10.1007/978-3-031-89277-6_5](https://img.shields.io/badge/DOI-10.1007%2F978--3--031--89277--6__5-red.svg)](https://doi.org/10.1007/978-3-031-89277-6_5)
[![arXiv](https://img.shields.io/badge/arXiv-2405.02095-2ebc4f.svg)](https://arxiv.org/abs/2405.02095)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![Citations](https://img.shields.io/badge/citations-4-blue)](https://scholar.google.com/citations?view_op=view_citation&hl=en&citation_for_view=X1pRUYcAAAAJ:DTjSuSUbmXsC)

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
@inproceedings{MartinezGil2025,
  author    = {Jorge Martinez-Gil},
  editor    = {Jannik Fischbach and
               Rudolf Ramler and
               Dietmar Winkler and
               Johannes Bergsmann},
  title     = {Advanced Detection of Source Code Clones via an Ensemble of Unsupervised Similarity Measures},
  booktitle = {Balancing Software Innovation and Regulatory Compliance - 17th International
               Conference on Software Quality, {SWQD} 2025, Munich, Germany, May 20-22, 2025, Proceedings},
  series    = {Lecture Notes in Business Information Processing},
  volume    = {544},
  pages     = {72--90},
  year      = {2025},
  publisher = {Springer},
  doi       = {10.1007/978-3-031-89277-6\_5},
  url       = {https://link.springer.com/chapter/10.1007/978-3-031-89277-6\_5}
}
```
---
### Research that has cited this work

1. **[An Enhanced Transformer-Based Framework for Interpretable Code Clone Detection](https://www.sciencedirect.com/science/article/pii/S0164121225000159)**
   - **Authors:** M. Nashaat, R. Amin, A.H. Eid, R.F. Abdel-Kader
   - **Journal:** *Journal of Systems and Software*, 2025 (Elsevier)
   - **Abstract:** Code cloning is a common practice in software development for reusing code segments. This study proposes an enhanced transformer-based framework to improve interpretable clone detection.

2. **[A Novel Method for Code Clone Detection Based on Minimally Random Kernel Convolutional Transform](https://ieeexplore.ieee.org/abstract/document/10731684/)**
   - **Author:** M. Abdelkader
   - **Journal:** *IEEE Access*, 2024
   - **Abstract:** This paper introduces a novel approach for code clone detection using minimally random kernel convolutional transforms, aiming to enhance accuracy in software maintenance.

3. **[Improving Source Code Similarity Detection Through GraphCodeBERT and Integration of Additional Features](https://arxiv.org/abs/2408.08903)**
   - **Author:** J. Martinez-Gil
   - **Journal:** *arXiv preprint arXiv:2408.08903*, 2024
   - **Abstract:** This study presents a method for improving source code similarity detection by integrating additional output features into the classification process, leveraging GraphCodeBERT.
---
## 📄 License

The project is provided under the MIT License. 
