# Source Code Clone Detection Using an Ensemble of Unsupervised Semantic Similarity Measures

This repository contains the implementation for the paper "Source Code Clone Detection Using an Ensemble of Unsupervised Semantic Similarity Measures" by Jorge Martinez-Gil. It focuses on evaluating code similarity using a novel ensemble learning approach, integrating multiple unsupervised similarity measures.

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

## Dependencies
Ensure the following dependencies are installed:

Python 3.x
NumPy
scikit-learn
tqdm

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
