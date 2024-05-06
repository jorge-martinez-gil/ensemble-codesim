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
@InProceedings{10.1007/978-3-031-56281-5_2,
	author="Martinez-Gil, Jorge",
	editor="Bludau, Peter
	and Ramler, Rudolf
	and Winkler, Dietmar
	and Bergsmann, Johannes",
	title="Source Code Clone Detection Using Unsupervised Similarity Measures",
	booktitle="Software Quality as a Foundation for Security",
	year="2024",
	publisher="Springer Nature Switzerland",
	address="Cham",
	pages="21--37",
	isbn="978-3-031-56281-5"
}
```

## 📄 License

The project is provided under the MIT License. 