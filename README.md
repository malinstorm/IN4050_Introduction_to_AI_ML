
# IN4050 – Introduction to Artificial Intelligence and Machine Learning

Coursework from **IN4050 – Introduction to Artificial Intelligence and Machine Learning** at the University of Oslo (UiO), 2024.

The repository contains three mandatory assignments covering core techniques in artificial intelligence and machine learning. The assignments include implementations of search and optimization algorithms, supervised learning methods, neural networks, dimensionality reduction, and clustering.

## Assignment 1 – Traveling Salesman Problem

The first assignment explores search and evolutionary optimization through the **Traveling Salesman Problem (TSP)**. The objective is to find short routes between a set of European cities using increasingly sophisticated optimization techniques.

### Techniques

- Exhaustive search
- Hill climbing
- Genetic algorithms
- Population-based optimization
- Selection, crossover, and mutation
- Fitness evaluation
- Lamarckian learning
- Baldwinian learning
- Hybrid genetic/local-search algorithms
- Visualization and comparison of optimized routes

The assignment compares deterministic exhaustive search with stochastic optimization methods and examines the trade-off between computational cost and solution quality.

---

## Assignment 2 – Supervised Learning

The second assignment focuses on **supervised machine learning**, implementing classification algorithms and neural networks largely from scratch.

### Techniques

- Feature scaling and preprocessing
- Linear classifiers
- Logistic regression
- Binary classification
- One-vs-Rest multiclass classification
- Multinomial / Softmax logistic regression
- Multi-Layer Perceptrons (MLP)
- Feed-forward neural networks
- Backpropagation
- Hyperparameter tuning
- Training and validation
- Model evaluation and comparison

The assignment explores how different classification models learn decision boundaries and how model architecture and hyperparameters affect predictive performance.

---

## Assignment 3 – Unsupervised Learning

The third assignment focuses on **unsupervised learning**, dimensionality reduction, and clustering.

### Techniques

- Principal Component Analysis (PCA)
- Covariance matrices
- Eigenvalues and eigenvectors
- Dimensionality reduction
- Data projection and reconstruction
- Image compression using PCA
- Explained variance
- K-Means clustering
- Cluster evaluation
- Logistic regression for quantitative evaluation
- Data visualization

PCA is implemented and used on several datasets, including image data, to investigate dimensionality reduction and reconstruction. K-Means clustering is then used to explore structure in unlabeled data and evaluate the resulting clusters.

---

## Technologies

- Python
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook

## Repository Structure

```text
IN4050_Introduction_to_AI_ML/
├── ASS1_IN4050/
│   ├── IN4050_Assignment1_orig.ipynb
│   ├── IN4050_Assignment_1.py
│   ├── european_cities.csv
│   └── map.png
│
├── ASS2_IN4050/
│   ├── IN4050_Assignment2_orig.ipynb
│   └── IN4050_Assignment2.py
│
└── ASS3_IN4050/
    ├── IN4050_Assignment_3_orig.ipynb
    ├── IN4050_Assignment_3.py
    ├── syntheticdata.py
    ├── PCAanswer1.pkl
    └── PCAanswer2.pkl
```

The original Jupyter notebooks are included alongside Python versions of the assignments. The notebooks preserve the original coursework structure, experiments, visualizations, and results, while the Python files provide an easier way to inspect the source code.

## Course

**IN4050 – Introduction to Artificial Intelligence and Machine Learning**  
University of Oslo (UiO)  
2024