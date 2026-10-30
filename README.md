# Breast Cancer Classification with Machine Learning

An educational machine-learning project that classifies samples in Scikit-learn’s Wisconsin Diagnostic Breast Cancer dataset using Logistic Regression. The notebook includes exploratory data analysis, a stratified train/test split, feature scaling, and evaluation with accuracy, a classification report, and a confusion matrix.

> **Disclaimer:** This project is for learning and demonstration only. It is not a medical device and must not be used to diagnose or make treatment decisions.

## Dataset

The notebook loads the dataset using `sklearn.datasets.load_breast_cancer(as_frame=True)`. It contains 569 samples and 30 numeric predictor features. The target encoding in Scikit-learn is `0 = malignant` and `1 = benign`. No separate dataset file is needed.

## Methods

1. Inspect data types, summary statistics, missing values, and class balance.
2. Visualize the target distribution and feature correlations.
3. Split data into 75% training and 25% test sets using stratification.
4. Standardize predictors and fit Logistic Regression using a Scikit-learn pipeline.
5. Report held-out accuracy, precision, recall, F1-score, and a labeled confusion matrix.

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook BreastCancerClassification.ipynb
```

Run the notebook cells from top to bottom. The model metrics are generated when you run it; results can vary if the data split or model settings are changed.

## Repository contents

- `BreastCancerClassification.ipynb` — analysis, model training, and evaluation.
- `requirements.txt` — Python package dependencies.
- `.gitignore` — common local and generated files to exclude.

## Limitations

This is a small benchmark dataset and a basic baseline model. Test-set metrics alone do not prove clinical validity, fairness, or generalization to real-world clinical settings.


Author
Preeti Dube

GitHub: PreetiDube
Project Repository: Breast Cancer Classification with Machine Learning
