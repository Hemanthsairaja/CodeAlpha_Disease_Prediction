# CodeAlpha Task 4 — Disease Prediction from Medical Data

## Objective
Classify breast tumor samples as malignant (0) or benign (1).

## Dataset
Wisconsin Diagnostic Breast Cancer dataset, bundled with scikit-learn.
It has 569 examples and 30 numeric features. No patient identifying data is used.

## Method
Stratified 80/20 train-test split, training-only standardization via pipeline,
Logistic Regression, accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
For precision/recall/F1/ROC-AUC, malignant (0) is treated as the positive class.

## Run
```bash
pip install -r requirements.txt
python main.py
python main.py --demo
```

## Output
`outputs/disease_model.joblib`, `metrics.json`,
`classification_report.txt`, `confusion_matrix.png`.

## Limitations
This is a classroom classification project, **not** a clinical diagnosis
or validated medical device. A small benchmark dataset cannot establish
performance on real patients.
