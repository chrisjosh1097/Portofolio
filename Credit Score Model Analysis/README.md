# Credit Score Classification

This project aims to predict the credit score category of individuals using various classification algorithms. 
The dataset used is from Kaggle: [Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification).

## Dataset

The dataset contains individual-level financial and demographic information with a target variable `Credit_Score` having three classes:
- Poor
- Standard
- Good

### Features:
- Age
- Monthly Income
- Number of Bank Accounts
- Number of Credit Cards
- Outstanding Debt
- Credit History Age
- etc

##  Objective

The goal is to classify individuals into one of the credit score categories using machine learning models, which are:

- Multinomial Logistic Regression
- Linear Discriminant Analysis (LDA)
- Random Forest
- XGBoost

## Data Preprocessing:

The data preprocessing steps include:
- Handling missing values
- Feature engineering and encoding categorical variables
- Balancing the dataset using oversampling techniques
- Feature scaling and dimensionality reduction (PCA)

## Hyperparameter Tuning

- Random Forest and XGBoost were tuned using `GridSearchCV` for optimal parameters.
- Multinom and LDA used standard settings for baseline comparison.

## Evaluation Metrics

- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- ROC Curve and AUC
- Cross-validation (K-Fold with k=5)

## Results
- XGBoost performed the best in terms of accuracy and stability across folds.
- Multinom and LDA gave competitive results with interpretability benefits.

## Visualizations

- Feature distributions and pairplots for EDA
- Confusion matrices and ROC curves for each model
- Bar charts for model comparisons
