# Task 2: End-to-End ML Pipeline with Scikit-learn

## Overview

Built a production-ready ML pipeline for predicting customer churn using the Telco Customer Churn dataset.

## Implementation Details

### Data Processing

- Loaded telco customer churn dataset
- Converted TotalCharges to numeric format
- Removed customerID column
- Encoded target variable (Churn: Yes/No → 1/0)
- Split data into 80% training and 20% test sets

### Feature Engineering

- Identified 7 numeric columns
- Identified 13 categorical columns
- Created separate preprocessing pipelines for numeric and categorical data:
  - **Numeric**: Median imputation → StandardScaler
  - **Categorical**: Most frequent imputation → OneHotEncoder

### Model Development

- Built two complete pipelines with ColumnTransformer:
  1. **Logistic Regression Pipeline**: Preprocessor + LogisticRegression
  2. **Random Forest Pipeline**: Preprocessor + RandomForestClassifier (100 estimators)

### Model Optimization

- Performed hyperparameter tuning on Random Forest using GridSearchCV (5-fold CV)
- Tuned parameters:
  - n_estimators: [100, 200]
  - max_depth: [None, 10, 20]
  - min_samples_split: [2, 5]
- Used F1 score as optimization metric

### Model Persistence

- Saved the best-tuned pipeline using joblib
- Verified model persistence and reproducibility

## Output

- `churn_pipeline.pkl`: Best-trained Random Forest pipeline ready for production

## Key Technologies

- Pandas (data manipulation)
- Scikit-learn (ML pipeline, preprocessing, modeling, hyperparameter tuning)
- Joblib (model serialization)
