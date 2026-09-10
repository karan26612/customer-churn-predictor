# Customer Churn Prediction

## Problem
Predict whether a telecom customer will churn using historical account and
service data, to help the business proactively identify and retain at-risk
customers.

## Dataset
Telco Customer Churn dataset — 7,043 customers, 21 features (demographics,
account info, subscribed services). Source: IBM sample dataset.

## Approach
1. Cleaned data (fixed `TotalCharges` type, handled missing values)
2. EDA to understand churn drivers (contract type, tenure, monthly charges)
3. Feature engineering (avg charge per month, tenure buckets, one-hot encoding)
4. Trained 3 models: Logistic Regression, Decision Tree, Random Forest
5. Evaluated using Accuracy, Precision, Recall, F1, ROC-AUC (accuracy alone
   is misleading here due to ~27% class imbalance)

## Key Findings
- Month-to-month contract customers churn significantly more than customers
  on 1-2 year contracts
- Low tenure combined with high monthly charges is the strongest churn signal
- Fiber optic internet customers show higher churn than DSL customers

## Results


| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.797 | 0.642 | 0.537 | 0.585 | 0.838 |
| Decision Tree | 0.786 | 0.612 | 0.532 | 0.569 | 0.817 |
| Random Forest | 0.790 | 0.634 | 0.500 | 0.559 | 0.834 |

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## Project Structure
```
├── data/
│   └── telco_churn.csv
├── notebook.ipynb
├── requirements.txt
└── README.md
```

## Future Improvements
- Try XGBoost / LightGBM
- Hyperparameter tuning with GridSearchCV
- Deploy as a Streamlit app for live predictions
