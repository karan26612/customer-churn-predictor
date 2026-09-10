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
_Run `notebook.ipynb` and paste your own numbers here — they'll vary
slightly by environment/library version._

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | | | | | |
| Decision Tree | | | | | |
| Random Forest | | | | | |

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
