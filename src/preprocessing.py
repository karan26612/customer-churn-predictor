import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path='data/telco_churn.csv'):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    # TotalCharges is loaded as object due to blank strings for new customers -> convert to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop rows with missing TotalCharges (very few rows, new customers with 0 tenure)
    df = df.dropna(subset=['TotalCharges'])

    # customerID is just an identifier, not predictive
    df = df.drop('customerID', axis=1)

    return df


def engineer_features(df):
    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # New feature: average charge per month of tenure
    df['AvgChargePerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)

    # Bucket tenure
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 60, 72],
                                 labels=['0-1yr', '1-2yr', '2-4yr', '4-5yr', '5-6yr'])

    # Binary Yes/No columns -> 0/1
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # One-hot encode remaining categorical columns
    df = pd.get_dummies(df, columns=['gender', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod', 'tenure_group'],
        drop_first=True)

    return df


def split_and_scale(df, test_size=0.2, random_state=42):
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    num_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgChargePerMonth']
    X_train[num_features] = scaler.fit_transform(X_train[num_features])
    X_test[num_features] = scaler.transform(X_test[num_features])

    return X_train, X_test, y_train, y_test, scaler
