from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

from src.preprocessing import load_data, clean_data, engineer_features, split_and_scale


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, balanced=False):
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42,
        class_weight='balanced' if balanced else None
    )
    model.fit(X_train, y_train)
    return model


if __name__ == '__main__':
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)

    log_reg = train_logistic_regression(X_train, y_train)
    dt = train_decision_tree(X_train, y_train)
    rf = train_random_forest(X_train, y_train)

    joblib.dump(rf, 'models/churn_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Models trained and saved.")
