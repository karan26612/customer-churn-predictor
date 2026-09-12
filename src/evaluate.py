import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix)


def evaluate(name, y_test, y_pred, model, X_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    }


def compare_models(models_and_preds, y_test, X_test):
    # models_and_preds: list of (name, model, y_pred) tuples
    results = pd.DataFrame([
        evaluate(name, y_test, pred, model, X_test)
        for name, model, pred in models_and_preds
    ])
    return results.sort_values('F1', ascending=False).reset_index(drop=True)


def plot_confusion_matrices(models_and_preds, y_test, save_path=None):
    fig, axes = plt.subplots(1, len(models_and_preds), figsize=(5 * len(models_and_preds), 4))
    for ax, (name, _, pred) in zip(axes, models_and_preds):
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(name)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def plot_feature_importance(model, feature_names, top_n=15, save_path=None):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    plt.figure(figsize=(8, 6))
    importances.sort_values(ascending=False).head(top_n).plot(kind='barh')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
