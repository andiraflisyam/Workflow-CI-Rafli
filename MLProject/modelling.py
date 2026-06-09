import os
import argparse
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

TRAIN_PATH = os.path.join('heart_disease_preprocessing', 'heart_disease_preprocessing_train.csv')
TEST_PATH  = os.path.join('heart_disease_preprocessing', 'heart_disease_preprocessing_test.csv')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators",      type=int, default=100)
    parser.add_argument("--max_depth",         type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--random_state",      type=int, default=42)
    return parser.parse_args()

def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)
    X_train  = train_df.drop('condition', axis=1)
    y_train  = train_df['condition']
    X_test   = test_df.drop('condition', axis=1)
    y_test   = test_df['condition']
    print(f"Data latih : {X_train.shape}")
    print(f"Data uji   : {X_test.shape}")
    return X_train, X_test, y_train, y_test

def save_confusion_matrix(y_test, y_pred):
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Tidak Sakit', 'Sakit Jantung'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=120)
    plt.close()

def save_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    fig, ax     = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(importances)), importances[indices], color='steelblue')
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.set_title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=120)
    plt.close()

def save_classification_report(y_test, y_pred):
    report = classification_report(y_test, y_pred, target_names=['Tidak Sakit', 'Sakit Jantung'])
    with open('classification_report.txt', 'w') as f:
        f.write(report)

def main():
    args = parse_args()
    print("=" * 60)
    print("  WORKFLOW CI - Heart Disease")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_data()

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=args.random_state
    )

    with mlflow.start_run():
        model.fit(X_train, y_train)
        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        mlflow.log_param("n_estimators",      args.n_estimators)
        mlflow.log_param("max_depth",         args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("random_state",      args.random_state)

        acc       = accuracy_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        auc       = roc_auc_score(y_test, y_pred_prob)

        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("f1_score",  f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall",    recall)
        mlflow.log_metric("roc_auc",   auc)

        mlflow.sklearn.log_model(model, "model")

        save_confusion_matrix(y_test, y_pred)
        save_feature_importance(model, list(X_train.columns))
        save_classification_report(y_test, y_pred)

        mlflow.log_artifact('confusion_matrix.png')
        mlflow.log_artifact('feature_importance.png')
        mlflow.log_artifact('classification_report.txt')

        mlflow.set_tag("model_name", "RandomForestClassifier")
        mlflow.set_tag("dataset",    "Heart Disease UCI")
        mlflow.set_tag("author",     "Andi Rafli")

        print(f"\nAccuracy : {acc:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC AUC  : {auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=['Tidak Sakit', 'Sakit Jantung']))

if __name__ == '__main__':
    main()
