import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_FILE = Path(__file__).resolve().parent / "churn_data.csv"
MODEL_FILE = Path(__file__).resolve().parent / "churn_pipeline.joblib"


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0}).astype(int)

    if "SeniorCitizen" in df.columns and not np.issubdtype(df["SeniorCitizen"].dtype, np.number):
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce")

    return df


def build_pipeline(classifier=None) -> Pipeline:
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    categorical_features = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    if classifier is None:
        classifier = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline


def get_model_candidates() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            solver="liblinear", class_weight="balanced", random_state=42, max_iter=1000
        ),
        "random_forest": RandomForestClassifier(
            random_state=42, n_jobs=-1, class_weight="balanced"
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }


def get_search_spaces() -> dict:
    return {
        "logistic_regression": {
            "classifier__C": [0.01, 0.1, 1, 10],
        },
        "random_forest": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 10, 20, 30],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
        },
        "gradient_boosting": {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.01, 0.05, 0.1],
            "classifier__max_depth": [3, 5, 7],
        },
    }


def evaluate_model(model, X_valid, y_valid):
    y_pred = model.predict(X_valid)
    y_proba = model.predict_proba(X_valid)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_valid, y_pred),
        "precision": precision_score(y_valid, y_pred),
        "recall": recall_score(y_valid, y_pred),
        "f1_score": f1_score(y_valid, y_pred),
        "roc_auc": roc_auc_score(y_valid, y_proba) if y_proba is not None else None,
    }

    print("Evaluation results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")

    print("\nClassification report:")
    print(classification_report(y_valid, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_valid, y_pred))

    return metrics


def print_model_summary(results):
    summary = []
    for name, _, metrics, best_params in results:
        summary.append(
            {
                "Model": name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1 Score": metrics["f1_score"],
                "ROC AUC": metrics["roc_auc"],
                "Best Params": best_params,
            }
        )

    df_summary = pd.DataFrame(summary).sort_values(by="F1 Score", ascending=False)
    print("\nModel summary:")
    print(df_summary.to_string(index=False, float_format="{:.4f}".format))


def fit_candidate_model(name: str, estimator, X_train, y_train, X_valid, y_valid):
    print(f"\nTraining candidate: {name}")
    pipeline = build_pipeline(classifier=estimator)
    search_space = get_search_spaces()[name]

    total_combinations = int(np.prod([len(values) for values in search_space.values()]))
    if total_combinations <= 20:
        search = GridSearchCV(
            pipeline,
            param_grid=search_space,
            cv=5,
            scoring="f1",
            verbose=0,
            n_jobs=-1,
        )
    else:
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=search_space,
            n_iter=min(20, total_combinations),
            cv=5,
            scoring="f1",
            verbose=0,
            n_jobs=-1,
            random_state=42,
        )

    search.fit(X_train, y_train)

    print(f"Best parameters for {name}: {search.best_params_}")
    print(f"Best CV F1 for {name}: {search.best_score_:.4f}")

    best_model = search.best_estimator_
    metrics = evaluate_model(best_model, X_valid, y_valid)
    return name, best_model, metrics, search.best_params_


def train_and_tune_model(df: pd.DataFrame) -> Pipeline:
    df = preprocess_dataframe(df)

    if "Churn" not in df.columns:
        raise ValueError("The input dataset must contain a 'Churn' target column.")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    candidate_estimators = get_model_candidates()
    results = []

    for name, estimator in candidate_estimators.items():
        candidate_name, best_model, metrics, best_params = fit_candidate_model(
            name, estimator, X_train, y_train, X_valid, y_valid
        )
        results.append((candidate_name, best_model, metrics, best_params))

    print_model_summary(results)

    results.sort(key=lambda item: item[2]["f1_score"], reverse=True)
    best_name, best_model, best_metrics, _ = results[0]

    print(f"\nSelected best model: {best_name}")
    print(f"Best validation F1 score: {best_metrics['f1_score']:.4f}")

    return best_model


def save_model(model, path: Path):
    joblib.dump(model, path)
    print(f"Saved trained churn pipeline to: {path}")


def load_model(path: Path):
    return joblib.load(path)


def predict_new_customers(model, customers: pd.DataFrame) -> pd.Series:
    customers = preprocess_dataframe(customers)
    predictions = model.predict(customers)
    probabilities = model.predict_proba(customers)[:, 1] if hasattr(model, "predict_proba") else None
    result = pd.Series(predictions, index=customers.index, name="ChurnPrediction")
    if probabilities is not None:
        result = pd.DataFrame({"ChurnPrediction": predictions, "ChurnProbability": probabilities}, index=customers.index)
    return result


def main(dataset_path: Path = DATA_FILE, model_path: Path = MODEL_FILE):
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}. Please place the churn dataset in this folder or update the path."
        )

    data = load_data(dataset_path)
    model = train_and_tune_model(data)
    save_model(model, model_path)

    print("Pipeline training complete.")
    print("You can now load the pipeline and call predict_new_customers(model, new_data).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a customer churn prediction pipeline.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(DATA_FILE),
        help="Path to the churn dataset CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(MODEL_FILE),
        help="Path where the trained model pipeline will be saved.",
    )

    args = parser.parse_args()
    main(Path(args.data), Path(args.output))
