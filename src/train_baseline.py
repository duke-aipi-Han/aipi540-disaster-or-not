import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.disasters import STRESS_TESTS, combine_tweet_fields


DATA_PATH = Path("data/train.csv")
MODEL_PATH = Path("models/tfidf_logreg.joblib")
METRICS_PATH = Path("models/baseline_metrics.json")


AUGMENTED_ROWS = [
    {"text": "this exam was a disaster and my brain is fried", "target": 0, "keyword": "disaster", "location": ""},
    {"text": "my fantasy team got destroyed this week lol", "target": 0, "keyword": "destroyed", "location": ""},
    {"text": "downtown bridge collapsed after flooding, avoid the area", "target": 1, "keyword": "flood", "location": "downtown"},
    {"text": "TORNADO WARNING issued for county residents take shelter now", "target": 1, "keyword": "tornado", "location": "county"},
    {"text": "phone battery at 1 percent this is an emergency", "target": 0, "keyword": "emergency", "location": ""},
    {"text": "smoke and flames visible near the school, crews responding", "target": 1, "keyword": "fire", "location": "school"},
]


def load_training_data(use_augmentation: bool) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["text_for_model"] = df.apply(
        lambda row: combine_tweet_fields(row["text"], row.get("keyword", ""), row.get("location", "")),
        axis=1,
    )

    if use_augmentation:
        aug = pd.DataFrame(AUGMENTED_ROWS)
        aug["text_for_model"] = aug.apply(
            lambda row: combine_tweet_fields(row["text"], row.get("keyword", ""), row.get("location", "")),
            axis=1,
        )
        df = pd.concat([df, aug], ignore_index=True)

    return df


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    strip_accents="unicode",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_predictions(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision_disaster": float(precision_score(y_true, y_pred)),
        "recall_disaster": float(recall_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def evaluate_stress_tests(model: Pipeline) -> dict:
    texts = [item["text"] for item in STRESS_TESTS]
    targets = [item["target"] for item in STRESS_TESTS]
    preds = model.predict(texts)
    rows = []
    for item, pred in zip(STRESS_TESTS, preds):
        rows.append(
            {
                "category": item["category"],
                "text": item["text"],
                "target": item["target"],
                "prediction": int(pred),
                "correct": bool(item["target"] == int(pred)),
            }
        )
    return {
        "stress_accuracy": float(accuracy_score(targets, preds)),
        "stress_rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-augmentation", action="store_true", help="Train only on Kaggle rows.")
    args = parser.parse_args()

    Path("models").mkdir(exist_ok=True)
    df = load_training_data(use_augmentation=not args.no_augmentation)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["target"],
    )

    model = build_model()
    model.fit(train_df["text_for_model"], train_df["target"])

    val_pred = model.predict(val_df["text_for_model"])
    metrics = evaluate_predictions(val_df["target"], val_pred)
    stress = evaluate_stress_tests(model)
    metrics.update(stress)
    metrics["model"] = "tfidf_logistic_regression"
    metrics["train_rows"] = int(len(train_df))
    metrics["validation_rows"] = int(len(val_df))
    metrics["augmentation_used"] = not args.no_augmentation

    joblib.dump({"model": model, "metrics": metrics}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
