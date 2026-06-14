import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.disasters import combine_tweet_fields
from src.train_transformer import positive_probabilities


TEST_PATH = Path("data/test.csv")
SAMPLE_SUBMISSION_PATH = Path("data/sample_submission.csv")
SUBMISSION_DIR = Path("submission")
BASELINE_MODEL_PATH = Path("models/tfidf_logreg.joblib")
BASELINE_METRICS_PATH = Path("models/baseline_metrics.json")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def transformer_metric_files() -> list[Path]:
    return sorted(Path("models").glob("transformer_metrics_*.json"))


def available_models() -> list[dict]:
    models = []

    if BASELINE_MODEL_PATH.exists() and BASELINE_METRICS_PATH.exists():
        models.append(
            {
                "kind": "baseline",
                "slug": "tfidf-logreg",
                "name": "TF-IDF + Logistic Regression",
                "metrics": load_json(BASELINE_METRICS_PATH),
                "model_path": BASELINE_MODEL_PATH,
            }
        )

    for metrics_path in transformer_metric_files():
        metrics = load_json(metrics_path)
        model_dir = Path(metrics.get("model_dir", ""))
        if not model_dir.exists():
            continue

        models.append(
            {
                "kind": "transformer",
                "slug": metrics.get("model_slug", model_dir.name),
                "name": metrics.get("model", model_dir.name),
                "metrics": metrics,
                "model_dir": model_dir,
                "metrics_path": metrics_path,
            }
        )

    return models


def choose_model(models: list[dict], metric: str, model_slug: str | None) -> dict:
    if not models:
        raise RuntimeError("No saved models found. Train a baseline or transformer model first.")

    if model_slug:
        for model in models:
            if model["slug"] == model_slug:
                return model
        choices = ", ".join(model["slug"] for model in models)
        raise RuntimeError(f"Model slug '{model_slug}' not found. Available slugs: {choices}")

    return max(models, key=lambda model: model["metrics"].get(metric, -1))


def load_test_texts() -> pd.DataFrame:
    df = pd.read_csv(TEST_PATH)
    df["text_for_model"] = df.apply(
        lambda row: combine_tweet_fields(row["text"], row.get("keyword", ""), row.get("location", "")),
        axis=1,
    )
    return df


def predict_baseline(model_info: dict, test_df: pd.DataFrame) -> list[int]:
    bundle = joblib.load(model_info["model_path"])
    model = bundle["model"]
    return model.predict(test_df["text_for_model"]).astype(int).tolist()


def predict_transformer(model_info: dict, test_df: pd.DataFrame, batch_size: int, device_name: str) -> list[int]:
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    threshold = float(model_info["metrics"].get("decision_threshold", 0.5))

    tokenizer = AutoTokenizer.from_pretrained(str(model_info["model_dir"]))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_info["model_dir"]))
    model.to(device)
    model.eval()

    predictions = []
    texts = test_df["text_for_model"].tolist()
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        encodings = {key: value.to(device) for key, value in encodings.items()}
        with torch.no_grad():
            logits = model(**encodings).logits.cpu().numpy()
        probs = positive_probabilities(logits)
        predictions.extend((probs >= threshold).astype(int).tolist())

    return predictions


def write_submission(test_df: pd.DataFrame, predictions: list[int], model_info: dict) -> Path:
    sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    submission = sample[["id"]].copy()
    submission["target"] = predictions

    if len(submission) != len(test_df):
        raise RuntimeError(
            f"Submission row count {len(submission)} does not match test row count {len(test_df)}."
        )

    SUBMISSION_DIR.mkdir(exist_ok=True)
    output_path = SUBMISSION_DIR / f"submission_{model_info['slug']}.csv"
    submission.to_csv(output_path, index=False)

    latest_path = SUBMISSION_DIR / "submission.csv"
    submission.to_csv(latest_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="f1", choices=["f1", "recall_disaster", "precision_disaster", "accuracy"])
    parser.add_argument("--model-slug", default=None, help="Optional exact model slug to use.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    models = available_models()
    model_info = choose_model(models, args.metric, args.model_slug)
    test_df = load_test_texts()

    print(f"Selected model: {model_info['name']} ({model_info['slug']})")
    print(f"Selection metric: {args.metric}={model_info['metrics'].get(args.metric)}")

    if model_info["kind"] == "baseline":
        predictions = predict_baseline(model_info, test_df)
    else:
        predictions = predict_transformer(model_info, test_df, args.batch_size, args.device)

    output_path = write_submission(test_df, predictions, model_info)
    print(f"Wrote {output_path}")
    print(f"Wrote {SUBMISSION_DIR / 'submission.csv'}")
    print(f"Predicted disaster rate: {sum(predictions) / len(predictions):.3f}")


if __name__ == "__main__":
    main()
