import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from src.disasters import STRESS_TESTS, combine_tweet_fields


DATA_PATH = Path("data/train.csv")


def model_slug(model_name: str) -> str:
    return (
        model_name.lower()
        .replace("/", "-")
        .replace("\\", "-")
        .replace("_", "-")
        .replace(".", "-")
    )


def model_output_paths(model_name: str) -> tuple[str, Path, Path, Path]:
    slug = model_slug(model_name)
    model_dir = Path("models/transformers") / slug
    metrics_path = Path("models") / f"transformer_metrics_{slug}.json"
    checkpoint_dir = Path("models/transformer-checkpoints") / slug
    return slug, model_dir, metrics_path, checkpoint_dir


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TweetDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        class_weights = None
        if self.class_weights is not None:
            class_weights = self.class_weights.to(device=outputs.logits.device, dtype=outputs.logits.dtype)
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision_disaster": precision_score(labels, preds),
        "recall_disaster": recall_score(labels, preds),
    }


def positive_probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    return probs[:, 1]


def tune_threshold(labels: np.ndarray, positive_probs: np.ndarray) -> tuple[float, dict]:
    best_threshold = 0.5
    best_metrics = {}
    best_f1 = -1.0

    for threshold in np.arange(0.20, 0.81, 0.01):
        preds = (positive_probs >= threshold).astype(int)
        current_f1 = f1_score(labels, preds)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = float(round(threshold, 2))
            best_metrics = {
                "accuracy": float(accuracy_score(labels, preds)),
                "f1": float(current_f1),
                "precision_disaster": float(precision_score(labels, preds, zero_division=0)),
                "recall_disaster": float(recall_score(labels, preds, zero_division=0)),
                "confusion_matrix": confusion_matrix(labels, preds).tolist(),
            }

    return best_threshold, best_metrics


def balanced_class_weights(labels: pd.Series, device: torch.device) -> torch.Tensor:
    counts = labels.value_counts().sort_index()
    total = counts.sum()
    weights = [total / (len(counts) * counts[label]) for label in sorted(counts.index)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable balanced class-weighted loss.")
    parser.add_argument("--output-name", default=None, help="Optional short name for the saved model folder.")
    args = parser.parse_args()
    device = choose_device(args.device)
    slug, model_dir, metrics_path, checkpoint_dir = model_output_paths(args.output_name or args.model_name)
    print(f"Using device: {device}")
    print(f"Saving model to: {model_dir}")
    print(f"Saving metrics to: {metrics_path}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    df = pd.read_csv(DATA_PATH)
    df["text_for_model"] = df.apply(
        lambda row: combine_tweet_fields(row["text"], row.get("keyword", ""), row.get("location", "")),
        axis=1,
    )

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["target"],
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model.to(device)

    train_dataset = TweetDataset(train_df["text_for_model"].tolist(), train_df["target"].tolist(), tokenizer)
    val_dataset = TweetDataset(val_df["text_for_model"].tolist(), val_df["target"].tolist(), tokenizer)

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        use_cpu=device.type == "cpu",
        fp16=args.fp16 and device.type == "cuda",
    )

    class_weights = None if args.no_class_weights else balanced_class_weights(train_df["target"], device)
    if class_weights is not None:
        print(f"Class weights: not disaster={class_weights[0]:.3f}, disaster={class_weights[1]:.3f}")

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )
    trainer.train()

    predictions = trainer.predict(val_dataset)
    labels = val_df["target"].to_numpy()
    positive_probs = positive_probabilities(predictions.predictions)
    default_preds = (positive_probs >= 0.5).astype(int)
    best_threshold, tuned_metrics = tune_threshold(labels, positive_probs)
    preds = (positive_probs >= best_threshold).astype(int)

    metrics = {
        "model": args.model_name,
        "model_slug": slug,
        "model_dir": str(model_dir),
        "decision_threshold": best_threshold,
        "class_weighted_loss": class_weights is not None,
        "default_threshold": {
            "accuracy": float(accuracy_score(labels, default_preds)),
            "f1": float(f1_score(labels, default_preds)),
            "precision_disaster": float(precision_score(labels, default_preds, zero_division=0)),
            "recall_disaster": float(recall_score(labels, default_preds, zero_division=0)),
            "confusion_matrix": confusion_matrix(labels, default_preds).tolist(),
        },
        **tuned_metrics,
    }

    stress_encodings = tokenizer(
        [item["text"] for item in STRESS_TESTS],
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    model.to(device)
    model.eval()
    stress_encodings = {key: value.to(device) for key, value in stress_encodings.items()}
    with torch.no_grad():
        stress_logits = model(**stress_encodings).logits.cpu().numpy()
        stress_preds = (positive_probabilities(stress_logits) >= best_threshold).astype(int)
    stress_targets = np.array([item["target"] for item in STRESS_TESTS])
    metrics["stress_accuracy"] = float(accuracy_score(stress_targets, stress_preds))
    metrics["stress_rows"] = [
        {
            "category": item["category"],
            "text": item["text"],
            "target": item["target"],
            "prediction": int(pred),
            "correct": bool(item["target"] == int(pred)),
        }
        for item, pred in zip(STRESS_TESTS, stress_preds)
    ]

    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
