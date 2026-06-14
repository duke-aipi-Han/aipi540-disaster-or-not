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
MODEL_DIR = Path("models/distilbert-disaster")
METRICS_PATH = Path("models/transformer_metrics.json")


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


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision_disaster": precision_score(labels, preds),
        "recall_disaster": recall_score(labels, preds),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

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

    train_dataset = TweetDataset(train_df["text_for_model"].tolist(), train_df["target"].tolist(), tokenizer)
    val_dataset = TweetDataset(val_df["text_for_model"].tolist(), val_df["target"].tolist(), tokenizer)

    training_args = TrainingArguments(
        output_dir="models/transformer-checkpoints",
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    predictions = trainer.predict(val_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = val_df["target"].to_numpy()
    metrics = {
        "model": args.model_name,
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
        "precision_disaster": float(precision_score(labels, preds)),
        "recall_disaster": float(recall_score(labels, preds)),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }

    stress_encodings = tokenizer(
        [item["text"] for item in STRESS_TESTS],
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    model.eval()
    with torch.no_grad():
        stress_preds = model(**stress_encodings).logits.argmax(dim=1).cpu().numpy()
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

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
