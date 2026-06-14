import json
from pathlib import Path


METRIC_FILES = {
    "Baseline TF-IDF + Logistic Regression": Path("models/baseline_metrics.json"),
    "DistilBERT classifier": Path("models/transformer_metrics.json"),
}


def main() -> None:
    found = False
    for name, path in METRIC_FILES.items():
        if not path.exists():
            print(f"{name}: no metrics found at {path}")
            continue

        found = True
        metrics = json.loads(path.read_text())
        print(f"\n{name}")
        print(f"  F1: {metrics.get('f1', 0):.3f}")
        print(f"  Disaster recall: {metrics.get('recall_disaster', 0):.3f}")
        print(f"  Disaster precision: {metrics.get('precision_disaster', 0):.3f}")
        print(f"  Stress accuracy: {metrics.get('stress_accuracy', 0):.3f}")
        print(f"  Confusion matrix [[TN, FP], [FN, TP]]: {metrics.get('confusion_matrix')}")

        misses = [row for row in metrics.get("stress_rows", []) if not row["correct"]]
        if misses:
            print("  Stress misses:")
            for row in misses:
                print(f"    - {row['category']}: predicted {row['prediction']} for {row['text']}")

    if not found:
        print("No model metrics found. Run src/train_baseline.py first, then src/train_transformer.py if desired.")


if __name__ == "__main__":
    main()
