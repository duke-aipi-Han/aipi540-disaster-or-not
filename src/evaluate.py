import json
from pathlib import Path


def metric_files() -> list[tuple[str, Path]]:
    files = [("Baseline TF-IDF + Logistic Regression", Path("models/baseline_metrics.json"))]

    legacy_path = Path("models/transformer_metrics.json")
    if legacy_path.exists():
        files.append(("Legacy transformer", legacy_path))

    for path in sorted(Path("models").glob("transformer_metrics_*.json")):
        try:
            metrics = json.loads(path.read_text())
            name = metrics.get("model", path.stem.replace("transformer_metrics_", ""))
        except json.JSONDecodeError:
            name = path.stem.replace("transformer_metrics_", "")
        files.append((name, path))

    return files


def main() -> None:
    found = False
    for name, path in metric_files():
        if not path.exists():
            print(f"{name}: no metrics found at {path}")
            continue

        found = True
        metrics = json.loads(path.read_text())
        print(f"\n{name}")
        if "decision_threshold" in metrics:
            print(f"  Decision threshold: {metrics.get('decision_threshold'):.2f}")
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
