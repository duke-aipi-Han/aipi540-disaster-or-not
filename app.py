import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import streamlit as st

from src.disasters import EXAMPLE_TWEETS, Prediction, heuristic_predict, simple_explanation


HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "hw391/disaster-or-not-tweet-model")
LOCAL_MODEL_ROOT = PROJECT_ROOT / "models"
BASELINE_MODEL_PATH = LOCAL_MODEL_ROOT / "tfidf_logreg.joblib"
BASELINE_METRICS_PATH = LOCAL_MODEL_ROOT / "baseline_metrics.json"
TRANSFORMER_ROOT = LOCAL_MODEL_ROOT / "transformers"
LEGACY_TRANSFORMER_MODEL_DIR = LOCAL_MODEL_ROOT / "distilbert-disaster"
LEGACY_TRANSFORMER_METRICS_PATH = LOCAL_MODEL_ROOT / "transformer_metrics.json"
REMOTE_MODEL_SUBFOLDERS = [
    "cardiffnlp-twitter-roberta-base",
    "distilbert-base-uncased",
]
REMOTE_IGNORE_PATTERNS = [
    "models/transformer-checkpoints/**",
    "**/optimizer.pt",
    "**/scheduler.pt",
    "**/rng_state.pth",
    "**/scaler.pt",
]


st.set_page_config(
    page_title="Disaster Tweet Evaluator",
    page_icon="DT",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_baseline_model():
    return joblib.load(resolve_baseline_model_path())


@st.cache_resource(show_spinner=False)
def load_transformer_model(model_dir: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if model_dir.startswith("hf://"):
        model_dir = str(download_remote_transformer_model(model_dir.removeprefix("hf://")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model, device


@st.cache_resource(show_spinner="Downloading transformer model from Hugging Face...")
def download_remote_transformer_model(model_subfolder: str) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        allow_patterns=[f"{model_subfolder}/**"],
        ignore_patterns=REMOTE_IGNORE_PATTERNS,
    )
    return Path(snapshot_path) / model_subfolder


def artifact_roots() -> list[Path]:
    return [PROJECT_ROOT]


def first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def resolve_baseline_model_path() -> Path:
    path = first_existing_path(
        [root / "models" / "tfidf_logreg.joblib" for root in artifact_roots()]
        + [root / "tfidf_logreg.joblib" for root in artifact_roots()]
    )
    if path is None:
        raise FileNotFoundError(f"Could not find tfidf_logreg.joblib locally or in {HF_MODEL_REPO}.")
    return path


def resolve_baseline_metrics_path() -> Path | None:
    return first_existing_path(
        [root / "models" / "baseline_metrics.json" for root in artifact_roots()]
        + [root / "baseline_metrics.json" for root in artifact_roots()]
    )


def metric_path_for_model(root: Path, transformer_root: Path, model_name: str) -> Path | None:
    return first_existing_path(
        [
            root / "models" / f"transformer_metrics_{model_name}.json",
            root / f"transformer_metrics_{model_name}.json",
            transformer_root.parent / f"transformer_metrics_{model_name}.json",
        ]
    )


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def discover_transformers() -> list[dict]:
    artifacts = []
    seen_dirs = set()
    seen_model_names = set()

    for root in artifact_roots():
        for transformer_root in [root / "models" / "transformers", root / "transformers", root]:
            if not transformer_root.exists():
                continue
            for model_dir in sorted(path for path in transformer_root.iterdir() if path.is_dir()):
                if model_dir.resolve() in seen_dirs or model_dir.name in seen_model_names:
                    continue
                seen_dirs.add(model_dir.resolve())
                metrics_path = metric_path_for_model(root, transformer_root, model_dir.name)
                metrics = read_json(metrics_path) if metrics_path is not None else {}
                if (model_dir / "config.json").exists():
                    seen_model_names.add(model_dir.name)
                    artifacts.append(
                        {
                            "name": metrics.get("model", model_dir.name),
                            "model_dir": model_dir,
                            "metrics_path": metrics_path,
                            "metrics": metrics,
                        }
                    )

        if (root / "config.json").exists() and root.resolve() not in seen_dirs:
            seen_dirs.add(root.resolve())
            seen_model_names.add(root.name)
            metrics_path = first_existing_path([root / "transformer_metrics.json"])
            metrics = read_json(metrics_path) if metrics_path is not None else {}
            artifacts.append(
                {
                    "name": metrics.get("model", root.name),
                    "model_dir": root,
                    "metrics_path": metrics_path,
                    "metrics": metrics,
                }
            )

    legacy_metrics_path = first_existing_path(
        [root / "models" / "transformer_metrics.json" for root in artifact_roots()]
        + [root / "transformer_metrics.json" for root in artifact_roots()]
    )
    for root in artifact_roots():
        legacy_dir = first_existing_path([root / "models" / "distilbert-disaster", root / "distilbert-disaster"])
        if legacy_dir is not None and legacy_dir.resolve() not in seen_dirs and legacy_dir.name not in seen_model_names:
            seen_dirs.add(legacy_dir.resolve())
            if (legacy_dir / "config.json").exists():
                seen_model_names.add(legacy_dir.name)
                metrics = read_json(legacy_metrics_path) if legacy_metrics_path is not None else {}
                artifacts.append(
                    {
                        "name": metrics.get("model", "distilbert-base-uncased"),
                        "model_dir": legacy_dir,
                        "metrics_path": legacy_metrics_path,
                        "metrics": metrics,
                    }
                )

    for model_subfolder in REMOTE_MODEL_SUBFOLDERS:
        if model_subfolder in seen_model_names:
            continue
        metrics_path = first_existing_path(
            [root / "models" / f"transformer_metrics_{model_subfolder}.json" for root in artifact_roots()]
            + [root / f"transformer_metrics_{model_subfolder}.json" for root in artifact_roots()]
        )
        metrics = read_json(metrics_path) if metrics_path is not None else {}
        artifacts.append(
            {
                "name": metrics.get("model", model_subfolder),
                "model_dir": f"hf://{model_subfolder}",
                "metrics_path": metrics_path,
                "metrics": metrics,
            }
        )

    return artifacts


def baseline_predict(text: str) -> Prediction:
    bundle = load_baseline_model()
    model = bundle["model"]
    proba = model.predict_proba([text])[0]
    label = int(proba.argmax())
    confidence = float(proba[label])

    explanation = simple_explanation(text)
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    row = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    contributions = row.multiply(classifier.coef_[0]).tocsr()
    scored_terms = []
    for idx in contributions.nonzero()[1]:
        scored_terms.append((feature_names[idx], float(contributions[0, idx])))
    scored_terms = sorted(scored_terms, key=lambda item: abs(item[1]), reverse=True)
    if scored_terms:
        explanation = simple_explanation(text, [term for term, _ in scored_terms[:6]])

    return Prediction(
        label=label,
        confidence=confidence,
        explanation=explanation,
        model_name="TF-IDF + Logistic Regression",
    )


def transformer_predict(text: str, artifact: dict) -> Prediction:
    import torch

    metrics = artifact["metrics"]
    threshold = float(metrics.get("decision_threshold", 0.5))
    tokenizer, model, device = load_transformer_model(str(artifact["model_dir"]))
    inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1)[0]

    disaster_probability = float(probs[1].cpu())
    label = 1 if disaster_probability >= threshold else 0
    confidence = disaster_probability if label == 1 else 1.0 - disaster_probability

    return Prediction(
        label=label,
        confidence=confidence,
        explanation=simple_explanation(text),
        model_name=f"{artifact['name']} (threshold {threshold:.2f})",
    )


def all_predictions(text: str) -> list[Prediction]:
    predictions = []
    try:
        predictions.append(baseline_predict(text))
    except FileNotFoundError:
        pass

    for artifact in discover_transformers():
        try:
            predictions.append(transformer_predict(text, artifact))
        except Exception as exc:
            predictions.append(
                Prediction(
                    label=0,
                    confidence=0.0,
                    explanation=f"Could not load model: {exc}",
                    model_name=str(artifact["name"]),
                )
            )

    predictions.append(heuristic_predict(text))
    return predictions


def metric_row(name: str, metrics: dict) -> dict:
    return {
        "Model": name,
        "F1": metrics.get("f1"),
        "Recall": metrics.get("recall_disaster"),
        "Precision": metrics.get("precision_disaster"),
        "Stress": metrics.get("stress_accuracy"),
        "Threshold": metrics.get("decision_threshold"),
    }

def main() -> None:
    st.title("Disaster or Not Tweet Evaluator")
    st.caption("See how different models classify tweets about disasters.")

    example_name = st.selectbox("Example tweet", list(EXAMPLE_TWEETS.keys()))
    tweet = st.text_area("Tweet text", value=EXAMPLE_TWEETS[example_name], height=120)

    if st.button("Compare Models", type="primary"):
        predictions = all_predictions(tweet)
        rows = [
            {
                "Model": prediction.model_name,
                "Prediction": prediction.label_name,
                "Confidence": f"{prediction.confidence:.1%}",
                "Explanation": prediction.explanation,
            }
            for prediction in predictions
        ]
        st.dataframe(rows, hide_index=True, use_container_width=True)

        disaster_votes = sum(prediction.label for prediction in predictions)
        if disaster_votes:
            st.warning(f"{disaster_votes} of {len(predictions)} models flagged this as disaster-relevant.")
        else:
            st.success("No model flagged this as a real disaster.")


if __name__ == "__main__":
    main()
