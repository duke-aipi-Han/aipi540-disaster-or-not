import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import streamlit as st

from src.disasters import EXAMPLE_TWEETS, Prediction, heuristic_predict, simple_explanation


BASELINE_MODEL_PATH = Path("models/tfidf_logreg.joblib")
BASELINE_METRICS_PATH = Path("models/baseline_metrics.json")
TRANSFORMER_ROOT = Path("models/transformers")
LEGACY_TRANSFORMER_MODEL_DIR = Path("models/distilbert-disaster")
LEGACY_TRANSFORMER_METRICS_PATH = Path("models/transformer_metrics.json")


st.set_page_config(
    page_title="Disaster Tweet Evaluator",
    page_icon="DT",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_baseline_model():
    return joblib.load(BASELINE_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_transformer_model(model_dir: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def discover_transformers() -> list[dict]:
    artifacts = []

    if TRANSFORMER_ROOT.exists():
        for model_dir in sorted(path for path in TRANSFORMER_ROOT.iterdir() if path.is_dir()):
            metrics_path = Path("models") / f"transformer_metrics_{model_dir.name}.json"
            if (model_dir / "config.json").exists():
                metrics = read_json(metrics_path)
                artifacts.append(
                    {
                        "name": metrics.get("model", model_dir.name),
                        "model_dir": model_dir,
                        "metrics_path": metrics_path,
                        "metrics": metrics,
                    }
                )

    if LEGACY_TRANSFORMER_MODEL_DIR.exists() and (LEGACY_TRANSFORMER_MODEL_DIR / "config.json").exists():
        metrics = read_json(LEGACY_TRANSFORMER_METRICS_PATH)
        artifacts.append(
            {
                "name": metrics.get("model", "distilbert-base-uncased"),
                "model_dir": LEGACY_TRANSFORMER_MODEL_DIR,
                "metrics_path": LEGACY_TRANSFORMER_METRICS_PATH,
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
    if BASELINE_MODEL_PATH.exists():
        predictions.append(baseline_predict(text))

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


def show_model_metrics() -> None:
    rows = []
    if BASELINE_METRICS_PATH.exists():
        rows.append(metric_row("TF-IDF + Logistic Regression", read_json(BASELINE_METRICS_PATH)))

    for artifact in discover_transformers():
        rows.append(metric_row(artifact["name"], artifact["metrics"]))

    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.caption("No saved metrics found yet.")


def main() -> None:
    st.title("Disaster Tweet Evaluator")
    st.caption("Compare the baseline and every saved transformer model on the same tweet.")

    with st.sidebar:
        st.header("Saved Metrics")
        show_model_metrics()

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
