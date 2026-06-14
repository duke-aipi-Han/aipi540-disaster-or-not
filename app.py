from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import streamlit as st

from src.disasters import EXAMPLE_TWEETS, LABELS, Prediction, heuristic_predict, simple_explanation


BASELINE_MODEL_PATH = Path("models/tfidf_logreg.joblib")
BASELINE_METRICS_PATH = Path("models/baseline_metrics.json")
TRANSFORMER_MODEL_DIR = Path("models/distilbert-disaster")
TRANSFORMER_METRICS_PATH = Path("models/transformer_metrics.json")


st.set_page_config(
    page_title="Disaster Tweet Evaluator",
    page_icon="DT",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def load_baseline_model():
    return joblib.load(BASELINE_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_transformer_pipeline():
    from transformers import pipeline

    return pipeline("text-classification", model=str(TRANSFORMER_MODEL_DIR), tokenizer=str(TRANSFORMER_MODEL_DIR))


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


def transformer_predict(text: str) -> Prediction:
    classifier = load_transformer_pipeline()
    result = classifier(text, truncation=True)[0]
    raw_label = result["label"].upper()
    label = 1 if raw_label in {"LABEL_1", "1", "REAL DISASTER", "DISASTER"} else 0
    return Prediction(
        label=label,
        confidence=float(result["score"]),
        explanation=simple_explanation(text),
        model_name="DistilBERT classifier",
    )


def available_models() -> list[str]:
    models = []
    if BASELINE_MODEL_PATH.exists():
        models.append("Baseline")
    if TRANSFORMER_MODEL_DIR.exists():
        models.append("Transformer")
    models.append("Keyword fallback")
    return models


def predict(text: str, model_choice: str) -> Prediction:
    if model_choice == "Baseline" and BASELINE_MODEL_PATH.exists():
        return baseline_predict(text)
    if model_choice == "Transformer" and TRANSFORMER_MODEL_DIR.exists():
        return transformer_predict(text)
    return heuristic_predict(text)


def show_metric_file(path: Path, title: str) -> None:
    if not path.exists():
        st.caption(f"{title}: not trained yet.")
        return

    import json

    metrics = json.loads(path.read_text())
    st.caption(
        f"{title}: F1 {metrics.get('f1', 0):.3f}, "
        f"disaster recall {metrics.get('recall_disaster', 0):.3f}, "
        f"stress accuracy {metrics.get('stress_accuracy', 0):.3f}"
    )


def main() -> None:
    st.title("Disaster Tweet Evaluator")
    st.caption("Classify whether a social post reports a real disaster, then inspect where evaluation gets tricky.")

    example_name = st.selectbox("Example tweet", list(EXAMPLE_TWEETS.keys()))
    default_text = EXAMPLE_TWEETS[example_name]
    tweet = st.text_area("Tweet text", value=default_text, height=120)
    model_choice = st.radio("Model", available_models(), horizontal=True)

    if st.button("Classify", type="primary") or tweet:
        prediction = predict(tweet, model_choice)
        st.metric("Prediction", prediction.label_name, f"{prediction.confidence:.1%} confidence")
        st.progress(prediction.confidence)
        st.write(f"Model: {prediction.model_name}")
        st.write(f"Explanation: {prediction.explanation}")

        if prediction.label == 1:
            st.warning("Treat this as potentially emergency-relevant.")
        else:
            st.success("Likely not an emergency report.")

if __name__ == "__main__":
    main()
