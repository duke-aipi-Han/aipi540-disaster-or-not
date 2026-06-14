---
title: AIPI540 Disaster Tweets
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
app_port: 8501
tags:
- streamlit
- nlp
- classification
pinned: false
short_description: Evaluate disaster tweet classifiers with stress tests
license: apache-2.0
---

# Disaster Tweet Evaluator

Quick AIPI540 NLP evaluation prototype for Kaggle's **Natural Language Processing with Disaster Tweets** competition.

The task is to classify whether a social post is about a real disaster. Misinterpretation matters because emergency-response systems may waste attention on jokes, metaphors, sarcasm, or irrelevant posts.

## Models

1. **Baseline:** TF-IDF + logistic regression.
2. **Transfer learning:** DistilBERT sequence classifier.
3. **Fallback:** simple keyword heuristic so the Streamlit app still runs before training.

## Evaluation Focus

The app and scripts report:

- F1 score
- Recall on real disasters
- Precision on real disasters
- Stress-test accuracy on metaphor, sarcasm, slang, hashtags, emojis, all-caps alerts, and location-heavy examples

Pitch angle: **Evaluation reveals the model confuses metaphorical disaster language with real-world emergency reports.**

## Project Structure

```text
aipi540-disaster-tweets/
  assignment.md
  requirements.txt
  requirements-cuda-cu128.txt
  data/
    train.csv
    test.csv
    sample_submission.csv
  models/
  src/
    disasters.py
    evaluate.py
    train_baseline.py
    train_transformer.py
```

## Local Setup

Use Python 3.13 in the repo-local virtual environment.

```bash
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python src\train_baseline.py
.venv\Scripts\python src\evaluate.py
.venv\Scripts\streamlit run app.py
```

Optional transfer-learning run:

```bash
.venv\Scripts\python src\train_transformer.py --model-name distilbert-base-uncased --epochs 2 --batch-size 16
.venv\Scripts\python src\train_transformer.py --model-name cardiffnlp/twitter-roberta-base --device cuda --fp16 --epochs 3 --batch-size 8
.venv\Scripts\python src\evaluate.py
```

The transformer script saves each base model separately under `models/transformers/<model-name>/` and writes matching metric files like `models/transformer_metrics_cardiffnlp-twitter-roberta-base.json`. It uses balanced class-weighted loss by default and tunes the final disaster decision threshold on the validation split. This is useful because the assignment cares about missed real disasters, not accuracy alone.

Generate a Kaggle submission from the best saved model:

```bash
.venv\Scripts\python src\generate_submission.py
```

This writes `submission/submission.csv` plus a model-specific copy such as `submission/submission_cardiffnlp-twitter-roberta-base.csv`.

## Optional CUDA Training

The default `requirements.txt` keeps the app portable for Hugging Face Spaces CPU hosting. For local NVIDIA GPU training, install a CUDA PyTorch wheel before the main requirements:

```bash
.venv\Scripts\python -m pip install --force-reinstall -r requirements-cuda-cu128.txt
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

If CUDA is available, `src\train_transformer.py` uses it by default. You can force a device when needed:

```bash
.venv\Scripts\python src\train_transformer.py --device cuda --epochs 2 --batch-size 16
.venv\Scripts\python src\train_transformer.py --device cpu --epochs 2 --batch-size 16
```

Mixed precision can speed up supported NVIDIA GPUs:

```bash
.venv\Scripts\python src\train_transformer.py --device cuda --fp16 --epochs 2 --batch-size 16
```

The baseline script writes:

```text
models/tfidf_logreg.joblib
models/baseline_metrics.json
```

The transformer script writes:

```text
models/transformers/<model-name>/
models/transformer_metrics_<model-name>.json
```

## Dataset

Download Kaggle's competition files from:

https://www.kaggle.com/competitions/nlp-getting-started/overview

Expected local files:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
```

`train.csv` contains 7,613 labeled tweets with columns `id`, `keyword`, `location`, `text`, and `target`.

## Hugging Face Spaces

The app is deployed to a Docker-based Hugging Face Space:

https://huggingface.co/spaces/hw391/AIPI540-disaster-or-not

Large transformer model files are not committed to this repo or the Space repo. The app loads them
from the public model repository at runtime:

https://huggingface.co/hw391/disaster-or-not-tweet-model

The Space deploy includes only runtime code, the Dockerfile, requirements, the small TF-IDF baseline,
and metrics JSON files. To deploy the current branch to the Space, authenticate with Hugging Face once:

```powershell
hf auth login
```

Then run:

```powershell
.\scripts\deploy_hf_space.ps1
```

Optional parameters:

```powershell
.\scripts\deploy_hf_space.ps1 `
  -SpaceRepo "hw391/AIPI540-disaster-or-not" `
  -ModelRepo "hw391/disaster-or-not-tweet-model" `
  -CommitMessage "Deploy latest app"
```
