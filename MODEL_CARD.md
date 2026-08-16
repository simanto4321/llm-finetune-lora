# Model Card: DistilBERT + LoRA Sentiment Classifier

## Model details

| Field | Value |
|-------|-------|
| **Model name** | DistilBERT LoRA sentiment (binary) |
| **Developed by** | Mehedi Ashraf Simanto |
| **Model type** | Sequence classification (LoRA adapter on DistilBERT) |
| **Base model** | [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) |
| **Language** | English |
| **License** | MIT (this adapter + training code); base model subject to its own license |
| **Library** | Transformers + PEFT (LoRA) |
| **Repository** | https://github.com/simanto4321/llm-finetune-lora |

### Architecture

- Base encoder: DistilBERT (`distilbert-base-uncased`), 2 labels (`0` negative, `1` positive).
- Adaptation: LoRA on attention projections `q_lin` and `v_lin` (`r=8`, `alpha=16`, `dropout=0.1`).
- Only ~1% of parameters are trainable; the bundled artifact under `output/lora_adapter/` stores adapter weights + tokenizer files, not a full base checkpoint.

## Intended use

**Direct use:** Classify short English product/review-style text as positive or negative sentiment via `inference.py`.

**Out of scope:** Multilingual sentiment, sarcasm-heavy or domain-shifted text (medical, legal, social media slang), toxicity moderation, or any high-stakes automated decision without human review.

## Training data

- File: `data/sentiment.jsonl` (66 labeled examples).
- Format: `{"text": "...", "label": 0|1}` with stratified 80/20 train/validation split (`seed=42`).
- Synthetic / curated short reviews for demo and reproducibility—not a production-scale corpus.

See `run_manifest.json` / `metrics.json` → `reproducibility.dataset` for SHA-256 and size after a training run.

## Training procedure

```bash
pip install -r requirements.txt
python train.py
# smoke / CPU: python train.py --fast
```

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 3 |
| Batch size | 8 |
| Learning rate | 2e-4 |
| Max sequence length | 96 |
| Seed | 42 |
| LoRA r / alpha / dropout | 8 / 16 / 0.1 |

Hardware for the published metrics run: see `metrics.json` → `reproducibility.device` (CPU or CUDA depending on host).

## Evaluation

Published validation metrics (bundled `metrics.json`):

| Metric | Value |
|--------|-------|
| Accuracy | ~92.9% |
| Weighted F1 | ~92.8% |

**Baselines** (same split seed; see `baseline.py` / `baseline_metrics.json`):

- Majority-class on the validation split.
- Bag-of-words + logistic regression (scikit-learn, CPU).

LoRA is expected to beat majority; on this tiny set BoW can be competitive—compare deltas in `baseline_metrics.json`.

## Limitations and bias

- **Tiny dataset:** Metrics are optimistic and unstable; do not treat as SOTA or production-ready.
- **Domain bias:** Examples resemble product reviews; other domains will degrade.
- **Label balance / wording:** Model may latch onto polarity keywords rather than full sentence meaning.
- **English-only** uncased tokenization.

## Ethical considerations

This is an educational demo. Do not use alone for hiring, credit, content moderation, or clinical decisions. Misclassification of sentiment can amplify bias if deployed without monitoring.

## How to use

```bash
python inference.py --text "Amazing product, highly recommend!"
```

Requires downloading the base DistilBERT weights on first run; the LoRA adapter is shipped in-repo (see [ARTIFACTS.md](ARTIFACTS.md)).

## Citation

```
@software{llm_finetune_lora,
  author = {Simanto, Mehedi Ashraf},
  title  = {llm-finetune-lora: DistilBERT sentiment with LoRA},
  year   = {2026},
  url    = {https://github.com/simanto4321/llm-finetune-lora}
}
```
