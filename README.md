# LLM Fine-Tuning with LoRA

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![CI](https://github.com/simanto4321/llm-finetune-lora/actions/workflows/ci.yml/badge.svg)

Fine-tune **DistilBERT** for binary sentiment classification using **Hugging Face Transformers** and **PEFT (LoRA)**.

Uses a local JSONL dataset. Runs on CPU or GPU. Includes a scikit-learn bag-of-words baseline, reproducibility manifests, and a real model card.

Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · Model card: [MODEL_CARD.md](MODEL_CARD.md) · Artifacts: [ARTIFACTS.md](ARTIFACTS.md)

## Stack

| Component | Role |
|-----------|------|
| `transformers` | DistilBERT sequence classification |
| `peft` | LoRA adapters (~1% trainable parameters) |
| `datasets` | Train/validation pipeline |
| `scikit-learn` | Stratified split, metrics, BoW baseline |

## Project structure

```
llm-finetune-lora/
├── train.py                 # LoRA fine-tuning (--fast / --max_steps for smoke)
├── inference.py             # Classify text with bundled adapter
├── baseline.py              # Majority + BoW logistic vs LoRA metrics
├── repro.py                 # Dataset hash, lib versions, run_manifest helpers
├── data/sentiment.jsonl     # 66 labeled examples
├── metrics.json             # Validation metrics + reproducibility block
├── run_manifest.json        # Full run metadata (written by train.py)
├── baseline_metrics.json    # Baseline comparison (written by baseline.py)
├── MODEL_CARD.md            # Model card (intended use, limits, eval)
├── ARTIFACTS.md             # Adapter policy, checksums, retrain steps
├── tests/                   # Pure-Python + skippable heavy smokes
├── output/lora_adapter/     # Bundled trained adapter (see ARTIFACTS.md)
├── requirements.txt
├── run_train.bat
└── run_inference.bat
```

## Install

```bash
git clone https://github.com/simanto4321/llm-finetune-lora.git
cd llm-finetune-lora
pip install -r requirements.txt
pip install pytest   # for tests
```

## Train

```bash
python train.py
```

CPU / CI smoke (one step):

```bash
python train.py --fast
```

Windows:

```bat
run_train.bat
```

Training writes `output/lora_adapter/`, `metrics.json`, and `run_manifest.json`.

## Baseline

Compare LoRA (from `metrics.json`) to majority-class and bag-of-words logistic regression on the same split seed:

```bash
python baseline.py
python baseline.py --fast
```

## Inference

```bash
python inference.py --text "Amazing product, highly recommend!"
```

Windows:

```bat
run_inference.bat --text "Amazing product, highly recommend!"
```

The repository includes a trained LoRA adapter in `output/lora_adapter/` for immediate inference.

## Dataset

`data/sentiment.jsonl` — 66 labeled review examples.

```json
{"text": "your review here", "label": 1}
```

(`0` = negative, `1` = positive)

## Results

| Metric | Value |
|--------|-------|
| Validation accuracy | 92.9% |
| Validation F1 | 92.8% |
| Trainable params (LoRA) | ~1.09% of model |
| Full metrics | `metrics.json` |
| Model card | [MODEL_CARD.md](MODEL_CARD.md) |
| Artifacts / checksums | [ARTIFACTS.md](ARTIFACTS.md) |

## Tests

```bash
# Always-on (no torch required beyond sklearn for baseline import):
pytest tests/test_data_and_manifest.py -v

# Heavy (needs requirements.txt):
pytest tests/test_inference_smoke.py tests/test_train_smoke.py -v -m heavy
```

## License

MIT — see [LICENSE](LICENSE).
