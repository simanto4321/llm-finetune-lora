# LLM Fine-Tuning with LoRA

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![CI](https://github.com/simanto4321/llm-finetune-lora/actions/workflows/ci.yml/badge.svg)

Fine-tune **DistilBERT** for binary sentiment classification using **Hugging Face Transformers** and **PEFT (LoRA)**.

Uses a local JSONL dataset. Runs on CPU or GPU.

## Stack

| Component | Role |
|-----------|------|
| `transformers` | DistilBERT sequence classification |
| `peft` | LoRA adapters (~1% trainable parameters) |
| `datasets` | Train/validation pipeline |
| `scikit-learn` | Stratified split and metrics |

## Project structure

```
llm-finetune-lora/
├── train.py
├── inference.py
├── data/sentiment.jsonl
├── metrics.json
├── output/lora_adapter/   # bundled trained adapter
├── requirements.txt
├── run_train.bat
└── run_inference.bat
```

## Install

```bash
git clone https://github.com/simanto4321/llm-finetune-lora.git
cd llm-finetune-lora
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

Windows:

```bat
run_train.bat
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

`data/sentiment.jsonl` — 64 labeled review examples.

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

## License

MIT — see [LICENSE](LICENSE).
