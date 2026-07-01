# LLM Fine-Tuning with LoRA

Fine-tune **DistilBERT** for binary sentiment classification using **Hugging Face Transformers** and **PEFT (LoRA)**.

Uses a local JSONL dataset. Trains on **CPU or GPU**; small data size keeps runtime reasonable on a laptop.

## Stack

- `transformers` — DistilBERT sequence classification
- `peft` — LoRA adapters (train ~1% of parameters)
- `datasets` — train/val pipeline
- `scikit-learn` — stratified split & metrics

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

Or on Windows:

```bat
run_train.bat
```

## Inference

```bash
python inference.py --text "Amazing product, highly recommend!"
```

Or on Windows:

```bat
run_inference.bat --text "Amazing product, highly recommend!"
```

## Outputs

- `output/lora_adapter/` — saved LoRA weights + tokenizer
- `metrics.json` — validation accuracy, F1, training config

## Dataset

`data/sentiment.jsonl` — 64 labeled examples (positive/negative reviews).

Add lines:

```json
{"text": "your review here", "label": 1}
```

(`0` = negative, `1` = positive)

## Results

After training (~1 min on CPU with 64 samples):

- Validation **accuracy: 92.9%**
- Validation **F1: 92.8%**
- Trainable params: **~1.09%** of model (LoRA)
- Full metrics in `metrics.json`

## GPU

For faster training, run `train.py` on a machine with CUDA or Google Colab GPU runtime.
