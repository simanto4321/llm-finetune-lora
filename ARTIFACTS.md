# Artifacts policy

This repository keeps a **small LoRA adapter in-tree** so inference demos and CI smoke checks work without re-training.

## What is shipped

| Path | Purpose | In git? |
|------|---------|---------|
| `output/lora_adapter/adapter_model.safetensors` | LoRA weights | Yes (demo) |
| `output/lora_adapter/adapter_config.json` | PEFT config | Yes |
| `output/lora_adapter/tokenizer*` | Tokenizer files saved with the adapter | Yes |
| `output/checkpoint-*/` | Intermediate Trainer checkpoints | **No** (gitignored) |
| `metrics.json` / `run_manifest.json` | Published metrics + reproducibility | Yes (after train) |
| Full DistilBERT base weights | Loaded from Hugging Face Hub at runtime | **Not** vendored |

Keeping the adapter (~3 MB) in-repo is intentional for portfolio/demo UX. Intermediate checkpoints are excluded because they are large and redundant.

## Checksums (SHA-256)

Verify after clone or before publishing a new adapter:

```bash
# Windows PowerShell
Get-FileHash -Algorithm SHA256 output\lora_adapter\adapter_model.safetensors
Get-FileHash -Algorithm SHA256 output\lora_adapter\adapter_config.json
Get-FileHash -Algorithm SHA256 data\sentiment.jsonl
```

| File | SHA-256 |
|------|---------|
| `output/lora_adapter/adapter_model.safetensors` | `91d5078b53d983439dd0d544d6aa5d6e2ed7d9654edda347138b0f8d8fa205f4` |
| `output/lora_adapter/adapter_config.json` | `b88a8ea6af2ca255a6d242f1b6b5cfbafeef5d7f2aa45623674131bad0d3887f` |
| `data/sentiment.jsonl` | `82b971634b9d9faeb26459b27d50317b6e8053788cff9e0b7890b31a63565ad5` |

If you retrain and commit a new adapter, **update this table** (and preferably regenerate `metrics.json` / `run_manifest.json`).

## How to retrain

```bash
pip install -r requirements.txt
python train.py
# writes: output/lora_adapter/, metrics.json, run_manifest.json
```

CPU smoke (does not replace the published adapter unless you choose to commit it):

```bash
python train.py --fast
```

Baseline comparison (no transformers required beyond what sklearn needs):

```bash
python baseline.py
python baseline.py --fast
```

## Reproducibility

Every full training run should refresh:

1. `run_manifest.json` — seed, hyperparams, dataset SHA-256/size, library versions, device.
2. `metrics.json` — validation metrics plus the same block under `reproducibility`.

Do not hand-edit checksums or metrics without re-running the corresponding script.
