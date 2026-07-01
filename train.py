"""
Fine-tune DistilBERT with LoRA for binary sentiment classification.

Runs on CPU or GPU with a small local JSONL dataset.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sentiment.jsonl"
OUTPUT_DIR = ROOT / "output"
METRICS_PATH = ROOT / "metrics.json"

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 2
RANDOM_SEED = 42
MAX_LENGTH = 96


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    model_name: str = MODEL_NAME
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 2e-4
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    test_size: float = 0.2


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict[str, object]]:
    """Load records from a JSONL file."""
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON at line {line_no}: {exc}") from exc
    return records


def build_datasets(path: Path, test_size: float, seed: int) -> tuple[Dataset, Dataset]:
    """Load data and split into Hugging Face datasets."""
    records = load_jsonl(path)
    texts = [str(r["text"]) for r in records]
    labels = [int(r["label"]) for r in records]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})
    return train_ds, val_ds


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    """Tokenize text field."""

    def _tokenize(batch: dict[str, list]) -> dict[str, list]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    return dataset.map(_tokenize, batched=True)


def create_model(config: TrainConfig) -> torch.nn.Module:
    """Load DistilBERT and wrap with LoRA adapters."""
    base = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=NUM_LABELS,
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_lin", "v_lin"],
        bias="none",
    )
    return get_peft_model(base, lora_config)


def compute_metrics(eval_pred: tuple) -> dict[str, float]:
    """Compute accuracy and F1 for the Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="weighted")),
    }


def save_metrics(history: dict, val_metrics: dict[str, float], config: TrainConfig) -> None:
    """Persist training metrics to JSON."""
    payload = {
        "model_name": config.model_name,
        "lora": {
            "r": config.lora_r,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
        },
        "epochs": config.num_train_epochs,
        "validation": val_metrics,
        "train_loss_per_epoch": history.get("train_loss", []),
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Metrics saved to: {METRICS_PATH}")


def main() -> None:
    """Run LoRA fine-tuning pipeline."""
    config = TrainConfig()
    set_seed(RANDOM_SEED)

    print("=" * 60)
    print("DistilBERT Fine-Tuning with LoRA")
    print("=" * 60)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}", file=sys.stderr)
        sys.exit(1)

    try:
        print("\n[1/5] Loading dataset...")
        train_ds, val_ds = build_datasets(DATA_PATH, config.test_size, RANDOM_SEED)
        print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

        print("\n[2/5] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        print("\n[3/5] Tokenizing...")
        train_ds = tokenize_dataset(train_ds, tokenizer)
        val_ds = tokenize_dataset(val_ds, tokenizer)
        train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        print("\n[4/5] Creating LoRA model...")
        model = create_model(config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

        training_args = TrainingArguments(
            output_dir=str(OUTPUT_DIR),
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            learning_rate=config.learning_rate,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to="none",
            seed=RANDOM_SEED,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
        )

        print("\n[5/5] Training...")
        train_output = trainer.train()
        eval_metrics = trainer.evaluate()

        model.save_pretrained(OUTPUT_DIR / "lora_adapter")
        tokenizer.save_pretrained(OUTPUT_DIR / "lora_adapter")

        train_losses = [
            round(entry["loss"], 4)
            for entry in trainer.state.log_history
            if "loss" in entry and "eval_loss" not in entry
        ]
        save_metrics({"train_loss": train_losses}, eval_metrics, config)

        print("\n--- Final validation metrics ---")
        for key, value in eval_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")

        print(f"\nAdapter saved to: {OUTPUT_DIR / 'lora_adapter'}")
        print(f"Final train loss: {train_output.training_loss:.4f}")
        print("=" * 60)

    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
