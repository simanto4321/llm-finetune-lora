"""
Run inference with the fine-tuned LoRA sentiment classifier.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
ADAPTER_DIR = ROOT / "output" / "lora_adapter"
BASE_MODEL = "distilbert-base-uncased"
LABEL_MAP = {0: "negative", 1: "positive"}


def load_model(adapter_dir: Path):
    """Load base model and merge LoRA adapter."""
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter not found at {adapter_dir}. Run train.py first."
        )

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return tokenizer, model


def predict(text: str, tokenizer, model) -> tuple[str, float]:
    """Return predicted label and confidence."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=96,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())
        confidence = float(probs[pred_id].item())

    return LABEL_MAP[pred_id], confidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentiment inference with LoRA model")
    parser.add_argument(
        "--text",
        type=str,
        default="This product exceeded my expectations, fantastic quality!",
        help="Text to classify",
    )
    args = parser.parse_args()

    try:
        tokenizer, model = load_model(ADAPTER_DIR)
        label, confidence = predict(args.text, tokenizer, model)
        print(f"Text: {args.text}")
        print(f"Prediction: {label} (confidence: {confidence:.2%})")
    except Exception as exc:
        print(f"Inference failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
