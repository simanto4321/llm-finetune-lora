"""
Compare LoRA metrics against a lightweight bag-of-words logistic baseline.

Default path uses only scikit-learn (CPU-friendly). Optionally compare
against metrics.json from a prior LoRA run. Use --fast for tiny subsets.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from repro import (
    build_run_manifest,
    load_jsonl,
    merge_metrics_with_manifest,
    write_json,
    write_run_manifest,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sentiment.jsonl"
METRICS_PATH = ROOT / "metrics.json"
BASELINE_METRICS_PATH = ROOT / "baseline_metrics.json"
BASELINE_MANIFEST_PATH = ROOT / "baseline_run_manifest.json"
DEFAULT_SEED = 42


@dataclass
class BaselineConfig:
    """Hyperparameters for the BoW logistic baseline."""

    test_size: float = 0.2
    seed: int = DEFAULT_SEED
    max_features: int = 5000
    c: float = 1.0
    max_iter: int = 500
    fast: bool = False
    max_examples: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BoW logistic baseline vs LoRA (from metrics.json)"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Tiny CPU path: 16 examples, fewer features, fewer iterations.",
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--lora-metrics",
        type=Path,
        default=METRICS_PATH,
        help="Existing LoRA metrics.json to compare against.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASELINE_METRICS_PATH,
        help="Where to write baseline comparison JSON.",
    )
    return parser.parse_args()


def majority_baseline(y_train: list[int], y_val: list[int]) -> dict[str, float]:
    """Predict the majority training label on the validation split."""
    majority = max(set(y_train), key=y_train.count)
    preds = [majority] * len(y_val)
    return {
        "accuracy": float(accuracy_score(y_val, preds)),
        "f1": float(f1_score(y_val, preds, average="weighted", zero_division=0)),
        "majority_label": int(majority),
    }


def train_bow_baseline(
    texts: list[str],
    labels: list[int],
    config: BaselineConfig,
) -> tuple[dict[str, float], dict[str, float], int, int]:
    """Fit CountVectorizer + LogisticRegression; return val metrics and sizes."""
    if config.max_examples is not None and len(texts) > config.max_examples:
        texts = texts[: config.max_examples]
        labels = labels[: config.max_examples]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=labels,
    )

    pipe = Pipeline(
        [
            (
                "vec",
                CountVectorizer(
                    max_features=config.max_features,
                    ngram_range=(1, 2),
                    lowercase=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=config.c,
                    max_iter=config.max_iter,
                    random_state=config.seed,
                ),
            ),
        ]
    )
    pipe.fit(train_texts, train_labels)
    preds = pipe.predict(val_texts)
    bow = {
        "accuracy": float(accuracy_score(val_labels, preds)),
        "f1": float(f1_score(val_labels, preds, average="weighted")),
    }
    maj = majority_baseline(train_labels, val_labels)
    return bow, maj, len(train_texts), len(val_texts)


def load_lora_validation(path: Path) -> dict[str, float] | None:
    """Pull validation accuracy/F1 from an existing LoRA metrics file."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    validation = payload.get("validation") or {}
    out: dict[str, float] = {}
    if "eval_accuracy" in validation:
        out["accuracy"] = float(validation["eval_accuracy"])
    elif "accuracy" in validation:
        out["accuracy"] = float(validation["accuracy"])
    if "eval_f1" in validation:
        out["f1"] = float(validation["eval_f1"])
    elif "f1" in validation:
        out["f1"] = float(validation["f1"])
    return out or None


def main() -> None:
    args = parse_args()
    config = BaselineConfig(
        test_size=args.test_size,
        seed=args.seed,
        fast=args.fast,
        max_features=1000 if args.fast else 5000,
        max_iter=100 if args.fast else 500,
        max_examples=16 if args.fast else None,
    )

    if not args.data.exists():
        print(f"Dataset not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    records = load_jsonl(args.data)
    texts = [str(r["text"]) for r in records]
    labels = [int(r["label"]) for r in records]

    print("=" * 60)
    print("Baseline comparison (BoW logistic vs LoRA metrics)")
    print("=" * 60)
    if config.fast:
        print("Fast mode: tiny subset, CPU-only sklearn.")

    bow, maj, n_train, n_val = train_bow_baseline(texts, labels, config)
    lora = load_lora_validation(args.lora_metrics)

    hyperparams = asdict(config)
    manifest = build_run_manifest(
        seed=config.seed,
        hyperparams=hyperparams,
        dataset_path=args.data,
        extra={
            "pipeline": "bow_logistic_baseline",
            "split": {"train": n_train, "val": n_val},
        },
    )
    write_run_manifest(BASELINE_MANIFEST_PATH, manifest)

    comparison: dict = {
        "method": "comparison",
        "seed": config.seed,
        "split": {"train": n_train, "val": n_val, "test_size": config.test_size},
        "baselines": {
            "majority": maj,
            "bow_logistic": bow,
        },
        "lora_from_metrics": lora,
    }
    if lora is not None:
        comparison["delta_vs_bow"] = {
            "accuracy": round(lora["accuracy"] - bow["accuracy"], 4),
            "f1": round(lora["f1"] - bow["f1"], 4),
        }

    enriched = merge_metrics_with_manifest(comparison, manifest)
    write_json(args.output, enriched)

    print(f"  Majority  accuracy={maj['accuracy']:.4f}  f1={maj['f1']:.4f}")
    print(f"  BoW logreg accuracy={bow['accuracy']:.4f}  f1={bow['f1']:.4f}")
    if lora:
        print(
            f"  LoRA (metrics.json) accuracy={lora['accuracy']:.4f}  "
            f"f1={lora['f1']:.4f}"
        )
        if "delta_vs_bow" in comparison:
            d = comparison["delta_vs_bow"]
            print(f"  Delta (LoRA - BoW) acc={d['accuracy']:+.4f}  f1={d['f1']:+.4f}")
    else:
        print(f"  LoRA metrics not found at {args.lora_metrics} (skipped comparison)")
    print(f"Wrote: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
