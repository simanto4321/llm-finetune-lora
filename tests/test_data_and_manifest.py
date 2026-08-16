"""Pure-Python tests: data loading and run manifest writing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repro import (
    build_run_manifest,
    dataset_fingerprint,
    load_jsonl,
    merge_metrics_with_manifest,
    sha256_file,
    write_run_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sentiment.jsonl"


def test_dataset_exists_and_has_66_examples() -> None:
    assert DATA_PATH.is_file()
    records = load_jsonl(DATA_PATH)
    assert len(records) == 66
    assert all("text" in r and "label" in r for r in records)
    assert {int(r["label"]) for r in records} <= {0, 1}


def test_dataset_fingerprint_stable() -> None:
    fp = dataset_fingerprint(DATA_PATH)
    assert fp["num_examples"] == 66
    assert fp["size_bytes"] == DATA_PATH.stat().st_size
    assert fp["sha256"] == sha256_file(DATA_PATH)
    assert len(fp["sha256"]) == 64


def test_write_run_manifest(tmp_path: Path) -> None:
    manifest = build_run_manifest(
        seed=42,
        hyperparams={"learning_rate": 2e-4, "lora_r": 8},
        dataset_path=DATA_PATH,
        extra={"pipeline": "unit_test"},
    )
    out = tmp_path / "run_manifest.json"
    write_run_manifest(out, manifest)

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["seed"] == 42
    assert loaded["dataset"]["num_examples"] == 66
    assert "library_versions" in loaded
    assert "device" in loaded
    assert loaded["extra"]["pipeline"] == "unit_test"


def test_merge_metrics_with_manifest() -> None:
    manifest = build_run_manifest(
        seed=7,
        hyperparams={"epochs": 1},
        dataset_path=DATA_PATH,
    )
    metrics = {"validation": {"eval_accuracy": 0.9}}
    enriched = merge_metrics_with_manifest(metrics, manifest)
    assert enriched["validation"]["eval_accuracy"] == 0.9
    assert enriched["reproducibility"]["seed"] == 7
    assert enriched["reproducibility"]["dataset"]["sha256"]


def test_baseline_fast_importable() -> None:
    """baseline.py should import without torch/transformers when sklearn is present."""
    pytest.importorskip("sklearn", reason="scikit-learn not installed")
    import baseline  # noqa: F401

    assert callable(baseline.train_bow_baseline)
