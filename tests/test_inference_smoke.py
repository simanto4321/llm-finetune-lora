"""Smoke inference: load bundled LoRA adapter when heavy deps are present."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_DIR = ROOT / "output" / "lora_adapter"

torch = pytest.importorskip("torch", reason="torch not installed")
transformers = pytest.importorskip("transformers", reason="transformers not installed")
peft = pytest.importorskip("peft", reason="peft not installed")


@pytest.mark.heavy
def test_inference_smoke_loads_adapter() -> None:
    assert ADAPTER_DIR.is_dir(), "Bundled adapter missing; see ARTIFACTS.md"

    from inference import load_model, predict

    tokenizer, model = load_model(ADAPTER_DIR)
    label, confidence = predict("Great product, works perfectly!", tokenizer, model)
    assert label in {"positive", "negative"}
    assert 0.0 <= confidence <= 1.0
