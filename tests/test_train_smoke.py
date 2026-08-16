"""Train smoke: max_steps=1 when heavy deps exist; otherwise skip clearly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

torch = pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("transformers", reason="transformers not installed")
pytest.importorskip("peft", reason="peft not installed")
pytest.importorskip("datasets", reason="datasets not installed")


@pytest.mark.heavy
def test_train_smoke_max_steps_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One optimizer step on the real tiny dataset; write metrics under tmp."""
    import train

    out_dir = tmp_path / "output"
    metrics_path = tmp_path / "metrics.json"
    manifest_path = tmp_path / "run_manifest.json"

    monkeypatch.setattr(train, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(train, "METRICS_PATH", metrics_path)
    monkeypatch.setattr(train, "MANIFEST_PATH", manifest_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--fast",
            "--data",
            str(ROOT / "data" / "sentiment.jsonl"),
            "--output-dir",
            str(out_dir),
        ],
    )
    train.main()

    assert metrics_path.is_file()
    assert manifest_path.is_file()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "reproducibility" in metrics
    assert metrics["reproducibility"]["dataset"]["num_examples"] == 66
    assert (out_dir / "lora_adapter").is_dir()
