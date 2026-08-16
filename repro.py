"""
Reproducibility helpers: dataset fingerprints, library versions, run manifests.

Core APIs use only the standard library so tests can run without torch/transformers.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load records from a JSONL file."""
    records: list[dict[str, Any]] = []
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


def _repo_relative(path: Path) -> str:
    """Prefer a stable path relative to this package's repo root."""
    root = Path(__file__).resolve().parent
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def dataset_fingerprint(path: Path) -> dict[str, Any]:
    """SHA-256, byte size, and example count for a dataset file."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    records = load_jsonl(path)
    return {
        "path": _repo_relative(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "num_examples": len(records),
        "label_counts": _label_counts(records),
    }


def _label_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        key = str(row.get("label"))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def collect_library_versions() -> dict[str, str | None]:
    """Best-effort package versions; missing packages are None."""
    names = (
        "torch",
        "transformers",
        "peft",
        "datasets",
        "sklearn",
        "numpy",
        "accelerate",
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            mod = __import__(name if name != "sklearn" else "sklearn")
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = None
    return versions


def detect_device() -> dict[str, Any]:
    """CPU/GPU summary without requiring CUDA at import time."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cuda_available": False,
        "cuda_device_name": None,
    }
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["torch_device"] = "cuda"
        else:
            info["torch_device"] = "cpu"
    except ImportError:
        info["torch_device"] = "unavailable"
    return info


def build_run_manifest(
    *,
    seed: int,
    hyperparams: dict[str, Any],
    dataset_path: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble full reproducibility metadata for a training or baseline run."""
    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "hyperparams": hyperparams,
        "dataset": dataset_fingerprint(dataset_path),
        "library_versions": collect_library_versions(),
        "device": detect_device(),
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_run_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Persist a run manifest JSON file."""
    write_json(path, manifest)


def merge_metrics_with_manifest(
    metrics: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Embed reproducibility block into a metrics payload."""
    enriched = dict(metrics)
    enriched["reproducibility"] = {
        "seed": manifest.get("seed"),
        "hyperparams": manifest.get("hyperparams"),
        "dataset": manifest.get("dataset"),
        "library_versions": manifest.get("library_versions"),
        "device": manifest.get("device"),
        "created_at_utc": manifest.get("created_at_utc"),
    }
    return enriched
