"""
FAIRLens Experiment Manifest
=============================

Every experiment run must generate a manifest JSON that uniquely identifies
the exact experimental conditions. The manifest is the authoritative record
linking result files to the model, generation parameters, embedding model,
bootstrap configuration, corpus version, hardware, and software used.

Without a manifest, results from different configurations cannot be
reliably compared or traced back to their provenance.

Manifest Schema
---------------
See make_manifest() for the complete field structure (Part 31 schema).

Usage
-----
    manifest = make_manifest(
        model_id="phi3_mini",
        model_name="microsoft/Phi-3-mini-4k-instruct",
        config=FAIRLensConfig(),
        corpus_manifest_sha256="abc123...",
        hardware=collect_hardware(),
        software=collect_software(),
    )
    manifest.save("results/manifests/")
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any


# ---------------------------------------------------------------------------
# Hardware / software collection
# ---------------------------------------------------------------------------

def collect_hardware() -> dict:
    """Collect machine hardware info (GPU-safe: graceful fallback if no GPU)."""
    info: dict = {
        "os": platform.system(),
        "os_version": platform.version(),
        "cpu": platform.processor(),
        "python_executable": sys.executable,
    }

    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        info["cpu_cores_physical"] = psutil.cpu_count(logical=False)
        info["cpu_cores_logical"] = psutil.cpu_count(logical=True)
    except ImportError:
        info["ram_gb"] = None
        info["cpu_cores_physical"] = None
        info["cpu_cores_logical"] = None

    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_devices"] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["cuda_devices"].append({
                    "index": i,
                    "name": props.name,
                    "vram_gb": round(props.total_memory / 1e9, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                })
        else:
            info["cuda_device_count"] = 0
            info["cuda_devices"] = []
    except ImportError:
        info["cuda_available"] = False
        info["cuda_device_count"] = 0
        info["cuda_devices"] = []

    return info


def collect_software() -> dict:
    """Collect key package versions."""
    versions: dict = {
        "python": sys.version,
    }

    pkg_names = [
        "torch", "transformers", "accelerate", "bitsandbytes",
        "sentence_transformers", "datasets", "peft",
        "scipy", "numpy", "matplotlib",
    ]
    for pkg in pkg_names:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not_installed"

    return versions


# ---------------------------------------------------------------------------
# Manifest dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentManifest:
    """
    Immutable record of every parameter that affects experimental results.

    This is the Part 31 schema, extended with tracking information.
    """

    # Identity
    experiment_id: str = ""
    experiment_track: str = "fairlens_v1"
    framework_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Model
    model_name: str = ""
    model_id: str = ""           # short identifier (e.g., "phi3_mini")
    model_revision: Optional[str] = None
    precision: str = "float16"
    quantization: str = "nf4_4bit"

    # Generation
    primary_temperature: float = 0.7
    top_p: float = 0.95
    num_samples: int = 20
    max_new_tokens: int = 128
    do_sample: bool = True
    generation_seed: Optional[int] = None

    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize_embeddings: bool = True

    # Bootstrap
    bootstrap_method: str = "independent"
    n_bootstrap: int = 1000
    confidence_level: float = 0.95
    bootstrap_seed: int = 42

    # Corpus
    corpus_pair_count: int = 0
    corpus_manifest_sha256: Optional[str] = None
    corpus_track: str = ""
    selected_pair_ids: Optional[List[str]] = None

    # Hardware
    hardware: Dict[str, Any] = field(default_factory=dict)

    # Software
    software: Dict[str, Any] = field(default_factory=dict)

    # Result paths
    output_dir: str = ""
    checkpoint_db: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, output_dir: str) -> str:
        """
        Atomically write manifest to output_dir/manifests/{experiment_id}.json.
        Returns the path of the saved manifest.
        """
        path = Path(output_dir) / "manifests" / f"{self.experiment_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        tmp.replace(path)
        return str(path)

    @property
    def sha256(self) -> str:
        """SHA-256 of the canonical manifest content (for reproducibility cross-checks)."""
        content = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def make_manifest(
    model_name: str,
    model_id: str,
    experiment_track: str = "fairlens_v1",
    framework_version: str = "1.0.0",
    primary_temperature: float = 0.7,
    top_p: float = 0.95,
    num_samples: int = 20,
    max_new_tokens: int = 128,
    generation_seed: Optional[int] = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    bootstrap_method: str = "independent",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 42,
    model_revision: Optional[str] = None,
    precision: str = "float16",
    quantization: str = "nf4_4bit",
    corpus_pair_count: int = 0,
    corpus_manifest_sha256: Optional[str] = None,
    corpus_track: str = "",
    selected_pair_ids: Optional[List[str]] = None,
    output_dir: str = "results",
    hardware: Optional[dict] = None,
    software: Optional[dict] = None,
) -> ExperimentManifest:
    """
    Build an ExperimentManifest with a deterministic experiment_id.

    The experiment_id encodes key parameters so result filenames are
    self-describing:
      {track}_{model_id}_T{temp}_K{k}_B{bootstrap_method[:3]}_001
    """
    temp_str = str(primary_temperature).replace(".", "")
    bootstrap_abbr = bootstrap_method[:3]  # "ind" or "pai"
    experiment_id = (
        f"{experiment_track}_{model_id}"
        f"_T{temp_str}_K{num_samples}_{bootstrap_abbr}"
    )

    hw = hardware if hardware is not None else collect_hardware()
    sw = software if software is not None else collect_software()

    return ExperimentManifest(
        experiment_id=experiment_id,
        experiment_track=experiment_track,
        framework_version=framework_version,
        model_name=model_name,
        model_id=model_id,
        model_revision=model_revision,
        precision=precision,
        quantization=quantization,
        primary_temperature=primary_temperature,
        top_p=top_p,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        generation_seed=generation_seed,
        embedding_model=embedding_model,
        bootstrap_method=bootstrap_method,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        bootstrap_seed=bootstrap_seed,
        corpus_pair_count=corpus_pair_count,
        corpus_manifest_sha256=corpus_manifest_sha256,
        corpus_track=corpus_track,
        selected_pair_ids=selected_pair_ids,
        output_dir=output_dir,
        hardware=hw,
        software=sw,
    )
