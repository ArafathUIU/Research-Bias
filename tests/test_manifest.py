"""
Tests for experiment manifest generation (Phase 4).

Covers:
- Required fields presence
- Deterministic experiment_id
- SHA-256 of manifest content
- Hardware/software collection (graceful fallback)
- Manifest save/load roundtrip
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from fairlens.manifest import (
    make_manifest,
    ExperimentManifest,
    collect_hardware,
    collect_software,
)


# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------

REQUIRED_MANIFEST_FIELDS = {
    "experiment_id",
    "experiment_track",
    "framework_version",
    "model_name",
    "model_id",
    "primary_temperature",
    "num_samples",
    "bootstrap_method",
    "n_bootstrap",
    "corpus_pair_count",
    "hardware",
    "software",
    "created_at",
}


class TestManifestRequiredFields:

    def test_all_required_fields_present(self):
        manifest = make_manifest(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            model_id="phi3_mini",
        )
        d = manifest.to_dict()
        missing = REQUIRED_MANIFEST_FIELDS - set(d.keys())
        assert not missing, f"Missing required manifest fields: {missing}"

    def test_temperature_is_07(self):
        manifest = make_manifest(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            model_id="phi3_mini",
            primary_temperature=0.7,
        )
        assert manifest.primary_temperature == 0.7

    def test_bootstrap_method_recorded(self):
        manifest = make_manifest(
            model_name="test",
            model_id="test",
            bootstrap_method="independent",
        )
        assert manifest.bootstrap_method == "independent"

    def test_experiment_track_recorded(self):
        manifest = make_manifest(
            model_name="test",
            model_id="test",
            experiment_track="legacy_reproduction",
        )
        assert manifest.experiment_track == "legacy_reproduction"


# ---------------------------------------------------------------------------
# Experiment ID determinism
# ---------------------------------------------------------------------------

class TestExperimentID:

    def test_deterministic_id(self):
        """Same parameters always produce same experiment_id."""
        m1 = make_manifest(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            model_id="phi3_mini",
            primary_temperature=0.7,
            num_samples=20,
            bootstrap_method="independent",
            experiment_track="fairlens_v1",
        )
        m2 = make_manifest(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            model_id="phi3_mini",
            primary_temperature=0.7,
            num_samples=20,
            bootstrap_method="independent",
            experiment_track="fairlens_v1",
        )
        assert m1.experiment_id == m2.experiment_id

    def test_id_changes_with_model(self):
        m1 = make_manifest(model_name="m1", model_id="phi3_mini")
        m2 = make_manifest(model_name="m2", model_id="qwen25_7b")
        assert m1.experiment_id != m2.experiment_id

    def test_id_contains_model_id(self):
        m = make_manifest(model_name="test", model_id="phi3_mini")
        assert "phi3_mini" in m.experiment_id

    def test_id_contains_temperature(self):
        m = make_manifest(model_name="test", model_id="phi3_mini", primary_temperature=0.7)
        # T=0.7 → "T07" in the id
        assert "T07" in m.experiment_id

    def test_id_contains_bootstrap_method_abbrev(self):
        m = make_manifest(model_name="test", model_id="x", bootstrap_method="independent")
        assert "ind" in m.experiment_id


# ---------------------------------------------------------------------------
# Manifest SHA-256
# ---------------------------------------------------------------------------

class TestManifestSHA256:

    def test_sha256_is_64_hex(self):
        m = make_manifest(model_name="test", model_id="test")
        sha = m.sha256
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)

    def test_sha256_changes_with_content(self):
        m1 = make_manifest(model_name="test", model_id="phi3_mini")
        m2 = make_manifest(model_name="test", model_id="qwen25_7b")
        assert m1.sha256 != m2.sha256

    def test_sha256_deterministic(self):
        hw = {"cpu": "test", "ram_gb": 8, "cuda_available": False,
              "cuda_device_count": 0, "cuda_devices": []}
        sw = {"python": "3.11", "torch": "2.9.0"}
        m1 = make_manifest(model_name="t", model_id="t", hardware=hw, software=sw)
        m2 = make_manifest(model_name="t", model_id="t", hardware=hw, software=sw)
        assert m1.sha256 == m2.sha256


# ---------------------------------------------------------------------------
# Hardware / software collection
# ---------------------------------------------------------------------------

class TestHardwareCollection:

    def test_collect_hardware_has_required_keys(self):
        hw = collect_hardware()
        required = {"os", "cuda_available", "cuda_device_count", "cuda_devices"}
        missing = required - set(hw.keys())
        assert not missing, f"Hardware info missing keys: {missing}"

    def test_cuda_available_is_bool(self):
        hw = collect_hardware()
        assert isinstance(hw["cuda_available"], bool)

    def test_collect_software_has_python(self):
        sw = collect_software()
        assert "python" in sw


# ---------------------------------------------------------------------------
# Save / load roundtrip
# ---------------------------------------------------------------------------

class TestManifestSave:

    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = make_manifest(model_name="test", model_id="phi3_mini")
            path = m.save(tmp)
            assert Path(path).exists()

    def test_saved_file_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = make_manifest(model_name="test", model_id="phi3_mini")
            path = m.save(tmp)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            assert "experiment_id" in data

    def test_roundtrip_preserves_temperature(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = make_manifest(
                model_name="test",
                model_id="phi3_mini",
                primary_temperature=0.7,
            )
            path = m.save(tmp)
            with open(path) as f:
                data = json.load(f)
            assert data["primary_temperature"] == 0.7

    def test_five_canonical_models_produce_five_manifests(self):
        """Each model in the canonical suite must produce a distinct manifest."""
        from fairlens.config import CANONICAL_MODELS, MODEL_SHORT_IDS
        with tempfile.TemporaryDirectory() as tmp:
            ids = []
            for model_name in CANONICAL_MODELS:
                model_id = MODEL_SHORT_IDS[model_name]
                m = make_manifest(model_name=model_name, model_id=model_id)
                ids.append(m.experiment_id)
                m.save(tmp)
            assert len(ids) == len(set(ids)), "Duplicate experiment IDs among canonical models"
