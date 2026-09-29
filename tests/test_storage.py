"""
Tests for storage and checkpoint system (Phase 4).

Covers:
- GenerationStore: table creation, insert, resume (skip completed)
- Atomic JSONL append
- Progress summary
- JSONL export
- Result directory structure creation
- Sample record validation
"""

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from fairlens.storage import (
    GenerationStore,
    atomic_write_json,
    atomic_append_jsonl,
    create_result_structure,
    generation_db_path,
    generation_jsonl_path,
    validate_sample_record,
    REQUIRED_SAMPLE_FIELDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sample_record(**overrides) -> dict:
    """Return a valid sample record dict."""
    record = {
        "experiment_id": "fairlens_v1_phi3_mini_T07_K20_ind",
        "model_id": "phi3_mini",
        "model_revision": "main",
        "pair_id": "P1A2B3C4D5E",
        "group": "A",
        "sample_idx": 0,
        "domain": "hiring",
        "group_a": "Black_Female",
        "group_b": "White_Male",
        "template_idx": 0,
        "prompt": "Write a hiring recommendation for a Black female applicant.",
        "response": "I would highly recommend this candidate.",
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 128,
        "precision": "float16",
        "quantization": "nf4_4bit",
        "device": "cuda:0",
        "seed": 42,
        "runtime_ms": 123.4,
        "timestamp": "2026-09-29T16:00:00Z",
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# Sample record validation
# ---------------------------------------------------------------------------

class TestSampleRecordValidation:

    def test_valid_record_has_no_missing(self):
        record = make_sample_record()
        missing = validate_sample_record(record)
        assert missing == [], f"Expected no missing fields, got: {missing}"

    def test_missing_response(self):
        record = make_sample_record()
        del record["response"]
        missing = validate_sample_record(record)
        assert "response" in missing

    def test_missing_multiple_fields(self):
        record = make_sample_record()
        del record["temperature"]
        del record["device"]
        missing = validate_sample_record(record)
        assert "temperature" in missing
        assert "device" in missing

    def test_extra_fields_allowed(self):
        record = make_sample_record(extra_field="extra_value")
        missing = validate_sample_record(record)
        assert missing == []


# ---------------------------------------------------------------------------
# GenerationStore: create, insert, retrieve
# ---------------------------------------------------------------------------

class TestGenerationStore:

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test.db")
        self.store = GenerationStore(self.db_path)
        self.store.create_tables()
        self.exp_id = "fairlens_v1_phi3_mini_T07_K20_ind"

    def test_db_file_created(self):
        assert Path(self.db_path).exists()

    def test_empty_store_has_no_completed_keys(self):
        completed = self.store.get_completed_keys(self.exp_id)
        assert completed == set()

    def test_insert_sample_and_retrieve(self):
        record = make_sample_record()
        self.store.insert_sample(record)
        completed = self.store.get_completed_keys(self.exp_id)
        expected_key = (self.exp_id, "P1A2B3C4D5E", "A", 0)
        assert expected_key in completed

    def test_duplicate_insert_is_idempotent(self):
        """INSERT OR IGNORE: inserting same key twice must not raise or duplicate."""
        record = make_sample_record()
        self.store.insert_sample(record)
        self.store.insert_sample(record)  # second insert — must not raise
        completed = self.store.get_completed_keys(self.exp_id)
        assert len(completed) == 1

    def test_missing_required_field_raises(self):
        record = make_sample_record()
        del record["response"]
        with pytest.raises(ValueError, match="missing fields"):
            self.store.insert_sample(record)

    def test_insert_multiple_samples_and_count(self):
        for idx in range(5):
            record = make_sample_record(sample_idx=idx)
            self.store.insert_sample(record)
        completed = self.store.get_completed_keys(self.exp_id)
        assert len(completed) == 5

    def test_different_pairs_tracked_separately(self):
        r1 = make_sample_record(pair_id="PAAAAAAAAAAA", sample_idx=0)
        r2 = make_sample_record(pair_id="PBBBBBBBBBBB", sample_idx=0)
        self.store.insert_sample(r1)
        self.store.insert_sample(r2)
        completed = self.store.get_completed_keys(self.exp_id)
        assert len(completed) == 2

    def test_different_groups_tracked_separately(self):
        r_a = make_sample_record(group="A", sample_idx=0)
        r_b = make_sample_record(group="B", sample_idx=0)
        self.store.insert_sample(r_a)
        self.store.insert_sample(r_b)
        completed = self.store.get_completed_keys(self.exp_id)
        # Both (exp, pair, A, 0) and (exp, pair, B, 0) should be in completed
        assert len(completed) == 2


# ---------------------------------------------------------------------------
# Checkpoint resume test
# ---------------------------------------------------------------------------

class TestCheckpointResume:
    """
    Simulates a crashed run and tests that completed samples are skipped
    on resume. This is the core crash-safety guarantee.
    """

    def test_resume_skips_completed_samples(self):
        """
        Simulate: 20 samples × 2 groups = 40 total needed.
        After a crash with 15 done, resume should skip those 15.
        """
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            db = os.path.join(tmp, "run.db")
            store = GenerationStore(db)
            store.create_tables()
            exp_id = "test_exp"
            pair_id = "PAAAAAAAAAA1"

            # Simulate 15 completed samples for group A
            for idx in range(15):
                r = make_sample_record(
                    experiment_id=exp_id,
                    pair_id=pair_id,
                    group="A",
                    sample_idx=idx,
                )
                store.insert_sample(r)

            # "Resume": load completed keys and count what would be skipped
            completed = store.get_completed_keys(exp_id)
            skipped = 0
            for idx in range(20):
                key = (exp_id, pair_id, "A", idx)
                if key in completed:
                    skipped += 1

            assert skipped == 15, (
                f"Expected 15 completed samples to be skipped, got {skipped}"
            )
            # Only 5 remaining (idx 15..19) would need to be generated
            remaining = 20 - skipped
            assert remaining == 5


# ---------------------------------------------------------------------------
# Progress summary tests
# ---------------------------------------------------------------------------

class TestProgressSummary:

    def test_empty_store_progress(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            store = GenerationStore(os.path.join(tmp, "p.db"))
            store.create_tables()
            summary = store.get_progress_summary("exp1", total_pairs=200, k=20)
            assert summary["completed_samples"] == 0
            assert summary["fully_completed_pairs"] == 0
            assert summary["completion_pct"] == 0.0

    def test_progress_after_inserts(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            store = GenerationStore(os.path.join(tmp, "p.db"))
            store.create_tables()
            exp_id = "exp1"
            # Insert all K=20 samples for both A and B for one pair
            for g in ["A", "B"]:
                for idx in range(20):
                    r = make_sample_record(
                        experiment_id=exp_id,
                        pair_id="PAAAAAAAAAA1",
                        group=g,
                        sample_idx=idx,
                    )
                    store.insert_sample(r)
            summary = store.get_progress_summary(exp_id, total_pairs=200, k=20)
            assert summary["completed_samples"] == 40
            assert summary["fully_completed_pairs"] == 1


# ---------------------------------------------------------------------------
# JSONL export test
# ---------------------------------------------------------------------------

class TestJSONLExport:

    def test_export_produces_correct_lines(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            db = os.path.join(tmp, "export.db")
            store = GenerationStore(db)
            store.create_tables()
            exp_id = "exp_export"

            for idx in range(3):
                r = make_sample_record(
                    experiment_id=exp_id,
                    sample_idx=idx,
                )
                store.insert_sample(r)

            out = os.path.join(tmp, "out.jsonl")
            n = store.export_jsonl(out, exp_id)
            assert n == 3

            with open(out, encoding="utf-8") as f:
                lines = f.readlines()
            assert len(lines) == 3

            for line in lines:
                record = json.loads(line)
                assert "pair_id" in record
                assert "response" in record

    def test_export_atomic_replaces_existing(self):
        """A second export overwrites the first atomically."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            db = os.path.join(tmp, "exp.db")
            store = GenerationStore(db)
            store.create_tables()
            exp_id = "exp2"

            r = make_sample_record(experiment_id=exp_id, sample_idx=0)
            store.insert_sample(r)
            out = os.path.join(tmp, "out.jsonl")
            store.export_jsonl(out, exp_id)

            # Now add more and re-export
            for idx in range(1, 5):
                store.insert_sample(make_sample_record(experiment_id=exp_id, sample_idx=idx))
            n = store.export_jsonl(out, exp_id)
            assert n == 5
            with open(out) as f:
                assert len(f.readlines()) == 5


# ---------------------------------------------------------------------------
# Atomic write utilities
# ---------------------------------------------------------------------------

class TestAtomicWriteJson:

    def test_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.json")
            atomic_write_json(path, {"key": "value", "n": 42})
            with open(path) as f:
                d = json.load(f)
            assert d["key"] == "value"
            assert d["n"] == 42

    def test_overwrites_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.json")
            atomic_write_json(path, {"v": 1})
            atomic_write_json(path, {"v": 2})
            with open(path) as f:
                d = json.load(f)
            assert d["v"] == 2

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "data.json")
            atomic_write_json(path, {})
            assert Path(path).exists()


# ---------------------------------------------------------------------------
# Result directory structure test
# ---------------------------------------------------------------------------

class TestCreateResultStructure:

    def test_all_directories_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = create_result_structure(base=tmp)
            expected_keys = [
                "fairlens_v1/manifests",
                "fairlens_v1/generations",
                "fairlens_v1/corpus",
                "legacy_reproduction/manifests",
                "legacy_reproduction/generations",
            ]
            for key in expected_keys:
                assert key in paths, f"Missing path key: {key}"
                assert Path(paths[key]).is_dir(), f"Not a directory: {paths[key]}"

    def test_both_tracks_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = create_result_structure(base=tmp)
            fv1_gen = Path(paths["fairlens_v1/generations"])
            leg_gen = Path(paths["legacy_reproduction/generations"])
            assert fv1_gen != leg_gen
            assert fv1_gen.exists()
            assert leg_gen.exists()
