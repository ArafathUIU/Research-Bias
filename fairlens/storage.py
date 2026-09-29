"""
FAIRLens Storage and Checkpoint System
=======================================

Provides crash-safe, resumable storage for LLM generation runs.

Architecture
------------
GenerationStore — SQLite-backed checkpoint store for per-sample records.
    Uses WAL journal mode for crash safety. On restart, loads completed
    record keys and skips them. A crash at any point preserves all
    previously committed samples.

ResultStore — manages the structured result directory hierarchy.
    Writes are always atomic (temp file → os.replace).

Schema
------
Per-sample (checkpoint) record (Part 29 schema):
    experiment_id, model_id, model_revision, pair_id, group (A|B),
    sample_idx, prompt, response, seed, temperature, top_p,
    max_new_tokens, precision, quantization, device, runtime_ms, timestamp

The checkpoint key is (experiment_id, pair_id, group, sample_idx).
This is the atomic unit: if it is in the database, it is complete.

Usage
-----
    store = GenerationStore("results/generations/phi3_mini_t07.db")
    store.create_tables()

    # Check what's done
    completed = store.get_completed_keys(experiment_id)

    # Before generating, check:
    key = (experiment_id, pair_id, "A", sample_idx)
    if key in completed:
        continue  # skip this sample

    # After generating:
    store.insert_sample(record_dict)

    # On completion, export to JSONL:
    store.export_jsonl("results/generations/phi3_mini_t07.jsonl")
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Per-sample record schema
# ---------------------------------------------------------------------------

SAMPLE_SCHEMA_VERSION = "1.0"

REQUIRED_SAMPLE_FIELDS = frozenset([
    "experiment_id", "model_id", "model_revision",
    "pair_id", "group", "sample_idx",
    "prompt", "response",
    "temperature", "top_p", "max_new_tokens",
    "precision", "quantization", "device",
    "seed", "runtime_ms", "timestamp",
])

OPTIONAL_SAMPLE_FIELDS = frozenset([
    "domain", "group_a", "group_b", "template_idx",
])


def validate_sample_record(record: dict) -> List[str]:
    """
    Check that a sample record contains all required fields.
    Returns a list of missing field names (empty = valid).
    """
    return [f for f in REQUIRED_SAMPLE_FIELDS if f not in record]


# ---------------------------------------------------------------------------
# SQLite checkpoint store
# ---------------------------------------------------------------------------

CREATE_SAMPLES_TABLE = """
CREATE TABLE IF NOT EXISTS samples (
    experiment_id TEXT NOT NULL,
    pair_id       TEXT NOT NULL,
    grp           TEXT NOT NULL,  -- "A" or "B"
    sample_idx    INTEGER NOT NULL,
    model_id      TEXT,
    model_revision TEXT,
    domain        TEXT,
    group_a       TEXT,
    group_b       TEXT,
    template_idx  INTEGER,
    prompt        TEXT NOT NULL,
    response      TEXT NOT NULL,
    temperature   REAL,
    top_p         REAL,
    max_new_tokens INTEGER,
    precision     TEXT,
    quantization  TEXT,
    device        TEXT,
    seed          INTEGER,
    runtime_ms    REAL,
    timestamp     TEXT,
    PRIMARY KEY (experiment_id, pair_id, grp, sample_idx)
);
"""

CREATE_META_TABLE = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class GenerationStore:
    """
    SQLite-backed checkpoint store for LLM generation samples.

    Thread safety: assumes single-process access (one Kaggle/Colab kernel).
    WAL mode ensures crash safety for concurrent reads from analysis scripts.
    """

    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def create_tables(self) -> None:
        """Create tables if they don't exist. Safe to call on every run."""
        with self._connect() as conn:
            conn.execute(CREATE_SAMPLES_TABLE)
            conn.execute(CREATE_META_TABLE)
            conn.execute(
                "INSERT OR IGNORE INTO meta VALUES (?, ?)",
                ("schema_version", SAMPLE_SCHEMA_VERSION),
            )
            conn.commit()

    def get_completed_keys(
        self, experiment_id: str
    ) -> Set[Tuple[str, str, str, int]]:
        """
        Return the set of (experiment_id, pair_id, grp, sample_idx) tuples
        that have already been completed and stored.

        Use this at the start of a generation run to skip completed work.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT experiment_id, pair_id, grp, sample_idx FROM samples "
                "WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchall()
        return set(rows)

    def insert_sample(self, record: dict) -> None:
        """
        Atomically insert a completed sample record.

        Uses INSERT OR IGNORE so duplicate inserts (e.g. from a retry)
        do not raise errors.

        Parameters
        ----------
        record : dict — must include all REQUIRED_SAMPLE_FIELDS
        """
        missing = validate_sample_record(record)
        if missing:
            raise ValueError(f"Sample record missing fields: {missing}")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO samples
                    (experiment_id, pair_id, grp, sample_idx,
                     model_id, model_revision, domain, group_a, group_b, template_idx,
                     prompt, response,
                     temperature, top_p, max_new_tokens,
                     precision, quantization, device, seed,
                     runtime_ms, timestamp)
                VALUES
                    (:experiment_id, :pair_id, :group, :sample_idx,
                     :model_id, :model_revision, :domain, :group_a, :group_b, :template_idx,
                     :prompt, :response,
                     :temperature, :top_p, :max_new_tokens,
                     :precision, :quantization, :device, :seed,
                     :runtime_ms, :timestamp)
                """,
                record,
            )
            conn.commit()

    def count_completed(self, experiment_id: str) -> dict:
        """Return counts of completed samples per pair_id × group."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT pair_id, grp, COUNT(*) FROM samples "
                "WHERE experiment_id = ? GROUP BY pair_id, grp",
                (experiment_id,),
            ).fetchall()
        return {(r[0], r[1]): r[2] for r in rows}

    def export_jsonl(self, out_path: str, experiment_id: str) -> int:
        """
        Export all samples for experiment_id to a JSONL file.
        Returns number of records written.
        Writes atomically (temp → replace).
        """
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path + ".tmp"

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT experiment_id, pair_id, grp, sample_idx,
                       model_id, model_revision, domain, group_a, group_b, template_idx,
                       prompt, response,
                       temperature, top_p, max_new_tokens,
                       precision, quantization, device, seed,
                       runtime_ms, timestamp
                FROM samples
                WHERE experiment_id = ?
                ORDER BY pair_id, grp, sample_idx
                """,
                (experiment_id,),
            ).fetchall()

        columns = [
            "experiment_id", "pair_id", "group", "sample_idx",
            "model_id", "model_revision", "domain", "group_a", "group_b", "template_idx",
            "prompt", "response",
            "temperature", "top_p", "max_new_tokens",
            "precision", "quantization", "device", "seed",
            "runtime_ms", "timestamp",
        ]

        with open(tmp, "w", encoding="utf-8") as f:
            for row in rows:
                record = dict(zip(columns, row))
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        os.replace(tmp, out_path)
        return len(rows)

    def get_progress_summary(self, experiment_id: str, total_pairs: int, k: int) -> dict:
        """
        Return a human-readable progress summary.

        Parameters
        ----------
        total_pairs : expected number of prompt pairs
        k           : expected samples per condition
        """
        completed = self.count_completed(experiment_id)
        total_expected = total_pairs * 2 * k  # 2 conditions (A, B)
        total_done = sum(completed.values())

        pair_ids_done_a = {pid for (pid, g) in completed if g == "A"}
        pair_ids_done_b = {pid for (pid, g) in completed if g == "B"}
        fully_done = pair_ids_done_a & pair_ids_done_b

        return {
            "experiment_id": experiment_id,
            "total_expected_samples": total_expected,
            "completed_samples": total_done,
            "completion_pct": round(100 * total_done / max(total_expected, 1), 1),
            "fully_completed_pairs": len(fully_done),
            "total_pairs": total_pairs,
        }


# ---------------------------------------------------------------------------
# Atomic file write utilities
# ---------------------------------------------------------------------------

def atomic_write_json(path: str, data: dict) -> None:
    """Write data to path atomically (temp file → os.replace)."""
    tmp = path + ".tmp"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def atomic_append_jsonl(path: str, record: dict) -> None:
    """
    Append a single JSON record to a JSONL file.

    This is NOT fully crash-safe for mid-line writes, but is acceptable
    for logging completed results alongside the SQLite checkpoint. The SQLite
    store is the authoritative checkpoint; the JSONL is a human-readable export.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


# ---------------------------------------------------------------------------
# Result directory structure
# ---------------------------------------------------------------------------

RESULT_DIRS = [
    "manifests",
    "corpus",
    "generations",
    "embeddings",
    "prompt_metrics",
    "aggregate_metrics",
    "cross_model",
    "robustness",
    "human_validation",
    "figures",
    "reports",
]


def create_result_structure(base: str = "results") -> Dict[str, str]:
    """
    Create the complete result directory hierarchy and return path dict.

    The two experiment tracks are separated into subdirectories:
      results/fairlens_v1/
      results/legacy_reproduction/
    """
    paths = {}
    for track in ["fairlens_v1", "legacy_reproduction"]:
        for d in RESULT_DIRS:
            p = Path(base) / track / d
            p.mkdir(parents=True, exist_ok=True)
            paths[f"{track}/{d}"] = str(p)
    # Also create top-level manifests for corpus
    top_corpus = Path(base) / "corpus"
    top_corpus.mkdir(parents=True, exist_ok=True)
    paths["corpus"] = str(top_corpus)
    return paths


def generation_db_path(base: str, track: str, model_id: str, temperature: float) -> str:
    """Return the canonical checkpoint database path for a generation run."""
    temp_str = str(temperature).replace(".", "")
    return str(Path(base) / track / "generations" / f"{model_id}_T{temp_str}.db")


def generation_jsonl_path(base: str, track: str, model_id: str, temperature: float) -> str:
    """Return the canonical JSONL export path for a generation run."""
    temp_str = str(temperature).replace(".", "")
    return str(Path(base) / track / "generations" / f"{model_id}_T{temp_str}.jsonl")
