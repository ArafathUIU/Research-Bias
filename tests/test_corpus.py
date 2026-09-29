"""
Tests for FAIRLens Corpus Generation and Validation
===================================================

Covers:
1. FAIRLens v1 corpus pair counts:
   - 360 total controlled prompt pairs
   - 200 main race/gender pairs (5 contrasts x 10 domains x 4 templates)
   - 160 2x2 SES factorial pairs (4 contrasts x 10 domains x 4 templates)
   - 36 pairs per domain across all 10 domains
   - 40 pairs per group contrast across all 9 group contrasts
2. Legacy reproduction corpus pair counts:
   - 240 total pairs (6 contrasts x 10 domains x 4 templates)
   - 200 validated pairs
   - 40 ses_unimplemented pairs (preserving historical bug)
3. Deterministic pair_id generation:
   - Format 'P' + 11 hex characters (12 total)
   - Uniqueness across all 360 pairs
4. Candidate A+ phrasing & grammatical naturalness:
   - Uses "from a low socioeconomic background" / "from a high socioeconomic background"
   - Absence of metalinguistic "(low SES)" / "(high SES)"
   - Absence of forbidden SES proxies (uninsured, wealth, food stamps, coaching, etc.)
5. Strict counterfactual isolation:
   - SES comparisons differ ONLY by low -> high
   - Race comparisons differ ONLY by Black -> White
   - Reject any unintended lexical difference
6. 2x2 factorial contrast terminology & structure:
   - 4 pairwise contrasts constructed from 4 factorial cells (NOT "CDS values for four cells")
   - Descriptive comparison of SES disparity across racial groups / moderation signal
7. Deterministic per-sample seed strategy:
   - SHA-256 derived from (experiment_id, model_id, pair_id, condition, sample_idx)
   - Distinct seeds for distinct identities
   - Valid 31-bit integer range
8. File output, JSONL serialization, and SHA-256 manifest verification
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from fairlens.corpus import (
    CorpusValidator,
    make_pair_id,
    sha256_of_pair,
    sha256_of_file,
    CANONICAL_MAIN_PAIRS,
    CANONICAL_SES_PAIRS,
    CANONICAL_FAIRLENS_PAIRS,
    CANONICAL_LEGACY_PAIRS,
    SES_TEMPLATES,
    FORBIDDEN_SES_PROXIES,
)
from fairlens.seeds import derive_sample_seed, make_sample_identity_dict
from consist.config import INTERSECTIONAL_GROUPS
from consist.prompts import BIAS_DOMAINS, TEMPLATES


# ---------------------------------------------------------------------------
# 1. FAIRLens v1 Pair Counts (360 pairs)
# ---------------------------------------------------------------------------

class TestFairlensV1PairCounts:
    """Verify the 360-pair structure of the frozen FAIRLens v1 corpus."""

    def setup_method(self):
        self.validator = CorpusValidator(track="fairlens_v1")
        self.pairs = self.validator.generate_all()

    def test_total_pairs_360(self):
        """FAIRLens v1 corpus must contain exactly 360 pairs."""
        assert len(self.pairs) == 360, (
            f"Expected 360 pairs, got {len(self.pairs)}. "
            f"Expected 200 main + 160 SES factorial."
        )

    def test_all_360_pairs_validated(self):
        """All 360 pairs in FAIRLens v1 must pass audit validation."""
        validated = [p for p in self.pairs if p.ses_status == "validated"]
        assert len(validated) == 360, (
            f"Expected 360 validated pairs, got {len(validated)}"
        )

    def test_zero_rejected_pairs(self):
        """No pairs should fail hard audit checks."""
        rejected = [p for p in self.pairs if p.ses_status == "rejected"]
        assert len(rejected) == 0, (
            f"Found {len(rejected)} rejected pairs: "
            f"{[p.pair_id for p in rejected[:5]]}"
        )

    def test_zero_unimplemented_ses_pairs_in_v1(self):
        """In FAIRLens v1, SES is implemented; zero pairs should be ses_unimplemented."""
        unimplemented = [p for p in self.pairs if p.ses_status == "ses_unimplemented"]
        assert len(unimplemented) == 0

    def test_module_breakdown(self):
        """Exactly 200 main race/gender pairs and 160 SES factorial pairs."""
        main_pairs = [p for p in self.pairs if p.module == "main"]
        ses_pairs = [p for p in self.pairs if p.module == "ses_factorial"]
        assert len(main_pairs) == 200, f"Expected 200 main pairs, got {len(main_pairs)}"
        assert len(ses_pairs) == 160, f"Expected 160 SES factorial pairs, got {len(ses_pairs)}"

    def test_domain_counts_36_each(self):
        """Each of the 10 domains must have exactly 36 pairs (20 main + 16 SES)."""
        for domain in BIAS_DOMAINS:
            d_pairs = [p for p in self.pairs if p.domain == domain]
            assert len(d_pairs) == 36, (
                f"Domain {domain!r}: expected 36 pairs, got {len(d_pairs)}"
            )

    def test_group_contrast_counts_40_each(self):
        """Each of the 9 group contrasts must have exactly 40 pairs (10 domains x 4 templates)."""
        for ga, gb in CANONICAL_FAIRLENS_PAIRS:
            gp_pairs = [p for p in self.pairs if p.group_a == ga and p.group_b == gb]
            assert len(gp_pairs) == 40, (
                f"{ga} vs {gb}: expected 40 pairs, got {len(gp_pairs)}"
            )


# ---------------------------------------------------------------------------
# 2. Legacy Reproduction Pair Counts (240 pairs)
# ---------------------------------------------------------------------------

class TestLegacyPairCounts:
    """Verify legacy reproduction track maintains 240 pairs (with unrendered SES)."""

    def setup_method(self):
        self.validator = CorpusValidator(track="legacy_reproduction")
        self.pairs = self.validator.generate_all()

    def test_total_pairs_240(self):
        assert len(self.pairs) == 240

    def test_validated_pairs_200(self):
        validated = [p for p in self.pairs if p.ses_status == "validated"]
        assert len(validated) == 200

    def test_ses_unimplemented_pairs_40(self):
        unimplemented = [p for p in self.pairs if p.ses_status == "ses_unimplemented"]
        assert len(unimplemented) == 40

    def test_domain_counts_24_each(self):
        for domain in BIAS_DOMAINS:
            d_pairs = [p for p in self.pairs if p.domain == domain]
            assert len(d_pairs) == 24


# ---------------------------------------------------------------------------
# 3. Deterministic Pair IDs
# ---------------------------------------------------------------------------

class TestPairID:

    def test_pair_id_deterministic(self):
        id1 = make_pair_id("Black_Female", "White_Male", "hiring", 0)
        id2 = make_pair_id("Black_Female", "White_Male", "hiring", 0)
        assert id1 == id2

    def test_pair_id_format(self):
        pid = make_pair_id("Black_LowSES_Male", "Black_HighSES_Male", "crime_justice", 0)
        assert pid.startswith("P")
        assert len(pid) == 12

    def test_all_360_fairlens_pair_ids_unique(self):
        validator = CorpusValidator(track="fairlens_v1")
        pairs = validator.generate_all()
        ids = [p.pair_id for p in pairs]
        assert len(ids) == 360
        assert len(ids) == len(set(ids)), f"Found {len(ids) - len(set(ids))} duplicates"


# ---------------------------------------------------------------------------
# 4. Candidate A+ Phrasing & Absence of Proxies
# ---------------------------------------------------------------------------

class TestCandidateAPlus:

    def setup_method(self):
        self.validator = CorpusValidator(track="fairlens_v1")
        self.pairs = self.validator.generate_all()
        self.ses_pairs = [p for p in self.pairs if p.module == "ses_factorial"]

    def test_candidate_a_plus_phrasing_present(self):
        """Every SES prompt must contain natural socioeconomic background descriptor."""
        for p in self.ses_pairs:
            assert (
                "from a low socioeconomic background" in p.prompt_a
                or "from a high socioeconomic background" in p.prompt_a
            ), f"Missing A+ phrasing in prompt_a: {p.prompt_a}"
            assert (
                "from a low socioeconomic background" in p.prompt_b
                or "from a high socioeconomic background" in p.prompt_b
            ), f"Missing A+ phrasing in prompt_b: {p.prompt_b}"

    def test_no_metalinguistic_parentheses(self):
        """FAIRLens v1 prompts must not use metalinguistic '(low SES)' or '(high SES)'."""
        for p in self.pairs:
            combined = p.prompt_a + " " + p.prompt_b
            assert "(low ses)" not in combined.lower()
            assert "(high ses)" not in combined.lower()

    def test_absence_of_forbidden_proxies(self):
        """No prompt may contain forbidden SES proxies (uninsured, wealth, food stamps, etc.)."""
        for p in self.pairs:
            combined = (p.prompt_a + " " + p.prompt_b).lower()
            for proxy in FORBIDDEN_SES_PROXIES:
                assert proxy not in combined, (
                    f"Forbidden proxy {proxy!r} found in pair {p.pair_id}: {combined}"
                )

    def test_gender_fixed_as_male_in_ses_module(self):
        """All SES factorial pairs in FAIRLens v1 must have male gender."""
        for p in self.ses_pairs:
            assert p.group_a_gender == "male"
            assert p.group_b_gender == "male"
            assert "male" in p.prompt_a.lower()
            assert "male" in p.prompt_b.lower()


# ---------------------------------------------------------------------------
# 5. Strict Counterfactual Isolation
# ---------------------------------------------------------------------------

class TestStrictCounterfactualIsolation:

    def setup_method(self):
        self.validator = CorpusValidator(track="fairlens_v1")
        self.pairs = self.validator.generate_all()
        self.ses_pairs = [p for p in self.pairs if p.module == "ses_factorial"]

    def test_ses_comparisons_differ_only_by_low_high(self):
        """For Black Low vs High and White Low vs High, diff is strictly 'low' -> 'high'."""
        ses_contrasts = [
            p for p in self.ses_pairs
            if p.contrast_type == "ses_comparison"
        ]
        assert len(ses_contrasts) == 80  # 2 contrasts x 40 = 80 pairs
        for p in ses_contrasts:
            assert p.audit.counterfactual_isolated, (
                f"Counterfactual isolation failed for {p.pair_id}: {p.audit.notes}"
            )
            assert p.counterfactual_diff == "low -> high"

    def test_race_comparisons_differ_only_by_black_white(self):
        """For Black Low vs White Low and Black High vs White High, diff is strictly 'Black' -> 'White'."""
        race_contrasts = [
            p for p in self.ses_pairs
            if p.contrast_type == "race_comparison"
        ]
        assert len(race_contrasts) == 80  # 2 contrasts x 40 = 80 pairs
        for p in race_contrasts:
            assert p.audit.counterfactual_isolated, (
                f"Counterfactual isolation failed for {p.pair_id}: {p.audit.notes}"
            )
            assert p.counterfactual_diff == "Black -> White"

    def test_no_grammatical_artifacts(self):
        """No prompt in FAIRLens v1 has double spaces or doubled articles."""
        for p in self.pairs:
            assert "  " not in p.prompt_a
            assert "  " not in p.prompt_b
            assert p.audit.grammatically_valid, f"Grammar check failed for {p.pair_id}"


# ---------------------------------------------------------------------------
# 6. 2x2 Factorial Contrast Terminology & Structure
# ---------------------------------------------------------------------------

class TestFactorialDesignStructure:

    def setup_method(self):
        self.validator = CorpusValidator(track="fairlens_v1")
        self.pairs = self.validator.generate_all()
        self.ses_pairs = [p for p in self.pairs if p.module == "ses_factorial"]

    def test_four_pairwise_contrasts_present(self):
        """Verify the 4 controlled comparisons."""
        contrast_names = set(p.contrast_name for p in self.ses_pairs)
        expected_names = {
            "Black Low SES vs Black High SES",
            "White Low SES vs White High SES",
            "Black Low SES vs White Low SES",
            "Black High SES vs White High SES",
        }
        assert contrast_names == expected_names

    def test_held_constant_variables(self):
        """Verify race is held constant for SES comparisons, SES for race comparisons."""
        for p in self.ses_pairs:
            if p.contrast_type == "ses_comparison":
                assert p.held_constant == "race"
            elif p.contrast_type == "race_comparison":
                assert p.held_constant == "ses"


# ---------------------------------------------------------------------------
# 7. Deterministic Per-Sample Seed Strategy
# ---------------------------------------------------------------------------

class TestDeterministicSeeds:

    def test_deterministic_seed_derivation(self):
        seed1 = derive_sample_seed("exp_001", "phi3_mini", "P12345678901", "A", 0)
        seed2 = derive_sample_seed("exp_001", "phi3_mini", "P12345678901", "A", 0)
        assert seed1 == seed2

    def test_different_seeds_for_sample_indices(self):
        seed0 = derive_sample_seed("exp_001", "phi3_mini", "P12345678901", "A", 0)
        seed1 = derive_sample_seed("exp_001", "phi3_mini", "P12345678901", "A", 1)
        assert seed0 != seed1

    def test_different_seeds_for_conditions(self):
        seedA = derive_sample_seed("exp_001", "phi3_mini", "P12345678901", "A", 0)
        seedB = derive_sample_seed("exp_001", "phi3_mini", "P12345678901", "B", 0)
        assert seedA != seedB

    def test_different_seeds_for_models(self):
        s1 = derive_sample_seed("exp_001", "phi3_mini", "P12345678901", "A", 0)
        s2 = derive_sample_seed("exp_001", "qwen25_7b", "P12345678901", "A", 0)
        assert s1 != s2

    def test_seed_in_valid_31bit_range(self):
        seed = derive_sample_seed("exp_001", "phi3_mini", "P12345678901", "A", 0)
        assert 0 <= seed < 2147483647


# ---------------------------------------------------------------------------
# 8. Corpus Output & Manifest
# ---------------------------------------------------------------------------

class TestCorpusOutput:

    def test_fairlens_v1_save_and_manifest(self, tmp_path):
        validator = CorpusValidator(track="fairlens_v1")
        pairs = validator.generate_all()
        paths = validator.save(pairs, output_dir=str(tmp_path), track="fairlens_v1")

        corpus_file = Path(paths["corpus"])
        manifest_file = Path(paths["manifest"])
        assert corpus_file.exists()
        assert manifest_file.exists()

        with open(corpus_file, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 360

        with open(manifest_file, encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["counts"]["total_pairs"] == 360
        assert manifest["counts"]["validated"] == 360
        assert manifest["counts"]["main_race_gender_pairs"] == 200
        assert manifest["counts"]["ses_factorial_pairs"] == 160
        assert manifest["corpus_file_sha256"] == sha256_of_file(corpus_file)

    def test_frozen_corpus_manifest_hash_matches_file_bytes(self):
        """
        Regression test: Verify that the committed frozen corpus file on disk
        has an exact raw-byte SHA-256 matching its committed manifest.
        Guarantees cross-platform LF line endings and detects any drift.
        """
        repo_root = Path(__file__).parent.parent
        corpus_path = repo_root / "results" / "corpus" / "audit_corpus_fairlens_v1.jsonl"
        manifest_path = repo_root / "results" / "corpus" / "corpus_manifest_fairlens_v1.json"

        assert corpus_path.exists(), f"Committed corpus not found: {corpus_path}"
        assert manifest_path.exists(), f"Committed manifest not found: {manifest_path}"

        raw_bytes = corpus_path.read_bytes()
        actual_sha = hashlib.sha256(raw_bytes).hexdigest()

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        expected_sha = manifest["corpus_file_sha256"]
        assert actual_sha == expected_sha, (
            f"Corpus file SHA-256 mismatch!\n"
            f"  Actual raw bytes SHA: {actual_sha}\n"
            f"  Manifest recorded SHA: {expected_sha}"
        )



# ---------------------------------------------------------------------------
# 9. Pilot Pair Stratification
# ---------------------------------------------------------------------------

class TestPilotPairSelection:

    def test_stratified_pilot_selection_properties(self):
        from fairlens.corpus import select_stratified_pilot_pairs
        pilot_pairs = select_stratified_pilot_pairs()

        # Exactly 20 pairs
        assert len(pilot_pairs) == 20, f"Expected 20 pairs, got {len(pilot_pairs)}"

        # Both modules represented (10 main, 10 SES)
        main_pairs = [p for p in pilot_pairs if p.module == "main"]
        ses_pairs = [p for p in pilot_pairs if p.module == "ses_factorial"]
        assert len(main_pairs) == 10
        assert len(ses_pairs) == 10

        # All 10 domains represented
        domains = set(p.domain for p in pilot_pairs)
        assert domains == set(BIAS_DOMAINS)
        assert len(domains) == 10

        # Exactly 2 pairs per domain (1 main, 1 SES)
        for d in BIAS_DOMAINS:
            d_main = [p for p in main_pairs if p.domain == d]
            d_ses = [p for p in ses_pairs if p.domain == d]
            assert len(d_main) == 1, f"Expected 1 main pair for {d}, got {len(d_main)}"
            assert len(d_ses) == 1, f"Expected 1 SES pair for {d}, got {len(d_ses)}"

        # Volume calculation
        # 20 pairs x 2 conditions x K=20 = 800 generations
        assert len(pilot_pairs) * 2 * 20 == 800

