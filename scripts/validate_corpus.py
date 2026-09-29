"""
Corpus validation and generation script.
Generates the audit corpus JSONL and manifest for both experiment tracks.

Run: python scripts/validate_corpus.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fairlens.corpus import CorpusValidator


def main():
    validator = CorpusValidator()
    print("Generating all 240 prompt pairs...")
    pairs = validator.generate_all()
    validator.print_summary(pairs)

    print("Saving legacy_reproduction corpus (240 pairs)...")
    legacy_paths = validator.save(pairs, "results/corpus", track="legacy_reproduction")
    print(f"  corpus   : {legacy_paths['corpus']}")
    print(f"  manifest : {legacy_paths['manifest']}")

    print("\nSaving fairlens_v1 corpus...")
    v1_paths = validator.save(pairs, "results/corpus", track="fairlens_v1")
    print(f"  corpus   : {v1_paths['corpus']}")
    print(f"  manifest : {v1_paths['manifest']}")

    # Show 4 example pairs across contrasts
    v1_pairs = validator.filter_fairlens_v1(pairs)
    print(f"\nExample rendered pairs from fairlens_v1 corpus:")
    for pair in v1_pairs[:4]:
        print(f"\n  [{pair.pair_id}] {pair.group_a} vs {pair.group_b} | {pair.domain} T{pair.template_idx}")
        print(f"  A: {pair.prompt_a[:100]}{'...' if len(pair.prompt_a) > 100 else ''}")
        print(f"  B: {pair.prompt_b[:100]}{'...' if len(pair.prompt_b) > 100 else ''}")


if __name__ == "__main__":
    main()
