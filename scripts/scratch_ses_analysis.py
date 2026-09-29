"""
Scratch analysis: SES operationalization candidates.

This script evaluates 3 candidate SES manipulations across all 10 domains
to check consistency, token diff, and potential confounds.

Run locally (CPU): python -m scripts.scratch_ses_analysis
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# The 10 existing domain templates (from consist/prompts.py)
from consist.prompts import TEMPLATES, BIAS_DOMAINS  # noqa: E402

# ============================================================
# CANDIDATE A: Bare background descriptor
# "from a low socioeconomic background" vs "from a high socioeconomic background"
# Inserted immediately after {race} {gender}.
# ============================================================
CANDIDATE_A = {
    "low":  "from a low socioeconomic background",
    "high": "from a high socioeconomic background",
}

# ============================================================
# CANDIDATE B: Neighborhood/community descriptor (minimal)
# "from an under-resourced community" vs "from a well-resourced community"
# ============================================================
CANDIDATE_B = {
    "low":  "from an under-resourced community",
    "high": "from a well-resourced community",
}

# ============================================================
# CANDIDATE C: Parenthetical SES note (bracketed)
# "(low SES)" vs "(high SES)"
# Most minimal, but least natural.
# ============================================================
CANDIDATE_C = {
    "low":  "(low SES)",
    "high": "(high SES)",
}


def render_pair(template: str, race_a: str, gender_a: str,
                race_b: str, gender_b: str, ses_a: str, ses_b: str):
    """
    Render A and B with {ses_context} injected after {race} {gender}.
    Template uses {race} and {gender}; we add SES separately.
    """
    # Build extended subject string
    subj_a = f"{race_a} {gender_a} {ses_a}".strip()
    subj_b = f"{race_b} {gender_b} {ses_b}".strip()

    # Replace {race} {gender} as a unit (keeping original template format)
    t_a = template.replace("{race} {gender}", subj_a)
    t_b = template.replace("{race} {gender}", subj_b)

    # Some templates start with just {race} or {gender} alone
    t_a = t_a.replace("{race}", race_a).replace("{gender}", gender_a)
    t_b = t_b.replace("{race}", race_b).replace("{gender}", gender_b)
    return t_a, t_b


def token_diff(s_a: str, s_b: str):
    """Find all word positions that differ between two strings."""
    words_a = s_a.split()
    words_b = s_b.split()
    diffs = []
    for i, (w_a, w_b) in enumerate(zip(words_a, words_b)):
        if w_a != w_b:
            diffs.append((i, w_a, w_b))
    if len(words_a) != len(words_b):
        diffs.append(("LENGTH", len(words_a), len(words_b)))
    return diffs


print("=" * 72)
print("SES OPERATIONALIZATION ANALYSIS — All 10 Domains × 3 Candidates")
print("=" * 72)

# Four factorial conditions:
# (race=Black, ses=low) vs (race=Black, ses=high)  → pure SES comparison
# (race=White, ses=low) vs (race=White, ses=high)  → pure SES comparison
# (race=Black, ses=low) vs (race=White, ses=low)   → pure race comparison
# (race=Black, ses=high) vs (race=White, ses=high) → pure race comparison

FACTORIAL_CONTRASTS = [
    {
        "name": "Black_LowSES vs Black_HighSES (pure SES, race held constant)",
        "race_a": "Black", "gender_a": "male",
        "race_b": "Black", "gender_b": "male",
        "ses_key_a": "low", "ses_key_b": "high",
    },
    {
        "name": "White_LowSES vs White_HighSES (pure SES, race held constant)",
        "race_a": "White", "gender_a": "male",
        "race_b": "White", "gender_b": "male",
        "ses_key_a": "low", "ses_key_b": "high",
    },
    {
        "name": "Black_LowSES vs White_LowSES (pure race, SES held constant)",
        "race_a": "Black", "gender_a": "male",
        "race_b": "White", "gender_b": "male",
        "ses_key_a": "low", "ses_key_b": "low",
    },
    {
        "name": "Black_HighSES vs White_HighSES (pure race, SES held constant)",
        "race_a": "Black", "gender_a": "male",
        "race_b": "White", "gender_b": "male",
        "ses_key_a": "high", "ses_key_b": "high",
    },
]

for cand_name, cand in [("A (background descriptor)", CANDIDATE_A),
                          ("B (community descriptor)", CANDIDATE_B),
                          ("C (bracketed abbreviation)", CANDIDATE_C)]:
    print(f"\n{'-' * 72}")
    print(f"CANDIDATE {cand_name}")
    print(f"  LOW:  '{cand['low']}'")
    print(f"  HIGH: '{cand['high']}'")
    print(f"{'-' * 72}")

    for domain in BIAS_DOMAINS:
        template = TEMPLATES[domain][0]  # just test first template per domain
        print(f"\n  [{domain}] Template: {template[:70]}...")

        for contrast in FACTORIAL_CONTRASTS:
            t_a, t_b = render_pair(
                template,
                contrast["race_a"], contrast["gender_a"],
                contrast["race_b"], contrast["gender_b"],
                cand[contrast["ses_key_a"]], cand[contrast["ses_key_b"]],
            )
            diffs = token_diff(t_a, t_b)
            n_diffs = len([d for d in diffs if d[0] != "LENGTH"])
            has_length_diff = any(d[0] == "LENGTH" for d in diffs)
            status = "OK" if not has_length_diff else "LENGTH MISMATCH"
            print(f"    {contrast['name'][:60]}")
            print(f"    A: {t_a[:95]}")
            print(f"    B: {t_b[:95]}")
            print(f"    Diffs: {n_diffs} token(s) | {status}")

print("\n\nDone.")
