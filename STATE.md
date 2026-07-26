# State: Divergent Representations

## Phase status
- fuzzy: done (3 landscape searches — 4 months ago)
- source: 5 pages captured (notes/v1/) + 4 validation papers
- think: 5 sources processed + 1 experimental result (240 pairs)
- evolve: EVOLVED — hypothesis falsified, new direction established
- gap: pending (new direction needs gap analysis)
- revisit: pending
- synthesize: pending (after centroid separation analysis)
- cycle: 4 (experimental phase)

## Current question
What does the way an LLM generates text reveal about how it represents different demographic groups — and can we build a multi-signal behavioral profile from a single generation pass?

## Question evolution
v1: How to measure intersectional bias in LLMs with zero budget? (CCIQ, $16K, dead)
v2: Can self-consistency disparity serve as a zero-cost unsupervised bias metric? (CONSIST)
v3: What does CDS actually capture, if not bias? (DIVERGENT REPRESENTATIONS — current)

## Question drift
- detected: yes
- trigger: Pilot experiment results (240 pairs, Phi-3-mini, mean CDS = 0.0015, p = 0.66)
- signal: Clean falsification of original hypothesis. Domain-level structure (education +0.024 vs media −0.030) suggests behavioral profiling, not bias detection.

## Belief registry
- Claim: "Self-consistency disparity is not a universal unsupervised bias metric for small LLMs (3.8B at K=20)"
  - confidence: 5/5
  - evidence_for: 240 pairs, t(239)=0.45, p=0.66
  - evidence_against: None
  - tags: [FACT: EMPIRICAL]
- Claim: "CDS shows systematic domain-level structure even when aggregate mean is zero"
  - confidence: 3/5
  - evidence_for: education +0.024 (p=0.019), leadership +0.021 (p=0.035), media −0.030 (p=0.021)
  - evidence_against: Multiple comparisons not corrected; only 3/10 domains significant at α=0.05
  - tags: [INTERPRETATION]
- Claim: "LowSES_Black vs HighSES_White shows highest CDS magnitude among group pairs"
  - confidence: 2/5
  - evidence_for: Mean CDS = +0.011 (highest of 6 pairs)
  - evidence_against: p = 0.24, not significant
  - tags: [SPECULATION]
- Claim: "Proposition 1 (entropy-consistency link) may not be empirically detectable at 3.8B scale"
  - confidence: 3/5
  - evidence_for: Null CDS across 240 pairs
  - evidence_against: Only tested one model at one temperature
  - tags: [INTERPRETATION]

## Saturation
- conceptual: no — new direction just formed
- predictive: no
- model stability: no
- all three: no

## Parking lot
- Domain ordering (education +0.024 → media −0.030) may correlate with training data topic distribution
- SES > race > gender in CDS magnitude aligns with Boufaied et al. 2026
- Centroid separation may be a stronger signal than CDS (not yet analyzed)

## Next suggested action
1. Write emerging findings to _meta/
2. Analyze centroid separation patterns from existing data
3. Run Qwen2.5-7B for cross-model replication
4. Gap analysis on the new direction
