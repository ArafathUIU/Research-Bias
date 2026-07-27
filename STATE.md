# State: Comparative Evaluation of Contextual Dispersion Bias Across Open-Source LLMs

## Phase status
- fuzzy: done (3 landscape searches)
- source: 3 pages captured (BSM, Neutral Is Not Unbiased, F²Bench)
- think: 3 sources processed
- evolve: skipped (cycle 1, no drift detected)
- gap: done (cycle 1 — full)
- revisit: not yet
- synthesize: ready (proposal written at _index.md)
- cycle: 1

## Current question
How does semantic output dispersion across demographic counterfactual pairs vary across model families and scales?

## Question evolution
v1: Can CDS detect intersectional bias? → FALSIFIED (CDS ≈ 0)
v2: What does CDS actually capture? → Domain-structured behavioral fingerprint
v3: Can comparative CDS across models serve as a zero-cost fairness benchmark? (current)

## Question drift
- detected: no

## Belief registry
- Claim: "CDS is null across models at 3.8B scale" — confidence 4/5 (only tested Phi-3-mini)
- Claim: "Domain-level CDS structure (education + → media -) may replicate across models" — confidence 2/5 (unverified)
- Claim: "No prior work uses dispersion as a cross-model comparative metric" — confidence 5/5 (verified via fuzzy)

## Saturation
- conceptual: yes — comparative evaluation is well-defined
- predictive: no — results unknown
- model stability: no

## Next suggested action
User go/no-go on running the model suite (4 models, ~2.5 hours on Kaggle)
