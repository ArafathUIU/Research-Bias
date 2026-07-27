# Source: Intersectional Fairness in LLMs (arXiv 2604.20677, 2026)
**URL:** https://arxiv.org/abs/2604.20677
**Lens:** connector
**Pre-read:** I'm reading this to understand the most recent evaluation methodology for intersectional fairness.

## What it says
Systematic eval of 6 LLMs on BBQ-based intersectional datasets (Race×Gender, Race×SES). Uses bias scores, subgroup fairness metrics, accuracy, and consistency across repeated runs. Key findings: (1) LLMs perform well in ambiguous contexts but this limits informativeness due to sparse non-unknown predictions, (2) in disambiguated contexts, accuracy is tied to stereotype alignment, (3) no LLM achieves consistently fair behavior, (4) race×SES intersection is harder than race×gender.

## My reaction
This paper directly validates the proposal's focus on SES (race×SES is the hardest intersection). It also shows that accuracy-based metrics miss nuance — exactly the gap that the base paper's quality-gradient approach fills. The consistency-across-runs finding (even stereotype-aligned responses vary) supports the longitudinal drift framing. However, this uses BBQ's closed-ended MCQ format — open-ended generation quality is not captured.

## Contradictions
Broadly consistent with all prior sources. The finding that race×SES is the hardest intersection directly supports the base paper's focus on SES. No contradictions found.

## Question raised
The paper notes that "indeterminate predictions limit subgroup fairness metrics in ambiguous contexts" — does the same issue apply to the base paper's quality scoring (which uses continuous rather than discrete outputs)?
