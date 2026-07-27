# Source: Souani et al. 2025 — HInter
**URL:** https://arxiv.org/abs/2503.11962
**Lens:** data-hunter
**Pre-read:** I'm reading this to find automated (human-free) methods for intersectional bias detection.

## What it says
Combines mutation analysis + dependency parsing + metamorphic oracles to auto-generate test inputs. 14.61% of generated inputs expose intersectional bias. 16.62% of intersectional errors are "hidden" — their atomic (single-attribute) counterparts do not trigger bias. Evaluated on 18 models across 6 architectures.

## My reaction
This is directly relevant to the "no human evaluators" requirement. HInter proves that automated metamorphic testing can surface intersectional bias without human annotation. The dependency invariant reduces false positives by 10× — important for reliability. The "hidden bias" finding (16.62%) is the strongest argument for intersectional over single-axis testing. Key limitation: HInter tests classification/QA tasks, not generation quality.

## Contradictions
None yet.

## Question raised
Can HInter-style mutation analysis be applied to the SES×dialect×tone quality framework? The base paper uses regression-style quality scoring rather than classification — metamorphic oracles may need different design.
