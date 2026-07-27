# Source: Khan et al. 2025 — WinoIdentity / Confidence Disparities
**URL:** https://arxiv.org/abs/2508.07111
**Lens:** opportunity-spotter
**Pre-read:** I'm reading this to find out how the closest existing intersectional bias benchmark is designed and what gaps remain.

## What it says
Extends WinoBias with 25 demographic markers across 10 attributes × binary gender → 245,700 prompts, 50 bias patterns. Uses coreference confidence disparity (CCD) as an uncertainty-based metric rather than accuracy. Finds disparities up to 40% across body type, sexual orientation, SES. Key finding: confidence drops even for privileged markers → suggests memorization > reasoning.

## My reaction
This is the strongest existing methodological template. It confirms that uncertainty-based metrics reveal bias where accuracy-based ones don't. The double-disadvantage pattern (e.g., transgender_fem on mechanic worse than either attribute alone) directly supports the proposal's intersectional framing. However, CCD is a single-task metric (coreference) — it doesn't measure quality degradation across multiple dimensions (SES×dialect×tone) as the base paper does. This opens a clear gap: extend confidence-based evaluation to multi-dimensional quality scoring.

## Contradictions
Does not directly contradict any prior source. However, the finding that even privileged markers show reduced confidence challenges the base paper's implicit assumption that bias only harms marginalized groups — quality degradation may be more symmetric than previously thought.

## Question raised
Can the confidence-disparity approach be extended from coreference to multi-attribute quality evaluation? Or does it require a narrow task with verifiable ground truth?
