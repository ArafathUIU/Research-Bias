# Emerging Findings — Cycle 4 (Experimental Phase)

## Entry 1: CDS Hypothesis Falsified
**Source:** Pilot experiment (240 pairs, Phi-3-mini, 10 domains × 6 group pairs × 20 samples)
**Finding:** Mean CDS = 0.0015 (p = 0.66). Hypothesis cleanly falsified. Self-consistency disparity does not detect intersectional bias in this setting.
**Implication:** The causal chain (training imbalance → epistemic uncertainty → dispersion asymmetry) may not be empirically detectable at 3.8B scale, or requires more samples (K > 20), or only manifests at larger model scales.

## Entry 2: Domain-Structured CDS
**Source:** Same experiment, domain-level aggregation
**Finding:** CDS varies systematically by domain, from +0.024 (education, p=0.019) to −0.030 (media_portrayal, p=0.021). The sign reverses across domains — not compatible with a simple "marginalized = more variable" story.
**Implication:** CDS may be a domain-specific behavioral fingerprint rather than a bias metric. Potentially related to training data coverage per domain × demographic group.

## Entry 3: SES Shows Weakest Signal
**Source:** Same experiment, group-pair aggregation
**Finding:** LowSES_Black vs HighSES_White has highest absolute CDS (+0.011), consistent with Boufaied 2026 finding that Race×SES is the hardest intersection. But not statistically significant (p=0.24).
**Implication:** SES may amplify bias signals more than race or gender alone. Worth pursuing with larger N or larger model.

## Entry 4: New Research Direction Established
**Source:** Synthesis of all findings
**New direction:** Multi-signal behavioral profiling framework — jointly measure semantic stability (CDS), dispersion asymmetry, and centroid separation to characterize how LLMs represent demographic groups. Not a bias detector, but a characterization tool.
**Documented in:** DIVERGENT_REPRESENTATIONS.md
