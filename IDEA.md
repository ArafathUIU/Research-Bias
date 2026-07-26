# IDEA: Divergent Representations

## A Multi-Signal Framework for Zero-Cost Behavioral Probing of Demographic Representation in LLMs

---

### The Core Finding That Changed Direction

The CONSIST pilot experiment (Phi-3-mini, 240 matched pairs, 10 domains, 6 demographic comparisons) cleanly falsified our original hypothesis:

**CDS (Consistency Disparity Score) ≈ 0 across all group pairs.** Overall mean: 0.0015. No domain, no group pair, no interaction showed significant signal.

Self-consistency disparity is **not** a universal unsupervised bias metric — at least not for small models at K=20.

This null result is valuable: it constrains the theory (Proposition 1's entropy-consistency link may be too weak to detect at this scale) and saves the field from a false positive. Alone it supports a findings/negative-result paper. But combined with other signals from the same pipeline, it enables something stronger.

---

### The New Question

> What does the way an LLM generates text reveal about how it represents different demographic groups — and can we build a multi-signal behavioral profile from a single generation pass?

---

### Three Signals, One Pipeline

Every counterfactual prompt pair produces N=20 samples per group. From those samples we extract three independent signals using the existing infrastructure:

| # | Signal | What it measures | Expected direction | Status |
|---|--------|-----------------|-------------------|--------|
| **1** | **Semantic Stability (CDS)** | Within-group response variability | Marginalized groups → higher variability | MEASURED: null |
| **2** | **Dispersion Asymmetry** | Which group in a pair has higher within-group scatter | Prevalence of positive vs negative CDS per domain | NOT ANALYZED |
| **3** | **Centroid Separation** | Between-group distance in embedding space | Matched groups should be closer than unmatched groups | NOT ANALYZED |

**Key insight:** Signals 1 (CDS) and 3 (centroid distance) are **orthogonal** — a model could treat two groups identically (low centroid distance) while being highly variable for both (high dispersion for both → CDS ≈ 0). Or the reverse. Their joint distribution characterizes the model's demographic representation in a way neither signal alone captures.

The data from the pilot run already contains all three signals — no additional model inference needed. We only need the full JSON with dispersion and centroid distance columns.

---

### Research Questions

**RQ1 (Empirical):** Is centroid separation between matched demographic groups consistently structured by domain, group pair, or template type? Does it reveal bias patterns where CDS did not?

**RQ2 (Measurement):** How do CDS, dispersion asymmetry, and centroid separation correlate across 240 pairs? Do they form a coherent profile or are they independent?

**RQ3 (Methodological):** Can a multi-signal behavioral profile — rather than any single metric — characterize demographic representation in LLMs better than existing single-metric approaches?

**RQ4 (Scalability):** Do these patterns replicate across model scales (Phi-3-mini 3.8B → Qwen2.5-7B → larger)?

---

### What We Already Have

From the completed pilot run:
- 240 matched prompt pairs × 2 groups = 9600 generated texts
- CDS, dispersion_a, dispersion_b, centroid_dist, CI bounds for each pair
- 10 domains × 6 group pairs
- All on Phi-3-mini (3.8B)

The analysis is purely post-hoc computation on existing outputs — zero additional GPU cost.

---

### What We Need Next

| Item | Status | Action |
|------|--------|--------|
| Full JSON with dispersion + centroid dist | Kaggle | Download from Kaggle to `results/progress.csv` |
| Exploratory analysis | Module done (`consist/explore.py`) | Run on full data |
| Second model (Qwen2.5-7B, ~30 min on T4) | Not started | Replicate on 7B model for scale comparison |
| Lexical analysis of raw text | Not started | Re-run with text saving enabled |

---

### Paper Framing

**Title options:**
1. "Divergent Representations: A Multi-Signal Behavioral Probe of Demographic Representation in LLMs"
2. "Self-Consistency Disparity Is Not Bias: A Null Result and What It Reveals About LLM Representation"
3. "Three Signals, One Pipeline: Characterizing Demographic Representation Through Generation Behavior"

**Target venues:**
- *ACL Rolling Review / Findings* — for negative-result paper
- *NeurIPS 2026 Datasets & Benchmarks* — if we release the pipeline as benchmark
- *EACL 2027 / EMNLP 2026* — for full multi-signal paper

**Contribution:**
1. **Negative empirical result:** CDS does not detect intersectional bias (Phi-3-mini, 240 pairs) — constrains theory
2. **Multi-signal framework:** First systematic comparison of CDS, dispersion, and centroid distance as representation probes
3. **Open-source pipeline:** CONSIST pipeline released for zero-cost behavioral profiling
4. **Cross-scale comparison:** How signals change from 3.8B → 7B (conditional on running RQ4)

---

### Risk Analysis

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| All three signals are null | Medium | Still publishable as negative result + framework paper |
| Centroid separation is confounded by template | High | Control: compare within-domain centroid distances only |
| Only one model tested | Medium | RQ4 adds Qwen2.5-7B (same pipeline, 30 min) |
| Reviewers reject "not a bias metric" | Low | Frame as behavioral profiling, not bias detection |
| No raw text for lexical analysis | Medium | Re-run with text saving (Kaggle, 30 min) |

---

### Immediate Next Step

1. **Export the full Kaggle DataFrame** as JSON/CSV and get it into `results/`
2. **Run `consist/explore.py`** to get the complete report (domain × group interaction, centroid separation, outliers)
3. **Review the output** to decide: is there a signal to pursue, or is this a pure negative-result paper?
4. **Frame the paper** based on evidence

---

*This document replaces CONSIST_Research_Proposal.md as the active research direction. The old proposal is preserved for reference.*
