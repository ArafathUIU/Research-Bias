# Comparative Evaluation of Contextual Dispersion Bias Across Open-Source LLMs

## One-Line Pitch
How does semantic output dispersion across demographic counterfactual pairs vary across model families and scales — and can a multi-signal behavioral profile (CDS, centroid distance, dispersion asymmetry) serve as a zero-cost comparative benchmark for LLM fairness?

---

## Motivation

Our pilot experiment on Phi-3-mini (3.8B) found that **CDS (Consistency Disparity Score) ≈ 0** overall (mean = 0.0015, p = 0.66), falsifying the hypothesis that self-consistency disparity alone detects intersectional bias. However, two findings emerged that are worth pursuing:

1. **Domain-level CDS structure** — CDS varied systematically across domains (education +0.024 → media −0.030), suggesting a training-data-dependent behavioral fingerprint.
2. **The multi-signal framework** — CDS, centroid separation, and dispersion asymmetry together provide a richer characterization than any single metric.

The natural next step: **run the same pipeline across multiple models** to see which patterns replicate, which diverge, and whether dispersion-based profiling can serve as a comparative tool for LLM auditing.

---

## Why This Is Novel

The closest existing work is **BSM (Bias Similarity Measurement, Jeong et al. 2025)**, which compares 30 LLMs on fairness using BBQ/UnQover/StereoSet (labeled classification benchmarks). Our approach differs fundamentally:

| Dimension | BSM (Jeong 2025) | This work |
|-----------|-----------------|-----------|
| Metric | Labeled accuracy + bias scores | Unsupervised semantic dispersion |
| Data source | BBQ, UnQover, StereoSet | Counterfactual generation pairs |
| What it measures | Classification fairness | Output behavioral profile |
| Model access | Black-box API | Open-weight only |
| Cost | API-dependent ($) | Zero (local GPU) |
| Intersectional? | Single-axis | Multi-axis (race×gender×SES) |

No existing work uses **output variability across demographic counterfactuals** as a comparative cross-model metric. This is a genuine gap.

---

## Model Suite

| Model | Family | Size | Status | Est. time (T4) |
|-------|--------|------|--------|----------------|
| Phi-3-mini-4k-instruct | Microsoft | 3.8B | **COMPLETED** | — |
| Qwen2.5-7B-Instruct | Alibaba | 7B | Pending | ~35 min |
| Mistral-7B-Instruct-v0.3 | Mistral AI | 7B | Pending | ~35 min |
| Llama-3.2-3B-Instruct | Meta | 3.8B | Pending | ~25 min |
| Gemma-2-9B-it | Google | 9B | Pending | ~45 min |

All run on Kaggle T4 in 4-bit NF4. Same prompt set (240 pairs, 10 domains, 6 group pairs, K=20, T=0.7). Same CONSIST pipeline.

---

## Research Questions

**RQ1 (Cross-model CDS):** Does CDS remain null across all models, or do some models show significant consistency disparity? Does model size correlate with CDS magnitude?

**RQ2 (Domain profile):** Does the domain-level ordering (education + → media −) replicate across models, or is it model-specific?

**RQ3 (Centroid separation):** Is centroid distance between matched groups systematically structured by model family, scale, or domain?

**RQ4 (Behavioral fingerprint):** Can a 3-signal profile (CDS, centroid distance, dispersion asymmetry) distinguish model families and scales better than any single metric?

---

## Expected Outcomes

| Scenario | Probability | Publication angle |
|----------|------------|-------------------|
| All models null on CDS, domain structure replicates | Medium | "Consistent null + domain fingerprint across 5 models" — behavioral profiling paper |
| Larger models (7B+) show CDS signal | Medium | "Scale-dependent emergence of consistency disparity" — extends the theory |
| CDS null but centroid distance shows structure | High | "Centroid separation as a more sensitive behavioral probe" — multi-signal framework |
| All metrics null across all models | Low | "Comprehensive null result across 5 model families" — constrains the theory |

---

## Methodology

**Pipeline** (existing, `consist/` package):
1. Load model in 4-bit on T4
2. Generate K=20 samples for each of 240 counterfactual prompt pairs (10 domains × 6 group pairs × 4 templates)
3. Embed all outputs via sentence-transformers/all-MiniLM-L6-v2
4. Compute per-pair: CDS, dispersion_a, dispersion_b, centroid distance, bootstrap CI
5. Aggregate by domain, group pair, and model

**Analysis:**
- CDS t-test against zero per model
- Domain ordering correlation across models (Spearman)
- Model × domain interaction (ANOVA)
- Centroid distance × model comparison
- Multi-signal PCA/profiling per model

---

## Resource Requirements

| Resource | Cost | Duration |
|----------|------|---------|
| Kaggle T4 GPU | $0 (30h/week free) | ~2.5 hours total |
| Storage | $0 | <1GB per model |
| Existing code | $0 | CONSIST pipeline ready |
| **Total** | **$0** | **~2.5 hours** |

---

## Contributions

1. **First cross-model comparison using semantic dispersion** as a behavioral probe for demographic representation
2. **Systematic evaluation of 5 models** across 3 families and 2 scales (3.8B–9B) on 240 counterfactual pairs
3. **Multi-signal profiling framework** — CDS, centroid distance, and dispersion asymmetry as complementary signals
4. **Open-source release** of comparative evaluation pipeline + results

---

## Target Venues

- **ACL 2027 / EMNLP 2026 Findings** — empirical study
- **NeurIPS 2026 Datasets & Benchmarks** — if we release as a benchmark
- **EACL 2027** — full multi-model behavioral profiling paper
- **AIES / FAccT** — fairness-focused venue

---

## Risk Analysis

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| All models show null CDS | Medium | Centroid distance may still show signal; domain fingerprint still publishable |
| Kaggle limits interrupt runs | Low | Checkpointing saves after each pair; resumeable |
| BSM paper overlaps | Low | Different metric (dispersion vs labels), different models (open only), different cost ($0 vs API) |
| "Only 5 models" criticism | Medium | Comparable to existing work (Neutral Is Not Unbiased: 5 models); BSM covers 30 but uses cheaper classification metrics |

---

## Immediate Next Steps

1. **User go/no-go** on this direction
2. **Kaggle: run remaining 4 models** (sequential, ~2.5 hours total)
3. **Analyze results** — cross-model comparison, domain replication, centroid separation
4. **Write paper** — structure determined by findings
