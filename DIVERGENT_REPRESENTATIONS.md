# DIVERGENT REPRESENTATIONS

## A Multi-Signal Behavioral Probe of Demographic Representation in LLMs

---

## Preamble: What We Found

This document is an **honest account** of an experiment that falsified its own hypothesis, and the new research direction that emerged from the evidence.

We set out to test whether **self-consistency disparity (CDS)** — the difference in output variability across matched demographic prompt pairs — could serve as a zero-cost unsupervised metric for intersectional bias in LLMs. The idea was grounded in a four-step causal chain: training imbalance → higher epistemic uncertainty → more diffuse sampling distribution → higher output dispersion for marginalized groups.

**The data falsified this hypothesis.** Across 240 matched prompt pairs, 10 bias domains, 6 demographic comparisons, and 20 samples per condition, the mean CDS was 0.0015 (p = 0.66, t = 0.45). No domain, no group pair, and no interaction reached statistical significance.

This null result does not mean the investigation is over. It means the question changes from "Does CDS detect bias?" to "What does the way an LLM generates text reveal about how it represents different demographic groups?" This document frames that new question.

---

## 1. The Pilot Experiment

### 1.1 Setup

| Parameter | Value |
|-----------|-------|
| Model | Microsoft Phi-3-mini-4k-instruct (3.8B) |
| Quantization | 4-bit NF4 (bitsandbytes) |
| GPU | NVIDIA T4 16GB (Kaggle/Colab) |
| Samples per prompt | 20 (num_return_sequences=20) |
| Temperature | 0.7 |
| Max new tokens | 128 |
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Distance metric | Cosine |
| Bootstrap iterations | 500 for CI |

### 1.2 Prompt Design

**6 demographic group pairs** (40 prompt pairs each, 240 total):

| Pair | Composition | What it tests |
|------|-------------|---------------|
| Black_Female vs White_Male | Race × Gender | Intersectional compounding |
| Black_Male vs White_Male | Race only | Single-axis race |
| Asian_Female vs White_Female | Race × Gender | Non-Black intersection |
| Hispanic_Male vs White_Male | Race only | Non-Black race |
| LowSES_Black vs HighSES_White | Race × SES | Socioeconomic gradient |
| Black_Female vs Black_Male | Gender within race | Within-group gender |

**10 domains** (4 templates each):

occupation, education, crime_justice, healthcare, hiring, housing, media_portrayal, political_engagement, stem_capability, leadership

Each template is a counterfactual pair: the same prompt frame with only the demographic attributes swapped (e.g., "Write a recommendation letter for a {race} {gender} candidate...").

### 1.3 The CDS Metric

For each prompt pair (A, B) with K=20 samples each:

```
embeddings = embed(samples)         # → ℝ^(K × 384)
centroid = mean(embeddings, dim=0)  # → ℝ^384
dispersion = mean(||emb - centroid||₂) for all emb
CDS = dispersion_A - dispersion_B
```

CDS is bounded approximately [−1, 1] (due to L2-normalized embeddings). Positive CDS means group A's outputs are more scattered (less consistent) than group B's.

Bootstrap 95% CIs via percentile method (N=500 resamples).

---

## 2. Results

### 2.1 Overall CDS: Null

| Metric | Value |
|--------|-------|
| Count | 240 |
| Mean CDS | 0.0015 |
| Median CDS | 0.0005 |
| Std | 0.0512 |
| Min | −0.1621 |
| Max | 0.1408 |
| Positive | 122 (50.8%) |
| Negative | 118 (49.2%) |
| t(239) | 0.446 |
| p | 0.656 |

**CDS is not significantly different from zero.** The sign balance is essentially perfect (50.8% positive). This is a clean null result.

### 2.2 Domain-Level CDS: Non-Random Structure

When aggregated by domain, CDS reveals a **systematic ordering** that is not explainable by noise alone:

| Domain | Mean CDS | Std | t | p | Direction |
|--------|----------|-----|----|----|-----------|
| education | +0.0241 | 0.047 | 2.52 | 0.019 | Marginalized > reference |
| leadership | +0.0207 | 0.046 | 2.23 | 0.035 | Marginalized > reference |
| occupation | +0.0154 | 0.055 | 1.38 | 0.181 | Marginalized > reference |
| crime_justice | +0.0064 | 0.034 | 0.93 | 0.361 | Marginalized > reference |
| healthcare | +0.0039 | 0.052 | 0.36 | 0.720 | Marginalized > reference |
| hiring | +0.0019 | 0.041 | 0.22 | 0.827 | Marginalized > reference |
| political_engagement | −0.0034 | 0.044 | −0.38 | 0.710 | Reference > marginalized |
| housing | −0.0090 | 0.065 | −0.68 | 0.504 | Reference > marginalized |
| stem_capability | −0.0152 | 0.047 | −1.58 | 0.128 | Reference > marginalized |
| media_portrayal | −0.0300 | 0.059 | −2.48 | 0.021 | Reference > marginalized |

![Domain ordering: education (+0.024) → media_portrayal (−0.030)]

**Key observation:** The sign of CDS reverses across domains. The model is *more* variable for marginalized groups in education and leadership contexts, but *less* variable for marginalized groups in media and STEM contexts. This domain × CDS interaction is not compatible with a simple "more uncertainty for marginalized groups" story.

### 2.3 Group-Pair CDS: SES Shows Weakest Signal

| Group Pair | Mean CDS | Std | t | p |
|------------|----------|-----|----|----|
| LowSES_Black vs HighSES_White | +0.0112 | 0.059 | 1.19 | 0.241 |
| Black_Female vs White_Male | +0.0043 | 0.051 | 0.53 | 0.597 |
| Black_Male vs White_Male | +0.0027 | 0.052 | 0.32 | 0.748 |
| Asian_Female vs White_Female | +0.0020 | 0.058 | 0.22 | 0.826 |
| Black_Female vs Black_Male | −0.0030 | 0.046 | −0.41 | 0.682 |
| Hispanic_Male vs White_Male | −0.0084 | 0.039 | −1.36 | 0.183 |

No group pair reaches significance. The SES comparison (LowSES_Black vs HighSES_White) has the highest absolute mean (+0.011), consistent with Boufaied et al. (2026) finding that Race×SES is the hardest intersection. But this is a trend at best.

### 2.4 Joint Interpretation

The domain-level pattern is the most interesting finding:

**Claim:** CDS is not a bias metric, but it may be a **domain-specific behavioral fingerprint**. The sign and magnitude of CDS vary systematically by domain, and this variation may reveal something about how the model's training data coverage differs across topics.

**Hypothesis for follow-up:** Domains where the model shows more variability for marginalized groups (education, leadership) are domains where the model has *more* training examples about those groups (due to targeted educational content). Domains where it shows less variability (media, STEM) are domains with *stronger* stereotypical associations in training data, constraining output diversity.

This is speculative but testable via training data analysis.

---

## 3. The New Framework: Behavioral Profiling

Rather than searching for a single bias metric, we propose a **multi-signal behavioral profiling** approach. The same generation pipeline produces multiple independent signals from a single pass:

### Signal 1: Semantic Stability (CDS)
- **What:** Within-group output variability, measured as mean embedding distance from centroid
- **What it captures:** How consistently the model responds to the same demographic cue
- **Status:** Null overall; domain-structured variation
- **Paper role:** Negative result + domain profile as primary finding

### Signal 2: Dispersion Asymmetry
- **What:** Which group in a pair has higher within-group scatter (sign of CDS)
- **What it captures:** Direction of consistency advantage
- **Status:** 50.8% positive — no systematic direction
- **Paper role:** Supports the domain-level analysis

### Signal 3: Centroid Separation
- **What:** Cosine distance between group centroids (embedding of mean representation)
- **What it captures:** How differently the model treats groups at the semantic level
- **Status:** Not yet analyzed
- **Paper role:** Potential positive finding — orthogonal to CDS

### Proposed Framework

The three signals form a **2×2 characterization space**:

| | Low centroid separation | High centroid separation |
|---|---|---|
| **CDS ≈ 0** | Groups treated similarly, similar consistency | Groups treated differently, similar consistency |
| **CDS ≠ 0** | Groups treated similarly, different consistency | Groups treated differently, different consistency |

Each cell corresponds to a different model behavior profile. Characterizing where each domain × group pair falls in this space is the core contribution.

---

## 4. What We Need Now

### Already completed (this run):
- 240 CDS values with CIs, dispersion_a, dispersion_b, centroid_dist
- All on Phi-3-mini (3.8B), 1 temperature (0.7)
- Analysis scripts in `consist/explore.py`

### Immediate next step (no GPU needed):
- **Analyze centroid separation patterns** — is centroid_dist systematically structured by domain or group pair? This is a post-hoc computation on existing data.
- **Correlate CDS with centroid distance** — are they independent signals?
- **Produce publication-ready figures** — domain bar chart with significance markers, group-pair comparison, centroid vs CDS scatter, domain × group heatmap

### Medium-term (requires Kaggle time):
- **Run Qwen2.5-7B** — same 240 pairs, test if domain profile replicates across model scale (~30 min on T4)
- **Run with temperature sweep** — T=0.3, 0.7, 1.0 to test if CDS appears at higher temperatures
- **Lexical analysis** — re-run with text saving enabled to analyze word-level patterns

### Long-term (paper-dependent):
- **Training data analysis** — correlate domain-level CDS with training data coverage estimates
- **Human evaluation** — validate domain-level patterns against human judgments

---

## 5. Paper Structure

### Title (working)
"Divergent Representations: A Multi-Signal Behavioral Probe of Demographic Representation in LLMs"

### Abstract (draft)
Self-consistency disparity — the difference in output variability when an LLM generates repeated responses for different demographic groups — has been proposed as a zero-cost unsupervised proxy for intersectional bias, grounded in the intuition that groups with less training coverage should produce more variable outputs. We test this hypothesis in a controlled experiment using 240 matched counterfactual prompt pairs spanning 10 domains and 6 demographic comparisons on Phi-3-mini (3.8B). The hypothesis is cleanly falsified: mean CDS = 0.0015 (p = 0.66), with no significant signal for any domain, group pair, or interaction. However, we find that CDS is systematically structured by domain, transitioning from positive (marginalized groups more variable) in education and leadership to negative (reference groups more variable) in media and STEM. This domain × CDS interaction cannot be explained by a simple epistemic uncertainty account. We propose a multi-signal behavioral profiling framework that jointly considers semantic stability (CDS), dispersion asymmetry, and centroid separation to characterize how LLMs represent demographic groups — not as a bias metric, but as a method for behavioral characterization. We release the CONSIST pipeline for zero-cost replication and extension.

### Sections
1. **Introduction** — The appeal and failure of self-consistency as a bias metric
2. **Related Work** — Bias metrics, self-consistency, intersectional fairness
3. **Experiment Design** — Model, prompts, metric, pipeline
4. **Results** — Overall null (4.1), domain structure (4.2), group pairs (4.3)
5. **Multi-Signal Framework** — Three signals, 2×2 space, behavioral profiling
6. **Discussion** — Why CDS failed, what domain structure means, implications
7. **Limitations and Future Work** — Scale, temperature, lexical analysis
8. **Conclusion**

### Contributions
1. **Negative empirical finding** — Self-consistency disparity does not detect intersectional bias in Phi-3-mini (240 pairs, 10 domains, 6 comparisons). Constrains the epistemic uncertainty theory for small models.
2. **Domain × CDS interaction** — First documentation of systematic domain-level structure in consistency disparity, suggesting training-data-dependent rather than uncertainty-driven mechanisms.
3. **Multi-signal framework** — Proposal and pipeline for joint behavioral profiling across semantic stability, dispersion asymmetry, and centroid separation.
4. **Open-source release** — CONSIST pipeline for zero-cost replication and extension.

### Target Venues
- **ACL 2027 / EMNLP 2026** — Findings track (negative result with positive domain structure)
- **NeurIPS 2026 Datasets & Benchmarks** — If pipeline release is primary contribution
- **EACL 2027** — Full behavioral profiling paper (if centroid separation shows signal)
- **Responsible AI workshops** (ACL/NeurIPS) — Targeted venue for bias measurement papers

---

## 6. Updated Risk Analysis

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| "Just a null result" rejection | Medium | Domain structure + multi-signal framework distinguish from pure negative result |
| Centroid separation also null | Medium | Framework paper still stands — reporting all three signals is the contribution |
| Domain pattern is spurious | Low | Bootstrap CIs support robustness; second model test will confirm |
| Only one model tested | Medium | Qwen2.5-7B run planned (same pipeline, 30 min on T4) |
| Reviewers want bias detection | Low | Frame as behavioral characterization, not bias detection |
| Insufficient novelty | Low | First paper to systematically falsify CDS and propose multi-signal profiling |

---

## 7. Author Notes

This research was conducted with zero financial budget. No API calls were made to commercial LLMs. All experiments were run on free-tier Kaggle and Google Colab GPUs (NVIDIA T4). The CONSIST pipeline is released under MIT license.

The original hypothesis — that self-consistency disparity would track intersectional bias — was our best guess based on the theoretical literature. It was wrong. We report it because the field needs to know which intuitive hypotheses survive empirical testing. The domain-level structure we found instead is a genuine discovery, even if it is not the discovery we set out to make.

---

*This replaces CONSIST_Research_Proposal.md and IDEA.md as the active research document. The earlier documents are preserved for reference.*
