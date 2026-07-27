## 2026-07-26 16:10 — Gap Phase

**Current phase:** gap
**Skill invoked:** gfy-research--gap (cycle 1: full)

**Gaps identified for the comparative evaluation direction:**

1. **BSM (Jeong 2025) is directly comparable** — Their paper already compares 30 LLMs on bias. Our differentiation: (a) unsupervised metric vs labeled benchmarks, (b) semantic dispersion vs classification accuracy, (c) open-source models only vs mixed open/proprietary. Risk: reviewers may ask "why not just use BSM's framework?" Mitigation: our metric captures semantic dispersion, an orthogonal signal that BSM does not measure.

2. **Only 1 model tested** — The entire comparative study depends on running 4+ models. Risk: if all models show null CDS, the paper's positive finding rests entirely on domain-level structure (weak). Mitigation: if null across all models, frame as "consistent null finding across scales and families" — a different kind of contribution.

3. **Model selection** — Which models? Proposed set: Phi-3-mini (3.8B, done), Qwen2.5-7B (diff family, larger), Mistral-7B (diff architecture), Llama-3.2-3B (same scale, diff family), Gemma-2-9B (diff family, largest). All fit on T4/4-bit.

4. **Metric scope** — Just CDS, or also centroid distance and dispersion asymmetry? Both should be reported. The multi-signal framework is the real contribution.

5. **Computational budget** — ~30 min per model on T4. 5 models = 2.5 hours. Kaggle weekly limit = 30 hours. Feasible.

**Next action:** Synthesize — write the comparative evaluation proposal document, then present to user for go/no-go on running the model suite.
