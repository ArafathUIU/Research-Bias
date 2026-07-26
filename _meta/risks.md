# Risks — Divergent Representations

## Active Risks

1. **Domain pattern is spurious** — 3/10 domains significant at α=0.05, no multiple comparison correction. Could be noise.
   - *Mitigation:* Bootstrap CIs already computed. Second model (Qwen2.5-7B) will test replicability.

2. **Centroid separation also null** — If centroid distance shows no structure either, the entire positive finding rests on domain-level CDS (weak signal).
   - *Mitigation:* Framework paper still stands — reporting all three signals is itself a contribution, even if all are null.

3. **Reviewer skepticism about negative results** — Some venues de-prioritize null findings.
   - *Mitigation:* Target Findings track or workshops. Domain structure provides positive content beyond the null.

4. **Only one model at one temperature** — Results may not generalize.
   - *Mitigation:* Qwen2.5-7B replication planned. Temperature sweep deferred but documented as limitation.

5. **No raw text saved** — Lexical analysis requires re-running generation with text saving.
   - *Mitigation:* Kaggle re-run with modified code. Low cost (~30 min).

## Retired Risks

- ~~CDS fails as a bias metric~~ → Confirmed. Repurposed as behavioral profiling signal.
- ~~$16K budget too high~~ → Solved: entire pipeline is $0.
- ~~Gated model access~~ → Solved: switched to Phi-3-mini (MIT license).
