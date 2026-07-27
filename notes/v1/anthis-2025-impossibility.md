# Source: Anthis et al. 2025 — The Impossibility of Fair LLMs
**URL:** https://aclanthology.org/2025.acl-long.5/
**Lens:** skeptic
**Pre-read:** I'm reading this to understand the strongest argument against the proposal's premise.

## What it says
Technical fairness frameworks (group fairness, individual fairness, fair representations) do not logically extend to general-purpose LLMs. Reason: combinatorial explosion of populations × use cases × attributes; fairness does not compose across system components; unstructured training data lacks identifiable target populations. Concludes fair LLMs are intractable, but suggests context-specific evaluation as a viable path.

## My reaction
This is the most important counterargument the proposal must address head-on. The proposal already partially aligns with Anthis's recommended path (context-specific probes, deployable artifact). But the proposal needs to explicitly acknowledge the intractability claim and argue why the SES×dialect×tone probe is *sufficiently context-specific* to escape it. The key move: this isn't a claim about "fair LLMs" generally — it's a specific, bounded audit tool.

## Contradictions
Directly contradicts the premise of most fairness benchmarks (BBQ, WinoIdentity, etc.) that attempt general-purpose fairness measurement. However, Anthis et al. explicitly endorse "context-specific evaluations" — which is exactly what the proposal's CLI artifact would be.

## Question raised
What specific examples of "context-specific evaluation" does Anthis consider viable? The paper mentions this as a positive direction but doesn't elaborate on methodological requirements.
