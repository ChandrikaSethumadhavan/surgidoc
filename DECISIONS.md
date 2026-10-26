# SurgiDoc decisions (consult before every decision)

Source of truth for architecture: `Architecture Flowchart.png`. Durable records also live in `.genesis/`.

| Date | Decision | Why |
|---|---|---|
| 2026-10-26 | Python + LangGraph + pytest; every component has tests | user |
| 2026-10-26 | LLMs: Gemini free tier via Google AI Studio only (no OpenRouter) | user; free |
| 2026-10-26 | Embeddings: Voyage multimodal + Cohere rerank if free, else gemini-embedding-001 | user; free tiers unverified (see open items) |
| 2026-10-26 | EndoViT via free hosted inference, not local CPU; interface allows Gemini-vision fallback | weak local CPU |
| 2026-10-26 | Data: Kaggle `newslab/cholecseg8k` (kagglehub) + public EU/Germany endoscopy text for the KB; no patient data | user |
| 2026-10-26 | All reports are drafts needing human approval; no diagnostic claims | safety |
| 2026-10-26 | Secrets only in git-ignored `keys.txt` | security |
| 2026-10-26 | Scope: all diagram layers, production grade | user |

## Open items
- Hosted EndoViT option not found yet (research needed).
- Verify Voyage multimodal and Cohere free limits (Voyage trial reportedly 3 RPM / 10K TPM).
- `gh` CLI not installed; repo ChandrikaSethumadhavan/surgidoc not created yet (visibility: private unless told otherwise).
- Gemini free tier may use inputs for product improvement; OK only for public data.
