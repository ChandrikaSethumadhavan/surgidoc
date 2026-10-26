# Product specification — surgidoc

> Status: draft. A coding agent must not implement product code until this specification is approved through Genesis.

## Problem

Endoscopists spend significant time writing procedure documentation. SurgiDoc turns an endoscopy/surgical video into a source-referenced draft report that a clinician reviews, edits and approves before a PDF is issued. See `Architecture Flowchart.png`.

## Users

- Endoscopist (target end user): reviews, edits, approves reports.
- Developer (current sole tester): operates and evaluates the system.
- Affected: patients (indirectly); no real patient data is used in this release.

## Functional requirements

- FR-1: Accept a video upload and extract frames.
- FR-2: Describe frames as text findings via EndoViT served from a free hosted endpoint (not local CPU), behind a replaceable interface.
- FR-3: Ingest public documents (PDF, images, video transcripts) with text extraction, OCR and semantic chunking into a knowledge base covering EU/Germany-focused endoscopy surgery text, plus Kaggle `newslab/cholecseg8k`.
- FR-4: Embed chunks and store them in TigerCloud; retrieve with hybrid search (BM25 + ANN), RRF fusion to top 30, rerank to top 5.
- FR-5: A LangGraph Generator agent drafts a report from findings and retrieved context, citing sources.
- FR-6: A Judge agent validates the draft against retrieved context and approves or rejects; rejection loops back to regeneration up to a bounded retry count.
- FR-7: Guardrails enforce Pydantic schema validation, content-safety policy and output validation before human review.
- FR-8: All LLM calls go through a gateway applying routing and rate limits to Gemini models on Google AI Studio.
- FR-9: A human reviewer can view, edit, approve or reject the validated report; nothing is finalized without approval.
- FR-10: Approved reports are rendered to PDF with sources and saved to TigerCloud, along with events and rollups.
- FR-11: Redis caches embeddings, queries and session state.
- FR-12: OpenTelemetry traces and metrics, a real-time dashboard (usage, latency, quality) and an evaluation suite (accuracy, relevance, quality).
- FR-13: Security and governance: RBAC, audit logs, encryption of data, policy enforcement, compliance checks.
- FR-14: Secrets are read only from git-ignored `keys.txt`.

## Non-functional requirements

- NFR-1: Every component has automated tests; the full suite runs with one command and passes offline using mocks.
- NFR-2: No secret appears in git history, logs or traces.
- NFR-3: Every report claim is traceable to a retrieved source; unsupported claims are rejected by the Judge.
- NFR-4: Free-tier rate limits are respected via backoff and caching; no paid service is required.
- NFR-5: The system never makes diagnostic claims; every report is labelled draft until approved.
- NFR-6: Audit log entries are append-only and record who approved what and when.

## Constraints

- Python, LangGraph, pytest. Gemini via Google AI Studio only. Free tiers only.
- Public data only; no patient data.
- Weak local CPU: heavy model inference must be remote.
- Prototype data from Kaggle `newslab/cholecseg8k` (segmentation images, not report text).

## Non-goals

- Clinical certification, use with real patient data, or autonomous report release.
- Paid LLM providers, OpenRouter, GPT or Claude models.
- Access to the B. Braun internal knowledge base (public stand-in used).

## Acceptance criteria

- AC-1: Given a sample video, the pipeline produces a draft with cited sources and a Judge verdict, using mocked services in tests.
- AC-2: A rejected draft triggers regeneration and stops after the retry limit.
- AC-3: Report with no human approval cannot produce a final PDF.
- AC-4: Schema-invalid or policy-violating output is blocked by guardrails.
- AC-5: Repeating a query is served from Redis cache.
- AC-6: Retrieval returns top 5 chunks after hybrid search and rerank on a fixture corpus.
- AC-7: An unauthorized role is denied approval and the denial is audited.
- AC-8: A secret-scan test finds no key material in tracked files.
- AC-9: Full test suite passes with one command.

## Risks

- Hosted EndoViT may not exist for free: mitigate with Gemini-vision fallback behind the same interface.
- Free-tier limits (Voyage reportedly 3 RPM / 10K TPM without payment): mitigate with caching, batching, gemini-embedding-001 fallback.
- cholecseg8k has no report text, so report quality needs a separate evaluation set.
- Free Gemini tier may use inputs for product improvement: acceptable only for public data.
- Hallucination in medical text: Judge, guardrails and mandatory human approval.

## Open questions

- Which hosted service will serve EndoViT?
- GitHub repo visibility (private assumed) and installing `gh`.
- Which specific public EU/Germany endoscopy text sources to use.
