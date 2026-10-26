# SurgiDoc decisions (consult before every decision)

Generated from the Genesis records in `.genesis/project.json` (the source of truth). Architecture reference: `Architecture Flowchart.png`; requirements: `SPEC.md`; plan: `.genesis/PLAN.md`.

## Binding rules (invariants)

- **INVARIANT-cc1a3020** (2026-10-26): Whenever work needs TigerCloud (or any service credential: Redis, Google AI Studio, Kaggle, Voyage, Cohere, EndoVIT endpoint), STOP and ask the user in chat to fill the matching keys.txt entries (TIGERCLOUD_SERVICE_URL, TIGERCLOUD_DB_PASSWORD, ...). Never invent values; code must raise MissingKey naming the entry.


## Decisions

### Gemini-only LLMs via Google AI Studio
- id `DECISION-91208eb9` · 2026-10-26 · status accepted · source: user 2026-10-26

All agent LLM calls (generator, judge) use Gemini free-tier models via Google AI Studio API. No OpenRouter/GPT/Claude. Embedding: Voyage multimodal + Cohere rerank only if free tier verified at implementation time, else gemini-embedding-001.

### Data: public only, human approval mandatory
- id `DECISION-6fd5ba2e` · 2026-10-26 · status accepted · source: user 2026-10-26

Only public datasets (Kaggle newslab/cholecseg8k via kagglehub, plus public EU/Germany-focused endoscopy surgery text) are used. No real patient data. Every generated report requires human approval; output is a draft, never a diagnosis.

### Scope: all layers, production/enterprise grade, Python
- id `DECISION-0737e188` · 2026-10-26 · status accepted · source: user 2026-10-26

Build every layer in Architecture Flowchart.png (RAG, LangGraph generator/judge, guardrails, gateway, human approval, PDF, TigerCloud+Redis, observability, security/governance). Python, LangGraph, pytest, tests for every component. Secrets in git-ignored keys.txt. DECISIONS.md consulted before every decision.

### EndoViT runs remotely, not on local CPU
- id `DECISION-321ba356` · 2026-10-26 · status accepted · source: user 2026-10-26

User CPU is weak. EndoViT inference must go through a free hosted option (to be found; none confirmed). Frame-description sits behind an interface so a Gemini-vision fallback is possible.

### MIMIC-IV-Note rejected; licence limits on KB sources
- id `DECISION-112f0bfb` · 2026-10-26 · status accepted · source: user+verification 2026-10-26

MIMIC-IV-Note rejected (credentialed DUA, patient notes, US ICU). ESGE KB papers are CC BY-NC(-ND) and CholecT45/CAMMA CC BY-NC-SA: non-commercial prototype only. 4 of 6 ESGE papers are Cascade guidelines for resource-limited settings, not full ESGE guidelines.

### Embeddings: gemini-embedding-001 default, dim 768
- id `DECISION-06c14732` · 2026-10-26 · status accepted · source: docs check 2026-10-26

Docs (ai.google.dev, 2026-10-26): gemini-embedding-2 is multimodal but aggregates a batch into ONE embedding and has no task_type, so text RAG defaults to gemini-embedding-001 (batch, taskType, we L2-normalize truncated dims). Model/dim configurable; revisit embedding-2 for frame/image search. Voyage/Cohere not used until free limits verified. Tiger Cloud: pg_textsearch BM25 (<@>) + pgvectorscale diskann + RRF k=60.

### T03 scope baseline hand-refreshed (user-approved bypass)
- id `DECISION-70581bcb` · 2026-10-26 · status accepted · source: user 2026-10-26

T03 base_manifest predated T02A/T02B (Genesis sets it once via ||=; requeue does not refresh), so completion falsely flagged T02's files as out of scope. At the user's explicit instruction (option 2) the stale base_manifest was removed from project.json. T03's real scope was verified from git: only surgidoc/retrieval and tests/test_retrieval.py.

### T06 baseline cleared before activation (user-approved)
- id `DECISION-6d05bd85` · 2026-10-26 · status accepted · source: user 2026-10-26

T06 had no work when its stale base_manifest (predating T05/T05A) was removed while queued, then resumed to record a fresh baseline. User approved at 2026-10-26 ('clear T06 stale baseline').

### Removed malformed rejected T03A from project.json (user-approved)
- id `DECISION-ba644589` · 2026-10-26 · status accepted · source: user 2026-10-26

T03A was created by mistake with no gate (typo --depend instead of --depends) and rejected; the plan check counts rejected tasks and Genesis has no remove command, so the entry was deleted by hand at the user's choice ('1'). Known: --depend was ignored on all earlier tasks, so T06-T12 have empty dependencies; they run in queue order. User declined the optional dependency back-fill; use --depends from now on.

### T06 baseline cleared again after T03B (user-approved)
- id `DECISION-32c8e2b2` · 2026-10-26 · status accepted · source: user 2026-10-26

T06 had no work; stale base_manifest removed while queued and resumed for a fresh baseline, at the user's instruction 'clear T06 baseline' (2026-10-26).

### Vision: Gemini vision default, EndoViT optional via endpoint contract
- id `DECISION-00353606` · 2026-10-26 · status accepted · source: T06 design 2026-10-26

EndoViT is an MAE-pretrained encoder (feature backbone), not a captioner, and no free hosted EndoViT endpoint was found. FrameDescriber interface: EndoViTClient (ENDOVIT_INFERENCE_URL/TOKEN, contract POST {image_b64,mime} -> {findings,labels}) is tried first when configured; Gemini vision (gateway role 'vision', batched frames per request to save free-tier quota) is the working default and fallback. Frame extraction via opencv-python-headless (no ffmpeg here).

### T07 agent design: cited statements, fail-closed judge, bounded loop
- id `DECISION-3a48b78f` · 2026-10-26 · status accepted · source: T07 design 2026-10-26

LangGraph StateGraph retrieve->generate->(citation check)->judge->{approved|regenerate|stop}. Generator returns JSON statements each citing source ids (F<n> frame findings, K<n> KB chunks); uncited or unknown-id statements are rejected deterministically before the judge call (saves free-tier quota, enforces NFR-3). Judge (different model than generator) returns {approved, issues, feedback}; unparseable judge output = NOT approved (fail closed). Max attempts bounded (default 3); result status approved|rejected|blocked; always labelled draft. KB text and findings are wrapped as data in tags; system prompt says they are not instructions (prompt injection).

### T09: approver self-named, fpdf2 core fonts (latin-1), Postgres tables with hypertable events
- id `DECISION-aa097c87` · 2026-10-26 · status accepted · source: user + T09 design 2026-10-26

Approver types their own name in the name field (user is the approver); recorded as given, no authentication until T10 RBAC. Approval binds to a sha256 content hash; edits (re-checked by guardrails) reset approval; edit after approval is refused; final PDF only from an approved, hash-verified item (AC-3). PDF via fpdf2 core Helvetica: characters outside latin-1 are replaced with '?' (known ceiling, upgrade = bundle a TTF). Storage: <prefix>_reports (jsonb + bytea pdf), <prefix>_events (hypertable on at), <prefix>_rollups; parameterized SQL; prefix whitelisted.

### T10 security design: keys.txt user registry, RBAC with separation of duties, hash-chained audit, AES-GCM primitives
- id `DECISION-927acd26` · 2026-10-26 · status accepted · source: user + T10 design 2026-10-26

Users live in keys.txt as SURGIDOC_USERS=Name:role+role;Name2:role (git-ignored, no new files). Roles: reviewer (view/edit/approve/reject), operator (run pipeline/view), admin (audit view/users manage/view) - admin and operator cannot approve (separation of duties). GuardedReview binds approver name to the authenticated user and audits every allow/deny; denials are audited (AC-7). Audit log: append-only, sha256 hash-chained, redacts secret-like keys, verify() + external head anchor detects truncation; Tiger Cloud sink uses advisory lock + DB triggers forbidding UPDATE/DELETE/TRUNCATE. Crypto: AES-256-GCM, versioned envelope with key fingerprint, AAD binds ciphertext to record id, optional previous key for rotation. KNOWN GAP: ReportStore (T09, closed) stores plaintext jsonb/pdf; applying encryption to stored columns needs a later task (plan reopen). No real authentication (no passwords/login) - registry is name->roles only.

### Accepted risk: Redis without TLS (free tier)
- id `DECISION-b7c5513a` · 2026-10-26 · status accepted · source: user 2026-10-26

User's free Redis account offers no rediss:// TLS endpoint; the project uses the plain redis:// URL (user decision 2026-10-26). Consequences: the Redis password and cached values cross the network unencrypted, so use only on trusted networks. Mitigations already in place: cache keys are SHA-256 hashes, only public-data values are cached, cache fails open. compliance check 'redis-tls' will therefore report FAIL; this is an accepted, documented exception, to be revisited if a TLS endpoint or paid plan becomes available. Do not cache anything other than public data.


## Open assumptions

- **ASSUMPTION-834e5ce2** [open] Voyage multimodal-3.x free allowance (200M text tokens) is from a third-party page and free trial is 3 RPM / 10K TPM without payment method; Cohere trial limits unverified. Verify before committing.

- **ASSUMPTION-c96a274b** [open] Gemini free tier may use content for product improvement; acceptable only because data is public.


## Knowledge / findings

- **gh CLI missing** (`KNOWLEDGE-65a43d10`, 2026-10-26): gh is not installed in this WSL environment; GitHub repo ChandrikaSethumadhavan/surgidoc creation is user-authorized but blocked until gh is installed and authenticated.

- **Candidate public sources for KB/datasets** (`KNOWLEDGE-7f2fe807`, 2026-10-26): Images/video (not report text): HyperKvasir (datasets.simula.no/hyper-kvasir, CC-BY 4.0, 58.6GB full; labeled images 3.9GB, segmentation 46MB). Text KB (EU/DE): AWMF/DGVS S3 sedation guideline 021-014 (register.awmf.org PDF, 2023), ESGE guidelines in Endoscopy (Thieme, many open access). Unverified: Kvasir, Cholec80, EndoVis, Kaggle mirrors, licences of each PDF.

- **Gemini free-tier models and limits** (`KNOWLEDGE-bfc35385`, 2026-10-26): ai.google.dev pricing (2026-10-26): free of charge input/output for gemini-2.5-flash, 2.5-flash-lite, 3.6-flash, 3.5-flash-lite, 3.1-flash-lite. Per-model RPM/TPM/RPD are NOT published on the rate-limits page (view in AI Studio); limits are per project, RPD resets midnight Pacific, 429 RESOURCE_EXHAUSTED on breach. Free tier content may be used to improve products: public data only. Gateway: configurable routes+limits, conservative defaults, 429 is ground truth.

- **Live check 2026-10-26: Gemini 2.5 models 404; working models** (`KNOWLEDGE-412774e2`, 2026-10-26): With the user's key: gemini-2.5-flash and gemini-2.5-flash-lite return 404 'no longer available to new users' (still listed by models.list). Working generateContent: gemini-3.6-flash, gemini-3.5-flash-lite, gemini-3.1-flash-lite. gemini-embedding-001 works (768 dims). Redis works. Tiger Cloud: PostgreSQL 18.6, vectorscale installed, pg_textsearch available but not yet installed (PgStore.setup creates it). gateway.py ROUTES defaults are therefore stale and must be updated.

- **Live check: pg_textsearch needs to_bm25query** (`KNOWLEDGE-0e1b159c`, 2026-10-26): On Tiger Cloud (PG18, pg_textsearch), 'ORDER BY content <@> %s' with a bound parameter fails: 'no BM25 index found for text <@> text expression'. Working form: ORDER BY content <@> to_bm25query(%s, 'chunks_bm25_idx'). PgStore.setup/add/vector verified live; chunk ids currently embed absolute file paths (ingest source) and should be made relative.

- **Follow-up: PDF Sources section lists uncited sources** (`KNOWLEDGE-bad7881c`, 2026-10-26): pdf.py (T09) prints every source in the report, including frame findings no statement cites any more. When a clinician edits out a statement, its raw finding text still appears under Sources in the approved PDF. Fix in a follow-up task (T13): list only cited sources in the PDF (content hash still covers all). Also in T13: pyproject.toml dependency list; encryption of stored report columns; Redis TLS accepted-risk exception for compliance; real evaluation set.


## Open items (human to-do)
- Add Kaggle credentials (`KAGGLE_USERNAME`, `KAGGLE_KEY`) to `keys.txt` to run real endoscopy frames and build a real evaluation set.
- Provide a German/EU guideline text source (AWMF links tried were unreachable or unidentified).
- Redis TLS: accepted risk while on the free plan.
- T13 (in planning): dependency list, PDF cited-sources fix, encryption at rest.
