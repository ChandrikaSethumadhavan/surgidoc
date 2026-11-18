# SurgiDoc

Endoscopy documentation assistant: a surgical video goes in, a cited draft report comes out, and a human approves it before a PDF exists.
Frames are described by a vision model, guideline passages are retrieved with hybrid search (BM25 + vector, RRF, rerank), and a LangGraph
generator and judge write and check the draft. Every statement cites the frame or guideline it came from. The output is a draft, never a diagnosis.

Public data only, non-commercial prototype (see DECISIONS.md for licences and known gaps). It runs on loopback with no login.

## Run it

```bash
pip install -e ".[dev]"
(cd frontend && npm ci && npm run build)   # the React page, served by the Python server
python -m surgidoc.webui                   # needs keys.txt (git-ignored); open http://127.0.0.1:8765
```

Live UI development: run `python -m surgidoc.webui`, then `cd frontend && npm run dev`. Vite proxies the API to port 8765.

## Test it

```bash
python -m pytest -q                          # offline; live checks only run with SURGIDOC_LIVE=1 and real keys
cd frontend && npm run typecheck && npm test -- --run && npm run build
```

## CI and releases

- `.github/workflows/ci.yml` runs on pushes to `main` and on pull requests: the Python suite (3.12, no secrets) and the frontend typecheck, tests and build.
- `.github/workflows/release.yml` runs on `v*` tags: CI first, then a GitHub release with the built UI attached.
- Make `CI` a required status check on `main` in the repository settings so nothing merges red. There is no hosted deployment: the app is
  loopback-only and has no login, so only the static UI bundle is published.
