# ATS Backend — FastAPI + RAG + LLM + ChromaDB (only database)

An applicant-tracking system where **ChromaDB is the only datastore** — no
Postgres, MySQL, Mongo, or SQLite. Structured resume/job fields live in
ChromaDB's `metadata`; a single embedding (no chunking) is stored per
resume and per job description.

## Connecting to the database

ChromaDB isn't a server you connect to with a host/user/password like Postgres —
by default it's just a local folder. Two modes, controlled by `CHROMA_MODE` in `.env`:

**`embedded` (default)** — `chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)` writes
straight to `chroma_data/` in the same process. No server, no setup. Fine for one
backend instance (internship/demo scale).

```bash
CHROMA_MODE=embedded
CHROMA_PERSIST_DIR=./chroma_data
```

**`server`** — a standalone ChromaDB server that multiple FastAPI replicas can share
(needed once you scale horizontally, per the design doc's scalability section).

```bash
# run a chroma server (or use docker compose, see below)
docker run -p 8001:8000 -v chroma_data:/chroma/chroma chromadb/chroma:0.5.5
```
```bash
# .env
CHROMA_MODE=server
CHROMA_HOST=localhost
CHROMA_PORT=8001
```
`chroma_service.py` picks `HttpClient` vs `PersistentClient` automatically based on
`CHROMA_MODE` — nothing else in the codebase needs to change either way.

With `docker compose up --build`, a `chromadb` service is already included and wired
up for you — the backend connects to it over the Docker network at `chromadb:8000`.

**Verify the connection:** `GET /health` now actually queries ChromaDB and reports
`chroma_status`, `jobs_stored`, and `resumes_stored` — not just "the API process is up."

```bash
curl http://localhost:8000/health
```

## Quick start

```bash
cd ats_backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and set GEMINI_API_KEY (free tier: https://aistudio.google.com/apikey) and ATS_API_KEY

uvicorn app.main:app --reload
```

Open **http://localhost:8000** for the built-in frontend (`frontend/static/index.html`),
or **http://localhost:8000/docs** for interactive API docs.

### Docker

```bash
docker compose up --build
```

## What's inside

```
app/
├── main.py                  FastAPI app, CORS, request-id logging middleware
├── config.py                All settings, via .env
├── api/                     routes_jobs.py, routes_resumes.py, routes_search.py
├── services/
│   ├── extraction_service.py   ← the hybrid experience pipeline (see below)
│   ├── embedding_service.py    single embedding per document, local model
│   ├── llm_service.py          Claude calls: structuring + scoring, never math
│   ├── chroma_service.py       the only database — all reads/writes
│   └── rag_service.py          top-K retrieval → scoped LLM scoring → rank
├── models/schemas.py        Pydantic request/response models
├── core/                    logging, custom exceptions, API-key auth
└── utils/                   PDF/DOCX text extraction, upload validation
frontend/static/index.html   plain HTML/JS UI (also see ats_prototype.jsx separately)
tests/                       pytest — extraction pipeline + health check
```

## Auto-detection features

**Paste a rough job description.** `POST /jobs/parse` takes unstructured JD
text (responsibilities, skills, experience, education mixed in any order) and
returns suggested `job_title`, `required_skills`, `minimum_experience`,
`education_requirement` for HR to review and edit before saving with the
normal `POST /jobs`. Nothing is auto-saved.

**Job-title / profession mismatch.** Every resume gets a `detected_job_title`
— the LLM's read of what profession this candidate actually is (e.g. "AI
Engineer"), independent of whatever job they applied to. `check_role_alignment()`
in `extraction_service.py` does a simple, deterministic word-overlap check
against the posted role's title and flags `"different_field"` when there's no
overlap at all (e.g. applying an "AI Engineer" resume against an
"Administrative Assistant" posting) — surfaced as a ⚠ badge in the UI and
folded into the search result's `explanation`.

**Education — every qualification found, only the HIGHEST one compared.**
A candidate may list several qualifications (Matric, Intermediate, Bachelor's,
PhD...). The LLM extracts `education_records` - every one it finds, classified
into one of nine ordered buckets:
`No formal education stated` → `Certificate/Diploma (non-academic)` →
`Matric / Secondary School` → `Intermediate / High School` →
`Diploma (technical/associate)` → `Bachelor's` → `Master's` → `MPhil` →
`PhD / Doctorate`. Python then picks the single highest one
(`extraction_service.highest_education()`, order-independent - it doesn't
matter what order the LLM lists them in) and that's the ONLY one compared
against a job's requirement, via a strict index comparison
(`education_meets_requirement()`) - never an LLM judgment call, same
principle as the experience math. A candidate with only a bootcamp
certificate (no degree) is labeled `Certificate/Diploma (non-academic)`
rather than being mis-classified as, or bluntly rejected against, a degree
requirement.

**Skills — explicit vs. semantically inferred, confidence-scored.**
`skills_explicit` are skills literally written in the resume.
`skills_inferred` are skills NOT literally written but strongly implied by
the candidate's actual projects/experience (e.g. someone who used
Scikit-learn, TensorFlow, and Kaggle almost certainly knows Python and data
preprocessing, even if "Python" isn't listed) - each inferred skill carries a
`confidence` (0.0-1.0) and `evidence` string, and the LLM is explicitly told
not to hallucinate: if it's not confident, it leaves the skill out.

**Skill matching — semantic, not literal keyword matching.** During
`/search` scoring, a required skill counts as matched when the candidate's
skills/projects/experience strongly imply it, even if phrased differently -
e.g. a job wanting "REST APIs" matches a candidate who "built backend
services using FastAPI". Every match is tagged `"exact"` (literally stated)
or `"related"` (semantic equivalence) with the evidence that supports it, so
nothing is a black-box score.

## Experience extraction — hybrid, not LLM guesswork

`total_experience` is never asked of the LLM as a number. `extraction_service.py`
follows two rules, exactly:

**Rule 1 - explicit total stated anywhere.** A regex checks the FULL resume
text (summary, header, anywhere - not one section) for a claim like *"15
Years of Experience"*, *"Over 10 Years Experience"*, *"8+ Years Experience"*,
or *"Professional Experience: 12 Years"*. If found, that's the number - date
math is skipped entirely.

**Rule 2 - otherwise, calculate from employment history.** The LLM reads the
**entire resume** (Work Experience, Employment History, Internships,
Freelance Work, Research - wherever dated roles appear, not just one
heading-sliced section, per the deep-parsing requirement) and structures
every role into `{company, title, start_date_raw, end_date_raw, type}` -
structuring only, **no duration math**. Python (`dateutil`) then parses
whatever date format shows up, treats `Present` / `Current` / `Till Date` /
`To Date` as ongoing, **merges overlapping ranges** so concurrent or
back-to-back roles aren't double-counted, and sums real month/year durations
into one final `total_experience`.

This keeps the number deterministic and auditable — every candidate row
tells you whether experience was `stated` or `computed`, and the underlying
per-company breakdown (`work_history`) travels with it.

## API summary

| Method | Endpoint | Auth | Purpose |
|---|---|---|---|
| POST | `/jobs` | API key | Create job description (auto-embeds) |
| POST | `/jobs/parse` | API key | Auto-detect skills/experience/education from a pasted rough JD (review only, doesn't save) |
| GET | `/jobs` | API key | List jobs |
| GET | `/jobs/{id}` | API key | Get one job |
| PUT | `/jobs/{id}` | API key | Update job (re-embeds) |
| DELETE | `/jobs/{id}` | API key | Delete job |
| POST | `/resumes/upload` | — | **Step 1**: save files only, no parsing |
| GET | `/resumes/pending` | — | List saved-but-unscreened files |
| DELETE | `/resumes/pending/{file_id}` | — | Remove a pending file before screening |
| POST | `/resumes/screen` | API key | **Step 2**: run extraction + embed + store |
| GET | `/resumes` | — | List screened candidates |
| GET | `/resumes/{id}` | — | Get one candidate |
| DELETE | `/resumes/{id}` | API key | Delete one candidate |
| DELETE | `/resumes` | API key | **Delete all** candidates |
| POST | `/search` | API key | RAG search: query → top-K → LLM score → ranked list |
| GET | `/health` | — | Liveness check |

Send the API key as header `X-API-Key: <ATS_API_KEY from .env>`.

## Design notes

- **No chunking anywhere.** One embedding per full resume, one per full job
  description.
- **RAG scope is narrow by design.** `/search` only ever sends the LLM the
  top-K retrieved candidates' metadata + resume text + the one target job
  description — never the whole collection.
- **Embeddings are local by default** (`sentence-transformers`, no external
  key required) so the project runs out of the box. Swap
  `embedding_service.py` for a hosted provider (e.g. Voyage AI) if you want
  one — nothing else in the codebase needs to change.
- Full architecture write-up (diagrams, deployment, scaling, security) is in
  `ATS_System_Design.md` alongside this project.

## Tests

```bash
pytest
```

`tests/test_extraction_service.py` exercises the experience pipeline's pure
functions (regex + date math) without needing an LLM or network call.
