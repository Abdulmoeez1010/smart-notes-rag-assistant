# Smart Notes RAG Assistant — Architecture Reference

A 5-minute read to refresh your memory before a demo, interview, or just
picking this project back up. Read this instead of re-deriving everything
from scratch.

---

## 1. The one-sentence pitch

An AI study assistant that turns YouTube videos, PDFs, and slide decks into
a chat-based Q&A tool, plus auto-generated summaries, quizzes, and mindmaps —
built on a custom RAG pipeline, FastAPI backend, and React frontend.

---

## 2. The full request flow, end to end

```
User action (click/type in React)
   → React component calls a function in api.js
      → axios sends an HTTP request to FastAPI
         → FastAPI validates the request against a Pydantic schema
            → the route calls ONE orchestrator function
               → the orchestrator calls pipeline functions in order
                  → pipeline functions call Gemini / FAISS / embeddings
               ← orchestrator returns a plain Python dict
            ← FastAPI validates the response against a Pydantic schema, returns JSON
         ← axios receives the JSON response
      ← React updates state (setState) with the response data
   → React re-renders the UI with the new data
```

Every layer only talks to the layer directly next to it. React never knows
about FAISS. The pipeline files never know FastAPI exists. That separation
is why adding PDF/PPTX support only meant new *loader* files — nothing else
had to change.

---

## 3. What each file actually does

### Backend

| File | Job |
|---|---|
| `main.py` | Defines the 7 HTTP routes. Validates input via schemas, calls exactly one orchestrator function, catches errors into clean JSON responses. No AI/pipeline logic lives here. |
| `schemas.py` | Pydantic models — the exact JSON shape every endpoint accepts and returns. This is the contract the frontend's `api.js` is written against. |
| `orchestrator.py` | The conductor. One function per task (`ask_question`, `summarize_doc`, `quiz_doc`, `mindmap_doc`, three `ingest_*` functions). Strings pipeline steps together in the right order. Never imports FastAPI. |
| `ingestion/youtube_loader.py`, `pdf_loader.py`, `pptx_loader.py` | Source-specific: pull raw text out of a YouTube transcript, PDF, or PPTX, and generate a `doc_id` (video ID for YouTube, content hash for files) plus a human-readable title. |
| `pipeline/splitter.py` | Breaks raw text into ~1000-character chunks with overlap, so each chunk is small enough to embed and retrieve individually. |
| `pipeline/embeddings.py` | Wraps Gemini's embedding model — separate embedders for documents vs. queries (different `task_type`, improves retrieval quality). |
| `pipeline/vectorstore.py` | Builds/loads a FAISS index per `doc_id`, saved to `data/vectorstores/{doc_id}/`. This folder's existence IS the cache. |
| `pipeline/retriever.py` | The 3-phase retrieval system — see section 4 below. |
| `pipeline/generation.py` | Takes compressed context + question, runs it through an LCEL chain (`prompt | llm | parser`), returns the final answer string. |
| `pipeline/summarization.py`, `quiz.py`, `mindmap.py` | Each takes ALL chunks (no retrieval) and one task-specific prompt, one Gemini call each. |

### Frontend

| File | Job |
|---|---|
| `api.js` | Every backend call in one place. Each function matches one FastAPI route and its expected request/response shape exactly. |
| `App.jsx` | Owns the shared state: `docId`, `docTitle`, `activeTab`. Decides whether to show the input screen or the 4-tab workspace. |
| `DocumentInput.jsx` | The ingestion screen — 3 modes (YouTube/PDF/PPTX), calls the right `ingest*` function, reports the resulting `doc_id` back up to `App`. |
| `ChatView.jsx` | Owns its own `messages` state (chat history lives only in browser memory — refresh clears it). Calls `askQuestion` per message. |
| `SummaryView.jsx`, `QuizView.jsx`, `MindmapView.jsx` | Each auto-fetches its data via `useEffect` the moment the tab is opened, using the shared `docId` prop. |

---

## 4. The retrieval pipeline, in detail (this is the technical heart of the project)

**Pre-retrieval — multi-query generation** (1 Gemini call)
The user's exact wording might not match how the source material phrases the
same idea — especially true for Hindi/English code-switched transcripts.
Generate 2 alternate phrasings of the question, search with all 3 versions.

**During-retrieval — hybrid search + MMR** (free, local computation)
- **BM25** (keyword search) catches exact technical terms that semantic
  search might underweight.
- **Semantic search with MMR** (Maximal Marginal Relevance) balances
  relevance against diversity — so you don't get 3 near-duplicate chunks
  repeating the same sentence.
- **EnsembleRetriever** merges both, weighted 60% semantic / 40% keyword.

**Post-retrieval — contextual compression** (embedding calls only, not
generation calls)
Re-embeds each retrieved chunk against the question, drops anything below a
similarity threshold (0.60, empirically tuned — not guessed). **Known
limitation we fixed:** a fixed threshold doesn't generalize to every
document's embedding distribution. Fix: if compression filters everything
to zero, fall back to the top-3 uncompressed chunks instead of returning
empty context.

**Why this design, cost-wise:** the free tier is 20 Gemini generation
calls/day. Only 2 of those are spent per question (1 query-variation call +
1 final-answer call) — everything else (BM25, MMR, embeddings-based
compression) is free/local computation on a separate, much higher quota.

---

## 5. The three architectural decisions worth explaining well

**1. Retrieval-based Q&A vs. whole-document summarization are different tasks.**
Retrieval finds the top-k chunks *relevant to a specific question* — it
can't answer "summarize this" well, because that question doesn't point at
any specific slice of the document. Discovered this directly: asking for a
summary through the `/ask` retrieval path returned "no context found," even
though chunks existed — because the vague question didn't match anything
strongly, and compression filtered the weak matches to zero. Fix:
`summarize/quiz/mindmap` use ALL chunks, no retrieval, one task-specific
prompt each — a completely separate code path built for a different job.

**2. Unified `doc_id` contract across all sources.**
YouTube, PDF, and PPTX all end in the exact same shape: a `doc_id` string
that's also the vectorstore's folder name on disk. Every downstream feature
(`ask/doc`, `summarize/doc`, `quiz/doc`, `mindmap/doc`) only needs a
`doc_id` — it never needs to know or care what source produced it. This is
why adding a new source type never requires touching retrieval, generation,
summarization, or the frontend's 4 view components — only a new loader file
and one new orchestrator function.

**3. Thin routes, fat orchestrator, dumb pipeline files.**
`main.py` only does HTTP plumbing. `orchestrator.py` sequences steps.
Pipeline files (`retriever.py`, `generation.py`, etc.) know nothing about
FastAPI or HTTP at all — they just take `Document` objects and questions.
This means the whole pipeline could be reused in a completely different
context (a CLI tool, a notebook) with zero changes.

---

## 6. Known limitations (say these proactively, don't wait to be asked)

- **YouTube ingestion only works locally**, not on the deployed Render
  backend — YouTube blocks transcript requests from cloud-hosting IP ranges
  as an anti-bot measure. Real production constraint, not a code bug.
- **No timestamps in answers** — the transcript is stored as one flat text
  block, not time-coded segments, so "jump to this part" isn't possible yet.
- **Compression threshold (0.60) was tuned on one video's data** — the
  fallback (section 4) mitigates but doesn't fully solve this; a more
  robust version would use a relative threshold instead of an absolute one.
- **No database** — state is just FAISS folders on disk. No user accounts,
  no persisted chat history, no multi-user support.
- **No automated tests** — everything was verified manually through
  Swagger UI during development.

---

## 7. The one-paragraph "tell me about this project" answer

"I built a RAG-based study assistant that ingests YouTube videos, PDFs, and
slide decks, then supports Q&A, summarization, quiz generation, and mindmap
generation — all behind a FastAPI backend and React frontend, both
deployed. The retrieval pipeline uses multi-query generation, hybrid
BM25/semantic search with MMR, and embedding-based contextual compression,
tuned to stay within a free-tier LLM quota. The most interesting design
decision was realizing that retrieval-based Q&A and whole-document
summarization are fundamentally different tasks — I discovered this by
debugging a real failure case, not by reading about it — so I built them as
separate code paths sharing the same underlying document store instead of
forcing everything through one retrieval mechanism."
