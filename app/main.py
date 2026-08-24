import tempfile
import os
from fastapi import FastAPI, HTTPException, UploadFile, File

from app.pipeline.orchestrator import (
    ingest_youtube, ingest_pdf, ingest_pptx,
    ask_question, summarize_doc, quiz_doc, mindmap_doc,
)
from app.schemas import (
    IngestYouTubeRequest, IngestResponse,
    AskDocRequest, AskResponse,
    SummarizeDocRequest, SummarizeDocResponse,
    QuizDocRequest, QuizDocResponse,
    MindmapDocRequest, MindmapDocResponse,
)

app = FastAPI(title="Smart Notes Rag Assistant")

# adding CORS middleware
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://smart-notes-frontend-black.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── INGESTION ENDPOINTS ──────────────────────────────────────────
# Each of these takes a source (URL or uploaded file), runs it through
# loading -> splitting -> embedding -> vectorstore build/cache, and
# returns a doc_id. Call once per document; reuse the doc_id after.

@app.post("/ingest/youtube", response_model=IngestResponse)
def ingest_youtube_endpoint(request: IngestYouTubeRequest):
    try:
        doc_id, title = ingest_youtube(request.video_url)
        return {"doc_id": doc_id, "title": title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/pdf", response_model=IngestResponse)
def ingest_pdf_endpoint(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name
        doc_id, title = ingest_pdf(tmp_path, file.filename)
        os.unlink(tmp_path)
        return {"doc_id": doc_id, "title": title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/pptx", response_model=IngestResponse)
def ingest_pptx_endpoint(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pptx") as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name
        doc_id, title = ingest_pptx(tmp_path, file.filename)
        os.unlink(tmp_path)
        return {"doc_id": doc_id, "title": title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── USAGE ENDPOINTS ──────────────────────────────────────────────
# All four take only a doc_id (never care which source it came from).
# They load the already-cached vectorstore for that doc_id and run
# a different task on top of it — this is what makes summarize/quiz/
# mindmap reusable across YouTube, PDF, and PPTX with zero extra code.

@app.post("/ask/doc", response_model=AskResponse)
def ask_doc_endpoint(request: AskDocRequest):
    """Answer a specific question using retrieval (hybrid + MMR +
    contextual compression) — narrow, fact-based Q&A, not whole-doc tasks."""
    try:
        result = ask_question(request.doc_id, request.question)
        return {"video_id": result["doc_id"], "question": result["question"], "answer": result["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize/doc", response_model=SummarizeDocResponse)
def summarize_doc_endpoint(request: SummarizeDocRequest):
    """Summarize the whole document. Uses ALL chunks, no retrieval —
    retrieval would filter out content, which defeats a full summary."""
    try:
        return summarize_doc(request.doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quiz/doc", response_model=QuizDocResponse)
def quiz_doc_endpoint(request: QuizDocRequest):
    """Generate a 5-question MCQ quiz from the full document content."""
    try:
        return quiz_doc(request.doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mindmap/doc", response_model=MindmapDocResponse)
def mindmap_doc_endpoint(request: MindmapDocRequest):
    """Generate a hierarchical topic/subtopic mindmap (as JSON) from
    the full document content, for the frontend to render as a tree."""
    try:
        return mindmap_doc(request.doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))