import os
import json
import re
from app.ingestion.youtube_loader import load_transcript, extract_video_id
from app.ingestion.pdf_loader import load_pdf, generate_doc_id as pdf_doc_id
from app.ingestion.pptx_loader import load_pptx
from app.pipeline.splitter import split_document
from app.pipeline.vectorstore import build_or_load_vectorstore, get_all_chunks_from_vectorstore, VECTORSTORE_DIR
from app.pipeline.retriever import multi_query_retrieve, apply_contextual_compression
from app.pipeline.generation import generate_answer
from app.pipeline.augmentation import format_context
from app.pipeline.summarization import summarize_chunks
from app.pipeline.quiz import generate_quiz
from app.pipeline.mindmap import generate_mindmap
from app.ingestion.youtube_loader import get_video_title
# from app.ingestion.pdf_loader import get_video_title
# from app.ingestion.pptx_loader import get_video_title


def _get_or_build_vectorstore(doc_id: str, doc=None):
    index_path = os.path.join(VECTORSTORE_DIR, doc_id)
    if os.path.exists(index_path):
        return build_or_load_vectorstore(doc_id)
    if doc is None:
        raise ValueError(f"No cached index for {doc_id} and no document provided.")
    chunks = split_document(doc)
    return build_or_load_vectorstore(doc_id, chunks=chunks)


def ingest_youtube(video_url: str):
    doc_id = extract_video_id(video_url)
    doc = load_transcript(video_url)
    title = get_video_title(doc_id)
    _get_or_build_vectorstore(doc_id, doc)
    return doc_id, title


def ingest_pdf(file_path: str, original_filename: str = "document.pdf"):
    doc = load_pdf(file_path, original_filename)
    doc_id = doc.metadata["doc_id"]
    title = doc.metadata["title"]
    _get_or_build_vectorstore(doc_id, doc)
    return doc_id, title


def ingest_pptx(file_path: str, original_filename: str = "presentation.pptx"):
    doc = load_pptx(file_path, original_filename)
    doc_id = doc.metadata["doc_id"]
    title = doc.metadata["title"]
    _get_or_build_vectorstore(doc_id, doc)
    return doc_id, title


def _clean_json(raw: str):
    """Gemini sometimes wraps JSON in ```json fences despite instructions — strip defensively."""
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw.strip())
    return json.loads(cleaned)


def ask_question(doc_id: str, question: str) -> dict:
    vectorstore = _get_or_build_vectorstore(doc_id)
    retrieved = multi_query_retrieve(question, vectorstore, k=3)
    compressed = apply_contextual_compression(retrieved, question)
    context = format_context(compressed)
    answer = generate_answer(context, question)
    return {"doc_id": doc_id, "question": question, "answer": answer}


def summarize_doc(doc_id: str) -> dict:
    vectorstore = _get_or_build_vectorstore(doc_id)
    chunks = get_all_chunks_from_vectorstore(vectorstore)
    summary = summarize_chunks(chunks)
    return {"doc_id": doc_id, "summary": summary}


def quiz_doc(doc_id: str) -> dict:
    vectorstore = _get_or_build_vectorstore(doc_id)
    chunks = get_all_chunks_from_vectorstore(vectorstore)
    raw = generate_quiz(chunks)
    questions = _clean_json(raw)
    return {"doc_id": doc_id, "questions": questions}


def mindmap_doc(doc_id: str) -> dict:
    vectorstore = _get_or_build_vectorstore(doc_id)
    chunks = get_all_chunks_from_vectorstore(vectorstore)
    raw = generate_mindmap(chunks)
    mindmap = _clean_json(raw)
    return {"doc_id": doc_id, "mindmap": mindmap}