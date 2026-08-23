import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def generate_doc_id(file_path: str) -> str:
    """Stable ID from file content hash, so re-uploading the same file reuses cache."""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:16]


def load_pdf(file_path: str, original_filename: str = "document.pdf") -> Document:
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    if not pages:
        raise ValueError(f"No content extracted from PDF: {file_path}")
    full_text = "\n\n".join(p.page_content for p in pages)
    doc_id = generate_doc_id(file_path)
    title = get_pdf_title(file_path, original_filename)
    return Document(page_content=full_text, metadata={"doc_id": doc_id, "source_type": "pdf", "title": title})


def get_pdf_title(file_path: str, original_filename: str) -> str:
    """Try PDF metadata title first, fall back to the uploaded filename."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        meta_title = reader.metadata.title if reader.metadata else None
        if meta_title and meta_title.strip():
            return meta_title.strip()
    except Exception:
        pass
    return original_filename