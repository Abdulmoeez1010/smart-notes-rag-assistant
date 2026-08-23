import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from app.pipeline.embeddings import get_document_embedder, get_query_embedder

VECTORSTORE_DIR = "data/vectorstores"

def _index_path(video_id: str) -> str:
    return os.path.join(VECTORSTORE_DIR, video_id)


def build_or_load_vectorstore(video_id: str, chunks: list[Document] = None) -> FAISS:
    """
    Load an existing FAISS index for this video if one exists on disk.
    Otherwise, build a new one from the given chunks and save it.
    """
    path = _index_path(video_id)

    if os.path.exists(path):
        print(f"[vectorstore] Found existing index for {video_id}, loading from disk.")
        # allow_dangerous_deserialization is safe here since we only ever
        # load indexes WE created ourselves, never from an untrusted source
        return FAISS.load_local(
            path,
            get_query_embedder(),  # embedder used for loading must match search-time embedder
            allow_dangerous_deserialization=True,
        )

    if chunks is None:
        raise ValueError(f"No index found for {video_id} and no chunks provided to build one.")

    print(f"[vectorstore] No existing index for {video_id}, building new one ({len(chunks)} chunks).")
    vectorstore = FAISS.from_documents(chunks, get_document_embedder())

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(path)

    return vectorstore

def get_all_chunks_from_vectorstore(vectorstore: FAISS) -> list[Document]:
    """
    Extract all stored Document chunks from an existing FAISS index,
    for use in building a BM25 keyword index (BM25 needs raw text, not vectors).
    """
    return list(vectorstore.docstore._dict.values())