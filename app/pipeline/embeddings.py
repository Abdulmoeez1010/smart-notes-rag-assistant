from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import GEMINI_API_KEY, EMBEDDING_MODEL


def get_document_embedder() -> GoogleGenerativeAIEmbeddings:
    """Embedder for chunks going INTO the vector store."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=GEMINI_API_KEY,
        task_type="retrieval_document",
    )


def get_query_embedder() -> GoogleGenerativeAIEmbeddings:
    """Embedder for the user's question at search time."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=GEMINI_API_KEY,
        task_type="retrieval_query",
    )