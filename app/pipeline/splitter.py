from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_document(doc: Document, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Document]:
    """
    Split a single transcript Document into smaller chunks for embedding.
    Uses recursive splitting to prefer natural sentence/paragraph boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "। ", ". ", "? ", "! ", " ", ""],
        # "। " added explicitly for Hindi sentence-ending punctuation (danda)
    )

    chunks = splitter.split_documents([doc])

    # keep track of chunk order — useful later for citations
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks
