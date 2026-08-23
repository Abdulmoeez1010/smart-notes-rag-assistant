"""
AUGMENTATION
============
Takes the final retrieved+compressed chunks and assembles them into a structured
prompt for the LLM. Covers:
  - Prompt templating: structured, reusable prompt structure
  - Answer grounding: explicit instruction to answer ONLY from given context (anti-hallucination)
  - Context window optimization: clearly separated/numbered chunks so the LLM
    can distinguish sources (also enables citation in generation.py later)

No LLM calls here — pure string/prompt construction.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document


rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant answering questions about a YouTube video, "
        "based ONLY on the transcript excerpts provided below.\n\n"
        "IMPORTANT RULES:\n"
        "- Answer using ONLY the information in the context below. Do not use outside knowledge.\n"
        "- If the context does not contain enough information to answer, say so clearly — "
        "do not guess or make up an answer.\n"
        "- Always answer in English, even if the context is in Hindi/Urdu.\n"
        "- Be concise and direct.\n"
        "- At the end of your answer, cite which excerpt number(s) you used, "
        "like this: (Source: Excerpt 2).\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION: {question}\n\n"
        "ANSWER:"
    ),
)


def format_context(chunks: list[Document]) -> str:
    """
    Formats retrieved chunks into a clearly numbered context block.
    Numbering lets the LLM (and us, later) reference specific sources,
    which sets up citation support in the generation step.
    """
    formatted = []
    for i, doc in enumerate(chunks, start=1):
        formatted.append(f"[Excerpt {i}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def build_augmented_prompt(chunks: list[Document], question: str) -> str:
    """
    Combines formatted context + question into the final prompt string,
    ready to send to the LLM in generation.py.
    """
    context = format_context(chunks)
    return rag_prompt.format(context=context, question=question)


if __name__ == "__main__":
    from app.pipeline.vectorstore import build_or_load_vectorstore
    from app.pipeline.retriever import multi_query_retrieve, apply_contextual_compression

    video_id = "J5_-l7WIO_w"
    vectorstore = build_or_load_vectorstore(video_id)

    question = "what is RAG"
    results = multi_query_retrieve(question, vectorstore)
    compressed = apply_contextual_compression(results, question)

    final_prompt = build_augmented_prompt(compressed, question)
    print(final_prompt)