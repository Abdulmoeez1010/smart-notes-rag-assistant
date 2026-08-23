from langchain_core.prompts import PromptTemplate
from app.pipeline.generation import llm
from langchain_core.output_parsers import StrOutputParser

summary_prompt = PromptTemplate(
    input_variables=["transcript"],
    template="""You are summarizing a video transcript. The transcript may be in Hindi, Urdu, or English.

Write a clear, well-structured summary in English, covering the main points discussed, in logical order.
Use short paragraphs or bullet points. Do not add information that isn't in the transcript.

Transcript:
{transcript}

Summary:"""
)

summary_chain = summary_prompt | llm | StrOutputParser()


def summarize_chunks(chunks) -> str:
    """Join all chunks (no retrieval/filtering) and summarize in one call."""
    full_text = "\n".join(c.page_content for c in chunks)
    return summary_chain.invoke({"transcript": full_text})