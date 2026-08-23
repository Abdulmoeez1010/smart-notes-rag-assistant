from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.pipeline.generation import llm

quiz_prompt = PromptTemplate(
    input_variables=["transcript"],
    template="""Based on the transcript below (may be Hindi/Urdu/English), create 5 multiple-choice quiz questions in English testing understanding of the content.

Return ONLY valid JSON, no markdown fences, no preamble, in this exact structure:
[
  {{"question": "...", "options": ["A", "B", "C", "D"], "correct_answer": "A", "explanation": "..."}}
]

Transcript:
{transcript}

JSON:"""
)
quiz_chain = quiz_prompt | llm | StrOutputParser()


def generate_quiz(chunks) -> str:
    full_text = "\n".join(c.page_content for c in chunks)
    return quiz_chain.invoke({"transcript": full_text})