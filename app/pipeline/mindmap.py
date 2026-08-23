from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.pipeline.generation import llm

mindmap_prompt = PromptTemplate(
    input_variables=["transcript"],
    template="""Based on the transcript below (may be Hindi/Urdu/English), create a hierarchical mindmap in English of the main topic and its subtopics.

Return ONLY valid JSON, no markdown fences, no preamble, in this exact structure:
{{"topic": "Main Topic", "children": [{{"topic": "Subtopic 1", "children": [{{"topic": "Point A", "children": []}}]}}]}}

Transcript:
{transcript}

JSON:"""
)
mindmap_chain = mindmap_prompt | llm | StrOutputParser()


def generate_mindmap(chunks) -> str:
    full_text = "\n".join(c.page_content for c in chunks)
    return mindmap_chain.invoke({"transcript": full_text})