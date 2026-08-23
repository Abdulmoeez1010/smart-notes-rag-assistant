"""
GENERATION (LCEL style)
========================
Uses LangChain Expression Language (the `|` pipe operator) to compose the
final generation step: prompt -> llm -> output parser.

LCEL fits here because generation is genuinely linear (one input, one
straight-through flow) — unlike retrieval, which has branching/looping logic
(multi-query, merging, conditional compression) that doesn't map cleanly onto
simple pipe composition.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.pipeline.augmentation import rag_prompt  # reuse the same PromptTemplate from augmentation


llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    api_key=GEMINI_API_KEY,
    temperature=0.2,
)

# LCEL chain: prompt template -> LLM -> plain string output
# StrOutputParser extracts just the text content from the LLM's response object,
# so callers get a clean string instead of having to do response.content themselves.
generation_chain = rag_prompt | llm | StrOutputParser()


def generate_answer(context: str, question: str) -> str:
    """
    Runs the LCEL generation chain: formats the prompt with context+question,
    sends to Gemini, returns the parsed answer string.
    Costs 1 Gemini generation call.
    """
    try:
        answer = generation_chain.invoke({"context": context, "question": question})

        if not answer.strip():
            return "Sorry, I couldn't generate an answer. Please try rephrasing your question."

        return answer.strip()

    except Exception as e:
        return f"Sorry, something went wrong while generating the answer: {e}"

