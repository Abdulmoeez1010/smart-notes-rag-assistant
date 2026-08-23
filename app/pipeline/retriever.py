"""
RETRIEVAL PIPELINE
===================
This file implements the full retrieval stage of the RAG pipeline, in three phases:

1. PRE-RETRIEVAL   -> improve the QUERY itself before searching (multi-query generation)
2. DURING-RETRIEVAL -> improve HOW we search the vector store (MMR + hybrid BM25/semantic search)
3. POST-RETRIEVAL   -> improve the RETRIEVED CHUNKS before they reach the LLM (contextual compression)

Design principle followed throughout: minimize Gemini generation-LLM calls (20/day free-tier
quota), since a single user question already triggers other pipeline stages. Most of the
techniques below are deliberately "free" (pure local computation) rather than LLM-based,
except where an LLM call gives real value that a free technique cannot replicate
(query rephrasing genuinely needs language understanding, unlike vector math).
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.pipeline.vectorstore import get_all_chunks_from_vectorstore

# Post-retrieval specific imports
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
from app.pipeline.embeddings import get_query_embedder
import numpy as np


# =============================================================================
# PHASE 1: PRE-RETRIEVAL — Multi-Query Generation
# =============================================================================
# WHY THIS EXISTS:
# A user's exact phrasing might not lexically/semantically match how the source
# transcript expresses the same idea (especially true here: our transcript is
# spoken Hindi/English code-switched lecture content, not clean written text).
# By generating a few alternate phrasings of the same question, we widen the
# semantic net cast during search — increasing the chance of finding relevant
# chunks that the original phrasing alone might miss.
#
# COST: 1 Gemini generation call per user question (this is the ONLY LLM
# generation call anywhere in the retrieval pipeline — everything else below
# is either free local computation or embedding calls, which are on a
# separate quota from generation calls).

query_variation_prompt = PromptTemplate(
    input_variables=["num_variations", "question"],
    template=(
        "Generate {num_variations} alternative ways to ask the following question. "
        "Each variation should preserve the original meaning but use different words/phrasing. "
        "Return ONLY the variations, one per line, no numbering, no extra text.\n\n"
        "Original question: {question}"
    ),
)


def generate_query_variations(question: str, num_variations: int = 2) -> list[str]:
    """
    PRE-RETRIEVAL step.
    Uses the LLM to rephrase the user's question into alternate forms, so we can
    search with multiple phrasings instead of just one. Returns the original
    question plus `num_variations` rewritten versions (list length = 1 + num_variations).
    """
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        api_key=GEMINI_API_KEY,
        temperature=0.3,  # low-but-nonzero: genuine variation in phrasing, without drifting from the original meaning
    )

    prompt = query_variation_prompt.format(
        num_variations=num_variations,
        question=question,
    )

    response = llm.invoke(prompt)
    variations = [line.strip() for line in response.content.strip().split("\n") if line.strip()]

    return [question] + variations[:num_variations]


# =============================================================================
# PHASE 2: DURING-RETRIEVAL — MMR + Hybrid (BM25 + Semantic) Search
# =============================================================================
# WHY MMR (Maximal Marginal Relevance):
# Plain top-k similarity search tends to return several near-duplicate chunks,
# because the most similar chunk to a query is often surrounded by other very
# similar chunks (e.g. a speaker repeating the same idea across sentences).
# MMR balances RELEVANCE to the query against DIVERSITY from chunks already
# selected — so instead of 3 chunks all repeating the same point, you get
# chunks covering different angles of the answer. Pure local vector math
# inside FAISS — ZERO extra LLM/embedding calls beyond what search already needs.
#
# WHY HYBRID (BM25 keyword search + semantic search):
# Semantic (embedding-based) search is great at "meaning" but can underweight
# exact keyword matches — especially technical terms (RAG, LangChain, FAISS)
# that appear in English mid-sentence within an otherwise Hindi transcript.
# BM25 is a classic keyword/term-frequency algorithm with NO understanding of
# meaning — it just reliably catches exact term matches. Combining both via
# EnsembleRetriever means a chunk ranked highly by EITHER method gets pulled
# up in the final merged ranking. Also pure local computation — no LLM cost.
#
# WEIGHTING: 60% semantic / 40% keyword — semantic search is generally more
# useful for natural-language question-answering, but keyword search acts as
# a safety net for exact technical term matches.

def build_hybrid_retriever(vectorstore: FAISS, k: int = 3) -> EnsembleRetriever:
    """
    DURING-RETRIEVAL step (setup).
    Builds a hybrid retriever combining:
      - BM25Retriever: keyword/lexical search over the raw chunk text
      - FAISS + MMR: semantic search with diversity-aware selection
    merged via EnsembleRetriever's weighted reciprocal rank fusion.
    """
    # BM25 needs raw chunk TEXT (not vectors) — pulled back out of FAISS's
    # internal docstore rather than storing a duplicate copy anywhere.
    chunks = get_all_chunks_from_vectorstore(vectorstore)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 4, "lambda_mult": 0.6},
        # fetch_k: candidate pool MMR chooses from (bigger than k so MMR has room to diversify)
        # lambda_mult: 1.0 = pure relevance, 0.0 = pure diversity; 0.6 = relevance-leaning balance
    )

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6],  # 40% keyword, 60% semantic
    )


def multi_query_retrieve(question: str, vectorstore: FAISS, k: int = 3) -> list[Document]:
    """
    Orchestrates PRE-RETRIEVAL + DURING-RETRIEVAL together:
    1. Generate query variations (pre-retrieval)
    2. Run EACH variation through the hybrid MMR+BM25 retriever (during-retrieval)
    3. Merge all results across variations, deduplicating by exact chunk content
       (different query variations often retrieve overlapping/identical chunks)
    """
    queries = generate_query_variations(question)
    hybrid_retriever = build_hybrid_retriever(vectorstore, k=k)

    all_results = []
    seen_content = set()  # dedup key: raw chunk text (identical chunks retrieved by different query variations should only count once)

    for query in queries:
        results = hybrid_retriever.invoke(query)
        for doc in results:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                all_results.append(doc)

    return all_results


# =============================================================================
# PHASE 3: POST-RETRIEVAL — Contextual Compression (Embedding-Based Filtering)
# =============================================================================
# WHY THIS EXISTS:
# Hybrid retrieval can still let through a chunk that's only weakly relevant
# (it made the cut on ONE query variation or ONE retrieval method, but isn't
# strongly related to the actual question). Contextual compression re-checks
# each retrieved chunk's relevance and drops the weak ones before they reach
# the LLM — reducing noise in the final context window.
#
# IMPLEMENTATION CHOICE: EmbeddingsFilter (re-embed + similarity threshold),
# NOT LLMChainExtractor (LLM-based extraction). LLM-based compression would
# cost 1 Gemini generation call PER RETRIEVED CHUNK (e.g. 5-6 extra calls per
# question) — completely unworkable against a 20/day quota. EmbeddingsFilter
# only costs embedding calls (separate, much higher quota), for near-zero
# practical cost.
#
# THRESHOLD NOTE: 0.60 was NOT guessed — it was set by actually measuring real
# cosine similarity scores between a test query and its retrieved chunks
# (observed range: 0.60-0.68 for this cross-lingual Hindi/English transcript).
# Cross-lingual embedding similarity is inherently lower than same-language
# similarity, so a generic default like 0.75 was too strict and filtered out
# EVERYTHING. This threshold is tuned to this project's actual data — a known
# limitation is that it's an absolute cutoff, not adaptive per-question/video;
# a more robust version would use a relative threshold (e.g. "drop anything
# more than X below the top score") rather than a fixed number.

def apply_contextual_compression(chunks: list[Document], question: str, similarity_threshold: float = 0.60) -> list[Document]:
    """
    POST-RETRIEVAL step.
    Re-embeds each chunk and the question, drops any chunk whose similarity
    to the question falls below `similarity_threshold`. Zero generation-LLM
    calls — only (separately-quota'd) embedding calls.

    FALLBACK: the 0.60 threshold was tuned on one video's cross-lingual
    embedding distribution and doesn't generalize to every document (e.g.
    videos/documents with different language mixes, domains, or transcript
    styles may have naturally lower/higher similarity scores). Rather than
    risk silently returning zero chunks (and the LLM wrongly claiming "not
    in context" when relevant chunks actually exist), if compression filters
    everything out, fall back to the original uncompressed top-k chunks
    instead of an empty list.
    """
    embeddings_filter = EmbeddingsFilter(
        embeddings=get_query_embedder(),
        similarity_threshold=similarity_threshold,
    )

    compressed = embeddings_filter.compress_documents(chunks, query=question)
    compressed = list(compressed)

    if not compressed and chunks:
        # threshold filtered out everything -> fall back to uncompressed
        # top-k rather than starving generation of context entirely
        return chunks[:3]

    return compressed

# =============================================================================
# DIAGNOSTIC UTILITY (not part of the pipeline — used only to tune the
# similarity_threshold above against real data instead of guessing)
# =============================================================================

def debug_similarity_scores(chunks: list[Document], question: str):
    """
    Prints actual cosine similarity between the question and each chunk.
    Used once to determine a real similarity_threshold value for this project's
    data — not called during normal pipeline operation.
    """
    embedder = get_query_embedder()
    query_vec = np.array(embedder.embed_query(question))
    for i, doc in enumerate(chunks):
        doc_vec = np.array(embedder.embed_query(doc.page_content))
        similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
        print(f"Chunk {i+1} similarity: {similarity:.3f}")


# =============================================================================
# TEST BLOCK — runs the full pre -> during -> post retrieval pipeline end-to-end
# =============================================================================

if __name__ == "__main__":
    from app.pipeline.vectorstore import build_or_load_vectorstore

    video_id = "J5_-l7WIO_w"
    vectorstore = build_or_load_vectorstore(video_id)  # loads from disk cache, no re-embedding cost

    question = "what is RAG"

    # Pre-retrieval + during-retrieval combined
    results = multi_query_retrieve(question, vectorstore)
    print(f"Before compression: {len(results)} chunks\n")

    # Post-retrieval
    compressed_results = apply_contextual_compression(results, question)
    print(f"After compression: {len(compressed_results)} chunks\n")

    for i, doc in enumerate(compressed_results):
        print(f"--- Chunk {i+1} ---")
        print(doc.page_content[:150])
        print()