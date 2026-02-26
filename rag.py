"""
RAG Pipeline Conversational Chatbot: Research Paper Analysis
Enhanced with:
- Reranking
- Citation enforcement
- Idea-paper similarity checking
- Novelty scoring
- PDF-based retrieval
"""

import os
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from base import ResearchIdeaResponse


# -------------------- SETUP --------------------

def setup(top_k: int = 8):
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    llm = ChatGoogleGenerativeAI(
        api_key=api_key,
        model="gemini-3-flash-preview",
        temperature=0.4,
    )

    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_function,
        collection_name="articles",
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return llm, retriever, reranker, embedding_function


# -------------------- PDF INGESTION --------------------

def extract_text_from_pdf(file) -> str:
    """Extract raw text from uploaded PDF."""
    from pypdf import PdfReader
    reader = PdfReader(file)
    #reader = pypdf.PdfReader(file)
    text = []
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text.append(content)
    return "\n".join(text)


def chunk_pdf_text(text: str, chunk_size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)


def pdf_to_query(llm, chunks, max_chars=4000):
    """Convert PDF content into a semantic retrieval query."""
    text = " ".join(chunks)[:max_chars]
    prompt = f"""
You are a research assistant.
Summarize the core research themes and methods in the following document
into a single concise search query.

Document:
{text}
"""
    return llm.invoke(prompt).content.strip()


# -------------------- DOCUMENT PROCESSING --------------------

def parse_docs(doc):
    text = doc.page_content
    if "|||SUMMARY|||" in text:
        title, summary = text.split("|||SUMMARY|||", 1)
        return title.strip(), summary.strip()
    return doc.metadata.get("Title", "No Title"), text


def build_context(docs):
    blocks = []
    for i, doc in enumerate(docs):
        title, summary = parse_docs(doc)
        meta = doc.metadata
        block = f"""
[Paper {i+1}]
Title: {title}
Summary: {summary}
Author: {meta.get("Author", "Unknown")}
Primary Category: {meta.get("Primary Category", "Unknown")}
Link: {meta.get("Link", "No Link")}
"""
        blocks.append(block.strip())
    return "\n\n".join(blocks)


# -------------------- RETRIEVAL PIPELINE --------------------

def rerank_docs_with_scores(query, docs, reranker, top_n=10):
    if not docs:
        return []
    pairs = [(query, doc.page_content) for doc in docs]
    ce_scores = reranker.predict(pairs)
    
    # Normalize scores to 0-1 range using min-max normalization
    min_score = min(ce_scores)
    max_score = max(ce_scores)
    if max_score > min_score:
        normalized_scores = [(s - min_score) / (max_score - min_score) for s in ce_scores]
    else:
        normalized_scores = [0.5] * len(ce_scores)
    
    ranked = sorted(zip(docs, normalized_scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def comnbine_relevance_scores(ce_scores, cos_sim, alpha=0.5):

    return alpha*cos_sim + (1-alpha)*ce_scores


def retrieve_with_full_scores(query: str, retriever, reranker, embedding_function, top_k=10):
    raw_docs = retriever.invoke(query)
    if not raw_docs:
        return []
    reranked_with_ce = rerank_docs_with_scores(query, raw_docs, reranker, top_n=top_k)

    query_emb = embedding_function.embed_query(query)
    doc_embs = embedding_function.embed_documents(
        [doc.page_content for doc, _ in reranked_with_ce]
    )

    cos_scores=cosine_similarity([query_emb], doc_embs)[0]

    results =[]

    for (doc, ce), cos in zip(reranked_with_ce, cos_scores):
        combined_score = comnbine_relevance_scores(float(ce), float(cos))

        results.append({
            "doc": doc,
            "cosine_score": float(cos),
            "cross_encoder_score": float(ce),
            "combined_score": float(combined_score)})
        
    return sorted(results, key=lambda x: x["combined_score"], reverse=True)



# -------------------- IDEA GENERATION --------------------

RESEARCH_IDEA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a senior research scientist. Use the provided research papers to generate novel research ideas.

Research Papers:
{context}

TASK:
Generate exactly 5 novel research ideas.

STRICT RULES:
- Every idea MUST cite at least one paper using [Paper X]
- Do NOT invent citations
- Explain novelty by contrasting with cited papers
- Focus on "cross-pollination"—combine techniques from one paper with the domain of another.

For each idea provide:
1. Title
2. Motivation (with citations)
3. Key Research Question
4. Why it's novel (vs cited work)
5. Possible methodology

Question: {question}

Generate now:
"""
)

def generate_research_ideas(llm, context, question):
    structured_llm = llm.with_structured_output(ResearchIdeaResponse)
    chain = RESEARCH_IDEA_PROMPT | structured_llm
    return chain.invoke({"context": context, "question": question})


# -------------------- IDEA ANALYSIS --------------------

def idea_paper_similarity(ideas_response, docs, embedding_function):
    paper_embeddings = embedding_function.embed_documents(
        [doc.page_content for doc in docs]
    )
    similarities = []
    for idea in ideas_response.ideas:
        idea_text = f"""
        {idea.title}
        {idea.motivation}
        {idea.research_question}
        {idea.methodology}
        """
        emb = embedding_function.embed_query(idea_text)
        sims = cosine_similarity([emb], paper_embeddings)[0]
        similarities.append(round(float(np.max(sims)), 3))
    return similarities


def novelty_scores(ideas_response, docs, embedding_function):
    sims = idea_paper_similarity(ideas_response, docs, embedding_function)
    return [round(1 - s, 3) for s in sims]
