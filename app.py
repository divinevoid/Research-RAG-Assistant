import streamlit as st
import numpy as np
from config import RELEVANCE_THRESHOLD, NOVELTY_THRESHOLD

from a import (
    setup,
    parse_docs,
    build_context,
    generate_research_ideas,
    novelty_scores,
    retrieve_with_full_scores
)

from eval.logger import log_result
from eval.judge import llm_judge_score

# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="Research RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# -------------------- INIT ENGINE --------------------

@st.cache_resource(show_spinner=False)
def load_engine():
    return setup()

llm, retriever, reranker, embedding_function = load_engine()


# -------------------- SESSION STATE --------------------

if "results" not in st.session_state:
    st.session_state.results = None

if "docs" not in st.session_state:
    st.session_state.docs = None

if "context" not in st.session_state:
    st.session_state.context = None

if "ideas" not in st.session_state:
    st.session_state.ideas = None

if "novelty" not in st.session_state:
    st.session_state.novelty = None

if "active_query" not in st.session_state:
    st.session_state.active_query = None

if "logged_papers" not in st.session_state:
    st.session_state.logged_papers = set()

# -------------------- UI --------------------

st.title("📚 Research RAG Assistant")
st.caption("Retrieve papers • Generate novel ideas • Scientifically evaluate performance")

query = st.text_input(
    "Enter your research question or topic",
    placeholder="e.g. Agentic RAG for scientific discovery"
)

# uploaded_pdf = st.file_uploader("Upload your PDF here:", type=["pdf"])

col1, col2 = st.columns(2)


# -------------------- FIND PAPERS --------------------

with col1:
    if st.button("🔍 Find Relevant Papers", use_container_width=True):

        if not query.strip():
            st.warning("Please enter a query.")
            st.stop
            
        if query != st.session_state.active_query:
            st.session_state.logged_papers = set()

        st.session_state.active_query = query
        with st.spinner("🔎 Retrieving relevant papers..."):

            results = retrieve_with_full_scores(
                query, retriever, reranker, embedding_function, top_k=8
            )

            filtered=[
                r for r in results
                if r["combined_score"]>= RELEVANCE_THRESHOLD
            ]

            if not filtered:
                st.warning("No papers retrieved.")
                st.stop()

            st.session_state.results = filtered
            st.session_state.docs = [r["doc"] for r in filtered]
            st.session_state.context = build_context(st.session_state.docs)

            st.success("Papers retrieved.")

# -------------------- GENERATE IDEAS --------------------

with col2:
    if st.button("💡 Generate Research Ideas", use_container_width=True):

        if not st.session_state.docs:
            st.warning("Retrieve papers first.")
        else:
            with st.spinner("Generating grounded research ideas..."):

                ideas_response = generate_research_ideas(
                    llm,
                    st.session_state.context,
                    query
                )

                novelty = novelty_scores(
                    ideas_response,
                    st.session_state.docs,
                    embedding_function
                )

                st.session_state.ideas = ideas_response.ideas
                st.session_state.novelty = novelty

                st.success(f"Generated {len(ideas_response.ideas)} ideas.")


# -------------------- DISPLAY PAPERS --------------------

if st.session_state.results:

    st.subheader("📄 Retrieved Papers")

    for i, r in enumerate(st.session_state.results):

        doc = r["doc"]
        title, summary = parse_docs(doc)
        meta = doc.metadata

        combined = r["combined_score"]

        with st.expander(f"📘 {title} | Score: {combined:.3f}"):

            st.write(f"**Author:** {meta.get('Author', 'Unknown')}")
            st.write(f"**Category:** {meta.get('Primary Category', 'Unknown')}")
            st.write(f"**Summary:** {summary}")
            st.markdown(f"[🔗 Paper Link]({meta.get('Link', '#')})")

            # -------------------- HUMAN EVALUATION --------------------

            human_score = st.slider(
                "Human Relevance (1–5)",
                1, 5, 3,
                key=f"human_{i}"
            )

            confirm = st.checkbox(
                "Confirm this evaluation",
                key=f"confirm_{i}"
            )

            paper_id = meta.get("id", f"paper_{i}")
            paper_key = f"{st.session_state.active_query}_{paper_id}"

            if confirm and paper_key not in st.session_state.logged_papers:

                human_binary = 1 if human_score >= 4 else 0

                log_result({
                    "query": st.session_state.active_query,
                    "paper_id": paper_id,
                    "model_score": combined,
                    "human_score": human_score,
                    "human_binary": human_binary,
                    "threshold_used": RELEVANCE_THRESHOLD
                })

                st.session_state.logged_papers.add(paper_key)


# -------------------- DISPLAY IDEAS --------------------

if st.session_state.ideas:

    st.subheader("💡 Generated Research Ideas")

    for i, idea in enumerate(st.session_state.ideas):

        novelty = st.session_state.novelty[i]

        if novelty < NOVELTY_THRESHOLD:
           continue

        with st.expander(f"💡 {idea.title} | Novelty: {novelty:.3f}"):

            st.write(f"**Motivation:** {idea.motivation}")
            st.write(f"**Key Research Question:** {idea.research_question}")
            st.write(f"**Why it’s Novel:** {idea.novelty}")
            st.write(f"**Methodology:** {idea.methodology}")            
            # -------------------- LLM JUDGE FOR IDEAS --------------------
            col1, col2 = st.columns([1, 3])
            
            with col1:
                use_llm_judge = st.checkbox(
                    "Use LLM Judge",
                    key=f"idea_llm_judge_{i}"
                )
            
            with col2:
                if use_llm_judge:
                    with st.spinner("🤖 LLM Judge evaluating..."):
                        # Combine idea info for judge to evaluate
                        idea_text = f"""
                        Title: {idea.title}
                        Motivation: {idea.motivation}
                        Research Question: {idea.research_question}
                        Why it's Novel: {idea.novelty}
                        Methodology: {idea.methodology}
                        """
                        
                        judge_score = llm_judge_score(
                            llm,
                            query,
                            idea_text
                        )
                        
                        st.write(f"### 🎯 LLM Judge Score: {judge_score}/5")
                        
                        # Display score with visual indicator
                        score_color = "green" if judge_score >= 4 else "orange" if judge_score >= 3 else "red"
                        st.markdown(f":{score_color}[Score: {judge_score}/5]")