import streamlit as st
import time
from pathlib import Path

from legal_dc_baseline.config import ExperimentConfig
from legal_dc_baseline.data import load_corpus
from legal_dc_baseline.retrieval import HybridRetriever
from legal_dc_baseline.generation import LocalGenerator

# Page Setup
st.set_page_config(
    page_title="Legal-DC Hybrid RAG Explorer",
    page_icon="⚖️",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stMetric { background-color: #1E222D; padding: 15px; border-radius: 8px; border: 1px solid #2E3440; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    """Initialize Config, Corpus, Retriever, and Generator using exact repo classes."""
    cfg = ExperimentConfig(
        corpus_path=Path("data/document_corpus.json"),
        qa_path=Path("data/qa_pairs.json"),
        results_dir=Path("results")
    )
    
    chunks = load_corpus(cfg.corpus_path)
    
    retriever = HybridRetriever(
        chunks=chunks,
        dense_model=cfg.dense_model,
        reranker_model=cfg.reranker_model,
        dense_k=cfg.dense_k,
        bm25_k=cfg.bm25_k,
        final_k=cfg.final_k,
        batch_size=cfg.batch_size,
        device=cfg.device
    )
    
    generator = LocalGenerator(
        model_name=cfg.generator_model,
        max_new_tokens=cfg.max_new_tokens,
        device=cfg.device
    )
    
    return retriever, generator, cfg

st.title("⚖️ Legal-DC Hybrid RAG Explorer")
st.caption("Single Query Interactive Runner for BM25 + BGE Dense + Cross-Encoder Reranking")

with st.spinner("Loading models and indexing corpus..."):
    retriever, generator, cfg = load_pipeline()

with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown(f"**Dense Model:** `{cfg.dense_model}`")
    st.markdown(f"**Reranker:** `{cfg.reranker_model}`")
    st.markdown(f"**Generator:** `{cfg.generator_model}`")
    st.markdown(f"**Final Top-K Passages:** `{cfg.final_k}`")

query_input = st.text_area(
    "Enter Legal Query:", 
    placeholder="Under the Constitution of India, can Parliament establish a new state without any conditions?",
    height=100
)

if st.button("🔍 Run Single Query", type="primary", use_container_width=True):
    if not query_input.strip():
        st.warning("Please enter a valid query.")
    else:
        start_time = time.time()
        
        with st.spinner("Retrieving top passages..."):
            t_ret_start = time.time()
            retrieved_passages = retriever.retrieve(query_input)
            retrieval_latency = time.time() - t_ret_start
        
        # Check retrieval results FIRST before generating
        if not retrieved_passages:
            t_gen_start = time.time()
            generated_answer = "⚠️ No relevant legal passages found in the Indian Constitution corpus for this query."
            generation_latency = time.time() - t_gen_start
            total_time = time.time() - start_time

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Execution Time", f"{total_time:.2f}s")
            col2.metric("Retrieval Latency", f"{retrieval_latency:.2f}s")
            col3.metric("Generation Latency", f"{generation_latency:.2f}s")

            st.divider()
            st.warning("Query Out of Scope: The retrieved scores fell below the relevance threshold.")
            st.info(generated_answer)
        else:
            with st.spinner("Generating answer..."):
                t_gen_start = time.time()
                generated_answer = generator.answer(query_input, retrieved_passages)
                generation_latency = time.time() - t_gen_start
                total_time = time.time() - start_time

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Execution Time", f"{total_time:.2f}s")
            col2.metric("Retrieval Latency", f"{retrieval_latency:.2f}s")
            col3.metric("Generation Latency", f"{generation_latency:.2f}s")

            st.divider()

            st.subheader("💡 Generated Legal Answer")
            st.info(generated_answer)

            st.subheader(f"📚 Retrieved Top-{len(retrieved_passages)} Passages")
            for passage in retrieved_passages:
                rank = passage['rank']
                score = passage['reranker_score']
                art_ref = passage['article_reference']
                title = passage['title']
                text = passage['text']
                
                with st.expander(f"Rank #{rank} | {art_ref} - {title} (Reranker Score: {score:.4f})", expanded=(rank == 1)):
                    st.write(text)



# from pathlib import Path
# import time
# import streamlit as st

# from legal_dc_baseline.config import ExperimentConfig
# from legal_dc_baseline.data import load_corpus
# from legal_dc_baseline.retrieval import HybridRetriever

# # Page Setup
# st.set_page_config(
#     page_title="Legal-DC Fast Retrieval Metrics Explorer",
#     page_icon="⚖️",
#     layout="wide",
# )

# st.markdown(
#     """
#     <style>
#     .main { background-color: #0E1117; }
#     .stMetric { background-color: #1E222D; padding: 15px; border-radius: 8px; border: 1px solid #2E3440; }
#     </style>
# """,
#     unsafe_allow_html=True,
# )


# @st.cache_resource
# def load_retriever_pipeline():
#   cfg = ExperimentConfig(
#       corpus_path=Path("data/document_corpus.json"),
#       qa_path=Path("data/qa_pairs.json"),
#       results_dir=Path("results"),
#   )
#   chunks = load_corpus(cfg.corpus_path)

#   retriever = HybridRetriever(
#       chunks=chunks,
#       dense_model=cfg.dense_model,
#       reranker_model=cfg.reranker_model,
#       dense_k=cfg.dense_k,
#       bm25_k=cfg.bm25_k,
#       final_k=cfg.final_k,
#       batch_size=cfg.batch_size,
#       device=cfg.device,
#   )
#   return retriever, cfg


# def compute_retrieval_metrics(retrieved_passages, target_article, k):
#   """Computes Recall@K and MRR@K for the single query."""
#   found_rank = None
#   for idx, passage in enumerate(retrieved_passages[:k]):
#     if passage.get("article_reference") == target_article:
#       found_rank = idx + 1
#       break

#   recall = 1.0 if found_rank is not None else 0.0
#   mrr = (1.0 / found_rank) if found_rank is not None else 0.0
#   return recall, mrr, found_rank


# st.title("⚖️ Fast Legal-DC Retrieval Metrics Explorer")
# st.caption(
#     "Fast evaluation without LLM generation for BM25, Dense, and Hybrid Search"
# )

# with st.spinner("Loading indexed corpus and embedding models..."):
#   retriever, cfg = load_retriever_pipeline()

# with st.sidebar:
#   st.header("⚙️ Evaluation Config")
#   st.markdown(f"**Dense Model:** `{cfg.dense_model}`")
#   st.markdown(f"**Top-K Evaluation:** `{cfg.final_k}`")

# query_input = st.text_area(
#     "Enter Legal Query:",
#     placeholder=(
#         "Under the Constitution of India, can Parliament establish a new state"
#         " without any conditions?"
#     ),
#     height=80,
# )
# target_article_input = st.text_input(
#     "Target Article Reference (Optional for metrics verification):",
#     placeholder="Article 3",
# )

# if st.button("🔍 Evaluate Query Metrics", type="primary", use_container_width=True):
#   if not query_input.strip():
#     st.warning("Please enter a valid query.")
#   else:
#     t_start = time.time()

#     # Retrieval across all 3 modes (assuming retrieve method supports mode or fallback)
#     results = {}
#     for mode in ["bm25", "dense", "hybrid"]:
#       if hasattr(retriever, "retrieve_by_mode"):
#         passages = retriever.retrieve_by_mode(query_input, mode=mode)
#       else:
#         passages = retriever.retrieve(query_input)
#       results[mode] = passages

#     latency = time.time() - t_start

#     st.subheader(f"⚡ Retrieval Latency: {latency:.4f}s")
#     st.divider()

#     # Display Metrics for each Mode
#     st.subheader("📊 Method-Wise Metrics (Recall@K & MRR@K)")
#     cols = st.columns(3)

#     for i, mode in enumerate(["bm25", "dense", "hybrid"]):
#       passages = results[mode]
#       recall, mrr, rank = compute_retrieval_metrics(
#           passages, target_article_input, cfg.final_k
#       )

#       with cols[i]:
#         st.markdown(f"### `{mode.upper()}` Search")
#         st.metric(f"Recall@{cfg.final_k}", f"{recall:.4f}")
#         st.metric(f"MRR@{cfg.final_k}", f"{mrr:.4f}")
#         if rank:
#           st.success(f"Matched Target at Rank #{rank}")
#         else:
#           st.caption("Target Article not in Top-K")

#     st.divider()

#     # Show Top Retrieved Passages for Hybrid
#     st.subheader(f"📚 Top-{cfg.final_k} Passages (Hybrid Mode)")
#     for passage in results["hybrid"][: cfg.final_k]:
#       rank = passage.get("rank", 1)
#       score = passage.get("reranker_score", 0.0)
#       art_ref = passage.get("article_reference", "N/A")
#       title = passage.get("title", "")
#       text = passage.get("text", "")

#       with st.expander(
#           f"Rank #{rank} | {art_ref} - {title} (Score: {score:.4f})",
#           expanded=(rank == 1),
#       ):
#         st.write(text)