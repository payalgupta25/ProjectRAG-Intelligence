import argparse
from pathlib import Path
import time
from tqdm import tqdm

from legal_dc_baseline.config import ExperimentConfig
from legal_dc_baseline.data import load_corpus, load_qa
from legal_dc_baseline.retrieval import HybridRetriever


def compute_metrics_for_passages(retrieved_items, target_article, k):
  """Computes Recall@K and MRR@K."""
  found_rank = None
  for idx, item in enumerate(retrieved_items[:k]):
    if item.get("article_reference") == target_article:
      found_rank = idx + 1
      break

  recall = 1.0 if found_rank is not None else 0.0
  mrr = (1.0 / found_rank) if found_rank is not None else 0.0
  return recall, mrr


def evaluate_query(retriever, query_text, target_article, k):
  """Evaluates a single query across BM25, Dense, and Hybrid modes."""
  modes = ["bm25", "dense", "hybrid"]
  results = {}

  for mode in modes:
    if hasattr(retriever, "retrieve_by_mode"):
      items = retriever.retrieve_by_mode(query_text, mode=mode)
    else:
      items = retriever.retrieve(query_text)

    rec, mrr = compute_metrics_for_passages(items, target_article, k)
    results[mode] = {"recall": rec, "mrr": mrr}

  return results


def main():
  parser = argparse.ArgumentParser(
      description="Fast Retrieval-Only Metrics (Recall@K & MRR@K)"
  )
  parser.add_argument(
      "--index",
      type=int,
      default=None,
      help="Specific QA index to run. If omitted, runs whole dataset.",
  )
  parser.add_argument("--data_dir", type=str, default="data")
  args = parser.parse_args()

  data_path = Path(args.data_dir)
  corpus_file = (
      data_path / "corpus.json"
      if (data_path / "corpus.json").exists()
      else data_path / "document_corpus.json"
  )
  qa_file = (
      data_path / "qa.json"
      if (data_path / "qa.json").exists()
      else data_path / "qa_pairs.json"
  )

  config = ExperimentConfig(
      corpus_path=corpus_file,
      qa_path=qa_file,
      results_dir=Path("results"),
  )

  print("Loading corpus and dataset...")
  chunks = load_corpus(config.corpus_path)
  qa_records = load_qa(config.qa_path)

  print("Initializing Retriever...")
  retriever = HybridRetriever(
      chunks=chunks,
      dense_model=config.dense_model,
      reranker_model=config.reranker_model,
      dense_k=config.dense_k,
      bm25_k=config.bm25_k,
      final_k=config.final_k,
      batch_size=config.batch_size,
      device=config.device,
  )

  k = config.final_k

  if args.index is not None:
    # Single Query Execution
    record = qa_records[args.index]
    query = record["query"]
    target_art = record.get("article_reference", "")

    print(f"\n--- Testing Query Index {args.index} ---")
    print(f"Query: {query}")
    print(f"Target Article: {target_art}\n")

    t_start = time.time()
    res = evaluate_query(retriever, query, target_art, k)
    elapsed = time.time() - t_start

    print("=" * 55)
    print(f"SINGLE QUERY RESULTS (Latency: {elapsed:.3f}s)")
    print("=" * 55)
    for mode in ["bm25", "dense", "hybrid"]:
      print(
          f"[{mode.upper():<6}] Recall@{k}: {res[mode]['recall']:.4f} | MRR@{k}:"
          f" {res[mode]['mrr']:.4f}"
      )
    print("=" * 55)

  else:
    # Entire Dataset Execution (1000+ Queries)
    print(
        f"\n--- Running Evaluation on Entire Dataset ({len(qa_records)} queries)"
        " ---"
    )
    totals = {
        m: {"recall": 0.0, "mrr": 0.0} for m in ["bm25", "dense", "hybrid"]
    }

    t_start = time.time()
    for record in tqdm(qa_records, desc="Evaluating Queries"):
      query = record["query"]
      target_art = record.get("article_reference", "")

      res = evaluate_query(retriever, query, target_art, k)
      for mode in ["bm25", "dense", "hybrid"]:
        totals[mode]["recall"] += res[mode]["recall"]
        totals[mode]["mrr"] += res[mode]["mrr"]

    elapsed = time.time() - t_start
    num_q = len(qa_records)

    print("\n" + "=" * 60)
    print(f"OVERALL DATASET RESULTS ({num_q} Queries | Total Time:"
          f" {elapsed:.2f}s)")
    print("=" * 60)
    for mode in ["bm25", "dense", "hybrid"]:
      avg_rec = totals[mode]["recall"] / num_q
      avg_mrr = totals[mode]["mrr"] / num_q
      print(
          f"Mode: {mode.upper():<6} | Mean Recall@{k}: {avg_rec:.4f} | Mean"
          f" MRR@{k}: {avg_mrr:.4f}"
      )
    print("=" * 60)


if __name__ == "__main__":
  main()