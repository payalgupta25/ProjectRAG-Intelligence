"""Dense + lexical candidate retrieval and cross-encoder reranking."""

import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from .data import Chunk


def tokenize(text: str) -> list[str]:
    """English lexical tokenization for the Constitution corpus."""
    return re.findall(r"[A-Za-z0-9]+", text.lower())


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        dense_model: str,
        reranker_model: str,
        dense_k: int,
        bm25_k: int,
        final_k: int,
        batch_size: int,
        device: str | None,
    ) -> None:
        self.chunks = chunks
        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.final_k = final_k
        self.embedder = SentenceTransformer(dense_model, device=device)
        self.embeddings = self.embedder.encode(
            [chunk.text for chunk in chunks],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        self.bm25 = BM25Okapi([tokenize(chunk.text) for chunk in chunks])
        self.reranker = CrossEncoder(reranker_model, device=device)

    def _dense_candidates(self, query: str) -> list[int]:
        query_embedding = self.embedder.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        scores = self.embeddings @ query_embedding
        indices = np.argsort(-scores)[: self.dense_k]
        return [int(index) for index in indices]

    def _bm25_candidates(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        indices = np.argsort(-scores)[: self.bm25_k]
        return [int(index) for index in indices]

    def retrieve(self, query: str) -> list[dict]:
        """Merge candidates by text identity, then return the top reranked passages."""
        RELEVANCE_THRESHOLD = 0.05  # Threshold for bge-reranker-base
        
        candidate_indices = self._bm25_candidates(query) + self._dense_candidates(query)
        seen_texts: set[str] = set()
        candidates: list[Chunk] = []
        for index in candidate_indices:
            chunk = self.chunks[index]
            if chunk.text not in seen_texts:
                candidates.append(chunk)
                seen_texts.add(chunk.text)

        scores = self.reranker.predict([(query, chunk.text) for chunk in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda item: float(item[1]), reverse=True)[: self.final_k]
        
        # Filter out out-of-domain / irrelevant queries
        if not ranked or float(ranked[0][1]) < RELEVANCE_THRESHOLD:
            return []
            
        return [
            {
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "article_reference": chunk.article_reference,
                "title": chunk.title,
                "text": chunk.text,
                "reranker_score": float(score),
            }
            for rank, (chunk, score) in enumerate(ranked, start=1)
        ]