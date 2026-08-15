import os
from collections import defaultdict

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import (
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_RRF_K,
    DEFAULT_SEARCH_LIMIT,
    format_search_result,
    load_movies,
)

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        total_docs = len(self.documents)
        bm25_results = self._bm25_search(query, total_docs)
        semantic_results = self.semantic_search.search_chunks(query, total_docs)

        bm25_scores = self.__normalize_scores(bm25_results)
        semantic_scores = self.__normalize_scores(semantic_results)

        doc_lookup = {r["id"]: r for r in semantic_results}
        doc_lookup.update({r["id"]: r for r in bm25_results})

        combined_scores: dict[int, float] = {}
        for doc_id in doc_lookup:
            combined_scores[doc_id] = (
                alpha * bm25_scores.get(doc_id, 0.0)
                + (1 - alpha) * semantic_scores.get(doc_id, 0.0)
            )

        return self.__build_results(combined_scores, doc_lookup, limit)

    def rrf_search(self, query: str, k: int = DEFAULT_RRF_K, limit: int = 10) -> list[dict]:
        total_docs = len(self.documents)
        bm25_results = self._bm25_search(query, total_docs)
        semantic_results = self.semantic_search.search_chunks(query, total_docs)

        doc_lookup = {r["id"]: r for r in semantic_results}
        doc_lookup.update({r["id"]: r for r in bm25_results})

        rrf_scores: dict[int, float] = defaultdict(float)
        for rank, result in enumerate(bm25_results, start=1):
            rrf_scores[result["id"]] += 1.0 / (k + rank)
        for rank, result in enumerate(semantic_results, start=1):
            rrf_scores[result["id"]] += 1.0 / (k + rank)

        return self.__build_results(rrf_scores, doc_lookup, limit)

    def __normalize_scores(self, results: list[dict]) -> dict[int, float]:
        if not results:
            return {}

        scores = [r["score"] for r in results]
        min_score, max_score = min(scores), max(scores)

        if max_score == min_score:
            return {r["id"]: 1.0 for r in results}

        return {
            r["id"]: (r["score"] - min_score) / (max_score - min_score)
            for r in results
        }

    def __build_results(
        self, scores: dict[int, float], doc_lookup: dict[int, dict], limit: int
    ) -> list[dict]:
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]

        results = []
        for doc_id, score in sorted_ids:
            doc = doc_lookup[doc_id]
            results.append(format_search_result(
                doc_id=doc_id,
                title=doc["title"],
                document=doc["document"],
                score=score,
            ))
        return results


def build_command() -> None:
    documents = load_movies()
    HybridSearch(documents)


def weighted_search_command(
    query: str, alpha: float = DEFAULT_HYBRID_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[dict]:
    documents = load_movies()
    hybrid = HybridSearch(documents)
    return hybrid.weighted_search(query, alpha, limit)


def rrf_search_command(
    query: str, k: int = DEFAULT_RRF_K, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[dict]:
    documents = load_movies()
    hybrid = HybridSearch(documents)
    return hybrid.rrf_search(query, k, limit)
