import os
from collections import defaultdict

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import (
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_RRF_K,
    DEFAULT_SEARCH_LIMIT,
    DOCUMENT_PREVIEW_LENGTH,
    SCORE_PRECISION,
    format_search_result,
    hybrid_score,
    load_movies,
    normalize_scores,
)

WEIGHTED_SEARCH_POOL_MULTIPLIER = 500

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(
        self, query: str, alpha: float = DEFAULT_HYBRID_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[dict]:
        pool_size = limit * WEIGHTED_SEARCH_POOL_MULTIPLIER
        bm25_results = self._bm25_search(query, pool_size)
        semantic_results = self.semantic_search.search_chunks(query, pool_size)

        bm25_scores = dict(zip(
            (r["id"] for r in bm25_results),
            normalize_scores([r["score"] for r in bm25_results]),
        ))
        semantic_scores = dict(zip(
            (r["id"] for r in semantic_results),
            normalize_scores([r["score"] for r in semantic_results]),
        ))

        scored_docs: dict[int, dict] = {}
        for doc_id in set(bm25_scores) | set(semantic_scores):
            bm25 = bm25_scores.get(doc_id, 0.0)
            semantic = semantic_scores.get(doc_id, 0.0)
            scored_docs[doc_id] = {
                "document": self.document_map[doc_id],
                "bm25_score": bm25,
                "semantic_score": semantic,
                "hybrid_score": hybrid_score(bm25, semantic, alpha),
            }

        sorted_docs = sorted(
            scored_docs.values(), key=lambda entry: entry["hybrid_score"], reverse=True
        )

        return [
            format_search_result(
                doc_id=entry["document"]["id"],
                title=entry["document"]["title"],
                document=entry["document"]["description"][:DOCUMENT_PREVIEW_LENGTH],
                score=entry["hybrid_score"],
                bm25_score=round(entry["bm25_score"], SCORE_PRECISION),
                semantic_score=round(entry["semantic_score"], SCORE_PRECISION),
            )
            for entry in sorted_docs
        ]

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
    return hybrid.weighted_search(query, alpha, limit)[:limit]


def rrf_search_command(
    query: str, k: int = DEFAULT_RRF_K, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[dict]:
    documents = load_movies()
    hybrid = HybridSearch(documents)
    return hybrid.rrf_search(query, k, limit)
