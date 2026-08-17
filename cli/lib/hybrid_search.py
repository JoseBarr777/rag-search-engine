import os
from typing import Literal, TypedDict

from .keyword_search import InvertedIndex
from .query_enhancement import enhance_query
from .reranking import rerank_batch, rerank_individual
from .semantic_search import ChunkedSemanticSearch
from .search_utils import (
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_RRF_K,
    DEFAULT_SEARCH_LIMIT,
    DOCUMENT_PREVIEW_LENGTH,
    RERANK_POOL_MULTIPLIER,
    SCORE_PRECISION,
    format_search_result,
    hybrid_score,
    load_movies,
    normalize_scores,
    rrf_score,
)

SEARCH_POOL_MULTIPLIER = 500


class RRFSearchCommandResult(TypedDict):
    original_query: str
    enhanced_query: str | None
    enhance_method: Literal["spell", "rewrite", "expand"] | None
    query: str
    k: int
    rerank_method: Literal["individual", "batch"] | None
    results: list[dict]


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
        pool_size = limit * SEARCH_POOL_MULTIPLIER
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

    def rrf_search(
        self, query: str, k: int = DEFAULT_RRF_K, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[dict]:
        pool_size = limit * SEARCH_POOL_MULTIPLIER
        bm25_results = self._bm25_search(query, pool_size)
        semantic_results = self.semantic_search.search_chunks(query, pool_size)

        doc_ranks: dict[int, dict] = {}
        for rank, result in enumerate(bm25_results, start=1):
            entry = doc_ranks.setdefault(
                result["id"], {"document": self.document_map[result["id"]]}
            )
            entry["bm25_rank"] = rank
        for rank, result in enumerate(semantic_results, start=1):
            entry = doc_ranks.setdefault(
                result["id"], {"document": self.document_map[result["id"]]}
            )
            entry["semantic_rank"] = rank

        for entry in doc_ranks.values():
            entry["rrf_score"] = (
                rrf_score(entry["bm25_rank"], k) if "bm25_rank" in entry else 0.0
            ) + (
                rrf_score(entry["semantic_rank"], k) if "semantic_rank" in entry else 0.0
            )

        sorted_entries = sorted(
            doc_ranks.values(), key=lambda entry: entry["rrf_score"], reverse=True
        )

        return [
            format_search_result(
                doc_id=entry["document"]["id"],
                title=entry["document"]["title"],
                document=entry["document"]["description"][:DOCUMENT_PREVIEW_LENGTH],
                score=entry["rrf_score"],
                bm25_rank=entry.get("bm25_rank"),
                semantic_rank=entry.get("semantic_rank"),
            )
            for entry in sorted_entries
        ]


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
    query: str,
    k: int = DEFAULT_RRF_K,
    enhance: Literal["spell", "rewrite", "expand"] | None = None,
    limit: int = DEFAULT_SEARCH_LIMIT,
    rerank_method: Literal["individual", "batch"] | None = None,
) -> RRFSearchCommandResult:
    original_query = query
    enhanced_query = None
    if enhance:
        enhanced_query = enhance_query(query, method=enhance)
        query = enhanced_query

    documents = load_movies()
    hybrid = HybridSearch(documents)

    search_limit = limit * RERANK_POOL_MULTIPLIER if rerank_method else limit
    results = hybrid.rrf_search(query, k, search_limit)[:search_limit]

    if rerank_method == "individual":
        results = rerank_individual(query, results)
    elif rerank_method == "batch":
        results = rerank_batch(query, results)

    results = results[:limit]

    return {
        "original_query": original_query,
        "enhanced_query": enhanced_query,
        "enhance_method": enhance,
        "query": query,
        "k": k,
        "rerank_method": rerank_method,
        "results": results,
    }
