import json
import os
from typing import Any

DEFAULT_SEARCH_LIMIT = 5
DEFAULT_CHUNK_SIZE = 200
SCORE_PRECISION = 3
DEFAULT_CHUNK_OVERLAP = 1
DEFAULT_SEMANTIC_CHUNK_SIZE = 4
DOCUMENT_PREVIEW_LENGTH = 100

BM25_K1 = 1.5
BM25_B = 0.75

DEFAULT_HYBRID_ALPHA = 0.5
DEFAULT_RRF_K = 60
RERANK_POOL_MULTIPLIER = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords(path: str = STOPWORDS_PATH) -> list[str]:
    with open(path) as file:
        return file.read().splitlines()


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score, max_score = min(scores), max(scores)

    if max_score == min_score:
        return [1.0 for _ in scores]

    return [(score - min_score) / (max_score - min_score) for score in scores]


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank: int, k: int = DEFAULT_RRF_K) -> float:
    return 1.0 / (k + rank)


def format_search_result(
    doc_id: int, title: str, document: str, score: float, **metadata: Any
) -> dict[str, Any]:
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }
