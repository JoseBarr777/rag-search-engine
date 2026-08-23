from .hybrid_search import HybridSearch
from .search_utils import DEFAULT_RRF_K, load_golden_dataset, load_movies


def precision_at_k(retrieved_docs: list[str], relevant_docs: set[str], k: int) -> float:
    top_k = retrieved_docs[:k]
    relevant_count = sum(1 for doc in top_k if doc in relevant_docs)
    return relevant_count / k


def recall_at_k(retrieved_docs: list[str], relevant_docs: set[str], k: int) -> float:
    top_k = retrieved_docs[:k]
    relevant_count = sum(1 for doc in top_k if doc in relevant_docs)
    return relevant_count / len(relevant_docs)


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_command(limit: int = 5) -> dict:
    movies = load_movies()
    golden_data = load_golden_dataset()
    test_cases = golden_data["test_cases"]

    hybrid_search = HybridSearch(movies)

    results_by_query = {}
    for test_case in test_cases:
        query = test_case["query"]
        relevant_docs = test_case["relevant_docs"]

        search_results = hybrid_search.rrf_search(query, k=DEFAULT_RRF_K, limit=limit)
        retrieved_docs = [result["title"] for result in search_results]

        relevant_set = set(relevant_docs)
        precision = precision_at_k(retrieved_docs, relevant_set, limit)
        recall = recall_at_k(retrieved_docs, relevant_set, limit)
        f1 = f1_score(precision, recall)

        results_by_query[query] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "retrieved": retrieved_docs[:limit],
            "relevant": relevant_docs,
        }

    return {
        "test_cases_count": len(test_cases),
        "limit": limit,
        "results": results_by_query,
    }
