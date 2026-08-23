import json

from .hybrid_search import HybridSearch
from .query_enhancement import LLM_MODEL, client
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


def evaluate_relevance(query: str, results: list[dict]) -> list[int]:
    formatted_results = [
        f"{i}. {res['title']}: {res['document']}" for i, res in enumerate(results, 1)
    ]
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    content = (response.choices[0].message.content or "").strip()
    scores = json.loads(content)

    if len(scores) != len(results):
        raise ValueError(
            f"LLM response parsing error. Expected {len(results)} scores, "
            f"got {len(scores)}. Response: {scores}"
        )

    return scores


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
