import time

from .query_enhancement import LLM_MODEL, client

RERANK_SLEEP_SECONDS = 3


def _score_result(query: str, doc: dict) -> float:
    prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    content = (response.choices[0].message.content or "").strip()
    try:
        return float(content)
    except ValueError:
        return 0.0


def rerank_individual(query: str, results: list[dict]) -> list[dict]:
    reranked = []
    for i, res in enumerate(results):
        if i > 0:
            time.sleep(RERANK_SLEEP_SECONDS)
        score = _score_result(query, res)
        reranked.append({**res, "rerank_score": score})

    reranked.sort(key=lambda res: res["rerank_score"], reverse=True)
    return reranked
