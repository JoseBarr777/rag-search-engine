import json
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


def rerank_batch(query: str, results: list[dict]) -> list[dict]:
    doc_list_str = "\n".join(
        f"{res['id']}: {res['title']} - {res['document']}" for res in results
    )
    prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    content = (response.choices[0].message.content or "").strip()
    ranked_ids = json.loads(content)

    rank_by_id = {doc_id: rank for rank, doc_id in enumerate(ranked_ids, start=1)}

    reranked = [
        {**res, "rerank_rank": rank_by_id[res["id"]]}
        for res in results
        if res["id"] in rank_by_id
    ]
    reranked.sort(key=lambda res: res["rerank_rank"])
    return reranked
