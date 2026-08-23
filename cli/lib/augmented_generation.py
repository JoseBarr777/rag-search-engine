from typing import TypedDict

from .hybrid_search import HybridSearch
from .query_enhancement import LLM_MODEL, client
from .search_utils import DEFAULT_RRF_K, DEFAULT_SEARCH_LIMIT, load_movies


class RAGCommandResult(TypedDict):
    query: str
    results: list[dict]
    answer: str


class SummarizeCommandResult(TypedDict):
    query: str
    results: list[dict]
    summary: str


def generate_answer(query: str, results: list[dict]) -> str:
    docs = "\n".join(
        f"{i}. {res['title']}: {res['document']}" for i, res in enumerate(results, 1)
    )
    prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}

Answer:"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return (response.choices[0].message.content or "").strip()


def rag_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> RAGCommandResult:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    results = hybrid_search.rrf_search(query, k=DEFAULT_RRF_K, limit=limit)[:limit]

    answer = generate_answer(query, results)

    return {
        "query": query,
        "results": results,
        "answer": answer,
    }


def generate_summary(query: str, results: list[dict]) -> str:
    docs = "\n".join(
        f"{i}. {res['title']}: {res['document']}" for i, res in enumerate(results, 1)
    )
    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Webflyx users. Webflyx is a movie streaming service.

Query: {query}

Search results:
{docs}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return (response.choices[0].message.content or "").strip()


def summarize_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> SummarizeCommandResult:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    results = hybrid_search.rrf_search(query, k=DEFAULT_RRF_K, limit=limit)[:limit]

    summary = generate_summary(query, results)

    return {
        "query": query,
        "results": results,
        "summary": summary,
    }
