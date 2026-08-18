#!/usr/bin/env python3

import argparse

from lib.hybrid_search import build_command, rrf_search_command, weighted_search_command
from lib.search_utils import (
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_RRF_K,
    DEFAULT_SEARCH_LIMIT,
    normalize_scores,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the BM25 index and chunk embeddings")

    search_parser = subparsers.add_parser(
        "search", help="Search using weighted hybrid (BM25 + semantic) scoring"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--alpha", type=float, default=DEFAULT_HYBRID_ALPHA,
        help="Weight given to BM25 score vs semantic score (0.0-1.0)",
    )
    search_parser.add_argument(
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of results to return"
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Search using weighted hybrid (BM25 + semantic) scoring"
    )
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument(
        "--alpha", type=float, default=DEFAULT_HYBRID_ALPHA,
        help="Weight given to BM25 score vs semantic score (0.0-1.0)",
    )
    weighted_search_parser.add_argument(
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of results to return"
    )

    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="Search using Reciprocal Rank Fusion of BM25 and semantic results"
    )
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument(
        "-k", type=int, default=DEFAULT_RRF_K, help="RRF k constant (higher dampens rank influence)"
    )
    rrf_search_parser.add_argument(
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of results to return"
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="LLM re-ranking method to apply to RRF results",
    )

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores using min-max normalization"
    )
    normalize_parser.add_argument(
        "scores", type=float, nargs="*", help="Scores to normalize"
    )

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building hybrid search index...")
            build_command()
            print("Hybrid search index built successfully.")
        case "search":
            print("Searching for:", args.query)
            results = weighted_search_command(args.query, args.alpha, args.limit)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']} - Score: {res['score']:.3f}")
        case "weighted-search":
            results = weighted_search_command(args.query, args.alpha, args.limit)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
                print(f"  Hybrid Score: {res['score']:.3f}")
                print(
                    f"  BM25: {res['metadata']['bm25_score']:.3f}, "
                    f"Semantic: {res['metadata']['semantic_score']:.3f}"
                )
                print(f"  {res['document']}...")
        case "rrf-search":
            result = rrf_search_command(
                args.query, args.k, args.enhance, args.limit, args.rerank_method
            )
            if result["enhanced_query"]:
                print(
                    f"Enhanced query ({result['enhance_method']}): "
                    f"'{result['original_query']}' -> '{result['enhanced_query']}'\n"
                )
            if result["rerank_method"]:
                print(
                    f"Re-ranking top {result['rerank_pool_size']} results using "
                    f"{result['rerank_method']} method...\n"
                )
            print(
                f"Reciprocal Rank Fusion Results for '{result['query']}' "
                f"(k={result['k']}):\n"
            )
            for i, res in enumerate(result["results"], 1):
                if i > 1:
                    print()
                bm25_rank = res["metadata"]["bm25_rank"] or "N/A"
                semantic_rank = res["metadata"]["semantic_rank"] or "N/A"
                print(f"{i}. {res['title']}")
                if "rerank_score" in res:
                    print(f"   Re-rank Score: {res['rerank_score']:.3f}/10")
                if "rerank_rank" in res:
                    print(f"   Re-rank Rank: {res['rerank_rank']}")
                if "cross_encoder_score" in res:
                    print(f"   Cross Encoder Score: {res['cross_encoder_score']:.3f}")
                print(f"   RRF Score: {res['score']:.3f}")
                print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
                print(f"   {res['document']}...")
        case "normalize":
            for score in normalize_scores(args.scores):
                print(f"* {score:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
