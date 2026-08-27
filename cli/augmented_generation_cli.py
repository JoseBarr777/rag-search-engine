#!/usr/bin/env python3

import argparse

from lib.augmented_generation import citations_command, rag_command, summarize_command
from lib.search_utils import DEFAULT_SEARCH_LIMIT


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize search results for a query"
    )
    summarize_parser.add_argument("query", type=str, help="Search query to summarize")
    summarize_parser.add_argument(
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of results to return"
    )

    citations_parser = subparsers.add_parser(
        "citations", help="Answer a query with cited sources"
    )
    citations_parser.add_argument("query", type=str, help="Search query to answer")
    citations_parser.add_argument(
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of results to return"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            result = rag_command(args.query)
            if "error" in result:
                print(result["error"])
            else:
                print("Search Results:")
                for res in result["results"]:
                    print(f"- {res['title']}")
                print("\nRAG Response:")
                print(result["answer"])
        case "summarize":
            result = summarize_command(args.query, args.limit)
            if "error" in result:
                print(result["error"])
            else:
                print("Search Results:")
                for res in result["results"]:
                    print(f"  - {res['title']}")
                print("\nLLM Summary:")
                print(result["summary"])
        case "citations":
            result = citations_command(args.query, args.limit)
            if "error" in result:
                print(result["error"])
            else:
                print("Search Results:")
                for res in result["results"]:
                    print(f"  - {res['title']}")
                print("\nLLM Answer:")
                print(result["answer"])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
