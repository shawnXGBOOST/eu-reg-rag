"""Query the index from the command line.

Usage:
  python -m eu_reg_rag.search "When is a DPIA required?"
  python -m eu_reg_rag.search "biometric categorisation" --mode bm25 -k 8
  python -m eu_reg_rag.search "max fines for prohibited AI" --no-rerank   # before/after
"""

import argparse

from eu_reg_rag.rerank import cross_encoder_scorer
from eu_reg_rag.retrieval import (
    HybridRetriever,
    get_client,
    load_chunks,
    sentence_transformer_encoder,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid")
    p.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder")
    p.add_argument("-k", type=int, default=5)
    args = p.parse_args()

    reranker = None if args.no_rerank else cross_encoder_scorer()
    retriever = HybridRetriever(
        load_chunks(), sentence_transformer_encoder(), get_client(), reranker
    )
    results = retriever.search(args.query, k=args.k, mode=args.mode, rerank=not args.no_rerank)

    print(f"\nQuery: {args.query}   [mode={args.mode}, rerank={not args.no_rerank}]\n")
    for i, r in enumerate(results, start=1):
        c = r.chunk
        para = f" ({c['paragraphs']})" if c["paragraphs"] else ""
        first_line = c["text"].split("\n", 1)[0]
        snippet = c["text"].split("\n", 1)[-1][:160].replace("\n", " ")
        print(f"{i}. {first_line}{para}")
        print(f"   score={r.score:.4f}  dense_rank={r.dense_rank}  bm25_rank={r.bm25_rank}")
        print(f"   {snippet}...\n")


if __name__ == "__main__":
    main()
