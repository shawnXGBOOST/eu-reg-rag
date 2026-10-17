"""Query the index from the command line.

Usage:
  python -m eu_reg_rag.search "When is a DPIA required?"
  python -m eu_reg_rag.search "biometric categorisation" --mode bm25 -k 8
  python -m eu_reg_rag.search "max fines for prohibited AI" --rerank        # + cross-encoder
  python -m eu_reg_rag.search "what is a recital about" --include-recitals
"""

import argparse

from eu_reg_rag.rerank import cross_encoder_scorer
from eu_reg_rag.retrieval import (
    BINDING_KINDS,
    HybridRetriever,
    get_client,
    load_chunks,
    sentence_transformer_encoder,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid")
    # Defaults follow eval/results.md: articles-first wins; the reranker adds ~16 s/query on CPU
    # for no measurable gain, so it is opt-in.
    p.add_argument("--rerank", action="store_true", help="apply the cross-encoder (slow on CPU)")
    p.add_argument("--include-recitals", action="store_true", help="don't restrict to articles")
    p.add_argument(
        "--per-article", type=int, default=1, help="max chunks per article (eval: 1 is best)"
    )
    p.add_argument("-k", type=int, default=5)
    args = p.parse_args()

    reranker = cross_encoder_scorer() if args.rerank else None
    kinds = None if args.include_recitals else BINDING_KINDS
    retriever = HybridRetriever(
        load_chunks(), sentence_transformer_encoder(), get_client(), reranker
    )
    results = retriever.search(
        args.query,
        k=args.k,
        mode=args.mode,
        rerank=args.rerank,
        kinds=kinds,
        max_per_unit=args.per_article,
    )

    scope = "all" if args.include_recitals else "articles-first"
    print(f"\nQuery: {args.query}   [mode={args.mode}, {scope}, rerank={args.rerank}]\n")
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
