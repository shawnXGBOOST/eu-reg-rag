"""Retrieval evaluation: compares BM25, dense, hybrid and hybrid+rerank on a labelled question set.

Gold labels are at ARTICLE level (e.g. "aiact-art-99"): a hit is any retrieved chunk from a gold
article, regardless of which paragraph-chunk it is. That keeps labels stable when chunking changes.

Metrics
  Hit@k  - share of questions with at least one gold article in the top k
  MRR@10 - mean of 1/rank of the first gold hit (0 if none in top 10)

Usage:  python -m eu_reg_rag.evaluate            (writes eval/results.md)
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

BINDING = ("article", "annex")  # articles-first: recitals excluded from results
CONFIGS = {
    "BM25": {"mode": "bm25", "rerank": False},
    "Dense": {"mode": "dense", "rerank": False},
    "Hybrid (RRF)": {"mode": "hybrid", "rerank": False},
    "Hybrid + rerank": {"mode": "hybrid", "rerank": True},
    "Hybrid, articles-first": {"mode": "hybrid", "rerank": False, "kinds": BINDING},
    "Hybrid + rerank, articles-first": {"mode": "hybrid", "rerank": True, "kinds": BINDING},
    "Hybrid, articles-first, 1 per article": {
        "mode": "hybrid",
        "rerank": False,
        "kinds": BINDING,
        "max_per_unit": 1,
    },
}
DEFAULT_CONFIGS = [  # fast set; add the rerank configs explicitly (~16 s/query on CPU)
    "BM25",
    "Dense",
    "Hybrid (RRF)",
    "Hybrid, articles-first",
    "Hybrid, articles-first, 1 per article",
]
DIFFICULTIES = ["easy", "paraphrase", "hard"]


def article_key(chunk: dict) -> str:
    slug = "gdpr" if chunk["regulation"] == "GDPR" else "aiact"
    return f"{slug}-{chunk['kind'][:3]}-{chunk['number']}"


def first_hit_rank(retrieved: list[dict], gold: list[str]) -> int | None:
    """1-based rank of the first chunk belonging to a gold article."""
    for rank, chunk in enumerate(retrieved, start=1):
        if article_key(chunk) in gold:
            return rank
    return None


@dataclass
class QResult:
    qid: str
    difficulty: str
    question: str
    gold: list[str]
    rank: int | None
    top: list[str]
    latency_ms: float


def summarize(results: list[QResult]) -> dict:
    n = len(results) or 1

    def hit(k: int) -> float:
        return sum(r.rank is not None and r.rank <= k for r in results) / n

    mrr = sum(1 / r.rank for r in results if r.rank and r.rank <= 10) / n
    lat = [r.latency_ms for r in results] or [0.0]
    return {
        "n": len(results),
        "hit@1": hit(1),
        "hit@5": hit(5),
        "mrr@10": mrr,
        "latency_ms_median": statistics.median(lat),
    }


def load_questions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run(
    retriever, questions: list[dict], mode: str, rerank: bool, kinds=None, max_per_unit=None
) -> list[QResult]:
    out = []
    for q in questions:
        t0 = time.perf_counter()
        res = retriever.search(
            q["question"], k=10, mode=mode, rerank=rerank, kinds=kinds, max_per_unit=max_per_unit
        )
        ms = (time.perf_counter() - t0) * 1000
        chunks = [r.chunk for r in res]
        out.append(
            QResult(
                q["id"],
                q["difficulty"],
                q["question"],
                q["gold"],
                first_hit_rank(chunks, q["gold"]),
                [article_key(c) for c in chunks[:5]],
                ms,
            )
        )
    return out


def to_markdown(all_results: dict[str, list[QResult]]) -> str:
    lines = [
        "## Retrieval results",
        "",
        "| Config | Hit@1 | Hit@5 | MRR@10 | Median latency |",
        "|---|---|---|---|---|",
    ]
    for name, res in all_results.items():
        s = summarize(res)
        lines.append(
            f"| {name} | {s['hit@1']:.2f} | {s['hit@5']:.2f} | {s['mrr@10']:.3f} "
            f"| {s['latency_ms_median']:.0f} ms |"
        )

    lines += [
        "",
        "### Hit@5 by question difficulty",
        "",
        "| Config | "
        + " | ".join(
            f"{d} (n={sum(r.difficulty == d for r in next(iter(all_results.values())))})"
            for d in DIFFICULTIES
        )
        + " |",
        "|---|" + "---|" * len(DIFFICULTIES),
    ]
    for name, res in all_results.items():
        cells = [
            f"{summarize([r for r in res if r.difficulty == d])['hit@5']:.2f}" for d in DIFFICULTIES
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    best = list(all_results)[-1]
    misses = [r for r in all_results[best] if r.rank is None or r.rank > 5]
    lines += ["", f"### Misses for '{best}' (gold not in top 5)", ""]
    if not misses:
        lines.append("None.")
    for r in misses:
        lines.append(
            f"- **{r.qid}** ({r.difficulty}) {r.question}  \n"
            f"  gold: {', '.join(r.gold)} | got: {', '.join(r.top)} | rank: {r.rank}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default="eval/questions.jsonl")
    p.add_argument("--out", default="eval/results.md")
    p.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS, help=f"any of: {list(CONFIGS)}")
    args = p.parse_args()

    from eu_reg_rag.rerank import cross_encoder_scorer
    from eu_reg_rag.retrieval import (
        HybridRetriever,
        get_client,
        load_chunks,
        sentence_transformer_encoder,
    )

    questions = load_questions(args.questions)
    needs_rerank = any(CONFIGS[c]["rerank"] for c in args.configs)
    retriever = HybridRetriever(
        load_chunks(),
        sentence_transformer_encoder(),
        get_client(),
        cross_encoder_scorer() if needs_rerank else None,
    )
    retriever.search("warm-up", k=1)  # load models before timing

    all_results = {}
    for name in args.configs:
        print(f"Running {name} on {len(questions)} questions ...")
        all_results[name] = run(retriever, questions, **CONFIGS[name])

    md = to_markdown(all_results)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md, encoding="utf-8")
    print("\n" + md + f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
