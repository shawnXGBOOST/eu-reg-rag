"""End-to-end answer evaluation of the agent (needs Ollama running).

Positives (eval/questions.jsonl, gold articles known):
  Answer rate        - share answered (not refused)
  Citation accuracy  - of answered, share citing at least one gold article
  Invalid citations  - share of answers where the model cited a tag that doesn't exist
  Rescued by retry   - answered only on attempt 2 (what the fallback rewrite buys)
Negatives (eval/negatives.jsonl, off-topic or near-miss questions):
  Refusal accuracy   - share correctly refused

Usage:
  python -m eu_reg_rag.evaluate_answers              # all ~57 questions, ~30-45 min on CPU
  python -m eu_reg_rag.evaluate_answers --limit 10   # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def chunk_article_key(chunk_id: str) -> str:
    """'aiact-art-99-1' -> 'aiact-art-99'; 'gdpr-art-22' -> 'gdpr-art-22'."""
    return "-".join(chunk_id.split("-")[:3])


def score(positives: list[dict], negatives: list[dict]) -> dict:
    answered = [r for r in positives if r["supported"]]
    n_pos, n_ans, n_neg = len(positives) or 1, len(answered) or 1, len(negatives) or 1
    times = [r["seconds"] for r in positives + negatives] or [0.0]
    return {
        "n_pos": len(positives),
        "n_neg": len(negatives),
        "answer_rate": len(answered) / n_pos,
        "citation_accuracy": sum(r["gold_cited"] for r in answered) / n_ans,
        "invalid_citation_rate": sum(bool(r["invalid_tags"]) for r in positives) / n_pos,
        "rescued_by_retry": sum(r["supported"] and r["attempts"] > 1 for r in positives),
        "refusal_accuracy": sum(not r["supported"] for r in negatives) / n_neg,
        "median_seconds": statistics.median(times),
    }


def to_markdown(s: dict, positives: list[dict], negatives: list[dict], strategy: str) -> str:
    lines = [
        f"## Answer quality (agent, strategy={strategy})",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Answer rate (n={s['n_pos']}) | {s['answer_rate']:.2f} |",
        f"| Citation accuracy (cites a gold article) | {s['citation_accuracy']:.2f} |",
        f"| Answers with invalid citations | {s['invalid_citation_rate']:.2f} |",
        f"| Answered only after retry | {s['rescued_by_retry']} |",
        f"| Refusal accuracy on off-topic (n={s['n_neg']}) | {s['refusal_accuracy']:.2f} |",
        f"| Median time per question | {s['median_seconds']:.1f} s |",
        "",
        "### Problems",
        "",
    ]
    problems = (
        [f"- **{r['id']}** refused: {r['question']}" for r in positives if not r["supported"]]
        + [
            f"- **{r['id']}** cited {', '.join(r['cited']) or 'nothing'} instead of "
            f"{', '.join(r['gold'])}: {r['question']}"
            for r in positives
            if r["supported"] and not r["gold_cited"]
        ]
        + [
            f"- **{r['id']}** ({r['kind']}) answered an out-of-scope question: {r['question']}"
            for r in negatives
            if r["supported"]
        ]
    )
    lines += problems or ["None."]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default="eval/questions.jsonl")
    p.add_argument("--negatives", default="eval/negatives.jsonl")
    p.add_argument("--strategy", choices=["fallback", "rewrite_first"], default="fallback")
    p.add_argument("--limit", type=int, default=None, help="only the first N of each set")
    p.add_argument("--out", default="eval/answer_results.md")
    args = p.parse_args()

    from eu_reg_rag.agent import RAGAgent
    from eu_reg_rag.evaluate import load_questions

    agent = RAGAgent.from_settings(args.strategy)
    pos_q = load_questions(args.questions)[: args.limit]
    neg_q = load_questions(args.negatives)[: args.limit]

    details: list[dict] = []
    positives, negatives = [], []
    total = len(pos_q) + len(neg_q)
    for i, q in enumerate(pos_q + neg_q, start=1):
        out = agent.ask(q["question"])
        cited = sorted({chunk_article_key(s["id"]) for s in out["sources"]})
        row = {
            "id": q["id"],
            "question": q["question"],
            "kind": q.get("kind", q.get("difficulty")),
            "gold": q.get("gold", []),
            "supported": out["supported"],
            "cited": cited,
            "gold_cited": bool(set(cited) & set(q.get("gold", []))),
            "invalid_tags": out["invalid_tags"],
            "attempts": out["attempts"],
            "seconds": out["timing_ms"]["total"] / 1000,
            "answer": out["answer"],
        }
        (positives if "gold" in q else negatives).append(row)
        details.append(row)
        mark = "answered" if row["supported"] else "refused"
        print(f"[{i}/{total}] {q['id']} {mark} in {row['seconds']:.0f}s  cited={cited}")

    s = score(positives, negatives)
    md = to_markdown(s, positives, negatives, args.strategy)
    Path(args.out).write_text(md, encoding="utf-8")
    with open(Path(args.out).with_suffix(".jsonl"), "w", encoding="utf-8") as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("\n" + md + f"\nSaved to {args.out} (+ per-question details in .jsonl)")


if __name__ == "__main__":
    main()
