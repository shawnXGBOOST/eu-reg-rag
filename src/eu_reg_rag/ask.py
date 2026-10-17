"""Ask a question from the command line (runs the LangGraph agent).

Usage:
  python -m eu_reg_rag.ask "Can a bank reject my loan purely by an algorithm?"
  python -m eu_reg_rag.ask "..." --strategy rewrite_first
"""

import argparse

from eu_reg_rag.agent import RAGAgent
from eu_reg_rag.config import settings


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="+")
    p.add_argument(
        "--strategy", choices=["fallback", "rewrite_first"], default=settings.agent_strategy
    )
    args = p.parse_args()
    question = " ".join(args.question)

    out = RAGAgent.from_settings(args.strategy).ask(question)

    print(f"\nQ: {question}\n")
    print(out["answer"] + "\n")
    if out["sources"]:
        print("Sources:")
        for s in out["sources"]:
            print(f"  - {s['label']} - {s['title']}")
    if out["invalid_tags"]:
        print(f"\n(warning: invalid citations removed: {out['invalid_tags']})")

    print(f"\nAgent trace ({out['strategy']}, {out['attempts']} attempt(s)):")
    for t in out["trace"]:
        if t["step"] == "rewrite":
            print(f"  rewrite  -> {t['queries']}")
        elif t["step"] == "retrieve":
            print(f"  retrieve -> {', '.join(t['found'])}")
        else:
            print(f"  generate -> attempt {t['attempt']}, supported={t['supported']}")
    print(f"\n[total {out['timing_ms']['total']} ms]")


if __name__ == "__main__":
    main()
