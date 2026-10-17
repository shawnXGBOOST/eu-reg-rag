"""Ask a question from the command line.

Usage:  python -m eu_reg_rag.ask "What are the maximum fines for prohibited AI practices?"
"""

import sys

from eu_reg_rag.pipeline import RAGPipeline


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m eu_reg_rag.ask "your question"')
    question = " ".join(sys.argv[1:])
    out = RAGPipeline.from_settings().ask(question)

    print(f"\nQ: {question}\n")
    print(out["answer"] + "\n")
    if out["sources"]:
        print("Sources:")
        for s in out["sources"]:
            print(f"  - {s['label']} - {s['title']}")
    if out["invalid_tags"]:
        print(f"\n(warning: model produced invalid citations, removed: {out['invalid_tags']})")
    t = out["timing_ms"]
    print(f"\n[retrieval {t['retrieval']} ms | generation {t['generation']} ms]")


if __name__ == "__main__":
    main()
