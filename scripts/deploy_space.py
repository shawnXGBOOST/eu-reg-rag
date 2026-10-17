"""Create/update the Hugging Face Space for the live demo.

Prerequisites
  pip install huggingface_hub
  set HF_TOKEN=<token with WRITE access>       (your Hugging Face account token)
  set LLM_API_KEY=<key of the LLM provider>    (defaults to HF_TOKEN, which works for the HF router
                                                only if the account has inference credits)

Usage
  python scripts/deploy_space.py --repo YOUR_USER/eu-reg-rag
  python scripts/deploy_space.py --repo YOUR_USER/eu-reg-rag \
      --base-url https://api.groq.com/openai/v1 --model qwen/qwen3-32b

The Space gets the LLM key as a secret (never in the repo) and the model/base URL as variables.
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
FRONT_MATTER = """---
title: EU Reg RAG
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Cited answers over the EU AI Act and GDPR
---

"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="Space id, e.g. yourname/eu-reg-rag")
    p.add_argument("--base-url", default="https://router.huggingface.co/v1")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--rate-limit", default="20", help="questions per IP per hour")
    args = p.parse_args()

    token = os.environ["HF_TOKEN"]
    llm_key = os.environ.get("LLM_API_KEY") or token
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="space", space_sdk="docker", exist_ok=True)
    api.add_space_secret(args.repo, "LLM_API_KEY", llm_key)
    api.add_space_variable(args.repo, "LLM_BASE_URL", args.base_url)
    api.add_space_variable(args.repo, "LLM_MODEL", args.model)
    api.add_space_variable(args.repo, "RATE_LIMIT_PER_HOUR", args.rate_limit)

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for name in ("Dockerfile", ".dockerignore", "pyproject.toml"):
            shutil.copy(ROOT / name, stage / name)
        shutil.copytree(
            ROOT / "src", stage / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        if (ROOT / "docs").exists():  # README screenshots
            shutil.copytree(ROOT / "docs", stage / "docs")
        (stage / "data" / "processed").mkdir(parents=True)
        shutil.copy(ROOT / "data/processed/chunks.jsonl", stage / "data/processed/chunks.jsonl")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        (stage / "README.md").write_text(FRONT_MATTER + readme, encoding="utf-8")
        api.upload_folder(
            repo_id=args.repo, repo_type="space", folder_path=stage, commit_message="Deploy"
        )
    print(f"Done. Build logs and the app: https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
