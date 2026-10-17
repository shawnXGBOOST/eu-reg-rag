# Image for the hosted demo (Hugging Face Spaces, Docker SDK). Local use needs no Docker.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    LLM_BACKEND=openai \
    LLM_MODEL=Qwen/Qwen3-8B \
    RATE_LIMIT_PER_HOUR=20
WORKDIR /home/user/app

# CPU-only torch keeps the image small (~200 MB instead of ~2 GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user pyproject.toml .
COPY --chown=user src ./src
COPY --chown=user data/processed ./data/processed
RUN pip install --no-cache-dir .

# Bake the embedding model and the vector index into the image: no cold-start work
RUN python -m eu_reg_rag.index

EXPOSE 7860
CMD ["uvicorn", "eu_reg_rag.api:app", "--host", "0.0.0.0", "--port", "7860"]
