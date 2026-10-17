# EU Reg RAG

Production-style RAG + agent assistant for the **EU AI Act** and **GDPR**: hybrid retrieval, reranking, cited answers, an evaluation harness, and a LangGraph agent for multi-hop questions.

> Work in progress.

## Quickstart

```bash
ollama pull qwen3:8b
docker compose up -d qdrant
pip install -e ".[dev]"
uvicorn eu_reg_rag.api:app --reload
# open http://localhost:8000/health
```
