# EU Reg RAG

Ask questions about the **EU AI Act** and the **GDPR** and get answers that cite the exact article they come from. Hybrid retrieval, a LangGraph agent that rewrites failed searches, a streaming web UI, and an evaluation harness that is run in CI.

It runs fully locally: Qwen3 8B through Ollama, embedded Qdrant, no API keys, no Docker.

!\[UI](docs/demo.gif)

## What it does

* **Cited answers.** Every claim carries a legal-style citation (`GDPR Art. 22`, `AI Act Art. 5(1)(f)-(g)`). Citations are checked after generation: tags that do not match a retrieved source are removed and counted.
* **Refuses when the sources do not answer.** Off-topic and out-of-scope questions (other laws, other topics) get a refusal instead of a made-up answer.
* **Agent with a retry.** If the first answer is a refusal, the agent rewrites the question into legal vocabulary and searches again, so lay wording like "a bank rejects my loan with an algorithm" can reach the article that uses the legal term.
* **Streaming UI.** Answer text streams in, citations link to the article text in a source margin, and an agent trace shows what the agent did.

## How it works

```
EUR-Lex HTML
   │  ingest: structure-aware chunking (paragraph / point level, cross-reference enrichment)
   ▼
 961 chunks ──► BM25  ─┐
            └─► dense (bge-small, embedded Qdrant) ─┴─► Reciprocal Rank Fusion
                                                          │ articles only, 1 chunk per article
                                                          ▼
                              LangGraph agent:  retrieve ─► generate ─► (refused?) ─► rewrite ─► retrieve ─► generate
                                                          │
                                                          ▼
                              Qwen3 8B (Ollama) ─► post-processing: citation check, refusal detection
                                                          │
                                                          ▼
                                   FastAPI  /ask  /ask/stream (NDJSON)  +  web UI
```

Design decisions that the evaluation backs up:

|Decision|Why|
|-|-|
|Chunk by legal structure, not fixed size|Articles are split by paragraph and point, with the lead-in line repeated, so a chunk like "AI Act Art. 5(1)(f)-(g)" is self-contained and citable.|
|Hybrid BM25 + dense, fused with RRF|Legal text needs exact terms (BM25) and paraphrase matching (dense). Hybrid beat both alone (Hit@5 0.85 vs 0.70 / 0.79).|
|Articles first, recitals excluded|Recitals are explanatory and crowded binding articles out of the top 5 (Hit@5 0.85 to 0.91).|
|At most one chunk per article|A long article split into many chunks no longer fills the whole top 5 (0.91 to 0.94).|
|Cross-encoder reranker is off by default|Measured, then rejected: about 16 s extra per query on CPU with no measurable gain. It stays available behind `--rerank`.|
|One chunk per defined term in "Definitions" articles|GDPR Art. 4 has 26 near-identical definition chunks. Packing several into one chunk buried "personal data".|
|Exact-term boost for definitional questions|"What is the definition of X" promotes the chunk that defines exactly X. Added after the answer eval showed definitional questions being refused.|

## Results

Evaluated on 47 labelled questions (19 easy, 22 paraphrased, 6 hard) with gold articles, plus 10 negative questions (5 off-topic, 5 near-miss such as the CCPA or the DMA). Both sets are written by me, so treat the numbers as a regression benchmark, not a leaderboard.

### Retrieval

|Config|Hit@1|Hit@5|MRR@10|Median latency|
|-|-|-|-|-|
|BM25|0.36|0.70|0.484|2 ms|
|Dense|0.47|0.79|0.595|22 ms|
|Hybrid (RRF)|0.47|0.85|0.607|24 ms|
|Hybrid, articles-first|0.62|0.91|0.747|37 ms|
|**Hybrid, articles-first, 1 per article**|**0.62**|**0.94**|**0.757**|41 ms|

Hit@5 by difficulty for the final config: easy 1.00, paraphrase 0.86, hard 1.00. Gold labels are at article level. Full table and the list of misses are in [`eval/results.md`](eval/results.md).

### Answer quality (end to end, agent with fallback rewrite)

|Metric|Value|
|-|-|
|Answer rate (47 questions)|0.96|
|Cites a gold article (of answered)|0.93|
|Answers with an invalid citation|0.00|
|Answered only after the agent retried|2|
|Correct refusal on off-topic / near-miss (10 questions)|1.00|
|Median time per question|44 s (Qwen3 8B, CPU)|

Details and the failing questions: [`eval/answer\\\_results.md`](eval/answer_results.md).

## Known limitations

* **"Cites a gold article" is not "answer is correct".** The metric checks the citation, not the reasoning. I have not graded answer correctness by hand at scale.
* **Paraphrased questions are the weak spot.** Three retrieval misses (q08 bank loan and algorithm, q22 suing over misused data, q34 deployer obligations) are vocabulary mismatches between lay wording and legal wording. The agent's rewrite fixes some of them some of the time. I left them unfixed on purpose rather than tuning to the test set.
* **Two refusals on answerable questions** (q05 medical records and fingerprints, q08 loan decision).
* **One wrong-article answer I know of:** q47 cited AI Act Art. 27 (fundamental rights impact assessment) instead of Art. 26 and GDPR Art. 35.
* **The definitions fix was tuned on q01**, which is in the eval set. Treat that one score as optimistic.
* **Slow.** About 45 s per answer on CPU. A GPU or a smaller model changes that, but I did not benchmark either.
* **Not legal advice.** It is a retrieval demo over two texts. It does not cover national law, case law or guidance documents.

## Run it

```bash
python -m venv .venv
.venv\\\\Scripts\\\\activate            # Windows;  source .venv/bin/activate on Linux/macOS
pip install -e ".\\\[dev]"
ollama pull qwen3:8b

python -m eu\\\_reg\\\_rag.index        # builds the vector index from data/processed/chunks.jsonl
uvicorn eu\\\_reg\\\_rag.api:app        # open http://localhost:8000
```

The chunks are committed, so you do not need to download anything. To rebuild them from the source texts, save the two EUR-Lex pages as `data/raw/gdpr.html` and `data/raw/ai\\\_act.html` and run `python -m eu\\\_reg\\\_rag.ingest`.

Embedded Qdrant locks its folder: stop the API before running the index, search or eval commands.

### Command line

```bash
python -m eu\\\_reg\\\_rag.ask "Can a bank reject my loan purely by an algorithm?"
python -m eu\\\_reg\\\_rag.search "When is a DPIA required?" -k 5
python -m eu\\\_reg\\\_rag.evaluate                       # retrieval eval, writes eval/results.md
python -m eu\\\_reg\\\_rag.evaluate\\\_answers --limit 10    # answer eval (needs Ollama), \\\~45 min for the full set
```

### API

|Endpoint||
|-|-|
|`GET /`|web UI|
|`GET /health`|API, Ollama, Qdrant and pipeline status|
|`POST /ask`|`{"question": "..."}` returns the answer, sources, agent trace and timings|
|`POST /ask/stream`|the same as NDJSON events: `sources`, `rewrite`, `generating`, `token`, `attempt\\\_done`, `final`|

## Tests and CI

```bash
pytest          # 42 tests, no model downloads or Ollama needed
ruff check . \\\&\\\& ruff format --check .
```

GitHub Actions runs two jobs: lint and tests, and a retrieval quality gate that builds the index from the committed chunks and fails the build if Hit@5 drops below 0.90.

## Stack

Python 3.11, FastAPI, LangGraph, Qdrant (embedded), `rank-bm25`, `BAAI/bge-small-en-v1.5` embeddings, Qwen3 8B via Ollama, plain HTML/CSS/JS for the UI.

