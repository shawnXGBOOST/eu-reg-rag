"""Cross-encoder reranking.

Retrieval (BM25 + dense) is optimised for recall: get the right chunk somewhere in the
top ~30. A cross-encoder reads query and chunk *together* and scores true relevance,
which fixes the ordering. Too slow to run on all 820 chunks, cheap on 30.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from eu_reg_rag.config import settings

# (query, texts) -> relevance scores (higher = more relevant)
ScoreFn = Callable[[str, Sequence[str]], list[float]]


def cross_encoder_scorer(model_name: str | None = None) -> ScoreFn:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name or settings.rerank_model, max_length=512)

    def score(query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in model.predict([(query, t) for t in texts])]

    return score
