"""Embed all chunks and (re)build the Qdrant collection.

Usage:  python -m eu_reg_rag.index
"""

import time

from eu_reg_rag.config import settings
from eu_reg_rag.retrieval import build_index, get_client, load_chunks, sentence_transformer_encoder


def main() -> None:
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks. Loading embedding model {settings.embed_model} ...")
    encode = sentence_transformer_encoder()
    t0 = time.time()
    client = get_client()
    build_index(chunks, encode, client)
    n = client.count(settings.collection).count
    client.close()
    print(f"Indexed {n} chunks into '{settings.collection}' in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
