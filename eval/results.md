## Retrieval results

| Config | Hit@1 | Hit@5 | MRR@10 | Median latency |
|---|---|---|---|---|
| BM25 | 0.32 | 0.68 | 0.460 | 2 ms |
| Dense | 0.45 | 0.79 | 0.587 | 21 ms |
| Hybrid (RRF) | 0.51 | 0.87 | 0.637 | 24 ms |
| Hybrid, articles-first | 0.74 | 0.91 | 0.822 | 34 ms |
| Hybrid, articles-first, 1 per article | 0.74 | 0.94 | 0.827 | 34 ms |

### Hit@5 by question difficulty

| Config | easy (n=19) | paraphrase (n=22) | hard (n=6) |
|---|---|---|---|
| BM25 | 0.95 | 0.50 | 0.50 |
| Dense | 0.95 | 0.68 | 0.67 |
| Hybrid (RRF) | 1.00 | 0.77 | 0.83 |
| Hybrid, articles-first | 1.00 | 0.82 | 1.00 |
| Hybrid, articles-first, 1 per article | 1.00 | 0.86 | 1.00 |

### Misses for 'Hybrid, articles-first, 1 per article' (gold not in top 5)

- **q08** (paraphrase) Can a bank reject my loan application purely by an algorithm without any human looking at it?  
  gold: gdpr-art-22 | got: aiact-art-5, aiact-art-26, aiact-art-74, aiact-art-2, gdpr-art-2 | rank: None
- **q22** (paraphrase) Can I sue a company for money if they misused my personal data?  
  gold: gdpr-art-82 | got: gdpr-art-13, gdpr-art-14, gdpr-art-85, gdpr-art-15, gdpr-art-17 | rank: None
- **q34** (paraphrase) What does a company that uses a high-risk AI system (but did not build it) have to do?  
  gold: aiact-art-26 | got: aiact-art-25, aiact-art-6, aiact-art-24, aiact-art-80, aiact-art-20 | rank: None
