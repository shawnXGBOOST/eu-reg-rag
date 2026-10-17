## Retrieval results

| Config | Hit@1 | Hit@5 | MRR@10 | Median latency |
|---|---|---|---|---|
| Hybrid, articles-first, 1 per article | 0.74 | 0.94 | 0.827 | 35 ms |
| Articles-first + LLM query rewrite | 0.55 | 0.87 | 0.662 | 10694 ms |

### Hit@5 by question difficulty

| Config | easy (n=19) | paraphrase (n=22) | hard (n=6) |
|---|---|---|---|
| Hybrid, articles-first, 1 per article | 1.00 | 0.86 | 1.00 |
| Articles-first + LLM query rewrite | 0.79 | 0.91 | 1.00 |

### Misses for 'Articles-first + LLM query rewrite' (gold not in top 5)

- **q06** (easy) What is the right to erasure (right to be forgotten)?  
  gold: gdpr-art-17 | got: gdpr-art-19, gdpr-art-20, gdpr-art-58, gdpr-art-70, gdpr-art-80 | rank: 8
- **q16** (easy) What must be included in the records of processing activities?  
  gold: gdpr-art-30 | got: gdpr-art-9, gdpr-art-49, aiact-art-10, gdpr-art-6, gdpr-art-14 | rank: 7
- **q21** (paraphrase) What must the contract with an outsourcing vendor that handles our customer data contain?  
  gold: gdpr-art-28 | got: gdpr-art-13, gdpr-art-88, gdpr-art-7, gdpr-art-6, gdpr-art-14 | rank: 7
- **q22** (paraphrase) Can I sue a company for money if they misused my personal data?  
  gold: gdpr-art-82 | got: gdpr-art-80, gdpr-art-15, gdpr-art-17, gdpr-art-13, gdpr-art-14 | rank: 8
- **q37** (easy) What are the obligations for providers of general-purpose AI models?  
  gold: aiact-art-53 | got: aiact-art-55, aiact-art-54, aiact-ann-XI, aiact-art-88, aiact-ann-XII | rank: 6
- **q40** (easy) What are AI regulatory sandboxes?  
  gold: aiact-art-57 | got: aiact-art-58, aiact-art-3, aiact-art-59, aiact-art-76, aiact-art-108 | rank: 10
