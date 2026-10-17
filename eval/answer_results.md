## Answer quality (agent, strategy=fallback)

| Metric | Value |
|---|---|
| Answer rate (n=47) | 0.96 |
| Citation accuracy (cites a gold article) | 0.93 |
| Answers with invalid citations | 0.00 |
| Answered only after retry | 2 |
| Refusal accuracy on off-topic (n=10) | 1.00 |
| Median time per question | 44.1 s |

### Problems

- **q05** refused: Is a company allowed to process my medical records or fingerprints?
- **q08** refused: Can a bank reject my loan application purely by an algorithm without any human looking at it?
- **q18** cited gdpr-art-4 instead of gdpr-art-13: What does a privacy notice have to tell people when we collect their data directly from them?
- **q34** cited aiact-art-24, aiact-art-80 instead of aiact-art-26: What does a company that uses a high-risk AI system (but did not build it) have to do?
- **q47** cited aiact-art-27 instead of aiact-art-26, gdpr-art-35: Does a company deploying a high-risk AI system still need to do a GDPR data protection impact assessment?
