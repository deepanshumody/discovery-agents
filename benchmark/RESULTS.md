# Retrieval benchmark — Banking77 intent retrieval

Dataset: **banking77** · queries (test): 3076 · pool (train): 9993 · intents: 77 · train steps: 1200 · seed: 7.

Relevant = same intent. **hit@k** = fraction of queries with ≥1 same-intent item in top-k (success@k, not textbook recall); **map** = mAP over the top-10 window.

Reproduce: `python -m discovery_agents.mlinfra.cli benchmark --full --with-st`.

| embedder | hit@1 | hit@5 | hit@10 | mrr | map |
|---|---|---|---|---|---|
| hashing (lexical baseline) | 0.7685 | 0.9220 | 0.9519 | 0.8345 | 0.5028 |
| torch (supervised contrastive) | 0.8296 | 0.9106 | 0.9330 | 0.8649 | 0.7749 |
| sentence-transformers (reference) | 0.9207 | 0.9701 | 0.9805 | 0.9417 | 0.8417 |
