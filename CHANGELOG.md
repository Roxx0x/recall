# Changelog

## 0.1.0

- Standard retrieval metrics: recall@k, precision@k, hit@k, MRR, nDCG@k
- Memory-specific metrics: staleness@k (returned-but-outdated) and forgetting
  (did the store evict what it should)
- `MemoryAdapter` interface — three methods to grade any backend
- `LexicalAdapter` reference baseline, dependency-free
- JSON task format with `relevant` / `stale` / `should_forget` / `gains` labels,
  and `Task.validate()` for label errors
- Runner producing per-probe and averaged reports
- `recall` CLI: run, validate, datasets
- Built-in `ops_agent` dataset that exercises every metric
