# recall

[![test](https://github.com/Roxx0x/recall/actions/workflows/test.yml/badge.svg)](https://github.com/Roxx0x/recall/actions/workflows/test.yml)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

An evaluation harness for agent memory. Point it at a memory backend and get numbers instead of a feeling.

Everyone is shipping agent memory. Almost nobody is measuring it, because "it seems to remember" is not a measurement and there was no clean way to get a real one. recall is that way: standard retrieval metrics, plus the two a search benchmark structurally can't give you — **how much of what it returned is out of date**, and **whether it actually forgot what it should have**.

Those two are the whole difference between grading a search index and grading a memory. A document never stops being true. A memory does.

## Quickstart

```
pip install "git+https://github.com/Roxx0x/recall"
```

```python
from recall import Task, LexicalAdapter, evaluate

task = Task.builtin("ops_agent")          # or Task.load("your_task.json")
print(evaluate(LexicalAdapter(), task, k=5))
```

```
ops_agent  (k=5, 6 probes)
  recall@k             1.000
  precision@k          0.240
  hit@k                1.000
  mrr                  0.900
  ndcg@k               1.000
  staleness@k          0.300   ← 30% of returned memories are out of date
  forgetting           0.000   ← forgets nothing; an append-only log
```

That last pair is the story. The lexical baseline finds everything (recall 1.0) and is also confidently serving stale facts and never forgetting a thing — which recall@k alone would never tell you.

```
recall run ops_agent          # score the built-in task with the baseline
recall run mytask.json -k 10
recall validate mytask.json   # catch label errors before they tank your scores
```

## Grade your backend

Three methods and you're gradable:

```python
class MyBackend:
    def add(self, mem_id, text): ...          # load a memory under a stable id
    def search(self, query, k): ...           # return up to k ids, best first
    def surface(self, query): ...             # optional: all ids it could return (for forgetting)

evaluate(MyBackend(), task, k=5)
```

Any store fits behind it — mem0, a vector db, your own. See [docs/adapters.md](docs/adapters.md).

## The metrics

| metric | what it catches |
|---|---|
| recall@k, precision@k, hit@k | did the relevant memory come back |
| MRR, nDCG@k | was it near the top, in the right order |
| **staleness@k** | is what came back still *true*, or superseded |
| **forgetting** | did the store evict what it should, or just accumulate |

The two bold ones don't exist in an IR benchmark, and they're where memory systems actually fail. Full explanation, including why optimising recall@k alone gives you a confidently-wrong agent, in [docs/metrics.md](docs/metrics.md).

## Why this exists

Optimise recall@k and you get a store that returns everything remotely relevant — stale facts and trivia included — and reports a great score. High recall with high staleness is worse than mediocre recall with zero staleness, because an agent acts on what it's told, and a confidently-returned outdated fact is a wrong action, not a missing one. You cannot tell which kind of memory you have by feel. That's the gap this fills.

## Tasks

A task is JSON: a set of memories and probes with gold labels (`relevant`, `stale`, `should_forget`, `gains`). One ships in [`recall/datasets/ops_agent.json`](recall/datasets/ops_agent.json) so the harness runs immediately; write your own against your real traffic. Format in [docs/adapters.md](docs/adapters.md#writing-a-task).

## Install and test

```
git clone https://github.com/Roxx0x/recall && cd recall
pip install -e ".[dev]"
pytest
python examples/grade_a_backend.py     # grades a deliberately lossy backend so you can see the metrics bite
```

Python 3.9+, standard library, no dependencies.

## Related

Two memory backends you might grade with it: [still-true](https://github.com/Roxx0x/still-true) (temporal facts) and [mtrace](https://github.com/Roxx0x/mtrace) (activation-based cognitive memory).

MIT.
