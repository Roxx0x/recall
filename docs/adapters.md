# Adapters

To grade a memory backend, wrap it in an adapter — three methods. Anything that
can store text under an id and search it fits.

```python
class MyBackend:
    def add(self, mem_id: str, text: str) -> None:
        """Load one memory. The id is what the task's gold labels reference, so
        it must be stable — don't let the backend reassign ids."""
        ...

    def search(self, query: str, k: int) -> list[str]:
        """Return up to k memory ids, best first. This is what recall@k,
        precision@k, mrr, ndcg and staleness are computed from."""
        ...

    def surface(self, query: str) -> list[str]:
        """Optional. Every id the store could return for this query, ranked,
        ignoring the k cutoff. The forgetting metric uses it to check that
        evicted memories are genuinely absent. If you omit it, the runner falls
        back to search(query, 10000)."""
        ...
```

That's the whole contract. `add` and `search` are required; `surface` is only
needed if you want the forgetting metric to mean anything.

## Grading a real backend

The CLI only wires the built-in lexical baseline. Real backends are graded from
Python, because they need construction (a client, a connection, a model):

```python
from recall import Task, evaluate

task = Task.load("recall/datasets/ops_agent.json")

class Mem0Adapter:
    def __init__(self):
        from mem0 import Memory
        self.m = Memory()
    def add(self, mem_id, text):
        self.m.add(text, user_id="eval", metadata={"id": mem_id})
    def search(self, query, k):
        hits = self.m.search(query, user_id="eval", limit=k)
        return [h["metadata"]["id"] for h in hits]

print(evaluate(Mem0Adapter(), task, k=5))
```

The sketch above is the pattern, not a drop-in — every backend names its methods
differently, and mapping its result objects back to your ids is the only real
work.

## Writing a task

A task is JSON: a set of memories and a set of probes. See
[`recall/datasets/ops_agent.json`](../recall/datasets/ops_agent.json) for a
worked one. The labels per probe:

- **relevant** — ids that should be retrieved.
- **stale** — ids that look relevant but are out of date (feeds staleness).
- **should_forget** — ids a correct memory would have evicted (feeds forgetting);
  the probe's query should be one that would surface them if they were kept.
- **gains** — optional `{id: 0-3}` graded relevance for nDCG.

`Task.validate()` catches the usual mistake — a label pointing at a memory id
that doesn't exist, which silently drags your scores down. `recall validate
<task>` runs it from the shell.

## The lexical baseline as a floor

The built-in `LexicalAdapter` is bag-of-words overlap. It's there so the harness
runs with nothing installed, and as a deliberate negative baseline: it forgets
nothing (forgetting ≈ 0) and matches on words, not meaning. Any real backend you
grade should beat it, and if it doesn't, that result is worth more than a green
checkmark — it means the sophistication isn't buying you anything on this task.
