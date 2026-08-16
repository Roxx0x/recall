"""
Run a task against an adapter and produce a scored report.

Loads the task's memories into the adapter, runs every probe, and averages the
metrics across probes. The result is a plain dict you can print, diff between two
backends, or assert on in CI to catch a memory regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import metrics
from .adapter import MemoryAdapter
from .task import Task


@dataclass
class Report:
    task: str
    k: int
    n_probes: int
    scores: dict[str, float]
    per_probe: list[dict[str, float]] = field(default_factory=list)

    def __str__(self) -> str:
        head = f"{self.task}  (k={self.k}, {self.n_probes} probes)"
        rows = "\n".join(f"  {name:20} {val:.3f}" for name, val in self.scores.items())
        return f"{head}\n{rows}"


def _mean(rows: list[dict[str, float]], key: str) -> float:
    vals = [r[key] for r in rows if key in r]
    return sum(vals) / len(vals) if vals else 0.0


def evaluate(adapter: MemoryAdapter, task: Task, *, k: int = 5) -> Report:
    problems = task.validate()
    if problems:
        raise ValueError("task has label errors:\n  " + "\n  ".join(problems))

    for mem_id, text in task.memories.items():
        adapter.add(mem_id, text)

    per_probe: list[dict[str, float]] = []
    for p in task.probes:
        retrieved = adapter.search(p.query, k)
        row: dict[str, float] = {}
        if p.relevant:
            row["recall@k"] = metrics.recall_at_k(retrieved, p.relevant, k)
            row["precision@k"] = metrics.precision_at_k(retrieved, p.relevant, k)
            row["hit@k"] = metrics.hit_at_k(retrieved, p.relevant, k)
            row["mrr"] = metrics.mrr(retrieved, p.relevant)
        if p.gains:
            row["ndcg@k"] = metrics.ndcg_at_k(retrieved, p.gains, k)
        if p.stale:
            row["staleness@k"] = metrics.staleness_at_k(retrieved, p.stale, k)
        if p.should_forget:
            surfaced = adapter.surface(p.query) if hasattr(adapter, "surface") else adapter.search(p.query, 10_000)
            row["forgetting"] = metrics.forgetting_score(surfaced, p.should_forget)
        per_probe.append(row)

    keys = ["recall@k", "precision@k", "hit@k", "mrr", "ndcg@k", "staleness@k", "forgetting"]
    scores = {key: _mean(per_probe, key) for key in keys if any(key in r for r in per_probe)}
    return Report(task=task.name, k=k, n_probes=len(task.probes), scores=scores, per_probe=per_probe)
