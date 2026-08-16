"""
Grade a memory backend in ~15 lines.

Shows the whole shape: implement three methods, load a task, get numbers. Here
the "backend" is a toy that only stores the last N memories — a deliberately bad
memory — so you can see the metrics catch it: decent recall on recent facts,
terrible recall once its tiny window fills.

    python examples/grade_a_backend.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recall import Task, evaluate
from recall.adapter import LexicalAdapter


class TinyWindowBackend(LexicalAdapter):
    """A memory that only keeps the most recent `capacity` entries. A common
    naive design, and one the harness should expose as lossy."""

    def __init__(self, capacity: int = 5) -> None:
        super().__init__()
        self.capacity = capacity
        self._order: list[str] = []

    def add(self, mem_id: str, text: str) -> None:
        super().add(mem_id, text)
        self._order.append(mem_id)
        while len(self._order) > self.capacity:
            drop = self._order.pop(0)
            self._mem.pop(drop, None)
            self._text.pop(drop, None)


DATASET = Path(__file__).resolve().parents[1] / "recall" / "datasets" / "ops_agent.json"
task = Task.load(DATASET)

print("full memory (lexical baseline):")
print(evaluate(LexicalAdapter(), task, k=5))

print("\ntiny 5-item window (drops old memories):")
print(evaluate(TinyWindowBackend(capacity=5), task, k=5))

print("\nThe window backend's recall collapses once its 5-slot window fills: it drops")
print("old memories by age, not importance, so the facts you actually query are gone.")
print("This is the failure the harness exists to make visible before you ship it.")
