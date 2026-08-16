"""
recall — an evaluation harness for agent memory.

You can't improve memory you don't measure, and "it feels like it remembers" is
not a measurement. Point this at any memory backend and get numbers: not just
recall@k and MRR, but the two a search benchmark can't give you — how much of
what it returned is out of date, and whether it actually forgot what it should.

    from recall import Task, LexicalAdapter, evaluate

    task = Task.load("recall/datasets/ops_agent.json")
    print(evaluate(LexicalAdapter(), task, k=5))
"""

from .adapter import LexicalAdapter, MemoryAdapter
from .runner import Report, evaluate
from .task import Probe, Task

__version__ = "0.1.0"
__all__ = ["Task", "Probe", "MemoryAdapter", "LexicalAdapter", "evaluate", "Report", "__version__"]
