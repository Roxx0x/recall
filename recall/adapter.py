"""
The one interface a memory system has to implement to be graded.

Three methods: load a memory, search, and (optionally) report everything the
store could surface — the last one is what lets `forgetting_score` see whether
evicted memories are really gone. Any backend fits behind this: mem0, a vector
db, your own thing. A reference in-memory adapter ships so the harness runs and
tests with nothing installed.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

_WORD = re.compile(r"[a-z0-9]+")


@runtime_checkable
class MemoryAdapter(Protocol):
    def add(self, mem_id: str, text: str) -> None:
        """Load one memory under a stable id (the id the gold labels reference)."""
        ...

    def search(self, query: str, k: int) -> list[str]:
        """Return up to k memory ids, best first."""
        ...

    def surface(self, query: str) -> list[str]:
        """Every id the store could return for this query, ranked. Used by
        forgetting_score to check that evicted memories are actually absent.
        Default to the same as an unbounded search if the store has no notion of
        a separate surface."""
        ...


class LexicalAdapter:
    """A dependency-free reference backend: bag-of-words overlap scoring.

    It is intentionally simple — it exists so the harness has something to grade
    out of the box and so the metrics have a system to run against in tests. It
    does no forgetting, which makes it a useful negative baseline: it should score
    a forgetting_score near zero, and a good real backend should beat it there.
    """

    def __init__(self) -> None:
        self._mem: dict[str, set[str]] = {}
        self._text: dict[str, str] = {}

    def add(self, mem_id: str, text: str) -> None:
        self._mem[mem_id] = set(_WORD.findall(text.lower()))
        self._text[mem_id] = text

    def _ranked(self, query: str) -> list[str]:
        q = set(_WORD.findall(query.lower()))
        scored = [
            (mid, len(q & toks) / (len(q | toks) or 1))   # Jaccard overlap
            for mid, toks in self._mem.items()
        ]
        scored = [(mid, s) for mid, s in scored if s > 0]
        scored.sort(key=lambda x: -x[1])
        return [mid for mid, _ in scored]

    def search(self, query: str, k: int) -> list[str]:
        return self._ranked(query)[:k]

    def surface(self, query: str) -> list[str]:
        return self._ranked(query)   # no eviction: everything that matches is surfaced
