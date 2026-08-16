"""
The task format: a memory to load, and a set of probes to grade retrieval with.

A task is a small world. You load its `memories`, then for each `probe` you ask
the adapter the probe's query and check what came back against the labels:

  relevant       ids that *should* be retrieved
  stale          ids that are relevant-looking but out of date (staleness metric)
  should_forget  ids a correct memory would have evicted (forgetting metric)
  gains          optional graded relevance for nDCG

It's plain JSON, so a task is a file you can write by hand or generate. One ships
in datasets/ so the harness runs immediately.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Probe:
    query: str
    relevant: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    should_forget: list[str] = field(default_factory=list)
    gains: dict[str, float] = field(default_factory=dict)


@dataclass
class Task:
    name: str
    memories: dict[str, str]          # id -> text
    probes: list[Probe]

    @staticmethod
    def load(path: str | Path) -> "Task":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return Task.from_dict(data)

    @staticmethod
    def builtin(name: str = "ops_agent") -> "Task":
        """Load a task that ships with the package, by name, from anywhere —
        so the quickstart works whether you cloned the repo or pip-installed it."""
        path = Path(__file__).parent / "datasets" / (name if name.endswith(".json") else name + ".json")
        if not path.exists():
            raise FileNotFoundError(f"no built-in dataset {name!r}")
        return Task.load(path)

    @staticmethod
    def from_dict(data: dict) -> "Task":
        probes = [
            Probe(
                query=p["query"],
                relevant=p.get("relevant", []),
                stale=p.get("stale", []),
                should_forget=p.get("should_forget", []),
                gains={k: float(v) for k, v in p.get("gains", {}).items()},
            )
            for p in data["probes"]
        ]
        mems = data["memories"]
        if isinstance(mems, list):        # allow [{id, text}, ...] as well as {id: text}
            mems = {m["id"]: m["text"] for m in mems}
        return Task(name=data.get("name", "unnamed"), memories=mems, probes=probes)

    def validate(self) -> list[str]:
        """Return a list of problems, empty if the task is well-formed. A label
        that points at a memory id which doesn't exist is the usual mistake, and
        it silently tanks your scores, so we check for it up front."""
        problems: list[str] = []
        ids = set(self.memories)
        for i, p in enumerate(self.probes):
            for field_name in ("relevant", "stale", "should_forget"):
                for mid in getattr(p, field_name):
                    if mid not in ids:
                        problems.append(f"probe[{i}].{field_name}: unknown memory id {mid!r}")
            if not p.relevant and not p.should_forget:
                problems.append(f"probe[{i}]: has neither relevant nor should_forget labels")
        return problems
