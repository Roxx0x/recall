"""Command line interface. `recall --help`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapter import LexicalAdapter
from .runner import evaluate
from .task import Task

_BUILTIN = Path(__file__).parent / "datasets" / "ops_agent.json"


def _resolve_task(name: str) -> Task:
    p = Path(name)
    if not p.exists():
        builtin = Path(__file__).parent / "datasets" / (name if name.endswith(".json") else name + ".json")
        if builtin.exists():
            p = builtin
        else:
            raise SystemExit(f"no task file or built-in dataset named {name!r}")
    return Task.load(p)


def cmd_run(a) -> int:
    task = _resolve_task(a.task)
    # Only the built-in lexical baseline is wired to the CLI; a real backend is
    # graded from Python by passing your own adapter to evaluate().
    report = evaluate(LexicalAdapter(), task, k=a.k)
    if a.json:
        print(json.dumps({"task": report.task, "k": report.k, "scores": report.scores}, indent=2))
    else:
        print(report)
    return 0


def cmd_validate(a) -> int:
    task = _resolve_task(a.task)
    problems = task.validate()
    if problems:
        for p in problems:
            print("  " + p, file=sys.stderr)
        print(f"{len(problems)} problem(s)", file=sys.stderr)
        return 1
    print(f"{task.name}: ok ({len(task.memories)} memories, {len(task.probes)} probes)")
    return 0


def cmd_datasets(a) -> int:
    for f in sorted((Path(__file__).parent / "datasets").glob("*.json")):
        t = Task.load(f)
        print(f"  {f.stem:16} {len(t.memories):3} memories  {len(t.probes):2} probes")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="recall", description="Evaluate agent memory.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="score a task with the built-in lexical baseline")
    r.add_argument("task", nargs="?", default=str(_BUILTIN), help="task file or built-in name (e.g. ops_agent)")
    r.add_argument("-k", type=int, default=5)
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=cmd_run)

    v = sub.add_parser("validate", help="check a task's labels")
    v.add_argument("task")
    v.set_defaults(fn=cmd_validate)

    d = sub.add_parser("datasets", help="list built-in datasets")
    d.set_defaults(fn=cmd_datasets)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
