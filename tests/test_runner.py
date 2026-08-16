from pathlib import Path

import pytest

from recall import LexicalAdapter, Task, evaluate
from recall.task import Probe

DATASET = Path(__file__).resolve().parents[1] / "recall" / "datasets" / "ops_agent.json"


def test_builtin_task_loads_and_validates():
    task = Task.load(DATASET)
    assert task.validate() == []
    assert len(task.memories) >= 10


def test_evaluate_produces_the_expected_metric_set():
    task = Task.load(DATASET)
    report = evaluate(LexicalAdapter(), task, k=5)
    # the ops_agent task has relevant, stale, gains, and should_forget labels,
    # so every metric family should be present
    for key in ("recall@k", "precision@k", "hit@k", "mrr", "ndcg@k", "staleness@k", "forgetting"):
        assert key in report.scores, key
        assert 0.0 <= report.scores[key] <= 1.0


def test_lexical_baseline_recalls_current_facts():
    task = Task.load(DATASET)
    report = evaluate(LexicalAdapter(), task, k=5)
    # a bag-of-words baseline should still find the obviously-worded facts
    assert report.scores["recall@k"] > 0.5


def test_lexical_baseline_forgets_nothing():
    # the whole point of the forgetting metric: a store with no eviction should
    # score ~0, which is exactly what makes it a useful negative baseline
    task = Task.load(DATASET)
    report = evaluate(LexicalAdapter(), task, k=5)
    assert report.scores["forgetting"] == 0.0


def test_evaluate_rejects_bad_labels():
    task = Task(name="broken", memories={"a": "x"}, probes=[Probe(query="q", relevant=["nonexistent"])])
    with pytest.raises(ValueError):
        evaluate(LexicalAdapter(), task)


def test_task_from_list_memories():
    task = Task.from_dict({
        "name": "t",
        "memories": [{"id": "a", "text": "hello world"}],
        "probes": [{"query": "hello", "relevant": ["a"]}],
    })
    assert task.memories["a"] == "hello world"
    assert evaluate(LexicalAdapter(), task, k=1).scores["recall@k"] == 1.0
