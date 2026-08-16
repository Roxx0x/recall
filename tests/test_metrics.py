from recall import metrics as M


def test_recall_at_k():
    assert M.recall_at_k(["a", "b", "c"], {"a", "d"}, 3) == 0.5
    assert M.recall_at_k(["a", "b"], {"a", "b"}, 5) == 1.0
    assert M.recall_at_k(["x"], {"a"}, 3) == 0.0


def test_recall_empty_relevant_is_one():
    assert M.recall_at_k(["a"], set(), 3) == 1.0


def test_precision_at_k():
    assert M.precision_at_k(["a", "b", "c", "d"], {"a", "c"}, 4) == 0.5
    assert M.precision_at_k(["a"], {"a"}, 1) == 1.0
    assert M.precision_at_k([], {"a"}, 0) == 0.0


def test_hit_at_k():
    assert M.hit_at_k(["x", "a"], {"a"}, 2) == 1.0
    assert M.hit_at_k(["x", "a"], {"a"}, 1) == 0.0   # a is beyond top-1


def test_mrr():
    assert M.mrr(["x", "y", "a"], {"a"}) == 1 / 3
    assert M.mrr(["a"], {"a"}) == 1.0
    assert M.mrr(["x"], {"a"}) == 0.0


def test_ndcg_prefers_better_order():
    gains = {"a": 3, "b": 2, "c": 1}
    good = M.ndcg_at_k(["a", "b", "c"], gains, 3)
    bad = M.ndcg_at_k(["c", "b", "a"], gains, 3)
    assert abs(good - 1.0) < 1e-9
    assert bad < good


def test_staleness():
    # two of the three returned are outdated
    assert abs(M.staleness_at_k(["old1", "old2", "cur"], {"old1", "old2"}, 3) - 2 / 3) < 1e-9
    assert M.staleness_at_k(["cur1", "cur2"], {"old"}, 2) == 0.0


def test_forgetting_score():
    # should have forgotten 2; if both still surface, score 0
    assert M.forgetting_score(["a", "b", "c"], {"a", "b"}) == 0.0
    # both correctly absent → 1.0
    assert M.forgetting_score(["c", "d"], {"a", "b"}) == 1.0
    # one of two forgotten → 0.5
    assert M.forgetting_score(["a", "c"], {"a", "b"}) == 0.5
    assert M.forgetting_score(["a"], set()) == 1.0
