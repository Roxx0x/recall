"""
Retrieval metrics, plus the two that matter for memory and that nobody reports.

The standard IR metrics (recall@k, precision@k, MRR, nDCG) answer "did you
retrieve the relevant thing." For an agent's memory that's necessary but not
sufficient, because memory has a property a search index doesn't: it can return
something that's relevant and *wrong* — an outdated fact, a memory that should
have been forgotten. So there are two more here:

  staleness@k          — of what you returned, how much is out of date
  forgetting_score     — of what should be gone, how much did you correctly drop

Those two are the difference between grading a search engine and grading a memory.
All functions take ids; nothing here knows about embeddings or storage.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """|relevant ∩ top-k| / |relevant|. How much of what mattered you surfaced."""
    if not relevant:
        return 1.0  # nothing to find → trivially complete; caller may prefer to skip these
    top = set(retrieved[:k])
    return len(top & set(relevant)) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """|relevant ∩ top-k| / k. How much of what you surfaced mattered."""
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    rel = set(relevant)
    return sum(1 for r in top if r in rel) / k


def hit_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """1.0 if anything relevant is in the top-k, else 0.0. The lenient one."""
    rel = set(relevant)
    return 1.0 if any(r in rel for r in retrieved[:k]) else 0.0


def mrr(retrieved: Sequence[str], relevant: Collection[str]) -> float:
    """Reciprocal rank of the first relevant hit. Rewards getting it near the top."""
    rel = set(relevant)
    for i, r in enumerate(retrieved, start=1):
        if r in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gains: dict[str, float], k: int) -> float:
    """Normalised discounted cumulative gain with graded relevance.

    `gains` maps an id to how relevant it is (0, 1, 2, 3...). Discounts each hit
    by log2 of its rank, then normalises against the best possible ordering, so
    1.0 means you returned the most relevant items in the best order.
    """
    def dcg(order: Sequence[str]) -> float:
        return sum(gains.get(r, 0.0) / math.log2(i + 1) for i, r in enumerate(order[:k], start=1))

    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1) if g > 0)
    if idcg == 0:
        return 1.0 if not any(gains.get(r, 0) for r in retrieved[:k]) else 0.0
    return dcg(retrieved) / idcg


def staleness_at_k(retrieved: Sequence[str], stale: Collection[str], k: int) -> float:
    """Fraction of the top-k that is known to be out of date.

    This is the metric a search benchmark can't have, because a document doesn't
    stop being true. A memory does. `stale` is the set of ids that were correct
    once and aren't now (a superseded fact, an ended state). A memory that scores
    high recall and high staleness is confidently feeding the agent the past.
    Lower is better; 0.0 means nothing you returned is outdated.
    """
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    st = set(stale)
    return sum(1 for r in top if r in st) / len(top) if top else 0.0


def forgetting_score(retrieved_all: Collection[str], should_forget: Collection[str]) -> float:
    """Of the memories that should have been forgotten, how many stayed gone.

    Run the store's full retrieval surface for a query and pass everything it
    *could* return as `retrieved_all`; `should_forget` is what a correct memory
    would have evicted. Score is the fraction correctly absent. 1.0 means the
    system forgot everything it should have; 0.0 means it forgot nothing and is
    just an append-only log wearing a memory's name.
    """
    if not should_forget:
        return 1.0
    surfaced = set(retrieved_all)
    kept = sum(1 for f in should_forget if f in surfaced)
    return 1.0 - kept / len(should_forget)
