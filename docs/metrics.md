# The metrics

Seven numbers. The first five are standard information retrieval, included
because they're the common language. The last two are the point of this harness:
they grade the things that make memory different from search.

## The standard five

**recall@k** — of the memories that should have come back for a query, what
fraction landed in the top k. The headline number, and the most misread (see
below).

**precision@k** — of the k you returned, what fraction were relevant. Recall and
precision trade off: return everything and recall is perfect while precision is
dreadful.

**hit@k** — 1 if anything relevant is in the top k, else 0. The lenient check:
"did it find *something* useful." Good for tasks where the agent only needs one
good hit.

**MRR** — mean reciprocal rank of the first relevant result. Rewards putting the
right memory near the top, not just somewhere in the list. A memory that buries
the answer at rank 8 scores 0.125 here even if recall@10 is perfect.

**nDCG@k** — for graded relevance, when some memories matter more than others.
Give each a gain (0–3), and this scores both whether you returned the important
ones and whether you ordered them well, normalised so 1.0 is the ideal ordering.

## The two that matter for memory

A document in a search index doesn't stop being true, and it doesn't need to be
forgotten. A memory does both. These two metrics grade exactly the gap.

**staleness@k** — of what you returned, what fraction is out of date. The user
was on the free plan (true once) and is now on enterprise; a memory that returns
the free-plan fact for "what plan is the user on" scores high recall *and* high
staleness. It found a relevant memory and the memory was wrong. No IR metric
catches this because IR has no concept of a fact expiring. Lower is better.

**forgetting** — of the memories that a correct system would have evicted, how
many did it actually drop. You probe with a query that *would* surface the
forgettable content, and check it's gone. A store that never forgets scores 0
here — which is correct, because a store that never forgets is a log, and it
rots the context the same way an unbounded history does. Higher is better.

## Why recall@k lies for agents

The trap: optimise recall@k and you get a system that returns everything remotely
relevant, including the stale and the trivial, and calls it a win. High recall
with high staleness is worse than mediocre recall with zero staleness, because
the agent acts on what it's told, and a confidently-returned outdated fact is a
wrong action, not a missing one.

Read the numbers together. recall@k tells you it found the memory; staleness
tells you whether the memory was still true; forgetting tells you whether the
store is curating or just accumulating. A memory system that wins on all three is
rare, and you cannot tell which one you have by feel — which is the entire reason
this harness exists.

## Reading a report

```
recall@k     0.83   found most of what mattered
precision@k  0.31   returned a lot of chaff alongside it
mrr          0.72   put the right answer near the top
staleness@k  0.20   one in five returned memories is out of date  ← the quiet killer
forgetting   0.10   evicts almost nothing; effectively append-only ← the other one
```

That store has good raw retrieval and a memory problem. You'd never see it from
recall alone.
