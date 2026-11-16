"""Association-rule recommender built on the frequent pairs (PCY) and triples (Apriori).

Rules are A -> c with a one-game premise {a} or a two-game premise {a, b}; confidence is
P(c | A) = support(A + c) / support(A). A user's score for game c combines the confidences of
all rules into c whose premise is among the games they played: either the strongest rule
("max") or the sum of all of them ("sum").
"""

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from recommender.evaluate import popularity_counts
from recommender.itemsets import baskets

TIE_BREAK = 1e-9  # fall back to popularity when no rule fires for a game
AGGREGATIONS = ("max", "sum")


@dataclass(frozen=True)
class TripleRules:
    """Rules {a, b} -> c grouped by premise, for fast lookup while scoring.

    premise[a, b] (with a < b) is the row of that premise or -1; its rules are
    consequent[ptr[row]:ptr[row + 1]] with confidence[...] alongside.
    """

    premise: np.ndarray
    ptr: np.ndarray
    consequent: np.ndarray
    confidence: np.ndarray

    @property
    def n_rules(self):
        return self.consequent.size


def pair_rule_confidences(pairs):
    """Dense n x n matrix with conf(a -> c) = support(a, c) / support(a) for frequent pairs."""
    conf = pairs.pair_support / np.maximum(pairs.item_support, 1)[:, None]
    conf[~pairs.frequent] = 0.0
    np.fill_diagonal(conf, 0.0)
    return conf.astype(np.float32)


def triple_rules(pairs, triples):
    """Each frequent triple {a, b, c} gives three rules: {a,b} -> c, {a,c} -> b, {b,c} -> a."""
    n = pairs.pair_support.shape[0]
    a, b, c = triples.items.T
    first = np.concatenate([a, a, b])
    second = np.concatenate([b, c, c])
    consequent = np.concatenate([c, b, a])
    support = np.tile(triples.support, 3).astype(np.float32)
    confidence = support / pairs.pair_support[first, second]

    key = first * n + second
    order = np.argsort(key, kind="stable")
    key, consequent, confidence = key[order], consequent[order], confidence[order]
    unique_keys, starts = np.unique(key, return_index=True)

    premise = np.full((n, n), -1, dtype=np.int32)
    premise[unique_keys // n, unique_keys % n] = np.arange(unique_keys.size, dtype=np.int32)
    ptr = np.append(starts, key.size).astype(np.int64)
    return TripleRules(premise, ptr, consequent.astype(np.int32), confidence.astype(np.float32))


def no_triple_rules(n):
    """An empty rule set, for the pairs-only recommender."""
    return TripleRules(
        np.full((n, n), -1, dtype=np.int32), np.zeros(1, dtype=np.int64),
        np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32),
    )


def association_rule_scorer(train, pair_conf, triples=None, aggregate="max"):
    """Score games for a batch of users from the rules whose premise they played."""
    if aggregate not in AGGREGATIONS:
        raise ValueError(f"aggregate must be one of {AGGREGATIONS}, got {aggregate!r}")
    b = baskets(train)
    triples = triples if triples is not None else no_triple_rules(pair_conf.shape[0])
    tie_break = TIE_BREAK * popularity_counts(train)
    use_sum = aggregate == "sum"

    def score(rows):
        return _score(
            b.indptr, b.indices, pair_conf, triples.premise, triples.ptr,
            triples.consequent, triples.confidence, np.asarray(rows), use_sum,
        ) + tie_break

    return score


@njit(parallel=True, cache=True)
def _score(indptr, indices, pair_conf, premise, ptr, consequent, confidence, rows, use_sum):
    n = pair_conf.shape[1]
    out = np.zeros((rows.size, n), dtype=np.float32)
    for r in prange(rows.size):
        u = rows[r]
        items = indices[indptr[u]:indptr[u + 1]]
        for x in range(items.size):
            rules = pair_conf[items[x]]
            for c in range(n):
                out[r, c] = out[r, c] + rules[c] if use_sum else max(out[r, c], rules[c])
        if consequent.size == 0:
            continue
        for x in range(items.size):
            for y in range(x + 1, items.size):
                row = premise[items[x], items[y]]
                if row < 0:
                    continue
                for q in range(ptr[row], ptr[row + 1]):
                    c = consequent[q]
                    out[r, c] = out[r, c] + confidence[q] if use_sum else max(out[r, c], confidence[q])
    return out
