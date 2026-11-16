"""Frequent itemsets over the users' played-game baskets: PCY for pairs, Apriori for triples.

A basket is one user's set of played games in a (training) matrix; items are column indices.
Supports are absolute counts of baskets.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numba import njit

# splitmix64 constants, used to hash a pair (i, j) to a PCY bucket
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)


@dataclass(frozen=True)
class FrequentPairs:
    item_support: np.ndarray  # baskets containing each item
    pair_support: np.ndarray  # dense n x n counts (symmetric), counted only for PCY candidates
    min_support: int
    n_candidates: int         # pairs of frequent items whose hash bucket was frequent
    frequent_bucket_share: float

    @property
    def frequent(self):
        """Boolean n x n matrix of frequent pairs."""
        return self.pair_support >= self.min_support


@dataclass(frozen=True)
class FrequentTriples:
    items: np.ndarray    # (m, 3) item indices, each row sorted a < b < c
    support: np.ndarray  # (m,) baskets containing all three
    n_candidates: int    # triples whose three sub-pairs are all frequent


def baskets(train):
    """Binary CSR matrix with sorted indices: row u is user u's basket of played games."""
    train = train.tocsr()
    b = sp.csr_matrix(
        (np.ones(train.nnz, dtype=np.int8), train.indices.astype(np.int32), train.indptr),
        shape=train.shape,
    )
    b.sort_indices()
    return b


def scale_support(min_support, fit_on, apply_to, size):
    """Translate a support threshold tuned on `fit_on` to the bigger matrix `apply_to`.

    The per-user split keeps the same users but hides a fraction of each basket, so a k-item
    set's support grows roughly with (games kept per basket)^k when hidden games come back.
    """
    return int(round(min_support * (apply_to.nnz / fit_on.nnz) ** size))


def pcy_pairs(basket_matrix, min_support, n_buckets):
    """PCY: count items and hashed pair buckets in pass 1, then count candidate pairs in pass 2.

    A pair is a candidate only if both items are frequent and its bucket is frequent. Pass 2
    stores counts in a dense items x items array, which fits in memory at this catalogue size.
    """
    indptr, indices, n = basket_matrix.indptr, basket_matrix.indices, basket_matrix.shape[1]
    item_support, bucket_counts = _pcy_pass1(indptr, indices, n, n_buckets)
    frequent_items = item_support >= min_support
    bitmap = bucket_counts >= min_support

    pair_support, n_candidates = _pcy_pass2(indptr, indices, n, frequent_items, bitmap)
    return FrequentPairs(
        item_support, pair_support, min_support, int(n_candidates), float(bitmap.mean())
    )


def bucket_saturation(basket_matrix, min_support, bucket_sizes):
    """How many hash buckets are frequent, and how many candidate pairs PCY keeps, per table size."""
    indptr, indices, n = basket_matrix.indptr, basket_matrix.indices, basket_matrix.shape[1]
    rows = []
    for n_buckets in bucket_sizes:
        item_support, bucket_counts = _pcy_pass1(indptr, indices, n, n_buckets)
        frequent_items = item_support >= min_support
        bitmap = bucket_counts >= min_support
        rows.append({
            "buckets": n_buckets,
            "frequent bucket share": bitmap.mean(),
            "candidate pairs": int(_count_candidates(frequent_items, bitmap)),
            "pairs of frequent items": int(frequent_items.sum() * (frequent_items.sum() - 1) // 2),
        })
    return pd.DataFrame(rows)


def apriori_triples(basket_matrix, pairs, min_support):
    """Apriori step k = 3: count only triples whose three sub-pairs are frequent."""
    if pairs.min_support > min_support:
        raise ValueError(f"pairs were mined at support {pairs.min_support}; triples at {min_support} "
                         "need pairs mined at that support or lower")
    n = basket_matrix.shape[1]
    keys = _triangles(pairs.frequent)  # sorted keys (a * n + b) * n + c with a < b < c
    support = _count_triples(basket_matrix.indptr, basket_matrix.indices, pairs.frequent, keys, n)
    keep = support >= min_support
    keys = keys[keep]
    items = np.column_stack([keys // (n * n), (keys // n) % n, keys % n])
    return FrequentTriples(items, support[keep], int(keep.size))


@njit(cache=True)
def _bucket(i, j, n_buckets):
    # splitmix64 finaliser on the pair key; a plain (i * m + j) % n_buckets collapses to j
    # whenever n_buckets divides m. All arithmetic stays in uint64 so it wraps, never floats.
    z = (np.uint64(i) << np.uint64(32)) | np.uint64(j)
    z = z + _GOLDEN
    z = (z ^ (z >> np.uint64(30))) * _MIX1
    z = (z ^ (z >> np.uint64(27))) * _MIX2
    z = z ^ (z >> np.uint64(31))
    return np.int64(z % np.uint64(n_buckets))


@njit(cache=True)
def _pcy_pass1(indptr, indices, n, n_buckets):
    item_support = np.zeros(n, dtype=np.int64)
    bucket_counts = np.zeros(n_buckets, dtype=np.int32)  # 4 bytes per bucket, as in the textbook
    for u in range(indptr.size - 1):
        items = indices[indptr[u]:indptr[u + 1]]
        for x in range(items.size):
            item_support[items[x]] += 1
            for y in range(x + 1, items.size):
                bucket_counts[_bucket(items[x], items[y], n_buckets)] += 1
    return item_support, bucket_counts


@njit(cache=True)
def _pcy_pass2(indptr, indices, n, frequent_items, bitmap):
    counts = np.zeros((n, n), dtype=np.int32)
    n_buckets = bitmap.size
    for u in range(indptr.size - 1):
        items = indices[indptr[u]:indptr[u + 1]]
        for x in range(items.size):
            a = items[x]
            if not frequent_items[a]:
                continue
            for y in range(x + 1, items.size):
                b = items[y]
                if frequent_items[b] and bitmap[_bucket(a, b, n_buckets)]:
                    counts[a, b] += 1
    # Mirror the upper triangle so lookups work in either order.
    for a in range(n):
        for b in range(a + 1, n):
            counts[b, a] = counts[a, b]
    return counts, _count_candidates(frequent_items, bitmap)


@njit(cache=True)
def _count_candidates(frequent_items, bitmap):
    n, n_buckets, total = frequent_items.size, bitmap.size, 0
    for a in range(n):
        if frequent_items[a]:
            for b in range(a + 1, n):
                if frequent_items[b] and bitmap[_bucket(a, b, n_buckets)]:
                    total += 1
    return total


@njit(cache=True)
def _triangles(frequent):
    """Sorted keys of all triples a < b < c whose three sub-pairs are frequent."""
    n = frequent.shape[0]
    total = 0
    for a in range(n):
        for b in range(a + 1, n):
            if frequent[a, b]:
                for c in range(b + 1, n):
                    if frequent[a, c] and frequent[b, c]:
                        total += 1
    keys = np.empty(total, dtype=np.int64)
    k = 0
    for a in range(n):
        for b in range(a + 1, n):
            if frequent[a, b]:
                for c in range(b + 1, n):
                    if frequent[a, c] and frequent[b, c]:
                        keys[k] = (a * n + b) * n + c
                        k += 1
    return keys


@njit(cache=True)
def _count_triples(indptr, indices, frequent, keys, n):
    support = np.zeros(keys.size, dtype=np.int32)
    for u in range(indptr.size - 1):
        items = indices[indptr[u]:indptr[u + 1]]
        for x in range(items.size):
            a = items[x]
            for y in range(x + 1, items.size):
                b = items[y]
                if not frequent[a, b]:
                    continue
                for z in range(y + 1, items.size):
                    c = items[z]
                    if frequent[a, c] and frequent[b, c]:
                        support[np.searchsorted(keys, (a * n + b) * n + c)] += 1
    return support
