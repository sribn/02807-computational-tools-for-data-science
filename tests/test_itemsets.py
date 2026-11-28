from collections import Counter
from itertools import combinations

import numpy as np
import pytest
import scipy.sparse as sp

from recommender import itemsets


def random_baskets(seed=0, users=60, items=10, density=0.45):
    rng = np.random.default_rng(seed)
    return sp.csr_matrix((rng.random((users, items)) < density) * rng.integers(1, 100, (users, items)))


def brute_force(matrix, size):
    counts = Counter()
    for row in (matrix.toarray() > 0):
        counts.update(combinations(np.flatnonzero(row), size))
    return counts


@pytest.mark.parametrize("n_buckets", [3, 17, 10_000])
def test_pcy_finds_exactly_the_frequent_pairs(n_buckets):
    # Few buckets force many collisions; PCY must still return every frequent pair.
    matrix = random_baskets()
    pairs = itemsets.pcy_pairs(itemsets.baskets(matrix), min_support=12, n_buckets=n_buckets)

    expected = {pair for pair, c in brute_force(matrix, 2).items() if c >= 12}
    found = {(a, b) for a, b in zip(*np.nonzero(pairs.frequent)) if a < b}
    assert found == expected
    for a, b in expected:
        assert pairs.pair_support[a, b] == pairs.pair_support[b, a] == brute_force(matrix, 2)[(a, b)]


def test_pcy_prunes_candidates_only_when_buckets_are_rare():
    matrix = random_baskets(density=0.2, items=30)
    b = itemsets.baskets(matrix)
    few = itemsets.pcy_pairs(b, min_support=4, n_buckets=5)
    many = itemsets.pcy_pairs(b, min_support=4, n_buckets=100_003)
    assert few.frequent_bucket_share == 1.0
    assert many.n_candidates < few.n_candidates


@pytest.mark.parametrize("n_buckets", [1_000_003, 1 << 20])
def test_pair_hash_spreads_pairs_over_buckets(n_buckets):
    # With plenty of buckets, almost every distinct pair should get a bucket of its own.
    b = itemsets.baskets(sp.csr_matrix(np.ones((1, 40))))
    _, bucket_counts = itemsets._pcy_pass1(b.indptr, b.indices, 40, n_buckets)
    assert (bucket_counts > 0).sum() >= 0.99 * (40 * 39 // 2)


def test_apriori_finds_exactly_the_frequent_triples():
    matrix = random_baskets(seed=1)
    b = itemsets.baskets(matrix)
    pairs = itemsets.pcy_pairs(b, min_support=8, n_buckets=1_000)
    triples = itemsets.apriori_triples(b, pairs, min_support=8)

    truth = brute_force(matrix, 3)
    expected = {t for t, c in truth.items() if c >= 8}
    assert {tuple(t) for t in triples.items} == expected
    for t, s in zip(triples.items, triples.support):
        assert s == truth[tuple(t)]


def test_bucket_saturation_reports_share_of_frequent_buckets():
    table = itemsets.bucket_saturation(itemsets.baskets(random_baskets()), 12, [2, 50_000])
    assert table["frequent bucket share"].iloc[0] == 1.0
    assert table["candidate pairs"].iloc[1] <= table["candidate pairs"].iloc[0]


def test_scale_support_grows_with_basket_size_to_the_power_of_the_set_size():
    small = sp.csr_matrix(np.ones((4, 4)))
    big = sp.csr_matrix(np.ones((4, 5)))  # 1.25x the entries
    assert itemsets.scale_support(100, small, big, size=2) == 156
    assert itemsets.scale_support(100, small, big, size=3) == 195


def test_apriori_refuses_pairs_mined_at_a_higher_support():
    # A triple can be frequent at 8 while one of its pairs has support 8 or 9, so pairs mined
    # at 10 would silently drop it.
    b = itemsets.baskets(random_baskets(seed=1))
    pairs = itemsets.pcy_pairs(b, min_support=10, n_buckets=1_000)
    with pytest.raises(ValueError):
        itemsets.apriori_triples(b, pairs, min_support=8)
