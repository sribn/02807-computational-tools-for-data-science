from itertools import combinations

import numpy as np
import pytest
import scipy.sparse as sp

from recommender import baselines


def toy_train():
    # Games 0 and 1 are usually played together; game 2 is played alone by user 3.
    return sp.csr_matrix(np.array([
        [10.0, 5.0, 0.0],
        [3.0, 8.0, 0.0],
        [4.0, 0.0, 0.0],
        [0.0, 0.0, 9.0],
    ]))


def test_pair_counts_match_brute_force_frequent_pairs():
    rng = np.random.default_rng(0)
    train = sp.csr_matrix((rng.random((40, 8)) < 0.4) * 1.0)
    counts = baselines.pair_counts(train)

    brute = {}
    for row in train.toarray():
        for a, b in combinations(np.flatnonzero(row), 2):
            brute[(a, b)] = brute.get((a, b), 0) + 1

    min_support = 5
    expected = {pair for pair, c in brute.items() if c >= min_support}
    found = {(a, b) for a, b in zip(*np.nonzero(counts >= min_support)) if a < b}
    assert found == expected


def test_rule_confidence_is_conditional_probability():
    conf = baselines.rule_confidences(baselines.pair_counts(toy_train()), min_support=1)
    assert conf[0, 1] == pytest.approx(2 / 3)  # 2 of the 3 players of game 0 also played game 1
    assert conf[1, 0] == pytest.approx(1.0)
    assert conf[0, 0] == 0.0


def test_rules_below_min_support_are_dropped():
    conf = baselines.rule_confidences(baselines.pair_counts(toy_train()), min_support=3)
    assert not conf.any()


def test_association_rules_score_by_strongest_rule():
    train = toy_train()
    conf = baselines.rule_confidences(baselines.pair_counts(train), min_support=1)
    scores = baselines.association_rule_scorer(train, conf)(np.array([2]))
    # User 2 played only game 0, so game 1 scores conf(0 -> 1) and game 2 only the tie-break.
    assert scores[0, 1] == pytest.approx(2 / 3, abs=1e-6)
    assert scores[0, 2] < 1e-6


def test_cosine_neighbours_keep_top_k_and_no_self_similarity():
    counts = baselines.pair_counts(toy_train())
    sim = baselines.cosine_neighbours(counts, k=1)
    assert np.all(np.diag(sim) == 0)
    assert np.all((sim > 0).sum(axis=1) <= 1)
    assert sim[0, 1] == pytest.approx(2 / np.sqrt(3 * 2))


def test_item_knn_scores_sum_similarity_of_played_games():
    train = toy_train()
    sim = baselines.cosine_neighbours(baselines.pair_counts(train), k=2)
    scores = baselines.item_knn_scorer(train, sim)(np.array([2]))
    assert scores[0, 1] == pytest.approx(sim[0, 1], abs=1e-6)
