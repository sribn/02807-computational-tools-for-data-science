import numpy as np
import pytest
import scipy.sparse as sp

from recommender import evaluate


def test_top_k_skips_training_games_and_orders_by_score():
    train = sp.csr_matrix(np.array([[1.0, 0, 0, 0], [0, 0, 0, 1.0]]))
    scores = np.array([[9.0, 1.0, 5.0, 3.0], [0.0, 2.0, 1.0, 9.0]])

    recs = evaluate.top_k(lambda rows: scores[rows], train, k=2, batch_size=1)

    assert recs.tolist() == [[2, 3], [1, 2]]


def test_ranking_metrics_hand_computed():
    recs = np.array([[0, 1], [2, 3]])
    test = sp.csr_matrix(np.array([[0, 1.0, 1.0, 0], [0, 0, 0, 0]]))  # user 1 has no test games

    m = evaluate.ranking_metrics(recs, test)

    assert m["users"] == 1
    assert m["precision@2"] == pytest.approx(0.5)
    assert m["recall@2"] == pytest.approx(0.5)
    # Hit at rank 2: (1/log2(3)) / (1 + 1/log2(3))
    assert m["ndcg@2"] == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))


def test_popularity_counts_training_players():
    train = sp.csr_matrix(np.array([[5.0, 0, 1.0], [2.0, 0, 0]]))
    scores = evaluate.popularity_scorer(train)(np.array([0, 1]))
    assert scores.tolist() == [[2, 0, 1], [2, 0, 1]]


def test_ndcg_ideal_uses_at_most_k_test_games():
    # Three test games but k = 2: a perfect top-2 list scores NDCG 1, recall 2/3.
    recs = np.array([[0, 1]])
    test = sp.csr_matrix(np.array([[1.0, 1.0, 1.0, 0]]))
    m = evaluate.ranking_metrics(recs, test)
    assert m["ndcg@2"] == pytest.approx(1.0)
    assert m["recall@2"] == pytest.approx(2 / 3)


def test_long_tail_share_counts_slots_outside_the_most_played_games():
    popularity = np.array([100.0, 50, 10, 5, 1])  # head = top 20% = game 0
    recs = np.array([[0, 1], [0, 4]])
    assert evaluate.long_tail_share(recs, popularity, head_share=0.2) == pytest.approx(0.5)


def test_intra_list_similarity_averages_pairs_within_each_list():
    sim = np.array([[1.0, 0.8, 0.0], [0.8, 1.0, 0.2], [0.0, 0.2, 1.0]])
    recs = np.array([[0, 1], [0, 2]])
    assert evaluate.intra_list_similarity(recs, sim) == pytest.approx((0.8 + 0.0) / 2)


def test_paired_difference_interval_covers_a_known_shift():
    rng = np.random.default_rng(0)
    base = rng.random(5000)
    out = evaluate.paired_difference(base + 0.01, base)
    assert out["difference"] == pytest.approx(0.01)
    assert out["low"] == pytest.approx(0.01) and out["high"] == pytest.approx(0.01)

    noisy = evaluate.paired_difference(base + rng.normal(0.01, 0.1, base.size), base)
    assert noisy["low"] < 0.01 < noisy["high"]
