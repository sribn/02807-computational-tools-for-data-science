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
