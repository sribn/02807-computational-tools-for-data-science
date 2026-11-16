import numpy as np
import scipy.sparse as sp

from recommender import diversify, evaluate


def setup():
    # Six games in two clusters; the scores prefer cluster 0. User 0 already played game 0.
    labels = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([[9.0, 8.0, 7.0, 6.0, 5.0, 4.0]])
    train = sp.csr_matrix(np.array([[1.0, 0, 0, 0, 0, 0]]))
    return labels, (lambda rows: scores[rows]), train


def test_cap_limits_games_per_cluster_and_keeps_score_order():
    labels, score, train = setup()
    recs = diversify.capped_top_k(score, train, labels, cap=2, k=4)
    assert recs.tolist() == [[1, 2, 4, 5]]


def test_large_cap_gives_the_plain_top_k():
    labels, score, train = setup()
    assert np.array_equal(
        diversify.capped_top_k(score, train, labels, cap=10, k=4), evaluate.top_k(score, train, k=4)
    )


def test_slots_the_caps_cannot_fill_go_to_the_best_skipped_games():
    labels, score, train = setup()
    # cap=1 with two clusters fills two slots; the other two are the best skipped games (2, 3).
    recs = diversify.capped_top_k(score, train, labels, cap=1, k=4)
    assert recs.tolist() == [[1, 4, 2, 3]]
