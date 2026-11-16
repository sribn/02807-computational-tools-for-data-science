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


def test_pair_counts_are_co_play_counts_with_support_on_the_diagonal():
    counts = baselines.pair_counts(toy_train())
    assert counts[0, 1] == counts[1, 0] == 2
    assert counts.diagonal().tolist() == [3, 2, 1]


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
