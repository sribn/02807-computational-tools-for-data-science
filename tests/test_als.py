import numpy as np
import pytest
import scipy.sparse as sp

from recommender import als


def block_playtime():
    """Two taste groups: users 0-3 play games 0-2, users 4-7 play games 3-5 (one gap each)."""
    dense = np.zeros((8, 6))
    dense[:4, :3] = 300.0
    dense[4:, 3:] = 300.0
    dense[0, 2] = 0.0
    dense[4, 5] = 0.0
    return sp.csr_matrix(dense)


def test_confidence_is_one_plus_alpha_log_playtime():
    conf = als.confidence(sp.csr_matrix(np.array([[0.0, 9.0]])), alpha=2.0)
    assert conf.nnz == 1
    assert np.isclose(conf[0, 1], 1 + 2 * np.log(10))


def test_solve_rows_matches_dense_closed_form():
    rng = np.random.default_rng(1)
    conf = als.confidence(block_playtime(), alpha=5.0)
    items = rng.normal(size=(6, 3))
    reg = 0.5

    users = als.solve_rows(conf, items, reg)

    dense_c = np.ones(conf.shape)
    dense_p = np.zeros(conf.shape)
    coo = conf.tocoo()
    dense_c[coo.row, coo.col] = coo.data
    dense_p[coo.row, coo.col] = 1.0
    for u in range(conf.shape[0]):
        cu = np.diag(dense_c[u])
        expected = np.linalg.solve(items.T @ cu @ items + reg * np.eye(3), items.T @ cu @ dense_p[u])
        assert np.allclose(users[u], expected)


def dense_loss(conf, model, reg):
    """Reference objective over every cell of the dense matrix."""
    dense_c = np.ones(conf.shape)
    dense_p = np.zeros(conf.shape)
    coo = conf.tocoo()
    dense_c[coo.row, coo.col] = coo.data
    dense_p[coo.row, coo.col] = 1.0
    err = dense_p - model.user_factors @ model.item_factors.T
    penalty = (model.user_factors ** 2).sum() + (model.item_factors ** 2).sum()
    return float((dense_c * err ** 2).sum() + reg * penalty)


def test_sparse_loss_matches_dense_objective():
    rng = np.random.default_rng(2)
    conf = als.confidence(block_playtime(), alpha=5.0)
    model = als.ALSModel(rng.normal(size=(8, 3)), rng.normal(size=(6, 3)))
    assert als.loss(conf, model, 0.7) == pytest.approx(dense_loss(conf, model, 0.7), rel=1e-12)


def test_solve_rows_matches_the_implicit_library():
    implicit_als = pytest.importorskip("implicit.cpu.als")
    rng = np.random.default_rng(4)
    conf = als.confidence(block_playtime(), alpha=5.0)
    items = rng.normal(size=(6, 3))
    reg = 0.5

    ours = als.solve_rows(conf, items, reg)
    # implicit reads every stored value as the confidence itself, so we hand it ours.
    for u in range(conf.shape[0]):
        theirs = implicit_als.user_factor(items, items.T @ items, conf, u, reg, 3)
        assert np.allclose(ours[u], theirs, atol=1e-12)


def test_loss_never_increases():
    conf = als.confidence(block_playtime(), alpha=5.0)
    losses = []
    als.fit(conf, factors=2, reg=0.1, iterations=8, callback=lambda _, m: losses.append(als.loss(conf, m, 0.1)))
    assert all(b <= a + 1e-9 for a, b in zip(losses, losses[1:]))


def test_fills_gaps_from_the_users_own_taste_group():
    conf = als.confidence(block_playtime(), alpha=5.0)
    model = als.fit(conf, factors=2, reg=0.1, iterations=15)
    scores = model.user_factors @ model.item_factors.T
    # User 0 never played game 2 but their group did; user 4 likewise for game 5.
    assert scores[0, 2] > scores[0, 3:].max()
    assert scores[4, 5] > scores[4, :3].max()


def test_similar_items_stay_within_group():
    conf = als.confidence(block_playtime(), alpha=5.0)
    model = als.fit(conf, factors=2, reg=0.1, iterations=15)
    top, _ = als.similar_items(model, item=0, n=2)
    assert set(top) == {1, 2}


def test_idf_weighted_confidence_trusts_niche_games_more():
    playtime = sp.csr_matrix(np.array([[60.0, 60.0], [60.0, 0.0], [60.0, 0.0]]))
    conf = als.confidence(playtime, alpha=1.0, item_weights=als.idf_weights(playtime))
    assert conf[0, 1] > conf[0, 0]
