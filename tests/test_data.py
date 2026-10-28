import numpy as np
import pandas as pd
import scipy.sparse as sp

from recommender import data


def test_split_partitions_each_users_games():
    rng = np.random.default_rng(0)
    dense = (rng.random((50, 30)) < 0.3) * rng.integers(1, 500, size=(50, 30))
    matrix = sp.csr_matrix(dense.astype(float))

    train, test = data.split_per_user(matrix, test_frac=0.2, seed=3)

    assert (train + test - matrix).count_nonzero() == 0
    assert train.multiply(test).nnz == 0
    for u in range(matrix.shape[0]):
        n = matrix[u].nnz
        if n >= 2:
            assert test[u].nnz == max(1, round(0.2 * n))
            assert train[u].nnz >= 1


def test_split_is_reproducible():
    matrix = sp.random(40, 20, density=0.4, format="csr", random_state=1)
    a = data.split_per_user(matrix, seed=7)[1]
    b = data.split_per_user(matrix, seed=7)[1]
    assert (a != b).nnz == 0


def test_build_interactions_filters_rare_games_unplayed_and_inactive_users():
    user_items = pd.DataFrame(
        [
            ("u1", "g1", "A", 10), ("u1", "g2", "B", 5), ("u1", "rare", "R", 99),
            ("u2", "g1", "A", 7), ("u2", "g2", "B", 0),  # g2 owned but unplayed
            ("u3", "g2", "B", 3),
        ],
        columns=["user_id", "item_id", "item_name", "playtime"],
    )

    out = data.build_interactions(user_items, min_owners=2, min_played=1)

    assert list(out.item_ids) == ["g1", "g2"]
    assert list(out.user_ids) == ["u1", "u2", "u3"]
    assert out.playtime.toarray().tolist() == [[10, 5], [7, 0], [0, 3]]

    strict = data.build_interactions(user_items, min_owners=2, min_played=2)
    assert list(strict.user_ids) == ["u1"]
