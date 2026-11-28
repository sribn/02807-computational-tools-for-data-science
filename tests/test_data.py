import numpy as np
import pandas as pd
import scipy.sparse as sp

from recommender import data


def entries(rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id", "item_name", "playtime"])


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


def test_deduplicate_keeps_the_largest_playtime():
    raw = entries([("u1", "g1", "A", 10), ("u1", "g1", "A", 30), ("u1", "g2", "B", 5), ("u1", "g2", "B", 5)])
    out = data.deduplicate(raw)
    assert out[["item_id", "playtime"]].values.tolist() == [["g1", 30], ["g2", 5]]


def test_filter_counts_players_not_owners():
    # g2 has two owners but only one player, so with min_players=2 it is dropped.
    user_items = entries([
        ("u1", "g1", "A", 10), ("u1", "g2", "B", 5),
        ("u2", "g1", "A", 7), ("u2", "g2", "B", 0),
    ])
    out = data.build_interactions(user_items, min_players=2, min_played=1)
    assert list(out.item_ids) == ["g1"]
    assert out.playtime.toarray().tolist() == [[10], [7]]


def test_filters_repeat_until_both_rules_hold():
    # Pass 1 drops u3 (one played game); that leaves g3 with one player, so pass 2 drops g3,
    # which leaves u2 with one game, so pass 2 also drops u2.
    user_items = entries([
        ("u1", "g1", "A", 1), ("u1", "g2", "B", 1),
        ("u2", "g1", "A", 1), ("u2", "g3", "C", 1),
        ("u3", "g3", "C", 1),
        ("u4", "g1", "A", 1), ("u4", "g2", "B", 1),
    ])
    out = data.build_interactions(user_items, min_players=2, min_played=2)

    assert list(out.user_ids) == ["u1", "u4"]
    assert list(out.item_ids) == ["g1", "g2"]
    players = data.players_per_game(out.playtime)
    assert players.min() >= 2 and np.diff(out.playtime.indptr).min() >= 2
    assert out.filter_log["played pairs"].iloc[-1] == 4
