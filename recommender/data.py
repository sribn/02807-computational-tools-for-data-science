"""Load the Steam inventories and turn them into a sparse user x game playtime matrix."""

import ast
import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA_DIR = Path(__file__).resolve().parent.parent / "dataset"
ITEMS_FILE = "australian_users_items.json.gz"

MIN_OWNERS = 50  # same threshold as the Apriori/PCY baskets in steam_cleaning.ipynb
MIN_PLAYED = 5   # users need enough played games to hide 20% for testing


@dataclass(frozen=True)
class Interactions:
    """Playtime matrix (users x games, minutes) plus the ids behind each row and column."""

    playtime: sp.csr_matrix
    user_ids: np.ndarray
    item_ids: np.ndarray
    item_names: dict


def load_user_items(path=DATA_DIR / ITEMS_FILE):
    """Return one row per (user, game) with playtime in minutes, duplicates removed."""
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = ast.literal_eval(line)
            for item in record["items"]:
                rows.append((record["user_id"], item["item_id"], item["item_name"], item["playtime_forever"]))

    df = pd.DataFrame(rows, columns=["user_id", "item_id", "item_name", "playtime"])
    # The raw file repeats some users with identical inventories (see steam_cleaning.ipynb).
    return df.drop_duplicates(subset=["user_id", "item_id"], ignore_index=True)


def build_interactions(user_items, min_owners=MIN_OWNERS, min_played=MIN_PLAYED):
    """Keep popular games and active users, and only count games that were actually played.

    Owned-but-unplayed games are treated like not owned: bundles and sales make
    ownership alone weak evidence of interest.
    """
    owners = user_items.groupby("item_id")["user_id"].nunique()
    popular = owners.index[owners >= min_owners]

    played = user_items[user_items["item_id"].isin(popular) & (user_items["playtime"] > 0)]
    played_per_user = played.groupby("user_id")["item_id"].size()
    active = played_per_user.index[played_per_user >= min_played]
    played = played[played["user_id"].isin(active)]

    user_codes, user_ids = pd.factorize(played["user_id"], sort=True)
    item_codes, item_ids = pd.factorize(played["item_id"], sort=True)
    playtime = sp.csr_matrix(
        (played["playtime"].to_numpy(dtype=np.float64), (user_codes, item_codes)),
        shape=(len(user_ids), len(item_ids)),
    )
    item_names = (
        played.drop_duplicates("item_id").set_index("item_id")["item_name"].to_dict()
    )
    return Interactions(playtime, np.asarray(user_ids), np.asarray(item_ids), item_names)


def split_per_user(matrix, test_frac=0.2, seed=0):
    """Hide a random `test_frac` of each user's games (at least one) as the test set.

    Returns new (train, test) CSR matrices; every stored entry lands in exactly one of them.
    """
    rng = np.random.default_rng(seed)
    matrix = matrix.tocsr()
    is_test = np.zeros(matrix.nnz, dtype=bool)

    for u in range(matrix.shape[0]):
        start, end = matrix.indptr[u], matrix.indptr[u + 1]
        n = end - start
        if n < 2:
            continue
        n_test = max(1, int(round(test_frac * n)))
        is_test[start + rng.choice(n, size=n_test, replace=False)] = True

    return _keep_entries(matrix, ~is_test), _keep_entries(matrix, is_test)


def _keep_entries(matrix, mask):
    coo = matrix.tocoo()
    return sp.csr_matrix(
        (coo.data[mask], (coo.row[mask], coo.col[mask])), shape=matrix.shape
    )
