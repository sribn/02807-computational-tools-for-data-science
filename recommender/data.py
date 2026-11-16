"""Load the Steam inventories and turn them into a sparse user x game playtime matrix.

This is the one data definition shared by every method (ALS, Apriori/PCY rules, item-kNN),
so all of them see the same users, games and train/validation/test split.
"""

import ast
import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA_DIR = Path(__file__).resolve().parent.parent / "dataset"
ITEMS_FILE = "australian_users_items.json.gz"

MIN_PLAYERS = 50  # a game needs this many players for any method to learn something about it
MIN_PLAYED = 5    # users need enough played games to hide some for validation and testing


@dataclass(frozen=True)
class Interactions:
    """Playtime matrix (users x games, minutes) plus the ids behind each row and column."""

    playtime: sp.csr_matrix
    user_ids: np.ndarray
    item_ids: np.ndarray
    item_names: dict
    filter_log: pd.DataFrame  # users / games / played pairs after each filtering pass


def read_user_items(path=DATA_DIR / ITEMS_FILE):
    """Return one row per (user, game) entry in the raw file, duplicates included."""
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = ast.literal_eval(line)
            for item in record["items"]:
                rows.append((record["user_id"], item["item_id"], item["item_name"], item["playtime_forever"]))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "item_name", "playtime"])


def deduplicate(user_items):
    """Keep one row per (user, game).

    The raw file repeats some users; a few repeated entries disagree on playtime, and we keep
    the largest, since playtime_forever only grows over time.
    """
    return (
        user_items.sort_values("playtime", ascending=False, kind="stable")
        .drop_duplicates(subset=["user_id", "item_id"])
        .sort_values(["user_id", "item_id"], ignore_index=True)
    )


def load_user_items(path=DATA_DIR / ITEMS_FILE):
    """Raw entries with duplicates removed."""
    return deduplicate(read_user_items(path))


def filter_played(user_items, min_players=MIN_PLAYERS, min_played=MIN_PLAYED):
    """Keep played entries of games with enough players and users with enough played games.

    Owned-but-unplayed games count as absent: bundles and giveaways make ownership weak
    evidence of taste, and every method is trained and scored on plays. Dropping games can
    push users below `min_played` and the other way round, so the two filters repeat until
    nothing changes. Returns the kept rows and a log of the counts after each pass.
    """
    played = user_items[user_items["playtime"] > 0]
    log = [_counts(played, "played entries")]

    for n_pass in range(1, 100):
        players = played.groupby("item_id")["user_id"].size()
        played = played[played["item_id"].isin(players.index[players >= min_players])]
        games_per_user = played.groupby("user_id")["item_id"].size()
        kept = played[played["user_id"].isin(games_per_user.index[games_per_user >= min_played])]
        log.append(_counts(kept, f"filter pass {n_pass}"))
        if len(kept) == len(played):
            break
        played = kept

    return kept.reset_index(drop=True), pd.DataFrame(log).set_index("step")


def build_interactions(user_items, min_players=MIN_PLAYERS, min_played=MIN_PLAYED):
    """Filter the entries (see `filter_played`) and build the sparse playtime matrix."""
    played, log = filter_played(user_items, min_players, min_played)

    user_codes, user_ids = pd.factorize(played["user_id"], sort=True)
    item_codes, item_ids = pd.factorize(played["item_id"], sort=True)
    playtime = sp.csr_matrix(
        (played["playtime"].to_numpy(dtype=np.float64), (user_codes, item_codes)),
        shape=(len(user_ids), len(item_ids)),
    )
    item_names = played.drop_duplicates("item_id").set_index("item_id")["item_name"].to_dict()
    return Interactions(playtime, np.asarray(user_ids), np.asarray(item_ids), item_names, log)


def players_per_game(matrix):
    """Number of users with a stored entry for each game (column)."""
    return np.diff(matrix.tocsc().indptr)


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


def _counts(played, step):
    return {
        "step": step,
        "users": played["user_id"].nunique(),
        "games": played["item_id"].nunique(),
        "played pairs": len(played),
    }


def _keep_entries(matrix, mask):
    coo = matrix.tocoo()
    return sp.csr_matrix(
        (coo.data[mask], (coo.row[mask], coo.col[mask])), shape=matrix.shape
    )
