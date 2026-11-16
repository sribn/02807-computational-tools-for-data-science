"""Re-rank recommendations so that no cluster of games fills more than `cap` slots of a list."""

import numpy as np
from numba import njit, prange

from recommender.evaluate import BATCH_SIZE


def capped_top_k(score_batch, train, labels, cap, k=10, batch_size=BATCH_SIZE):
    """Top-k lists that take at most `cap` games from any one cluster.

    Walks each user's full ranking (training games excluded) and takes the best game whose
    cluster still has room; if the caps cannot fill all k slots, the remaining slots go to
    the best games skipped so far. cap >= k gives the plain top-k list.
    """
    train = train.tocsr()
    n_users = train.shape[0]
    labels = np.asarray(labels, dtype=np.int64)
    recs = np.empty((n_users, k), dtype=np.int64)

    for start in range(0, n_users, batch_size):
        rows = np.arange(start, min(start + batch_size, n_users))
        scores = np.array(score_batch(rows), dtype=np.float64)
        owned = train[rows].tocoo()
        scores[owned.row, owned.col] = -np.inf
        order = np.argsort(-scores, axis=1, kind="stable")
        n_allowed = scores.shape[1] - np.diff(train.indptr)[rows]
        recs[rows] = _greedy_cap(order, n_allowed, labels, cap, k, labels.max() + 1)

    return recs


@njit(parallel=True, cache=True)
def _greedy_cap(order, n_allowed, labels, cap, k, n_clusters):
    out = np.empty((order.shape[0], k), dtype=np.int64)
    for r in prange(order.shape[0]):
        used = np.zeros(n_clusters, dtype=np.int64)
        taken = np.zeros(order.shape[1], dtype=np.bool_)
        filled = 0
        for p in range(n_allowed[r]):
            if filled == k:
                break
            game = order[r, p]
            if used[labels[game]] < cap:
                used[labels[game]] += 1
                taken[p] = True
                out[r, filled] = game
                filled += 1
        # Not enough clusters among the candidates: fill up with the best games skipped.
        p = 0
        while filled < k and p < n_allowed[r]:
            if not taken[p]:
                out[r, filled] = order[r, p]
                filled += 1
            p += 1
    return out
