"""Top-k recommendation and ranking metrics on held-out games."""

import numpy as np

BATCH_SIZE = 4096  # users scored at once; keeps the score matrix to a few hundred MB


def top_k(score_batch, train, k=10, batch_size=BATCH_SIZE):
    """Return a (users x k) array of game indices, best first, never recommending train games.

    `score_batch(rows)` returns the scores for those user rows (len(rows) x n_items).
    """
    train = train.tocsr()
    n_users = train.shape[0]
    recs = np.empty((n_users, k), dtype=np.int64)

    for start in range(0, n_users, batch_size):
        rows = np.arange(start, min(start + batch_size, n_users))
        scores = np.array(score_batch(rows), dtype=np.float64)
        owned = train[rows].tocoo()
        scores[owned.row, owned.col] = -np.inf

        part = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
        recs[rows] = np.take_along_axis(part, order, axis=1)

    return recs


def als_scorer(model):
    return lambda rows: model.user_factors[rows] @ model.item_factors.T


def popularity_scorer(train):
    """Score every game by how many training users played it, the same for everyone."""
    counts = np.asarray((train > 0).sum(axis=0)).ravel().astype(np.float64)
    return lambda rows: np.broadcast_to(counts, (len(rows), counts.size))


def ranking_metrics(recs, test):
    """Mean precision@k, recall@k and NDCG@k over users with at least one test game."""
    test = test.tocsr()
    k = recs.shape[1]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    precision, recall, ndcg = [], [], []

    for u in range(test.shape[0]):
        truth = test.indices[test.indptr[u]:test.indptr[u + 1]]
        if truth.size == 0:
            continue
        hits = np.isin(recs[u], truth)
        precision.append(hits.sum() / k)
        recall.append(hits.sum() / truth.size)
        ideal = discounts[: min(k, truth.size)].sum()
        ndcg.append((hits * discounts).sum() / ideal)

    return {
        f"precision@{k}": float(np.mean(precision)),
        f"recall@{k}": float(np.mean(recall)),
        f"ndcg@{k}": float(np.mean(ndcg)),
        "users": len(precision),
    }


def catalogue_coverage(recs, n_items):
    """Share of all games that appear in at least one user's recommendations."""
    return np.unique(recs).size / n_items
