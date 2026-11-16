"""Top-k recommendation, ranking metrics on held-out games, and their uncertainty."""

import numpy as np

BATCH_SIZE = 4096  # users scored at once; keeps the score matrix to a few hundred MB
N_BOOT = 10_000    # bootstrap resamples of the test users (Bonferroni tails need many)
HEAD_SHARE = 0.2   # the most-played 20% of games count as the "head", the rest as the long tail


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


def popularity_counts(train):
    """Number of training users who played each game."""
    return np.asarray((train > 0).sum(axis=0)).ravel().astype(np.float64)


def popularity_scorer(train):
    """Score every game by how many training users played it, the same for everyone."""
    counts = popularity_counts(train)
    return lambda rows: np.broadcast_to(counts, (len(rows), counts.size))


def per_user_metrics(recs, test):
    """Precision@k, recall@k and NDCG@k for every user with at least one test game.

    Recall divides by all of the user's test games, so it stays below 1 for users with more
    than k of them; NDCG's ideal ranking puts min(k, |test|) hits first.
    """
    test = test.tocsr()
    n_users, k = recs.shape
    truth = test.copy()
    truth.data = np.ones_like(truth.data)
    hits = np.asarray(truth[np.repeat(np.arange(n_users), k), recs.ravel()]).reshape(n_users, k) > 0
    n_test = np.diff(test.indptr)
    has_test = n_test > 0

    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    ideal = np.cumsum(discounts)[np.minimum(k, np.maximum(n_test, 1)) - 1]
    n_hits = hits.sum(axis=1)
    return {
        f"precision@{k}": (n_hits / k)[has_test],
        f"recall@{k}": (n_hits / np.maximum(n_test, 1))[has_test],
        f"ndcg@{k}": ((hits * discounts).sum(axis=1) / ideal)[has_test],
    }


def ranking_metrics(recs, test):
    """Mean precision@k, recall@k and NDCG@k over users with at least one test game."""
    per_user = per_user_metrics(recs, test)
    means = {name: float(values.mean()) for name, values in per_user.items()}
    return {**means, "users": len(next(iter(per_user.values())))}


def catalogue_coverage(recs, n_items):
    """Share of all games that appear in at least one user's recommendations."""
    return np.unique(recs).size / n_items


def long_tail_share(recs, popularity, head_share=HEAD_SHARE):
    """Share of recommended slots filled with games outside the most-played `head_share`."""
    n_head = int(round(head_share * popularity.size))
    head = np.zeros(popularity.size, dtype=bool)
    head[np.argsort(-popularity, kind="stable")[:n_head]] = True
    return float(1.0 - head[recs].mean())


def intra_list_similarity(recs, sim, batch_size=BATCH_SIZE):
    """Mean similarity between the games within each list (lower = more varied lists)."""
    k = recs.shape[1]
    off_diagonal = ~np.eye(k, dtype=bool)
    totals = []
    for start in range(0, recs.shape[0], batch_size):
        batch = recs[start:start + batch_size]
        pairwise = sim[batch[:, :, None], batch[:, None, :]]
        totals.append(pairwise[:, off_diagonal].mean(axis=1))
    return float(np.concatenate(totals).mean())


def bootstrap_interval(values, n_boot=N_BOOT, seed=0, level=0.95):
    """Percentile interval of the mean, resampling users with replacement."""
    rng = np.random.default_rng(seed)
    means = np.array([values[rng.integers(0, values.size, values.size)].mean() for _ in range(n_boot)])
    tail = (1 - level) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def paired_difference(a, b, n_boot=N_BOOT, seed=0, level=0.95):
    """Mean of a - b over the same users, with a paired bootstrap interval and its SE.

    Both methods are scored on the same users, so resampling the per-user differences removes
    the variation from some users simply being easier to predict than others.
    """
    diff = np.asarray(a) - np.asarray(b)
    low, high = bootstrap_interval(diff, n_boot, seed, level)
    return {"difference": float(diff.mean()), "low": low, "high": high,
            "se": float(diff.std(ddof=1) / np.sqrt(diff.size))}
