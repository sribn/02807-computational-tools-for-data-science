"""Course-method recommenders to compare ALS against: association rules and item-item similarity.

Both start from the same pair counts: entry (i, j) of B^T B, where B is the binary
user x game "played" matrix, is the number of users who played both games. Keeping
pairs with count >= min_support gives exactly the frequent pairs that Apriori/PCY
find (those algorithms only prune which candidate pairs they count).
"""

import numpy as np
import scipy.sparse as sp
from numba import njit, prange

from recommender.evaluate import popularity_counts

TIE_BREAK = 1e-9  # fall back to popularity when no rule or neighbour fires for a game


def played(matrix):
    """Binary version of a playtime matrix."""
    matrix = matrix.tocsr()
    return sp.csr_matrix(
        (np.ones_like(matrix.data), matrix.indices, matrix.indptr), shape=matrix.shape
    )


def pair_counts(train):
    """Dense games x games co-play counts; the diagonal holds each game's own support."""
    b = played(train)
    return (b.T @ b).toarray().astype(np.float32)


def rule_confidences(counts, min_support=50, min_confidence=0.0):
    """Confidence of every rule i -> j, i.e. P(played j | played i), for frequent pairs only."""
    support = np.diag(counts).copy()
    conf = counts / np.maximum(support, 1)[:, None]
    conf[(counts < min_support) | (conf < min_confidence)] = 0.0
    np.fill_diagonal(conf, 0.0)
    return conf


def association_rule_scorer(train, conf):
    """Score game j for a user by the strongest rule i -> j over the games i they played."""
    train = train.tocsr()
    tie_break = TIE_BREAK * popularity_counts(train)

    def score(rows):
        return _max_rule(train.indptr, train.indices, conf, np.asarray(rows)) + tie_break

    return score


@njit(parallel=True, cache=True)
def _max_rule(indptr, indices, conf, rows):
    out = np.zeros((rows.size, conf.shape[1]), dtype=np.float32)
    for r in prange(rows.size):
        u = rows[r]
        for p in range(indptr[u], indptr[u + 1]):
            antecedent = conf[indices[p]]
            for j in range(conf.shape[1]):
                if antecedent[j] > out[r, j]:
                    out[r, j] = antecedent[j]
    return out


def cosine_neighbours(counts, k=50):
    """Item-item cosine similarity of the played vectors, keeping each game's top-k neighbours."""
    support = np.sqrt(np.maximum(np.diag(counts), 1))
    sim = counts / support[:, None] / support[None, :]
    np.fill_diagonal(sim, 0.0)
    if k < sim.shape[1]:
        drop = np.argpartition(-sim, k, axis=1)[:, k:]
        sim = sim.copy()
        np.put_along_axis(sim, drop, 0.0, axis=1)
    return sim


def item_knn_scorer(train, sim):
    """Score game j for a user by summing its similarity to every game they played."""
    b = played(train)
    tie_break = TIE_BREAK * popularity_counts(train)
    # Rows of `sim` are the played game's neighbours, so the user's score is b_u @ sim.
    return lambda rows: np.asarray(b[rows] @ sim) + tie_break
