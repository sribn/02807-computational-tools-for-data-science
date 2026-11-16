"""Item-item similarity recommender (the association-rule recommender lives in rules.py).

Entry (i, j) of B^T B, where B is the binary user x game "played" matrix, is the number of
users who played both games; the similarities are computed from these counts.
"""

import numpy as np
import scipy.sparse as sp

from recommender.evaluate import popularity_counts

TIE_BREAK = 1e-9  # fall back to popularity when no neighbour fires for a game


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
