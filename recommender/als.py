"""Implicit-feedback ALS (Hu, Koren & Volinsky, 2008), implemented with NumPy + Numba.

Every user-game cell has a preference p (1 if played, else 0) and a confidence
c = 1 + alpha * log(1 + playtime), with c = 1 for unplayed cells. ALS minimises

    sum_ui c_ui (p_ui - x_u . y_i)^2 + reg * (sum_u |x_u|^2 + sum_i |y_i|^2)

by alternately solving for all user vectors with the game vectors fixed, and vice versa.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numba import njit, prange


@dataclass(frozen=True)
class ALSModel:
    user_factors: np.ndarray
    item_factors: np.ndarray


def confidence(playtime, alpha):
    """Map a playtime matrix (minutes) to confidence values for the played cells."""
    playtime = playtime.tocsr()
    return sp.csr_matrix(
        (1.0 + alpha * np.log1p(playtime.data), playtime.indices, playtime.indptr),
        shape=playtime.shape,
    )


def fit(conf, factors=32, reg=1.0, iterations=15, seed=0, callback=None):
    """Fit user and game vectors to a CSR confidence matrix (users x games).

    `callback(iteration, model)` is called after each sweep, e.g. to track metrics.
    """
    rng = np.random.default_rng(seed)
    conf = conf.tocsr().astype(np.float64)
    conf_t = conf.T.tocsr()

    users = rng.normal(scale=0.01, size=(conf.shape[0], factors))
    items = rng.normal(scale=0.01, size=(conf.shape[1], factors))

    for it in range(iterations):
        users = solve_rows(conf, items, reg)
        items = solve_rows(conf_t, users, reg)
        if callback is not None:
            callback(it, ALSModel(users, items))

    return ALSModel(users, items)


def solve_rows(conf, fixed, reg):
    """Solve every row's vector given the other side's `fixed` vectors (one ALS half-step)."""
    fixed = np.ascontiguousarray(fixed, dtype=np.float64)
    # Shared by every row: the speed-up from the paper, Y^T C_u Y = Y^T Y + Y^T (C_u - I) Y.
    gram = fixed.T @ fixed + reg * np.eye(fixed.shape[1])
    return _solve_rows(conf.indptr, conf.indices, conf.data, fixed, gram)


@njit(parallel=True, cache=True)
def _solve_rows(indptr, indices, data, fixed, gram):
    n_rows = len(indptr) - 1
    out = np.zeros((n_rows, fixed.shape[1]))
    for u in prange(n_rows):
        start, end = indptr[u], indptr[u + 1]
        if start == end:
            continue
        y = fixed[indices[start:end]]
        c = data[start:end]
        # Only the played games add to the shared Gram matrix; p = 1 for exactly those.
        # Explicit loop: Numba's parallel mode returned zeros for `(c - 1.0)[:, None] * y`.
        weighted = np.empty_like(y)
        for j in range(y.shape[0]):
            weighted[j] = (c[j] - 1.0) * y[j]
        a = gram + y.T @ weighted
        b = y.T @ c
        out[u] = np.linalg.solve(a, b)
    return out


def loss(conf, model, reg):
    """Full objective, computed densely. Only meant for small matrices (tests, sanity checks)."""
    dense_c = np.ones(conf.shape)
    dense_p = np.zeros(conf.shape)
    coo = conf.tocoo()
    dense_c[coo.row, coo.col] = coo.data
    dense_p[coo.row, coo.col] = 1.0
    err = dense_p - model.user_factors @ model.item_factors.T
    penalty = (model.user_factors ** 2).sum() + (model.item_factors ** 2).sum()
    return float((dense_c * err ** 2).sum() + reg * penalty)


def similar_items(model, item, n=10):
    """Return the `n` games with the most similar vectors (cosine), excluding `item`."""
    vecs = model.item_factors
    norms = np.linalg.norm(vecs, axis=1)
    sims = vecs @ vecs[item] / (norms * norms[item] + 1e-12)
    sims[item] = -np.inf
    top = np.argpartition(-sims, n)[:n]
    return top[np.argsort(-sims[top])], sims
