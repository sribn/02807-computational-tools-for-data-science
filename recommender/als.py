"""Implicit-feedback ALS (Hu, Koren & Volinsky, 2008), implemented with NumPy + Numba.

Every user-game cell has a preference p (1 if played, else 0) and a confidence c:
c = 1 + alpha for played cells and c = 1 for unplayed cells. ALS minimises

    sum_ui c_ui (p_ui - x_u . y_i)^2 + reg * (sum_u |x_u|^2 + sum_i |y_i|^2)

by alternately solving for all user vectors with the game vectors fixed, and vice versa.
The confidence matrix must store exactly the played cells: every stored entry is read as
p = 1, so an explicit zero would wrongly count as played.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numba import njit, prange


@dataclass(frozen=True)
class ALSModel:
    user_factors: np.ndarray
    item_factors: np.ndarray


def confidence(played, alpha):
    """Confidence values for the played cells (the stored entries of `played`): 1 + alpha each."""
    played = played.tocsr()
    return sp.csr_matrix(
        (np.full(played.nnz, 1.0 + alpha), played.indices, played.indptr), shape=played.shape
    )


def initial_factors(shape, factors, seed=0):
    """Small random starting vectors for (users, games); the first sweep overwrites the users."""
    rng = np.random.default_rng(seed)
    users = rng.normal(scale=0.01, size=(shape[0], factors))
    items = rng.normal(scale=0.01, size=(shape[1], factors))
    return users, items


def fit(conf, factors=32, reg=1.0, iterations=15, seed=0, callback=None):
    """Fit user and game vectors to a CSR confidence matrix (users x games).

    `callback(iteration, model)` is called after each sweep, e.g. to track metrics.
    """
    conf = conf.tocsr().astype(np.float64)
    conf_t = conf.T.tocsr()
    users, items = initial_factors(conf.shape, factors, seed)

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
    """The ALS objective, computed without forming the dense users x games matrix.

    With s_ui = x_u . y_i and S the played cells,

        loss = trace(X^T X Y^T Y) + sum_{(u,i) in S} [c_ui (1 - s_ui)^2 - s_ui^2] + reg (|X|^2 + |Y|^2)

    The trace equals the sum of s_ui^2 over all cells, i.e. the loss if nothing were played;
    the sum then swaps in the right term for the played cells (p = 1, confidence c).
    """
    x, y = model.user_factors, model.item_factors
    conf = conf.tocsr()
    every_cell = float(np.sum((x.T @ x) * (y.T @ y)))
    played = _played_correction(conf.indptr, conf.indices, conf.data.astype(np.float64), x, y)
    penalty = float((x ** 2).sum() + (y ** 2).sum())
    return every_cell + played + reg * penalty


@njit(parallel=True, cache=True)
def _played_correction(indptr, indices, data, x, y):
    total = 0.0
    for u in prange(indptr.size - 1):
        for p in range(indptr[u], indptr[u + 1]):
            s = 0.0
            for f in range(x.shape[1]):
                s += x[u, f] * y[indices[p], f]
            total += data[p] * (1.0 - s) ** 2 - s ** 2
    return total


def similar_items(model, item, n=10):
    """Return the `n` games with the most similar vectors (cosine), excluding `item`."""
    vecs = model.item_factors
    norms = np.linalg.norm(vecs, axis=1)
    sims = vecs @ vecs[item] / (norms * norms[item] + 1e-12)
    sims[item] = -np.inf
    top = np.argpartition(-sims, n)[:n]
    return top[np.argsort(-sims[top])], sims
