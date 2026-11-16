"""Cluster the learned game vectors (spherical k-means) and compare the clusters with Steam tags.

ALS scores a user-game pair by a dot product, so a game's taste profile is the direction of
its vector, while the length mostly reflects how much data the game has. We therefore cluster
directions: spherical k-means assigns each unit vector to the centroid with the highest cosine
similarity and re-normalises each centroid to unit length after every update.
"""

import ast
import gzip

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, davies_bouldin_score

from recommender.data import DATA_DIR

GAMES_FILE = "steam_games.json.gz"
N_INIT = 10     # k-means restarts; the best (highest total cosine) is kept
MAX_ITER = 100
# Tags about business model, release stage, player count or production rather than content.
META_TAGS = frozenset({
    "Indie", "Free to Play", "Early Access", "Great Soundtrack", "Singleplayer", "Multiplayer",
    "Co-op", "Online Co-Op", "Local Co-Op", "Local Multiplayer", "Controller", "Classic",
    "Short", "Moddable",
})


def load_game_tags(path=DATA_DIR / GAMES_FILE):
    """Map game id -> list of Steam tags (most-voted first) for games that have tags."""
    tags = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            game = ast.literal_eval(line)
            if game.get("id") and game.get("tags"):
                tags[game["id"]] = list(game["tags"])
    return tags


def unit_rows(vectors):
    """Scale each vector to length 1, so games are compared by direction (cosine)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def spherical_kmeans(vectors, k, seed=0, n_init=N_INIT, max_iter=MAX_ITER):
    """Cluster vectors by cosine similarity; return (labels, unit centroids, total cosine)."""
    x = unit_rows(np.asarray(vectors, dtype=np.float64))
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        result = _spherical_kmeans_once(x, k, rng, max_iter)
        if best is None or result[2] > best[2]:
            best = result
    return best


def _spherical_kmeans_once(x, k, rng, max_iter):
    centroids = _kmeans_plus_plus(x, k, rng)
    labels = np.full(x.shape[0], -1)
    for _ in range(max_iter):
        new_labels = np.argmax(x @ centroids.T, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        sums = np.zeros_like(centroids)
        np.add.at(sums, labels, x)
        empty = ~sums.any(axis=1)
        if empty.any():
            # Re-seed empty clusters with the points least similar to their own centroid.
            fit = np.einsum("ij,ij->i", x, centroids[labels])
            sums[empty] = x[np.argsort(fit)[: empty.sum()]]
        centroids = unit_rows(sums)
    total_cosine = float(np.einsum("ij,ij->i", x, centroids[labels]).sum())
    return labels, centroids, total_cosine


def _kmeans_plus_plus(x, k, rng):
    """k-means++ seeding with the cosine distance 1 - cos (squared distance of unit vectors / 2)."""
    centroids = [x[rng.integers(x.shape[0])]]
    distance = 1.0 - x @ centroids[0]
    for _ in range(1, k):
        weights = np.maximum(distance, 0) ** 2
        pick = rng.choice(x.shape[0], p=weights / weights.sum()) if weights.sum() > 0 else rng.integers(x.shape[0])
        centroids.append(x[pick])
        distance = np.minimum(distance, 1.0 - x @ x[pick])
    return np.array(centroids)


def kmeans_sweep(vectors, ks, seed=0):
    """Spherical k-means for each k; return a table of Davies-Bouldin scores and the labels."""
    x = unit_rows(vectors)
    rows, labels = [], {}
    for k in ks:
        labels[k], _, _ = spherical_kmeans(x, k, seed)
        rows.append({"k": k, "davies_bouldin": davies_bouldin_score(x, labels[k])})
    return pd.DataFrame(rows), labels


def seed_agreement(vectors, k, seeds):
    """Mean adjusted Rand index between clusterings from different random seeds (1 = identical)."""
    runs = [spherical_kmeans(vectors, k, seed)[0] for seed in seeds]
    scores = [adjusted_rand_score(a, b) for i, a in enumerate(runs) for b in runs[i + 1:]]
    return float(np.mean(scores))


def largest_cluster_share(recs, labels):
    """Share of all recommended slots that fall in the single most-recommended cluster."""
    counts = np.bincount(labels[recs].ravel(), minlength=labels.max() + 1)
    return float(counts.max() / counts.sum())


def top_genre_tag(tags):
    """A game's most-voted tag that describes content (meta tags skipped), or None."""
    return next((t for t in tags if t not in META_TAGS), None)


def tag_profile(labels, tag_lists, min_share=0.2, top=4):
    """Per cluster, the tags that are both common in it and over-represented.

    A tag must be on at least `min_share` of the cluster's games; tags are ranked by lift,
    P(tag | cluster) / P(tag), and shown as "tag (share in cluster vs share overall)".
    """
    long = pd.DataFrame(
        [(c, t) for c, tags in zip(labels, tag_lists) for t in tags], columns=["cluster", "tag"]
    )
    games_per_cluster = pd.Series(labels).value_counts()
    overall = long["tag"].value_counts() / len(labels)

    rows = []
    for cluster, group in long.groupby("cluster"):
        share = group["tag"].value_counts() / games_per_cluster[cluster]
        share = share[share >= min_share]
        lift = (share / overall[share.index]).sort_values(ascending=False).head(top)
        rows.append({
            "cluster": cluster,
            "games": int(games_per_cluster[cluster]),
            "tags (share in cluster vs overall)": ", ".join(
                f"{t} ({share[t]:.0%} vs {overall[t]:.0%})" for t in lift.index
            ),
        })
    return pd.DataFrame(rows).sort_values("games", ascending=False, ignore_index=True)


def ami_with_top_tag(labels, tag_lists, seed=0, shuffles=20):
    """Adjusted mutual information between clusters and each game's top content tag.

    Games without a content tag are left out. Also returns the mean AMI of randomly shuffled
    cluster labels, which should be about 0.
    """
    top = [top_genre_tag(tags) for tags in tag_lists]
    keep = np.array([t is not None for t in top])
    top = [t for t in top if t is not None]
    labels = np.asarray(labels)[keep]
    ami = adjusted_mutual_info_score(top, labels)
    rng = np.random.default_rng(seed)
    shuffled = [adjusted_mutual_info_score(top, rng.permutation(labels)) for _ in range(shuffles)]
    return float(ami), float(np.mean(shuffled))
