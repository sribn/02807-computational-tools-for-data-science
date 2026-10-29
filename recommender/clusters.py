"""Cluster the learned game vectors and check whether the clusters line up with Steam tags."""

import ast
import gzip

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, normalized_mutual_info_score

from recommender.data import DATA_DIR

GAMES_FILE = "steam_games.json.gz"


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
    """Scale each vector to length 1, so k-means groups by direction (cosine), not popularity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def kmeans_sweep(vectors, ks, seed=0):
    """Fit k-means for each k; return a table of Davies-Bouldin scores and the fitted labels."""
    rows, labels = [], {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(vectors)
        labels[k] = km.labels_
        rows.append({"k": k, "davies_bouldin": davies_bouldin_score(vectors, km.labels_)})
    return pd.DataFrame(rows), labels


def tag_lift(labels, tag_lists, min_games=5, top=5):
    """Per cluster, the tags most over-represented relative to all games (lift = P(tag|c) / P(tag))."""
    long = pd.DataFrame(
        [(c, t) for c, tags in zip(labels, tag_lists) for t in tags], columns=["cluster", "tag"]
    )
    games_per_cluster = pd.Series(labels).value_counts()
    overall = long["tag"].value_counts() / len(labels)

    rows = []
    for cluster, group in long.groupby("cluster"):
        counts = group["tag"].value_counts()
        counts = counts[counts >= min_games]
        lift = (counts / games_per_cluster[cluster]) / overall[counts.index]
        best = lift.sort_values(ascending=False).head(top)
        rows.append({
            "cluster": cluster,
            "games": int(games_per_cluster[cluster]),
            "top tags (lift)": ", ".join(f"{t} ({v:.1f})" for t, v in best.items()),
        })
    return pd.DataFrame(rows).sort_values("games", ascending=False, ignore_index=True)


def nmi_with_top_tag(labels, tag_lists, seed=0, shuffles=20):
    """NMI between clusters and each game's most-voted tag, plus the same for shuffled labels."""
    top_tags = [tags[0] for tags in tag_lists]
    nmi = normalized_mutual_info_score(top_tags, labels)
    rng = np.random.default_rng(seed)
    shuffled = [normalized_mutual_info_score(top_tags, rng.permutation(labels)) for _ in range(shuffles)]
    return nmi, float(np.mean(shuffled))
