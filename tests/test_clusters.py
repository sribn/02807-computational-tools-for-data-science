import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from recommender import clusters


def three_directions(seed=0, per_group=40):
    """Points around three directions, with very different lengths (like popular vs niche games)."""
    rng = np.random.default_rng(seed)
    centres = np.eye(3)
    groups = [c + rng.normal(scale=0.15, size=(per_group, 3)) for c in centres]
    lengths = rng.uniform(0.1, 10.0, size=(3 * per_group, 1))
    return np.vstack(groups) * lengths, np.repeat([0, 1, 2], per_group)


def test_unit_rows_have_length_one():
    out = clusters.unit_rows(np.array([[3.0, 4.0], [0.0, 2.0]]))
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_spherical_kmeans_groups_by_direction_not_length():
    vectors, truth = three_directions()
    labels, centroids, _ = clusters.spherical_kmeans(vectors, 3)
    assert adjusted_rand_score(truth, labels) == 1.0
    assert np.allclose(np.linalg.norm(centroids, axis=1), 1.0)


def test_spherical_kmeans_agrees_with_sklearn_on_unit_vectors():
    vectors, _ = three_directions(seed=1)
    ours, _, _ = clusters.spherical_kmeans(vectors, 3)
    theirs = KMeans(3, n_init=10, random_state=0).fit_predict(clusters.unit_rows(vectors))
    assert adjusted_rand_score(ours, theirs) == 1.0


def test_more_clusters_than_distinct_directions_leaves_none_empty():
    vectors, _ = three_directions(per_group=10)
    labels, _, _ = clusters.spherical_kmeans(vectors, 6)
    assert np.unique(labels).size == 6


def test_top_genre_tag_skips_meta_tags():
    assert clusters.top_genre_tag(["Indie", "Free to Play", "Racing"]) == "Racing"
    assert clusters.top_genre_tag(["Indie"]) is None


def test_tag_profile_needs_tags_to_be_common_and_over_represented():
    labels = np.array([0] * 6 + [1] * 6)
    tags = [["Racing", "Indie"]] * 5 + [["Rare"]] + [["RPG", "Indie"]] * 6
    table = clusters.tag_profile(labels, tags, min_share=0.5, top=1).set_index("cluster")
    # "Rare" has the higher lift in cluster 0 but is on only 1 of its 6 games.
    assert table.loc[0, "tags (share in cluster vs overall)"].startswith("Racing (83% vs 42%)")
    assert table.loc[1, "tags (share in cluster vs overall)"].startswith("RPG (100% vs 50%)")


def test_ami_is_one_for_perfect_clusters_and_near_zero_when_shuffled():
    labels = np.repeat([0, 1, 2], 20)
    tags = [["Indie", "A"]] * 20 + [["B"]] * 20 + [["C"]] * 20
    ami, shuffled = clusters.ami_with_top_tag(labels, tags)
    assert np.isclose(ami, 1.0)
    assert abs(shuffled) < 0.1


def test_largest_cluster_share_counts_recommended_slots():
    labels = np.array([0, 0, 1, 2])
    recs = np.array([[0, 1], [0, 2]])
    assert clusters.largest_cluster_share(recs, labels) == 0.75


def test_tag_profile_leaves_out_common_tags_that_are_under_represented():
    labels = np.array([0] * 4 + [1] * 4)
    tags = [["Action", "Puzzle"]] * 2 + [["Puzzle"]] * 2 + [["Action"]] * 4
    table = clusters.tag_profile(labels, tags, min_share=0.2, top=4).set_index("cluster")
    # Action is on half of cluster 0's games but on 75% of all games.
    assert table.loc[0, "tags (share in cluster vs overall)"] == "Puzzle (100% vs 50%)"
