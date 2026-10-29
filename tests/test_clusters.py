import numpy as np

from recommender import clusters


def test_unit_rows_have_length_one():
    out = clusters.unit_rows(np.array([[3.0, 4.0], [0.0, 2.0]]))
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_tag_lift_finds_the_tag_that_defines_each_cluster():
    labels = np.array([0] * 6 + [1] * 6)
    tags = [["Racing", "Indie"]] * 6 + [["RPG", "Indie"]] * 6

    table = clusters.tag_lift(labels, tags, min_games=1, top=1).set_index("cluster")

    assert table.loc[0, "top tags (lift)"].startswith("Racing (2.0)")
    assert table.loc[1, "top tags (lift)"].startswith("RPG (2.0)")


def test_nmi_is_one_for_perfect_clusters_and_lower_when_shuffled():
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    tags = [["A"]] * 3 + [["B"]] * 3 + [["C"]] * 3
    nmi, shuffled = clusters.nmi_with_top_tag(labels, tags)
    assert np.isclose(nmi, 1.0)
    assert shuffled < 0.9

