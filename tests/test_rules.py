import numpy as np
import pytest
import scipy.sparse as sp

from recommender import itemsets, rules


def toy():
    # Users 0-2 play games 0, 1, 2 together; user 3 plays 0 and 1; user 4 plays 0 and 3.
    dense = np.zeros((5, 4))
    dense[:3, :3] = 1.0
    dense[3, [0, 1]] = 1.0
    dense[4, [0, 3]] = 1.0
    train = sp.csr_matrix(dense)
    b = itemsets.baskets(train)
    pairs = itemsets.pcy_pairs(b, min_support=1, n_buckets=101)
    return train, pairs, itemsets.apriori_triples(b, pairs, min_support=1)


def test_pair_rule_confidence_is_conditional_probability():
    _, pairs, _ = toy()
    conf = rules.pair_rule_confidences(pairs)
    assert conf[0, 1] == pytest.approx(4 / 5)  # 4 of the 5 players of game 0 played game 1
    assert conf[1, 0] == pytest.approx(1.0)
    assert conf[0, 0] == 0.0


def test_triple_rule_confidence_uses_the_pair_premise():
    _, pairs, triples = toy()
    tr = rules.triple_rules(pairs, triples)
    row = tr.premise[0, 1]
    found = dict(zip(tr.consequent[tr.ptr[row]:tr.ptr[row + 1]], tr.confidence[tr.ptr[row]:tr.ptr[row + 1]]))
    assert found[2] == pytest.approx(3 / 4)  # {0,1} played by 4 users, 3 of them played 2
    assert tr.n_rules == 3 * len(triples.support)


def test_max_and_sum_aggregate_the_rules_a_user_can_fire():
    train, pairs, triples = toy()
    conf = rules.pair_rule_confidences(pairs)
    tr = rules.triple_rules(pairs, triples)
    user = np.array([3])  # played games 0 and 1

    best = rules.association_rule_scorer(train, conf, tr, "max")(user)[0]
    total = rules.association_rule_scorer(train, conf, tr, "sum")(user)[0]
    # Rules into game 2: 0 -> 2 (3/5), 1 -> 2 (3/4), {0,1} -> 2 (3/4).
    assert best[2] == pytest.approx(3 / 4, abs=1e-6)
    assert total[2] == pytest.approx(3 / 5 + 3 / 4 + 3 / 4, abs=1e-6)


def test_pairs_only_scorer_ignores_triples():
    train, pairs, _ = toy()
    conf = rules.pair_rule_confidences(pairs)
    total = rules.association_rule_scorer(train, conf, None, "sum")(np.array([3]))[0]
    assert total[2] == pytest.approx(3 / 5 + 3 / 4, abs=1e-6)


def test_unknown_aggregation_is_rejected():
    train, pairs, _ = toy()
    with pytest.raises(ValueError):
        rules.association_rule_scorer(train, rules.pair_rule_confidences(pairs), aggregate="mean")
