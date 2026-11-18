import numpy as np
import pandas as pd

from recommender import experiments


def test_run_grid_caches_finished_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "RESULTS_DIR", tmp_path)
    calls = []

    def evaluate(cfg):
        calls.append(cfg["a"])
        return {"score": cfg["a"] * 2.0}

    first = experiments.run_grid("g", [{"a": 1, "b": "x"}, {"a": 2.0, "b": "x"}], evaluate)
    again = experiments.run_grid("g", [{"a": 1.0, "b": "x"}, {"a": 3, "b": "x"}], evaluate)

    assert calls == [1, 2.0, 3]  # 1 == 1.0 is found in the cache
    assert first["score"].tolist() == [2.0, 4.0]
    assert sorted(again["a"].tolist()) == [1, 3]


def test_one_se_rule_prefers_the_simpler_setting_within_noise():
    grid = pd.DataFrame({
        "factors": [64, 128, 256],
        "ndcg@10": [0.200, 0.2445, 0.2450],
        "ndcg_se": [0.001, 0.001, 0.001],
    })
    chosen, best = experiments.one_se_choice(grid, [("factors", True)])
    assert best["factors"] == 256 and chosen["factors"] == 128


def test_edge_check_flags_a_best_value_at_the_end_of_its_range():
    grid = pd.DataFrame({
        "factors": [64, 128, 256, 128, 128],
        "reg": [300.0, 300.0, 300.0, 100.0, 1000.0],
        "ndcg@10": [0.1, 0.2, 0.3, 0.15, 0.18],
    })
    best = grid.loc[grid["ndcg@10"].idxmax()]
    table = experiments.edge_check(grid, best, ["factors", "reg"], ["factors", "reg"]).set_index("setting")
    assert bool(table.loc["factors", "at edge"])
    assert table.loc["reg", "tried"] == [300.0]  # no lambda neighbours at 256 factors
    assert bool(table.loc["reg", "at edge"])  # an untested direction counts as an edge


def test_stage2_works_on_concatenated_grids_with_repeated_row_labels(monkeypatch):
    monkeypatch.setattr(experiments, "ALS_REGS", [100.0, 300.0, 1000.0])
    part = pd.DataFrame({"weighting": ["none"] * 2, "alpha": [3.0] * 2, "factors": [64, 128],
                         "reg": [100.0, 300.0], "ndcg@10": [0.1, 0.2]})
    configs = experiments.als_stage2_configs(pd.concat([part, part.assign(**{"ndcg@10": [0.0, 0.0]})]))
    assert {c["reg"] for c in configs} == {100.0, 300.0, 1000.0}
    assert all(c["factors"] == 128 for c in configs)


def test_reg_extension_steps_past_an_edge_best_lambda(monkeypatch):
    monkeypatch.setattr(experiments, "ALS_REGS", [100.0, 200.0, 300.0, 400.0])
    grid = pd.DataFrame({"weighting": ["idf"] * 3 + ["none"] * 3, "alpha": [3.0] * 6,
                         "factors": [512] * 6, "reg": [200.0, 300.0, 400.0] * 2,
                         "ndcg@10": [0.30, 0.20, 0.10, 0.10, 0.30, 0.20]})
    configs = experiments.reg_extensions(grid)
    # idf's best (200) is the smallest lambda tried; none's best (300) is inside its range.
    assert configs == [{"weighting": "idf", "alpha": 3.0, "factors": 512, "reg": 100.0}]
    assert experiments.reg_extensions(pd.concat([grid, pd.DataFrame([{**configs[0], "ndcg@10": 0.2}])])) == []


def test_reg_extension_checks_every_alpha_not_only_the_best(monkeypatch):
    monkeypatch.setattr(experiments, "ALS_REGS", [100.0, 200.0, 300.0, 400.0])
    grid = pd.DataFrame({"weighting": ["none"] * 6, "alpha": [3.0] * 3 + [10.0] * 3,
                         "factors": [512] * 6, "reg": [100.0, 200.0, 300.0] * 2,
                         "ndcg@10": [0.20, 0.30, 0.25, 0.10, 0.15, 0.29]})
    # alpha = 10 is still rising at the largest lambda, although alpha = 3 holds the overall best.
    assert experiments.reg_extensions(grid) == [{"weighting": "none", "alpha": 10.0, "factors": 512, "reg": 400.0}]


def test_reg_extension_also_checks_the_one_se_pick(monkeypatch):
    monkeypatch.setattr(experiments, "ALS_REGS", [100.0, 200.0, 300.0, 400.0])
    grid = pd.DataFrame({"weighting": ["none"] * 3, "alpha": [3.0] * 3, "factors": [512] * 3,
                         "reg": [100.0, 200.0, 300.0], "ndcg@10": [0.20, 0.30, 0.295]})
    assert experiments.reg_extensions(grid) == []  # the best (200) is inside its range
    pick = grid.iloc[2]  # but a one-SE pick at the largest lambda tried needs the next lambda
    assert experiments.reg_extensions(grid, chosen=[pick]) == [
        {"weighting": "none", "alpha": 3.0, "factors": 512, "reg": 400.0}]


def test_reg_extension_steps_through_the_small_lambdas(monkeypatch):
    monkeypatch.setattr(experiments, "ALS_REGS", [30.0, 100.0])
    monkeypatch.setattr(experiments, "ALS_SMALL_REGS", [1.0, 10.0])
    grid = pd.DataFrame({"weighting": ["none"] * 2, "alpha": [3.0] * 2, "factors": [128] * 2,
                         "reg": [10.0, 30.0], "ndcg@10": [0.30, 0.20]})
    assert experiments.reg_extensions(grid) == [{"weighting": "none", "alpha": 3.0, "factors": 128, "reg": 1.0}]


def test_alpha_extension_steps_past_an_edge_best_alpha(monkeypatch):
    monkeypatch.setattr(experiments, "ALS_REGS", [100.0, 200.0, 300.0, 400.0])
    monkeypatch.setattr(experiments, "ALS_ALPHA_STEPS", [1.0, 3.0, 10.0, 30.0])
    grid = pd.DataFrame({"weighting": ["none"] * 4, "alpha": [3.0, 3.0, 10.0, 10.0], "factors": [512] * 4,
                         "reg": [100.0, 200.0, 200.0, 300.0], "ndcg@10": [0.20, 0.25, 0.26, 0.28]})
    # The best alpha (10, at lambda 300) is the largest tried: try alpha = 30 around lambda 300.
    assert experiments.alpha_extensions(grid) == [
        {"weighting": "none", "alpha": 30.0, "factors": 512, "reg": r} for r in (200.0, 300.0, 400.0)]
    worse = pd.DataFrame({"weighting": ["none"], "alpha": [30.0], "factors": [512], "reg": [300.0], "ndcg@10": [0.27]})
    assert experiments.alpha_extensions(pd.concat([grid, worse])) == []


def test_edge_check_can_compare_alphas_across_their_own_lambdas():
    grid = pd.DataFrame({"alpha": [3.0, 10.0], "reg": [200.0, 375.0], "ndcg@10": [0.25, 0.26]})
    best = grid.loc[grid["ndcg@10"].idxmax()]
    fixed = experiments.edge_check(grid, best, ["alpha", "reg"], ["alpha"]).set_index("setting")
    free = experiments.edge_check(grid, best, ["alpha", "reg"], ["alpha"], free={"alpha": ["reg"]}).set_index("setting")
    assert fixed.loc["alpha", "tried"] == [10.0]  # no other alpha at lambda 375
    assert bool(free.loc["alpha", "at edge"]) and free.loc["alpha", "tried"] == [3.0, 10.0]


def test_alpha_extension_also_checks_the_one_se_pick(monkeypatch):
    monkeypatch.setattr(experiments, "ALS_REGS", [100.0, 200.0, 300.0])
    monkeypatch.setattr(experiments, "ALS_ALPHA_STEPS", [1.0, 3.0, 10.0])
    grid = pd.DataFrame({"weighting": ["none"] * 2, "alpha": [3.0, 10.0], "factors": [512] * 2,
                         "reg": [200.0, 200.0], "ndcg@10": [0.295, 0.30]})
    pick = grid.iloc[0]  # the one-SE rule prefers the smaller alpha, which is the smallest tried
    assert {c["alpha"] for c in experiments.alpha_extensions(grid, chosen=[pick])} == {1.0}


def test_als_simplicity_order_is_factors_then_smaller_alpha_then_larger_lambda():
    grid = pd.DataFrame({"factors": [512, 512, 512, 256], "alpha": [10.0, 3.0, 3.0, 3.0],
                         "reg": [450.0, 200.0, 300.0, 300.0], "ndcg@10": [0.300, 0.2995, 0.2992, 0.250],
                         "ndcg_se": [0.001] * 4})
    chosen, _ = experiments.one_se_choice(grid, [("factors", True), ("alpha", True), ("reg", False)])
    # 256 factors is too far below; among the rest the smaller alpha wins over the larger lambda.
    assert (chosen["alpha"], chosen["reg"]) == (3.0, 300.0)


def test_extra_test_ndcg_is_cached_per_settings(tmp_path, monkeypatch):
    import scipy.sparse as sp
    monkeypatch.setattr(experiments, "RESULTS_DIR", tmp_path)
    calls = []

    def fake(train_full, test, cfg, seed):
        calls.append(cfg["reg"])
        return np.full(train_full.shape[0], cfg["reg"])
    monkeypatch.setattr(experiments, "_als_test_ndcg", fake)
    playtime = sp.csr_matrix(np.ones((4, 6)))
    cfg = {"weighting": "none", "alpha": 3.0, "factors": 8, "reg": 1.0}
    first = experiments.extra_test_ndcg(playtime, 0, {"a": cfg})
    experiments.extra_test_ndcg(playtime, 0, {"a": cfg})
    assert calls == [1.0] and list(first["a"]) == [1.0] * 4
    experiments.extra_test_ndcg(playtime, 0, {"a": {**cfg, "factors": 16}})
    assert calls == [1.0, 1.0]  # a different setting is a different cache file


def _fake_test_run(calls):
    def run(playtime, seed, settings, cluster_ks, caps):
        calls.append(seed)
        scores = pd.DataFrame({"seed": [seed], "method": ["m"], "ndcg@10": [seed / 10]})
        diversity = pd.DataFrame([{"seed": seed, "clusters": k, "cap": c} for k in cluster_ks for c in caps])
        return scores, diversity, {"m": np.full(3, float(seed))}
    return run


def test_test_runs_recompute_a_deleted_diversity_table(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "RESULTS_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(experiments, "test_run", _fake_test_run(calls))
    experiments.test_runs(None, [0, 1], {"als": {"reg": 300.0}}, [10], [1])
    (tmp_path / "test_diversity.csv").unlink()
    _, diversity, _ = experiments.test_runs(None, [0, 1], {"als": {"reg": 300.0}}, [10], [1])
    assert calls == [0, 1, 0, 1] and set(diversity["seed"]) == {0, 1}


def test_test_runs_cache_restarts_when_caps_or_seeds_change(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "RESULTS_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(experiments, "test_run", _fake_test_run(calls))
    settings = {"als": {"reg": 300.0}}

    experiments.test_runs(None, [0, 1], settings, [10], [1, 2])
    experiments.test_runs(None, [0, 1], settings, [10], [1, 2])
    assert calls == [0, 1]  # second call is served from the cache

    _, diversity, _ = experiments.test_runs(None, [0, 1], settings, [10], [1, 2, 3])
    assert calls == [0, 1, 0, 1] and set(diversity["cap"]) == {1, 2, 3}

    scores, _, per_user = experiments.test_runs(None, [1, 2], settings, [10], [1, 2, 3])
    assert set(scores["seed"]) == {1, 2}
    assert np.all(per_user["m"] == 1.0)  # per-user NDCG belongs to the first requested seed
