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


def _fake_test_run(calls):
    def run(playtime, seed, settings, cluster_ks, caps):
        calls.append(seed)
        scores = pd.DataFrame({"seed": [seed], "method": ["m"], "ndcg@10": [seed / 10]})
        diversity = pd.DataFrame([{"seed": seed, "clusters": k, "cap": c} for k in cluster_ks for c in caps])
        return scores, diversity, {"m": np.full(3, float(seed))}
    return run


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
