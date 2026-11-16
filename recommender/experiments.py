"""Experiment plumbing for recommender.ipynb: build each recommender from its settings, score it,
choose settings with one rule for every method, and cache slow grids as CSV files.

The tuning grids take hours, so every finished setting is appended to results/<grid>.csv and
skipped on the next run; delete a file (or set RECOMMENDER_RESULTS to an empty folder) to
recompute it.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from recommender import als, baselines, clusters, data, diversify, evaluate, itemsets, rules

RESULTS_DIR = Path(os.environ.get(
    "RECOMMENDER_RESULTS", Path(__file__).resolve().parent.parent / "results"
))
PCY_BUCKETS = 10_000_000
ITERATIONS = 15
K = 10

# Tuning grids (supports are counts of users in the tuning split).
RULE_PAIR_SUPPORTS = [30, 60, 120, 250]
RULE_TRIPLE_SUPPORTS = [0, 60, 150, 300]   # 0 = pairs only
RULE_AGGREGATIONS = ["max", "sum"]
KNN_MEASURES = ["cosine", "jaccard"]
KNN_NEIGHBOURS = [1, 2, 5, 10, 20, 50, 100, 200, 500]  # plus "all games"
ALS_ALPHA = 3.0
ALS_FACTORS = [64, 128, 256]
ALS_MAX_FACTORS = 512                      # compute cap, tried only near the best lambda
ALS_REGS = [30.0, 100.0, 200.0, 250.0, 300.0, 375.0, 450.0, 700.0, 1000.0, 1500.0, 3000.0]
ALS_SMALL_REGS = [1.0, 10.0]               # shows what happens without strong regularisation
ALS_ALPHAS = [1.0, 3.0, 10.0]
ALS_WEIGHTINGS = ["none", "idf"]


# --- building recommenders -------------------------------------------------------------

def rules_scorer(train, pair_support, triple_support, aggregate, _cache=None):
    """Association rules: pair rules at `pair_support`; triple rules from Apriori at `triple_support`."""
    b = itemsets.baskets(train)
    pairs = itemsets.pcy_pairs(b, int(pair_support), PCY_BUCKETS)
    triples = None
    if triple_support:
        triples = _triple_rules(b, int(triple_support), _cache)
    return rules.association_rule_scorer(train, rules.pair_rule_confidences(pairs), triples, aggregate)


def _triple_rules(b, support, cache):
    if cache is not None and support in cache:
        return cache[support]
    pairs = itemsets.pcy_pairs(b, support, PCY_BUCKETS)
    triple_rules = rules.triple_rules(pairs, itemsets.apriori_triples(b, pairs, support))
    if cache is not None:
        cache[support] = triple_rules
    return triple_rules


def knn_scorer(train, measure, k):
    sim = baselines.item_similarity(baselines.pair_counts(train), measure)
    return baselines.item_knn_scorer(train, baselines.nearest_neighbours(sim, int(k)))


def fit_als(train, weighting, alpha, factors, reg, seed=0, iterations=ITERATIONS, callback=None):
    weights = als.idf_weights(train) if weighting == "idf" else None
    conf = als.confidence(train, float(alpha), weights)
    return als.fit(conf, int(factors), float(reg), iterations, seed, callback)


def als_with_convergence(train, cfg, seed=0):
    """Fit ALS and report the relative loss change in the last sweep (small = converged)."""
    weights = als.idf_weights(train) if cfg["weighting"] == "idf" else None
    conf = als.confidence(train, float(cfg["alpha"]), weights)
    losses = []

    def track(it, model):
        if it >= ITERATIONS - 2:
            losses.append(als.loss(conf, model, float(cfg["reg"])))

    model = als.fit(conf, int(cfg["factors"]), float(cfg["reg"]), ITERATIONS, seed, track)
    return model, (losses[0] - losses[1]) / losses[0]


# --- scoring ----------------------------------------------------------------------------

def summarise(recs, truth, n_items):
    """Mean metrics, the standard error of NDCG, and catalogue coverage of one set of lists."""
    per_user = evaluate.per_user_metrics(recs, truth)
    ndcg = per_user[f"ndcg@{K}"]
    return {
        **{name: float(values.mean()) for name, values in per_user.items()},
        "ndcg_se": float(ndcg.std(ddof=1) / np.sqrt(ndcg.size)),
        "coverage": evaluate.catalogue_coverage(recs, n_items),
    }


def score(scorer, train, truth):
    return summarise(evaluate.top_k(scorer, train, K), truth, train.shape[1])


# --- grids ------------------------------------------------------------------------------

def run_grid(name, configs, evaluate_config):
    """Evaluate every config not already in results/<name>.csv; return the rows for `configs`."""
    path = RESULTS_DIR / f"{name}.csv"
    done = pd.read_csv(path) if path.exists() else pd.DataFrame()
    keys = list(configs[0])
    wanted = [_key(cfg, keys) for cfg in configs]
    seen = set(done[keys].apply(lambda r: _key(r, keys), axis=1)) if len(done) else set()

    for cfg, key in zip(configs, wanted):
        if key in seen:
            continue
        row = {**cfg, **evaluate_config(cfg)}
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        done.to_csv(path, index=False)
        seen.add(key)

    mask = done[keys].apply(lambda r: _key(r, keys), axis=1).isin(set(wanted))
    return done[mask].reset_index(drop=True)


def _key(cfg, keys):
    return tuple(f"{float(cfg[k]):g}" if _is_number(cfg[k]) else str(cfg[k]) for k in keys)


def _is_number(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def rule_configs():
    return [
        {"pair_support": s2, "triple_support": s3, "aggregate": agg}
        for s2 in RULE_PAIR_SUPPORTS for s3 in RULE_TRIPLE_SUPPORTS for agg in RULE_AGGREGATIONS
    ]


def knn_configs(n_items):
    return [
        {"measure": m, "k": k}
        for m in KNN_MEASURES for k in KNN_NEIGHBOURS + [n_items]
    ]


def als_stage1_configs():
    grid = [
        {"weighting": "none", "alpha": ALS_ALPHA, "factors": f, "reg": reg}
        for f in ALS_FACTORS for reg in ALS_REGS
    ]
    small = [{"weighting": "none", "alpha": ALS_ALPHA, "factors": 128, "reg": reg} for reg in ALS_SMALL_REGS]
    return grid + small


def neighbouring_regs(reg, below=1, above=1):
    """The grid's lambda values around `reg`."""
    i = ALS_REGS.index(float(reg))
    return ALS_REGS[max(0, i - below): i + above + 1]


def als_cap_configs(stage1):
    """The compute cap: 512 factors only at lambdas around the best 256-factor lambda."""
    at_256 = stage1[stage1["factors"] == max(ALS_FACTORS)].reset_index(drop=True)
    best_reg = at_256.loc[at_256[f"ndcg@{K}"].idxmax(), "reg"]
    return [
        {"weighting": "none", "alpha": ALS_ALPHA, "factors": ALS_MAX_FACTORS, "reg": reg}
        for reg in neighbouring_regs(best_reg, below=1, above=2)
    ]


def als_stage2_configs(stage1):
    """alpha and idf weighting around the best (factors, lambda) found in stage 1."""
    stage1 = stage1.reset_index(drop=True)
    best = stage1.loc[stage1[f"ndcg@{K}"].idxmax()]
    return [
        {"weighting": w, "alpha": a, "factors": int(best["factors"]), "reg": reg}
        for w in ALS_WEIGHTINGS for a in ALS_ALPHAS for reg in neighbouring_regs(best["reg"])
    ]


# --- choosing settings ------------------------------------------------------------------

def one_se_choice(grid, simplicity, metric=f"ndcg@{K}"):
    """The simplest setting within one standard error of the best (the "one-SE rule").

    `simplicity` lists (column, ascending) pairs, simplest first; remaining ties go to the
    higher score. Returns (chosen row, best row).
    """
    best = grid.loc[grid[metric].idxmax()]
    within = grid[grid[metric] >= best[metric] - best["ndcg_se"]]
    columns = [c for c, _ in simplicity] + [metric]
    ascending = [a for _, a in simplicity] + [False]
    return within.sort_values(columns, ascending=ascending).iloc[0], best


def edge_check(grid, best, keys, ordinal, natural_bounds=None):
    """For each ordinal setting, is the best value at the edge of what was tried around it?

    Only rows that match `best` on every other setting count as its neighbours. Values in
    `natural_bounds` (e.g. "all neighbours", a stated compute cap) are allowed edges.
    """
    natural_bounds = natural_bounds or {}
    rows = []
    for axis in ordinal:
        others = [k for k in keys if k != axis]
        line = grid[(grid[others] == best[others]).all(axis=1)]
        tried = sorted(line[axis].unique())
        at_edge = len(tried) > 1 and best[axis] in (tried[0], tried[-1])
        allowed = best[axis] in natural_bounds.get(axis, ())
        rows.append({"setting": axis, "best": best[axis], "tried": tried,
                     "at edge": at_edge, "allowed edge": allowed})
    return pd.DataFrame(rows)


# --- convergence, test runs over several seeds ------------------------------------------

def cached_table(name, compute):
    """Load results/<name>.csv, or compute the table and save it there."""
    path = RESULTS_DIR / f"{name}.csv"
    if path.exists():
        return pd.read_csv(path)
    table = compute()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def convergence_trace(train, valid, cfg, seed, sweeps):
    """Training loss and validation NDCG after every ALS sweep."""
    weights = als.idf_weights(train) if cfg["weighting"] == "idf" else None
    conf = als.confidence(train, float(cfg["alpha"]), weights)
    rows = []

    def track(it, model):
        rows.append({
            "seed": seed, "sweep": it + 1, "loss": als.loss(conf, model, float(cfg["reg"])),
            f"ndcg@{K}": score(evaluate.als_scorer(model), train, valid)[f"ndcg@{K}"],
        })

    als.fit(conf, int(cfg["factors"]), float(cfg["reg"]), sweeps, seed, track)
    return pd.DataFrame(rows)


def list_metrics(recs, truth, popularity, sim):
    """Accuracy and variety of one set of top-k lists."""
    per_user = evaluate.per_user_metrics(recs, truth)
    return {
        **{name: float(values.mean()) for name, values in per_user.items()},
        "coverage": evaluate.catalogue_coverage(recs, popularity.size),
        "long tail share": evaluate.long_tail_share(recs, popularity),
        "intra-list similarity": evaluate.intra_list_similarity(recs, sim),
    }, per_user[f"ndcg@{K}"]


def final_methods(train_full, settings, seed):
    """Every method refit on train + validation with its chosen settings (supports already scaled).

    Returns the scorers by name and the two fitted ALS models by name.
    """
    pairs, triples, knn = settings["rules_pairs"], settings["rules_triples"], settings["knn"]
    als_model = fit_als(train_full, **settings["als"], seed=seed)
    idf_model = fit_als(train_full, **settings["als_idf"], seed=seed)
    methods = {
        "Popularity": evaluate.popularity_scorer(train_full),
        "Rules (pairs)": rules_scorer(train_full, pairs["pair_support"], 0, pairs["aggregate"]),
        "Rules (pairs + triples)": rules_scorer(
            train_full, triples["pair_support"], triples["triple_support"], triples["aggregate"]),
        "Item-kNN": knn_scorer(train_full, knn["measure"], knn["k"]),
        "ALS": evaluate.als_scorer(als_model),
        "ALS + idf": evaluate.als_scorer(idf_model),
    }
    return methods, {"ALS": als_model, "ALS + idf": idf_model}


def test_run(playtime, seed, settings, cluster_ks, caps):
    """Split with `seed`, refit every method, score it on the test games, and trace how
    capping games per cluster changes the lists of the headline ALS variant
    (settings["headline"]). Returns (scores, diversity, per-user NDCG)."""
    train_full, test = data.split_per_user(playtime, test_frac=0.2, seed=seed)
    methods, models = final_methods(train_full, settings, seed)
    als_model = models[settings["headline"]]
    popularity = evaluate.popularity_counts(train_full)
    sim = baselines.item_similarity(baselines.pair_counts(train_full))

    rows, per_user = [], {}
    for name, scorer in methods.items():
        metrics, per_user[name] = list_metrics(evaluate.top_k(scorer, train_full, K), test, popularity, sim)
        rows.append({"seed": seed, "method": name, **metrics})

    diversity = []
    for k in cluster_ks:
        labels, _, _ = clusters.spherical_kmeans(als_model.item_factors, k, seed)
        for cap in caps:
            recs = diversify.capped_top_k(evaluate.als_scorer(als_model), train_full, labels, cap, K)
            metrics, _ = list_metrics(recs, test, popularity, sim)
            diversity.append({"seed": seed, "clusters": k, "cap": cap, **metrics,
                              "largest cluster share": clusters.largest_cluster_share(recs, labels)})
    return pd.DataFrame(rows), pd.DataFrame(diversity), per_user


def test_runs(playtime, seeds, settings, cluster_ks, caps):
    """Run `test_run` for every seed not cached yet; per-user NDCG is kept for the first seed."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    names = {part: RESULTS_DIR / f"test_{part}.csv" for part in ("scores", "diversity")}
    per_user_path = RESULTS_DIR / "test_per_user_ndcg.npz"
    settings_path = RESULTS_DIR / "test_settings.json"
    as_json = json.dumps(settings, sort_keys=True, default=float)
    if not settings_path.exists() or settings_path.read_text() != as_json:
        for path in [*names.values(), per_user_path]:
            path.unlink(missing_ok=True)  # settings changed: old test results no longer apply
        settings_path.write_text(as_json)

    tables = {part: pd.read_csv(p) if p.exists() else pd.DataFrame() for part, p in names.items()}
    for seed in seeds:
        done = len(tables["scores"]) and seed in set(tables["scores"]["seed"])
        if done and (seed != seeds[0] or per_user_path.exists()):
            continue
        scores, diversity, per_user = test_run(playtime, seed, settings, cluster_ks, caps)
        for part, new in (("scores", scores), ("diversity", diversity)):
            old = tables[part]
            old = old[old["seed"] != seed] if len(old) else old
            tables[part] = pd.concat([old, new], ignore_index=True)
            tables[part].to_csv(names[part], index=False)
        if seed == seeds[0]:
            np.savez_compressed(per_user_path, **per_user)

    per_user = dict(np.load(per_user_path))
    return tables["scores"], tables["diversity"], per_user
