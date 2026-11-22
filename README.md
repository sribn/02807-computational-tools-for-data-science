# Steam game recommendations

Which games will a Steam user go on to play? This project compares implicit-feedback ALS (our method from outside the course) with association rules from frequent itemsets and two baselines, and clusters the learned game vectors to diversify the recommendations.

## Data

The data are the Australian Steam data sets from Julian McAuley's group at UCSD (https://cseweb.ucsd.edu/~jmcauley/datasets.html). The dataset page asks users to cite: Kang & McAuley, "Self-attentive sequential recommendation" (ICDM 2018); Wan & McAuley, "Item recommendation on monotonic behavior chains" (RecSys 2018); Pathak, Gupta & McAuley, "Generating and personalizing bundle recommendations on Steam" (SIGIR 2017).

- `dataset/australian_users_items.json.gz` - games in each user's inventory, with playtime (the input of every method).
- `dataset/steam_games.json.gz` - game metadata (only the tags are used, to describe clusters).
- `dataset/australian_user_reviews.json.gz` - user reviews (not used by the results).

The notebooks read the compressed files directly from `dataset/`; they do not need to be extracted.

## Notebooks, in order

| Notebook | Contents |
|---|---|
| `data_preparation.ipynb` | the data every result uses: duplicates, played games only, the player/game filters, the train/validation/test split, and the popularity figure |
| `frequent_itemsets.ipynb` | frequent pairs (PCY) and triples (Apriori) on all baskets, and whether the PCY bitmap saves memory at this scale |
| `recommender.ipynb` | tuning and test comparison of ALS, association rules, item-kNN and popularity; cross-check against `implicit`; clusters and diversification |
| `steam_cleaning.ipynb` | legacy: the group's first exploration of all three files (metadata and review cleaning, plots); no result depends on it |

## Run

Use Python 3 from the repository root so the relative `dataset/` path resolves:

```bash
python -m pip install -r requirements-recommender.txt
python -m pytest
jupyter notebook
```

`data_preparation.ipynb` takes about a minute and `frequent_itemsets.ipynb` about three (most of it the Apriori triples).

## Recommender (ALS against a course method and baselines)

`recommender.ipynb` compares implicit-feedback ALS (Hu, Koren & Volinsky, 2008; our outside-the-course method, implemented from scratch) with the course method association rules from frequent itemsets (PCY pairs, Apriori triples) and two baselines, item-item similarity and popularity. It also clusters the learned game vectors with spherical k-means and uses the clusters to diversify the recommendation lists.

| Path | Contents |
|---|---|
| `recommender/data.py` | shared data definition: played games, filters, train/validation/test split |
| `recommender/itemsets.py`, `recommender/rules.py` | PCY and Apriori, association-rule recommender |
| `recommender/baselines.py` | item-kNN (cosine or Jaccard) |
| `recommender/als.py` | implicit-feedback ALS (NumPy + Numba), sparse loss |
| `recommender/evaluate.py` | top-k lists, ranking metrics, bootstrap intervals, diversity measures |
| `recommender/clusters.py`, `recommender/diversify.py` | spherical k-means, tag comparison, cluster-capped lists |
| `recommender/experiments.py` | tuning grids, the one-SE selection rule, cached test runs |
| `results/` | cached tuning grids and test runs (delete a file to recompute it) |
| `figures/` | figures and the test-results table for the report |

```bash
python -m pip install -r requirements-recommender.txt
python -m pytest
jupyter notebook recommender.ipynb
```

Run the notebook from the repository root. With the cached `results/` it takes about 15 minutes; recomputing everything takes several hours, mostly the 512-factor ALS fits at about 6 minutes each.
