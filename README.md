# Steam Data Cleaning and Exploration

This project explores Steam game, review, and user-inventory data. The notebook covers data loading, schema and missing-value inspection, game-data cleaning, duplicate checks, and visual summaries.

## Project Files

- `steam_cleaning.ipynb` - analysis and visualizations.
- `dataset/steam_games.json.gz` - Steam game metadata.
- `dataset/australian_user_reviews.json.gz` - user reviews.
- `dataset/australian_users_items.json.gz` - games listed in user inventories.

The notebook reads the compressed JSON files directly from `dataset/`; they do not need to be extracted.

## Notebook Workflow

- Inspect the three raw datasets and compare their schemas.
- Measure missing values and review columns with substantial missingness.
- Clean game sentiment and prices, fill missing publisher and developer names, and remove selected columns and incomplete rows.
- Check duplicate identifiers and inspect examples from the datasets.
- Plot the most common game tags and the most frequently listed games in user inventories.

## Run

Use Python 3 from the repository root so the relative `dataset/` path resolves. Install the notebook dependencies:

```bash
python -m pip install pandas matplotlib notebook
```

Then open and run the notebook:

```bash
jupyter notebook steam_cleaning.ipynb
```
## Recommender (ALS against course methods)

`recommender.ipynb` compares implicit-feedback ALS (Hu, Koren & Volinsky, 2008; our outside-the-course method, implemented from scratch) with course-method recommenders: association rules from frequent itemsets (PCY pairs, Apriori triples), item-item similarity, and popularity. It also clusters the learned game vectors with spherical k-means and uses the clusters to diversify the recommendation lists.

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

Run the notebook from the repository root. With the cached `results/` it takes about 15 minutes; recomputing everything takes several hours on a 12-core laptop.
