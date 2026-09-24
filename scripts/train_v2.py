"""
Corrected training and evaluation (2026) for the Selangor flood classifier.

The 2023 notebooks have two evaluation problems:
  1. SMOTE was applied before the train/test split, so synthetic copies of
     training floods leaked into the test set and inflated accuracy.
  2. The split was random. Adjacent days share weather, so a random split also
     leaks. Floods need a chronological split.
They also predicted a flood from the same day's rainfall, which is nowcasting.

This script keeps the same four models and the same data files, and fixes all
three points:
  * chronological split: train 2001-2008, test 2009-2010
  * SMOTE on the training fold only, test keeps its natural imbalance
  * lagged features: today's 225-cell grid plus the previous 3 days (900 features)
  * standardisation for Logistic Regression and SVM
  * reports precision / recall / F1 for the flood class, not just accuracy

Usage
    python scripts/train_v2.py --rain "data/HQprecipitation Data.csv" --col HqPrecips
    python scripts/train_v2.py --rain "data/precipitationCal Data.csv" --col PrecipCals
    python scripts/train_v2.py --synthetic        # smoke test, no data needed

Models are saved to webapp/models/<col>_<model>_v2.pkl. Note they expect 900
features, so app.py must be given four days of rainfall before it can use them.
"""

from __future__ import annotations

import argparse
import ast
import os
import pickle

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
FLOODS = os.path.join(ROOT, "data", "Flood_Events.xlsx")
MODEL_DIR = os.path.join(ROOT, "webapp", "models")
GRID = 225
LAGS = 3
SPLIT_YEAR = 2009  # first test year


def load_rain(path: str, col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"].str[:10])
    cells = np.vstack([np.asarray(ast.literal_eval(s), dtype=float) for s in df[col]])
    out = pd.DataFrame(cells, columns=[f"{col}_{i}" for i in range(GRID)])
    out.insert(0, "date", df["time"].values)
    return out.sort_values("date").reset_index(drop=True)


def load_labels(start: str, end: str) -> pd.DataFrame:
    floods = pd.read_excel(FLOODS)
    floods = floods.drop(columns=[c for c in floods.columns if c.startswith("Unnamed")])
    floods["Date"] = pd.to_datetime(floods["Date"])
    rivers = [c for c in floods.columns if c != "Date"]
    daily = floods.groupby("Date")[rivers].max()
    days = pd.date_range(start, end, freq="D")
    daily = daily.reindex(days, fill_value=0)
    return pd.DataFrame({"date": days, "flood": (daily[rivers].sum(axis=1) > 0).astype(int).values})


def add_lags(rain: pd.DataFrame, lags: int) -> pd.DataFrame:
    """Forward-fill missing days, then append the previous `lags` days as features."""
    full = pd.date_range(rain["date"].min(), rain["date"].max(), freq="D")
    missing = len(full) - len(rain)
    if missing:
        print(f"forward-filling {missing} missing day(s) in the rainfall series")
    rain = rain.set_index("date").reindex(full).ffill()
    frames = [rain]
    for k in range(1, lags + 1):
        frames.append(rain.shift(k).add_suffix(f"_lag{k}"))
    out = pd.concat(frames, axis=1).dropna()
    return out.reset_index().rename(columns={"index": "date"})


def synthetic(col: str) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    days = pd.date_range("2001-01-01", "2010-12-31", freq="D")
    cells = rng.gamma(0.6, 6.0, size=(len(days), GRID))
    out = pd.DataFrame(cells, columns=[f"{col}_{i}" for i in range(GRID)])
    out.insert(0, "date", days)
    return out


def evaluate(name: str, model, x_tr, y_tr, x_te, y_te) -> dict:
    model.fit(x_tr, y_tr)
    pred = model.predict(x_te)
    cm = metrics.confusion_matrix(y_te, pred, labels=[0, 1])
    row = {
        "model": name,
        "accuracy": metrics.accuracy_score(y_te, pred),
        "precision": metrics.precision_score(y_te, pred, zero_division=0),
        "recall": metrics.recall_score(y_te, pred, zero_division=0),
        "f1": metrics.f1_score(y_te, pred, zero_division=0),
    }
    print(f"\n{name}\n  confusion matrix [[TN FP] [FN TP]]: {cm.tolist()}")
    print("  " + "  ".join(f"{k}={v:.3f}" for k, v in row.items() if k != "model"))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rain", help="rainfall CSV produced by imerg_download.py")
    ap.add_argument("--col", default="HqPrecips", choices=["HqPrecips", "PrecipCals"])
    ap.add_argument("--synthetic", action="store_true", help="run on random rainfall to check the pipeline")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    if args.synthetic:
        rain = synthetic(args.col)
    elif args.rain:
        rain = load_rain(args.rain, args.col)
    else:
        ap.error("give --rain <csv> or --synthetic")

    rain = add_lags(rain, LAGS)
    labels = load_labels(rain["date"].min(), rain["date"].max())
    data = rain.merge(labels, on="date", how="inner")
    features = [c for c in data.columns if c not in ("date", "flood")]
    print(f"{len(data)} days, {len(features)} features, {int(data['flood'].sum())} flood days "
          f"({data['flood'].mean():.1%})")

    train = data[data["date"].dt.year < SPLIT_YEAR]
    test = data[data["date"].dt.year >= SPLIT_YEAR]
    x_tr, y_tr = train[features].values, train["flood"].values
    x_te, y_te = test[features].values, test["flood"].values
    print(f"train {train['date'].min().date()}..{train['date'].max().date()} ({len(train)} days, {int(y_tr.sum())} floods)")
    print(f"test  {test['date'].min().date()}..{test['date'].max().date()} ({len(test)} days, {int(y_te.sum())} floods)")

    x_tr, y_tr = SMOTE(random_state=42).fit_resample(x_tr, y_tr)
    print(f"after SMOTE on training fold only: {len(y_tr)} rows")

    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000, class_weight="balanced")),
        "NaiveBayes": GaussianNB(),
        "SVM": make_pipeline(StandardScaler(), SVC()),
    }
    results = [evaluate(n, m, x_tr, y_tr, x_te, y_te) for n, m in models.items()]

    print("\nSummary (test set 2009-2010, flood class):")
    print(pd.DataFrame(results).set_index("model").round(3).to_string())

    if not args.no_save and not args.synthetic:
        os.makedirs(MODEL_DIR, exist_ok=True)
        for name, model in models.items():
            path = os.path.join(MODEL_DIR, f"{args.col}_{name}_v2.pkl")
            with open(path, "wb") as fh:
                pickle.dump(model, fh)
        print(f"\nsaved 4 models to {MODEL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
