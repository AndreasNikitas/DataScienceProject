"""
Validate different recent-form windows (e.g. 3/5/8/10 matches).

Prints train/test accuracy for both Random Forest and Logistic Regression
so you can justify the selected window in class.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from db import engine
from feature_columns import TRAINING_FEATURE_COLUMNS
from generate_features import generate_features_for_match

FEATURE_COLUMNS = TRAINING_FEATURE_COLUMNS
LOGISTIC_REGRESSION_C = 0.5


def load_finished_matches() -> pd.DataFrame:
    query = text(
        """
        SELECT id, match_date, home_team_id, away_team_id, result
        FROM matches
        WHERE result IS NOT NULL
        ORDER BY match_date ASC
        """
    )
    return pd.read_sql(query, engine)


def build_dataset_for_window(matches_df: pd.DataFrame, form_window: int) -> pd.DataFrame:
    rows: List[Dict] = []
    total = len(matches_df)
    for idx, row in matches_df.iterrows():
        if idx % 100 == 0:
            print(f"  Window {form_window}: {idx}/{total} matches")
        features = generate_features_for_match(
            int(row["id"]),
            row["match_date"],
            int(row["home_team_id"]),
            int(row["away_team_id"]),
            form_window=form_window,
        )
        features["result"] = row["result"]
        rows.append(features)
    return pd.DataFrame(rows)


def evaluate_window_for_seed(dataset: pd.DataFrame, form_window: int, seed: int) -> Dict:
    X = dataset[FEATURE_COLUMNS]
    y = dataset["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=seed,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    rf_train_acc = accuracy_score(y_train, rf_model.predict(X_train))
    rf_test_acc = accuracy_score(y_test, rf_model.predict(X_test))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_model = LogisticRegression(
        random_state=seed,
        max_iter=2000,
        C=LOGISTIC_REGRESSION_C,
        solver="lbfgs",
        n_jobs=-1,
    )
    lr_model.fit(X_train_scaled, y_train)
    lr_train_acc = accuracy_score(y_train, lr_model.predict(X_train_scaled))
    lr_test_acc = accuracy_score(y_test, lr_model.predict(X_test_scaled))

    return {
        "window": form_window,
        "seed": seed,
        "rf_train_accuracy": rf_train_acc,
        "rf_test_accuracy": rf_test_acc,
        "lr_train_accuracy": lr_train_acc,
        "lr_test_accuracy": lr_test_acc,
    }


def summarize_window_results(per_seed_results: List[Dict], samples: int) -> Dict:
    rf_train = [r["rf_train_accuracy"] for r in per_seed_results]
    rf_test = [r["rf_test_accuracy"] for r in per_seed_results]
    lr_train = [r["lr_train_accuracy"] for r in per_seed_results]
    lr_test = [r["lr_test_accuracy"] for r in per_seed_results]

    return {
        "window": per_seed_results[0]["window"],
        "samples": samples,
        "seeds": [r["seed"] for r in per_seed_results],
        "runs": len(per_seed_results),
        "rf_train_mean": statistics.mean(rf_train),
        "rf_train_std": statistics.pstdev(rf_train) if len(rf_train) > 1 else 0.0,
        "rf_test_mean": statistics.mean(rf_test),
        "rf_test_std": statistics.pstdev(rf_test) if len(rf_test) > 1 else 0.0,
        "rf_test_min": min(rf_test),
        "rf_test_max": max(rf_test),
        "lr_train_mean": statistics.mean(lr_train),
        "lr_train_std": statistics.pstdev(lr_train) if len(lr_train) > 1 else 0.0,
        "lr_test_mean": statistics.mean(lr_test),
        "lr_test_std": statistics.pstdev(lr_test) if len(lr_test) > 1 else 0.0,
        "lr_test_min": min(lr_test),
        "lr_test_max": max(lr_test),
        "per_seed": per_seed_results,
    }


def print_results_table(results: List[Dict]) -> None:
    print("\n" + "=" * 136)
    print("FORM WINDOW VALIDATION (80/20 split, repeated seeds)")
    print("=" * 136)
    print(
        f"{'Window':<8} {'Runs':<6} {'Samples':<8} "
        f"{'RF Test mean±std':<22} {'RF Test range':<18} "
        f"{'LR Test mean±std':<22} {'LR Test range':<18}"
    )
    print("-" * 136)
    for r in results:
        print(
            f"{r['window']:<8} {r['runs']:<6} {r['samples']:<8} "
            f"{r['rf_test_mean']*100:>6.2f}% ± {r['rf_test_std']*100:<6.2f}% "
            f"{r['rf_test_min']*100:>6.2f}%-{r['rf_test_max']*100:<6.2f}% "
            f"{r['lr_test_mean']*100:>6.2f}% ± {r['lr_test_std']*100:<6.2f}% "
            f"{r['lr_test_min']*100:>6.2f}%-{r['lr_test_max']*100:<6.2f}%"
        )

    best_rf = max(results, key=lambda x: x["rf_test_mean"])
    best_lr = max(results, key=lambda x: x["lr_test_mean"])
    print("-" * 136)
    print(
        f"Best RF window by mean test accuracy: {best_rf['window']} "
        f"({best_rf['rf_test_mean']*100:.2f}% ± {best_rf['rf_test_std']*100:.2f}%)"
    )
    print(
        f"Best LR window by mean test accuracy: {best_lr['window']} "
        f"({best_lr['lr_test_mean']*100:.2f}% ± {best_lr['lr_test_std']*100:.2f}%)"
    )
    print("=" * 136)


def save_results(results: List[Dict]) -> None:
    output_path = Path("models")
    output_path.mkdir(exist_ok=True)
    filepath = output_path / "form_window_validation_repeated.json"
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"Saved validation results to {filepath}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate form windows for training.")
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[5, 8, 10, 15, 20],
        help="Form windows to evaluate (e.g. --windows 3 5 8 10)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Random seeds for repeated splits (e.g. --seeds 1 2 3 4 5)",
    )
    args = parser.parse_args()

    matches_df = load_finished_matches()
    if matches_df.empty:
        print("No finished matches found.")
        return

    results: List[Dict] = []
    for window in args.windows:
        if window <= 0:
            print(f"Skipping invalid window: {window}")
            continue
        print(f"\nEvaluating window={window}...")
        dataset = build_dataset_for_window(matches_df, window)
        per_seed_results: List[Dict] = []
        for seed in args.seeds:
            print(f"  Training/evaluating seed={seed}")
            per_seed_results.append(evaluate_window_for_seed(dataset, window, seed))
        results.append(summarize_window_results(per_seed_results, samples=len(dataset)))

    if not results:
        print("No valid windows to evaluate.")
        return

    print_results_table(results)
    save_results(results)


if __name__ == "__main__":
    main()
