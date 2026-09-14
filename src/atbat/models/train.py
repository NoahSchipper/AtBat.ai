"""Compare baseline classifiers and predictor groups with temporal validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUTCOMES = (
    "strikeout",
    "walk_hbp",
    "single",
    "double",
    "triple",
    "home_run",
    "ball_in_play_out",
    "reached_other",
)


def _number(value: str) -> float:
    return float(value) if value not in ("", None) else 0.0


def _feature_groups(columns: list[str]) -> dict[str, list[str]]:
    history = [
        column
        for column in columns
        if column.startswith(("batter_", "pitcher_"))
        and not column.startswith(("batter_vs_", "pitcher_vs_"))
        and column.endswith(("_prior", "_rate_prior"))
    ]
    matchup = [
        column
        for column in columns
        if column.startswith(("batter_vs_pitcher_hand_", "pitcher_vs_batter_hand_"))
    ]
    game_state = [
        column
        for column in ("inning", "outs_pre", "score_v", "score_h", "balls", "strikes")
        if column in columns
    ]
    environment = [
        column
        for column in ("temp", "windspeed", "attendance")
        if column in columns
    ]
    return {
        "game_state": game_state,
        "player_history": history,
        "handedness_splits": matchup,
        "environment": environment,
    }


def _brier_score(probabilities: np.ndarray, y_test: np.ndarray, ordered_labels: list[str]) -> float:
    """Multiclass Brier score: mean squared error between one-hot labels and predicted probs."""
    label_index = {label: i for i, label in enumerate(ordered_labels)}
    one_hot = np.zeros_like(probabilities)
    for row, label in enumerate(y_test):
        one_hot[row, label_index[label]] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def _per_outcome_metrics(
    y_test: np.ndarray,
    predictions: np.ndarray,
    ordered_labels: list[str],
) -> dict[str, dict[str, float]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=ordered_labels,
        zero_division=0,
    )
    return {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(ordered_labels)
    }


def load_data(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
    if not rows or not reader.fieldnames:
        raise ValueError("Feature dataset is empty")
    return rows, reader.fieldnames


def evaluate(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    ordered_labels: list[str],
    model_name: str = "multinomial_logistic_regression",
) -> dict[str, object]:
    if model_name == "histogram_gradient_boosting":
        model = HistGradientBoostingClassifier()
    else:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=250),
        )
    model.fit(x_train, y_train)
    classes = list(model.classes_) if hasattr(model, "classes_") else list(model[-1].classes_)
    probabilities = model.predict_proba(x_test)
    # Reindex probability columns to a stable, sorted label order.
    reindex = [classes.index(label) for label in ordered_labels]
    probabilities = probabilities[:, reindex]
    predictions = model.predict(x_test)
    return {
        "model": model_name,
        "log_loss": float(log_loss(y_test, probabilities, labels=ordered_labels)),
        "brier_score": _brier_score(probabilities, y_test, ordered_labels),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "per_outcome": _per_outcome_metrics(y_test, predictions, ordered_labels),
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "features": x_train.shape[1],
    }


def _split_by_season(
    rows: list[dict[str, str]],
    train_seasons: list[str] | None,
    validation_season: str | None,
    test_season: str | None,
    test_fraction: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]] | None, list[dict[str, str]]]:
    rows = sorted(rows, key=lambda row: (row["date"], row["gid"]))
    if test_season is None:
        split = int(len(rows) * (1.0 - test_fraction))
        return rows[:split], None, rows[split:]

    train_rows = (
        [row for row in rows if row["season"] in set(train_seasons)]
        if train_seasons
        else [row for row in rows if row["season"] < test_season]
    )
    validation_rows = (
        [row for row in rows if row["season"] == validation_season]
        if validation_season
        else None
    )
    test_rows = [row for row in rows if row["season"] == test_season]
    if not train_rows or not test_rows:
        raise ValueError("Season-based split produced an empty train or test set")
    return train_rows, validation_rows, test_rows


def train(
    path: Path,
    output: Path,
    test_fraction: float,
    train_seasons: list[str] | None = None,
    validation_season: str | None = None,
    test_season: str | None = None,
) -> Path:
    rows, columns = load_data(path)
    train_rows, validation_rows, test_rows = _split_by_season(
        rows, train_seasons, validation_season, test_season, test_fraction
    )
    groups = _feature_groups(columns)
    ordered_labels = sorted(OUTCOMES)
    results: list[dict[str, object]] = []

    train_labels = np.array([row["outcome"] for row in train_rows])
    test_labels = np.array([row["outcome"] for row in test_rows])
    frequencies = np.array(
        [np.mean(train_labels == label) for label in ordered_labels],
        dtype=float,
    )
    baseline = np.tile(frequencies, (len(test_labels), 1))
    baseline_label = ordered_labels[int(frequencies.argmax())]
    baseline_predictions = np.full(len(test_labels), baseline_label, dtype=object)
    results.append(
        {
            "model": "training_frequency_baseline",
            "log_loss": float(log_loss(test_labels, baseline, labels=ordered_labels)),
            "brier_score": _brier_score(baseline, test_labels, ordered_labels),
            "accuracy": float(accuracy_score(test_labels, baseline_predictions)),
            "per_outcome": _per_outcome_metrics(test_labels, baseline_predictions, ordered_labels),
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "features": 0,
        }
    )

    selected: list[str] = []
    all_feature_columns: list[str] = []
    for group_name, group_columns in groups.items():
        selected.extend(group_columns)
        all_feature_columns.extend(group_columns)
        if not selected:
            continue
        x_train = np.array([[_number(row[column]) for column in selected] for row in train_rows])
        x_test = np.array([[_number(row[column]) for column in selected] for row in test_rows])
        result = evaluate(x_train, x_test, train_labels, test_labels, ordered_labels)
        result["feature_group"] = f"through_{group_name}"
        results.append(result)

    if all_feature_columns:
        x_train = np.array(
            [[_number(row[column]) for column in all_feature_columns] for row in train_rows]
        )
        x_test = np.array(
            [[_number(row[column]) for column in all_feature_columns] for row in test_rows]
        )
        tree_result = evaluate(
            x_train,
            x_test,
            train_labels,
            test_labels,
            ordered_labels,
            model_name="histogram_gradient_boosting",
        )
        tree_result["feature_group"] = "through_environment"
        results.append(tree_result)

    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "results": results,
        "train_seasons": sorted({row["season"] for row in train_rows}),
        "test_season": sorted({row["season"] for row in test_rows}),
    }
    if validation_rows:
        payload["validation_season"] = sorted({row["season"] for row in validation_rows})
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/model_results.json"))
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--train-seasons", nargs="+", help="Seasons (years) to train on, e.g. 2019 2021 2022")
    parser.add_argument("--validation-season", help="Season to report as a validation slice")
    parser.add_argument("--test-season", help="Future season held out as the test set")
    args = parser.parse_args()
    print(
        train(
            args.input,
            args.output,
            args.test_fraction,
            args.train_seasons,
            args.validation_season,
            args.test_season,
        )
    )


if __name__ == "__main__":
    main()
