"""Build point-in-time batter and pitcher features for model training."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

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


def _history() -> dict[str, int]:
    return {"pa": 0, **{outcome: 0 for outcome in OUTCOMES}}


def _rate(history: dict[str, int], outcome: str) -> str:
    pa = history["pa"]
    return f"{history[outcome] / pa:.8f}" if pa else ""


def _add_history_features(
    row: dict[str, str],
    prefix: str,
    history: dict[str, int],
) -> None:
    row[f"{prefix}_pa_prior"] = str(history["pa"])
    for outcome in OUTCOMES:
        row[f"{prefix}_{outcome}_prior"] = str(history[outcome])
        row[f"{prefix}_{outcome}_rate_prior"] = _rate(history, outcome)


def _update(history: dict[str, int], outcome: str) -> None:
    history["pa"] += 1
    history[outcome] += 1


def build_features(input_path: Path, output_path: Path) -> Path:
    """Create leakage-safe historical features from a combined PA CSV.

    Rows must be in chronological order. Each row receives features from
    histories before that PA, then its outcome is added to those histories.
    """
    with input_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError("Input dataset has no columns")
        required = {
            "date",
            "gid",
            "batter",
            "pitcher",
            "bathand",
            "pithand",
            "outcome",
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Feature input is missing columns: {sorted(missing)}")
        rows = list(reader)

    rows = [
        row
        for _, row in sorted(
            enumerate(rows),
            key=lambda item: (item[1]["date"], item[1]["gid"], item[0]),
        )
    ]

    batter_history: defaultdict[str, dict[str, int]] = defaultdict(_history)
    pitcher_history: defaultdict[str, dict[str, int]] = defaultdict(_history)
    batter_split: defaultdict[tuple[str, str], dict[str, int]] = defaultdict(_history)
    pitcher_split: defaultdict[tuple[str, str], dict[str, int]] = defaultdict(_history)

    feature_prefixes = (
        "batter",
        "pitcher",
        "batter_vs_pitcher_hand",
        "pitcher_vs_batter_hand",
    )
    feature_columns = [
        f"{prefix}_{suffix}"
        for prefix in feature_prefixes
        for suffix in (
            ["pa_prior"]
            + [f"{outcome}_prior" for outcome in OUTCOMES]
            + [f"{outcome}_rate_prior" for outcome in OUTCOMES]
        )
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as destination:
        fieldnames = list(reader.fieldnames) + feature_columns
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            batter_id = row["batter"]
            pitcher_id = row["pitcher"]
            batter_hand = row["bathand"]
            pitcher_hand = row["pithand"]
            batter = batter_history[batter_id]
            pitcher = pitcher_history[pitcher_id]
            batter_matchup = batter_split[(batter_id, pitcher_hand)]
            pitcher_matchup = pitcher_split[(pitcher_id, batter_hand)]

            _add_history_features(row, "batter", batter)
            _add_history_features(row, "pitcher", pitcher)
            _add_history_features(row, "batter_vs_pitcher_hand", batter_matchup)
            _add_history_features(row, "pitcher_vs_batter_hand", pitcher_matchup)
            writer.writerow(row)

            outcome = row["outcome"]
            if outcome not in OUTCOMES:
                raise ValueError(f"Unknown outcome: {outcome}")
            _update(batter, outcome)
            _update(pitcher, outcome)
            _update(batter_matchup, outcome)
            _update(pitcher_matchup, outcome)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/model_features.csv"),
    )
    args = parser.parse_args()
    print(build_features(args.input, args.output))


if __name__ == "__main__":
    main()
