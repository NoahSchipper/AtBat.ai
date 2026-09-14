"""Join leakage-safe rolling Statcast power features onto the feature table.

Statcast batted-ball events (exit velocity, launch angle, barrel/hard-hit
classification) are keyed by MLBAM player IDs, not Retrosheet IDs, so they
are first translated with the Chadwick Bureau crosswalk produced by
``download-statcast crosswalk``. Because Statcast data cannot be reliably
sequenced within a single game against our Retrosheet-derived rows, prior
power stats only include batted balls from strictly earlier game dates
(same-day games are excluded from a row's "prior" window). This keeps the
join leakage-safe without requiring intra-game event ordering across the two
data sources.
"""

from __future__ import annotations

import argparse
import bisect
import csv
from collections import defaultdict
from pathlib import Path

HARD_HIT_THRESHOLD = 95.0
BARREL_LAUNCH_SPEED_ANGLE = "6"  # Statcast's launch_speed_angle==6 means "Barrel"

Snapshot = tuple[int, float, float, int, int]  # pa, speed_sum, angle_sum, barrel, hard_hit


def _load_crosswalk(path: Path) -> dict[str, str]:
    """Map MLBAM player id (string) -> Retrosheet id."""
    mapping: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            mlbam = row.get("key_mlbam", "")
            retro = row.get("key_retro", "")
            if mlbam and retro:
                mapping[mlbam] = retro
    return mapping


def _load_events(paths: list[Path], crosswalk: dict[str, str]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                batter_retro = crosswalk.get(row.get("batter", ""), "")
                pitcher_retro = crosswalk.get(row.get("pitcher", ""), "")
                if not batter_retro or not pitcher_retro:
                    continue
                date = row.get("game_date", "").replace("-", "")
                if not date:
                    continue
                launch_speed = row.get("launch_speed") or None
                launch_angle = row.get("launch_angle") or None
                events.append(
                    {
                        "date": date,
                        "batter": batter_retro,
                        "pitcher": pitcher_retro,
                        "launch_speed": float(launch_speed) if launch_speed else None,
                        "launch_angle": float(launch_angle) if launch_angle else None,
                        "is_barrel": row.get("launch_speed_angle") == BARREL_LAUNCH_SPEED_ANGLE,
                        "is_hard_hit": float(launch_speed) >= HARD_HIT_THRESHOLD if launch_speed else False,
                    }
                )
    return events


def _build_player_history(
    events: list[dict[str, object]], id_key: str
) -> dict[str, tuple[list[str], list[Snapshot]]]:
    """Build, per player, cumulative power snapshots at each distinct game date."""
    grouped: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for event in events:
        grouped[event[id_key]][event["date"]].append(event)

    history: dict[str, tuple[list[str], list[Snapshot]]] = {}
    for player_id, by_date in grouped.items():
        dates = sorted(by_date)
        snapshots: list[Snapshot] = []
        pa = 0
        speed_sum = 0.0
        angle_sum = 0.0
        barrel = 0
        hard_hit = 0
        for date in dates:
            for event in by_date[date]:
                pa += 1
                if event["launch_speed"] is not None:
                    speed_sum += event["launch_speed"]
                if event["launch_angle"] is not None:
                    angle_sum += event["launch_angle"]
                if event["is_barrel"]:
                    barrel += 1
                if event["is_hard_hit"]:
                    hard_hit += 1
            snapshots.append((pa, speed_sum, angle_sum, barrel, hard_hit))
        history[player_id] = (dates, snapshots)
    return history


def _prior_snapshot(
    history: dict[str, tuple[list[str], list[Snapshot]]], player_id: str, date: str
) -> Snapshot:
    entry = history.get(player_id)
    if entry is None:
        return (0, 0.0, 0.0, 0, 0)
    dates, snapshots = entry
    index = bisect.bisect_left(dates, date)
    if index == 0:
        return (0, 0.0, 0.0, 0, 0)
    return snapshots[index - 1]


def _snapshot_columns(prefix: str, snapshot: Snapshot) -> dict[str, str]:
    pa, speed_sum, angle_sum, barrel, hard_hit = snapshot
    return {
        f"{prefix}_batted_balls_prior": str(pa),
        f"{prefix}_avg_exit_velo_prior": f"{speed_sum / pa:.4f}" if pa else "",
        f"{prefix}_avg_launch_angle_prior": f"{angle_sum / pa:.4f}" if pa else "",
        f"{prefix}_barrel_rate_prior": f"{barrel / pa:.8f}" if pa else "",
        f"{prefix}_hard_hit_rate_prior": f"{hard_hit / pa:.8f}" if pa else "",
    }


def build_power_features(
    features_path: Path,
    batted_ball_paths: list[Path],
    crosswalk_path: Path,
    output_path: Path,
) -> Path:
    crosswalk = _load_crosswalk(crosswalk_path)
    events = _load_events(batted_ball_paths, crosswalk)
    batter_history = _build_player_history(events, "batter")
    pitcher_history = _build_player_history(events, "pitcher")

    with features_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError("Feature dataset has no columns")
        power_columns = [
            f"{prefix}_{suffix}"
            for prefix in ("batter", "pitcher_allowed")
            for suffix in (
                "batted_balls_prior",
                "avg_exit_velo_prior",
                "avg_launch_angle_prior",
                "barrel_rate_prior",
                "hard_hit_rate_prior",
            )
        ]
        fieldnames = list(reader.fieldnames) + power_columns
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                batter_snapshot = _prior_snapshot(batter_history, row["batter"], row["date"])
                pitcher_snapshot = _prior_snapshot(pitcher_history, row["pitcher"], row["date"])
                row.update(_snapshot_columns("batter", batter_snapshot))
                row.update(_snapshot_columns("pitcher_allowed", pitcher_snapshot))
                writer.writerow(row)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--batted-balls", type=Path, nargs="+", required=True)
    parser.add_argument("--crosswalk", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/model_features_with_power.csv")
    )
    args = parser.parse_args()
    print(build_power_features(args.features, args.batted_balls, args.crosswalk, args.output))


if __name__ == "__main__":
    main()
