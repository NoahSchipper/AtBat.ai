"""Download Statcast batted-ball data and the MLBAM/Retrosheet ID crosswalk.

Requires the optional ``statcast`` extra (``pip install -e .[statcast]``),
which pulls in ``pybaseball`` and ``pandas``. This module is intentionally
kept separate from the rest of the pipeline, which avoids pandas.
"""

from __future__ import annotations

import argparse
from pathlib import Path

BATTED_BALL_COLUMNS = [
    "game_date",
    "game_pk",
    "batter",
    "pitcher",
    "events",
    "description",
    "launch_speed",
    "launch_angle",
    "launch_speed_angle",
    "hit_distance_sc",
    "bb_type",
]


def download_crosswalk(output_path: Path) -> Path:
    """Download the MLBAM <-> Retrosheet player ID crosswalk (Chadwick Bureau register)."""
    from pybaseball import chadwick_register

    register = chadwick_register()
    columns = ["key_mlbam", "key_retro", "name_last", "name_first"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    register[columns].to_csv(output_path, index=False)
    return output_path


def download_season(season: str, output_path: Path) -> Path:
    """Download all batted-ball events (balls actually put in play) for one season.

    Only rows with ``type == "X"`` (ball in play) carry launch metrics, so
    non-batted-ball pitches are dropped before saving to keep the file small.
    """
    from pybaseball import statcast

    data = statcast(start_dt=f"{season}-03-01", end_dt=f"{season}-11-15")
    batted = data[data["type"] == "X"].copy()
    available = [column for column in BATTED_BALL_COLUMNS if column in batted.columns]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    batted[available].to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    crosswalk_parser = subparsers.add_parser(
        "crosswalk", help="Download the MLBAM/Retrosheet player ID crosswalk"
    )
    crosswalk_parser.add_argument(
        "--output", type=Path, default=Path("data/raw/statcast/id_crosswalk.csv")
    )

    season_parser = subparsers.add_parser("season", help="Download one season of batted-ball events")
    season_parser.add_argument("season")
    season_parser.add_argument("--output", type=Path)

    args = parser.parse_args()
    if args.command == "crosswalk":
        print(download_crosswalk(args.output))
    elif args.command == "season":
        output = args.output or Path(f"data/raw/statcast/{args.season}_batted_balls.csv")
        print(download_season(args.season, output))


if __name__ == "__main__":
    main()
