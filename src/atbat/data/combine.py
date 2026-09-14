"""Normalize Retrosheet plays and join Lahman player and team metadata."""

from __future__ import annotations

import argparse
import csv
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


def is_flag(row: dict[str, str], column: str) -> bool:
    return row.get(column, "0") == "1"


def classify(row: dict[str, str]) -> str:
    """Map a Retrosheet plate appearance to one MVP outcome."""
    if is_flag(row, "k"):
        return "strikeout"
    if is_flag(row, "walk") or is_flag(row, "hbp"):
        return "walk_hbp"
    if is_flag(row, "single"):
        return "single"
    if is_flag(row, "double"):
        return "double"
    if is_flag(row, "triple"):
        return "triple"
    if is_flag(row, "hr"):
        return "home_run"
    if is_flag(row, "bip") and not is_flag(row, "roe"):
        return "ball_in_play_out"
    return "reached_other"


def read_lookup(path: Path, key: str, columns: tuple[str, ...]) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return {
            row[key]: {column: row.get(column, "") for column in columns}
            for row in csv.DictReader(source)
            if row.get(key)
        }


def read_game_context(path: Path) -> dict[str, dict[str, str]]:
    columns = (
        "site",
        "visteam",
        "hometeam",
        "daynight",
        "usedh",
        "fieldcond",
        "precip",
        "sky",
        "temp",
        "winddir",
        "windspeed",
        "attendance",
    )
    return read_lookup(path, "gid", columns)


_ENRICHED_COLUMNS = [
    "outcome",
    "batter_nameFirst",
    "batter_nameLast",
    "batter_bats",
    "batter_throws",
    "pitcher_nameFirst",
    "pitcher_nameLast",
    "pitcher_bats",
    "pitcher_throws",
    "season",
    "batting_league",
    "park_name",
    "park_city",
    "park_state",
    "park_country",
]

_GAME_CONTEXT_COLUMNS = [
    "daynight",
    "usedh",
    "fieldcond",
    "precip",
    "sky",
    "temp",
    "winddir",
    "windspeed",
    "attendance",
    "visteam",
    "hometeam",
]


def _process_plays(
    plays_path: Path,
    people: dict[str, dict[str, str]],
    teams: dict[str, dict[str, str]],
    gameinfo: dict[str, dict[str, str]],
    parks: dict[str, dict[str, str]],
    include_gameinfo: bool,
    writer: csv.DictWriter,
) -> None:
    with plays_path.open(newline="", encoding="utf-8") as source:
        plays = csv.DictReader(source)
        required = {"pa", "batter", "pitcher", "date", "batteam"}
        missing = required.difference(plays.fieldnames or ())
        if missing:
            raise ValueError(f"Retrosheet plays is missing columns: {sorted(missing)}")

        for row in plays:
            if row.get("pa") != "1":
                continue
            batter = people.get(row["batter"], {})
            pitcher = people.get(row["pitcher"], {})
            season = row["date"][:4]
            team = teams.get(row["batteam"], {})
            enriched = dict(row)
            enriched["outcome"] = classify(row)
            for prefix, metadata in (("batter_", batter), ("pitcher_", pitcher)):
                for column in ("nameFirst", "nameLast", "bats", "throws"):
                    enriched[f"{prefix}{column}"] = metadata.get(column, "")
            enriched["season"] = season
            enriched["batting_league"] = (
                team.get("lgID", "") if team.get("yearID", "") == season else ""
            )
            context = gameinfo.get(row["gid"], {})
            park = parks.get(context.get("site", ""), {})
            for source, target in (
                ("park.name", "park_name"),
                ("city", "park_city"),
                ("state", "park_state"),
                ("country", "park_country"),
            ):
                enriched[target] = park.get(source, "")
            for column in _GAME_CONTEXT_COLUMNS:
                if include_gameinfo:
                    enriched[column] = context.get(column, "")
            writer.writerow(enriched)


def build_dataset(
    plays_paths: list[Path],
    people_path: Path,
    teams_path: Path,
    output_path: Path,
    gameinfo_paths: list[Path] | None = None,
    parks_path: Path | None = None,
) -> Path:
    """Build a normalized, metadata-enriched plate-appearance CSV.

    Accepts one or more seasons of Retrosheet plays files (and matching
    gameinfo files, aligned by position) so that a single combined dataset
    can span multiple seasons.
    """
    if gameinfo_paths and len(gameinfo_paths) != len(plays_paths):
        raise ValueError("gameinfo_paths must match plays_paths one-to-one when provided")

    people = read_lookup(
        people_path,
        "retroID",
        ("nameFirst", "nameLast", "bats", "throws"),
    )
    teams = read_lookup(teams_path, "teamID", ("yearID", "lgID"))
    parks = (
        read_lookup(parks_path, "park.key", ("park.name", "city", "state", "country"))
        if parks_path
        else {}
    )

    with plays_paths[0].open(newline="", encoding="utf-8") as source:
        base_fieldnames = csv.DictReader(source).fieldnames or []

    include_gameinfo = bool(gameinfo_paths)
    fieldnames = list(base_fieldnames) + _ENRICHED_COLUMNS
    if include_gameinfo:
        fieldnames = fieldnames + _GAME_CONTEXT_COLUMNS

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for index, plays_path in enumerate(plays_paths):
            gameinfo = (
                read_game_context(gameinfo_paths[index]) if include_gameinfo else {}
            )
            _process_plays(
                plays_path,
                people,
                teams,
                gameinfo,
                parks,
                include_gameinfo,
                writer,
            )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plays", type=Path, nargs="+", required=True)
    parser.add_argument("--people", type=Path, required=True)
    parser.add_argument("--teams", type=Path, required=True)
    parser.add_argument("--gameinfo", type=Path, nargs="+")
    parser.add_argument("--parks", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/plate_appearances.csv"))
    args = parser.parse_args()
    print(
        build_dataset(
            args.plays,
            args.people,
            args.teams,
            args.output,
            args.gameinfo,
            args.parks,
        )
    )


if __name__ == "__main__":
    main()
