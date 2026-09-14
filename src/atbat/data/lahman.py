"""Download the Lahman tables used by the initial data pipeline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from urllib.request import urlopen

BASE_URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/Lahman"
TABLES = ("People", "Batting", "Pitching", "Teams", "Parks")


def download_table(table: str, output_dir: Path) -> Path:
    """Download one Lahman table and return its local CSV path."""
    if table not in TABLES:
        raise ValueError(f"Unsupported Lahman table: {table}")

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{table}.csv"
    url = f"{BASE_URL}/{table}.csv"

    with urlopen(url, timeout=60) as response:
        destination.write_bytes(response.read())

    # Validate the download immediately so an HTML error page is not treated
    # as a valid dataset by later pipeline stages.
    with destination.open(newline="", encoding="utf-8") as csv_file:
        headers = next(csv.reader(csv_file), [])
    if not headers:
        raise ValueError(f"Downloaded Lahman table is empty: {table}")
    return destination


def download_tables(output_dir: Path, tables: tuple[str, ...] = TABLES) -> list[Path]:
    """Download the requested Lahman tables."""
    return [download_table(table, output_dir) for table in tables]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/lahman"),
        help="Directory in which to save Lahman CSV files.",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=TABLES,
        default=list(TABLES),
        help="Lahman tables to download.",
    )
    args = parser.parse_args()
    paths = download_tables(args.output_dir, tuple(args.tables))
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
