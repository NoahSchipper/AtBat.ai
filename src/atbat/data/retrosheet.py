"""Download parsed Retrosheet CSV data for a selected season."""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

BASE_URL = "https://www.retrosheet.org/downloads"
DEFAULT_TABLES = ("allplayers", "gameinfo", "plays")


def download_season(
    season: int,
    output_dir: Path,
    tables: tuple[str, ...] = DEFAULT_TABLES,
) -> list[Path]:
    """Download and extract a Retrosheet season archive."""
    if not 1871 <= season <= 2100:
        raise ValueError(f"Invalid season: {season}")

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_url = f"{BASE_URL}/{season}/{season}csvs.zip"
    with urlopen(archive_url, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))

    available = {
        Path(name).stem.lower(): name
        for name in archive.namelist()
        if name.lower().endswith(".csv")
    }
    paths: list[Path] = []
    for table in tables:
        archive_name = available.get(f"{season}{table}".lower())
        if archive_name is None:
            raise ValueError(f"Retrosheet table not found for {season}: {table}")

        destination = output_dir / Path(archive_name).name
        with archive.open(archive_name) as source:
            destination.write_bytes(source.read())
        with destination.open(newline="", encoding="utf-8") as csv_file:
            if not next(csv.reader(csv_file), []):
                raise ValueError(f"Downloaded Retrosheet table is empty: {table}")
        paths.append(destination)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("season", type=int, help="Season to download, such as 2023.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/retrosheet"),
        help="Directory in which to save extracted CSV files.",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        default=list(DEFAULT_TABLES),
        help="Tables to extract from the season archive.",
    )
    args = parser.parse_args()
    for path in download_season(args.season, args.output_dir, tuple(args.tables)):
        print(path)


if __name__ == "__main__":
    main()
