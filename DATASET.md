# Primary Modeling Dataset

## Overview

The current primary dataset is a combined, normalized plate-appearance dataset
built from:

- **Retrosheet** parsed play-by-play data, which supplies the event outcome and
  pre-appearance game state.
- **Lahman** player and team metadata, which supplies names, handedness, and
  team/league context.

The generated file is:

```text
data/processed/plate_appearances.csv
```

This file is generated locally and is intentionally not committed to Git.
Raw downloads are also ignored. Re-run the ingestion commands to reproduce
the dataset.

## Current sample

The current sample spans four seasons (2019, 2021, 2022, 2023 — 2020 is
skipped because its shortened 60-game schedule is not representative) and
contains **746,428 plate appearances**:

| Outcome | Count |
| --- | ---: |
| Ball-in-play out | 334,179 |
| Strikeout | 170,669 |
| Single | 104,461 |
| Walk/HBP | 71,742 |
| Double | 33,015 |
| Home run | 24,204 |
| Reached other | 5,326 |
| Triple | 2,832 |

All eight outcome classes are represented in every season, and the sample has
complete batter metadata coverage. Season sizes are roughly even
(~184k–190k plate appearances per season), so no single season dominates the
combined dataset.

For model training, 2019/2021/2022 are used as the training set and 2023 is
held out as a genuine future-season test set (see the "Predictor experiments"
section in [README.md](README.md)).

## Outcome labels

Each row is one plate appearance and has exactly one `outcome` value:

- `strikeout`
- `walk_hbp`
- `single`
- `double`
- `triple`
- `home_run`
- `ball_in_play_out`
- `reached_other`

The labels are derived from Retrosheet's parsed event flags. `reached_other`
currently includes reached-on-error and other uncommon reaching outcomes.

## Important fields

The dataset retains the source fields needed for the initial simulator and
feature pipeline, including:

- `gid`, `date`, `season`
- `inning`, `top_bot`, `vis_home`
- `batteam`, `pitteam`
- `outs_pre`
- `br1_pre`, `br2_pre`, `br3_pre`
- `score_v`, `score_h`
- `batter`, `pitcher`
- `bathand`, `pithand`
- `batter_nameFirst`, `batter_nameLast`, `batter_bats`, `batter_throws`
- `pitcher_nameFirst`, `pitcher_nameLast`, `pitcher_bats`, `pitcher_throws`
- `park_name`, `park_city`, `park_state`, `park_country`
- `temp`, `precip`, `sky`, `winddir`, `windspeed`
- `outcome`

The `br1_pre`, `br2_pre`, and `br3_pre` fields identify the runners occupying
first, second, and third before the plate appearance. Empty values mean that
base was unoccupied.

## Reproducing the dataset

Install the project, download the source files, and build the combined output:

```text
python -m pip install -e .
download-lahman --tables People Teams Parks --output-dir data/raw/lahman
download-retrosheet 2019 --tables plays gameinfo --output-dir data/raw/retrosheet
download-retrosheet 2021 --tables plays gameinfo --output-dir data/raw/retrosheet
download-retrosheet 2022 --tables plays gameinfo --output-dir data/raw/retrosheet
download-retrosheet 2023 --tables plays gameinfo --output-dir data/raw/retrosheet
build-plate-appearances `
  --plays data/raw/retrosheet/2019plays.csv data/raw/retrosheet/2021plays.csv data/raw/retrosheet/2022plays.csv data/raw/retrosheet/2023plays.csv `
  --gameinfo data/raw/retrosheet/2019gameinfo.csv data/raw/retrosheet/2021gameinfo.csv data/raw/retrosheet/2022gameinfo.csv data/raw/retrosheet/2023gameinfo.csv `
  --people data/raw/lahman/People.csv `
  --teams data/raw/lahman/Teams.csv `
  --parks data/raw/lahman/Parks.csv `
  --output data/processed/plate_appearances.csv
```

The backtick line continuation is for PowerShell. On other shells, place the
command on one line or use that shell's continuation syntax.

## Modeling limitations

This is the primary **outcome and game-state dataset**, not yet the final
feature table. Before training a model, we still need to:

- Build point-in-time rolling batter and pitcher statistics.
- Prevent future plate appearances from influencing earlier rows.
- Add park factors and historical weather where available.
- Define handling for rare batter-pitcher matchups.
- Validate Retrosheet and Lahman identifier coverage across additional seasons.

The leakage-safe feature table is generated separately as
`data/processed/model_features.csv`. It contains prior-history counts and
rates for each batter and pitcher, plus batter-versus-pitcher-handedness and
pitcher-versus-batter-handedness splits. Current-row outcomes are incorporated
only after that row's features are written.

Retrosheet game information also supplies the initial environmental features:
temperature, precipitation, sky condition, wind direction, and wind speed.
These are observed game conditions rather than a forward-looking weather
forecast, so they are appropriate only when the simulator is modeling a
historical game or when equivalent pre-game inputs are supplied.

Retrosheet and Lahman have separate data-use terms. Review their requirements
before redistributing downloaded data or derived files.
