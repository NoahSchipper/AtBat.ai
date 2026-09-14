# AtBat.ai

AtBat.ai is a baseball at-bat simulator. The initial milestone is a machine
learning model that predicts a plate-appearance outcome from the game state
and player context. The longer-term goal is to expand this into a complete
baseball game simulator.

## MVP prediction target

The model will predict one mutually exclusive outcome for each plate
appearance:

- Strikeout
- Walk or hit by pitch
- Single
- Double
- Triple
- Home run
- Ball-in-play out
- Reached on error or other reaching outcome

The first user-facing input contract includes:

- Batter and pitcher
- Away and home teams
- Inning number and top/bottom half
- Outs
- Runner occupancy and base placement
- Batter handedness, including an explicit left/right choice for switch hitters

The batting team can be derived from the inning half and the home/away teams.
The initial model will use time-aware historical validation so features only
use information available before the simulated plate appearance.

## Data setup

The first pipeline stage downloads the Lahman tables used for player, team,
park, batting, and pitching history. Install the project in a virtual
environment, then run:

```text
python -m pip install -e .
download-lahman
```

The files are written to `data/raw/lahman/`, which is intentionally ignored by
Git because historical datasets should be downloaded reproducibly rather than
committed to the repository. Lahman is a useful foundation for season-level
features, but it does not contain the plate-appearance outcome labels needed
for the final model.

Retrosheet supplies the missing historical play-by-play data. Its parsed
season archives include `plays.csv`, along with game and player context. To
download a manageable first sample:

```text
download-retrosheet 2023 --tables allplayers gameinfo plays
```

The extracted files are written to `data/raw/retrosheet/` with the season
prefix used by Retrosheet, such as `2023plays.csv`. Retrosheet data should be
reviewed against its
[data-use terms](https://www.retrosheet.org/datause.html) before
redistribution. The next pipeline stage will normalize the plays file into
the eight MVP outcome classes and join it to Lahman player/team IDs.

See [DATASET.md](DATASET.md) for the primary dataset's provenance, schema,
outcome definitions, current sample counts, reproduction commands, and
modeling limitations.

To combine multiple downloaded seasons into one dataset, pass each season's
`--plays`/`--gameinfo` files in matching order:

```text
build-plate-appearances `
  --plays data/raw/retrosheet/2019plays.csv data/raw/retrosheet/2021plays.csv data/raw/retrosheet/2022plays.csv data/raw/retrosheet/2023plays.csv `
  --gameinfo data/raw/retrosheet/2019gameinfo.csv data/raw/retrosheet/2021gameinfo.csv data/raw/retrosheet/2022gameinfo.csv data/raw/retrosheet/2023gameinfo.csv `
  --people data/raw/lahman/People.csv `
  --teams data/raw/lahman/Teams.csv `
  --parks data/raw/lahman/Parks.csv
```

(2020 is intentionally skipped — it was a shortened, non-representative
60-game season.)

This creates `data/processed/plate_appearances.csv`. The resulting rows
represent plate appearances, retain pre-at-bat state such as inning, outs, and
base occupancy, and include normalized outcome labels plus batter/pitcher,
park, and game-weather metadata. Features derived from future rows must still
be computed with point-in-time rolling windows before model training.

To build leakage-safe training features:

```text
build-model-features `
  --input data/processed/plate_appearances.csv `
  --output data/processed/model_features.csv
```

The feature builder sorts by date and game ID, emits each row's prior batter,
pitcher, and handedness-split histories, and only then incorporates that row's
outcome into subsequent histories. A first plate appearance therefore has
empty rates and zero prior counts rather than statistics that include its own
label.

## Predictor experiments

Train the predictor ablation benchmark with a genuine future-season holdout:

```text
python -m pip install -e .
train-models `
  --input data/processed/model_features.csv `
  --output data/processed/model_results.json `
  --train-seasons 2019 2021 2022 `
  --test-season 2023
```

This compares a training-frequency baseline, multinomial logistic regression
(as feature groups are added in sequence: game state, player history,
handedness splits, environment), and a histogram gradient boosting classifier
on the full feature set. Training rows are restricted to `--train-seasons`
and evaluated only against `--test-season`, a season the model never saw
during training — this is a real chronological holdout, not just a
within-season split. (Omit both flags to fall back to a within-file
chronological `--test-fraction` split.)

Each result now reports, in addition to log loss and accuracy:

- **Brier score** — mean squared error between predicted probabilities and
  the one-hot true outcome, another calibration-sensitive metric.
- **Per-outcome precision/recall/F1** — because accuracy and log loss are
  dominated by the frequent classes (`ball_in_play_out`, `strikeout`), the
  per-outcome breakdown is what actually shows whether the model captures
  rare-but-important outcomes like `home_run`, `double`, and `triple`.

The latest run (train 2019/2021/2022, test 2023) is written to
`data/processed/model_results.json`. Results so far: logistic regression and
gradient boosting both reach ~54% accuracy and a Brier score around 0.60,
comfortably ahead of the ~44.7% frequency baseline — but the per-outcome
breakdown shows the models are still essentially only distinguishing
`ball_in_play_out`, `strikeout`, and `walk_hbp` well; recall for `single`,
`double`, `home_run`, and `triple` is close to zero. This means the current
feature set (rolling career/split rates + game state + weather) is not yet
separating power/contact outcomes from generic outs. Likely next steps:
richer batter/pitcher power indicators (e.g. isolated power, barrel rate
proxies), park factors by outcome type, and probability-aware evaluation
(e.g. top-k or expected-value metrics) rather than hard classification
accuracy for rare classes.
