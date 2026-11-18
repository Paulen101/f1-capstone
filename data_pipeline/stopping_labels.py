"""Pit-timing labels that value waiting: replay any lap-by-lap policy on the stored Monte Carlo runs (.venv-train).

The situation folders of integration/run_situations.py hold, for every driver-race and Monte Carlo run r, the
outcome of stopping on every candidate lap L (runs.csv) and the race situation at every lap (situations.csv).
In run r every candidate shares the same random race up to its stop (same seed, no race was re-run), so a
policy that decides lap by lap is scored exactly: it stops on the first lap it says pit (or on the last
candidate lap, forced), and its outcome is that lap's outcome in run r.

The first labels compared pitting now with pitting on one fixed later lap, the best on average. Staying out
was therefore never credited with what a good decision later could gain, such as stopping under a safety car
that comes out later. Here the value of staying out at lap L is the outcome of following a policy from lap
L + 1 on, in the same run:

    advantage(r, L) = outcome(pit at L, run r) - outcome(follow the policy from L + 1, run r)

Training a network on these targets and replaying it again is policy iteration with exact policy evaluation
(data_pipeline/train_stopping_network.py). Rows without a situation (the driver retired earlier) are never
training rows and never decide anything; every later candidate has the same retired outcome anyway.
"""
from pathlib import Path

import numpy as np
import pandas as pd

GROUP = ["Race", "Driver", "Run"]
OUTCOME = ["PitPosition", "PitRaceTime_s", "PitFinished", "PitLaps"]


def load_folder(folder: Path) -> pd.DataFrame:
    """One row per (run, candidate lap): the outcome of stopping on that lap and the situation at it."""
    runs = pd.read_csv(folder / "runs.csv")
    situations = pd.read_csv(folder / "situations.csv")
    if (runs.Seed != runs.Run).any() or (situations.Seed != situations.Run).any():
        raise ValueError("%s: a race was re-run with another seed; the replay would not be exact" % folder)
    rows = runs.rename(columns={"PitLap": "Lap", "Compound": "StopCompound", "Position": "PitPosition",
                                "RaceTime_s": "PitRaceTime_s", "Finished": "PitFinished",
                                "SimulatedLaps": "PitLaps"})
    rows = rows[["Race", "Driver", "StopCompound", "Run", "Lap"] + OUTCOME]
    situations = situations.drop(columns=["Race", "Driver", "StopCompound", "Seed"])
    rows = rows.merge(situations, on=["Run", "Lap"], how="left", validate="one_to_one")
    rows["HasSituation"] = rows.Position.notna()
    rows["LastLap"] = rows.groupby(["Race", "Driver"]).Lap.transform("max")
    return rows


def load_folders(folders) -> pd.DataFrame:
    rows = pd.concat([load_folder(Path(f)) for f in folders], ignore_index=True)
    return rows.sort_values(GROUP + ["Lap"]).reset_index(drop=True)


def follow(rows: pd.DataFrame, pit: np.ndarray) -> pd.DataFrame:
    """Outcome of following a policy from each row's lap on: stop at the first lap >= L where `pit` is true
    (forced on the last lap). `pit` is one boolean per row of `rows` (sorted by run, then lap)."""
    stop = np.asarray(pit, dtype=bool) & rows.HasSituation.to_numpy() | (rows.Lap == rows.LastLap).to_numpy()
    index = pd.Series(np.where(stop, np.arange(len(rows)), np.nan))
    nxt = index.groupby([rows[c] for c in GROUP]).bfill().to_numpy().astype(int)
    out = rows[OUTCOME].iloc[nxt].reset_index(drop=True)
    out["StopLap"] = rows.Lap.to_numpy()[nxt]
    return out


def targets(rows: pd.DataFrame, pit: np.ndarray) -> pd.DataFrame:
    """Advantage of pitting now over following the policy from the next lap, for every row before the last lap.

    Position: when the driver finished both ways. Race time: when both also completed the same number of laps
    (a lapped driver's time is that of the last completed lap), as in data_pipeline/build_situations.py.
    """
    later = follow(rows, pit)
    nxt = later.groupby([rows[c] for c in GROUP]).shift(-1)       # following the policy from L + 1
    both = rows.PitFinished.to_numpy(bool) & nxt.PitFinished.eq(True).to_numpy()
    same = both & (rows.PitLaps.to_numpy() == nxt.PitLaps.to_numpy())
    return pd.DataFrame({
        "AdvantagePosition": np.where(both, rows.PitPosition - nxt.PitPosition, np.nan),
        "AdvantageTime_s": np.where(same, rows.PitRaceTime_s - nxt.PitRaceTime_s, np.nan),
        "LaterStopLap": nxt.StopLap.to_numpy(),
    })


def policy_outcomes(rows: pd.DataFrame, pit: np.ndarray) -> pd.DataFrame:
    """Each run's outcome under the policy: one row per (race, driver, run), from the first candidate lap."""
    later = follow(rows, pit)
    first = (rows.groupby(GROUP).cumcount() == 0).to_numpy()
    out = rows.loc[first, GROUP].reset_index(drop=True)
    return pd.concat([out, later[first].reset_index(drop=True)], axis=1)


def fixed_lap_pit(rows: pd.DataFrame, choose_from: np.ndarray) -> np.ndarray:
    """Policy that stops on one lap per driver-race: best mean position, then more laps, then race time
    (label_generator.pick_label), chosen only from the runs in `choose_from`."""
    part = rows[choose_from & rows.PitFinished.to_numpy(bool)]
    mean = part.groupby(["Race", "Driver", "Lap"]).agg(P=("PitPosition", "mean"), L=("PitLaps", "mean"),
                                                       T=("PitRaceTime_s", "mean")).reset_index()
    mean = mean.assign(negL=-mean.L).sort_values(["Race", "Driver", "P", "negL", "T"])
    label = mean.groupby(["Race", "Driver"]).Lap.first()
    lap = label.reindex(pd.MultiIndex.from_frame(rows[["Race", "Driver"]])).to_numpy()
    return rows.Lap.to_numpy() == lap


def compare(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Paired difference a - b over the same runs (position: both finished; time: same laps)."""
    both = a.PitFinished.to_numpy(bool) & b.PitFinished.to_numpy(bool)
    same = both & (a.PitLaps.to_numpy() == b.PitLaps.to_numpy())
    dp = (a.PitPosition - b.PitPosition).to_numpy()[both]
    dt = (a.PitRaceTime_s - b.PitRaceTime_s).to_numpy()[same]
    return {"PosDiff": float(dp.mean()), "PosDiff_SE": float(dp.std(ddof=1) / np.sqrt(len(dp))),
            "TimeDiff_s": float(dt.mean()), "TimeDiff_s_SE": float(dt.std(ddof=1) / np.sqrt(len(dt))),
            "Runs": int(len(a))}
