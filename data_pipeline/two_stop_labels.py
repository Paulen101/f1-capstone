"""Exact replay of one- and two-stop policies on the stored two-stop runs, and labels from it (.venv-train).

Input: folders of integration/two_stop_generator.py. In Monte Carlo run r every plan shares the same random
race up to its first difference, so a policy that decides lap by lap is replayed exactly:

    first stint   it stops on the first lap L1 it says pit (forced on the second-to-last lap)
    second stint  after a stop on L1 that allows a second stop, it stops again on the first lap L2 of the
                  window [L1 + min stint, laps - min stint] it says pit, or never
    outcome       the stored outcome of the plan (L1, L2), or (L1, no second stop)

Two tables, one row per decision:

    stint 2  (run, L1, lap in the window) plus one terminal row per (run, L1): no second stop
    stint 1  (run, lap): pitting now is worth what following the stint-2 policy after a stop now is worth

Targets, as in data_pipeline/stopping_labels.py: pitting now minus following the policy from the next lap,
in the same run (position: finished both ways; race time: also the same number of laps).

Compound choice: with the runs of integration/compound_runs.py (`extra`), stint 1 has one row per lap and legal
stop compound. The stored compound C1 comes first in each lap (Default). A policy then stops on at most one row
per lap, the compound it chooses (choose_compound). The forced stop on the last lap and a second stop stay
with C1, as in the runs: a first stop onto another compound is a one-stop race.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUTCOME = ["PitPosition", "PitRaceTime_s", "PitFinished", "PitLaps"]
G1 = ["Race", "Driver", "Run"]
G2 = ["Race", "Driver", "Run", "L1"]
TERMINAL = 10 ** 4          # lap of the "no second stop" row, after every real lap


def read_runs(folder: Path) -> pd.DataFrame:
    runs = pd.read_csv(folder / "runs.csv").rename(columns={
        "Position": "PitPosition", "RaceTime_s": "PitRaceTime_s", "Finished": "PitFinished",
        "SimulatedLaps": "PitLaps"})
    if (runs.Seed != runs.Run).any():
        raise ValueError("%s: a race was re-run with another seed; the replay would not be exact" % folder)
    return runs


def load_folder(folder: Path, extra: Path = None):
    """Tables of one driver-race; `extra`: its folder of integration/compound_runs.py, if any."""
    runs = read_runs(folder)
    situations = pd.read_csv(folder / "situations.csv").drop(columns=["Seed"])
    c1, c2 = runs.C1.iloc[0], runs.loc[runs.L2 > 0, "C2"].iloc[0] if (runs.L2 > 0).any() else None
    race, driver = runs.Race.iloc[0], runs.Driver.iloc[0]
    one = runs[runs.L2 == 0].set_index(["Run", "L1"])[OUTCOME]
    two = runs[runs.L2 > 0]

    # stint 2: every second-stop lap, then the terminal "no second stop" row
    t2 = two[["Run", "L1", "L2"] + OUTCOME].rename(columns={"L2": "Lap"})
    stay = one.loc[one.index.isin(pd.MultiIndex.from_frame(two[["Run", "L1"]].drop_duplicates()))].reset_index()
    stay["Lap"] = TERMINAL
    t2 = pd.concat([t2, stay], ignore_index=True)
    s2 = situations[situations.L1 > 0]
    t2 = t2.merge(s2, on=["Run", "L1", "Lap"], how="left", validate="one_to_one")
    t2["StopCompound"] = c2

    # stint 1: a stop on every lap
    t1 = one.reset_index().rename(columns={"L1": "Lap"})
    s1 = situations[situations.L1 == 0].drop(columns=["L1"])
    t1 = t1.merge(s1, on=["Run", "Lap"], how="left", validate="one_to_one")
    t1["StopCompound"] = c1
    t1["HasSecond"] = t1.Lap.isin(two.L1.unique())
    t1["Default"] = True
    if extra is not None:
        other = read_runs(Path(extra))
        if other.L1.max() != t1.Lap.max() or (other.C1 == c1).any():
            raise ValueError("%s does not match %s" % (extra, folder))
        o1 = other[["Run", "L1", "C1"] + OUTCOME].rename(columns={"L1": "Lap", "C1": "StopCompound"})
        o1 = o1.merge(s1, on=["Run", "Lap"], how="left", validate="one_to_one")
        o1["HasSecond"], o1["Default"] = False, False
        t1 = pd.concat([t1, o1], ignore_index=True)
    t2["FirstCompound"] = c1
    t1["FirstCompound"] = t1.StopCompound
    for table in (t1, t2):
        table["Race"], table["Driver"] = race, driver
        table["HasSituation"] = table.Position.notna()
    return t1, t2


def load_folders(folders, extra_root: Path = None):
    """All driver-races; with `extra_root`, the compound runs of each folder of the same name found there."""
    def extra(folder):
        found = Path(extra_root) / Path(folder).name if extra_root is not None else None
        return found if found is not None and (found / "manifest.json").is_file() else None
    parts = [load_folder(Path(f), extra(f)) for f in folders]
    t1 = pd.concat([p[0] for p in parts], ignore_index=True)
    t1["Alternative"] = ~t1.Default.astype(bool)
    t1 = t1.sort_values(G1 + ["Lap", "Alternative", "StopCompound"]).drop(columns="Alternative").reset_index(drop=True)
    t2 = pd.concat([p[1] for p in parts], ignore_index=True).sort_values(G2 + ["Lap"]).reset_index(drop=True)
    t1["LastLap"] = t1.groupby(["Race", "Driver"]).Lap.transform("max")
    t2["LastLap"] = TERMINAL
    return t1, t2


def follow(table: pd.DataFrame, pit: np.ndarray, group: list, values: pd.DataFrame = None) -> pd.DataFrame:
    """Outcome of following a policy from each row on: the first row at or after it where `pit` is true and
    a situation exists, or the forced stop on the last lap (stored compound). `values` replaces the stored outcomes (stint 1: what a stop
    is worth with the stint-2 policy after it)."""
    forced = (table.Lap == table.LastLap).to_numpy()
    if "Default" in table:
        forced &= table.Default.to_numpy(bool)      # the forced last-lap stop goes onto the stored compound
    stop = np.asarray(pit, bool) & table.HasSituation.to_numpy() | forced
    index = pd.Series(np.where(stop, np.arange(len(table)), np.nan))
    groups = [table[c] for c in group]
    # rows after the forced stop (other compounds on the last lap) point back to it
    nxt = index.groupby(groups).bfill().groupby(groups).ffill().to_numpy().astype(int)
    source = table[OUTCOME] if values is None else values
    out = source.iloc[nxt].reset_index(drop=True)
    out["StopLap"] = table.Lap.to_numpy()[nxt]
    if "FirstCompound" in table:
        out["StopCompound"] = table.FirstCompound.to_numpy()[nxt]
    return out


def stop_values(t1: pd.DataFrame, t2: pd.DataFrame, pit2: np.ndarray) -> pd.DataFrame:
    """What a first stop on each stint-1 row is worth when the stint-2 policy `pit2` follows it."""
    later = follow(t2, pit2, G2)
    first = (t2.groupby(G2).cumcount() == 0).to_numpy()
    worth = pd.concat([t2.loc[first, G2].reset_index(drop=True), later[first].reset_index(drop=True)], axis=1)
    worth = worth.rename(columns={"L1": "Lap", "StopLap": "SecondStop"})
    merged = t1[G1 + ["Lap"]].merge(worth, on=G1 + ["Lap"], how="left")
    values = t1[OUTCOME].copy().reset_index(drop=True)
    has = merged.PitPosition.notna().to_numpy() & t1.Default.to_numpy(bool)
    for column in OUTCOME:
        values[column] = np.where(has, merged[column].to_numpy(), values[column].to_numpy())
    values["SecondStop"] = np.where(has, merged.SecondStop.fillna(TERMINAL), TERMINAL)
    values["PitFinished"] = values.PitFinished.astype(bool)
    return values


def advantage(now: pd.DataFrame, later: pd.DataFrame) -> pd.DataFrame:
    both = now.PitFinished.to_numpy(bool) & later.PitFinished.eq(True).to_numpy()
    same = both & (now.PitLaps.to_numpy() == later.PitLaps.to_numpy())
    return pd.DataFrame({"AdvantagePosition": np.where(both, now.PitPosition - later.PitPosition, np.nan),
                         "AdvantageTime_s": np.where(same, now.PitRaceTime_s - later.PitRaceTime_s, np.nan)})


def targets(t1, t2, pit1, pit2):
    """Targets of both tables for the policy (pit1 on stint-1 rows, pit2 on stint-2 rows)."""
    nxt2 = follow(t2, pit2, G2).groupby([t2[c] for c in G2]).shift(-1)
    y2 = advantage(t2[OUTCOME].reset_index(drop=True), nxt2)
    values = stop_values(t1, t2, pit2)
    later = follow(t1, pit1, G1, values)
    # following the policy from the next lap: the value from that lap's first row (every row of one lap
    # shares the race up to the stop, and the policy stops on at most one of them)
    lap = t1[G1 + ["Lap"]].reset_index(drop=True)
    first = ~lap.duplicated().to_numpy()
    starts = pd.concat([lap[first].reset_index(drop=True), later[first].drop(columns="StopCompound", errors="ignore")
                        .reset_index(drop=True)], axis=1)
    shifted = starts.groupby(G1)[OUTCOME + ["StopLap"]].shift(-1)
    starts = pd.concat([starts[G1 + ["Lap"]], shifted], axis=1)
    nxt1 = lap.merge(starts, on=G1 + ["Lap"], how="left", validate="many_to_one")
    y1 = advantage(values, nxt1)
    usable2 = t2.HasSituation.to_numpy() & (t2.Lap < TERMINAL).to_numpy()
    usable1 = t1.HasSituation.to_numpy() & (t1.Lap < t1.LastLap).to_numpy()
    return y1, y2, usable1, usable2


def outcomes(t1, t2, pit1, pit2) -> pd.DataFrame:
    """Each run's outcome under the policy, with its stops: one row per (race, driver, run)."""
    later = follow(t1, pit1, G1, stop_values(t1, t2, pit2))
    first = (t1.groupby(G1).cumcount() == 0).to_numpy()
    out = t1.loc[first, G1].reset_index(drop=True)
    return pd.concat([out, later[first].reset_index(drop=True)], axis=1)


def choose_compound(table: pd.DataFrame, prediction: np.ndarray, position_tie: float) -> np.ndarray:
    """The row a policy would stop on in each (race, driver, run, lap): the compound with the best predicted
    position advantage; compounds within `position_tie` of it are decided by predicted time. Rows without a
    situation are never chosen. integration/pit_network.NetworkVSE chooses the same way."""
    pos, t = prediction[:, 0], prediction[:, 1]
    has = table.HasSituation.to_numpy(bool)
    key = table[G1 + ["Lap"]]
    frame = pd.DataFrame({"pos": np.where(has, pos, np.inf), "t": np.where(has, t, np.inf)})
    best_pos = frame.pos.groupby([key[c] for c in key]).transform("min").to_numpy()
    frame["t"] = np.where(frame.pos.to_numpy() <= best_pos + position_tie, frame.t, np.inf)
    best_t = frame.t.groupby([key[c] for c in key]).transform("min").to_numpy()
    chosen = has & (frame.t.to_numpy() == best_t) & np.isfinite(best_t)
    return chosen & ~pd.Series(chosen).groupby([key[c] for c in key]).cumsum().gt(1).to_numpy()


def best_plan(t1, t2, choose_from_runs) -> tuple:
    """Fixed plans per driver-race chosen on the given runs (best mean position, then race time): the best
    one-stop lap onto the stored compound, the best one-stop lap and compound, and the best plan of all.
    Returned as (pit1, pit2) policies."""
    def pick(frame, keys):
        f = frame[frame.PitFinished.astype(bool)]
        mean = f.groupby(["Race", "Driver"] + keys).agg(P=("PitPosition", "mean"), T=("PitRaceTime_s", "mean"))
        return mean.reset_index().sort_values(["Race", "Driver", "P", "T"]).groupby(["Race", "Driver"]).first()
    r1 = t1[choose_from_runs(t1)]
    one = pick(r1[r1.Default.astype(bool)], ["Lap", "StopCompound"])
    any_compound = pick(r1, ["Lap", "StopCompound"])
    r2 = t2[choose_from_runs(t2)]
    plans = pd.concat([r1.assign(L1=r1.Lap, L2=TERMINAL)[["Race", "Driver", "L1", "StopCompound", "L2"] + OUTCOME],
                       r2.assign(L2=r2.Lap, StopCompound=r2.FirstCompound)[
                           ["Race", "Driver", "L1", "StopCompound", "L2"] + OUTCOME]])
    plans = plans.drop_duplicates()   # one-stop plans on the stored compound appear in both tables
    both = pick(plans, ["L1", "StopCompound", "L2"])
    policies = {}
    for name, frame, l1col, l2col in (("fixed one-stop", one, "Lap", None),
                                      ("fixed one-stop, any compound", any_compound, "Lap", None),
                                      ("fixed best plan", both, "L1", "L2")):
        index = pd.MultiIndex.from_frame(t1[["Race", "Driver"]])
        lap1 = frame[l1col].reindex(index).to_numpy()
        compound1 = frame["StopCompound"].reindex(index).to_numpy()
        pit1 = (t1.Lap.to_numpy() == lap1) & (t1.StopCompound.to_numpy() == compound1)
        if l2col is None:
            pit2 = np.zeros(len(t2), bool)
        else:
            lap2 = frame[l2col].reindex(pd.MultiIndex.from_frame(t2[["Race", "Driver"]])).to_numpy()
            pit2 = t2.Lap.to_numpy() == lap2
        policies[name] = (pit1, pit2)
    return policies


def compare(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    both = a.PitFinished.to_numpy(bool) & b.PitFinished.to_numpy(bool)
    same = both & (a.PitLaps.to_numpy() == b.PitLaps.to_numpy())
    dp = (a.PitPosition - b.PitPosition).to_numpy()[both]
    dt = (a.PitRaceTime_s - b.PitRaceTime_s).to_numpy()[same]
    return {"PosDiff": float(dp.mean()), "PosDiff_SE": float(dp.std(ddof=1) / np.sqrt(len(dp))),
            "TimeDiff_s": float(dt.mean()), "TimeDiff_s_SE": float(dt.std(ddof=1) / np.sqrt(len(dt))),
            "Runs": int(len(a))}
