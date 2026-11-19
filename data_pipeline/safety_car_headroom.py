"""How much could reacting to safety cars add to v4 + compound head? (.venv-train). Notes: pit_network.md, round 7.

    .venv-train/Scripts/python.exe data_pipeline/safety_car_headroom.py

On the two-stop runs with both compounds (data_pipeline/two_stop_labels.py, exact replay), against the best
policy so far, v4 + compound head (base), on the held-out runs 40-49, split by races with and without a safety
car or VSC before the last lap:

    hindsight, SC laps only   per run, the best of: the base plan; a first stop on any SC/VSC lap onto either
                              compound (v4's second-stint decisions after it); a second stop on any SC/VSC lap
                              after the base plan's first stop. No policy that only changes what it does under
                              a safety car can beat this.
    hindsight, any plan       the best stored plan of each run: the bound for any policy
    first stop under SC       base, but stop at once under SC/VSC once the race progress is at least p (the
                              head picks the compound); p chosen on the training runs 0-39
    second stop under SC      base, plus a second stop under SC/VSC (after a first stop onto the stored
                              compound) while at least r laps remain; r chosen on the training runs
    both                      the two rules together, with the chosen p and r
    hold for SC               when v4 first says pit (lap L), wait up to k laps for a safety car: stop at once
                              under SC/VSC from lap L on, otherwise on lap L + k; k chosen on the training runs
    SC lowers the bar         also stop under SC/VSC when v4's predicted position advantage of pitting now is
                              below theta (pit_now() needs -0.1); theta chosen on the training runs

The SC-only hindsight bound is also split by where its stop is: an SC lap before the base plan's first stop
(reacting), after it (waiting for a safety car that has not come yet), or a second stop.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_pipeline.train_compound_head import pairs, pit_now_vec, switch  # noqa: E402
from data_pipeline.train_stopping_network import decisions  # noqa: E402
from data_pipeline.train_two_stop_network import COMPOUND_RUNS, TWO_STOP, features, first_stint_decisions  # noqa: E402
from data_pipeline.two_stop_labels import G1, OUTCOME, TERMINAL, compare, load_folders, outcomes, stop_values  # noqa: E402
from integration.pit_network import PitNetwork  # noqa: E402

MODELS = ROOT / "integration/models"
NEUTRAL = ["SC", "VSC"]


def best_per_run(frames: list) -> pd.DataFrame:
    """Best outcome per run over candidate frames (G1 + OUTCOME): finished first, then position, then time."""
    all_ = pd.concat([f[G1 + OUTCOME] for f in frames], ignore_index=True)
    all_["Unfinished"] = ~all_.PitFinished.astype(bool)
    best = all_.sort_values(["Unfinished", "PitPosition", "PitRaceTime_s"]).groupby(G1).first().reset_index()
    return best.drop(columns="Unfinished")


def main():
    folders = sorted(TWO_STOP.glob("*_2019_fitted_*_mc50"))
    t1, t2 = load_folders(folders, COMPOUND_RUNS)
    x1, x2 = features(t1), features(t2)
    has2 = t2.HasSituation.to_numpy()
    timing = PitNetwork(MODELS / "pit_network_v4_two_stop.npz")
    head = PitNetwork(MODELS / "pit_compound_head_v1.npz")
    stored, other = pairs(t1)
    use_head = pit_now_vec(head.predict(x1[other]))
    pit2 = decisions(timing, x2, has2)
    pit1_v4 = first_stint_decisions(timing, x1, t1, stored_only=True)

    def run_policy(p1, p2=pit2):
        return outcomes(t1, t2, switch(p1, stored, other, use_head), p2)

    base = run_policy(pit1_v4)
    sc1 = (t1.FlagState.isin(NEUTRAL) & t1.HasSituation & (t1.Lap < t1.LastLap)).to_numpy()
    sc2 = (t2.FlagState.isin(NEUTRAL) & t2.HasSituation & (t2.Lap < TERMINAL)).to_numpy()
    default = t1.Default.to_numpy(bool)
    sc_race = t1[sc1].groupby(G1).size().reindex(pd.MultiIndex.from_frame(base[G1])).notna().to_numpy()
    train_runs = (base.Run < 40).to_numpy()

    # hindsight over SC laps only
    values = stop_values(t1, t2, pit2)
    first_sc = pd.concat([t1.loc[sc1, G1].reset_index(drop=True), values[sc1].reset_index(drop=True)], axis=1)
    base_stop = base[G1 + ["StopLap", "StopCompound"]].rename(columns={"StopLap": "L1"})
    stored_compound = t1[default].groupby(["Race", "Driver"]).StopCompound.first().rename("Stored").reset_index()
    base_stop = base_stop.merge(stored_compound, on=["Race", "Driver"])
    base_stop = base_stop[base_stop.StopCompound == base_stop.Stored][G1 + ["L1"]]
    second_sc = t2[sc2].merge(base_stop, on=G1 + ["L1"])
    first_sc["StopLap"] = t1.Lap.to_numpy()[sc1]
    first_sc = first_sc.merge(base[G1 + ["StopLap"]].rename(columns={"StopLap": "BaseLap"}), on=G1)
    policies = {"base: v4 + compound head": base,
                "hindsight, SC laps only": best_per_run([base, first_sc, second_sc]),
                "  ... SC stop before the base stop": best_per_run([base, first_sc[first_sc.StopLap < first_sc.BaseLap]]),
                "  ... SC stop after the base stop": best_per_run([base, first_sc[first_sc.StopLap > first_sc.BaseLap]]),
                "  ... second stop under SC": best_per_run([base, second_sc])}
    plans = pd.concat([t1[G1 + OUTCOME], t2[t2.Lap < TERMINAL][G1 + OUTCOME]])
    policies["hindsight, any plan"] = best_per_run([plans])

    # causal rules, parameters chosen on the training runs
    def score(out, runs):
        return compare(out[runs].reset_index(drop=True), base[runs].reset_index(drop=True))

    progress = t1.RaceProgress.to_numpy()
    remaining2 = t2.LapsRemaining.to_numpy()
    grid1 = {p: run_policy(pit1_v4 | (sc1 & default & (progress >= p))) for p in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)}
    grid2 = {r: run_policy(pit1_v4, pit2 | (sc2 & (remaining2 >= r))) for r in (10, 15, 20, 25, 30, 40)}
    p_best = min(grid1, key=lambda p: (score(grid1[p], train_runs)["PosDiff"], score(grid1[p], train_runs)["TimeDiff_s"]))
    r_best = min(grid2, key=lambda r: (score(grid2[r], train_runs)["PosDiff"], score(grid2[r], train_runs)["TimeDiff_s"]))
    policies["first stop under SC, progress >= %.1f" % p_best] = grid1[p_best]
    policies["second stop under SC, >= %d laps left" % r_best] = grid2[r_best]
    policies["both"] = run_policy(pit1_v4 | (sc1 & default & (progress >= p_best)), pit2 | (sc2 & (remaining2 >= r_best)))

    # hold for SC: v4's first pit lap per run, then wait up to k laps for a safety car
    lap = t1.Lap.to_numpy()
    first_pit = pd.Series(np.where(pit1_v4, lap, np.nan)).groupby([t1[c] for c in G1]).transform("min").to_numpy()
    grid3 = {k: run_policy(default & ((lap == first_pit + k) | (sc1 & (lap >= first_pit) & (lap < first_pit + k))))
             for k in (1, 2, 3, 4, 6, 8, 10)}
    # SC lowers the bar: v4's prediction on the stored compound
    pred = np.zeros((len(t1), 2))
    rows = t1.HasSituation.to_numpy() & default
    pred[rows] = timing.predict(x1[rows])
    grid4 = {th: run_policy(pit1_v4 | (sc1 & default & (pred[:, 0] < th))) for th in (0.0, 0.25, 0.5, 1.0, 1.5)}
    k_best = min(grid3, key=lambda k: (score(grid3[k], train_runs)["PosDiff"], score(grid3[k], train_runs)["TimeDiff_s"]))
    th_best = min(grid4, key=lambda t: (score(grid4[t], train_runs)["PosDiff"], score(grid4[t], train_runs)["TimeDiff_s"]))
    policies["hold for SC up to %d laps" % k_best] = grid3[k_best]
    policies["SC lowers the bar to %.2f" % th_best] = grid4[th_best]

    held = ~train_runs
    track = base.Race.str.extract(r"pars_(\w+?)_\d{4}")[0].to_numpy()
    table, grids = [], {"FirstStop": {}, "SecondStop": {}, "Hold": {}, "LowerBar": {}}
    for name, out in policies.items():
        out = base[G1].merge(out, on=G1, how="left")
        out["PitFinished"] = out.PitFinished.eq(True)
        for part, mask in (("all", held), ("SC or VSC", held & sc_race), ("green only", held & ~sc_race)):
            c = score(out, mask)
            table.append({"Policy": name, "Track": "all", "Races": part, **c})
        for t in sorted(set(track)):
            c = score(out, held & (track == t) & sc_race)
            table.append({"Policy": name, "Track": t, "Races": "SC or VSC", **c})
    for p, out in grid1.items():
        grids["FirstStop"]["%.1f" % p] = {"train": score(out, train_runs), "held_out": score(out, held)}
    for r, out in grid2.items():
        grids["SecondStop"][str(r)] = {"train": score(out, train_runs), "held_out": score(out, held)}
    for k, out in grid3.items():
        grids["Hold"][str(k)] = {"train": score(out, train_runs), "held_out": score(out, held)}
        for t in sorted(set(track)):
            grids["Hold"][str(k)][t] = {"train": score(out, train_runs & (track == t)),
                                        "held_out": score(out, held & (track == t))}
    for th, out in grid4.items():
        grids["LowerBar"][str(th)] = {"train": score(out, train_runs), "held_out": score(out, held)}
    table = pd.DataFrame(table)
    pd.set_option("display.width", 200)
    show = table.assign(Places=table.PosDiff.map("{:+.3f}".format) + " ± " + table.PosDiff_SE.map("{:.3f}".format),
                        Time=table.TimeDiff_s.map("{:+.2f} s".format))
    print("minus base (v4 + compound head), held-out runs 40-49 (negative: better)")
    print(show[["Policy", "Track", "Races", "Places", "Time", "Runs"]].to_string(index=False))
    print("share of held-out runs with SC/VSC before the last lap: %.2f" % sc_race[held].mean())
    evidence = ROOT / "notes/evidence/pit_network"
    table.round(4).to_csv(evidence / "safety_car_headroom.csv", index=False)
    (evidence / "safety_car_headroom_grid.json").write_text(json.dumps(grids, indent=1))


if __name__ == "__main__":
    main()
