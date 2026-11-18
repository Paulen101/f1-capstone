"""How much could any pit-timing policy gain on the stored runs? (.venv-train). Notes: pit_network.md, round 2.

    .venv-train/Scripts/python.exe data_pipeline/stopping_headroom.py

On the held-out Monte Carlo runs 160-199 of the 80 driver-races, against the best fixed lap (chosen on runs
0-159), split by races with and without a safety car or VSC:

    hindsight          the best candidate lap of each run, knowing its future: an upper bound no policy reaches
    wait for SC        stop at once under a safety car or VSC from `early` laps before the fixed lap on,
                       otherwise `late` laps after it
    pit_network_v1/v2  the trained networks, replayed exactly (data_pipeline/stopping_labels.py)
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_pipeline.stopping_labels import (GROUP, OUTCOME, compare, fixed_lap_pit, load_folders,  # noqa: E402
                                           policy_outcomes)
from data_pipeline.train_pit_network import VAL_FIRST_RUN, add_tyre_features  # noqa: E402
from data_pipeline.train_stopping_network import SITUATIONS, decisions  # noqa: E402
from integration.pit_network import FEATURES, PitNetwork, feature_vector  # noqa: E402


def main():
    rows = load_folders(sorted(SITUATIONS.glob("*_2019_fitted_*_mc200")))
    has = rows.HasSituation.to_numpy()
    val = (rows.Run >= VAL_FIRST_RUN).to_numpy()
    held = rows[val].reset_index(drop=True)
    fixed_pit = fixed_lap_pit(rows, ~val)
    fixed = policy_outcomes(held, fixed_pit[val])
    lap = rows.Lap.to_numpy()
    label = rows[fixed_pit].groupby(["Race", "Driver"]).Lap.first()
    label = label.reindex(pd.MultiIndex.from_frame(rows[["Race", "Driver"]])).to_numpy()
    neutral = rows.FlagState.isin(["SC", "VSC"]).to_numpy()
    sc_race = held.assign(N=held.FlagState.isin(["SC", "VSC"])).groupby(GROUP).N.any()
    sc_race = sc_race.reindex(pd.MultiIndex.from_frame(fixed[GROUP])).fillna(False).to_numpy(bool)

    policies = {}
    finished = held[held.PitFinished.astype(bool)].sort_values(["PitPosition", "PitRaceTime_s"])
    best = finished.groupby(GROUP).first().reset_index()[GROUP + OUTCOME + ["Lap"]].rename(columns={"Lap": "StopLap"})
    hindsight = fixed[GROUP].merge(best, on=GROUP, how="left")
    hindsight["PitFinished"] = hindsight.PitFinished.eq(True)
    policies["hindsight"] = hindsight
    for early, late in ((0, 0), (5, 0), (10, 0), (15, 0), (5, 2), (5, 4), (5, 8)):
        pit = (lap >= label + late) | (neutral & (lap >= label - early))
        policies["wait for SC: from fixed-%d, else fixed+%d" % (early, late)] = policy_outcomes(held, pit[val])
    situated = add_tyre_features(rows[has].reset_index())
    x = np.zeros((len(rows), len(FEATURES)))
    x[situated["index"].to_numpy()] = np.array([feature_vector(r) for r in situated.to_dict("records")])
    for name in ("pit_network_v1", "pit_network_v2"):
        network = PitNetwork(ROOT / "integration/models" / (name + ".npz"))
        policies[name] = policy_outcomes(held, decisions(network, x, has)[val])

    table = []
    for name, outcome in policies.items():
        entry = {"Policy": name}
        for part, mask in (("All", np.ones(len(fixed), bool)), ("SC or VSC", sc_race), ("Green only", ~sc_race)):
            c = compare(outcome[mask].reset_index(drop=True), fixed[mask].reset_index(drop=True))
            entry[part] = "%+.3f ± %.3f" % (c["PosDiff"], c["PosDiff_SE"])
        table.append(entry)
    table = pd.DataFrame(table)
    print("places against the best fixed lap, held-out runs (negative: better)\n" + table.to_string(index=False))
    table.to_csv(ROOT / "notes/evidence/pit_network/stopping_headroom.csv", index=False)


if __name__ == "__main__":
    main()
