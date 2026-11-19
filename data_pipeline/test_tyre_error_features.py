"""Training rows with a wrong tyre model equal what the simulator-side strategy would compute (.venv-train).

    .venv-train/Scripts/python.exe -m unittest data_pipeline.test_tyre_error_features
"""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.train_pit_network import TRAIN_ERROR_SALT, add_tyre_features
from integration.pit_network import tyre_features
from integration.tyre_uncertainty import believed_tires
from integration.write_fitted_pars import parse_pars

ROOT = Path(__file__).resolve().parents[1]
RACES = ["pars_Catalunya_2019_fitted.ini", "pars_Monza_2019_fitted.ini"]   # quadratic wear, both


class WrongTyreFeatureTests(unittest.TestCase):
    def test_fast_path_equals_believed_tyres(self):
        pars = {r: parse_pars((ROOT / "integration/parameters" / r).read_bytes().decode("utf-8")) for r in RACES}
        rng = np.random.default_rng(1)
        rows = []
        for race in RACES:
            dry = pars[race]["vse_pars"]["param_dry_compounds"]
            for _ in range(30):
                compound, stop = rng.choice(dry, 2, replace=False)
                rows.append({"Race": race, "Driver": rng.choice(sorted(pars[race]["tireset_pars"])),
                             "Compound": compound, "StopCompound": stop, "TyreAge": int(rng.integers(0, 40)),
                             "LapsRemaining": int(rng.integers(1, 50)), "Run": int(rng.integers(0, 50))})
        rows = pd.DataFrame(rows)
        exact, wrong = add_tyre_features(rows), add_tyre_features(rows, "moderate")
        for e, w in zip(exact.itertuples(), wrong.itertuples()):
            p = pars[e.Race]
            believed = believed_tires(p["tireset_pars"], p["vse_pars"]["param_dry_compounds"], e.Race.split("_")[1],
                                      e.Run + TRAIN_ERROR_SALT, "moderate")[e.Driver]
            reference = tyre_features(believed, e.Compound, e.StopCompound, e.TyreAge, e.LapsRemaining)
            truth = tyre_features(p["tireset_pars"][e.Driver], e.Compound, e.StopCompound, e.TyreAge, e.LapsRemaining)
            for name, value in reference.items():
                self.assertAlmostEqual(getattr(w, name), value, places=9)
                self.assertAlmostEqual(getattr(e, name), truth[name], places=9)


if __name__ == "__main__":
    unittest.main()
