"""Pit-timing network: features, numpy forward pass and the VSE adapter (Python 3.8 and 3.10, numpy only).

The network is trained in .venv-train (data_pipeline/train_pit_network.py, PyTorch) and saved as plain
numpy weights (.npz), so the simulator's Python 3.8 environment runs it without PyTorch or TensorFlow.

At each decision lap it sees the race situation of integration/situations.py plus four tyre features from
the race file, and predicts the pit-now advantage of notes/situation_sampling.md: pitting now minus pitting
at the best later lap, in finishing position and in race time (negative: pitting now is better). The
driver pits when pitting now is predicted to be at least as good (DECISION_RULE below).
"""
import math
from typing import Optional

import numpy as np

GAP_CAP_S = 30.0          # gaps above this (or missing: leader / last car) count as free air
POSITION_TIE = 0.1        # predicted position advantages within +-0.1 places count as a tie, decided by time

# Order of the network's inputs. Changing it needs a retrained network.
FEATURES = [
    "RaceProgress", "LapsRemaining", "TyreAge", "TyreAgeProgress",
    "RelHard", "RelMedium", "RelSoft",
    "Position", "GapAhead_s", "GapBehind_s", "NoCarAhead", "NoCarBehind",
    "CloseAheadPrevLap", "PursuerPittedPrevLap", "FlagSC", "FlagVSC",
    "LocationCat", "PitLoss_s",
    "TyreDelta_s", "WearRate_s", "StopWearRate_s", "StayCost_s",
]
TARGETS = ["AdvantagePosition", "AdvantageTime_s"]


def degradation(tire_pars: dict, compound: str, age: float) -> float:
    """Lap-time penalty (s) of `compound` at `age`, TUM's lin/quad/cub/ln models (calc_tire_degradation)."""
    k = tire_pars[compound]
    model = tire_pars["tire_deg_model"]
    if model == "lin":
        return k["k_0"] + k["k_1_lin"] * age
    if model == "quad":
        return k["k_0"] + k["k_1_quad"] * age + k["k_2_quad"] * age ** 2
    if model == "cub":
        return k["k_0"] + k["k_1_cub"] * age + k["k_2_cub"] * age ** 2 + k["k_3_cub"] * age ** 3
    if model == "ln":
        return k["k_0"] + k["k_1_ln"] * math.log(k["k_2_ln"] * age + 1.0)
    raise ValueError("Unknown tyre model %s" % model)


def tyre_features(tire_pars: dict, compound: str, stop_compound: str, age: int, laps_remaining: int) -> dict:
    """How much the current tyres cost, from the race file's degradation model of this driver.

    TyreDelta_s     current tyres at `age` minus fresh tyres of the stop compound, one lap
    WearRate_s      how much slower the current tyres get on the next lap
    StopWearRate_s  mean per-lap wear of the stop compound over the laps left
    StayCost_s      time the current tyres lose over the laps left compared with fresh stop tyres now
    """
    remaining = max(int(laps_remaining), 1)
    stay = sum(degradation(tire_pars, compound, age + a) - degradation(tire_pars, stop_compound, a)
               for a in range(remaining))
    return {"TyreDelta_s": degradation(tire_pars, compound, age) - degradation(tire_pars, stop_compound, 0),
            "WearRate_s": degradation(tire_pars, compound, age + 1) - degradation(tire_pars, compound, age),
            "StopWearRate_s": (degradation(tire_pars, stop_compound, remaining)
                               - degradation(tire_pars, stop_compound, 0)) / remaining,
            "StayCost_s": stay}


def _flag(value) -> float:
    if isinstance(value, str):
        return 1.0 if value.strip().lower() == "true" else 0.0
    return 1.0 if bool(value) else 0.0


def _gap(value):
    """(capped gap, missing flag); a missing gap means no car there (leader / last car)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return GAP_CAP_S, 1.0
    return min(float(value), GAP_CAP_S), 0.0


def feature_vector(situation: dict) -> list:
    """Network inputs from one situation (integration/situations.py fields plus tyre_features)."""
    ahead, no_ahead = _gap(situation["GapAhead_s"])
    behind, no_behind = _gap(situation["GapBehind_s"])
    rel = int(situation["RelCompound"])
    flag = situation["FlagState"]
    values = {
        "RaceProgress": situation["RaceProgress"], "LapsRemaining": situation["LapsRemaining"],
        "TyreAge": situation["TyreAge"], "TyreAgeProgress": situation["TyreAgeProgress"],
        "RelHard": float(rel == 0), "RelMedium": float(rel == 1), "RelSoft": float(rel == 2),
        "Position": situation["Position"], "GapAhead_s": ahead, "GapBehind_s": behind,
        "NoCarAhead": no_ahead, "NoCarBehind": no_behind,
        "CloseAheadPrevLap": _flag(situation["CloseAheadPrevLap"]),
        "PursuerPittedPrevLap": _flag(situation["PursuerPittedPrevLap"]),
        "FlagSC": float(flag == "SC"), "FlagVSC": float(flag == "VSC"),
        "LocationCat": situation["LocationCat"], "PitLoss_s": situation["PitLoss_s"],
    }
    for name in ("TyreDelta_s", "WearRate_s", "StopWearRate_s", "StayCost_s"):
        values[name] = situation[name]
    return [float(values[name]) for name in FEATURES]


class PitNetwork:
    """Multilayer perceptron with ReLU, standardized inputs; outputs TARGETS in their own units."""

    def __init__(self, path):
        data = np.load(str(path), allow_pickle=False)
        if [str(x) for x in data["features"]] != FEATURES:
            raise ValueError("Network was trained on other features: retrain it")
        self.mean, self.std = data["x_mean"], data["x_std"]
        self.y_scale = data["y_scale"]
        n = int(data["n_layers"])
        self.layers = [(data["w%d" % i], data["b%d" % i]) for i in range(n)]

    def predict(self, x) -> np.ndarray:
        h = (np.atleast_2d(np.asarray(x, dtype=np.float64)) - self.mean) / self.std
        for i, (w, b) in enumerate(self.layers):
            h = h @ w.T + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        return h * self.y_scale


def pit_now(advantage_position: float, advantage_time: float) -> bool:
    """DECISION_RULE: the project's label rule. Position first; a near-tie in position is decided by time."""
    if advantage_position < -POSITION_TIE:
        return True
    if advantage_position > POSITION_TIE:
        return False
    return advantage_time <= 0.0


class NetworkVSE:
    """One driver's single stop is timed by the network; other drivers delegate to TUM's realstrategy.

    Install after Race construction, before simulate_race(), through the public Race.vse property, like
    FixedLapVSE. The adapter keeps the Race object to read the situation the way situations.py does. It
    stops at most once, onto `stop_compound`, from `first_lap` on, and stops on the last possible lap
    (tot_no_laps - 1) if it has not yet, because TUM requires two compounds.
    """

    def __init__(self, original_vse, race, driver: str, network: PitNetwork, stop_compound: str,
                 location_cat: int, pit_loss_s: float, first_lap: int = 2):
        if original_vse is None:
            raise ValueError("Enable the original VSE before attaching this adapter")
        if any(kind != "realstrategy" for kind in original_vse.vse_pars["vse_type"].values()):
            raise ValueError("Background drivers must use realstrategy")
        self.original_vse = original_vse
        self.race = race
        self.driver = driver
        self.idx = next(i for i, d in enumerate(race.drivers_list) if d.initials == driver)
        self.network = network
        self.stop_compound = stop_compound
        self.location_cat = location_cat
        self.pit_loss_s = pit_loss_s
        self.first_lap = first_lap
        self.dry_compounds = original_vse.param_dry_compounds
        self.decisions = []

    def __getattr__(self, name):
        return getattr(self.original_vse, name)

    def situation(self, lap: int) -> dict:
        from integration.situations import features
        situation = features(self.race, self.idx, lap, self.dry_compounds, self.location_cat, self.pit_loss_s)
        tire_pars = self.race.drivers_list[self.idx].tireset_pars
        situation.update(tyre_features(tire_pars, situation["Compound"], self.stop_compound,
                                       situation["TyreAge"], situation["LapsRemaining"]))
        return situation

    def decide(self, lap: int, tot_no_laps: int) -> Optional[str]:
        if lap >= tot_no_laps - 1:
            return self.stop_compound
        if lap < self.first_lap:
            return None
        situation = self.situation(lap)
        prediction = self.network.predict(feature_vector(situation))[0]
        pit = pit_now(float(prediction[0]), float(prediction[1]))
        self.decisions.append({"Lap": lap, "PredPosition": round(float(prediction[0]), 3),
                               "PredTime_s": round(float(prediction[1]), 3), "Pit": pit})
        return self.stop_compound if pit else None

    def decide_pitstop(self, **state):
        decisions = list(self.original_vse.decide_pitstop(**state))
        idx = state["driver_initials"].index(self.driver)
        decision = None
        if bool(state["bool_driving"][idx]) and state["no_past_tirechanges"][idx] == 0:
            decision = self.decide(state["cur_lap"], state["tot_no_laps"])
        decisions[idx] = decision
        return decisions
