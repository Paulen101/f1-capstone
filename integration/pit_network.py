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

# Order of the network's inputs. A saved network stores its own list (v1 and v2: without StopsMade).
FEATURES = [
    "RaceProgress", "LapsRemaining", "TyreAge", "TyreAgeProgress",
    "RelHard", "RelMedium", "RelSoft",
    "Position", "GapAhead_s", "GapBehind_s", "NoCarAhead", "NoCarBehind",
    "CloseAheadPrevLap", "PursuerPittedPrevLap", "FlagSC", "FlagVSC",
    "LocationCat", "PitLoss_s",
    "TyreDelta_s", "WearRate_s", "StopWearRate_s", "StayCost_s",
    "StopsMade",
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


def feature_vector(situation: dict, features: list = FEATURES) -> list:
    """Network inputs from one situation (integration/situations.py fields plus tyre_features).

    StopsMade is 0 when a situation has none (rows recorded before two-stop decisions existed)."""
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
    stops = situation.get("StopsMade", 0)
    values["StopsMade"] = 0.0 if stops is None or (isinstance(stops, float) and math.isnan(stops)) else float(stops)
    return [float(values[name]) for name in features]


class PitNetwork:
    """Multilayer perceptron with ReLU, standardized inputs; outputs TARGETS in their own units."""

    def __init__(self, path):
        data = np.load(str(path), allow_pickle=False)
        self.features = [str(x) for x in data["features"]]
        if any(name not in FEATURES for name in self.features):
            raise ValueError("Network was trained on unknown features: retrain it")
        self.columns = [FEATURES.index(name) for name in self.features]
        self.mean, self.std = data["x_mean"], data["x_std"]
        self.y_scale = data["y_scale"]
        n = int(data["n_layers"])
        self.layers = [(data["w%d" % i], data["b%d" % i]) for i in range(n)]

    def predict(self, x) -> np.ndarray:
        """x: rows in the network's own feature order, or in the full FEATURES order (columns are picked)."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if x.shape[1] == len(FEATURES) and len(self.features) != len(FEATURES):
            x = x[:, self.columns]
        h = (x - self.mean) / self.std
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


def best_choice(predictions) -> int:
    """Index of the compound to stop onto, from one (position, time) prediction per candidate: the best
    position advantage; candidates within POSITION_TIE of it are decided by time; the first on a full tie.
    data_pipeline/two_stop_labels.choose_compound is the same rule over the stored runs."""
    best_position = min(p[0] for p in predictions)
    close = [i for i, p in enumerate(predictions) if p[0] <= best_position + POSITION_TIE]
    return min(close, key=lambda i: predictions[i][1])


class FixedHead:
    """A compound head that always (or never) prefers the other compound: the per-driver rule of
    data_pipeline/train_compound_head.py, in the shape of a network."""

    def __init__(self, use_other: bool):
        self.use_other = bool(use_other)
        self.features = FEATURES

    def predict(self, x) -> np.ndarray:
        return np.full((len(np.atleast_2d(x)), 2), -1.0 if self.use_other else 1.0)


class NetworkVSE:
    """One driver's single stop is timed by the network; other drivers delegate to TUM's realstrategy.

    Install after Race construction, before simulate_race(), through the public Race.vse property, like
    FixedLapVSE. The adapter keeps the Race object to read the situation the way situations.py does. It
    stops at most once, from `first_lap` on, and stops on the last possible lap (tot_no_laps - 1) if it has
    not yet, because TUM requires two compounds.

    The compound is `stop_compound` (a fixed compound, as in the training rows), or, with `compound_choice`
    (integration/tum_compound_choice.py), what TUM's compound-choice network would mount if the driver pitted
    on this lap. The tyre features then describe that compound, so the network weighs the stop it would make.

    With `second_compound` the network may also make an optional second stop onto it, on laps at least
    `min_stint` after the first stop and at least `min_stint` before the end, as in the two-stop runs
    (integration/two_stop_generator.py). The first stop is then no longer forced to be the only one.

    With `other_compounds` (integration/compound_runs.py) the network chooses the first stop's compound as
    well: it predicts the advantage of a stop onto `stop_compound` and onto each other compound, takes the best
    (best_choice) and pits if that one says pit. The forced last-lap stop goes onto `stop_compound`, and a
    second stop only follows a first stop onto it, as in the runs it was trained on.

    With `other_compounds` and `compound_head` (data_pipeline/train_compound_head.py) the network times the stop
    on `stop_compound` as without a choice; only when it stops does the head predict the other compounds' advantage
    over `stop_compound` and the stop goes onto the best one if pit_now() says it is better.

    With `limits` (integration/race_file_plan.limits, for tracks the network has not trained on) each stop must
    fall in a window around the race file's plan: limits["first"] = [lo, hi] means no first stop before lap lo
    and a forced stop on lap hi if the network has not stopped; limits["second"] the same for the second stop,
    or None for no second stop. The windows of `min_stint` still apply.
    """

    def __init__(self, original_vse, race, driver: str, network: PitNetwork, stop_compound: Optional[str],
                 location_cat: int, pit_loss_s: float, first_lap: int = 2, compound_choice=None,
                 second_compound: Optional[str] = None, min_stint: int = 10, other_compounds=None,
                 compound_head=None, limits: Optional[dict] = None):
        if (stop_compound is None) == (compound_choice is None):
            raise ValueError("Give either a fixed stop compound or a compound choice")
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
        self.compound_choice = compound_choice
        self.location_cat = location_cat
        self.pit_loss_s = pit_loss_s
        self.first_lap = first_lap
        self.second_compound = second_compound
        self.min_stint = min_stint
        self.other_compounds = list(other_compounds or [])
        self.compound_head = compound_head
        self.limits = limits
        if limits is not None and limits.get("second") is not None and second_compound is None:
            raise ValueError("A second-stop window needs a second compound")
        if self.other_compounds and stop_compound is None:
            raise ValueError("Compound choice by the network needs the stored stop compound")
        self.dry_compounds = original_vse.param_dry_compounds
        self.decisions = []

    def __getattr__(self, name):
        return getattr(self.original_vse, name)

    def compound_for(self, lap: int) -> str:
        """Compound a stop decided on `lap` would mount."""
        if self.compound_choice is None:
            return self.stop_compound
        current = self.race.drivers_list[self.idx].strategy_info[0][1]
        tot = self.race.race_pars["tot_no_laps"]
        compound = self.compound_choice.choose(current, (lap - 1) / tot)   # TUM: raceprogress_prevlap
        if compound is None:
            raise RuntimeError("TUM's compound choice found no legal compound on lap %d" % lap)
        return compound

    def situation(self, lap: int, stop_compound: Optional[str] = None) -> dict:
        from integration.situations import features
        situation = features(self.race, self.idx, lap, self.dry_compounds, self.location_cat, self.pit_loss_s)
        tire_pars = self.race.drivers_list[self.idx].tireset_pars
        situation.update(tyre_features(tire_pars, situation["Compound"], stop_compound or self.compound_for(lap),
                                       situation["TyreAge"], situation["LapsRemaining"]))
        return situation

    def decide(self, lap: int, tot_no_laps: int) -> Optional[str]:
        if lap >= tot_no_laps - 1:
            return self.compound_for(lap)
        if lap < self.first_lap:
            return None
        if self.limits is not None:
            lo, hi = self.limits["first"]
            if lap < lo:
                return None
            if lap >= hi:
                self.decisions.append({"Lap": lap, "Pit": True, "Forced": "first", "StopsMade": 0})
                return self.compound_for(lap)
        if self.other_compounds and self.compound_head is not None:
            return self.ask_head(lap)
        if self.other_compounds:
            return self.ask_choice(lap, [self.stop_compound] + self.other_compounds)
        return self.ask(lap, self.compound_for(lap))

    def decide_second(self, lap: int, tot_no_laps: int) -> Optional[str]:
        first, first_compound = self.race.drivers_list[self.idx].strategy_info[1][:2]
        if self.second_compound is None or not first + self.min_stint <= lap <= tot_no_laps - self.min_stint:
            return None
        if self.other_compounds and first_compound != self.stop_compound:
            return None
        if self.limits is not None:
            if self.limits.get("second") is None:
                return None
            lo, hi = self.limits["second"]
            if lap < lo:
                return None
            if lap >= hi:
                self.decisions.append({"Lap": lap, "Pit": True, "Forced": "second", "StopsMade": 1})
                return self.second_compound
        return self.ask(lap, self.second_compound)

    def ask(self, lap: int, compound: str) -> Optional[str]:
        situation = self.situation(lap, compound)
        features = getattr(self.network, "features", FEATURES)
        prediction = self.network.predict(feature_vector(situation, features))[0]
        pit = pit_now(float(prediction[0]), float(prediction[1]))
        self.decisions.append({"Lap": lap, "PredPosition": round(float(prediction[0]), 3),
                               "PredTime_s": round(float(prediction[1]), 3), "Pit": pit, "Compound": compound,
                               "StopsMade": situation["StopsMade"]})
        return compound if pit else None

    def ask_head(self, lap: int) -> Optional[str]:
        if self.ask(lap, self.stop_compound) is None:
            return None
        features = getattr(self.compound_head, "features", FEATURES)
        rows = [feature_vector(self.situation(lap, c), features) for c in self.other_compounds]
        predictions = [(float(p[0]), float(p[1])) for p in self.compound_head.predict(rows)]
        best = best_choice(predictions)
        compound = self.other_compounds[best] if pit_now(*predictions[best]) else self.stop_compound
        self.decisions[-1].update({"Compound": compound, "Head": {c: [round(v, 3) for v in p]
                                                                  for c, p in zip(self.other_compounds, predictions)}})
        return compound

    def ask_choice(self, lap: int, compounds: list) -> Optional[str]:
        features = getattr(self.network, "features", FEATURES)
        rows = [feature_vector(self.situation(lap, c), features) for c in compounds]
        predictions = [(float(p[0]), float(p[1])) for p in self.network.predict(rows)]
        best = best_choice(predictions)
        pit = pit_now(*predictions[best])
        self.decisions.append({"Lap": lap, "PredPosition": round(predictions[best][0], 3),
                               "PredTime_s": round(predictions[best][1], 3), "Pit": pit,
                               "Compound": compounds[best], "StopsMade": 0,
                               "Candidates": {c: [round(v, 3) for v in p] for c, p in zip(compounds, predictions)}})
        return compounds[best] if pit else None

    def decide_pitstop(self, **state):
        decisions = list(self.original_vse.decide_pitstop(**state))
        idx = state["driver_initials"].index(self.driver)
        decision = None
        if bool(state["bool_driving"][idx]) and state["no_past_tirechanges"][idx] == 0:
            decision = self.decide(state["cur_lap"], state["tot_no_laps"])
        elif bool(state["bool_driving"][idx]) and state["no_past_tirechanges"][idx] == 1:
            decision = self.decide_second(state["cur_lap"], state["tot_no_laps"])
        decisions[idx] = decision
        return decisions
