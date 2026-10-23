"""One planned stop, with a predetermined dry compound (Python 3.8 compatible)."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FixedLapPolicy:
    pit_lap: int
    compound: str

    def __post_init__(self):
        if type(self.pit_lap) is not int or self.pit_lap < 1:
            raise ValueError("pit_lap must be a positive integer")
        if not self.compound or self.compound in ("I", "W"):
            raise ValueError("Choose a parameterized dry compound")

    def decide(self, cur_lap: int, tot_no_laps: int,
               is_driving: bool, no_past_tirechanges: int) -> Optional[str]:
        if not 1 <= self.pit_lap < tot_no_laps:
            raise ValueError("The stop needs an in-lap and an out-lap before the race ends")
        if is_driving and no_past_tirechanges == 0 and cur_lap == self.pit_lap:
            return self.compound
        return None


class FixedLapVSE:
    """Replace one driver's decisions; delegate other drivers to TUM's realstrategy.

    Install after Race construction, before simulate_race(). This uses the public
    Race.vse property and requires no changes to LGPL simulator source files.
    """

    def __init__(self, original_vse, driver: str, policy: FixedLapPolicy):
        if original_vse is None:
            raise ValueError("Enable the original VSE before attaching this adapter")
        if policy.compound not in original_vse.param_dry_compounds:
            raise ValueError("The fixed compound is not parameterized for this race")
        if any(kind != "realstrategy" for kind in original_vse.vse_pars["vse_type"].values()):
            raise ValueError("This experiment requires realstrategy for all background drivers")
        self.original_vse = original_vse
        self.driver = driver
        self.policy = policy
        self.decisions = []

    def __getattr__(self, name):
        return getattr(self.original_vse, name)

    def decide_pitstop(self, **state):
        decisions = list(self.original_vse.decide_pitstop(**state))
        idx = state["driver_initials"].index(self.driver)
        decision = self.policy.decide(
            cur_lap=state["cur_lap"], tot_no_laps=state["tot_no_laps"],
            is_driving=bool(state["bool_driving"][idx]),
            no_past_tirechanges=state["no_past_tirechanges"][idx])
        decisions[idx] = decision
        self.decisions.append({"Lap": state["cur_lap"], "Driver": self.driver,
                               "Pit": decision is not None, "Compound": decision or ""})
        return decisions
