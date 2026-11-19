"""A wrong tyre model for the strategy, while the simulator keeps the true one (any Python 3.8+, numpy).

The race-file plan (integration/race_file_plan.py) and the network's tyre inputs (integration/pit_network.py)
read the fitted race file's degradation model, the same model the simulator races on. A real team only has an
estimate. `believed_tires` returns a copy of the race file's tyre parameters with an error per compound,
shared by every driver (as if the estimate came from the team's practice running):

    k_0                      + N(0, sigma_k0) seconds        (the compound's pace offset)
    wear terms (k_1, k_2, ...) * exp(N(0, sigma_wear))       (how fast the compound wears)

The error is drawn from (track, run, level), so every policy in a paired comparison believes the same wrong
model in the same race. TUM's VSE does not read the race file's tyre model and is not affected.
"""
import copy
import zlib

import numpy as np

LEVELS = {"exact": (0.0, 0.0), "moderate": (0.3, 0.3), "large": (0.6, 0.6)}   # (sigma_k0 s, sigma_wear)
WEAR_TERMS = ("k_1_lin", "k_1_quad", "k_2_quad", "k_1_cub", "k_2_cub", "k_3_cub", "k_1_ln")


def draw(compounds: list, track: str, run: int, level: str) -> dict:
    """{compound: (k0 offset s, wear factor)} for this race."""
    sigma_k0, sigma_wear = LEVELS[level]
    rng = np.random.default_rng([zlib.crc32(track.encode("utf-8")), int(run), zlib.crc32(level.encode("utf-8"))])
    return {c: (float(rng.normal(0.0, sigma_k0)) if sigma_k0 else 0.0,
                float(np.exp(rng.normal(0.0, sigma_wear))) if sigma_wear else 1.0) for c in sorted(compounds)}


def perturb(tire_pars: dict, errors: dict) -> dict:
    """One driver's tyre parameters with the errors applied (a copy)."""
    out = copy.deepcopy(tire_pars)
    for compound, (offset, factor) in errors.items():
        if compound not in out:
            continue
        values = out[compound]
        values["k_0"] = values["k_0"] + offset
        for term in WEAR_TERMS:
            if term in values:
                values[term] = values[term] * factor
    return out


def believed_tires(all_tire_pars: dict, compounds: list, track: str, run: int, level: str) -> dict:
    """{driver: tyre parameters as the strategy believes them} for one race."""
    errors = draw(compounds, track, run, level)
    return {driver: perturb(pars, errors) for driver, pars in all_tire_pars.items()}
