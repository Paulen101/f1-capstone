"""Race situation of one driver at each decision lap, taken from a finished TUM race (Python 3.8).

TUM's VSE decides during lap L, after the lap's pace and on-track fights, before pit entry. A stop
decided then makes L the in-lap. The features below describe what is known at that moment: the state
at the end of lap L - 1, the tyre age including lap L and the flag state of lap L. Nothing later.

Used by label_generator.py --situations on the run whose candidate pit lap is the latest, because in
that run the driver has not stopped before any earlier candidate, so its history up to every
candidate lap is the one all candidates share. See notes/situation_sampling.md.
"""
CLOSE_AHEAD_S = 1.5   # TUM's "close ahead" threshold


def _gap(race, lap: int, idx: int, offset: int):
    """Race-time gap at the end of `lap` to the car `offset` positions away (-1 ahead, +1 behind), or None."""
    position = int(race.positions[lap, idx])
    other = [i for i in range(race.no_drivers) if int(race.positions[lap, i]) == position + offset]
    if not other or not race.bool_driving[lap, other[0]]:
        return None
    return round(float(abs(race.racetimes[lap, idx] - race.racetimes[lap, other[0]])), 3)


def _pitted_on(driver, lap: int) -> bool:
    return any(stop[0] == lap for stop in driver.strategy_info[1:])


def features(race, idx: int, lap: int, dry_compounds: list, location_cat: int, pit_loss_s: float) -> dict:
    """Features of driver `idx` at decision lap `lap` (>= 2). `dry_compounds` is ordered hardest first."""
    driver = race.drivers_list[idx]
    tot = race.race_pars["tot_no_laps"]
    prev = lap - 1
    start_compound, start_age = driver.strategy_info[0][1], driver.strategy_info[0][2]
    position = int(race.positions[prev, idx])
    behind = [i for i in range(race.no_drivers) if int(race.positions[prev, i]) == position + 1]
    gap_ahead, gap_behind = _gap(race, prev, idx, -1), _gap(race, prev, idx, +1)
    gap_ahead_before = _gap(race, prev - 1, idx, -1) if prev >= 1 else None
    return {"Lap": lap, "LapsRemaining": tot - lap, "RaceProgress": round(lap / tot, 4),
            "TyreAge": start_age + lap, "TyreAgeProgress": round((start_age + lap) / tot, 4),
            "Compound": start_compound, "RelCompound": dry_compounds.index(start_compound),
            "Position": position,
            "GapAhead_s": gap_ahead, "GapBehind_s": gap_behind,
            "CloseAheadPrevLap": gap_ahead_before is not None and gap_ahead_before <= CLOSE_AHEAD_S,
            "PursuerPittedPrevLap": bool(behind) and _pitted_on(race.drivers_list[behind[0]], prev),
            "FlagState": race.flagstates[lap], "RemainingStops": 1, "LocationCat": location_cat,
            "PitLoss_s": pit_loss_s}
