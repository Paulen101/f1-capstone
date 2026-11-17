"""Find a race parameter file: a TUM file by name, or one of ours by path (Python 3.8 compatible)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUM_PARAMETERS = ROOT / "race-simulation" / "racesim" / "input" / "parameters"
# Workbook location -> TUM track name, where they differ.
WORKBOOK_TO_TUM = {"Barcelona": "Catalunya", "Montr\u00e9al": "Montreal", "Le Castellet": "LeCastellet",
                   "Spa-Francorchamps": "Spa", "Mexico City": "MexicoCity", "S\u00e3o Paulo": "SaoPaulo",
                   "Yas Island": "YasMarina"}


def tum_track(location: str) -> str:
    """TUM's track name for a workbook location, e.g. Barcelona -> Catalunya."""
    return WORKBOOK_TO_TUM.get(location, location.replace(" ", ""))


def race_name(path) -> str:
    """<Track>_<year> from a TUM or fitted race file name, e.g. pars_Monza_2019_fitted.ini -> Monza_2019."""
    stem = Path(path).stem
    stem = stem[len("pars_"):] if stem.startswith("pars_") else stem
    return "_".join(stem.split("_")[:2])


def resolve_race_file(race: str) -> Path:
    """Return the absolute path of a race file.

    TUM's import_pars() joins its own parameter folder with the name it is given. Joining with an
    absolute path returns that path, so our fitted files load from outside the TUM tree and no
    TUM file is added or changed.
    """
    for candidate in (Path(race), ROOT / race, TUM_PARAMETERS / race):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("No race parameter file: %s" % race)


def is_tum_file(path: Path) -> bool:
    return path.resolve().parent == TUM_PARAMETERS.resolve()


def allow_recorded_fcy(pars: dict) -> list:
    """Let TUM run a race whose recorded FCY phases are closer together than its minimum distance.

    TUM rejects a race whose recorded phases are closer than min_dist_sc / min_dist_vsc (1.5 laps in
    its files), so its own Monza 2019 file does not run: two VSC phases 1.46 laps apart. The
    simulator only needs one phase per lap, so where a recorded gap is at least one lap the minimum
    is lowered to that gap, in memory. Returns the changes, (key, old, new).
    """
    fcy, limits = pars["event_pars"]["fcy_data"], pars["monte_carlo_pars"]
    if fcy["domain"] != "progress":
        return []
    changes = []
    phases = sorted(fcy["phases"])
    for first, second in zip(phases, phases[1:]):
        key = "min_dist_vsc" if first[2] == "VSC" else "min_dist_sc"
        gap = second[0] - first[1]
        if 1.0 <= gap < limits[key]:
            new = int(gap * 1000) / 1000.0 - 0.001
            changes.append((key, limits[key], new))
            limits[key] = new
    return changes
