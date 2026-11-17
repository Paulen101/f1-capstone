"""Find a race parameter file: a TUM file by name, or one of ours by path (Python 3.8 compatible)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUM_PARAMETERS = ROOT / "race-simulation" / "racesim" / "input" / "parameters"


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
