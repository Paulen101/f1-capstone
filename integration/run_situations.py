"""Record race situations for every driver of the fitted 2019 races (.venv-tum). Methodology step 4.

    run_situations.py --runs 200 --jobs 28 [--tracks Spielberg Monza Singapore Austin] [--drivers VER HAM]

For each driver it runs label_generator.py --situations with one stop onto the driver's real second
compound; a driver who never changed compound takes the compound most other drivers finished on.
Finished drivers are skipped, so the batch can be restarted. Output:
integration/output/situations/<Track>_2019_fitted_<driver>_mc<runs>/ and a log per driver. Then:

    .venv-data/bin/python data_pipeline/build_situations.py integration/output/situations/*_2019_fitted_*_mc200
"""
import argparse
from collections import Counter
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from integration.write_fitted_pars import parse_pars  # noqa: E402

OUT = ROOT / "integration/output/situations"
YEAR = 2019


def stop_compounds(pars: dict) -> dict:
    """Compound of each driver's single stop: the first compound after the start one."""
    strategies = {d: v["strategy_info"] for d, v in pars["driver_pars"].items()}
    finals = Counter(s[-1][1] for s in strategies.values() if len({x[1] for x in s}) > 1)
    result = {}
    for driver, strategy in strategies.items():
        start = strategy[0][1]
        later = [stint[1] for stint in strategy[1:] if stint[1] != start]
        result[driver] = later[0] if later else next(c for c, _ in finals.most_common() if c != start)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracks", nargs="+", default=["Spielberg", "Monza", "Singapore", "Austin"])
    parser.add_argument("--drivers", nargs="+", help="default: every driver of the race")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=28)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for track in args.tracks:
        race_file = ROOT / "integration/parameters" / ("pars_%s_%d_fitted.ini" % (track, YEAR))
        compounds = stop_compounds(parse_pars(race_file.read_bytes().decode("utf-8")))
        for driver in args.drivers or sorted(compounds):
            if driver not in compounds:
                continue
            name = "%s_%d_fitted_%s_mc%d" % (track, YEAR, driver, args.runs)
            if (OUT / name / "situations.csv").is_file():
                print("%s: already done" % name, flush=True)
                continue
            started = time.time()
            with open(OUT / (name + ".log"), "w") as log:
                code = subprocess.call([sys.executable, str(ROOT / "integration/label_generator.py"),
                                        "--race", str(race_file), "--driver", driver,
                                        "--compound", compounds[driver], "--runs", str(args.runs),
                                        "--jobs", str(args.jobs), "--situations", "--out", str(OUT / name)],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            print("%s: %s (%.0f s)" % (name, "done" if code == 0 else "FAILED, see log", time.time() - started),
                  flush=True)


if __name__ == "__main__":
    main()
