"""Run TUM's neural-network VSEs for all drivers using .venv-tum; needs a CPU with AVX."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TUM = ROOT / "race-simulation"
RACE = "pars_Spielberg_2019.ini"


def main():
    if sys.version_info[:2] != (3, 8):
        sys.exit("Run this script with the .venv-tum Python (Python 3.8)")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tools" / "matplotlib"))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    sys.path.insert(0, str(TUM))
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from racesim.src.import_pars import import_pars
    from racesim.src.race import Race

    print("TensorFlow", tf.__version__)
    out = ROOT / "integration" / "output" / "demo_vse"
    out.mkdir(parents=True, exist_ok=True)
    for vse_type in ("supervised", "reinforcement"):
        print("\nAll drivers on the %s VSE" % vse_type)
        pars, paths = import_pars(use_print=False, use_vse=True, race_pars_file=RACE, mcs_pars_file="pars_mcs.ini")
        pars["vse_pars"]["vse_type"] = dict.fromkeys(pars["race_pars"]["participants"], vse_type)
        race = Race(**pars, vse_paths=paths, use_prob_infl=False, create_rand_events=False)
        race.simulate_race()
        if race.result_status != 0:
            raise RuntimeError("TUM rejected the simulation: result_status=%s" % race.result_status)
        race.check_valid_result()
        race.print_result()
        (out / vse_type).mkdir(exist_ok=True)
        race.export_results_as_csv(str(out / vse_type))
        # TUM's plot method ends in plt.show(), so save the figure at that point.
        figure = out / ("%s_racetime_gaps.png" % vse_type)
        plt.show = lambda: plt.savefig(str(figure), dpi=130, bbox_inches="tight")
        # np.sort puts the NaN race times of lapped cars last, as in TUM's main_racesim.py.
        t_winner = np.sort(race.racetimes[race.cur_lap, :])[0]
        race.plot_racetime_diffto_reflaptime(ref_laptime=t_winner / race.cur_lap)
        plt.close("all")
    print("\nCSVs and plots:", out)


if __name__ == "__main__":
    main()
