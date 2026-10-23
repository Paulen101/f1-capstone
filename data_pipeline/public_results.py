"""Save public FastF1/Jolpica results when lap/weather archives are unavailable.

This creates a separate results-only workbook. It does not certify a dry race or
provide any observations for pace, tire degradation, or pit-loss fitting.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import fastf1
    import pandas as pd
    from f1_to_excel import build_results, write_workbook, DICTIONARY

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--rounds", type=int, nargs="+", default=[9])
    parser.add_argument("--offline", action="store_true", help="Use existing FastF1 HTTP cache only")
    args = parser.parse_args()
    cache = ROOT / "fastf1_cache"
    cache.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))
    fastf1.Cache.offline_mode(args.offline)
    results, races = [], []
    for rnd in args.rounds:
        session = fastf1.get_session(args.year, rnd, "R")
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        meta = {"Year": args.year, "Round": rnd, "Event": session.event["EventName"]}
        table = build_results(session.results, meta)
        if table.empty:
            raise RuntimeError("No public results returned for round %s" % rnd)
        results.append(table)
        races.append({**meta, "Condition": "UNVERIFIED", "Included": False,
                      "ExcludeReason": "No lap/weather data; dry/wet rule cannot be checked",
                      "LoadStatus": "RESULTS ONLY"})
    out = ROOT / "f1_data" / ("F1_%s_results_only.xlsx" % args.year)
    out.parent.mkdir(exist_ok=True)
    sheets = {
        "README": pd.DataFrame({"Item": ["Source", "Generated (UTC)", "Scope", "Dry/wet validation", "Times"],
                                 "Value": ["FastF1 %s; Jolpica public race results" % fastf1.__version__,
                                           datetime.now(timezone.utc).isoformat(),
                                           "Results only; NOT the dry-race dataset",
                                           "UNVERIFIED: timing and weather archives unavailable",
                                           "Raw FastF1 Time: winner total, other finishers may be gaps. Verify before comparing; NaN where unavailable."]}),
        "Races": pd.DataFrame(races),
        "Results": pd.concat(results, ignore_index=True),
        "DataDictionary": pd.DataFrame(
            [row for row in DICTIONARY if row[0] == "Results"] + [
                ("Races", "Condition", "UNVERIFIED: this results-only export cannot check tire or rainfall observations."),
                ("Races", "Included", "False: no session in this file is certified for the dry-race dataset."),
                ("Races", "LoadStatus", "RESULTS ONLY: public finishing results loaded; no lap/weather data.")],
            columns=["Sheet", "Column", "Meaning"]),
    }
    write_workbook(out, sheets)
    print("Wrote %s: %s results; no dry-race certification" % (out, sum(map(len, results))))


if __name__ == "__main__":
    main()
