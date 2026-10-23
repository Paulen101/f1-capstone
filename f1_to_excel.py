#!/usr/bin/env python3
"""
f1_to_excel.py - Pull F1 race data from FastF1 and write one Excel workbook per season.

Scope: race sessions only (no sprints, no practice/quali), DRY races only by default.
A race counts as wet/mixed if any lap was run on INTERMEDIATE or WET tyres, or if
rainfall was recorded while the race was running. Excluded races still appear in the
"Races" sheet with the reason, so nothing disappears silently.

Usage:
    pip install fastf1 openpyxl pandas
    python f1_to_excel.py --years 2024                    # one season
    python f1_to_excel.py --years 2022 2023 2024 2025     # several seasons
    python f1_to_excel.py --years 2024 --rounds 1 2 3     # quick test on 3 races
    python f1_to_excel.py --years 2019 --include-wet      # keep wet races too

Seasons outside 2021-2022 need an F1TV token (paid tier): python -m fastf1 auth f1tv --authenticate

Output: <out>/F1_<year>_dry_races.xlsx
FastF1 caches everything it downloads in <cache>, so re-running is fast and a run
that stops halfway (rate limit, network) resumes where it left off.
"""
import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

WET_COMPOUNDS = {"INTERMEDIATE", "WET"}
WEATHER_COLS = ["AirTemp", "TrackTemp", "Humidity", "Pressure", "Rainfall", "WindSpeed", "WindDirection"]

log = logging.getLogger("f1_to_excel")


# --------------------------------------------------------------------------- helpers
def to_seconds(s: pd.Series) -> pd.Series:
    """Timedelta column -> float seconds (Excel can't store timedeltas cleanly)."""
    return pd.to_timedelta(s).dt.total_seconds().round(3)


def status_has(ts: pd.Series, code: str) -> pd.Series:
    """FastF1 lap TrackStatus is a string of every status code seen during the lap, e.g. '1', '24', '4'."""
    return ts.str.contains(code, regex=False)


# --------------------------------------------------------------------------- per-race processing
def build_laps(laps: pd.DataFrame, meta: dict) -> pd.DataFrame:
    df = pd.DataFrame({
        "Year": meta["Year"],
        "Round": meta["Round"],
        "Event": meta["Event"],
        "Driver": laps["Driver"].values,
        "DriverNumber": laps["DriverNumber"].values,
        "Team": laps["Team"].values,
        "LapNumber": laps["LapNumber"].astype("Int64").values,
        "Position": laps["Position"].astype("Int64").values,
        "LapTime_s": to_seconds(laps["LapTime"]).values,
        "Sector1_s": to_seconds(laps["Sector1Time"]).values,
        "Sector2_s": to_seconds(laps["Sector2Time"]).values,
        "Sector3_s": to_seconds(laps["Sector3Time"]).values,
        "Compound": laps["Compound"].values,
        "TyreLife": laps["TyreLife"].astype("Int64").values,
        "FreshTyre": laps["FreshTyre"].values,
        "Stint": laps["Stint"].astype("Int64").values,
        "PitInTime_s": to_seconds(laps["PitInTime"]).values,
        "PitOutTime_s": to_seconds(laps["PitOutTime"]).values,
        "LapStartTime_s": to_seconds(laps["LapStartTime"]).values,
        "LapEndTime_s": to_seconds(laps["Time"]).values,
        "SpeedST_kph": pd.to_numeric(laps["SpeedST"], errors="coerce").values,
        "TrackStatus": laps["TrackStatus"].fillna("").astype(str).values,
        "Deleted": laps["Deleted"].fillna(False).astype(bool).values,
        "IsAccurate": laps["IsAccurate"].fillna(False).astype(bool).values,
    })
    ts = df["TrackStatus"]
    df["IsInLap"] = df["PitInTime_s"].notna()
    df["IsOutLap"] = df["PitOutTime_s"].notna() & (df["LapNumber"] > 1)
    df["GreenFlag"] = ts == "1"
    df["Yellow"] = status_has(ts, "2")
    df["SC"] = status_has(ts, "4")
    df["VSC"] = status_has(ts, "6") | status_has(ts, "7")
    df["RedFlag"] = status_has(ts, "5")
    # Clean lap = usable for degradation / pace fitting.
    df["CleanLap"] = (
        df["LapTime_s"].notna()
        & df["GreenFlag"]
        & ~df["IsInLap"]
        & ~df["IsOutLap"]
        & df["IsAccurate"]
        & ~df["Deleted"]
        & (df["LapNumber"] > 1)
        & ~df["Compound"].isin(WET_COMPOUNDS)
    )
    return df


def build_stints(lapdf: pd.DataFrame) -> pd.DataFrame:
    d = lapdf.dropna(subset=["Stint"])
    if d.empty:
        return pd.DataFrame()
    g = d.groupby(["Year", "Round", "Event", "Driver", "Team", "Stint"], sort=False)
    out = g.agg(
        Compound=("Compound", "first"),
        StartLap=("LapNumber", "min"),
        EndLap=("LapNumber", "max"),
        StintLaps=("LapNumber", "count"),
        TyreLifeAtStart=("TyreLife", "first"),
        FreshTyre=("FreshTyre", "first"),
        CleanLaps=("CleanLap", "sum"),
    ).reset_index()
    clean = d[d["CleanLap"]].groupby(["Driver", "Stint"])["LapTime_s"]
    med = clean.median().rename("MedianCleanLapTime_s")
    best = clean.min().rename("BestCleanLapTime_s")
    out = out.join(med, on=["Driver", "Stint"]).join(best, on=["Driver", "Stint"])
    return out.sort_values(["Driver", "Stint"]).reset_index(drop=True)


def build_pitstops(lapdf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for drv, dl in lapdf.groupby("Driver", sort=False):
        dl = dl.set_index("LapNumber")
        stop_no = 0
        for lap_no, r in dl[dl["IsInLap"]].iterrows():
            stop_no += 1
            nxt = dl.loc[lap_no + 1] if (lap_no + 1) in dl.index else None
            has_out = nxt is not None and pd.notna(nxt["PitOutTime_s"])
            rows.append({
                "Year": r["Year"], "Round": r["Round"], "Event": r["Event"],
                "Driver": drv, "Team": r["Team"], "StopNumber": stop_no, "InLap": int(lap_no),
                "CompoundBefore": r["Compound"],
                "CompoundAfter": nxt["Compound"] if has_out else None,
                "PitLaneTime_s": round(nxt["PitOutTime_s"] - r["PitInTime_s"], 3) if has_out else np.nan,
                "InLapTime_s": r["LapTime_s"],
                "OutLapTime_s": nxt["LapTime_s"] if has_out else np.nan,
                "TrackStatusInLap": r["TrackStatus"],
                "UnderSCorVSC": bool(r["SC"] or r["VSC"]),
                "UnderRedFlag": bool(r["RedFlag"] or (has_out and nxt["RedFlag"])),
                "Note": "" if has_out else "No out-lap (retired in pits or race ended)",
            })
    return pd.DataFrame(rows)


def race_window(laps: pd.DataFrame):
    return laps["LapStartTime"].min(), laps["Time"].max()


def classify_conditions(laps: pd.DataFrame, weather: pd.DataFrame):
    start, end = race_window(laps)
    wet_laps = int(laps["Compound"].isin(WET_COMPOUNDS).sum())
    rain = False
    if weather is not None and not weather.empty and "Rainfall" in weather:
        w = weather[(weather["Time"] >= start) & (weather["Time"] <= end)]
        rain = bool(w["Rainfall"].fillna(False).astype(bool).any())
    reasons = []
    if wet_laps:
        reasons.append(f"{wet_laps} laps on INTERMEDIATE/WET")
    if rain:
        reasons.append("rainfall recorded during race")
    return ("WET/MIXED" if reasons else "DRY"), wet_laps, rain, "; ".join(reasons)


def build_weather(weather: pd.DataFrame, laps: pd.DataFrame, meta: dict) -> pd.DataFrame:
    if weather is None or weather.empty:
        return pd.DataFrame()
    start, end = race_window(laps)
    w = weather[(weather["Time"] >= start) & (weather["Time"] <= end)].copy()
    out = pd.DataFrame({"Year": meta["Year"], "Round": meta["Round"], "Event": meta["Event"],
                        "SessionTime_s": to_seconds(w["Time"]).values})
    for c in WEATHER_COLS:
        if c in w:
            out[c] = w[c].values
    return out


def build_track_status(track_status: pd.DataFrame, meta: dict) -> pd.DataFrame:
    if track_status is None or track_status.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "Year": meta["Year"], "Round": meta["Round"], "Event": meta["Event"],
        "SessionTime_s": to_seconds(track_status["Time"]).values,
        "Status": track_status["Status"].astype(str).values,
        "Message": track_status["Message"].values,
    })


def build_results(results: pd.DataFrame, meta: dict) -> pd.DataFrame:
    if results is None or results.empty:
        return pd.DataFrame()
    cols = ["Position", "ClassifiedPosition", "GridPosition", "Abbreviation", "FullName",
            "TeamName", "Status", "Points", "Laps"]
    out = results[[c for c in cols if c in results]].copy()
    if "Time" in results:
        out["ResultTime_s"] = to_seconds(results["Time"])
    out.insert(0, "Event", meta["Event"])
    out.insert(0, "Round", meta["Round"])
    out.insert(0, "Year", meta["Year"])
    return out.reset_index(drop=True)


def summarize_race(lapdf, pits, tstat, wdf, results, meta) -> dict:
    ts = tstat["Status"] if not tstat.empty else pd.Series(dtype=str)
    green_pits = pits[~pits["UnderRedFlag"] & pits["PitLaneTime_s"].notna()] if not pits.empty else pits
    classified = results[pd.to_numeric(results.get("ClassifiedPosition"), errors="coerce").notna()] \
        if not results.empty and "ClassifiedPosition" in results else pd.DataFrame()
    if not classified.empty and not pits.empty:
        stops = pits.groupby("Driver").size().reindex(classified["Abbreviation"]).fillna(0)
        mean_stops = round(float(stops.mean()), 2)
    else:
        mean_stops = np.nan
    return {
        "Year": meta["Year"], "Round": meta["Round"], "Event": meta["Event"],
        "RaceLaps": int(lapdf["LapNumber"].max()),
        "Drivers": int(lapdf["Driver"].nunique()),
        "Classified": len(classified) if not classified.empty else np.nan,
        "MeanStopsPerClassifiedDriver": mean_stops,
        "MedianPitLaneTime_s": round(float(green_pits["PitLaneTime_s"].median()), 3) if not green_pits.empty else np.nan,
        "StopsUnderSCorVSC": int(pits["UnderSCorVSC"].sum()) if not pits.empty else 0,
        "SCDeployments": int((ts == "4").sum()),
        "VSCDeployments": int((ts == "6").sum()),
        "RedFlags": int((ts == "5").sum()),
        "CleanLapShare": round(float(lapdf["CleanLap"].mean()), 3),
        "CompoundsUsed": ", ".join(sorted(lapdf["Compound"].dropna().astype(str).unique())),
        "MeanAirTemp_C": round(float(wdf["AirTemp"].mean()), 1) if not wdf.empty else np.nan,
        "MeanTrackTemp_C": round(float(wdf["TrackTemp"].mean()), 1) if not wdf.empty else np.nan,
    }


def process_race(laps, results, weather, track_status, meta, include_wet):
    """Pure-DataFrame core: everything except the network call. Returns (race_row, tables|None)."""
    condition, wet_laps, rain, reason = classify_conditions(laps, weather)
    race_row = {**meta, "Condition": condition, "WetLapCount": wet_laps,
                "RainfallDuringRace": rain, "Included": True, "ExcludeReason": "", "LoadStatus": "OK"}
    if condition != "DRY" and not include_wet:
        race_row.update(Included=False, ExcludeReason=reason)
        return race_row, None
    lapdf = build_laps(laps, meta)
    pits = build_pitstops(lapdf)
    wdf = build_weather(weather, laps, meta)
    tstat = build_track_status(track_status, meta)
    res = build_results(results, meta)
    tables = {
        "Laps": lapdf,
        "Stints": build_stints(lapdf),
        "PitStops": pits,
        "TrackStatus": tstat,
        "Weather": wdf,
        "Results": res,
        "TrackSummary": pd.DataFrame([summarize_race(lapdf, pits, tstat, wdf, res, meta)]),
    }
    return race_row, tables


# --------------------------------------------------------------------------- F1TV auth
F1_ARCHIVE_HOST = "livetiming.formula1.com"


class _F1TVAuth:
    """Adds the F1TV token to requests for F1's own timing archive only (never the mirror or Ergast)."""
    def __init__(self, token):
        self.token = token

    def __call__(self, r):
        if urlparse(r.url).hostname == F1_ARCHIVE_HOST:
            r.headers["Authorization"] = f"Bearer {self.token}"
        return r


def attach_f1tv_token():
    """The timing archive now returns 401 without an F1TV login. FastF1 3.8 only sends the
    token for live timing, so attach it to FastF1's archive requests ourselves."""
    import fastf1
    import jwt
    from fastf1.internals.f1auth import AUTH_DATA_FILE
    token = AUTH_DATA_FILE.read_text().strip() if AUTH_DATA_FILE.exists() else ""
    if not token:
        log.warning("No F1TV token found; only seasons on the free mirror (2021-2022) will load. "
                    "Run: python -m fastf1 auth f1tv --authenticate")
        return
    exp = jwt.decode(token, options={"verify_signature": False}).get("exp")
    if exp and exp < datetime.now(timezone.utc).timestamp():
        log.warning("F1TV token expired %s UTC; re-authenticate or archive downloads will fail.",
                    datetime.fromtimestamp(exp, timezone.utc).strftime("%Y-%m-%d %H:%M"))
    for sess in (fastf1.Cache._requests_session, fastf1.Cache._requests_session_cached):
        if sess is not None:
            sess.auth = _F1TVAuth(token)


# --------------------------------------------------------------------------- FastF1 I/O
def load_season(year, rounds, include_wet, cache_dir, offline=False):
    import fastf1
    fastf1.Cache.enable_cache(str(cache_dir))
    if offline:
        # Cache only: no network requests, so no F1TV token is needed. Uncached races show FAILED.
        fastf1.Cache.offline_mode(True)
    else:
        attach_f1tv_token()
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    if rounds:
        schedule = schedule[schedule["RoundNumber"].isin(rounds)]
    today = pd.Timestamp(datetime.now(timezone.utc).date())

    race_rows, collected = [], {}
    for _, ev in schedule.iterrows():
        meta = {"Year": year, "Round": int(ev["RoundNumber"]), "Event": ev["EventName"],
                "Country": ev["Country"], "Location": ev["Location"],
                "Date": pd.Timestamp(ev["EventDate"]).date()}
        if pd.Timestamp(ev["EventDate"]).normalize() >= today:
            race_rows.append({**meta, "Included": False, "LoadStatus": "NOT RUN YET"})
            continue
        log.info("%d R%02d %s", year, meta["Round"], meta["Event"])
        try:
            s = fastf1.get_session(year, meta["Round"], "R")
            s.load(laps=True, telemetry=False, weather=True, messages=True)
            if s.laps is None or s.laps.empty:
                raise RuntimeError("no lap data returned")
            row, tables = process_race(s.laps, s.results, s.weather_data, s.track_status, meta, include_wet)
        except Exception as e:  # keep going; record the failure in the Races sheet
            log.warning("  failed: %s", e)
            race_rows.append({**meta, "Included": False, "LoadStatus": f"FAILED: {type(e).__name__}: {e}"[:250]})
            continue
        race_rows.append(row)
        if tables:
            for k, v in tables.items():
                if not v.empty:
                    collected.setdefault(k, []).append(v)
    sheets = {k: pd.concat(v, ignore_index=True) for k, v in collected.items()}
    return pd.DataFrame(race_rows), sheets


# --------------------------------------------------------------------------- Excel output
DICTIONARY = [
    ("Races", "Condition", "DRY, or WET/MIXED if any lap used INTERMEDIATE/WET tyres or rainfall was recorded during the race."),
    ("Races", "Included", "True if the race's data is in the other sheets."),
    ("Races", "LoadStatus", "OK, NOT RUN YET (future event), or FAILED with the error. Re-run the script to retry failures."),
    ("Laps", "LapTime_s / Sector*_s", "Lap and sector times in seconds."),
    ("Laps", "TyreLife", "Laps on this set of tyres, including laps from earlier sessions if the set was used before."),
    ("Laps", "FreshTyre", "True if the set was new at the start of the stint."),
    ("Laps", "PitInTime_s / PitOutTime_s", "Session time (s) the car crossed pit entry / exit. Present on in-laps / out-laps only."),
    ("Laps", "LapEndTime_s", "Session time (s) when the lap ended. Gap to the car ahead at a lap end = difference of LapEndTime_s on the same LapNumber."),
    ("Laps", "SpeedST_kph", "Speed-trap speed on this lap, km/h. Max over a driver's race or qualifying approximates TUM's vel_max."),
    ("Laps", "TrackStatus","Every status code seen during the lap: 1 green, 2 yellow, 4 safety car, 5 red flag, 6 VSC deployed, 7 VSC ending."),
    ("Laps", "IsAccurate", "FastF1's own flag that the lap timing is trustworthy."),
    ("Laps", "CleanLap", "Green flag, not in/out lap, not lap 1, accurate, not deleted, dry tyres. Use these laps for degradation and pace fitting."),
    ("Stints", "TyreLifeAtStart", "Tyre age on the first lap of the stint (>1 means a used set)."),
    ("Stints", "MedianCleanLapTime_s", "Median of clean laps in the stint. NOT fuel-corrected."),
    ("PitStops", "PitLaneTime_s", "Pit entry to pit exit (stationary time + lane driving). This is not the full pit loss vs a racing lap."),
    ("PitStops", "InLapTime_s / OutLapTime_s", "Lap times of the laps either side of the stop. Pit loss = (in + out) - 2 x reference clean lap."),
    ("PitStops", "UnderSCorVSC", "Stop made on a lap with SC/VSC. These stops are cheaper and should be modelled separately."),
    ("PitStops", "UnderRedFlag", "Tyre change during a red flag. Excluded from MedianPitLaneTime_s."),
    ("TrackStatus", "Status", "Track status change events with session time. Codes as in Laps.TrackStatus."),
    ("Weather", "*", "Roughly one sample per minute during the race. Temperatures in C, wind in m/s."),
    ("Results", "ResultTime_s", "Results Time field from FastF1 in seconds. The winner's value is the total race time; check whether other drivers show totals or gaps before using it."),
    ("TrackSummary", "SCDeployments / VSCDeployments", "Number of times the status switched to 4 / 6 during the session."),
    ("TrackSummary", "MedianPitLaneTime_s", "Median pit-lane time for stops not made under a red flag."),
    ("TrackSummary", "CleanLapShare", "Fraction of all laps that are CleanLap."),
]
SHEET_ORDER = ["README", "Races", "TrackSummary", "Laps", "Stints", "PitStops",
               "TrackStatus", "Weather", "Results", "DataDictionary"]


def readme_frame(year, include_wet, races):
    import importlib.metadata as md
    try:
        ff1v = md.version("fastf1")
    except Exception:
        ff1v = "unknown"
    n_inc = int(races["Included"].fillna(False).sum()) if not races.empty else 0
    return pd.DataFrame({"Item": [
        "Season", "Generated (UTC)", "Source", "Session type", "Filter",
        "Races included", "Races in schedule", "Time units", "Notes"],
        "Value": [
        year, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"), f"FastF1 {ff1v}",
        "Grand Prix race only (sprints excluded)",
        "All races" if include_wet else "Dry races only (see Races sheet for exclusions)",
        n_inc, len(races), "Seconds",
        "Generated by f1_to_excel.py. Do not edit by hand; re-run the script instead."]})


def write_workbook(path, sheets):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name in SHEET_ORDER:
            if name in sheets and not sheets[name].empty:
                sheets[name].to_excel(xw, sheet_name=name, index=False)
        wb = xw.book
        head_font = Font(name="Arial", bold=True, color="FFFFFF")
        head_fill = PatternFill("solid", fgColor="1F3864")
        body_font = Font(name="Arial", size=10)
        for ws in wb.worksheets:
            for row in ws.iter_rows(min_row=2):
                for c in row:
                    c.font = body_font
            for c in ws[1]:
                c.font, c.fill = head_font, head_fill
                c.alignment = Alignment(vertical="center")
            ws.freeze_panes = "A2"
            if ws.max_row > 1:
                ws.auto_filter.ref = ws.dimensions
            for i, col in enumerate(ws.iter_cols(min_row=1, max_row=min(ws.max_row, 200)), start=1):
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[get_column_letter(i)].width = min(max(width + 2, 8), 60)
            if ws.title in ("README", "DataDictionary"):
                ws.column_dimensions["B" if ws.title == "README" else "C"].width = 100
                for row in ws.iter_rows(min_row=2):
                    for c in row:
                        c.alignment = Alignment(wrap_text=True, vertical="top")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--rounds", type=int, nargs="*", help="only these round numbers (for testing)")
    ap.add_argument("--include-wet", action="store_true", help="keep wet/mixed races")
    ap.add_argument("--out", default="f1_data", help="output folder for workbooks")
    ap.add_argument("--cache", default="fastf1_cache", help="FastF1 cache folder")
    ap.add_argument("--offline", action="store_true",
                    help="use the FastF1 cache only (no network, no F1TV token); uncached races show FAILED")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("fastf1").setLevel(logging.WARNING)
    out, cache = Path(args.out), Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    for year in args.years:
        races, sheets = load_season(year, args.rounds, args.include_wet, cache, offline=args.offline)
        sheets["Races"] = races
        sheets["README"] = readme_frame(year, args.include_wet, races)
        sheets["DataDictionary"] = pd.DataFrame(DICTIONARY, columns=["Sheet", "Column", "Meaning"])
        suffix = "all_races" if args.include_wet else "dry_races"
        path = out / f"F1_{year}_{suffix}.xlsx"
        write_workbook(path, sheets)
        ok = races["LoadStatus"].eq("OK").sum() if "LoadStatus" in races else 0
        failed = races["LoadStatus"].astype(str).str.startswith("FAILED").sum()
        log.info("%d: wrote %s  (%d loaded, %d included, %d failed)", year, path, ok,
                 int(races["Included"].fillna(False).sum()), failed)
        if failed:
            log.info("  Some races failed (a rate limit, or a missing/expired F1TV token). Re-run the same command later; cached races are skipped quickly.")


if __name__ == "__main__":
    sys.exit(main())
