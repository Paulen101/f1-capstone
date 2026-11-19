"""Compare policies raced in different evaluation folders on the same seeds (.venv-data or .venv-train).

    compare_policies.py --base all_five_tracks_mc100:tum new_tracks_baselines_mc100:tum \
        --real all_five_tracks_mc100:real new_tracks_baselines_mc100:real \
        --policy v8=eight_v8_exact_mc100:network_head ... --out notes/evidence/pit_network/eight_tracks.csv

Each source is FOLDER:POLICY, FOLDER relative to integration/output/evaluation/. Every row of the output is one
(label, tyre error) pair: the mean finishing-position difference to the base (TUM's VSE) with its standard error, the
race-time difference, the split into green and SC/VSC races, the difference to the real strategies, the share of
two-stop races and the position difference per track. --diff A-B adds rows for label A minus label B, paired by
track, driver, run and tyre error. Negative is better for the first policy. Uses the pairing of
summarise_evaluation.py.
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "integration/output/evaluation"
from summarise_evaluation import describe, paired  # noqa: E402  (same folder)


def load(sources, label):
    parts = []
    for source in sources:
        folder, policy = source.rsplit(":", 1)
        runs = pd.read_csv(EVALUATION / folder / "runs.csv")
        runs = runs[runs.Policy == policy].copy()
        if runs.empty:
            raise SystemExit("no rows for policy %s in %s" % (policy, folder))
        runs["Policy"] = label
        if "TyreError" not in runs:
            runs["TyreError"] = "exact"
        parts.append(runs)
    runs = pd.concat(parts, ignore_index=True)
    duplicated = runs.duplicated(["Track", "Driver", "Run", "TyreError"])
    if duplicated.any():
        raise SystemExit("%s: %d duplicated races" % (label, duplicated.sum()))
    return runs


def fmt(value, se=None, digits=2):
    if pd.isna(value):
        return ""
    text = "%+.*f" % (digits, value)
    return text + ("±%.*f" % (digits, se) if se is not None else "")


def row(diffs, tracks):
    whole = describe(diffs)
    out = {"VsTUM": fmt(whole["PosDiff"], whole["PosDiff_SE"]), "Time": fmt(whole["TimeDiff_s"], digits=1)}
    for events, name in (("green only", "Green"), ("SC or VSC", "SC")):
        part = diffs[diffs.Neutralised == events]
        out[name] = fmt(part.PosDiff.mean()) if len(part) else ""
    for track in tracks:
        out[track[:5]] = fmt(diffs[diffs.Track == track].PosDiff.mean())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", nargs="+", required=True, help="TUM's VSE, FOLDER:POLICY")
    parser.add_argument("--real", nargs="+", required=True, help="the real strategies, FOLDER:POLICY")
    parser.add_argument("--policy", nargs="+", required=True, action="append",
                        help="LABEL=FOLDER:POLICY [FOLDER:POLICY ...]; repeat the flag for more labels")
    parser.add_argument("--diff", nargs="*", default=[], help="LABEL_A-LABEL_B")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    base, real = load(args.base, "tum"), load(args.real, "real")
    tracks = sorted(set(base.Track))
    policies = {}
    for spec in args.policy:
        label, first = spec[0].split("=", 1)
        policies[label] = load([first] + spec[1:], label)

    rows = []
    for label, runs in policies.items():
        for error, part in runs.groupby("TyreError", sort=False):
            both = pd.concat([part, base.assign(Policy="tum")])
            vs_tum = paired(both, label, "tum")
            vs_real = paired(pd.concat([part, real]), label, "real")
            missing = sorted(set(tracks) - set(part.Track))
            rows.append({"Error": error, "Policy": label, **row(vs_tum, tracks),
                         "VsReal": fmt(vs_real.PosDiff.mean()),
                         "TwoStop": "%.0f%%" % (100 * (part.Stops >= 2).mean()),
                         "Missing": " ".join(missing)})
    for spec in args.diff:
        a, b = spec.split("-", 1)
        for error in policies[a].TyreError.unique():
            pa = policies[a][policies[a].TyreError == error]
            pb = policies[b][policies[b].TyreError == error]
            if pb.empty:
                continue
            diffs = paired(pd.concat([pa, pb]), a, b)
            rows.append({"Error": error, "Policy": "%s - %s" % (a, b), **row(diffs, tracks)})
    table = pd.DataFrame(rows).fillna("")
    pd.set_option("display.width", 250)
    print(table.to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
        print("Wrote %s" % args.out)


if __name__ == "__main__":
    main()
