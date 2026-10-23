#!/usr/bin/env bash
# Pit-lap sweep: run the one-stop fixed-lap policy for every possible in-lap and collect
# each driver's finishing position and race time. Run from Main/ with .venv-tum.
#
#   bash integration/run_sweep.sh                 # VER, LEC, HAM onto A3, laps 2-70
#   DRIVERS="VER" COMPOUND=A3 bash integration/run_sweep.sh
#
# Output: integration/output/sweep/<DRIVER>_<LAP>/ (one TUM run each) and
#         integration/output/sweep/pit_lap_sweep.csv (one row per run).
# Every run goes through run_experiment.py, which asserts that the simulator executed
# exactly the requested stop (pitinlap on N, pitoutlap on N+1).
set -euo pipefail
PY=${PY:-.venv-tum/bin/python}
DRIVERS=${DRIVERS:-"VER LEC HAM"}
COMPOUND=${COMPOUND:-A3}
FIRST=${FIRST:-2}
LAST=${LAST:-70}
OUT=integration/output/sweep
mkdir -p "$OUT"
for D in $DRIVERS; do
  for L in $(seq "$FIRST" "$LAST"); do
    "$PY" integration/run_experiment.py fixed --driver "$D" --pit-lap "$L" --compound "$COMPOUND" \
      --out "$OUT/${D}_${L}" > /dev/null
  done
done
"$PY" - "$OUT" "$DRIVERS" "$FIRST" "$LAST" "$COMPOUND" <<'EOF'
import csv, sys
out, drivers, first, last, compound = sys.argv[1], sys.argv[2].split(), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
def row(path, drv):
    with open(path) as f:
        return next(r for r in csv.DictReader(f) if r["Driver"] == drv)
rows = []
for d in drivers:
    real = row("integration/output/realstrategy/classification.csv", d)
    for lap in range(first, last + 1):
        r = row("%s/%s_%d/classification.csv" % (out, d, lap), d)
        rows.append({"Driver": d, "PitLap": lap, "Compound": compound, "Position": int(r["Position"]),
                     "SimulatedLaps": float(r["SimulatedLaps"]), "RaceTime_s": round(float(r["RaceTime_s"]), 3),
                     "RealStrategyTime_s": round(float(real["RaceTime_s"]), 3),
                     "RealStrategyPos": int(real["Position"]),
                     "DeltaVsReal_s": round(float(r["RaceTime_s"]) - float(real["RaceTime_s"]), 3)})
with open("%s/pit_lap_sweep.csv" % out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote %s/pit_lap_sweep.csv (%d runs)" % (out, len(rows)))
EOF
