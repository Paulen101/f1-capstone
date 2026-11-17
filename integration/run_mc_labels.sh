#!/usr/bin/env bash
# Monte Carlo pit-lap labels on our fitted 2019 files: every one-stop top finisher we label.
# Sampled safety cars and retirements, 500 runs per candidate lap. See notes/label_generator.md.
#
#   integration/run_mc_labels.sh [runs] [jobs]
#
# Output: integration/output/labels/<Track>_2019_fitted_<driver>_mc<runs>/ and a log per driver.
set -euo pipefail
cd "$(dirname "$0")/.."
RUNS="${1:-500}"
JOBS="${2:-28}"
PY=.venv-tum/bin/python
[ -x "$PY" ] || PY=.venv-tum/Scripts/python.exe

# track driver compound-of-the-real-second-stint
LABELS="
Spielberg VER A3
Spielberg HAM A3
Spielberg LEC A3
Spielberg BOT A3
Monza LEC A3
Monza BOT A4
Monza RIC A4
Singapore VET A4
Singapore LEC A4
Singapore VER A4
Singapore HAM A4
Austin HAM A3
Austin RIC A3
Austin SAI A3
"
mkdir -p integration/output/labels
echo "$LABELS" | while read -r track driver compound; do
  [ -n "$track" ] || continue
  log="integration/output/labels/${track}_2019_fitted_${driver}_mc${RUNS}.log"
  if [ -f "integration/output/labels/${track}_2019_fitted_${driver}_mc${RUNS}/manifest.json" ]; then
    echo "$track $driver: already done"
    continue
  fi
  # stdin from /dev/null: the label list is this loop's stdin, and Python must not read it
  "$PY" integration/label_generator.py --race "integration/parameters/pars_${track}_2019_fitted.ini" \
    --driver "$driver" --compound "$compound" --runs "$RUNS" --jobs "$JOBS" > "$log" 2>&1 < /dev/null \
    || { echo "$track $driver: FAILED, see $log"; continue; }
  echo "$track $driver: $(tail -1 "$log" | cut -c1-120)"
done
