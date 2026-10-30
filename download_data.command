#!/bin/zsh
# Double-click to download the missing FastF1 races and rebuild the workbooks.
# Output is also saved to notes/evidence/data_download.log.
cd "$(dirname "$0")"
source .venv-data/bin/activate
LOG=notes/evidence/data_download.log
mkdir -p notes/evidence

# F1TV login is needed for 2023+ races. Log in only if the saved token is missing or expired.
python - <<'PY'
import sys, time, jwt
from fastf1.internals.f1auth import AUTH_DATA_FILE
if not AUTH_DATA_FILE.exists():
    sys.exit(1)
try:
    t = AUTH_DATA_FILE.read_text().strip()
    exp = jwt.decode(t, options={"verify_signature": False}).get("exp")
except Exception:
    sys.exit(0)  # unknown token format: let f1_to_excel.py report any problem
sys.exit(0 if exp and exp > time.time() + 600 else 1)
PY
if [ $? -ne 0 ]; then
  echo "F1TV login needed: a browser window will open. Log in there, then come back here."
  python -m fastf1 auth f1tv --authenticate
fi

{
  echo "=== $(date) ==="
  python f1_to_excel.py --years 2026 --cache ../fastf1_cache
  python f1_to_excel.py --years 2022
} 2>&1 | tee -a "$LOG"

echo ""
echo "Done. Tell Claude it finished."
read -k1 "?Press any key to close this window."
