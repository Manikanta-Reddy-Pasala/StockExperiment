#!/bin/bash
# Quarterly NSE free-float market-cap refresh.
# MUST run on a residential-IP machine with a full browser (NSE 403s the VM +
# plain scripts; only headless full-Chromium from a residential IP works).
# Scheduled via ~/Library/LaunchAgents/com.stockexp.mcaprefresh.plist (local Mac).
# Steps: rebuild candidate list (FULL NSE equity universe) -> scrape FF-mcap -> push CSV to VM.
set -uo pipefail
cd /Users/manip/Documents/codeRepo/StockExperiment || exit 1
LOG="/tmp/mcap_refresh_$(date +%Y%m%d).log"
echo "=== mcap refresh $(date) ===" >> "$LOG"

# 1. rebuild candidate list = FULL NSE equity universe (all -EQ with recent data).
#    No ADV cap: a stock climbing toward N500 can sit below top-800 by liquidity.
python3 - >> "$LOG" 2>&1 <<'PY'
import sys; sys.path.insert(0,'.')
from tools.shared.ohlcv_cache import _get_engine
from sqlalchemy import text
import json
e=_get_engine()
q='''SELECT DISTINCT symbol FROM historical_data
     WHERE data_source='fyers' AND symbol LIKE 'NSE:%-EQ'
       AND date >= (SELECT MAX(date)-120 FROM historical_data)'''
with e.connect() as c:
    rows=[r[0] for r in c.execute(text(q))]
syms=sorted(s.replace('NSE:','').replace('-EQ','') for s in rows)
open('/tmp/mcap_candidates.json','w').write(json.dumps(syms))
print('full equity candidates:',len(syms))
PY

# 2. fresh scrape (rm old so shares re-pull; scraper is otherwise resumable)
[ -f exports/nse_mcap.csv ] && mv exports/nse_mcap.csv "exports/nse_mcap_$(date +%Y%m%d).csv"
python3 tools/analysis/nse_mcap_scraper.py >> "$LOG" 2>&1

# 2.5. persist this snapshot to Postgres for a permanent historical track:
#       market_cap_history (every run — plist fires the 1st of every month) +
#       nifty_index_membership (gated to Apr & Sep, the NSE semi-annual reviews).
python3 tools/analysis/mcap_db.py load-mcap >> "$LOG" 2>&1
MONTH=$(date +%m)
if [ "$MONTH" = "04" ] || [ "$MONTH" = "09" ]; then
  python3 tools/analysis/mcap_db.py snapshot-membership >> "$LOG" 2>&1
fi

# 3. push fresh CSV to the VM (exports/ is rsync-excluded on deploy, so push direct)
rsync -z exports/nse_mcap.csv root@77.42.45.12:/opt/trading_system/exports/nse_mcap.csv >> "$LOG" 2>&1 \
  && ssh root@77.42.45.12 'docker cp /opt/trading_system/exports/nse_mcap.csv trading_system_app:/app/exports/nse_mcap.csv' >> "$LOG" 2>&1

echo "=== done $(date), rows=$(wc -l < exports/nse_mcap.csv) ===" >> "$LOG"
