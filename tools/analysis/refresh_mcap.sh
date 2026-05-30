#!/bin/bash
# Monthly NSE index-membership refresh.
# REPLACES the old NSE get-quotes scraper (nse_mcap_scraper.py, deleted):
# niftyindices.com is NOT WAF-blocked, so this works from ANY IP (laptop OR the
# VM) over plain HTTPS — no headless Chromium, no residential-IP dependency.
# Scheduled via ~/Library/LaunchAgents/com.stockexp.mcaprefresh.plist (monthly).
# Steps: download current index constituents -> mark nifty_index_membership ->
#        snapshot membership (Apr/Sep) -> push constituent CSVs to the VM.
set -uo pipefail
cd /Users/manip/Documents/codeRepo/StockExperiment || exit 1
LOG="/tmp/mcap_refresh_$(date +%Y%m%d).log"
echo "=== index refresh $(date) ===" >> "$LOG"

# 1. download + mark membership (review_date = today)
python3 tools/analysis/download_niftyindices.py --load-db >> "$LOG" 2>&1

# 2. on the NSE review months (Apr/Sep) also snapshot the PIT membership tables
MONTH=$(date +%m)
if [ "$MONTH" = "04" ] || [ "$MONTH" = "09" ]; then
  python3 tools/analysis/mcap_db.py snapshot-membership >> "$LOG" 2>&1
fi

# 3. push the fresh constituent CSVs to the VM (exports/ is rsync-excluded on deploy)
rsync -rz exports/index_constituents/current/ \
  root@77.42.45.12:/opt/trading_system/exports/index_constituents/current/ >> "$LOG" 2>&1

echo "=== done $(date) ===" >> "$LOG"
