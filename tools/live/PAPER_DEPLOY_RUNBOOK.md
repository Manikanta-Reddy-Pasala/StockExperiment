# Model 3 Paper Trading — Deployed

**Date deployed:** 2026-05-14 00:13 IST
**Host:** 77.42.45.12 (root)
**Container:** `trading_system_app` (Docker)
**Strategy:** Monthly Momentum Rotation (winner Model 3)

## Config

| Param | Value |
|-------|-------|
| Universe | pseudo-N100 (top 100 by 20-day ADV) |
| Top-N rank | 5 |
| Max concurrent | 1 (full deploy) |
| Lookback | 60 days |
| Capital | ₹10,00,000 |
| Rebalance trigger | First weekday of month (day_of_month ≤ 7, Mon-Fri) |
| Min price | ₹10 |

## State Paths (host-persistent via bind mount)

```
/opt/trading_system/logs/momrot/
├── universes/n100_current.json     # active N100 list (replaced weekly)
├── universes/n100_<date>.json      # snapshots
├── signals/<date>_momrot_n100.json # daily emitted signals
├── ledger/momrot_ledger.json       # paper portfolio state (LIVE)
└── run_logs/<TS>.log               # per-run output
```

Container sees these at `/app/logs/momrot/` (bind from `/opt/trading_system/logs/`).

## Cron Schedule

```
1 4 * * 1-5 /opt/trading_system/momrot_daily.sh             # 09:31 IST Mon-Fri
0 3 * * 0   /opt/trading_system/momrot_rebuild_universe.sh  # 08:30 IST Sun
```

09:31 IST = 04:01 UTC — runs after market open (09:15 IST), price stable.

## Initial Position (forced entry at deploy)

```
Symbol: HFCL
60d return: +113.23%   (rank-1 of N100)
Entry: ₹153.46
Qty: 6,516 shares
Deploy: ₹9,99,945
Cash: ₹55 (idle)
```

## How a Cron Run Works

```
1. momrot_daily.sh runs at 09:31 IST
2. docker exec into trading_system_app
3. python tools/live/momentum_rotation_signal.py --rebalance-only
   - If day_of_month > 7 or weekend: emit [] (no signals)
   - Else: rank N100 by 60d return, compare to held
     - If held NOT in top-5: emit STOP_HIT for held + ENTRY1 for rank-1
     - If held IN top-5 or no held: emit ENTRY1 for rank-1 (if none)
4. If signals non-empty, python tools/live/paper_executor.py
   - Reads ledger, applies risk limits, updates ledger
5. Log + ledger persist on host
```

## Manual Commands

```bash
# Run daily flow manually
ssh root@77.42.45.12 /opt/trading_system/momrot_daily.sh

# View current portfolio
ssh root@77.42.45.12 'cat /opt/trading_system/logs/momrot/ledger/momrot_ledger.json'

# View today's signals
ssh root@77.42.45.12 'ls /opt/trading_system/logs/momrot/signals/ | tail -5'

# Tail latest run log
ssh root@77.42.45.12 'ls -t /opt/trading_system/logs/momrot/run_logs/*.log | head -1 | xargs cat'

# Force rebalance NOW (bypass first-week gate)
ssh root@77.42.45.12 'docker exec -e CAPITAL_INR=1000000 -e MAX_CONCURRENT=1 -e MIN_PRICE=10 \
  trading_system_app bash -c "cd /app && python tools/live/momentum_rotation_signal.py \
    --universe-file /app/logs/momrot/universes/n100_current.json --top-n 5 --force \
    --ledger /app/logs/momrot/ledger/momrot_ledger.json \
    --signals-out /app/logs/momrot/signals/manual_$(date +%FT%H%M)_momrot.json"'

# Rebuild N100 universe manually
ssh root@77.42.45.12 /opt/trading_system/momrot_rebuild_universe.sh

# Disable everything
ssh root@77.42.45.12 'crontab -l | grep -v momrot | crontab -'
```

## Expected Behavior

- **Days 1-7 (Mon-Fri) of month**: rebalance check. If rank-1 changed, swap stock.
- **Days 8-end of month**: hold whatever is in ledger. Daily cron logs "NO_REBALANCE".
- **Weekends**: no run (cron Mon-Fri only).
- **First-of-month**: typical rotation point. ~12 trades/year expected per backtest.

## Risk Notes

- **Paper only.** `fyers_executor` exists but `LIVE_TRADING=true` gate enforced separately.
- **Capital locked at ₹10L.** No add/withdraw without ledger surgery.
- **Single-position concentration** (max=1). One bad pick = full drawdown exposure. Backtest worst DD 10.39% over 3yr — live could exceed.
- **No SL.** Pure ranking-driven exit. If HFCL crashes mid-month, no protection until 1st of next month rotation.
- **Cache freshness**: rank depends on `historical_data` table being current. If Fyers refresh fails, ranking goes stale.

## Backtest Forward Expectation

- Backtest: +86.93%/yr avg (3 yrs)
- Realistic live after slippage/STT/STCG: **35-55% CAGR**
- Worst year backtest: +46% (2025-26 single-trade BSE)
- Worst-case live: +10-15% in regime-crash year (2018-style)

## Monitoring Plan

| When | What | How |
|------|------|-----|
| Daily 09:31 IST | Cron auto-run | Check `run_logs/` |
| Weekly Sun 08:30 IST | Universe refresh | Check `universes/` |
| Monthly 1st-7th | Rotation check | Read ledger after rebalance day |
| Anytime | Drawdown | `(current_value / 1000000 - 1) * 100` |

To compute current portfolio value: query Fyers for held symbol current price × qty + cash.
