# StockExperiment — Momentum Rotation

NSE momentum-based stock rotation system on Nifty-100 (pseudo, ADV-ranked).
Picks rank-1 stock by 60-day return, holds for the month, rebalances on
the 1st weekday of every month.

Production: `77.42.45.12` · App: <https://stock.oneshell.in>

---

## Strategy — Momentum Rotation

```
1. Universe = pseudo-Nifty-100 (top 100 NSE stocks by 20-day ADV)
2. Every rebalance day (1st weekday of month):
   - Rank all 100 stocks by 60-day price return descending
   - Identify top-5
3. Hold rank-1 only (max-concurrent = 1, full capital deploy)
4. Rotation rule:
   - If held stock STILL in top-5 → hold
   - If held stock dropped out → SELL at close, BUY new rank-1 at close
5. No stop loss. No profit target. Pure ranking-driven exits.
6. Capital ceiling: ₹10,00,000 reference.
```

### Backtest Results (May 2023 → May 2026, 3 years)

Walk-forward validated monthly max=1 (see `exports/backtests/MODEL3_TRADE_LEDGER.md`):

| Year    | ROI    | MaxDD |
|---------|-------:|------:|
| 2023-24 | +80.87% | 10.39% |
| 2024-25 | +133.78% | 8.04% |
| 2025-26 | +46.14% | 0.00% |
| **Avg** | **+86.93%** | **6.14%** |

Compound: ₹10L → ₹61.80L (+518% over 3 yrs).

**Realistic live expectation:** 35-55% CAGR after slippage / STT / STCG.

---

## Components

### Live Production

| Path | Purpose |
|------|---------|
| `tools/live/momentum_rotation_signal.py` | Daily ranker, emits BUY/SELL signals JSON |
| `tools/live/paper_executor.py` | Executes signals against ledger |
| `tools/live/daily_summary.py` | Sends short Telegram digest each day |
| `tools/live/telegram_notify.py` | Bot API helper |
| `tools/live/fyers_executor.py` | Real-order placement (gated by `LIVE_TRADING=true`) |
| `/opt/trading_system/momrot_daily.sh` | Host cron wrapper (Mon-Fri 04:01 UTC) |
| `/opt/trading_system/momrot_rebuild_universe.sh` | Weekly N100 rebuild (Sun 03:00 UTC) |

### Backtest

| Path | Purpose |
|------|---------|
| `tools/backtests/momentum_rotation_backtest.py` | Monthly engine — used for `optimize_p18/p19` results |
| `tools/backtests/momrot_freq_backtest.py` | Daily/weekly/monthly variant |
| `tools/backtests/tiered_momentum_rotation.py` | 3-tier cap-stratified variant (lost in test) |
| `tools/backtests/build_universe_by_adv.py` | Builds pseudo-N100 by ADV ranking |
| `tools/backtests/realistic_capital_sim.py` | Capital simulator with max_concurrent + compounding |
| `tools/backtests/ohlcv_cache.py` | Postgres-backed OHLCV access layer |

### UI

| Path | URL |
|------|-----|
| `src/web/momrot_routes.py` | `/admin/momrot/*` backend endpoints |
| `src/web/templates/admin/momrot_dashboard.html` | Live portfolio + ranking + Fyers card |
| `src/web/templates/v2/dashboard.html` | Summary page |
| `src/web/templates/v2/portfolio.html` | Holdings (from Fyers) |
| `src/web/templates/v2/picks.html` | Top-20 momentum ranking |
| `src/web/templates/v2/history.html` | Closed-trade ledger |
| `src/web/templates/v2/settings.html` | Capital + rotation rules + Fyers broker |

Portfolio + Dashboard now read from **live Fyers account** (funds + holdings)
via `FyersService`. Paper ledger at `/app/logs/momrot/ledger/` is retained
for simulation tracking but no longer drives portfolio display.

---

## Daily Telegram Digest

Bot: `@stocks_momrot_bot`. Sends short summary every weekday 09:31 IST:

```
*Momrot 2026-05-14* ⏸️ HOLD
NAV ₹10,00,000  (+0.00%)
Day P&L ₹+0
Hold HFCL (+0.0%)
Top1 ✓ HFCL (+113.2%)
Next rebalance: 2026-06-01
```

Status icons: `⏸️ HOLD` · `🔄 REBALANCED` · `❌ FAILED` · `⚠️ NO_DATA`

---

## Documentation

| File | Content |
|------|---------|
| `exports/backtests/WINNERS_FOUND.md` | 3 winning models from 19-phase journey |
| `exports/backtests/MODEL3_TRADE_LEDGER.md` | Full trade-by-trade ledger for winner |
| `exports/backtests/HOW_WE_GOT_HERE.md` | Methodology + 19-phase research journey |
| `exports/backtests/REBALANCE_FREQ_COMPARISON.md` | Daily/weekly/monthly × max-1/2 comparison |
| `exports/backtests/TIERED_VS_MODEL3.md` | Tier-stratified variant test (failed) |
| `tools/live/PAPER_DEPLOY_RUNBOOK.md` | Deployment runbook |

---

## Quick Commands

```bash
# View live ledger
ssh root@77.42.45.12 'cat /opt/trading_system/logs/momrot/ledger/momrot_ledger.json'

# Latest cron run log
ssh root@77.42.45.12 'ls -t /opt/trading_system/logs/momrot/run_logs/*.log | head -1 | xargs cat'

# Force rebalance via UI
curl -X POST https://stock.oneshell.in/admin/momrot/run-now

# Disable daily cron
ssh root@77.42.45.12 'crontab -l | grep -v momrot | crontab -'
```

---

## Realistic Caveats

- 3-year backtest is a small sample.
- Indian momentum strategies have documented 30-40% CAGR over decades but with
  20-30% worst-year DD in 2018-style mom-crashes.
- Live forward could see 10-15% returns in a regime shift year.
- All three winners are momentum-correlated and will suffer together in a
  mom-crash year.
