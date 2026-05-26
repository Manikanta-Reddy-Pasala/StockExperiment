# momentum_n100_top5_max1 — SUMMARY

**Real NSE Nifty 100 monthly momentum rotation (top-1 by 30d ret). No price filter — honest baseline.**

## When it BUYS (entry rules)

Single position (`max_concurrent=1`). When flat at a rebalance:
1. Universe = real NSE Nifty 100 (`src/data/symbols/nifty100.csv`, ~104 stocks). No price/SMA/ADV filter — honest baseline.
2. Rank every stock by **30-day return** (`lookback_days=30`).
3. Buy **rank-1**.
- Code: `rank_universe()` + `emit_signals()` in `live_signal.py:100-178`.

## When it SELLS (exit rules)

Monthly rotation, single position. **Sells only on rank rotation — there is NO price stop or target:**
- At each **monthly rebalance** (1st–7th weekday): SELLS the held stock **the moment it is no longer rank-1** by 30-day return, and buys the new rank-1 (`retain_top_n=1`). If still rank-1, keeps it.
- At the **mid-month day-15 check**: rotates **only if the new rank-1 leads the held by ≥5pp** of 30-day return (`MID_MONTH_LEAD_PCT=5.0`).
- SELL labelled `TARGET_HIT`/`STOP_HIT` by exit-vs-entry price only — the **trigger is the rank drop, not a price level.**

> **✅ Live == backtest (fixed 2026-05-26).** Two live bugs, now corrected:
> 1. **Stateless / stuck:** live `emit_signals` got `held=[]` (cron passed no ledger), so it never
>    emitted a rotation SELL — the executor (`max_concurrent=1`) then silently skipped the new
>    rank-1 BUY. The model bought once (ADANIGREEN, 2026-05-04) and **could not rotate**.
>    Fixed: `live_signal.py` now reads its open position from the DB (`held_from_db()`).
> 2. **Top-5 band:** exit used `top_picks[:5]`; backtest used top-1. Fixed: `retain_top_n=1`.
>
> With both fixes live now does top-1 rotation + mid-month, matching `backtest.py --retain-top-n 1
> --mid-month-check` (**+125.13%** on real fyers data). **Requires redeploy** to take effect.

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2023-05-15 → 2026-05-12** (~3.00 years) |
| First entry | 2023-05-15 |
| Last exit | 2026-05-04 |
| Total trades | 42 |
| Trades per year | ~14.0 |
| Rebalance | Monthly (1st trading day) + mid-month day-15 check |
| Config | top-1 rotation + mid-month (`--retain-top-n 1 --mid-month-check`) = live |
| Data source | **Fyers (split-adjusted cont_flag=1)** |

## Stock pick logic

1. Universe: src/data/symbols/nifty100.csv (104 NSE Nifty 100 stocks)
2. Rank by 30-day return, pick top-1
3. Rebalance: 1st trading day of month + mid-month day-15 check (lead ≥5pp)
4. Exit: rotation only — sell when not rank-1 (top-1 retention)

## Headline result (live config: top-1 + mid-month, real fyers data)

| Metric | Value |
|---|---:|
| Final NAV (cap + open MTM) | **Rs.11,341,351** |
| Total return | **+1034.14%** |
| 2.99-yr CAGR | **+125.13%** |
| Max DD | **28.21%** |
| Calmar (CAGR / Max DD) | **4.44** |
| Trades closed | 42 |
| Wins / Losses | 28 / 14 |
| Win rate | 66.7% |
| Live deployment | YES (top-1 fix pending redeploy) |
| Open position | **ADANIGREEN** qty 8,670 entry Rs.1,290.70 (2026-05-04) last Rs.1,308.00 unrealized +149,991 |

> Prior published figure was **+65.10%** — that was the monthly-only (no mid-month) top-1 run.
> Live runs the mid-month check too, so the live-faithful number is **+125.13%**.

## NSE cap segment breakdown

| Cap | Trades | Wins | Losses | WR |
|---|---:|---:|---:|---:|
| **Large** | 42 | 28 | 14 | 66.7% |

## Top 5 winners

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| ADANIPOWER   | 2026-04-01 → 2026-05-04 | 157.11 | +44.68% | +3,455,875 |
| SOLARINDS    | 2025-05-15 → 2025-07-01 | 13,880.00 | +23.89% | +1,376,140 |
| MAZDOCK      | 2024-06-18 → 2024-08-16 | 2,089.13 | +19.07% | +916,205 |
| SOLARINDS    | 2025-04-01 → 2025-05-02 | 11,131.60 | +17.22% | +782,299 |
| SHRIRAMFIN   | 2026-02-01 → 2026-02-16 | 997.60 | +8.80% | +688,879 |

## Top 5 losses

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| CGPOWER      | 2024-10-15 → 2024-11-01 | 832.70 | -13.46% | -718,561 |
| ENRIN        | 2026-03-02 → 2026-03-16 | 2,972.70 | -6.22% | -521,885 |
| HINDZINC     | 2024-11-01 → 2024-11-18 | 558.25 | -11.09% | -512,161 |
| ETERNAL      | 2024-08-16 → 2024-09-02 | 264.43 | -7.56% | -432,287 |
| VEDL         | 2026-01-16 → 2026-02-01 | 255.69 | -4.04% | -329,218 |

Full trade-by-trade ledger: see [TRADE_LEDGER.md](TRADE_LEDGER.md).
