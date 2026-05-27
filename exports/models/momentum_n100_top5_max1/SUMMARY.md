# momentum_n100_top5_max1 — SUMMARY

**Real NSE Nifty 100 momentum rotation (top-1 by 15-trading-day ret), monthly + mid-month check. No price filter — honest baseline.**

## When it BUYS (entry rules)

Single position (`max_concurrent=1`). When flat at a rebalance:
1. Universe = real NSE Nifty 100 (`src/data/symbols/nifty100.csv`, ~104 stocks). No price/SMA/ADV filter — honest baseline.
2. Rank every stock by **15-trading-day return** (`lookback_days=15`, ~3 weeks).
3. Buy **rank-1**.
- Code: SELECTION (universe/filters/rank) in `live_signal.py` + `backtest.py`; RULE = shared `tools/shared/rotation_strategy.decide_rotation` + `midmonth_lead_ok` (same calls live + backtest make); backtest EXECUTION = shared `tools/shared/backtest_engine`. Parity-tested.

## When it SELLS (exit rules)

Monthly rotation, single position. **Sells only on rank rotation — there is NO price stop or target:**
- At each **monthly rebalance** (1st–7th weekday): SELLS the held stock **the moment it is no longer rank-1** by 15-trading-day return, and buys the new rank-1 (`retain_top_n=1`). If still rank-1, keeps it.
- At the **mid-month day-15 check**: rotates **only if the new rank-1 leads the held by ≥5pp** of 15-trading-day return (`MID_MONTH_LEAD_PCT=5.0`, unchanged).
- SELL labelled `TARGET_HIT`/`STOP_HIT` by exit-vs-entry price only — the **trigger is the rank drop, not a price level.**

> **✅ Live == backtest (fixed 2026-05-26).** Two live bugs, now corrected:
> 1. **Stateless / stuck:** live `emit_signals` got `held=[]` (cron passed no ledger), so it never
>    emitted a rotation SELL — the executor (`max_concurrent=1`) then silently skipped the new
>    rank-1 BUY. The model bought once (ADANIGREEN, 2026-05-04) and **could not rotate**.
>    Fixed: `live_signal.py` now reads its open position from the DB (`held_from_db()`).
> 2. **Top-5 band:** exit used `top_picks[:5]`; backtest used top-1. Fixed: `retain_top_n=1`.
>
> With both fixes live now does top-1 rotation, matching `backtest.py --retain-top-n 1 --mid-month-check`
> — the live cron runs both the monthly rebalance and the mid-month check, so the mid-month backtest
> is the live-faithful figure. **Requires redeploy** to take effect.
> *(The previously-quoted **+125.13%** was the LB=30 figure; the momentum lookback is now **15 trading
> days** — see the headline table below for the LB=15 result.)*

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2023-05-15 → 2026-05-15** (~3.00 years) |
| First entry | 2023-05-15 |
| Total trades | 57 |
| Trades per year | ~19 |
| Lookback | **15 trading days (~3 weeks)** — changed 2026-05-27 from 30 (see note) |
| Rebalance | Monthly (1st trading day) + mid-month day-15 check |
| Config | top-1 rotation + mid-month check (`--retain-top-n 1 --mid-month-check`) = live |
| Data source | **Fyers (split-adjusted cont_flag=1)** |

> **Lookback chosen via 6-year sweep (2020-2026):** 15 trading days (+151.72% CAGR / 45.69% max DD)
> beat 30 trading days (+129.01% CAGR / 57.29% max DD), so the momentum window was changed 30→15 on
> 2026-05-27 (`live_signal.rank_universe lookback_days=15`, `backtest.LOOKBACK=15`).

## Stock pick logic

1. Universe: src/data/symbols/nifty100.csv (104 NSE Nifty 100 stocks)
2. Rank by 15-trading-day return, pick top-1
3. Rebalance: 1st trading day of month + mid-month day-15 check (lead ≥5pp)
4. Exit: rotation only — sell when not rank-1 (top-1 retention)

## Headline result (live config: top-1 rotation + mid-month check, 15-trading-day lookback, real fyers data)

3-year standard window (**2023-05-15 → 2026-05-15**):

| Metric | Value |
|---|---:|
| Final NAV (cap + open MTM) | **Rs.22,814,268** |
| Total return | **+2181.43%** |
| 3-yr CAGR | **+184.36%** |
| Max DD (3yr) | **14.89%** (UNDERSTATES real risk — see note) |
| Calmar (CAGR / Max DD) | **12.38** |
| Trades closed | 57 |
| Wins / Losses | 37 / 20 |
| Win rate | 64.9% |
| Live deployment | YES (top-1 fix pending redeploy) |
| Open position | **ADANIPOWER** qty 108,831 entry Rs.227.30 (2026-05-04) last Rs.209.63 unrealized -1,923,044 |

> ## ⚠️ Drawdown — honest 6-year risk
>
> **The 3yr Max DD of 14.89% UNDERSTATES real risk** — the 2023-2026 window never saw a major
> correction. The honest 6-year backtest (**2020-05-15 → 2026-05-15**) shows the true max drawdown is
> **~45.69%**: the 2022 correction cut this model roughly **-40% in a year**. Over that full 6-year
> window the model returned **+151.72% CAGR with 45.69% max DD (Calmar 3.32)**.
>
> **Size positions for ~46% max drawdown, not the optimistic ~15%.** This applies to BOTH the new
> LB=15 and the old LB=30 configs — it was simply hidden by the short 3-year window.

## NSE cap segment breakdown

| Cap | Trades | Wins | Losses | WR |
|---|---:|---:|---:|---:|
| **Large** | 57 | 37 | 20 | 64.9% |

Regenerated 2026-05-27 from the LB=15 `--mid-month-check` 57-trade ledger (`trade_ledger.json`).

## Top 5 winners

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| MAZDOCK      | 2024-06-03 → 2024-08-01 | 1,628.80 | +56.79% | +4,891,506 |
| ADANIGREEN   | 2026-04-15 → 2026-05-04 | 1,096.05 | +17.76% | +3,730,467 |
| ADANIPOWER   | 2026-04-01 → 2026-04-15 | 157.11 | +16.75% | +3,014,219 |
| IRFC         | 2024-01-01 → 2024-02-15 | 100.40 | +58.27% | +2,381,944 |
| MAZDOCK      | 2025-05-02 → 2025-06-02 | 2,996.60 | +12.94% | +1,761,454 |

## Top 5 losses

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| ADANIENSOL   | 2024-08-01 → 2024-08-16 | 1,275.20 | -14.89% | -2,011,041 |
| HINDZINC     | 2024-11-01 → 2024-11-18 | 558.25 | -11.09% | -1,449,017 |
| ENRIN        | 2026-03-02 → 2026-03-16 | 2,972.70 | -6.22% | -1,222,850 |
| CGPOWER      | 2025-09-15 → 2025-10-01 | 791.35 | -6.51% | -1,172,505 |
| HINDZINC     | 2025-10-15 → 2025-11-03 | 513.20 | -6.34% | -1,102,241 |

Full trade-by-trade ledger: see [TRADE_LEDGER.md](TRADE_LEDGER.md).
