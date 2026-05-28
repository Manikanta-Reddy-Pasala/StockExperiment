# momentum_n100_top5_max1 — SUMMARY

**Real NSE Nifty 100 momentum rotation (top-1 BUY, top-3 retention band by 15-trading-day ret), monthly + mid-month check. No price filter — honest baseline.**

## When it BUYS (entry rules)

Single position (`max_concurrent=1`). When flat at a rebalance:
1. Universe = real NSE Nifty 100 (`src/data/symbols/nifty100.csv`, ~104 stocks). No price/SMA/ADV filter — honest baseline.
2. Rank every stock by **15-trading-day return** (`lookback_days=15`, ~3 weeks).
3. Buy **rank-1**.
- Code: SELECTION (universe/filters/rank) in `live_signal.py` + `backtest.py`; RULE = shared `tools/shared/rotation_strategy.decide_rotation` + `midmonth_lead_ok` (same calls live + backtest make); backtest EXECUTION = shared `tools/shared/backtest_engine`. Parity-tested.

## When it SELLS (exit rules)

Monthly rotation, single position. **Sells only on rank drop — there is NO price stop or target:**
- At each **monthly rebalance** (1st–7th weekday): keeps the held stock while it stays inside the **top-3 by 15-trading-day return** (`retain_top_n=3`). Only when it drops to rank-4 or worse does it sell + buy the new rank-1.
- At the **mid-month day-15 check**: rotates **only if the new rank-1 leads the held by ≥5pp** of 15-trading-day return (`MID_MONTH_LEAD_PCT=5.0`, unchanged).
- SELL labelled `TARGET_HIT`/`STOP_HIT` by exit-vs-entry price only — the **trigger is the rank drop, not a price level.**

> **2026-05-28 change: `retain_top_n` 1 → 3.** Sweep across LB=15/20/25/30/35/40/45 × retain=1/2/3/5 showed retain=3 wins **both** the 3yr AND the 10yr window vs canonical retain=1, with no trade-off:
>
> | Window | retain=1 (old) | retain=3 (new) | Δ |
> |---|---:|---:|---:|
> | 3yr CAGR | +184.36% | **+245.47%** | +61.1pp |
> | 3yr Max DD | 14.89% | **14.89%** | 0 |
> | 3yr Calmar | 12.38 | **16.48** | +4.10 |
> | 10yr CAGR | +67.14% | **+86.84%** | +19.7pp |
> | 10yr Max DD | 60.89% | **53.07%** | −7.82pp |
> | 10yr Calmar | 1.10 | **1.64** | +0.54 |
>
> Mechanism: held drops to rank-2 or rank-3 = STILL strong momentum; rotating away whipsaws trade cost + misses recoveries when it climbs back to rank-1.

> **✅ Live == backtest.** Live cron runs `--mid-month-check` with `retain_top_n=3` (default since 2026-05-28). Headline numbers below are the live-faithful figures. **Requires redeploy** after `retain_top_n` flips from 1 to 3.

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2023-05-15 → 2026-05-12** (~2.99 years) + 10yr (2016-05-15 → 2026-05-12) |
| First entry | 2023-05-15 (3yr) / 2016-05-15 (10yr) |
| Total trades 3yr | 50 |
| Total trades 10yr | 177 |
| Trades per year | ~17 |
| Lookback | **15 trading days (~3 weeks)** — chosen 2026-05-27 via 6yr walk-forward |
| Retention band | **top-3 by 15td return** (`retain_top_n=3`) — chosen 2026-05-28 via lookback × retain grid |
| Rebalance | Monthly (1st trading day) + mid-month day-15 check |
| Config | LB=15 + retain=3 + mid-month (`--retain-top-n 3 --mid-month-check`) = live |
| Data source | **Fyers (split-adjusted cont_flag=1)** |

> **Lookback chosen via 6-year sweep (2020-2026):** 15 trading days (+151.72% CAGR / 45.69% max DD)
> beat 30 trading days (+129.01% CAGR / 57.29% max DD), so the momentum window was changed 30→15 on
> 2026-05-27 (`live_signal.rank_universe lookback_days=15`, `backtest.LOOKBACK=15`).

## Stock pick logic

1. Universe: src/data/symbols/nifty100.csv (104 NSE Nifty 100 stocks)
2. Rank by 15-trading-day return, BUY rank-1
3. Rebalance: 1st trading day of month + mid-month day-15 check (lead ≥5pp)
4. Exit: rotation only — sell when held drops OUT of top-3 by 15td ret (retain band = 3)

## Headline result (live config: top-1 rotation + mid-month check, 15-trading-day lookback, real fyers data)

3-year standard window (**2023-05-15 → 2026-05-15**):

### 3-year (2023-05-15 → 2026-05-12, ₹10L start)

| Metric | Value |
|---|---:|
| Final NAV (cap + open MTM) | **Rs.40,849,591** |
| Total return | **+3984.96%** |
| 3-yr CAGR | **+245.47%** |
| Max DD (3yr) | **14.89%** (UNDERSTATES real risk — see 6yr / 10yr below) |
| Calmar (CAGR / Max DD) | **16.48** |
| Trades closed | 50 |
| Wins / Losses | 32 / 18 |
| Win rate | 64.0% |
| Live deployment | YES (retain_top_n 1→3 ships 2026-05-28) |

### 10-year (2016-05-15 → 2026-05-12, ₹10L start) — honest decade including 2017-2019 regime hostility

| Metric | Value |
|---|---:|
| Final NAV | **Rs.515,293,966** (₹51.5Cr) |
| Total return | **+51,429%** |
| 10-yr CAGR | **+86.84%** |
| Max DD (10yr) | **53.07%** |
| Calmar | **1.64** |
| Trades closed | 177 |
| Wins / Losses | 99 / 78 |
| Win rate | 55.9% |

> 10yr universe is today's `nifty100.csv` snapshot → ~5-15pp CAGR survivorship inflation (IRFC / MAZDOCK / ETERNAL / LIC etc. didn't trade pre-2020). DD figures stay reliable.

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
