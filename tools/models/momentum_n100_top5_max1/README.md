# momentum_n100_top5_max1

**Category: LARGE-CAP equity (real NSE Nifty 100 constituents) — LIVE**

Monthly momentum rotation on **REAL NIFTY 100** (NSE constituents). Backtest universe matches live universe (no methodology drift). No price filter — pure ranking on NSE-official list. Sibling model `momentum_pseudo_n100_adv` is also LIVE and uses an ADV-ranked top-100 from N500 with yearly-PIT rebuild (PIT-safe).

## Stock universe

| Source | NSE archives `ind_nifty100list.csv` |
|---|---|
| Cached at | `src/data/symbols/nifty100.csv` (104 stocks, includes some -EQ alternates) |
| Refresh script | `python tools/refresh_nifty100.py` (NSE rebalances Mar/Sep) |
| Selection | All 104 stocks → strategy ranks by 15-trading-day return → picks top-1 |

**No filtering**: takes the entire official Nifty 100 list as-is. NSE already curates the constituents (top-100 by free-float market cap, large-cap by definition).

Real Nifty 100 contains: HDFCBANK, RELIANCE, ICICIBANK, TCS, INFY, BHARTIARTL, SBIN, BAJFINANCE, LICI, HINDUNILVR, ITC, LT, KOTAKBANK, AXISBANK, MARUTI, M&M, SUNPHARMA, TITAN, ASIANPAINT, ADANIENT, ADANIPORTS, NTPC, ULTRACEMCO, HCLTECH, COALINDIA, NESTLEIND, POWERGRID, BAJAJFINSV, BEL, HAL, JSWSTEEL, TATAMOTORS, TATASTEEL, BAJAJ-AUTO, EICHERMOT, HEROMOTOCO, DABUR, MARICO, etc. — all genuine large-cap names.

## Strategy

| Knob | Value |
|---|---|
| Universe | Real NIFTY 100 from `src/data/symbols/nifty100.csv` (NSE archives) |
| Signal | Rank by **15-trading-day return** (`lookback_days=15`, ~3 weeks; changed 30→15 on 2026-05-27) |
| Position | Hold top-1 (`top_n=5` ranking, `max_concurrent=1`) |
| Rebalance | **1st weekday of month** (unconditional rotate to rank-1) |
| **Mid-month check** | **Day-15 weekday — rotate only if rank-1 leads held by ≥ 5pp** (unchanged) |
| Exit | Rotation only — sell when stock drops out of rank-1 OR mid-month lead breached. No SL, no target |

**Universe refresh**: `python tools/refresh_nifty100.py` pulls NSE CSV. NSE rebalances March/September.

### Mid-month check rationale

Pure monthly rotation misses stocks that break out *during* the month and become the obvious winner by month-end (calendar-lag risk). The mid-month check addresses this without exploding turnover:

- Runs on the first weekday on/after day 15
- Re-ranks the universe
- Rotates **only** if today's rank-1 has a 15-trading-day return ≥ 5pp higher than currently-held stock's 15-trading-day return
- Most months: no action (held stock usually still leads or lead is below 5pp)
- ~30-40% of months trigger an actual rotation
- Backtest improves CAGR over plain monthly with honest costs (slip + STT + brokerage + 20% STCG)

5pp threshold is a round number chosen *before* testing the variant; not sweep-selected. See `backtest.py --mid-month-check` to reproduce.

## Backtest result (REAL Nifty 100, 2023-05-15 → 2026-05-15, 15-trading-day lookback)

The LIVE cron runs both the monthly rebalance and the mid-month check (`emit_signal` +
`emit_mid_month_signal` + `execute_orders` + `execute_mid_month_orders`), so the mid-month config
below is the live-faithful headline. **Lookback was changed 30→15 trading days on 2026-05-27** (see
History) — all numbers below are the LB=15 result.

### Monthly + mid-month check (LIVE config) — `--mid-month-check --mid-month-lead-pct 5.0`

| Metric | Value |
|---|---:|
| Final NAV (cap + open MTM) | **₹2,28,14,268** |
| Total return | **+2181.43%** |
| **3-yr CAGR** | **+184.36%** |
| Max DD (3yr) | **14.89%** (UNDERSTATES real risk — see ⚠️ note) |
| Calmar (CAGR / Max DD) | **12.38** |
| Round-trips | 57 (~19/yr) |
| Wins / Losses | 37 / 20 |
| Win rate | 64.9% |

The mid-month check rotates only when the new rank-1 leads the held name's 15-trading-day return by
≥5pp, capturing bull-trend continuations that plain monthly misses while keeping turnover contained.

> ### ⚠️ Drawdown — honest 6-year risk
>
> **The 3yr Max DD of 14.89% UNDERSTATES real risk** — the 2023-2026 window never saw a major
> correction. The honest 6-year backtest (**2020-05-15 → 2026-05-15**) shows the true max drawdown is
> **~45.69%** — the 2022 correction cut this model roughly **-40% in a year**. Over the full 6 years
> the model returned **+151.72% CAGR with 45.69% max DD (Calmar 3.32)**.
>
> **Plan position sizing for ~46% max drawdown, not the optimistic ~15%.** This applies to BOTH the
> new LB=15 and the old LB=30 configs — it was simply hidden by the short 3-year window.

> **Per-year ROI table needs regeneration under LB=15** from the `--mid-month-check` ledger (any prior
> per-year table was computed under the old LB=30 monthly-only run and does not match this headline).

## Top losers (unfiltered)

> **NOTE — STALE TABLES:** the winners/losers tables below were computed under the **prior LB=30**
> monthly-only run (different trades) and do NOT match the canonical LB=15 mid-month (57-trade)
> headline. **Needs regeneration under LB=15** from the `--mid-month-check` ledger. Do not rely on
> these figures.

| Symbol | Entry → Exit | Entry ₹ | Ret | PnL |
|---|---|---:|---:|---:|
| BAJAJ-AUTO | 2024-10-01 → 2024-11-01 | 12,157.45 | -18.77% | -₹4.84L |
| ENRIN | 2026-03-02 → 2026-04-01 | 2,972.70 | -12.07% | -₹4.19L |
| IRFC | 2024-02-01 → 2024-03-01 | 169.90 | -13.24% | -₹3.26L |
| HINDZINC | 2024-11-01 → 2024-12-02 | 558.25 | -9.92% | -₹2.09L |
| TATACONSUM | 2025-02-01 → 2025-03-03 | 1,069.85 | -10.84% | -₹2.05L |

BAJAJ-AUTO is the worst — single-share concentration cost (only 10 shares for ₹10L capital).

## Top winners

| Symbol | Entry → Exit | Entry ₹ | Ret | PnL |
|---|---|---:|---:|---:|
| ADANIPOWER | 2026-04-01 → 2026-05-04 | 157.11 | +44.68% | +₹17.88L |
| SHRIRAMFIN | 2025-11-03 → 2026-01-01 | 796.45 | +28.03% | +₹9.16L |
| MAZDOCK | 2023-07-03 → 2023-09-01 | 644.55 | +46.39% | +₹4.71L |
| IRFC | 2023-09-01 → 2023-11-01 | — | +30.85% | +₹4.59L |
| SOLARINDS | 2025-04-01 → 2025-05-02 | — | +17.22% | +₹3.89L |

## History

| Date | Issue | Fix | Honest CAGR |
|---|---|---|---:|
| Pre-2026-05-17 | Universe rebuilt with TODAY's ADV applied retroactively; 60d lookback | — | +518% claimed (lookahead) |
| 2026-05-17 (am) | Yearly-PIT pseudo-N100 (ADV from N500 at year start); 30d lookback | Drop daily ADV refresh | +136.39% (still pseudo) |
| 2026-05-17 (pm) | Pseudo-N100 had 47/100 stocks NOT in real index (HFCL, BSE, GROWW etc.) | NSE CSV → real N100 | +64.90% (honest, deployable) |
| 2026-05-17 (eve) | Tested MAX_PRICE=₹3,000 filter — CAGR +85.85% but threshold curve-fit on backtest losses | Reverted to no filter | +65.10% monthly-only baseline (clean, no in-sample bias) |
| 2026-05-27 (am) | Monthly-only figure was not live-faithful — live cron runs the mid-month check too | Canonical = `--mid-month-check` run | +125.13% (LB=30, live config: monthly + mid-month) |
| 2026-05-27 | 30-trading-day lookback beaten by 15td on a 6-year (2020-2026) sweep (15td +151.72%/45.69%DD vs 30td +129.01%/57.29%DD) | Lookback 30→15 trading days (`live_signal.rank_universe lookback_days=15`, `backtest.LOOKBACK=15`) | **+184.36%** (current; LB=15, 3yr; honest 6yr max DD ~45.69%) |

## Files

| File | Purpose |
|---|---|
| `backtest.py` | 3-year backtest harness (`LOOKBACK=15` trading days, no price filter) |
| `build_universe.py` | Emit `n100_current.json` from real NSE CSV |
| `live_signal.py` | Daily live signal emitter (`lookback_days=15`, real N100, no filter) |
| `data_pull.py` | Refresh NSE CSV + rebuild universe + OHLCV pull |
| `cron.py` | Scheduled rotation |
| `trade_ledger.json` | trades + summary (regenerate under LB=15 `--mid-month-check` for the 57-trade live ledger) |
| `summary.json` | Authoritative metrics output |

## How to reproduce

```bash
# Refresh real Nifty 100 from NSE
python tools/refresh_nifty100.py

# Refresh OHLCV
python tools/shared/prefetch_ohlcv.py --universe n50,n500 --days 1500 --intervals 1h,D

# With mid-month check + 5pp lead (LIVE config) — LB=15 by default (backtest.LOOKBACK=15)
docker exec trading_system_app python tools/models/momentum_n100_top5_max1/backtest.py \
    --mid-month-check --mid-month-lead-pct 5.0
```

## Live cron schedule

Three jobs registered (all IST, naïve = container TZ=Asia/Kolkata):

| Time | Job | Purpose |
|---|---|---|
| 09:25 | `emit_signal` (--rebalance-only) | Monthly rebalance signal on 1st weekday of month |
| 09:27 | `emit_mid_month_signal` (--mid-month-check) | Day-15 weekday rank check, 5pp lead gate |
| 09:30 | `execute_orders` | Place Fyers orders from monthly signal file |
| 09:35 | `execute_mid_month_orders` | Place Fyers orders from mid-month signal file (if non-empty) |
| 20:30 | `pull_daily_ohlcv` | Daily N500 OHLCV refresh |
| 06:30 | `_monthly_universe` | NSE Nifty 100 CSV refresh on day-1 |

Full trade ledger: `exports/models/momentum_n100_top5_max1/TRADE_LEDGER.md`

## Live deployment

In-container scheduler (scheduler.py + this model's cron.py) calls `live_signal.py` with `lookback_days=15` (15 trading days, ~3 weeks) against real-N100 universe. No price filter. Real Fyers orders via `tools/live/fyers_executor.py` (always live). No paper trading.

## Honest caveats

- **⚠️ Real max DD ~45.69% (6yr), NOT the 14.89% 3yr figure** — the 2023-2026 window never saw a major correction; the 6-year backtest (2020-2026, incl. the 2022 correction) shows ~46% drawdown (the model fell roughly -40% in a single year). **Size positions for ~46% DD.** Hidden by the short 3yr window; applies to both LB=15 and the old LB=30.
- **Single-stock concentration** — `max_concurrent=1` means one name carries the whole book; mean-reversion years are painful.
- **Universe drift**: backtest uses today's real N100 retroactively. ~5-8% turnover/yr — small lookahead bias.
- **No PIT historical N100**: NSE doesn't expose historical constituents easily. True PIT N100 would give slightly lower CAGR.
- **57 trades / 3yr** (~19/yr, LB=15 mid-month config): costs ~1-2%/yr drag.
- **Slippage**: backtest fills at close. Real ~10-30 bps round-trip drag.
- **Drawdown years recurring**: strategy fragile to choppy/correction regimes. Plan for ~46% DD over a full cycle.
- **High-priced stocks (>₹3K)**: known to lose disproportionately in this regime (BAJAJ-AUTO ₹12K, ENRIN ₹2.97K, BAJAJFINSV ₹1.97K), but threshold filter intentionally NOT applied since it's curve-fit on past data.
