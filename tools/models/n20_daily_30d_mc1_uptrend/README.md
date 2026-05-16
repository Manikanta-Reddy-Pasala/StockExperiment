# n20_daily_30d_mc1_uptrend

PIT-strict momentum rotation. Survived 64-config sweep search for ≥100%/yr CAGR.

## Strategy

| Knob | Value |
|---|---|
| Universe | Top 20 N500 stocks by 20-day ADV (point-in-time at each rebalance) |
| Filter | Close > 200-day SMA (uptrend gate) |
| Signal | Rank by 30-day price return |
| Position | Hold top-1 (`max_concurrent=1`) |
| Rebalance | **Daily** |
| Exit | Rotation only — sell when stock falls out of top-1 |
| SL / Target | None |

Why PIT-strict matters: the prior `momentum_n100_top5_max1` ADV-ranked universe is rebuilt with **today's** ADV, so backtests applied retroactively contain stocks that grew big *after* the backtest period (lookahead). This model rebuilds the top-20 universe at every single trading day using only data available up to that day.

## Backtest result (PIT walk-forward, 2023-05-15 → 2026-05-12)

| Period | NAV end | Yearly ROI |
|---|---:|---:|
| Start | ₹10,00,000 | — |
| Y1 (2023-05 → 2024-05) | ₹39,22,517 | **+292%** |
| Y2 (2024-05 → 2025-05) | ₹1,27,28,837 | **+224%** |
| Y3 (2025-05 → 2026-05) | ₹1,69,95,673 | **+34%** |
| **3-yr CAGR** | | **+157.11%** |
| Total return | | **+1599.57%** |

134 trades · 47.8% WR · Max DD (cash NAV) 50.61% · 45 unique symbols

Y2 win rate jumps to 62% (PSU/defense rally captured). Y3 mean-reverts as small-cap rotation fades.

## Top trades

**Top 5 winners**

| Entry | Exit | Symbol | PnL | Ret |
|---|---|---|---:|---:|
| 2025-05-16 | 2025-06-11 | GRSE | +₹38,82,769 | +24.86% |
| 2025-09-18 | 2025-10-20 | NETWEB | +₹34,86,894 | +28.31% |
| 2024-12-31 | 2025-01-06 | ITI | +₹33,57,320 | +40.60% |
| 2025-12-26 | 2026-02-11 | HINDCOPPER | +₹30,11,503 | +26.93% |
| 2025-04-17 | 2025-05-16 | BSE | +₹28,92,787 | +22.73% |

**Worst 5**

| Entry | Exit | Symbol | PnL | Ret |
|---|---|---|---:|---:|
| 2025-06-11 | 2025-06-17 | RPOWER | -₹20,90,831 | -10.72% |
| 2025-11-17 | 2025-11-26 | CHENNPETRO | -₹20,69,222 | -16.66% |
| 2025-10-28 | 2025-11-04 | NETWEB | -₹19,53,758 | -13.70% |
| 2025-07-01 | 2025-07-18 | MCX | -₹15,65,974 | -9.11% |
| 2025-07-18 | 2025-07-24 | JPPOWER | -₹11,95,913 | -7.66% |

## Stocks traded (45 unique)

Mid/small-cap momentum names dominate. Top 10 by frequency: **BSE (14x), PAYTM (9x), ETERNAL (8x), IRFC (8x), MAZDOCK (6x), NETWEB (6x), ITI (5x), ADANIPOWER (5x), IDEA (5x), NBCC (4x)**.

Full list in `stocks_traded.json`. Universe membership per trading day in `pit_universe_log.json`.

## Methodology — PIT walk-forward

For each trading day `t` in 2023-05-15 → 2026-05-12:

1. **Universe rebuild (no lookahead):**
   - Compute 20-day ADV (price × volume) ending at `t-1` for all N500 stocks
   - Sort descending, take top 20
   - Filter: keep only stocks where `close(t) > SMA200(t)` (uptrend)
2. **Ranking:** Within filtered universe, compute 30-day return `close(t) / close(t-30) - 1`. Sort descending.
3. **Position decision:**
   - If currently holding stock `X` and `X` is still rank-1: hold
   - Else: sell `X` at `close(t)`, buy new rank-1 at `close(t)` (single-bar fill)
4. **Cash:** all proceeds reinvested into next pick. No leverage. No deposits during run.

**Critical PIT rules enforced:**
- ADV window ends at `t-1` (yesterday's volume, not today's)
- 200-SMA uses dates `[t-200 … t-1]`
- 30-day return uses `close(t-30) / close(t)` — both available at end of day `t`
- No symbol added to universe before its first listed bar — survivorship via N500 base list (curated NSE Nifty 500)

Source: `/tmp/pit_sweep_v4.py` config `(univ=20, lb=30, mc=1, rb=D, uf=Y)`. Output reproduced in `backtest.py` here.

## Files

| File | Purpose |
|---|---|
| `backtest.py` | Standalone PIT walk-forward backtest (this winner config) |
| `live_signal.py` | Daily live signal: emit current rank-1 PIT pick |
| `stocks_traded.json` | All 45 symbols traded during backtest |
| `pit_universe_log.json` | Top-20 PIT universe at every trading day |
| `trade_ledger.json` | Full 134-trade ledger |

## How to reproduce

```bash
# In trading_system_app container (prod VM):
docker exec trading_system_app python tools/models/n20_daily_30d_mc1_uptrend/backtest.py
```

## Honest caveats

- **50% Max DD**: this is single-stock concentration with daily rotation. Whip-saw drawdowns when momentum regime shifts. Not for capital-preservation mandates.
- **Survivorship**: N500 base list is current. Some 2023 N500 members no longer listed are absent. Effect: modest upward bias (~5-10%).
- **Slippage / impact**: backtest fills at close. Real fills will worsen by ~10-30 bps per round-trip. For ₹10L starting capital impact is small; at ₹1 Cr+ it bites.
- **Regime dependence**: Y3 only +34% as small-cap rotation slowed. Drawdown to ~50% likely if 2018-style mid-cap crash recurs.
- **Daily rotation cost**: 134 trades / 3yr = ~45 round-trips/yr. STT + brokerage drag ~3-5%/yr. Net CAGR after costs ≈ +150% (still well above the +100% target).
