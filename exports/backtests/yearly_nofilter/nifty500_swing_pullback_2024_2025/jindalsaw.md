# Jindal Saw Ltd. (JINDALSAW)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (664 bars)
- **Last close:** 242.81
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** 4.71% / 7.60%
- **Sum % (uncompounded):** 18.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 4.71% | 18.8% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 4.71% | 18.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 4.71% | 18.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 00:00:00 | 284.83 | 242.52 | 276.23 | Stage2 pullback-breakout RSI=59 vol=1.9x ATR=10.45 |
| Stop hit — per-position SL triggered | 2024-07-19 00:00:00 | 269.15 | 245.41 | 278.25 | SL hit (bars_held=8) |

### Cycle 2 — BUY (started 2024-07-26 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 00:00:00 | 289.38 | 247.04 | 278.69 | Stage2 pullback-breakout RSI=61 vol=1.5x ATR=10.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 00:00:00 | 311.36 | 248.27 | 284.33 | T1 booked 50% @ 311.36 |
| Target hit | 2024-10-03 00:00:00 | 353.75 | 283.06 | 357.43 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2024-12-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 00:00:00 | 336.00 | 297.19 | 314.47 | Stage2 pullback-breakout RSI=64 vol=2.3x ATR=12.32 |
| Stop hit — per-position SL triggered | 2024-12-13 00:00:00 | 317.51 | 298.76 | 320.08 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-08 00:00:00 | 284.83 | 2024-07-19 00:00:00 | 269.15 | STOP_HIT | 1.00 | -5.51% |
| BUY | retest1 | 2024-07-26 00:00:00 | 289.38 | 2024-07-30 00:00:00 | 311.36 | PARTIAL | 0.50 | 7.60% |
| BUY | retest1 | 2024-07-26 00:00:00 | 289.38 | 2024-10-03 00:00:00 | 353.75 | TARGET_HIT | 0.50 | 22.24% |
| BUY | retest1 | 2024-12-06 00:00:00 | 336.00 | 2024-12-13 00:00:00 | 317.51 | STOP_HIT | 1.00 | -5.50% |
