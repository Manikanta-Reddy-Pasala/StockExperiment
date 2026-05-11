# NMDC Steel Ltd. (NSLNISP)

## Backtest Summary

- **Window:** 2023-02-20 05:30:00 → 2026-05-08 05:30:00 (795 bars)
- **Last close:** 43.66
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
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 1 / 1 / 2
- **Avg / median % per leg:** 7.30% / 10.63%
- **Sum % (uncompounded):** 29.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 1 | 2 | 7.30% | 29.2% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 1 | 2 | 7.30% | 29.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 1 | 2 | 7.30% | 29.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-01-20 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-20 05:30:00 | 52.15 | 44.96 | 49.60 | Stage2 pullback-breakout RSI=62 vol=2.8x ATR=1.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-25 05:30:00 | 55.93 | 45.26 | 51.04 | T1 booked 50% @ 55.93 |
| Target hit | 2024-02-12 05:30:00 | 58.05 | 47.60 | 61.89 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-04-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 05:30:00 | 60.95 | 50.82 | 57.37 | Stage2 pullback-breakout RSI=58 vol=1.7x ATR=3.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-12 05:30:00 | 67.43 | 51.71 | 60.27 | T1 booked 50% @ 67.43 |
| Stop hit — per-position SL triggered | 2024-04-15 05:30:00 | 60.95 | 51.82 | 60.46 | SL hit (bars_held=9) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-20 05:30:00 | 52.15 | 2024-01-25 05:30:00 | 55.93 | PARTIAL | 0.50 | 7.25% |
| BUY | retest1 | 2024-01-20 05:30:00 | 52.15 | 2024-02-12 05:30:00 | 58.05 | TARGET_HIT | 0.50 | 11.31% |
| BUY | retest1 | 2024-04-01 05:30:00 | 60.95 | 2024-04-12 05:30:00 | 67.43 | PARTIAL | 0.50 | 10.63% |
| BUY | retest1 | 2024-04-01 05:30:00 | 60.95 | 2024-04-15 05:30:00 | 60.95 | STOP_HIT | 0.50 | 0.00% |
