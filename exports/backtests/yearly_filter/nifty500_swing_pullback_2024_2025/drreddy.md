# Dr. Reddy's Laboratories Ltd. (DRREDDY)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (662 bars)
- **Last close:** 1293.90
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
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 2 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 0
- **Target hits / Stop hits / Partials:** 1 / 0 / 1
- **Avg / median % per leg:** 6.52% / 9.06%
- **Sum % (uncompounded):** 13.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 2 | 100.0% | 1 | 0 | 1 | 6.52% | 13.0% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 6.52% | 13.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 2 | 100.0% | 1 | 0 | 1 | 6.52% | 13.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-27 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 00:00:00 | 1247.18 | 1182.99 | 1204.67 | Stage2 pullback-breakout RSI=65 vol=3.8x ATR=24.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 00:00:00 | 1296.89 | 1186.71 | 1228.82 | T1 booked 50% @ 1296.89 |
| Target hit | 2024-08-14 00:00:00 | 1360.21 | 1229.34 | 1362.38 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-12-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 00:00:00 | 1325.60 | 1265.56 | 1252.93 | Stage2 pullback-breakout RSI=67 vol=3.2x ATR=29.72 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-06-27 00:00:00 | 1247.18 | 2024-07-03 00:00:00 | 1296.89 | PARTIAL | 0.50 | 3.99% |
| BUY | retest1 | 2024-06-27 00:00:00 | 1247.18 | 2024-08-14 00:00:00 | 1360.21 | TARGET_HIT | 0.50 | 9.06% |
