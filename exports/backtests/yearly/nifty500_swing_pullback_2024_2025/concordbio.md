# Concord Biotech Ltd. (CONCORDBIO)

## Backtest Summary

- **Window:** 2023-09-04 05:30:00 → 2026-05-08 05:30:00 (663 bars)
- **Last close:** 1173.90
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
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 2 / 1 / 2
- **Avg / median % per leg:** 8.20% / 8.88%
- **Sum % (uncompounded):** 41.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 2 | 1 | 2 | 8.20% | 41.0% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 2 | 1 | 2 | 8.20% | 41.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 5 | 4 | 80.0% | 2 | 1 | 2 | 8.20% | 41.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-29 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 05:30:00 | 1705.60 | 1468.98 | 1619.48 | Stage2 pullback-breakout RSI=61 vol=12.4x ATR=75.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 05:30:00 | 1856.99 | 1483.32 | 1678.97 | T1 booked 50% @ 1856.99 |
| Target hit | 2024-09-27 05:30:00 | 2047.80 | 1577.38 | 2063.58 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-11-11 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 05:30:00 | 1882.95 | 1656.92 | 1867.42 | Stage2 pullback-breakout RSI=51 vol=2.6x ATR=81.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-25 05:30:00 | 2046.35 | 1678.63 | 1914.86 | T1 booked 50% @ 2046.35 |
| Target hit | 2024-12-17 05:30:00 | 2059.10 | 1745.02 | 2083.24 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2025-01-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-08 05:30:00 | 2262.90 | 1803.06 | 2147.96 | Stage2 pullback-breakout RSI=63 vol=2.0x ATR=90.12 |
| Stop hit — per-position SL triggered | 2025-01-22 05:30:00 | 2127.72 | 1843.45 | 2188.46 | SL hit (bars_held=10) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-29 05:30:00 | 1705.60 | 2024-09-05 05:30:00 | 1856.99 | PARTIAL | 0.50 | 8.88% |
| BUY | retest1 | 2024-08-29 05:30:00 | 1705.60 | 2024-09-27 05:30:00 | 2047.80 | TARGET_HIT | 0.50 | 20.06% |
| BUY | retest1 | 2024-11-11 05:30:00 | 1882.95 | 2024-11-25 05:30:00 | 2046.35 | PARTIAL | 0.50 | 8.68% |
| BUY | retest1 | 2024-11-11 05:30:00 | 1882.95 | 2024-12-17 05:30:00 | 2059.10 | TARGET_HIT | 0.50 | 9.36% |
| BUY | retest1 | 2025-01-08 05:30:00 | 2262.90 | 2025-01-22 05:30:00 | 2127.72 | STOP_HIT | 1.00 | -5.97% |
