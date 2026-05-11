# Union Bank of India (UNIONBANK)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 163.92
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 3
- **Target hits / Stop hits / Partials:** 3 / 3 / 3
- **Avg / median % per leg:** 3.57% / 4.11%
- **Sum % (uncompounded):** 32.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 6 | 66.7% | 3 | 3 | 3 | 3.57% | 32.2% |
| BUY @ 2nd Alert (retest1) | 9 | 6 | 66.7% | 3 | 3 | 3 | 3.57% | 32.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 6 | 66.7% | 3 | 3 | 3 | 3.57% | 32.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-30 00:00:00 | 72.25 | 66.74 | 70.89 | Stage2 pullback-breakout RSI=57 vol=2.0x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-04 00:00:00 | 75.22 | 66.92 | 71.74 | T1 booked 50% @ 75.22 |
| Target hit | 2023-08-25 00:00:00 | 89.50 | 73.45 | 90.10 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2023-09-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-14 00:00:00 | 94.50 | 75.46 | 89.46 | Stage2 pullback-breakout RSI=61 vol=3.0x ATR=3.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-20 00:00:00 | 100.69 | 76.13 | 91.76 | T1 booked 50% @ 100.69 |
| Target hit | 2023-10-09 00:00:00 | 99.35 | 79.23 | 100.34 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2023-11-13 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 00:00:00 | 112.00 | 84.44 | 103.55 | Stage2 pullback-breakout RSI=69 vol=2.7x ATR=3.75 |
| Stop hit — per-position SL triggered | 2023-11-17 00:00:00 | 106.38 | 85.21 | 105.32 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2023-12-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 00:00:00 | 114.20 | 87.45 | 107.69 | Stage2 pullback-breakout RSI=65 vol=2.4x ATR=3.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-12 00:00:00 | 121.42 | 89.23 | 112.56 | T1 booked 50% @ 121.42 |
| Target hit | 2023-12-20 00:00:00 | 116.45 | 91.18 | 116.94 | Trail-exit close<EMA20 |

### Cycle 5 — BUY (started 2024-03-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-05 00:00:00 | 157.90 | 110.49 | 145.64 | Stage2 pullback-breakout RSI=68 vol=1.7x ATR=6.13 |
| Stop hit — per-position SL triggered | 2024-03-12 00:00:00 | 148.70 | 112.20 | 148.37 | SL hit (bars_held=4) |

### Cycle 6 — BUY (started 2024-04-29 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 00:00:00 | 156.75 | 122.01 | 149.83 | Stage2 pullback-breakout RSI=59 vol=1.5x ATR=5.18 |
| Stop hit — per-position SL triggered | 2024-05-06 00:00:00 | 148.98 | 123.19 | 150.54 | SL hit (bars_held=4) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-06-30 00:00:00 | 72.25 | 2023-07-04 00:00:00 | 75.22 | PARTIAL | 0.50 | 4.11% |
| BUY | retest1 | 2023-06-30 00:00:00 | 72.25 | 2023-08-25 00:00:00 | 89.50 | TARGET_HIT | 0.50 | 23.88% |
| BUY | retest1 | 2023-09-14 00:00:00 | 94.50 | 2023-09-20 00:00:00 | 100.69 | PARTIAL | 0.50 | 6.55% |
| BUY | retest1 | 2023-09-14 00:00:00 | 94.50 | 2023-10-09 00:00:00 | 99.35 | TARGET_HIT | 0.50 | 5.13% |
| BUY | retest1 | 2023-11-13 00:00:00 | 112.00 | 2023-11-17 00:00:00 | 106.38 | STOP_HIT | 1.00 | -5.02% |
| BUY | retest1 | 2023-12-04 00:00:00 | 114.20 | 2023-12-12 00:00:00 | 121.42 | PARTIAL | 0.50 | 6.32% |
| BUY | retest1 | 2023-12-04 00:00:00 | 114.20 | 2023-12-20 00:00:00 | 116.45 | TARGET_HIT | 0.50 | 1.97% |
| BUY | retest1 | 2024-03-05 00:00:00 | 157.90 | 2024-03-12 00:00:00 | 148.70 | STOP_HIT | 1.00 | -5.82% |
| BUY | retest1 | 2024-04-29 00:00:00 | 156.75 | 2024-05-06 00:00:00 | 148.98 | STOP_HIT | 1.00 | -4.95% |
