# Bank of India (BANKINDIA)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 143.47
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
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 5
- **Target hits / Stop hits / Partials:** 1 / 5 / 3
- **Avg / median % per leg:** 1.21% / 0.00%
- **Sum % (uncompounded):** 10.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 4 | 44.4% | 1 | 5 | 3 | 1.21% | 10.9% |
| BUY @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 1 | 5 | 3 | 1.21% | 10.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 9 | 4 | 44.4% | 1 | 5 | 3 | 1.21% | 10.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 00:00:00 | 77.30 | 72.81 | 73.82 | Stage2 pullback-breakout RSI=61 vol=3.2x ATR=1.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-06 00:00:00 | 81.19 | 73.03 | 75.52 | T1 booked 50% @ 81.19 |
| Stop hit — per-position SL triggered | 2023-07-13 00:00:00 | 77.30 | 73.29 | 76.64 | SL hit (bars_held=8) |

### Cycle 2 — BUY (started 2023-09-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 00:00:00 | 93.40 | 77.27 | 87.67 | Stage2 pullback-breakout RSI=67 vol=2.8x ATR=2.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-11 00:00:00 | 98.99 | 77.94 | 90.00 | T1 booked 50% @ 98.99 |
| Target hit | 2023-10-09 00:00:00 | 104.30 | 82.78 | 104.91 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2023-12-01 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-01 00:00:00 | 107.50 | 88.91 | 104.16 | Stage2 pullback-breakout RSI=59 vol=2.7x ATR=3.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-06 00:00:00 | 113.69 | 89.60 | 106.26 | T1 booked 50% @ 113.69 |
| Stop hit — per-position SL triggered | 2023-12-20 00:00:00 | 107.50 | 91.83 | 110.26 | SL hit (bars_held=13) |

### Cycle 4 — BUY (started 2024-03-05 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-05 00:00:00 | 143.70 | 108.39 | 137.39 | Stage2 pullback-breakout RSI=60 vol=1.7x ATR=5.75 |
| Stop hit — per-position SL triggered | 2024-03-13 00:00:00 | 135.08 | 109.97 | 138.40 | SL hit (bars_held=5) |

### Cycle 5 — BUY (started 2024-04-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 00:00:00 | 145.30 | 113.11 | 137.33 | Stage2 pullback-breakout RSI=61 vol=1.7x ATR=6.02 |
| Stop hit — per-position SL triggered | 2024-04-15 00:00:00 | 136.27 | 115.20 | 140.49 | SL hit (bars_held=7) |

### Cycle 6 — BUY (started 2024-04-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 00:00:00 | 150.35 | 117.00 | 141.56 | Stage2 pullback-breakout RSI=63 vol=2.1x ATR=5.37 |
| Stop hit — per-position SL triggered | 2024-05-06 00:00:00 | 142.29 | 119.01 | 145.85 | SL hit (bars_held=6) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-03 00:00:00 | 77.30 | 2023-07-06 00:00:00 | 81.19 | PARTIAL | 0.50 | 5.03% |
| BUY | retest1 | 2023-07-03 00:00:00 | 77.30 | 2023-07-13 00:00:00 | 77.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-05 00:00:00 | 93.40 | 2023-09-11 00:00:00 | 98.99 | PARTIAL | 0.50 | 5.99% |
| BUY | retest1 | 2023-09-05 00:00:00 | 93.40 | 2023-10-09 00:00:00 | 104.30 | TARGET_HIT | 0.50 | 11.67% |
| BUY | retest1 | 2023-12-01 00:00:00 | 107.50 | 2023-12-06 00:00:00 | 113.69 | PARTIAL | 0.50 | 5.76% |
| BUY | retest1 | 2023-12-01 00:00:00 | 107.50 | 2023-12-20 00:00:00 | 107.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-05 00:00:00 | 143.70 | 2024-03-13 00:00:00 | 135.08 | STOP_HIT | 1.00 | -6.00% |
| BUY | retest1 | 2024-04-03 00:00:00 | 145.30 | 2024-04-15 00:00:00 | 136.27 | STOP_HIT | 1.00 | -6.21% |
| BUY | retest1 | 2024-04-25 00:00:00 | 150.35 | 2024-05-06 00:00:00 | 142.29 | STOP_HIT | 1.00 | -5.36% |
