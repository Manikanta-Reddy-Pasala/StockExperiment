# Tata Power Co. Ltd. (TATAPOWER)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 431.85
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
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 3
- **Target hits / Stop hits / Partials:** 2 / 4 / 4
- **Avg / median % per leg:** 5.71% / 4.20%
- **Sum % (uncompounded):** 57.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 7 | 70.0% | 2 | 4 | 4 | 5.71% | 57.1% |
| BUY @ 2nd Alert (retest1) | 10 | 7 | 70.0% | 2 | 4 | 4 | 5.71% | 57.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 10 | 7 | 70.0% | 2 | 4 | 4 | 5.71% | 57.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-06 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 00:00:00 | 229.60 | 214.38 | 220.65 | Stage2 pullback-breakout RSI=70 vol=3.1x ATR=4.04 |
| Stop hit — per-position SL triggered | 2023-07-13 00:00:00 | 223.54 | 214.91 | 222.40 | SL hit (bars_held=5) |

### Cycle 2 — BUY (started 2023-07-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 00:00:00 | 234.65 | 215.59 | 222.15 | Stage2 pullback-breakout RSI=70 vol=5.0x ATR=4.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-01 00:00:00 | 243.62 | 216.08 | 225.52 | T1 booked 50% @ 243.62 |
| Stop hit — per-position SL triggered | 2023-08-02 00:00:00 | 234.65 | 216.28 | 226.49 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2023-08-22 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 00:00:00 | 243.50 | 218.59 | 233.16 | Stage2 pullback-breakout RSI=64 vol=1.5x ATR=5.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-01 00:00:00 | 254.63 | 220.82 | 241.30 | T1 booked 50% @ 254.63 |
| Target hit | 2023-09-21 00:00:00 | 256.65 | 225.98 | 257.15 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2023-11-08 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 00:00:00 | 254.95 | 232.66 | 247.41 | Stage2 pullback-breakout RSI=59 vol=1.8x ATR=5.48 |
| Stop hit — per-position SL triggered | 2023-11-22 00:00:00 | 262.10 | 235.08 | 254.85 | Time-stop (10d <3%) |

### Cycle 5 — BUY (started 2023-11-28 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-28 00:00:00 | 270.80 | 235.91 | 257.12 | Stage2 pullback-breakout RSI=68 vol=2.9x ATR=5.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-04 00:00:00 | 282.16 | 237.43 | 263.06 | T1 booked 50% @ 282.16 |
| Target hit | 2024-02-12 00:00:00 | 361.70 | 280.88 | 374.03 | Trail-exit close<EMA20 |

### Cycle 6 — BUY (started 2024-03-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-04 00:00:00 | 391.75 | 295.04 | 376.84 | Stage2 pullback-breakout RSI=62 vol=1.8x ATR=10.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-07 00:00:00 | 412.81 | 298.31 | 384.52 | T1 booked 50% @ 412.81 |
| Stop hit — per-position SL triggered | 2024-03-13 00:00:00 | 391.75 | 301.12 | 386.55 | SL hit (bars_held=6) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-07-06 00:00:00 | 229.60 | 2023-07-13 00:00:00 | 223.54 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest1 | 2023-07-28 00:00:00 | 234.65 | 2023-08-01 00:00:00 | 243.62 | PARTIAL | 0.50 | 3.82% |
| BUY | retest1 | 2023-07-28 00:00:00 | 234.65 | 2023-08-02 00:00:00 | 234.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-22 00:00:00 | 243.50 | 2023-09-01 00:00:00 | 254.63 | PARTIAL | 0.50 | 4.57% |
| BUY | retest1 | 2023-08-22 00:00:00 | 243.50 | 2023-09-21 00:00:00 | 256.65 | TARGET_HIT | 0.50 | 5.40% |
| BUY | retest1 | 2023-11-08 00:00:00 | 254.95 | 2023-11-22 00:00:00 | 262.10 | STOP_HIT | 1.00 | 2.80% |
| BUY | retest1 | 2023-11-28 00:00:00 | 270.80 | 2023-12-04 00:00:00 | 282.16 | PARTIAL | 0.50 | 4.20% |
| BUY | retest1 | 2023-11-28 00:00:00 | 270.80 | 2024-02-12 00:00:00 | 361.70 | TARGET_HIT | 0.50 | 33.57% |
| BUY | retest1 | 2024-03-04 00:00:00 | 391.75 | 2024-03-07 00:00:00 | 412.81 | PARTIAL | 0.50 | 5.37% |
| BUY | retest1 | 2024-03-04 00:00:00 | 391.75 | 2024-03-13 00:00:00 | 391.75 | STOP_HIT | 0.50 | 0.00% |
