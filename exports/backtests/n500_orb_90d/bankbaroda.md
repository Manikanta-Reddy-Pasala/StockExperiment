# Bank of Baroda (BANKBARODA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 263.50
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
| ENTRY1 | 12 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 5 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 7
- **Target hits / Stop hits / Partials:** 5 / 7 / 6
- **Avg / median % per leg:** 0.48% / 0.43%
- **Sum % (uncompounded):** 8.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 10 | 71.4% | 5 | 4 | 5 | 0.63% | 8.8% |
| BUY @ 2nd Alert (retest1) | 14 | 10 | 71.4% | 5 | 4 | 5 | 0.63% | 8.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.04% | -0.2% |
| SELL @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.04% | -0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 18 | 11 | 61.1% | 5 | 7 | 6 | 0.48% | 8.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:05:00 | 291.50 | 290.55 | 0.00 | ORB-long ORB[289.25,291.20] vol=2.7x ATR=0.70 |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 290.80 | 290.67 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 11:00:00 | 288.20 | 286.67 | 0.00 | ORB-long ORB[284.20,286.60] vol=4.3x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:40:00 | 289.11 | 287.21 | 0.00 | T1 1.5R @ 289.11 |
| Target hit | 2026-02-16 15:20:00 | 292.50 | 290.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:20:00 | 295.40 | 294.02 | 0.00 | ORB-long ORB[291.20,295.00] vol=1.8x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:30:00 | 296.72 | 294.36 | 0.00 | T1 1.5R @ 296.72 |
| Target hit | 2026-02-17 15:20:00 | 303.30 | 300.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-02-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:40:00 | 306.65 | 306.20 | 0.00 | ORB-long ORB[302.55,306.60] vol=2.4x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:25:00 | 307.97 | 306.59 | 0.00 | T1 1.5R @ 307.97 |
| Target hit | 2026-02-20 14:10:00 | 307.60 | 307.62 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — BUY (started 2026-03-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 11:00:00 | 289.55 | 286.57 | 0.00 | ORB-long ORB[282.95,287.00] vol=3.0x ATR=1.02 |
| Stop hit — per-position SL triggered | 2026-03-12 11:05:00 | 288.53 | 286.66 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:45:00 | 282.50 | 284.85 | 0.00 | ORB-short ORB[285.50,287.50] vol=1.7x ATR=0.91 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 283.41 | 284.77 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:35:00 | 272.27 | 270.25 | 0.00 | ORB-long ORB[268.01,271.88] vol=2.3x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 10:30:00 | 274.72 | 271.17 | 0.00 | T1 1.5R @ 274.72 |
| Target hit | 2026-04-08 15:20:00 | 275.79 | 273.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2026-04-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:40:00 | 271.32 | 269.29 | 0.00 | ORB-long ORB[266.70,270.48] vol=1.7x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 12:00:00 | 272.81 | 270.25 | 0.00 | T1 1.5R @ 272.81 |
| Target hit | 2026-04-13 15:20:00 | 275.60 | 272.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2026-04-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 11:10:00 | 278.06 | 279.83 | 0.00 | ORB-short ORB[279.02,283.00] vol=6.5x ATR=0.87 |
| Stop hit — per-position SL triggered | 2026-04-15 11:20:00 | 278.93 | 279.77 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:55:00 | 284.46 | 283.83 | 0.00 | ORB-long ORB[281.36,284.40] vol=1.6x ATR=0.63 |
| Stop hit — per-position SL triggered | 2026-04-22 11:05:00 | 283.83 | 283.86 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:15:00 | 262.31 | 263.66 | 0.00 | ORB-short ORB[263.30,266.22] vol=2.0x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 11:45:00 | 261.08 | 262.48 | 0.00 | T1 1.5R @ 261.08 |
| Stop hit — per-position SL triggered | 2026-04-30 12:25:00 | 262.31 | 262.27 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:40:00 | 270.70 | 268.89 | 0.00 | ORB-long ORB[264.90,268.40] vol=1.9x ATR=1.06 |
| Stop hit — per-position SL triggered | 2026-05-04 09:50:00 | 269.64 | 269.03 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-12 10:05:00 | 291.50 | 2026-02-12 10:15:00 | 290.80 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-16 11:00:00 | 288.20 | 2026-02-16 11:40:00 | 289.11 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2026-02-16 11:00:00 | 288.20 | 2026-02-16 15:20:00 | 292.50 | TARGET_HIT | 0.50 | 1.49% |
| BUY | retest1 | 2026-02-17 10:20:00 | 295.40 | 2026-02-17 10:30:00 | 296.72 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-17 10:20:00 | 295.40 | 2026-02-17 15:20:00 | 303.30 | TARGET_HIT | 0.50 | 2.67% |
| BUY | retest1 | 2026-02-20 10:40:00 | 306.65 | 2026-02-20 11:25:00 | 307.97 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-02-20 10:40:00 | 306.65 | 2026-02-20 14:10:00 | 307.60 | TARGET_HIT | 0.50 | 0.31% |
| BUY | retest1 | 2026-03-12 11:00:00 | 289.55 | 2026-03-12 11:05:00 | 288.53 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-03-13 10:45:00 | 282.50 | 2026-03-13 10:50:00 | 283.41 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-08 09:35:00 | 272.27 | 2026-04-08 10:30:00 | 274.72 | PARTIAL | 0.50 | 0.90% |
| BUY | retest1 | 2026-04-08 09:35:00 | 272.27 | 2026-04-08 15:20:00 | 275.79 | TARGET_HIT | 0.50 | 1.29% |
| BUY | retest1 | 2026-04-13 10:40:00 | 271.32 | 2026-04-13 12:00:00 | 272.81 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-04-13 10:40:00 | 271.32 | 2026-04-13 15:20:00 | 275.60 | TARGET_HIT | 0.50 | 1.58% |
| SELL | retest1 | 2026-04-15 11:10:00 | 278.06 | 2026-04-15 11:20:00 | 278.93 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-22 10:55:00 | 284.46 | 2026-04-22 11:05:00 | 283.83 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-04-30 10:15:00 | 262.31 | 2026-04-30 11:45:00 | 261.08 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-04-30 10:15:00 | 262.31 | 2026-04-30 12:25:00 | 262.31 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-04 09:40:00 | 270.70 | 2026-05-04 09:50:00 | 269.64 | STOP_HIT | 1.00 | -0.39% |
