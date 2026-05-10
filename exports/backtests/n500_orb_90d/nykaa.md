# FSN E-Commerce Ventures Ltd. (NYKAA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 273.00
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
| ENTRY1 | 15 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 2 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 13
- **Target hits / Stop hits / Partials:** 2 / 13 / 6
- **Avg / median % per leg:** 0.07% / 0.00%
- **Sum % (uncompounded):** 1.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 4 | 44.4% | 2 | 5 | 2 | 0.08% | 0.7% |
| BUY @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 2 | 5 | 2 | 0.08% | 0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 4 | 33.3% | 0 | 8 | 4 | 0.06% | 0.8% |
| SELL @ 2nd Alert (retest1) | 12 | 4 | 33.3% | 0 | 8 | 4 | 0.06% | 0.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 21 | 8 | 38.1% | 2 | 13 | 6 | 0.07% | 1.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:55:00 | 277.73 | 276.44 | 0.00 | ORB-long ORB[275.01,277.45] vol=3.8x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:25:00 | 278.81 | 276.80 | 0.00 | T1 1.5R @ 278.81 |
| Target hit | 2026-02-12 15:20:00 | 279.80 | 278.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:25:00 | 275.51 | 277.13 | 0.00 | ORB-short ORB[277.55,280.25] vol=6.0x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 11:05:00 | 274.21 | 276.78 | 0.00 | T1 1.5R @ 274.21 |
| Stop hit — per-position SL triggered | 2026-02-13 11:50:00 | 275.51 | 276.22 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 10:40:00 | 269.82 | 272.59 | 0.00 | ORB-short ORB[272.66,276.38] vol=1.6x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:50:00 | 268.58 | 271.68 | 0.00 | T1 1.5R @ 268.58 |
| Stop hit — per-position SL triggered | 2026-02-17 13:05:00 | 269.82 | 270.59 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:55:00 | 268.25 | 271.14 | 0.00 | ORB-short ORB[271.20,274.40] vol=1.8x ATR=0.74 |
| Stop hit — per-position SL triggered | 2026-02-19 11:00:00 | 268.99 | 270.95 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-20 11:00:00 | 264.62 | 267.37 | 0.00 | ORB-short ORB[266.20,269.38] vol=1.6x ATR=0.87 |
| Stop hit — per-position SL triggered | 2026-02-20 11:30:00 | 265.49 | 266.27 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 261.40 | 263.05 | 0.00 | ORB-short ORB[262.51,266.44] vol=2.1x ATR=0.85 |
| Stop hit — per-position SL triggered | 2026-02-24 09:45:00 | 262.25 | 262.45 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:35:00 | 265.40 | 265.17 | 0.00 | ORB-long ORB[262.66,265.39] vol=2.2x ATR=0.96 |
| Stop hit — per-position SL triggered | 2026-02-25 09:40:00 | 264.44 | 265.11 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-02-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:50:00 | 269.67 | 267.45 | 0.00 | ORB-long ORB[264.77,268.50] vol=2.2x ATR=0.77 |
| Stop hit — per-position SL triggered | 2026-02-26 11:25:00 | 268.90 | 267.96 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:10:00 | 256.10 | 257.60 | 0.00 | ORB-short ORB[257.25,259.45] vol=1.5x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:45:00 | 254.66 | 256.79 | 0.00 | T1 1.5R @ 254.66 |
| Stop hit — per-position SL triggered | 2026-03-05 12:25:00 | 256.10 | 256.20 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 10:15:00 | 254.40 | 252.41 | 0.00 | ORB-long ORB[251.25,253.15] vol=2.6x ATR=1.04 |
| Stop hit — per-position SL triggered | 2026-03-10 10:20:00 | 253.36 | 252.46 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 09:45:00 | 240.60 | 238.73 | 0.00 | ORB-long ORB[235.65,238.60] vol=1.5x ATR=1.21 |
| Stop hit — per-position SL triggered | 2026-03-16 09:55:00 | 239.39 | 238.82 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-03-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 11:05:00 | 240.20 | 241.48 | 0.00 | ORB-short ORB[241.00,243.65] vol=2.7x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 11:15:00 | 239.03 | 241.03 | 0.00 | T1 1.5R @ 239.03 |
| Stop hit — per-position SL triggered | 2026-03-27 13:15:00 | 240.20 | 239.31 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 10:05:00 | 264.22 | 265.14 | 0.00 | ORB-short ORB[265.00,267.20] vol=2.9x ATR=0.76 |
| Stop hit — per-position SL triggered | 2026-04-21 11:35:00 | 264.98 | 264.52 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:55:00 | 266.90 | 265.13 | 0.00 | ORB-long ORB[263.19,266.21] vol=3.1x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 12:00:00 | 268.11 | 265.98 | 0.00 | T1 1.5R @ 268.11 |
| Target hit | 2026-04-27 15:20:00 | 269.38 | 268.32 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2026-05-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:10:00 | 268.80 | 266.86 | 0.00 | ORB-long ORB[264.40,267.80] vol=2.1x ATR=0.71 |
| Stop hit — per-position SL triggered | 2026-05-04 11:35:00 | 268.09 | 267.37 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-12 10:55:00 | 277.73 | 2026-02-12 11:25:00 | 278.81 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-12 10:55:00 | 277.73 | 2026-02-12 15:20:00 | 279.80 | TARGET_HIT | 0.50 | 0.75% |
| SELL | retest1 | 2026-02-13 10:25:00 | 275.51 | 2026-02-13 11:05:00 | 274.21 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-02-13 10:25:00 | 275.51 | 2026-02-13 11:50:00 | 275.51 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-17 10:40:00 | 269.82 | 2026-02-17 10:50:00 | 268.58 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-02-17 10:40:00 | 269.82 | 2026-02-17 13:05:00 | 269.82 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 10:55:00 | 268.25 | 2026-02-19 11:00:00 | 268.99 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-20 11:00:00 | 264.62 | 2026-02-20 11:30:00 | 265.49 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-24 09:30:00 | 261.40 | 2026-02-24 09:45:00 | 262.25 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-25 09:35:00 | 265.40 | 2026-02-25 09:40:00 | 264.44 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-02-26 10:50:00 | 269.67 | 2026-02-26 11:25:00 | 268.90 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-03-05 10:10:00 | 256.10 | 2026-03-05 11:45:00 | 254.66 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-03-05 10:10:00 | 256.10 | 2026-03-05 12:25:00 | 256.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-10 10:15:00 | 254.40 | 2026-03-10 10:20:00 | 253.36 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-03-16 09:45:00 | 240.60 | 2026-03-16 09:55:00 | 239.39 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2026-03-27 11:05:00 | 240.20 | 2026-03-27 11:15:00 | 239.03 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-03-27 11:05:00 | 240.20 | 2026-03-27 13:15:00 | 240.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-21 10:05:00 | 264.22 | 2026-04-21 11:35:00 | 264.98 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-27 10:55:00 | 266.90 | 2026-04-27 12:00:00 | 268.11 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-04-27 10:55:00 | 266.90 | 2026-04-27 15:20:00 | 269.38 | TARGET_HIT | 0.50 | 0.93% |
| BUY | retest1 | 2026-05-04 11:10:00 | 268.80 | 2026-05-04 11:35:00 | 268.09 | STOP_HIT | 1.00 | -0.26% |
