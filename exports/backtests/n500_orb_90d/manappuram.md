# Manappuram Finance Ltd. (MANAPPURAM)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 315.05
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
| ENTRY1 | 20 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 17
- **Target hits / Stop hits / Partials:** 3 / 17 / 5
- **Avg / median % per leg:** 0.06% / -0.28%
- **Sum % (uncompounded):** 1.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 5 | 33.3% | 2 | 10 | 3 | 0.16% | 2.4% |
| BUY @ 2nd Alert (retest1) | 15 | 5 | 33.3% | 2 | 10 | 3 | 0.16% | 2.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 3 | 30.0% | 1 | 7 | 2 | -0.09% | -0.9% |
| SELL @ 2nd Alert (retest1) | 10 | 3 | 30.0% | 1 | 7 | 2 | -0.09% | -0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 25 | 8 | 32.0% | 3 | 17 | 5 | 0.06% | 1.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:30:00 | 306.15 | 304.08 | 0.00 | ORB-long ORB[302.70,305.10] vol=1.9x ATR=1.04 |
| Stop hit — per-position SL triggered | 2026-02-10 10:40:00 | 305.11 | 304.28 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 09:40:00 | 308.35 | 307.07 | 0.00 | ORB-long ORB[306.10,308.00] vol=1.7x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 09:50:00 | 310.29 | 308.02 | 0.00 | T1 1.5R @ 310.29 |
| Stop hit — per-position SL triggered | 2026-02-11 10:10:00 | 308.35 | 308.32 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 10:50:00 | 299.50 | 301.73 | 0.00 | ORB-short ORB[300.95,303.70] vol=1.7x ATR=0.98 |
| Stop hit — per-position SL triggered | 2026-02-17 11:15:00 | 300.48 | 301.51 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:35:00 | 308.95 | 307.53 | 0.00 | ORB-long ORB[306.00,308.10] vol=2.3x ATR=0.86 |
| Stop hit — per-position SL triggered | 2026-02-18 09:45:00 | 308.09 | 307.85 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-02-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 09:55:00 | 307.60 | 308.06 | 0.00 | ORB-short ORB[307.90,310.35] vol=2.3x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:35:00 | 306.44 | 307.41 | 0.00 | T1 1.5R @ 306.44 |
| Stop hit — per-position SL triggered | 2026-02-19 10:50:00 | 307.60 | 307.35 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:45:00 | 307.20 | 306.08 | 0.00 | ORB-long ORB[303.35,306.00] vol=3.0x ATR=0.84 |
| Stop hit — per-position SL triggered | 2026-02-20 11:05:00 | 306.36 | 306.14 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:45:00 | 304.80 | 302.61 | 0.00 | ORB-long ORB[301.05,304.65] vol=2.3x ATR=1.09 |
| Stop hit — per-position SL triggered | 2026-02-24 10:00:00 | 303.71 | 302.81 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-02-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 10:40:00 | 301.80 | 303.68 | 0.00 | ORB-short ORB[304.20,305.85] vol=1.6x ATR=0.89 |
| Stop hit — per-position SL triggered | 2026-02-25 10:55:00 | 302.69 | 303.45 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:00:00 | 268.25 | 269.17 | 0.00 | ORB-short ORB[269.45,273.10] vol=3.1x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:25:00 | 266.99 | 268.95 | 0.00 | T1 1.5R @ 266.99 |
| Target hit | 2026-03-05 14:40:00 | 267.40 | 267.32 | 0.00 | Trail-exit close>VWAP |

### Cycle 10 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 264.95 | 267.11 | 0.00 | ORB-short ORB[266.30,270.05] vol=1.8x ATR=0.93 |
| Stop hit — per-position SL triggered | 2026-03-06 11:05:00 | 265.88 | 267.00 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 09:45:00 | 264.35 | 262.54 | 0.00 | ORB-long ORB[260.45,263.90] vol=2.1x ATR=1.53 |
| Stop hit — per-position SL triggered | 2026-03-10 10:00:00 | 262.82 | 262.71 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:25:00 | 254.30 | 252.13 | 0.00 | ORB-long ORB[249.70,253.50] vol=2.2x ATR=1.14 |
| Stop hit — per-position SL triggered | 2026-03-17 10:35:00 | 253.16 | 252.22 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-03-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-23 09:40:00 | 253.15 | 251.48 | 0.00 | ORB-long ORB[249.50,253.00] vol=1.5x ATR=1.42 |
| Stop hit — per-position SL triggered | 2026-03-23 09:50:00 | 251.73 | 251.55 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-01 11:05:00 | 256.50 | 258.06 | 0.00 | ORB-short ORB[257.05,260.45] vol=2.9x ATR=0.98 |
| Stop hit — per-position SL triggered | 2026-04-01 11:30:00 | 257.48 | 258.00 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 10:40:00 | 266.50 | 265.02 | 0.00 | ORB-long ORB[262.00,265.70] vol=2.0x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 10:55:00 | 268.17 | 265.40 | 0.00 | T1 1.5R @ 268.17 |
| Target hit | 2026-04-08 15:20:00 | 269.95 | 267.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2026-04-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 10:05:00 | 264.10 | 265.53 | 0.00 | ORB-short ORB[264.95,267.75] vol=1.8x ATR=0.88 |
| Stop hit — per-position SL triggered | 2026-04-10 10:15:00 | 264.98 | 265.42 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-04-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 09:30:00 | 271.40 | 272.72 | 0.00 | ORB-short ORB[271.45,274.75] vol=1.6x ATR=1.14 |
| Stop hit — per-position SL triggered | 2026-04-15 10:10:00 | 272.54 | 272.17 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-04-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:00:00 | 274.25 | 272.84 | 0.00 | ORB-long ORB[270.10,273.45] vol=2.4x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:35:00 | 275.54 | 273.58 | 0.00 | T1 1.5R @ 275.54 |
| Target hit | 2026-04-21 15:20:00 | 282.15 | 278.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:15:00 | 297.35 | 295.66 | 0.00 | ORB-long ORB[293.10,297.30] vol=1.9x ATR=1.20 |
| Stop hit — per-position SL triggered | 2026-04-29 11:05:00 | 296.15 | 295.82 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 11:15:00 | 313.95 | 312.03 | 0.00 | ORB-long ORB[310.25,313.65] vol=3.4x ATR=0.76 |
| Stop hit — per-position SL triggered | 2026-05-07 11:20:00 | 313.19 | 312.06 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 10:30:00 | 306.15 | 2026-02-10 10:40:00 | 305.11 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-11 09:40:00 | 308.35 | 2026-02-11 09:50:00 | 310.29 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-02-11 09:40:00 | 308.35 | 2026-02-11 10:10:00 | 308.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-17 10:50:00 | 299.50 | 2026-02-17 11:15:00 | 300.48 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-18 09:35:00 | 308.95 | 2026-02-18 09:45:00 | 308.09 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-19 09:55:00 | 307.60 | 2026-02-19 10:35:00 | 306.44 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-02-19 09:55:00 | 307.60 | 2026-02-19 10:50:00 | 307.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-20 10:45:00 | 307.20 | 2026-02-20 11:05:00 | 306.36 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-02-24 09:45:00 | 304.80 | 2026-02-24 10:00:00 | 303.71 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-25 10:40:00 | 301.80 | 2026-02-25 10:55:00 | 302.69 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-05 11:00:00 | 268.25 | 2026-03-05 11:25:00 | 266.99 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-03-05 11:00:00 | 268.25 | 2026-03-05 14:40:00 | 267.40 | TARGET_HIT | 0.50 | 0.32% |
| SELL | retest1 | 2026-03-06 10:45:00 | 264.95 | 2026-03-06 11:05:00 | 265.88 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-03-10 09:45:00 | 264.35 | 2026-03-10 10:00:00 | 262.82 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2026-03-17 10:25:00 | 254.30 | 2026-03-17 10:35:00 | 253.16 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-03-23 09:40:00 | 253.15 | 2026-03-23 09:50:00 | 251.73 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2026-04-01 11:05:00 | 256.50 | 2026-04-01 11:30:00 | 257.48 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-08 10:40:00 | 266.50 | 2026-04-08 10:55:00 | 268.17 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-04-08 10:40:00 | 266.50 | 2026-04-08 15:20:00 | 269.95 | TARGET_HIT | 0.50 | 1.29% |
| SELL | retest1 | 2026-04-10 10:05:00 | 264.10 | 2026-04-10 10:15:00 | 264.98 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-04-15 09:30:00 | 271.40 | 2026-04-15 10:10:00 | 272.54 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-04-21 10:00:00 | 274.25 | 2026-04-21 10:35:00 | 275.54 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-04-21 10:00:00 | 274.25 | 2026-04-21 15:20:00 | 282.15 | TARGET_HIT | 0.50 | 2.88% |
| BUY | retest1 | 2026-04-29 10:15:00 | 297.35 | 2026-04-29 11:05:00 | 296.15 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-05-07 11:15:00 | 313.95 | 2026-05-07 11:20:00 | 313.19 | STOP_HIT | 1.00 | -0.24% |
