# Crompton Greaves Consumer Electricals Ltd. (CROMPTON)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 293.55
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
| ENTRY1 | 17 |
| ENTRY2 | 0 |
| PARTIAL | 12 |
| TARGET_HIT | 6 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 11
- **Target hits / Stop hits / Partials:** 6 / 11 / 12
- **Avg / median % per leg:** 0.44% / 0.44%
- **Sum % (uncompounded):** 12.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 12 | 70.6% | 4 | 5 | 8 | 0.59% | 10.0% |
| BUY @ 2nd Alert (retest1) | 17 | 12 | 70.6% | 4 | 5 | 8 | 0.59% | 10.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 6 | 50.0% | 2 | 6 | 4 | 0.23% | 2.8% |
| SELL @ 2nd Alert (retest1) | 12 | 6 | 50.0% | 2 | 6 | 4 | 0.23% | 2.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 29 | 18 | 62.1% | 6 | 11 | 12 | 0.44% | 12.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:00:00 | 270.49 | 270.76 | 0.00 | ORB-short ORB[270.67,273.00] vol=2.5x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:15:00 | 269.31 | 270.71 | 0.00 | T1 1.5R @ 269.31 |
| Target hit | 2026-02-19 15:20:00 | 265.86 | 268.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 10:10:00 | 262.17 | 263.21 | 0.00 | ORB-short ORB[262.75,264.91] vol=1.7x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 10:25:00 | 261.07 | 262.82 | 0.00 | T1 1.5R @ 261.07 |
| Stop hit — per-position SL triggered | 2026-02-24 11:25:00 | 262.17 | 262.17 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:40:00 | 265.04 | 263.17 | 0.00 | ORB-long ORB[261.82,264.54] vol=1.9x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 12:10:00 | 266.23 | 263.96 | 0.00 | T1 1.5R @ 266.23 |
| Stop hit — per-position SL triggered | 2026-02-25 12:55:00 | 265.04 | 264.41 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:45:00 | 257.67 | 260.12 | 0.00 | ORB-short ORB[261.04,263.73] vol=2.1x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:35:00 | 256.46 | 259.27 | 0.00 | T1 1.5R @ 256.46 |
| Stop hit — per-position SL triggered | 2026-02-27 12:25:00 | 257.67 | 258.82 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 247.75 | 249.75 | 0.00 | ORB-short ORB[248.75,251.05] vol=3.0x ATR=0.70 |
| Stop hit — per-position SL triggered | 2026-03-06 11:00:00 | 248.45 | 249.46 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:55:00 | 250.25 | 248.34 | 0.00 | ORB-long ORB[246.35,249.75] vol=1.6x ATR=1.33 |
| Stop hit — per-position SL triggered | 2026-03-17 10:40:00 | 248.92 | 249.12 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:45:00 | 243.83 | 241.89 | 0.00 | ORB-long ORB[240.00,242.80] vol=2.2x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 10:55:00 | 245.90 | 242.96 | 0.00 | T1 1.5R @ 245.90 |
| Target hit | 2026-04-08 15:20:00 | 246.21 | 245.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2026-04-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 10:45:00 | 241.76 | 242.52 | 0.00 | ORB-short ORB[243.45,246.33] vol=1.8x ATR=0.85 |
| Stop hit — per-position SL triggered | 2026-04-09 11:00:00 | 242.61 | 242.49 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 10:10:00 | 240.86 | 239.17 | 0.00 | ORB-long ORB[237.60,240.59] vol=2.8x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 11:15:00 | 242.53 | 239.85 | 0.00 | T1 1.5R @ 242.53 |
| Stop hit — per-position SL triggered | 2026-04-10 14:25:00 | 240.86 | 240.64 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:15:00 | 245.30 | 243.63 | 0.00 | ORB-long ORB[241.49,244.80] vol=2.4x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 10:30:00 | 246.68 | 243.96 | 0.00 | T1 1.5R @ 246.68 |
| Stop hit — per-position SL triggered | 2026-04-15 10:45:00 | 245.30 | 244.34 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 10:30:00 | 259.03 | 261.13 | 0.00 | ORB-short ORB[260.12,263.47] vol=2.6x ATR=1.29 |
| Stop hit — per-position SL triggered | 2026-04-17 10:45:00 | 260.32 | 260.99 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 11:00:00 | 261.05 | 263.61 | 0.00 | ORB-short ORB[262.00,265.57] vol=1.7x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 11:40:00 | 259.95 | 262.92 | 0.00 | T1 1.5R @ 259.95 |
| Target hit | 2026-04-21 15:20:00 | 258.67 | 260.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2026-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:30:00 | 252.16 | 253.48 | 0.00 | ORB-short ORB[253.35,257.00] vol=6.7x ATR=1.07 |
| Stop hit — per-position SL triggered | 2026-04-24 09:40:00 | 253.23 | 253.34 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:45:00 | 261.53 | 258.48 | 0.00 | ORB-long ORB[257.71,260.80] vol=2.1x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:50:00 | 263.04 | 259.09 | 0.00 | T1 1.5R @ 263.04 |
| Target hit | 2026-04-28 15:20:00 | 269.45 | 266.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2026-04-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:30:00 | 274.16 | 271.32 | 0.00 | ORB-long ORB[269.63,273.57] vol=3.0x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 09:50:00 | 276.40 | 272.40 | 0.00 | T1 1.5R @ 276.40 |
| Target hit | 2026-04-29 14:25:00 | 276.05 | 276.18 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — BUY (started 2026-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:40:00 | 280.40 | 278.28 | 0.00 | ORB-long ORB[276.10,279.75] vol=2.1x ATR=1.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:55:00 | 282.43 | 280.47 | 0.00 | T1 1.5R @ 282.43 |
| Target hit | 2026-05-06 11:20:00 | 281.45 | 281.85 | 0.00 | Trail-exit close<VWAP |

### Cycle 17 — BUY (started 2026-05-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:50:00 | 288.70 | 286.54 | 0.00 | ORB-long ORB[283.25,287.05] vol=1.6x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 10:05:00 | 290.98 | 287.84 | 0.00 | T1 1.5R @ 290.98 |
| Stop hit — per-position SL triggered | 2026-05-07 11:05:00 | 288.70 | 289.24 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-19 11:00:00 | 270.49 | 2026-02-19 11:15:00 | 269.31 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-02-19 11:00:00 | 270.49 | 2026-02-19 15:20:00 | 265.86 | TARGET_HIT | 0.50 | 1.71% |
| SELL | retest1 | 2026-02-24 10:10:00 | 262.17 | 2026-02-24 10:25:00 | 261.07 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-02-24 10:10:00 | 262.17 | 2026-02-24 11:25:00 | 262.17 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 10:40:00 | 265.04 | 2026-02-25 12:10:00 | 266.23 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-25 10:40:00 | 265.04 | 2026-02-25 12:55:00 | 265.04 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 10:45:00 | 257.67 | 2026-02-27 11:35:00 | 256.46 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-02-27 10:45:00 | 257.67 | 2026-02-27 12:25:00 | 257.67 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-06 10:45:00 | 247.75 | 2026-03-06 11:00:00 | 248.45 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-03-17 09:55:00 | 250.25 | 2026-03-17 10:40:00 | 248.92 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2026-04-08 09:45:00 | 243.83 | 2026-04-08 10:55:00 | 245.90 | PARTIAL | 0.50 | 0.85% |
| BUY | retest1 | 2026-04-08 09:45:00 | 243.83 | 2026-04-08 15:20:00 | 246.21 | TARGET_HIT | 0.50 | 0.98% |
| SELL | retest1 | 2026-04-09 10:45:00 | 241.76 | 2026-04-09 11:00:00 | 242.61 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-10 10:10:00 | 240.86 | 2026-04-10 11:15:00 | 242.53 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2026-04-10 10:10:00 | 240.86 | 2026-04-10 14:25:00 | 240.86 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-15 10:15:00 | 245.30 | 2026-04-15 10:30:00 | 246.68 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-04-15 10:15:00 | 245.30 | 2026-04-15 10:45:00 | 245.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-17 10:30:00 | 259.03 | 2026-04-17 10:45:00 | 260.32 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2026-04-21 11:00:00 | 261.05 | 2026-04-21 11:40:00 | 259.95 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-04-21 11:00:00 | 261.05 | 2026-04-21 15:20:00 | 258.67 | TARGET_HIT | 0.50 | 0.91% |
| SELL | retest1 | 2026-04-24 09:30:00 | 252.16 | 2026-04-24 09:40:00 | 253.23 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-28 10:45:00 | 261.53 | 2026-04-28 10:50:00 | 263.04 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-04-28 10:45:00 | 261.53 | 2026-04-28 15:20:00 | 269.45 | TARGET_HIT | 0.50 | 3.03% |
| BUY | retest1 | 2026-04-29 09:30:00 | 274.16 | 2026-04-29 09:50:00 | 276.40 | PARTIAL | 0.50 | 0.82% |
| BUY | retest1 | 2026-04-29 09:30:00 | 274.16 | 2026-04-29 14:25:00 | 276.05 | TARGET_HIT | 0.50 | 0.69% |
| BUY | retest1 | 2026-05-06 09:40:00 | 280.40 | 2026-05-06 09:55:00 | 282.43 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2026-05-06 09:40:00 | 280.40 | 2026-05-06 11:20:00 | 281.45 | TARGET_HIT | 0.50 | 0.37% |
| BUY | retest1 | 2026-05-07 09:50:00 | 288.70 | 2026-05-07 10:05:00 | 290.98 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2026-05-07 09:50:00 | 288.70 | 2026-05-07 11:05:00 | 288.70 | STOP_HIT | 0.50 | 0.00% |
