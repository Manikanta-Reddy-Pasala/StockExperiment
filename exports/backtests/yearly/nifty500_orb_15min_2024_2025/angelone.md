# Angel One Ltd. (ANGELONE)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-06-04 15:25:00 (19758 bars)
- **Last close:** 312.01
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
| ENTRY1 | 32 |
| ENTRY2 | 0 |
| PARTIAL | 14 |
| TARGET_HIT | 8 |
| STOP_HIT | 24 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 46 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 24
- **Target hits / Stop hits / Partials:** 8 / 24 / 14
- **Avg / median % per leg:** 0.44% / 0.00%
- **Sum % (uncompounded):** 20.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 12 | 60.0% | 4 | 8 | 8 | 0.58% | 11.6% |
| BUY @ 2nd Alert (retest1) | 20 | 12 | 60.0% | 4 | 8 | 8 | 0.58% | 11.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 26 | 10 | 38.5% | 4 | 16 | 6 | 0.33% | 8.6% |
| SELL @ 2nd Alert (retest1) | 26 | 10 | 38.5% | 4 | 16 | 6 | 0.33% | 8.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 46 | 22 | 47.8% | 8 | 24 | 14 | 0.44% | 20.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 09:30:00 | 248.25 | 249.77 | 0.00 | ORB-short ORB[248.70,250.80] vol=1.7x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 09:35:00 | 247.10 | 249.48 | 0.00 | T1 1.5R @ 247.10 |
| Stop hit — per-position SL triggered | 2024-05-30 09:40:00 | 248.25 | 249.41 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-06-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 09:55:00 | 263.37 | 262.01 | 0.00 | ORB-long ORB[260.21,262.60] vol=1.8x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 10:35:00 | 264.79 | 262.51 | 0.00 | T1 1.5R @ 264.79 |
| Stop hit — per-position SL triggered | 2024-06-11 11:00:00 | 263.37 | 262.66 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-06-25 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 10:20:00 | 255.79 | 257.49 | 0.00 | ORB-short ORB[257.91,259.83] vol=1.7x ATR=0.62 |
| Stop hit — per-position SL triggered | 2024-06-25 11:00:00 | 256.41 | 257.18 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-06-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 10:40:00 | 254.60 | 255.98 | 0.00 | ORB-short ORB[256.01,258.71] vol=2.2x ATR=0.84 |
| Stop hit — per-position SL triggered | 2024-06-26 14:00:00 | 255.44 | 255.04 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 09:45:00 | 264.27 | 262.86 | 0.00 | ORB-long ORB[260.30,263.93] vol=2.3x ATR=1.23 |
| Stop hit — per-position SL triggered | 2024-06-28 10:05:00 | 263.04 | 263.00 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-07-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 11:10:00 | 236.20 | 237.16 | 0.00 | ORB-short ORB[236.53,238.99] vol=2.1x ATR=0.59 |
| Stop hit — per-position SL triggered | 2024-07-08 11:20:00 | 236.79 | 237.14 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-08-01 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 10:05:00 | 221.46 | 219.15 | 0.00 | ORB-long ORB[216.91,219.99] vol=2.0x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 10:35:00 | 223.19 | 220.23 | 0.00 | T1 1.5R @ 223.19 |
| Stop hit — per-position SL triggered | 2024-08-01 10:50:00 | 221.46 | 220.50 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-08-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 09:30:00 | 255.25 | 257.25 | 0.00 | ORB-short ORB[257.11,259.90] vol=2.6x ATR=1.25 |
| Stop hit — per-position SL triggered | 2024-08-30 09:40:00 | 256.50 | 256.90 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-09-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 09:40:00 | 253.55 | 254.51 | 0.00 | ORB-short ORB[253.71,257.38] vol=1.8x ATR=0.81 |
| Stop hit — per-position SL triggered | 2024-09-05 09:50:00 | 254.36 | 254.43 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-09-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-12 09:30:00 | 238.71 | 240.01 | 0.00 | ORB-short ORB[239.22,241.44] vol=1.5x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 09:35:00 | 237.52 | 239.63 | 0.00 | T1 1.5R @ 237.52 |
| Stop hit — per-position SL triggered | 2024-09-12 09:40:00 | 238.71 | 239.55 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-09-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:50:00 | 259.33 | 262.56 | 0.00 | ORB-short ORB[262.21,265.40] vol=1.7x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:10:00 | 256.89 | 261.55 | 0.00 | T1 1.5R @ 256.89 |
| Target hit | 2024-09-19 15:20:00 | 254.10 | 255.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2024-09-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 09:30:00 | 259.24 | 260.56 | 0.00 | ORB-short ORB[259.60,263.20] vol=1.5x ATR=1.09 |
| Stop hit — per-position SL triggered | 2024-09-24 09:35:00 | 260.33 | 260.52 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-10-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 11:00:00 | 257.80 | 261.11 | 0.00 | ORB-short ORB[260.00,263.44] vol=1.6x ATR=0.97 |
| Stop hit — per-position SL triggered | 2024-10-10 11:10:00 | 258.77 | 260.91 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-10-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 10:10:00 | 264.40 | 262.44 | 0.00 | ORB-long ORB[260.90,263.44] vol=3.7x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 10:25:00 | 266.16 | 263.33 | 0.00 | T1 1.5R @ 266.16 |
| Stop hit — per-position SL triggered | 2024-10-11 11:30:00 | 264.40 | 264.53 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-10-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-21 09:30:00 | 316.66 | 313.28 | 0.00 | ORB-long ORB[310.40,314.80] vol=2.4x ATR=1.99 |
| Stop hit — per-position SL triggered | 2024-10-21 09:35:00 | 314.67 | 313.56 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-10-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 09:30:00 | 299.19 | 301.50 | 0.00 | ORB-short ORB[300.59,304.35] vol=1.9x ATR=1.74 |
| Stop hit — per-position SL triggered | 2024-10-22 09:35:00 | 300.93 | 301.27 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-11-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-05 09:30:00 | 286.53 | 290.20 | 0.00 | ORB-short ORB[288.71,293.00] vol=2.6x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 09:45:00 | 284.06 | 288.62 | 0.00 | T1 1.5R @ 284.06 |
| Stop hit — per-position SL triggered | 2024-11-05 09:55:00 | 286.53 | 288.17 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-11-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 09:35:00 | 281.60 | 277.34 | 0.00 | ORB-long ORB[273.31,277.20] vol=6.1x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-25 09:40:00 | 284.01 | 279.13 | 0.00 | T1 1.5R @ 284.01 |
| Target hit | 2024-11-25 10:30:00 | 281.80 | 283.04 | 0.00 | Trail-exit close<VWAP |

### Cycle 19 — BUY (started 2024-12-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 09:30:00 | 298.31 | 296.37 | 0.00 | ORB-long ORB[293.61,297.82] vol=2.8x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 09:35:00 | 299.77 | 298.07 | 0.00 | T1 1.5R @ 299.77 |
| Target hit | 2024-12-03 15:20:00 | 308.00 | 303.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2024-12-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 09:30:00 | 308.83 | 306.34 | 0.00 | ORB-long ORB[303.95,307.15] vol=2.6x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-05 09:35:00 | 310.53 | 307.40 | 0.00 | T1 1.5R @ 310.53 |
| Stop hit — per-position SL triggered | 2024-12-05 09:40:00 | 308.83 | 307.60 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-12-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 09:50:00 | 301.04 | 305.09 | 0.00 | ORB-short ORB[305.10,308.46] vol=1.8x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 10:00:00 | 299.02 | 303.22 | 0.00 | T1 1.5R @ 299.02 |
| Target hit | 2024-12-20 15:20:00 | 288.11 | 295.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2024-12-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 09:40:00 | 290.14 | 288.06 | 0.00 | ORB-long ORB[285.80,288.90] vol=1.7x ATR=1.29 |
| Stop hit — per-position SL triggered | 2024-12-24 09:45:00 | 288.85 | 288.22 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-12-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:30:00 | 289.46 | 291.20 | 0.00 | ORB-short ORB[290.50,293.50] vol=1.9x ATR=1.15 |
| Stop hit — per-position SL triggered | 2024-12-26 09:40:00 | 290.61 | 291.04 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-12-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-31 09:40:00 | 284.99 | 288.68 | 0.00 | ORB-short ORB[287.80,291.92] vol=1.6x ATR=1.54 |
| Stop hit — per-position SL triggered | 2024-12-31 10:15:00 | 286.53 | 287.22 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-01-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 09:30:00 | 297.02 | 295.90 | 0.00 | ORB-long ORB[293.26,297.00] vol=3.2x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 09:40:00 | 298.70 | 296.95 | 0.00 | T1 1.5R @ 298.70 |
| Target hit | 2025-01-01 11:35:00 | 300.28 | 300.72 | 0.00 | Trail-exit close<VWAP |

### Cycle 26 — SELL (started 2025-01-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 10:05:00 | 296.12 | 298.24 | 0.00 | ORB-short ORB[296.40,299.90] vol=2.0x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 10:10:00 | 294.31 | 297.76 | 0.00 | T1 1.5R @ 294.31 |
| Target hit | 2025-01-03 15:20:00 | 285.34 | 289.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — SELL (started 2025-01-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 09:30:00 | 263.80 | 265.03 | 0.00 | ORB-short ORB[264.50,266.82] vol=1.8x ATR=1.34 |
| Stop hit — per-position SL triggered | 2025-01-09 09:35:00 | 265.14 | 264.96 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-01-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-17 11:00:00 | 245.77 | 247.59 | 0.00 | ORB-short ORB[246.00,249.00] vol=1.8x ATR=1.16 |
| Target hit | 2025-01-17 15:20:00 | 245.42 | 246.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — BUY (started 2025-01-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 09:30:00 | 222.27 | 220.31 | 0.00 | ORB-long ORB[218.32,221.35] vol=1.9x ATR=1.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 09:40:00 | 224.48 | 221.69 | 0.00 | T1 1.5R @ 224.48 |
| Target hit | 2025-01-29 15:20:00 | 230.62 | 225.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — SELL (started 2025-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-30 09:30:00 | 228.34 | 230.17 | 0.00 | ORB-short ORB[229.02,232.14] vol=1.5x ATR=1.18 |
| Stop hit — per-position SL triggered | 2025-01-30 09:35:00 | 229.52 | 230.11 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-03-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:40:00 | 227.66 | 230.24 | 0.00 | ORB-short ORB[229.02,232.45] vol=2.0x ATR=1.30 |
| Stop hit — per-position SL triggered | 2025-03-26 09:50:00 | 228.96 | 229.80 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-05-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:45:00 | 235.10 | 232.50 | 0.00 | ORB-long ORB[230.18,233.29] vol=2.1x ATR=1.11 |
| Stop hit — per-position SL triggered | 2025-05-05 10:00:00 | 233.99 | 232.80 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-30 09:30:00 | 248.25 | 2024-05-30 09:35:00 | 247.10 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-05-30 09:30:00 | 248.25 | 2024-05-30 09:40:00 | 248.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-11 09:55:00 | 263.37 | 2024-06-11 10:35:00 | 264.79 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-06-11 09:55:00 | 263.37 | 2024-06-11 11:00:00 | 263.37 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-25 10:20:00 | 255.79 | 2024-06-25 11:00:00 | 256.41 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-06-26 10:40:00 | 254.60 | 2024-06-26 14:00:00 | 255.44 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-06-28 09:45:00 | 264.27 | 2024-06-28 10:05:00 | 263.04 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-07-08 11:10:00 | 236.20 | 2024-07-08 11:20:00 | 236.79 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-08-01 10:05:00 | 221.46 | 2024-08-01 10:35:00 | 223.19 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2024-08-01 10:05:00 | 221.46 | 2024-08-01 10:50:00 | 221.46 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-30 09:30:00 | 255.25 | 2024-08-30 09:40:00 | 256.50 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-09-05 09:40:00 | 253.55 | 2024-09-05 09:50:00 | 254.36 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-09-12 09:30:00 | 238.71 | 2024-09-12 09:35:00 | 237.52 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-09-12 09:30:00 | 238.71 | 2024-09-12 09:40:00 | 238.71 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-19 09:50:00 | 259.33 | 2024-09-19 10:10:00 | 256.89 | PARTIAL | 0.50 | 0.94% |
| SELL | retest1 | 2024-09-19 09:50:00 | 259.33 | 2024-09-19 15:20:00 | 254.10 | TARGET_HIT | 0.50 | 2.02% |
| SELL | retest1 | 2024-09-24 09:30:00 | 259.24 | 2024-09-24 09:35:00 | 260.33 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-10-10 11:00:00 | 257.80 | 2024-10-10 11:10:00 | 258.77 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-10-11 10:10:00 | 264.40 | 2024-10-11 10:25:00 | 266.16 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2024-10-11 10:10:00 | 264.40 | 2024-10-11 11:30:00 | 264.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-21 09:30:00 | 316.66 | 2024-10-21 09:35:00 | 314.67 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest1 | 2024-10-22 09:30:00 | 299.19 | 2024-10-22 09:35:00 | 300.93 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2024-11-05 09:30:00 | 286.53 | 2024-11-05 09:45:00 | 284.06 | PARTIAL | 0.50 | 0.86% |
| SELL | retest1 | 2024-11-05 09:30:00 | 286.53 | 2024-11-05 09:55:00 | 286.53 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-25 09:35:00 | 281.60 | 2024-11-25 09:40:00 | 284.01 | PARTIAL | 0.50 | 0.86% |
| BUY | retest1 | 2024-11-25 09:35:00 | 281.60 | 2024-11-25 10:30:00 | 281.80 | TARGET_HIT | 0.50 | 0.07% |
| BUY | retest1 | 2024-12-03 09:30:00 | 298.31 | 2024-12-03 09:35:00 | 299.77 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-12-03 09:30:00 | 298.31 | 2024-12-03 15:20:00 | 308.00 | TARGET_HIT | 0.50 | 3.25% |
| BUY | retest1 | 2024-12-05 09:30:00 | 308.83 | 2024-12-05 09:35:00 | 310.53 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-12-05 09:30:00 | 308.83 | 2024-12-05 09:40:00 | 308.83 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-20 09:50:00 | 301.04 | 2024-12-20 10:00:00 | 299.02 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2024-12-20 09:50:00 | 301.04 | 2024-12-20 15:20:00 | 288.11 | TARGET_HIT | 0.50 | 4.30% |
| BUY | retest1 | 2024-12-24 09:40:00 | 290.14 | 2024-12-24 09:45:00 | 288.85 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-12-26 09:30:00 | 289.46 | 2024-12-26 09:40:00 | 290.61 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-12-31 09:40:00 | 284.99 | 2024-12-31 10:15:00 | 286.53 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2025-01-01 09:30:00 | 297.02 | 2025-01-01 09:40:00 | 298.70 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-01-01 09:30:00 | 297.02 | 2025-01-01 11:35:00 | 300.28 | TARGET_HIT | 0.50 | 1.10% |
| SELL | retest1 | 2025-01-03 10:05:00 | 296.12 | 2025-01-03 10:10:00 | 294.31 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-01-03 10:05:00 | 296.12 | 2025-01-03 15:20:00 | 285.34 | TARGET_HIT | 0.50 | 3.64% |
| SELL | retest1 | 2025-01-09 09:30:00 | 263.80 | 2025-01-09 09:35:00 | 265.14 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2025-01-17 11:00:00 | 245.77 | 2025-01-17 15:20:00 | 245.42 | TARGET_HIT | 1.00 | 0.14% |
| BUY | retest1 | 2025-01-29 09:30:00 | 222.27 | 2025-01-29 09:40:00 | 224.48 | PARTIAL | 0.50 | 0.99% |
| BUY | retest1 | 2025-01-29 09:30:00 | 222.27 | 2025-01-29 15:20:00 | 230.62 | TARGET_HIT | 0.50 | 3.76% |
| SELL | retest1 | 2025-01-30 09:30:00 | 228.34 | 2025-01-30 09:35:00 | 229.52 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2025-03-26 09:40:00 | 227.66 | 2025-03-26 09:50:00 | 228.96 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2025-05-05 09:45:00 | 235.10 | 2025-05-05 10:00:00 | 233.99 | STOP_HIT | 1.00 | -0.47% |
