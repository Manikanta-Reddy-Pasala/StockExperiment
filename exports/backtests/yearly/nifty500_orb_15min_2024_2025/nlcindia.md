# NLC India Ltd. (NLCINDIA)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2024-12-06 15:25:00 (10683 bars)
- **Last close:** 266.30
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
| ENTRY1 | 37 |
| ENTRY2 | 0 |
| PARTIAL | 18 |
| TARGET_HIT | 10 |
| STOP_HIT | 27 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 27
- **Target hits / Stop hits / Partials:** 10 / 27 / 18
- **Avg / median % per leg:** 0.27% / 0.11%
- **Sum % (uncompounded):** 14.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 13 | 44.8% | 5 | 16 | 8 | 0.23% | 6.5% |
| BUY @ 2nd Alert (retest1) | 29 | 13 | 44.8% | 5 | 16 | 8 | 0.23% | 6.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 26 | 15 | 57.7% | 5 | 11 | 10 | 0.32% | 8.3% |
| SELL @ 2nd Alert (retest1) | 26 | 15 | 57.7% | 5 | 11 | 10 | 0.32% | 8.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 55 | 28 | 50.9% | 10 | 27 | 18 | 0.27% | 14.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 10:50:00 | 243.35 | 245.57 | 0.00 | ORB-short ORB[245.90,249.15] vol=2.0x ATR=1.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-22 13:15:00 | 241.31 | 244.76 | 0.00 | T1 1.5R @ 241.31 |
| Target hit | 2024-05-22 15:20:00 | 241.20 | 243.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2024-05-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 09:50:00 | 225.15 | 226.52 | 0.00 | ORB-short ORB[226.70,229.90] vol=1.7x ATR=1.26 |
| Stop hit — per-position SL triggered | 2024-05-28 10:10:00 | 226.41 | 226.28 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-06-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 09:30:00 | 220.77 | 222.64 | 0.00 | ORB-short ORB[221.30,224.20] vol=1.6x ATR=1.15 |
| Stop hit — per-position SL triggered | 2024-06-10 09:40:00 | 221.92 | 222.22 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-06-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 09:30:00 | 234.21 | 236.11 | 0.00 | ORB-short ORB[235.56,238.10] vol=1.8x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 11:10:00 | 232.75 | 234.77 | 0.00 | T1 1.5R @ 232.75 |
| Target hit | 2024-06-13 15:20:00 | 233.20 | 233.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2024-06-14 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:50:00 | 234.65 | 233.17 | 0.00 | ORB-long ORB[231.40,233.94] vol=1.8x ATR=0.78 |
| Stop hit — per-position SL triggered | 2024-06-14 09:55:00 | 233.87 | 233.34 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-18 09:55:00 | 230.79 | 231.97 | 0.00 | ORB-short ORB[231.38,233.80] vol=1.5x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-18 10:15:00 | 229.57 | 231.62 | 0.00 | T1 1.5R @ 229.57 |
| Stop hit — per-position SL triggered | 2024-06-18 10:30:00 | 230.79 | 231.41 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 09:30:00 | 239.95 | 238.18 | 0.00 | ORB-long ORB[235.02,238.45] vol=4.7x ATR=1.25 |
| Stop hit — per-position SL triggered | 2024-06-20 09:50:00 | 238.70 | 238.66 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-24 09:30:00 | 231.16 | 232.31 | 0.00 | ORB-short ORB[231.63,234.18] vol=1.7x ATR=1.00 |
| Stop hit — per-position SL triggered | 2024-06-24 09:45:00 | 232.16 | 232.04 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 11:15:00 | 231.90 | 232.68 | 0.00 | ORB-short ORB[233.68,235.95] vol=15.9x ATR=0.71 |
| Stop hit — per-position SL triggered | 2024-06-25 11:20:00 | 232.61 | 232.68 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:40:00 | 236.80 | 235.72 | 0.00 | ORB-long ORB[233.80,236.75] vol=6.7x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 11:35:00 | 238.45 | 236.32 | 0.00 | T1 1.5R @ 238.45 |
| Target hit | 2024-06-26 13:00:00 | 241.42 | 241.87 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2024-06-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 09:30:00 | 247.40 | 245.72 | 0.00 | ORB-long ORB[243.19,246.00] vol=1.9x ATR=1.61 |
| Stop hit — per-position SL triggered | 2024-06-28 10:00:00 | 245.79 | 247.03 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 09:40:00 | 245.23 | 243.51 | 0.00 | ORB-long ORB[241.41,245.00] vol=1.6x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 10:00:00 | 247.30 | 246.84 | 0.00 | T1 1.5R @ 247.30 |
| Target hit | 2024-07-02 12:50:00 | 255.69 | 256.23 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 11:15:00 | 265.27 | 268.11 | 0.00 | ORB-short ORB[267.33,271.18] vol=2.7x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 11:30:00 | 263.20 | 267.64 | 0.00 | T1 1.5R @ 263.20 |
| Stop hit — per-position SL triggered | 2024-07-23 11:40:00 | 265.27 | 267.36 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:30:00 | 284.91 | 283.31 | 0.00 | ORB-long ORB[281.50,284.60] vol=2.2x ATR=1.71 |
| Stop hit — per-position SL triggered | 2024-07-26 09:55:00 | 283.20 | 283.79 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-29 09:30:00 | 288.45 | 290.78 | 0.00 | ORB-short ORB[289.45,293.57] vol=1.6x ATR=1.81 |
| Stop hit — per-position SL triggered | 2024-07-29 09:55:00 | 290.26 | 290.45 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-08-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 09:35:00 | 268.60 | 266.71 | 0.00 | ORB-long ORB[264.60,268.40] vol=2.2x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 09:45:00 | 270.83 | 267.74 | 0.00 | T1 1.5R @ 270.83 |
| Stop hit — per-position SL triggered | 2024-08-09 10:10:00 | 268.60 | 268.44 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-08-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-16 09:30:00 | 259.45 | 260.32 | 0.00 | ORB-short ORB[259.50,262.20] vol=1.7x ATR=1.13 |
| Stop hit — per-position SL triggered | 2024-08-16 09:40:00 | 260.58 | 260.22 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 11:15:00 | 274.00 | 271.48 | 0.00 | ORB-long ORB[269.45,272.50] vol=1.8x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 13:35:00 | 275.19 | 272.52 | 0.00 | T1 1.5R @ 275.19 |
| Stop hit — per-position SL triggered | 2024-08-23 13:50:00 | 274.00 | 272.70 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-08-26 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 10:25:00 | 274.75 | 276.38 | 0.00 | ORB-short ORB[276.55,279.00] vol=2.0x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 10:55:00 | 273.45 | 275.90 | 0.00 | T1 1.5R @ 273.45 |
| Stop hit — per-position SL triggered | 2024-08-26 11:05:00 | 274.75 | 275.85 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-09-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 11:00:00 | 269.75 | 271.75 | 0.00 | ORB-short ORB[271.10,273.90] vol=1.5x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 14:05:00 | 268.61 | 270.71 | 0.00 | T1 1.5R @ 268.61 |
| Stop hit — per-position SL triggered | 2024-09-18 14:40:00 | 269.75 | 270.41 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-09-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 11:10:00 | 262.00 | 266.44 | 0.00 | ORB-short ORB[268.35,271.75] vol=4.1x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 11:20:00 | 260.67 | 265.95 | 0.00 | T1 1.5R @ 260.67 |
| Stop hit — per-position SL triggered | 2024-09-19 12:15:00 | 262.00 | 263.91 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-09-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 11:10:00 | 266.80 | 265.60 | 0.00 | ORB-long ORB[263.10,265.35] vol=3.7x ATR=0.89 |
| Stop hit — per-position SL triggered | 2024-09-20 11:50:00 | 265.91 | 265.67 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-09-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 11:10:00 | 271.85 | 269.72 | 0.00 | ORB-long ORB[268.30,271.50] vol=3.7x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 12:50:00 | 273.40 | 270.67 | 0.00 | T1 1.5R @ 273.40 |
| Target hit | 2024-09-23 14:25:00 | 272.15 | 272.26 | 0.00 | Trail-exit close<VWAP |

### Cycle 24 — BUY (started 2024-09-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 09:40:00 | 276.80 | 275.00 | 0.00 | ORB-long ORB[273.00,275.75] vol=5.2x ATR=0.84 |
| Stop hit — per-position SL triggered | 2024-09-24 09:45:00 | 275.96 | 275.37 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-09-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 09:40:00 | 291.05 | 288.93 | 0.00 | ORB-long ORB[286.20,290.50] vol=2.1x ATR=1.91 |
| Stop hit — per-position SL triggered | 2024-09-25 10:00:00 | 289.14 | 289.14 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:55:00 | 280.75 | 279.78 | 0.00 | ORB-long ORB[276.15,280.30] vol=2.3x ATR=1.02 |
| Stop hit — per-position SL triggered | 2024-09-27 11:55:00 | 279.73 | 279.99 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-10-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 11:05:00 | 277.30 | 275.55 | 0.00 | ORB-long ORB[273.10,277.00] vol=2.8x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-09 11:10:00 | 278.88 | 275.83 | 0.00 | T1 1.5R @ 278.88 |
| Stop hit — per-position SL triggered | 2024-10-09 11:40:00 | 277.30 | 276.41 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-10-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 09:35:00 | 269.75 | 268.16 | 0.00 | ORB-long ORB[266.30,268.15] vol=4.2x ATR=0.94 |
| Stop hit — per-position SL triggered | 2024-10-15 09:50:00 | 268.81 | 268.66 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 10:15:00 | 270.15 | 268.48 | 0.00 | ORB-long ORB[267.30,269.20] vol=2.8x ATR=1.00 |
| Stop hit — per-position SL triggered | 2024-10-16 12:15:00 | 269.15 | 269.18 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-10-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 10:55:00 | 263.95 | 264.70 | 0.00 | ORB-short ORB[265.10,268.10] vol=5.5x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:25:00 | 262.66 | 264.57 | 0.00 | T1 1.5R @ 262.66 |
| Target hit | 2024-10-17 15:20:00 | 261.30 | 263.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — SELL (started 2024-10-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:35:00 | 260.85 | 262.18 | 0.00 | ORB-short ORB[261.70,265.05] vol=1.6x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 09:50:00 | 259.73 | 261.66 | 0.00 | T1 1.5R @ 259.73 |
| Target hit | 2024-10-21 11:10:00 | 260.55 | 260.52 | 0.00 | Trail-exit close>VWAP |

### Cycle 32 — BUY (started 2024-10-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 09:45:00 | 253.30 | 251.58 | 0.00 | ORB-long ORB[249.45,252.80] vol=1.5x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-31 10:25:00 | 255.10 | 253.35 | 0.00 | T1 1.5R @ 255.10 |
| Target hit | 2024-10-31 12:40:00 | 254.55 | 254.92 | 0.00 | Trail-exit close<VWAP |

### Cycle 33 — SELL (started 2024-11-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 09:50:00 | 245.50 | 247.51 | 0.00 | ORB-short ORB[247.05,250.40] vol=1.6x ATR=1.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 12:10:00 | 243.86 | 245.99 | 0.00 | T1 1.5R @ 243.86 |
| Target hit | 2024-11-12 15:20:00 | 238.30 | 243.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — BUY (started 2024-11-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 10:05:00 | 264.85 | 263.70 | 0.00 | ORB-long ORB[261.10,264.35] vol=3.5x ATR=1.76 |
| Stop hit — per-position SL triggered | 2024-11-25 13:00:00 | 263.09 | 263.97 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-11-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 11:10:00 | 261.25 | 260.09 | 0.00 | ORB-long ORB[257.75,260.95] vol=1.7x ATR=0.86 |
| Stop hit — per-position SL triggered | 2024-11-27 11:20:00 | 260.39 | 260.12 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-11-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:30:00 | 263.25 | 262.45 | 0.00 | ORB-long ORB[260.50,263.20] vol=1.5x ATR=0.90 |
| Stop hit — per-position SL triggered | 2024-11-28 09:45:00 | 262.35 | 262.58 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-12-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 10:35:00 | 266.70 | 265.59 | 0.00 | ORB-long ORB[263.10,266.45] vol=1.6x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 10:45:00 | 268.59 | 266.80 | 0.00 | T1 1.5R @ 268.59 |
| Target hit | 2024-12-06 11:25:00 | 267.15 | 267.19 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-22 10:50:00 | 243.35 | 2024-05-22 13:15:00 | 241.31 | PARTIAL | 0.50 | 0.84% |
| SELL | retest1 | 2024-05-22 10:50:00 | 243.35 | 2024-05-22 15:20:00 | 241.20 | TARGET_HIT | 0.50 | 0.88% |
| SELL | retest1 | 2024-05-28 09:50:00 | 225.15 | 2024-05-28 10:10:00 | 226.41 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2024-06-10 09:30:00 | 220.77 | 2024-06-10 09:40:00 | 221.92 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-06-13 09:30:00 | 234.21 | 2024-06-13 11:10:00 | 232.75 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-06-13 09:30:00 | 234.21 | 2024-06-13 15:20:00 | 233.20 | TARGET_HIT | 0.50 | 0.43% |
| BUY | retest1 | 2024-06-14 09:50:00 | 234.65 | 2024-06-14 09:55:00 | 233.87 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-06-18 09:55:00 | 230.79 | 2024-06-18 10:15:00 | 229.57 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-06-18 09:55:00 | 230.79 | 2024-06-18 10:30:00 | 230.79 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-20 09:30:00 | 239.95 | 2024-06-20 09:50:00 | 238.70 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-06-24 09:30:00 | 231.16 | 2024-06-24 09:45:00 | 232.16 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-06-25 11:15:00 | 231.90 | 2024-06-25 11:20:00 | 232.61 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-06-26 10:40:00 | 236.80 | 2024-06-26 11:35:00 | 238.45 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-06-26 10:40:00 | 236.80 | 2024-06-26 13:00:00 | 241.42 | TARGET_HIT | 0.50 | 1.95% |
| BUY | retest1 | 2024-06-28 09:30:00 | 247.40 | 2024-06-28 10:00:00 | 245.79 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest1 | 2024-07-02 09:40:00 | 245.23 | 2024-07-02 10:00:00 | 247.30 | PARTIAL | 0.50 | 0.84% |
| BUY | retest1 | 2024-07-02 09:40:00 | 245.23 | 2024-07-02 12:50:00 | 255.69 | TARGET_HIT | 0.50 | 4.27% |
| SELL | retest1 | 2024-07-23 11:15:00 | 265.27 | 2024-07-23 11:30:00 | 263.20 | PARTIAL | 0.50 | 0.78% |
| SELL | retest1 | 2024-07-23 11:15:00 | 265.27 | 2024-07-23 11:40:00 | 265.27 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-26 09:30:00 | 284.91 | 2024-07-26 09:55:00 | 283.20 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest1 | 2024-07-29 09:30:00 | 288.45 | 2024-07-29 09:55:00 | 290.26 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest1 | 2024-08-09 09:35:00 | 268.60 | 2024-08-09 09:45:00 | 270.83 | PARTIAL | 0.50 | 0.83% |
| BUY | retest1 | 2024-08-09 09:35:00 | 268.60 | 2024-08-09 10:10:00 | 268.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-16 09:30:00 | 259.45 | 2024-08-16 09:40:00 | 260.58 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-08-23 11:15:00 | 274.00 | 2024-08-23 13:35:00 | 275.19 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-08-23 11:15:00 | 274.00 | 2024-08-23 13:50:00 | 274.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-26 10:25:00 | 274.75 | 2024-08-26 10:55:00 | 273.45 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-08-26 10:25:00 | 274.75 | 2024-08-26 11:05:00 | 274.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-18 11:00:00 | 269.75 | 2024-09-18 14:05:00 | 268.61 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-09-18 11:00:00 | 269.75 | 2024-09-18 14:40:00 | 269.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-19 11:10:00 | 262.00 | 2024-09-19 11:20:00 | 260.67 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-09-19 11:10:00 | 262.00 | 2024-09-19 12:15:00 | 262.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-20 11:10:00 | 266.80 | 2024-09-20 11:50:00 | 265.91 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-09-23 11:10:00 | 271.85 | 2024-09-23 12:50:00 | 273.40 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-09-23 11:10:00 | 271.85 | 2024-09-23 14:25:00 | 272.15 | TARGET_HIT | 0.50 | 0.11% |
| BUY | retest1 | 2024-09-24 09:40:00 | 276.80 | 2024-09-24 09:45:00 | 275.96 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-09-25 09:40:00 | 291.05 | 2024-09-25 10:00:00 | 289.14 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest1 | 2024-09-27 10:55:00 | 280.75 | 2024-09-27 11:55:00 | 279.73 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-10-09 11:05:00 | 277.30 | 2024-10-09 11:10:00 | 278.88 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-10-09 11:05:00 | 277.30 | 2024-10-09 11:40:00 | 277.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-15 09:35:00 | 269.75 | 2024-10-15 09:50:00 | 268.81 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-10-16 10:15:00 | 270.15 | 2024-10-16 12:15:00 | 269.15 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-10-17 10:55:00 | 263.95 | 2024-10-17 11:25:00 | 262.66 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-10-17 10:55:00 | 263.95 | 2024-10-17 15:20:00 | 261.30 | TARGET_HIT | 0.50 | 1.00% |
| SELL | retest1 | 2024-10-21 09:35:00 | 260.85 | 2024-10-21 09:50:00 | 259.73 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-10-21 09:35:00 | 260.85 | 2024-10-21 11:10:00 | 260.55 | TARGET_HIT | 0.50 | 0.12% |
| BUY | retest1 | 2024-10-31 09:45:00 | 253.30 | 2024-10-31 10:25:00 | 255.10 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2024-10-31 09:45:00 | 253.30 | 2024-10-31 12:40:00 | 254.55 | TARGET_HIT | 0.50 | 0.49% |
| SELL | retest1 | 2024-11-12 09:50:00 | 245.50 | 2024-11-12 12:10:00 | 243.86 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2024-11-12 09:50:00 | 245.50 | 2024-11-12 15:20:00 | 238.30 | TARGET_HIT | 0.50 | 2.93% |
| BUY | retest1 | 2024-11-25 10:05:00 | 264.85 | 2024-11-25 13:00:00 | 263.09 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest1 | 2024-11-27 11:10:00 | 261.25 | 2024-11-27 11:20:00 | 260.39 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-11-28 09:30:00 | 263.25 | 2024-11-28 09:45:00 | 262.35 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-12-06 10:35:00 | 266.70 | 2024-12-06 10:45:00 | 268.59 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2024-12-06 10:35:00 | 266.70 | 2024-12-06 11:25:00 | 267.15 | TARGET_HIT | 0.50 | 0.17% |
