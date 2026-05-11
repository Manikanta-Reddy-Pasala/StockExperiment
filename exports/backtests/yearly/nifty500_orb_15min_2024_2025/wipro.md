# Wipro Ltd. (WIPRO)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 197.88
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
| ENTRY1 | 78 |
| ENTRY2 | 0 |
| PARTIAL | 35 |
| TARGET_HIT | 15 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 113 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 50 / 63
- **Target hits / Stop hits / Partials:** 15 / 63 / 35
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 19.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 61 | 25 | 41.0% | 7 | 36 | 18 | 0.20% | 12.1% |
| BUY @ 2nd Alert (retest1) | 61 | 25 | 41.0% | 7 | 36 | 18 | 0.20% | 12.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 52 | 25 | 48.1% | 8 | 27 | 17 | 0.14% | 7.3% |
| SELL @ 2nd Alert (retest1) | 52 | 25 | 48.1% | 8 | 27 | 17 | 0.14% | 7.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 113 | 50 | 44.2% | 15 | 63 | 35 | 0.17% | 19.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 11:15:00 | 229.48 | 231.60 | 0.00 | ORB-short ORB[230.60,233.75] vol=2.0x ATR=0.62 |
| Stop hit — per-position SL triggered | 2024-05-16 11:30:00 | 230.10 | 231.49 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-21 09:55:00 | 230.60 | 231.16 | 0.00 | ORB-short ORB[230.83,231.93] vol=1.5x ATR=0.46 |
| Stop hit — per-position SL triggered | 2024-05-21 11:00:00 | 231.06 | 231.04 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 09:35:00 | 232.93 | 232.03 | 0.00 | ORB-long ORB[230.88,232.05] vol=3.5x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 09:40:00 | 233.69 | 232.42 | 0.00 | T1 1.5R @ 233.69 |
| Stop hit — per-position SL triggered | 2024-05-23 09:45:00 | 232.93 | 232.58 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 09:45:00 | 227.75 | 229.14 | 0.00 | ORB-short ORB[228.33,231.08] vol=1.7x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 09:55:00 | 226.69 | 228.41 | 0.00 | T1 1.5R @ 226.69 |
| Stop hit — per-position SL triggered | 2024-05-27 10:05:00 | 227.75 | 228.19 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-29 11:15:00 | 225.45 | 226.86 | 0.00 | ORB-short ORB[226.23,227.75] vol=1.7x ATR=0.43 |
| Stop hit — per-position SL triggered | 2024-05-29 11:20:00 | 225.88 | 226.83 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 09:30:00 | 222.75 | 223.53 | 0.00 | ORB-short ORB[223.05,224.75] vol=1.8x ATR=0.47 |
| Stop hit — per-position SL triggered | 2024-05-30 09:40:00 | 223.22 | 223.41 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-05 10:20:00 | 224.30 | 221.83 | 0.00 | ORB-long ORB[219.53,222.73] vol=1.9x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-05 10:35:00 | 225.96 | 223.03 | 0.00 | T1 1.5R @ 225.96 |
| Stop hit — per-position SL triggered | 2024-06-05 11:20:00 | 224.30 | 224.20 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 10:35:00 | 246.48 | 245.39 | 0.00 | ORB-long ORB[244.65,246.40] vol=1.6x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 10:45:00 | 247.11 | 245.70 | 0.00 | T1 1.5R @ 247.11 |
| Stop hit — per-position SL triggered | 2024-06-25 11:00:00 | 246.48 | 245.83 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-07-01 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:55:00 | 262.05 | 259.44 | 0.00 | ORB-long ORB[257.05,260.08] vol=1.7x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 10:20:00 | 263.53 | 260.70 | 0.00 | T1 1.5R @ 263.53 |
| Target hit | 2024-07-01 14:35:00 | 264.20 | 264.47 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2024-07-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 11:10:00 | 271.73 | 269.75 | 0.00 | ORB-long ORB[267.75,270.95] vol=4.4x ATR=0.86 |
| Stop hit — per-position SL triggered | 2024-07-03 11:50:00 | 270.87 | 270.03 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 11:00:00 | 264.83 | 266.27 | 0.00 | ORB-short ORB[265.52,267.85] vol=1.9x ATR=0.58 |
| Stop hit — per-position SL triggered | 2024-07-05 11:25:00 | 265.41 | 265.99 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:10:00 | 268.90 | 269.99 | 0.00 | ORB-short ORB[269.45,271.68] vol=1.9x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:25:00 | 267.98 | 269.63 | 0.00 | T1 1.5R @ 267.98 |
| Target hit | 2024-07-10 15:20:00 | 267.55 | 268.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2024-07-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 09:35:00 | 273.83 | 272.25 | 0.00 | ORB-long ORB[270.77,273.50] vol=1.9x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 09:50:00 | 275.59 | 273.01 | 0.00 | T1 1.5R @ 275.59 |
| Target hit | 2024-07-12 15:20:00 | 279.55 | 277.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2024-07-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:30:00 | 258.15 | 256.59 | 0.00 | ORB-long ORB[254.25,257.48] vol=1.9x ATR=0.84 |
| Stop hit — per-position SL triggered | 2024-07-26 09:50:00 | 257.31 | 257.28 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 11:05:00 | 262.48 | 261.60 | 0.00 | ORB-long ORB[260.38,262.27] vol=1.7x ATR=0.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 11:10:00 | 263.13 | 261.69 | 0.00 | T1 1.5R @ 263.13 |
| Stop hit — per-position SL triggered | 2024-07-30 11:40:00 | 262.48 | 261.88 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-08-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-05 10:55:00 | 244.55 | 246.13 | 0.00 | ORB-short ORB[244.98,247.93] vol=2.6x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 11:10:00 | 243.23 | 245.79 | 0.00 | T1 1.5R @ 243.23 |
| Target hit | 2024-08-05 15:20:00 | 242.70 | 243.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2024-08-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 11:10:00 | 245.90 | 244.96 | 0.00 | ORB-long ORB[243.58,244.70] vol=1.9x ATR=0.50 |
| Stop hit — per-position SL triggered | 2024-08-13 11:40:00 | 245.40 | 245.04 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-08-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:30:00 | 252.25 | 250.94 | 0.00 | ORB-long ORB[248.98,251.68] vol=2.1x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-16 09:55:00 | 253.30 | 251.90 | 0.00 | T1 1.5R @ 253.30 |
| Target hit | 2024-08-16 15:20:00 | 258.02 | 255.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — BUY (started 2024-08-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 10:00:00 | 260.10 | 259.42 | 0.00 | ORB-long ORB[256.85,259.70] vol=1.8x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 10:40:00 | 261.20 | 259.61 | 0.00 | T1 1.5R @ 261.20 |
| Stop hit — per-position SL triggered | 2024-08-19 11:05:00 | 260.10 | 259.75 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 09:45:00 | 265.05 | 264.07 | 0.00 | ORB-long ORB[262.83,264.63] vol=1.9x ATR=0.64 |
| Stop hit — per-position SL triggered | 2024-08-22 09:50:00 | 264.41 | 264.15 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 09:30:00 | 257.20 | 258.67 | 0.00 | ORB-short ORB[258.08,260.73] vol=1.5x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 11:00:00 | 256.22 | 257.55 | 0.00 | T1 1.5R @ 256.22 |
| Stop hit — per-position SL triggered | 2024-08-23 11:05:00 | 257.20 | 257.54 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:45:00 | 261.90 | 260.92 | 0.00 | ORB-long ORB[257.98,261.63] vol=2.2x ATR=0.70 |
| Stop hit — per-position SL triggered | 2024-08-26 10:05:00 | 261.20 | 261.16 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 10:50:00 | 259.18 | 260.10 | 0.00 | ORB-short ORB[259.50,261.63] vol=2.2x ATR=0.53 |
| Stop hit — per-position SL triggered | 2024-08-27 10:55:00 | 259.71 | 260.05 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-28 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 10:20:00 | 261.75 | 260.14 | 0.00 | ORB-long ORB[259.10,260.85] vol=2.3x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-28 10:25:00 | 262.75 | 260.64 | 0.00 | T1 1.5R @ 262.75 |
| Target hit | 2024-08-28 15:20:00 | 267.30 | 266.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — BUY (started 2024-08-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 10:45:00 | 270.75 | 268.89 | 0.00 | ORB-long ORB[267.02,270.45] vol=1.9x ATR=0.85 |
| Stop hit — per-position SL triggered | 2024-08-29 10:55:00 | 269.90 | 269.05 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-09-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-02 10:55:00 | 267.65 | 268.84 | 0.00 | ORB-short ORB[268.77,271.00] vol=2.6x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-02 11:05:00 | 266.74 | 268.40 | 0.00 | T1 1.5R @ 266.74 |
| Stop hit — per-position SL triggered | 2024-09-02 11:45:00 | 267.65 | 268.00 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-09-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 10:00:00 | 269.95 | 267.54 | 0.00 | ORB-long ORB[264.75,267.73] vol=1.8x ATR=0.78 |
| Stop hit — per-position SL triggered | 2024-09-03 10:05:00 | 269.17 | 267.72 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-09-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-04 10:50:00 | 260.73 | 262.19 | 0.00 | ORB-short ORB[261.25,264.13] vol=1.5x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 11:00:00 | 259.64 | 261.86 | 0.00 | T1 1.5R @ 259.64 |
| Target hit | 2024-09-04 15:20:00 | 259.63 | 260.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — SELL (started 2024-09-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:30:00 | 260.50 | 262.37 | 0.00 | ORB-short ORB[261.83,264.83] vol=3.2x ATR=0.83 |
| Stop hit — per-position SL triggered | 2024-09-06 10:40:00 | 261.33 | 262.28 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 09:40:00 | 269.50 | 267.45 | 0.00 | ORB-long ORB[265.43,267.20] vol=1.6x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-13 09:45:00 | 271.02 | 269.32 | 0.00 | T1 1.5R @ 271.02 |
| Target hit | 2024-09-13 15:20:00 | 275.43 | 273.32 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — BUY (started 2024-09-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 09:30:00 | 275.93 | 274.67 | 0.00 | ORB-long ORB[273.52,275.27] vol=1.5x ATR=0.68 |
| Stop hit — per-position SL triggered | 2024-09-17 09:50:00 | 275.25 | 275.28 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-09-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-23 10:50:00 | 266.25 | 268.02 | 0.00 | ORB-short ORB[267.90,270.20] vol=1.8x ATR=0.70 |
| Stop hit — per-position SL triggered | 2024-09-23 11:05:00 | 266.95 | 267.86 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 10:55:00 | 268.00 | 266.14 | 0.00 | ORB-long ORB[265.00,266.85] vol=2.2x ATR=0.59 |
| Stop hit — per-position SL triggered | 2024-09-24 11:20:00 | 267.41 | 266.43 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-10-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-04 10:30:00 | 270.30 | 267.47 | 0.00 | ORB-long ORB[264.02,267.48] vol=2.6x ATR=0.96 |
| Stop hit — per-position SL triggered | 2024-10-04 10:35:00 | 269.34 | 267.59 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-10-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:35:00 | 266.10 | 268.56 | 0.00 | ORB-short ORB[268.58,270.85] vol=1.7x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 11:00:00 | 264.71 | 267.53 | 0.00 | T1 1.5R @ 264.71 |
| Stop hit — per-position SL triggered | 2024-10-07 11:40:00 | 266.10 | 266.71 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-10-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 11:00:00 | 267.48 | 267.28 | 0.00 | ORB-long ORB[264.25,266.70] vol=1.5x ATR=0.70 |
| Stop hit — per-position SL triggered | 2024-10-09 11:45:00 | 266.78 | 267.31 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-10-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 09:30:00 | 269.00 | 270.38 | 0.00 | ORB-short ORB[269.35,272.75] vol=2.0x ATR=1.09 |
| Stop hit — per-position SL triggered | 2024-10-14 10:00:00 | 270.09 | 269.95 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-10-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-21 10:05:00 | 279.73 | 277.96 | 0.00 | ORB-long ORB[275.50,278.60] vol=1.5x ATR=1.25 |
| Stop hit — per-position SL triggered | 2024-10-21 10:45:00 | 278.48 | 278.34 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-10-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-22 09:55:00 | 277.40 | 275.86 | 0.00 | ORB-long ORB[274.05,276.30] vol=1.7x ATR=0.92 |
| Stop hit — per-position SL triggered | 2024-10-22 10:10:00 | 276.48 | 276.04 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-10-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-23 10:30:00 | 275.38 | 274.09 | 0.00 | ORB-long ORB[271.10,274.48] vol=2.3x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 11:50:00 | 276.91 | 274.79 | 0.00 | T1 1.5R @ 276.91 |
| Stop hit — per-position SL triggered | 2024-10-23 12:15:00 | 275.38 | 275.00 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-10-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:55:00 | 271.50 | 273.31 | 0.00 | ORB-short ORB[273.00,276.00] vol=2.5x ATR=0.79 |
| Stop hit — per-position SL triggered | 2024-10-25 11:15:00 | 272.29 | 273.03 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-10-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-28 10:30:00 | 276.40 | 273.54 | 0.00 | ORB-long ORB[271.35,273.77] vol=1.6x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 10:45:00 | 277.64 | 274.58 | 0.00 | T1 1.5R @ 277.64 |
| Target hit | 2024-10-28 15:20:00 | 279.50 | 277.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2024-10-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 09:30:00 | 285.38 | 282.83 | 0.00 | ORB-long ORB[280.55,283.00] vol=2.9x ATR=0.91 |
| Stop hit — per-position SL triggered | 2024-10-30 09:35:00 | 284.47 | 283.23 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-11-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-04 10:00:00 | 270.68 | 273.48 | 0.00 | ORB-short ORB[273.55,275.50] vol=1.6x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-04 10:30:00 | 269.16 | 272.44 | 0.00 | T1 1.5R @ 269.16 |
| Target hit | 2024-11-04 15:10:00 | 270.55 | 270.36 | 0.00 | Trail-exit close>VWAP |

### Cycle 45 — BUY (started 2024-11-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 09:40:00 | 286.00 | 284.64 | 0.00 | ORB-long ORB[282.50,285.45] vol=2.0x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-08 09:45:00 | 287.20 | 285.08 | 0.00 | T1 1.5R @ 287.20 |
| Stop hit — per-position SL triggered | 2024-11-08 09:50:00 | 286.00 | 285.22 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-11-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 11:00:00 | 290.18 | 286.95 | 0.00 | ORB-long ORB[283.02,285.35] vol=3.5x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 11:05:00 | 291.41 | 287.41 | 0.00 | T1 1.5R @ 291.41 |
| Stop hit — per-position SL triggered | 2024-11-11 11:20:00 | 290.18 | 287.90 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-11-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 10:55:00 | 281.98 | 280.85 | 0.00 | ORB-long ORB[277.35,280.88] vol=1.7x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 11:30:00 | 283.01 | 281.23 | 0.00 | T1 1.5R @ 283.01 |
| Stop hit — per-position SL triggered | 2024-11-19 12:25:00 | 281.98 | 281.66 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-11-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 10:30:00 | 285.73 | 282.90 | 0.00 | ORB-long ORB[278.95,281.27] vol=1.6x ATR=0.82 |
| Stop hit — per-position SL triggered | 2024-11-22 10:45:00 | 284.91 | 283.89 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-11-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 11:05:00 | 290.43 | 288.77 | 0.00 | ORB-long ORB[286.95,290.00] vol=1.8x ATR=0.75 |
| Stop hit — per-position SL triggered | 2024-11-25 11:40:00 | 289.68 | 289.33 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-11-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:35:00 | 288.25 | 290.29 | 0.00 | ORB-short ORB[289.70,293.00] vol=1.7x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 10:45:00 | 286.94 | 289.87 | 0.00 | T1 1.5R @ 286.94 |
| Stop hit — per-position SL triggered | 2024-11-28 13:55:00 | 288.25 | 288.38 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-12-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:30:00 | 295.55 | 293.79 | 0.00 | ORB-long ORB[291.80,294.65] vol=1.8x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 09:40:00 | 296.63 | 294.48 | 0.00 | T1 1.5R @ 296.63 |
| Stop hit — per-position SL triggered | 2024-12-04 10:10:00 | 295.55 | 295.09 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-12-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 09:40:00 | 312.50 | 311.10 | 0.00 | ORB-long ORB[307.45,311.60] vol=1.9x ATR=0.89 |
| Stop hit — per-position SL triggered | 2024-12-12 10:25:00 | 311.61 | 311.94 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-12-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 10:30:00 | 309.00 | 307.45 | 0.00 | ORB-long ORB[305.65,308.30] vol=2.2x ATR=0.80 |
| Stop hit — per-position SL triggered | 2024-12-24 10:40:00 | 308.20 | 307.56 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-12-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 10:35:00 | 306.90 | 305.74 | 0.00 | ORB-long ORB[304.95,306.75] vol=3.3x ATR=0.79 |
| Stop hit — per-position SL triggered | 2024-12-26 10:40:00 | 306.11 | 305.75 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 10:15:00 | 309.90 | 306.91 | 0.00 | ORB-long ORB[304.85,308.00] vol=2.4x ATR=0.84 |
| Stop hit — per-position SL triggered | 2024-12-27 10:25:00 | 309.06 | 307.46 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-12-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-30 10:35:00 | 304.65 | 305.98 | 0.00 | ORB-short ORB[307.00,309.40] vol=3.3x ATR=0.76 |
| Stop hit — per-position SL triggered | 2024-12-30 10:45:00 | 305.41 | 305.88 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-01-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:55:00 | 302.60 | 300.06 | 0.00 | ORB-long ORB[297.15,300.50] vol=3.5x ATR=0.97 |
| Stop hit — per-position SL triggered | 2025-01-02 11:05:00 | 301.63 | 300.20 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-01-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 10:45:00 | 296.15 | 299.33 | 0.00 | ORB-short ORB[301.15,303.80] vol=1.9x ATR=0.79 |
| Stop hit — per-position SL triggered | 2025-01-03 10:50:00 | 296.94 | 299.15 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-01-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:10:00 | 293.25 | 295.48 | 0.00 | ORB-short ORB[294.55,297.80] vol=1.9x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:40:00 | 292.10 | 295.03 | 0.00 | T1 1.5R @ 292.10 |
| Stop hit — per-position SL triggered | 2025-01-06 12:35:00 | 293.25 | 294.47 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-01-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 09:45:00 | 292.60 | 295.20 | 0.00 | ORB-short ORB[294.20,297.50] vol=1.5x ATR=1.10 |
| Stop hit — per-position SL triggered | 2025-01-07 10:25:00 | 293.70 | 293.75 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-01-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:50:00 | 294.35 | 295.36 | 0.00 | ORB-short ORB[295.00,298.00] vol=1.7x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 12:25:00 | 292.83 | 294.57 | 0.00 | T1 1.5R @ 292.83 |
| Target hit | 2025-01-09 15:20:00 | 292.15 | 293.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — SELL (started 2025-01-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 09:55:00 | 291.30 | 292.06 | 0.00 | ORB-short ORB[292.60,294.70] vol=1.7x ATR=0.90 |
| Stop hit — per-position SL triggered | 2025-01-15 10:00:00 | 292.20 | 292.05 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-01-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-22 09:30:00 | 302.90 | 301.53 | 0.00 | ORB-long ORB[299.35,302.00] vol=3.8x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:35:00 | 304.23 | 302.34 | 0.00 | T1 1.5R @ 304.23 |
| Target hit | 2025-01-22 12:20:00 | 304.80 | 305.25 | 0.00 | Trail-exit close<VWAP |

### Cycle 64 — BUY (started 2025-01-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-24 10:55:00 | 320.25 | 318.52 | 0.00 | ORB-long ORB[316.85,320.00] vol=1.8x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 11:20:00 | 321.67 | 319.12 | 0.00 | T1 1.5R @ 321.67 |
| Stop hit — per-position SL triggered | 2025-01-24 12:35:00 | 320.25 | 319.76 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-01-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 10:45:00 | 311.80 | 315.42 | 0.00 | ORB-short ORB[315.60,319.15] vol=2.1x ATR=1.03 |
| Stop hit — per-position SL triggered | 2025-01-27 11:10:00 | 312.83 | 314.83 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-01-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 10:35:00 | 302.75 | 304.43 | 0.00 | ORB-short ORB[303.25,307.60] vol=1.7x ATR=1.17 |
| Stop hit — per-position SL triggered | 2025-01-28 11:10:00 | 303.92 | 303.96 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-31 11:15:00 | 311.45 | 313.08 | 0.00 | ORB-short ORB[311.50,315.70] vol=1.7x ATR=0.88 |
| Stop hit — per-position SL triggered | 2025-01-31 11:45:00 | 312.33 | 312.84 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-02-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 11:05:00 | 308.45 | 309.87 | 0.00 | ORB-short ORB[308.75,313.25] vol=2.9x ATR=0.81 |
| Stop hit — per-position SL triggered | 2025-02-01 11:15:00 | 309.26 | 309.76 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-02-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-05 11:00:00 | 316.95 | 318.07 | 0.00 | ORB-short ORB[317.20,321.45] vol=1.8x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-05 11:20:00 | 315.74 | 317.93 | 0.00 | T1 1.5R @ 315.74 |
| Stop hit — per-position SL triggered | 2025-02-05 14:00:00 | 316.95 | 316.98 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-02-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-10 11:10:00 | 320.90 | 318.59 | 0.00 | ORB-long ORB[315.85,319.80] vol=2.7x ATR=0.80 |
| Stop hit — per-position SL triggered | 2025-02-10 11:20:00 | 320.10 | 318.74 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-02-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-11 11:05:00 | 314.70 | 317.36 | 0.00 | ORB-short ORB[317.00,320.40] vol=2.4x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 11:15:00 | 313.56 | 317.11 | 0.00 | T1 1.5R @ 313.56 |
| Target hit | 2025-02-11 15:20:00 | 313.00 | 314.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 72 — SELL (started 2025-02-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-12 11:05:00 | 311.55 | 312.74 | 0.00 | ORB-short ORB[313.10,315.75] vol=1.9x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 11:35:00 | 309.89 | 312.41 | 0.00 | T1 1.5R @ 309.89 |
| Stop hit — per-position SL triggered | 2025-02-12 11:55:00 | 311.55 | 312.03 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2025-02-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 10:40:00 | 307.60 | 309.59 | 0.00 | ORB-short ORB[309.45,312.40] vol=3.8x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 11:00:00 | 306.37 | 309.31 | 0.00 | T1 1.5R @ 306.37 |
| Stop hit — per-position SL triggered | 2025-02-14 14:35:00 | 307.60 | 307.25 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-02-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 09:35:00 | 313.90 | 312.63 | 0.00 | ORB-long ORB[310.25,313.55] vol=2.1x ATR=0.94 |
| Stop hit — per-position SL triggered | 2025-02-20 09:45:00 | 312.96 | 312.74 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2025-02-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 10:50:00 | 307.90 | 309.99 | 0.00 | ORB-short ORB[310.30,312.80] vol=1.5x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-21 12:10:00 | 306.46 | 309.28 | 0.00 | T1 1.5R @ 306.46 |
| Target hit | 2025-02-21 15:20:00 | 306.40 | 308.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 76 — SELL (started 2025-02-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-24 09:35:00 | 298.90 | 301.05 | 0.00 | ORB-short ORB[300.30,303.30] vol=1.9x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-24 09:55:00 | 297.51 | 300.09 | 0.00 | T1 1.5R @ 297.51 |
| Target hit | 2025-02-24 15:20:00 | 295.05 | 296.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 77 — BUY (started 2025-04-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 11:05:00 | 263.95 | 262.83 | 0.00 | ORB-long ORB[261.10,263.80] vol=2.2x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-04-02 11:15:00 | 263.35 | 262.85 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2025-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 09:30:00 | 245.60 | 243.41 | 0.00 | ORB-long ORB[241.20,244.25] vol=1.8x ATR=0.78 |
| Stop hit — per-position SL triggered | 2025-04-16 09:35:00 | 244.82 | 243.71 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-16 11:15:00 | 229.48 | 2024-05-16 11:30:00 | 230.10 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-05-21 09:55:00 | 230.60 | 2024-05-21 11:00:00 | 231.06 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-05-23 09:35:00 | 232.93 | 2024-05-23 09:40:00 | 233.69 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-05-23 09:35:00 | 232.93 | 2024-05-23 09:45:00 | 232.93 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-27 09:45:00 | 227.75 | 2024-05-27 09:55:00 | 226.69 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-05-27 09:45:00 | 227.75 | 2024-05-27 10:05:00 | 227.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-29 11:15:00 | 225.45 | 2024-05-29 11:20:00 | 225.88 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-05-30 09:30:00 | 222.75 | 2024-05-30 09:40:00 | 223.22 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-06-05 10:20:00 | 224.30 | 2024-06-05 10:35:00 | 225.96 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2024-06-05 10:20:00 | 224.30 | 2024-06-05 11:20:00 | 224.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-25 10:35:00 | 246.48 | 2024-06-25 10:45:00 | 247.11 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2024-06-25 10:35:00 | 246.48 | 2024-06-25 11:00:00 | 246.48 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-01 09:55:00 | 262.05 | 2024-07-01 10:20:00 | 263.53 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-07-01 09:55:00 | 262.05 | 2024-07-01 14:35:00 | 264.20 | TARGET_HIT | 0.50 | 0.82% |
| BUY | retest1 | 2024-07-03 11:10:00 | 271.73 | 2024-07-03 11:50:00 | 270.87 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-07-05 11:00:00 | 264.83 | 2024-07-05 11:25:00 | 265.41 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-07-10 10:10:00 | 268.90 | 2024-07-10 10:25:00 | 267.98 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-07-10 10:10:00 | 268.90 | 2024-07-10 15:20:00 | 267.55 | TARGET_HIT | 0.50 | 0.50% |
| BUY | retest1 | 2024-07-12 09:35:00 | 273.83 | 2024-07-12 09:50:00 | 275.59 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-07-12 09:35:00 | 273.83 | 2024-07-12 15:20:00 | 279.55 | TARGET_HIT | 0.50 | 2.09% |
| BUY | retest1 | 2024-07-26 09:30:00 | 258.15 | 2024-07-26 09:50:00 | 257.31 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-07-30 11:05:00 | 262.48 | 2024-07-30 11:10:00 | 263.13 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2024-07-30 11:05:00 | 262.48 | 2024-07-30 11:40:00 | 262.48 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-05 10:55:00 | 244.55 | 2024-08-05 11:10:00 | 243.23 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-08-05 10:55:00 | 244.55 | 2024-08-05 15:20:00 | 242.70 | TARGET_HIT | 0.50 | 0.76% |
| BUY | retest1 | 2024-08-13 11:10:00 | 245.90 | 2024-08-13 11:40:00 | 245.40 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-08-16 09:30:00 | 252.25 | 2024-08-16 09:55:00 | 253.30 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-08-16 09:30:00 | 252.25 | 2024-08-16 15:20:00 | 258.02 | TARGET_HIT | 0.50 | 2.29% |
| BUY | retest1 | 2024-08-19 10:00:00 | 260.10 | 2024-08-19 10:40:00 | 261.20 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-08-19 10:00:00 | 260.10 | 2024-08-19 11:05:00 | 260.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-22 09:45:00 | 265.05 | 2024-08-22 09:50:00 | 264.41 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-08-23 09:30:00 | 257.20 | 2024-08-23 11:00:00 | 256.22 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-08-23 09:30:00 | 257.20 | 2024-08-23 11:05:00 | 257.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-26 09:45:00 | 261.90 | 2024-08-26 10:05:00 | 261.20 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-08-27 10:50:00 | 259.18 | 2024-08-27 10:55:00 | 259.71 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-08-28 10:20:00 | 261.75 | 2024-08-28 10:25:00 | 262.75 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-08-28 10:20:00 | 261.75 | 2024-08-28 15:20:00 | 267.30 | TARGET_HIT | 0.50 | 2.12% |
| BUY | retest1 | 2024-08-29 10:45:00 | 270.75 | 2024-08-29 10:55:00 | 269.90 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-09-02 10:55:00 | 267.65 | 2024-09-02 11:05:00 | 266.74 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-09-02 10:55:00 | 267.65 | 2024-09-02 11:45:00 | 267.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-03 10:00:00 | 269.95 | 2024-09-03 10:05:00 | 269.17 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-09-04 10:50:00 | 260.73 | 2024-09-04 11:00:00 | 259.64 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-09-04 10:50:00 | 260.73 | 2024-09-04 15:20:00 | 259.63 | TARGET_HIT | 0.50 | 0.42% |
| SELL | retest1 | 2024-09-06 10:30:00 | 260.50 | 2024-09-06 10:40:00 | 261.33 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-09-13 09:40:00 | 269.50 | 2024-09-13 09:45:00 | 271.02 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-09-13 09:40:00 | 269.50 | 2024-09-13 15:20:00 | 275.43 | TARGET_HIT | 0.50 | 2.20% |
| BUY | retest1 | 2024-09-17 09:30:00 | 275.93 | 2024-09-17 09:50:00 | 275.25 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-09-23 10:50:00 | 266.25 | 2024-09-23 11:05:00 | 266.95 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-09-24 10:55:00 | 268.00 | 2024-09-24 11:20:00 | 267.41 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-10-04 10:30:00 | 270.30 | 2024-10-04 10:35:00 | 269.34 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-10-07 10:35:00 | 266.10 | 2024-10-07 11:00:00 | 264.71 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-10-07 10:35:00 | 266.10 | 2024-10-07 11:40:00 | 266.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-09 11:00:00 | 267.48 | 2024-10-09 11:45:00 | 266.78 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-10-14 09:30:00 | 269.00 | 2024-10-14 10:00:00 | 270.09 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-10-21 10:05:00 | 279.73 | 2024-10-21 10:45:00 | 278.48 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-10-22 09:55:00 | 277.40 | 2024-10-22 10:10:00 | 276.48 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-10-23 10:30:00 | 275.38 | 2024-10-23 11:50:00 | 276.91 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-10-23 10:30:00 | 275.38 | 2024-10-23 12:15:00 | 275.38 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-25 10:55:00 | 271.50 | 2024-10-25 11:15:00 | 272.29 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-10-28 10:30:00 | 276.40 | 2024-10-28 10:45:00 | 277.64 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-10-28 10:30:00 | 276.40 | 2024-10-28 15:20:00 | 279.50 | TARGET_HIT | 0.50 | 1.12% |
| BUY | retest1 | 2024-10-30 09:30:00 | 285.38 | 2024-10-30 09:35:00 | 284.47 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-11-04 10:00:00 | 270.68 | 2024-11-04 10:30:00 | 269.16 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-11-04 10:00:00 | 270.68 | 2024-11-04 15:10:00 | 270.55 | TARGET_HIT | 0.50 | 0.05% |
| BUY | retest1 | 2024-11-08 09:40:00 | 286.00 | 2024-11-08 09:45:00 | 287.20 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-11-08 09:40:00 | 286.00 | 2024-11-08 09:50:00 | 286.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-11 11:00:00 | 290.18 | 2024-11-11 11:05:00 | 291.41 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-11-11 11:00:00 | 290.18 | 2024-11-11 11:20:00 | 290.18 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-19 10:55:00 | 281.98 | 2024-11-19 11:30:00 | 283.01 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-11-19 10:55:00 | 281.98 | 2024-11-19 12:25:00 | 281.98 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-22 10:30:00 | 285.73 | 2024-11-22 10:45:00 | 284.91 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-11-25 11:05:00 | 290.43 | 2024-11-25 11:40:00 | 289.68 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-11-28 10:35:00 | 288.25 | 2024-11-28 10:45:00 | 286.94 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-11-28 10:35:00 | 288.25 | 2024-11-28 13:55:00 | 288.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-04 09:30:00 | 295.55 | 2024-12-04 09:40:00 | 296.63 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-12-04 09:30:00 | 295.55 | 2024-12-04 10:10:00 | 295.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-12 09:40:00 | 312.50 | 2024-12-12 10:25:00 | 311.61 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-12-24 10:30:00 | 309.00 | 2024-12-24 10:40:00 | 308.20 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-12-26 10:35:00 | 306.90 | 2024-12-26 10:40:00 | 306.11 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-12-27 10:15:00 | 309.90 | 2024-12-27 10:25:00 | 309.06 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-12-30 10:35:00 | 304.65 | 2024-12-30 10:45:00 | 305.41 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-01-02 10:55:00 | 302.60 | 2025-01-02 11:05:00 | 301.63 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-01-03 10:45:00 | 296.15 | 2025-01-03 10:50:00 | 296.94 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-01-06 11:10:00 | 293.25 | 2025-01-06 11:40:00 | 292.10 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-01-06 11:10:00 | 293.25 | 2025-01-06 12:35:00 | 293.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-07 09:45:00 | 292.60 | 2025-01-07 10:25:00 | 293.70 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-01-09 10:50:00 | 294.35 | 2025-01-09 12:25:00 | 292.83 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-01-09 10:50:00 | 294.35 | 2025-01-09 15:20:00 | 292.15 | TARGET_HIT | 0.50 | 0.75% |
| SELL | retest1 | 2025-01-15 09:55:00 | 291.30 | 2025-01-15 10:00:00 | 292.20 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-01-22 09:30:00 | 302.90 | 2025-01-22 09:35:00 | 304.23 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-01-22 09:30:00 | 302.90 | 2025-01-22 12:20:00 | 304.80 | TARGET_HIT | 0.50 | 0.63% |
| BUY | retest1 | 2025-01-24 10:55:00 | 320.25 | 2025-01-24 11:20:00 | 321.67 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-01-24 10:55:00 | 320.25 | 2025-01-24 12:35:00 | 320.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-27 10:45:00 | 311.80 | 2025-01-27 11:10:00 | 312.83 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-01-28 10:35:00 | 302.75 | 2025-01-28 11:10:00 | 303.92 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-01-31 11:15:00 | 311.45 | 2025-01-31 11:45:00 | 312.33 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-02-01 11:05:00 | 308.45 | 2025-02-01 11:15:00 | 309.26 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-02-05 11:00:00 | 316.95 | 2025-02-05 11:20:00 | 315.74 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-02-05 11:00:00 | 316.95 | 2025-02-05 14:00:00 | 316.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-10 11:10:00 | 320.90 | 2025-02-10 11:20:00 | 320.10 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-02-11 11:05:00 | 314.70 | 2025-02-11 11:15:00 | 313.56 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-02-11 11:05:00 | 314.70 | 2025-02-11 15:20:00 | 313.00 | TARGET_HIT | 0.50 | 0.54% |
| SELL | retest1 | 2025-02-12 11:05:00 | 311.55 | 2025-02-12 11:35:00 | 309.89 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-02-12 11:05:00 | 311.55 | 2025-02-12 11:55:00 | 311.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-14 10:40:00 | 307.60 | 2025-02-14 11:00:00 | 306.37 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-02-14 10:40:00 | 307.60 | 2025-02-14 14:35:00 | 307.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-20 09:35:00 | 313.90 | 2025-02-20 09:45:00 | 312.96 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-02-21 10:50:00 | 307.90 | 2025-02-21 12:10:00 | 306.46 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-02-21 10:50:00 | 307.90 | 2025-02-21 15:20:00 | 306.40 | TARGET_HIT | 0.50 | 0.49% |
| SELL | retest1 | 2025-02-24 09:35:00 | 298.90 | 2025-02-24 09:55:00 | 297.51 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-02-24 09:35:00 | 298.90 | 2025-02-24 15:20:00 | 295.05 | TARGET_HIT | 0.50 | 1.29% |
| BUY | retest1 | 2025-04-02 11:05:00 | 263.95 | 2025-04-02 11:15:00 | 263.35 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-04-16 09:30:00 | 245.60 | 2025-04-16 09:35:00 | 244.82 | STOP_HIT | 1.00 | -0.32% |
