# Bank of Baroda (BANKBARODA)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-05-05 15:25:00 (18108 bars)
- **Last close:** 249.50
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
| ENTRY1 | 94 |
| ENTRY2 | 0 |
| PARTIAL | 34 |
| TARGET_HIT | 18 |
| STOP_HIT | 76 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 128 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 52 / 76
- **Target hits / Stop hits / Partials:** 18 / 76 / 34
- **Avg / median % per leg:** 0.13% / 0.00%
- **Sum % (uncompounded):** 16.42%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 60 | 21 | 35.0% | 7 | 39 | 14 | 0.10% | 6.0% |
| BUY @ 2nd Alert (retest1) | 60 | 21 | 35.0% | 7 | 39 | 14 | 0.10% | 6.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 68 | 31 | 45.6% | 11 | 37 | 20 | 0.15% | 10.4% |
| SELL @ 2nd Alert (retest1) | 68 | 31 | 45.6% | 11 | 37 | 20 | 0.15% | 10.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 128 | 52 | 40.6% | 18 | 76 | 34 | 0.13% | 16.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 10:20:00 | 264.65 | 262.48 | 0.00 | ORB-long ORB[260.25,263.85] vol=2.1x ATR=0.86 |
| Stop hit — per-position SL triggered | 2024-05-21 10:30:00 | 263.79 | 262.66 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 10:50:00 | 270.10 | 269.31 | 0.00 | ORB-long ORB[264.60,268.30] vol=2.4x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 11:40:00 | 271.57 | 269.51 | 0.00 | T1 1.5R @ 271.57 |
| Stop hit — per-position SL triggered | 2024-05-23 13:10:00 | 270.10 | 269.78 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 11:00:00 | 269.60 | 269.77 | 0.00 | ORB-short ORB[270.00,271.80] vol=4.9x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 11:25:00 | 268.41 | 269.59 | 0.00 | T1 1.5R @ 268.41 |
| Target hit | 2024-05-28 15:20:00 | 263.70 | 267.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2024-05-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 09:30:00 | 265.00 | 263.74 | 0.00 | ORB-long ORB[261.80,264.65] vol=2.0x ATR=0.88 |
| Stop hit — per-position SL triggered | 2024-05-29 10:05:00 | 264.12 | 264.24 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 10:35:00 | 271.90 | 269.44 | 0.00 | ORB-long ORB[266.75,269.90] vol=2.0x ATR=1.04 |
| Stop hit — per-position SL triggered | 2024-06-07 10:40:00 | 270.86 | 269.56 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 10:15:00 | 279.10 | 276.73 | 0.00 | ORB-long ORB[274.80,277.60] vol=1.5x ATR=0.97 |
| Stop hit — per-position SL triggered | 2024-06-11 11:00:00 | 278.13 | 277.21 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 10:30:00 | 278.55 | 276.88 | 0.00 | ORB-long ORB[273.75,275.95] vol=2.0x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 10:40:00 | 279.75 | 277.35 | 0.00 | T1 1.5R @ 279.75 |
| Target hit | 2024-06-12 15:20:00 | 283.65 | 281.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2024-06-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 11:10:00 | 284.90 | 283.02 | 0.00 | ORB-long ORB[281.60,283.30] vol=2.2x ATR=0.72 |
| Stop hit — per-position SL triggered | 2024-06-14 11:25:00 | 284.18 | 283.13 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 09:35:00 | 285.65 | 286.45 | 0.00 | ORB-short ORB[285.75,288.45] vol=1.5x ATR=0.91 |
| Stop hit — per-position SL triggered | 2024-06-19 09:50:00 | 286.56 | 286.30 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-24 09:30:00 | 275.30 | 276.41 | 0.00 | ORB-short ORB[275.55,279.50] vol=2.3x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-24 09:40:00 | 273.73 | 275.90 | 0.00 | T1 1.5R @ 273.73 |
| Stop hit — per-position SL triggered | 2024-06-24 09:45:00 | 275.30 | 275.86 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 09:45:00 | 280.45 | 276.48 | 0.00 | ORB-long ORB[272.90,276.50] vol=1.9x ATR=1.31 |
| Stop hit — per-position SL triggered | 2024-06-28 10:00:00 | 279.14 | 277.27 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 09:45:00 | 259.85 | 261.28 | 0.00 | ORB-short ORB[260.85,263.50] vol=1.5x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 09:55:00 | 258.93 | 260.94 | 0.00 | T1 1.5R @ 258.93 |
| Target hit | 2024-07-10 15:20:00 | 256.50 | 257.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2024-07-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 10:45:00 | 256.40 | 257.65 | 0.00 | ORB-short ORB[257.20,258.70] vol=1.6x ATR=0.76 |
| Stop hit — per-position SL triggered | 2024-07-11 11:00:00 | 257.16 | 257.52 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 09:40:00 | 258.80 | 257.75 | 0.00 | ORB-long ORB[256.60,258.20] vol=2.0x ATR=0.67 |
| Stop hit — per-position SL triggered | 2024-07-12 09:50:00 | 258.13 | 257.86 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 10:50:00 | 254.90 | 252.65 | 0.00 | ORB-long ORB[251.20,252.95] vol=1.6x ATR=0.74 |
| Stop hit — per-position SL triggered | 2024-07-15 11:15:00 | 254.16 | 253.06 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:40:00 | 256.00 | 256.70 | 0.00 | ORB-short ORB[257.15,258.85] vol=1.5x ATR=0.76 |
| Stop hit — per-position SL triggered | 2024-07-18 09:45:00 | 256.76 | 256.76 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 10:55:00 | 251.65 | 254.80 | 0.00 | ORB-short ORB[253.55,256.50] vol=1.8x ATR=0.95 |
| Stop hit — per-position SL triggered | 2024-07-19 11:30:00 | 252.60 | 254.27 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 09:30:00 | 252.60 | 251.52 | 0.00 | ORB-long ORB[250.00,252.15] vol=1.8x ATR=0.98 |
| Stop hit — per-position SL triggered | 2024-07-24 09:35:00 | 251.62 | 251.54 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-08-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-07 11:10:00 | 239.55 | 240.58 | 0.00 | ORB-short ORB[240.20,243.00] vol=1.7x ATR=0.79 |
| Stop hit — per-position SL triggered | 2024-08-07 12:05:00 | 240.34 | 240.35 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 11:15:00 | 244.70 | 242.09 | 0.00 | ORB-long ORB[241.30,243.50] vol=2.0x ATR=0.87 |
| Stop hit — per-position SL triggered | 2024-08-08 11:20:00 | 243.83 | 242.16 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 10:55:00 | 245.95 | 245.18 | 0.00 | ORB-long ORB[242.50,245.55] vol=2.2x ATR=0.89 |
| Stop hit — per-position SL triggered | 2024-08-12 12:40:00 | 245.06 | 245.42 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 09:35:00 | 240.50 | 242.01 | 0.00 | ORB-short ORB[241.30,243.40] vol=1.5x ATR=0.85 |
| Stop hit — per-position SL triggered | 2024-08-14 09:45:00 | 241.35 | 241.84 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:30:00 | 249.70 | 248.51 | 0.00 | ORB-long ORB[246.95,249.25] vol=3.7x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 09:40:00 | 250.63 | 249.51 | 0.00 | T1 1.5R @ 250.63 |
| Target hit | 2024-08-20 15:20:00 | 254.75 | 252.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — SELL (started 2024-08-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-21 11:05:00 | 252.25 | 253.87 | 0.00 | ORB-short ORB[253.65,255.40] vol=1.7x ATR=0.57 |
| Stop hit — per-position SL triggered | 2024-08-21 11:20:00 | 252.82 | 253.77 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:35:00 | 256.35 | 254.80 | 0.00 | ORB-long ORB[253.75,255.00] vol=1.7x ATR=0.66 |
| Stop hit — per-position SL triggered | 2024-08-22 10:55:00 | 255.69 | 255.02 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-08-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 10:35:00 | 251.80 | 252.77 | 0.00 | ORB-short ORB[252.35,253.70] vol=1.6x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 11:55:00 | 251.04 | 252.21 | 0.00 | T1 1.5R @ 251.04 |
| Stop hit — per-position SL triggered | 2024-08-26 12:10:00 | 251.80 | 252.04 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-08-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:35:00 | 249.50 | 250.26 | 0.00 | ORB-short ORB[249.55,251.65] vol=2.8x ATR=0.58 |
| Stop hit — per-position SL triggered | 2024-08-28 09:40:00 | 250.08 | 250.24 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:40:00 | 252.95 | 252.57 | 0.00 | ORB-long ORB[250.45,252.85] vol=2.7x ATR=0.64 |
| Stop hit — per-position SL triggered | 2024-08-30 11:20:00 | 252.31 | 252.70 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-09-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-04 10:30:00 | 244.95 | 246.72 | 0.00 | ORB-short ORB[246.75,249.45] vol=1.7x ATR=0.68 |
| Stop hit — per-position SL triggered | 2024-09-04 11:05:00 | 245.63 | 246.33 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-09-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:05:00 | 240.10 | 241.80 | 0.00 | ORB-short ORB[242.15,244.20] vol=2.1x ATR=0.67 |
| Stop hit — per-position SL triggered | 2024-09-06 10:10:00 | 240.77 | 241.65 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-10 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 10:40:00 | 235.75 | 236.02 | 0.00 | ORB-short ORB[235.90,237.45] vol=1.8x ATR=0.74 |
| Stop hit — per-position SL triggered | 2024-09-10 12:00:00 | 236.49 | 235.96 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:30:00 | 236.20 | 235.18 | 0.00 | ORB-long ORB[233.55,236.00] vol=1.6x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 09:45:00 | 237.17 | 235.64 | 0.00 | T1 1.5R @ 237.17 |
| Stop hit — per-position SL triggered | 2024-09-11 09:50:00 | 236.20 | 235.72 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-09-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-12 11:00:00 | 234.05 | 234.33 | 0.00 | ORB-short ORB[234.30,236.35] vol=3.5x ATR=0.57 |
| Stop hit — per-position SL triggered | 2024-09-12 11:30:00 | 234.62 | 234.18 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-09-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 10:40:00 | 239.50 | 240.43 | 0.00 | ORB-short ORB[239.85,241.65] vol=2.1x ATR=0.57 |
| Stop hit — per-position SL triggered | 2024-09-16 11:00:00 | 240.07 | 240.28 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-09-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 10:45:00 | 236.05 | 238.36 | 0.00 | ORB-short ORB[238.55,241.60] vol=1.8x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 11:15:00 | 234.68 | 237.81 | 0.00 | T1 1.5R @ 234.68 |
| Target hit | 2024-09-19 15:00:00 | 235.40 | 234.98 | 0.00 | Trail-exit close>VWAP |

### Cycle 36 — BUY (started 2024-09-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 10:45:00 | 239.35 | 237.82 | 0.00 | ORB-long ORB[237.10,238.55] vol=1.9x ATR=0.68 |
| Stop hit — per-position SL triggered | 2024-09-20 11:00:00 | 238.67 | 237.95 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-09-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 09:30:00 | 239.90 | 237.92 | 0.00 | ORB-long ORB[236.00,238.15] vol=1.5x ATR=0.80 |
| Stop hit — per-position SL triggered | 2024-09-23 09:35:00 | 239.10 | 238.11 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 11:15:00 | 240.85 | 243.10 | 0.00 | ORB-short ORB[243.60,245.15] vol=1.8x ATR=0.46 |
| Stop hit — per-position SL triggered | 2024-09-25 11:35:00 | 241.31 | 242.90 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-09-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 10:30:00 | 245.20 | 243.53 | 0.00 | ORB-long ORB[242.50,244.40] vol=4.1x ATR=0.65 |
| Stop hit — per-position SL triggered | 2024-09-26 10:45:00 | 244.55 | 243.81 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-09-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 09:50:00 | 246.65 | 245.71 | 0.00 | ORB-long ORB[244.60,246.20] vol=1.9x ATR=0.70 |
| Stop hit — per-position SL triggered | 2024-09-27 10:10:00 | 245.95 | 245.81 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-10-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:55:00 | 244.11 | 249.34 | 0.00 | ORB-short ORB[251.10,253.28] vol=1.8x ATR=1.22 |
| Stop hit — per-position SL triggered | 2024-10-07 11:10:00 | 245.33 | 248.71 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-10-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 09:45:00 | 249.85 | 249.20 | 0.00 | ORB-long ORB[247.83,249.69] vol=2.0x ATR=0.96 |
| Stop hit — per-position SL triggered | 2024-10-09 09:55:00 | 248.89 | 248.02 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-10-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 10:05:00 | 251.45 | 249.30 | 0.00 | ORB-long ORB[246.40,247.83] vol=1.6x ATR=0.82 |
| Stop hit — per-position SL triggered | 2024-10-10 10:35:00 | 250.63 | 249.88 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-10-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 10:55:00 | 245.26 | 246.85 | 0.00 | ORB-short ORB[245.51,247.20] vol=1.7x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 11:25:00 | 244.35 | 246.25 | 0.00 | T1 1.5R @ 244.35 |
| Target hit | 2024-10-11 15:20:00 | 242.73 | 243.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — BUY (started 2024-10-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 09:45:00 | 245.00 | 243.89 | 0.00 | ORB-long ORB[243.05,244.34] vol=1.6x ATR=0.69 |
| Stop hit — per-position SL triggered | 2024-10-14 10:05:00 | 244.31 | 244.04 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-10-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:40:00 | 242.90 | 243.81 | 0.00 | ORB-short ORB[243.41,245.19] vol=2.3x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 09:50:00 | 241.85 | 243.00 | 0.00 | T1 1.5R @ 241.85 |
| Target hit | 2024-10-17 12:25:00 | 241.89 | 241.78 | 0.00 | Trail-exit close>VWAP |

### Cycle 47 — BUY (started 2024-10-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 09:55:00 | 241.74 | 239.98 | 0.00 | ORB-long ORB[237.85,241.10] vol=1.8x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 10:10:00 | 243.00 | 240.50 | 0.00 | T1 1.5R @ 243.00 |
| Target hit | 2024-10-18 15:20:00 | 247.82 | 244.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — SELL (started 2024-10-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 11:15:00 | 247.65 | 248.59 | 0.00 | ORB-short ORB[247.97,251.20] vol=2.2x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 11:30:00 | 246.52 | 248.53 | 0.00 | T1 1.5R @ 246.52 |
| Target hit | 2024-10-21 15:20:00 | 245.36 | 247.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — BUY (started 2024-10-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-24 10:30:00 | 243.26 | 240.52 | 0.00 | ORB-long ORB[236.30,239.35] vol=3.8x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 10:50:00 | 244.88 | 241.58 | 0.00 | T1 1.5R @ 244.88 |
| Stop hit — per-position SL triggered | 2024-10-24 11:15:00 | 243.26 | 242.02 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-10-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 11:05:00 | 237.77 | 239.47 | 0.00 | ORB-short ORB[242.75,245.90] vol=1.5x ATR=1.38 |
| Stop hit — per-position SL triggered | 2024-10-25 11:15:00 | 239.15 | 239.41 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-11-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-04 10:45:00 | 249.55 | 251.10 | 0.00 | ORB-short ORB[251.20,253.50] vol=4.8x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-04 11:15:00 | 248.17 | 250.53 | 0.00 | T1 1.5R @ 248.17 |
| Stop hit — per-position SL triggered | 2024-11-04 11:20:00 | 249.55 | 250.38 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-11-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 09:35:00 | 264.70 | 262.99 | 0.00 | ORB-long ORB[261.00,263.35] vol=1.9x ATR=0.92 |
| Stop hit — per-position SL triggered | 2024-11-07 09:50:00 | 263.78 | 263.26 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-11-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 10:55:00 | 257.90 | 255.20 | 0.00 | ORB-long ORB[252.80,256.00] vol=1.7x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 13:15:00 | 259.46 | 256.67 | 0.00 | T1 1.5R @ 259.46 |
| Stop hit — per-position SL triggered | 2024-11-11 13:40:00 | 257.90 | 256.77 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 11:15:00 | 257.15 | 258.26 | 0.00 | ORB-short ORB[258.30,260.50] vol=2.4x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 12:05:00 | 256.01 | 258.02 | 0.00 | T1 1.5R @ 256.01 |
| Target hit | 2024-11-12 15:20:00 | 252.35 | 255.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — SELL (started 2024-11-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:45:00 | 248.10 | 250.56 | 0.00 | ORB-short ORB[249.70,252.85] vol=1.6x ATR=1.17 |
| Stop hit — per-position SL triggered | 2024-11-13 09:50:00 | 249.27 | 250.45 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-11-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 09:35:00 | 244.00 | 243.31 | 0.00 | ORB-long ORB[241.80,243.70] vol=2.9x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 10:05:00 | 245.20 | 243.98 | 0.00 | T1 1.5R @ 245.20 |
| Stop hit — per-position SL triggered | 2024-11-19 11:00:00 | 244.00 | 244.30 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2024-11-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:35:00 | 247.90 | 247.29 | 0.00 | ORB-long ORB[245.60,247.65] vol=2.0x ATR=0.75 |
| Stop hit — per-position SL triggered | 2024-11-28 09:40:00 | 247.15 | 247.30 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 10:15:00 | 246.95 | 248.80 | 0.00 | ORB-short ORB[248.75,250.45] vol=2.3x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 11:30:00 | 245.59 | 247.75 | 0.00 | T1 1.5R @ 245.59 |
| Stop hit — per-position SL triggered | 2024-11-29 14:00:00 | 246.95 | 247.00 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2024-12-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:30:00 | 256.39 | 255.44 | 0.00 | ORB-long ORB[254.22,255.79] vol=1.9x ATR=0.61 |
| Stop hit — per-position SL triggered | 2024-12-04 10:10:00 | 255.78 | 255.96 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-12-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 10:05:00 | 259.48 | 260.80 | 0.00 | ORB-short ORB[259.51,261.56] vol=2.4x ATR=1.02 |
| Stop hit — per-position SL triggered | 2024-12-06 10:20:00 | 260.50 | 261.09 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2024-12-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 09:55:00 | 254.25 | 255.58 | 0.00 | ORB-short ORB[256.15,258.90] vol=3.7x ATR=0.72 |
| Stop hit — per-position SL triggered | 2024-12-13 10:00:00 | 254.97 | 255.50 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-12-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 10:20:00 | 256.87 | 257.86 | 0.00 | ORB-short ORB[257.23,259.30] vol=1.9x ATR=0.58 |
| Stop hit — per-position SL triggered | 2024-12-17 10:40:00 | 257.45 | 257.63 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-12-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 11:00:00 | 252.30 | 252.92 | 0.00 | ORB-short ORB[252.38,255.80] vol=1.7x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 12:35:00 | 251.13 | 252.57 | 0.00 | T1 1.5R @ 251.13 |
| Stop hit — per-position SL triggered | 2024-12-18 13:05:00 | 252.30 | 252.46 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2024-12-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 09:50:00 | 246.85 | 247.78 | 0.00 | ORB-short ORB[247.20,249.99] vol=1.6x ATR=0.77 |
| Stop hit — per-position SL triggered | 2024-12-20 10:35:00 | 247.62 | 247.43 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-12-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-23 09:50:00 | 240.60 | 240.88 | 0.00 | ORB-short ORB[240.75,242.90] vol=2.9x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 09:55:00 | 239.11 | 240.84 | 0.00 | T1 1.5R @ 239.11 |
| Stop hit — per-position SL triggered | 2024-12-23 10:00:00 | 240.60 | 240.81 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2024-12-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 10:05:00 | 247.17 | 246.40 | 0.00 | ORB-long ORB[244.55,246.95] vol=1.6x ATR=0.75 |
| Stop hit — per-position SL triggered | 2024-12-24 10:40:00 | 246.42 | 246.51 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-01-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:55:00 | 241.47 | 240.30 | 0.00 | ORB-long ORB[239.26,241.29] vol=1.9x ATR=0.59 |
| Stop hit — per-position SL triggered | 2025-01-01 11:10:00 | 240.88 | 240.51 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-01-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 10:40:00 | 233.00 | 232.31 | 0.00 | ORB-long ORB[231.22,232.86] vol=2.1x ATR=0.56 |
| Stop hit — per-position SL triggered | 2025-01-09 10:45:00 | 232.44 | 232.32 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-01-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:35:00 | 227.42 | 228.84 | 0.00 | ORB-short ORB[228.02,230.82] vol=1.6x ATR=0.64 |
| Stop hit — per-position SL triggered | 2025-01-10 09:40:00 | 228.06 | 228.59 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-01-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-17 10:55:00 | 226.38 | 227.85 | 0.00 | ORB-short ORB[226.53,228.60] vol=1.6x ATR=0.57 |
| Stop hit — per-position SL triggered | 2025-01-17 11:15:00 | 226.95 | 227.50 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-01-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 10:25:00 | 230.65 | 228.92 | 0.00 | ORB-long ORB[226.81,229.82] vol=2.0x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 11:15:00 | 231.61 | 229.64 | 0.00 | T1 1.5R @ 231.61 |
| Target hit | 2025-01-20 15:15:00 | 231.95 | 232.23 | 0.00 | Trail-exit close<VWAP |

### Cycle 72 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-22 11:15:00 | 228.46 | 229.54 | 0.00 | ORB-short ORB[228.51,231.10] vol=2.2x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 12:00:00 | 227.43 | 229.22 | 0.00 | T1 1.5R @ 227.43 |
| Target hit | 2025-01-22 14:55:00 | 227.90 | 227.62 | 0.00 | Trail-exit close>VWAP |

### Cycle 73 — SELL (started 2025-01-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 09:55:00 | 226.12 | 227.68 | 0.00 | ORB-short ORB[227.61,230.01] vol=2.7x ATR=0.70 |
| Stop hit — per-position SL triggered | 2025-01-24 10:25:00 | 226.82 | 227.38 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-02-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 11:10:00 | 216.22 | 214.97 | 0.00 | ORB-long ORB[213.85,216.00] vol=1.6x ATR=0.59 |
| Stop hit — per-position SL triggered | 2025-02-01 11:20:00 | 215.63 | 215.03 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2025-02-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-04 09:35:00 | 213.15 | 212.21 | 0.00 | ORB-long ORB[210.11,212.78] vol=1.8x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-04 09:45:00 | 214.29 | 212.77 | 0.00 | T1 1.5R @ 214.29 |
| Stop hit — per-position SL triggered | 2025-02-04 10:05:00 | 213.15 | 213.08 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2025-02-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 09:40:00 | 217.40 | 216.00 | 0.00 | ORB-long ORB[214.12,216.50] vol=1.8x ATR=0.68 |
| Stop hit — per-position SL triggered | 2025-02-05 09:45:00 | 216.72 | 216.16 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2025-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 11:15:00 | 219.50 | 220.89 | 0.00 | ORB-short ORB[220.35,222.58] vol=1.9x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-02-06 11:20:00 | 220.10 | 220.87 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2025-02-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 10:15:00 | 215.80 | 218.58 | 0.00 | ORB-short ORB[218.20,220.70] vol=2.4x ATR=1.16 |
| Stop hit — per-position SL triggered | 2025-02-07 10:20:00 | 216.96 | 218.37 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2025-02-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 09:45:00 | 214.21 | 215.55 | 0.00 | ORB-short ORB[215.64,217.16] vol=1.9x ATR=0.70 |
| Stop hit — per-position SL triggered | 2025-02-10 09:50:00 | 214.91 | 215.51 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2025-02-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-13 10:55:00 | 212.26 | 213.00 | 0.00 | ORB-short ORB[212.70,214.21] vol=2.5x ATR=0.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-13 13:10:00 | 211.39 | 212.63 | 0.00 | T1 1.5R @ 211.39 |
| Target hit | 2025-02-13 15:20:00 | 210.80 | 212.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 81 — SELL (started 2025-02-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 10:05:00 | 211.78 | 213.47 | 0.00 | ORB-short ORB[212.03,214.95] vol=1.5x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-21 10:15:00 | 210.80 | 213.08 | 0.00 | T1 1.5R @ 210.80 |
| Stop hit — per-position SL triggered | 2025-02-21 10:20:00 | 211.78 | 212.93 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2025-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-25 10:15:00 | 206.93 | 208.32 | 0.00 | ORB-short ORB[207.80,209.57] vol=1.8x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 12:00:00 | 206.09 | 207.74 | 0.00 | T1 1.5R @ 206.09 |
| Target hit | 2025-02-25 15:20:00 | 204.51 | 206.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 83 — BUY (started 2025-03-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-06 09:35:00 | 209.47 | 207.83 | 0.00 | ORB-long ORB[206.24,208.70] vol=2.1x ATR=0.76 |
| Stop hit — per-position SL triggered | 2025-03-06 09:50:00 | 208.71 | 208.42 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2025-03-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-10 11:10:00 | 203.70 | 205.35 | 0.00 | ORB-short ORB[204.37,206.70] vol=2.0x ATR=0.67 |
| Stop hit — per-position SL triggered | 2025-03-10 11:20:00 | 204.37 | 205.21 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2025-03-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 09:50:00 | 205.76 | 203.93 | 0.00 | ORB-long ORB[202.30,203.84] vol=2.3x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-13 09:55:00 | 206.78 | 204.61 | 0.00 | T1 1.5R @ 206.78 |
| Stop hit — per-position SL triggered | 2025-03-13 11:00:00 | 205.76 | 206.00 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2025-03-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:00:00 | 207.64 | 206.85 | 0.00 | ORB-long ORB[205.75,207.25] vol=3.3x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 10:55:00 | 208.45 | 207.37 | 0.00 | T1 1.5R @ 208.45 |
| Target hit | 2025-03-18 15:20:00 | 209.38 | 208.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 87 — BUY (started 2025-03-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 09:30:00 | 211.85 | 211.24 | 0.00 | ORB-long ORB[209.42,211.83] vol=2.6x ATR=0.54 |
| Stop hit — per-position SL triggered | 2025-03-19 09:35:00 | 211.31 | 211.27 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2025-03-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-24 10:55:00 | 221.50 | 219.81 | 0.00 | ORB-long ORB[217.10,220.30] vol=2.0x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-24 12:20:00 | 222.31 | 220.39 | 0.00 | T1 1.5R @ 222.31 |
| Target hit | 2025-03-24 15:20:00 | 224.42 | 222.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 89 — SELL (started 2025-03-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-25 10:30:00 | 221.89 | 223.80 | 0.00 | ORB-short ORB[222.61,225.90] vol=1.8x ATR=0.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-25 11:40:00 | 220.78 | 222.82 | 0.00 | T1 1.5R @ 220.78 |
| Stop hit — per-position SL triggered | 2025-03-25 11:55:00 | 221.89 | 222.79 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2025-04-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-09 10:45:00 | 230.75 | 232.74 | 0.00 | ORB-short ORB[234.01,235.88] vol=5.0x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-09 11:05:00 | 229.48 | 232.12 | 0.00 | T1 1.5R @ 229.48 |
| Stop hit — per-position SL triggered | 2025-04-09 11:20:00 | 230.75 | 231.87 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2025-04-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-11 11:10:00 | 233.73 | 232.32 | 0.00 | ORB-long ORB[231.20,233.69] vol=2.8x ATR=0.71 |
| Stop hit — per-position SL triggered | 2025-04-11 11:15:00 | 233.02 | 232.38 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2025-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:30:00 | 246.10 | 244.76 | 0.00 | ORB-long ORB[243.00,245.45] vol=1.8x ATR=0.70 |
| Stop hit — per-position SL triggered | 2025-04-21 10:00:00 | 245.40 | 245.42 | 0.00 | SL hit |

### Cycle 93 — BUY (started 2025-04-22 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 10:25:00 | 254.03 | 252.59 | 0.00 | ORB-long ORB[249.79,252.95] vol=1.9x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 10:35:00 | 255.23 | 252.77 | 0.00 | T1 1.5R @ 255.23 |
| Target hit | 2025-04-22 13:35:00 | 254.22 | 254.32 | 0.00 | Trail-exit close<VWAP |

### Cycle 94 — SELL (started 2025-04-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:05:00 | 249.48 | 250.54 | 0.00 | ORB-short ORB[250.51,253.80] vol=1.9x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 10:15:00 | 248.44 | 250.12 | 0.00 | T1 1.5R @ 248.44 |
| Target hit | 2025-04-23 11:15:00 | 249.35 | 249.32 | 0.00 | Trail-exit close>VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-21 10:20:00 | 264.65 | 2024-05-21 10:30:00 | 263.79 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-05-23 10:50:00 | 270.10 | 2024-05-23 11:40:00 | 271.57 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-05-23 10:50:00 | 270.10 | 2024-05-23 13:10:00 | 270.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-28 11:00:00 | 269.60 | 2024-05-28 11:25:00 | 268.41 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-05-28 11:00:00 | 269.60 | 2024-05-28 15:20:00 | 263.70 | TARGET_HIT | 0.50 | 2.19% |
| BUY | retest1 | 2024-05-29 09:30:00 | 265.00 | 2024-05-29 10:05:00 | 264.12 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-06-07 10:35:00 | 271.90 | 2024-06-07 10:40:00 | 270.86 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-06-11 10:15:00 | 279.10 | 2024-06-11 11:00:00 | 278.13 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-06-12 10:30:00 | 278.55 | 2024-06-12 10:40:00 | 279.75 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-06-12 10:30:00 | 278.55 | 2024-06-12 15:20:00 | 283.65 | TARGET_HIT | 0.50 | 1.83% |
| BUY | retest1 | 2024-06-14 11:10:00 | 284.90 | 2024-06-14 11:25:00 | 284.18 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-06-19 09:35:00 | 285.65 | 2024-06-19 09:50:00 | 286.56 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-06-24 09:30:00 | 275.30 | 2024-06-24 09:40:00 | 273.73 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-06-24 09:30:00 | 275.30 | 2024-06-24 09:45:00 | 275.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-28 09:45:00 | 280.45 | 2024-06-28 10:00:00 | 279.14 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-07-10 09:45:00 | 259.85 | 2024-07-10 09:55:00 | 258.93 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-07-10 09:45:00 | 259.85 | 2024-07-10 15:20:00 | 256.50 | TARGET_HIT | 0.50 | 1.29% |
| SELL | retest1 | 2024-07-11 10:45:00 | 256.40 | 2024-07-11 11:00:00 | 257.16 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-12 09:40:00 | 258.80 | 2024-07-12 09:50:00 | 258.13 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-07-15 10:50:00 | 254.90 | 2024-07-15 11:15:00 | 254.16 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-07-18 09:40:00 | 256.00 | 2024-07-18 09:45:00 | 256.76 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-07-19 10:55:00 | 251.65 | 2024-07-19 11:30:00 | 252.60 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-07-24 09:30:00 | 252.60 | 2024-07-24 09:35:00 | 251.62 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-08-07 11:10:00 | 239.55 | 2024-08-07 12:05:00 | 240.34 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-08-08 11:15:00 | 244.70 | 2024-08-08 11:20:00 | 243.83 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-08-12 10:55:00 | 245.95 | 2024-08-12 12:40:00 | 245.06 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-08-14 09:35:00 | 240.50 | 2024-08-14 09:45:00 | 241.35 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-08-20 09:30:00 | 249.70 | 2024-08-20 09:40:00 | 250.63 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-08-20 09:30:00 | 249.70 | 2024-08-20 15:20:00 | 254.75 | TARGET_HIT | 0.50 | 2.02% |
| SELL | retest1 | 2024-08-21 11:05:00 | 252.25 | 2024-08-21 11:20:00 | 252.82 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-08-22 10:35:00 | 256.35 | 2024-08-22 10:55:00 | 255.69 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-08-26 10:35:00 | 251.80 | 2024-08-26 11:55:00 | 251.04 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-08-26 10:35:00 | 251.80 | 2024-08-26 12:10:00 | 251.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-28 09:35:00 | 249.50 | 2024-08-28 09:40:00 | 250.08 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-08-30 10:40:00 | 252.95 | 2024-08-30 11:20:00 | 252.31 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-09-04 10:30:00 | 244.95 | 2024-09-04 11:05:00 | 245.63 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-09-06 10:05:00 | 240.10 | 2024-09-06 10:10:00 | 240.77 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-09-10 10:40:00 | 235.75 | 2024-09-10 12:00:00 | 236.49 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-09-11 09:30:00 | 236.20 | 2024-09-11 09:45:00 | 237.17 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-09-11 09:30:00 | 236.20 | 2024-09-11 09:50:00 | 236.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-12 11:00:00 | 234.05 | 2024-09-12 11:30:00 | 234.62 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-09-16 10:40:00 | 239.50 | 2024-09-16 11:00:00 | 240.07 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-09-19 10:45:00 | 236.05 | 2024-09-19 11:15:00 | 234.68 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-09-19 10:45:00 | 236.05 | 2024-09-19 15:00:00 | 235.40 | TARGET_HIT | 0.50 | 0.28% |
| BUY | retest1 | 2024-09-20 10:45:00 | 239.35 | 2024-09-20 11:00:00 | 238.67 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-09-23 09:30:00 | 239.90 | 2024-09-23 09:35:00 | 239.10 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-09-25 11:15:00 | 240.85 | 2024-09-25 11:35:00 | 241.31 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-09-26 10:30:00 | 245.20 | 2024-09-26 10:45:00 | 244.55 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-27 09:50:00 | 246.65 | 2024-09-27 10:10:00 | 245.95 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-10-07 10:55:00 | 244.11 | 2024-10-07 11:10:00 | 245.33 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-10-09 09:45:00 | 249.85 | 2024-10-09 09:55:00 | 248.89 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-10-10 10:05:00 | 251.45 | 2024-10-10 10:35:00 | 250.63 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-10-11 10:55:00 | 245.26 | 2024-10-11 11:25:00 | 244.35 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-10-11 10:55:00 | 245.26 | 2024-10-11 15:20:00 | 242.73 | TARGET_HIT | 0.50 | 1.03% |
| BUY | retest1 | 2024-10-14 09:45:00 | 245.00 | 2024-10-14 10:05:00 | 244.31 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-10-17 09:40:00 | 242.90 | 2024-10-17 09:50:00 | 241.85 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-10-17 09:40:00 | 242.90 | 2024-10-17 12:25:00 | 241.89 | TARGET_HIT | 0.50 | 0.42% |
| BUY | retest1 | 2024-10-18 09:55:00 | 241.74 | 2024-10-18 10:10:00 | 243.00 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-10-18 09:55:00 | 241.74 | 2024-10-18 15:20:00 | 247.82 | TARGET_HIT | 0.50 | 2.52% |
| SELL | retest1 | 2024-10-21 11:15:00 | 247.65 | 2024-10-21 11:30:00 | 246.52 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-10-21 11:15:00 | 247.65 | 2024-10-21 15:20:00 | 245.36 | TARGET_HIT | 0.50 | 0.92% |
| BUY | retest1 | 2024-10-24 10:30:00 | 243.26 | 2024-10-24 10:50:00 | 244.88 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2024-10-24 10:30:00 | 243.26 | 2024-10-24 11:15:00 | 243.26 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-25 11:05:00 | 237.77 | 2024-10-25 11:15:00 | 239.15 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2024-11-04 10:45:00 | 249.55 | 2024-11-04 11:15:00 | 248.17 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-11-04 10:45:00 | 249.55 | 2024-11-04 11:20:00 | 249.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-07 09:35:00 | 264.70 | 2024-11-07 09:50:00 | 263.78 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-11-11 10:55:00 | 257.90 | 2024-11-11 13:15:00 | 259.46 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-11-11 10:55:00 | 257.90 | 2024-11-11 13:40:00 | 257.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-12 11:15:00 | 257.15 | 2024-11-12 12:05:00 | 256.01 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-11-12 11:15:00 | 257.15 | 2024-11-12 15:20:00 | 252.35 | TARGET_HIT | 0.50 | 1.87% |
| SELL | retest1 | 2024-11-13 09:45:00 | 248.10 | 2024-11-13 09:50:00 | 249.27 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-11-19 09:35:00 | 244.00 | 2024-11-19 10:05:00 | 245.20 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-11-19 09:35:00 | 244.00 | 2024-11-19 11:00:00 | 244.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-28 09:35:00 | 247.90 | 2024-11-28 09:40:00 | 247.15 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-11-29 10:15:00 | 246.95 | 2024-11-29 11:30:00 | 245.59 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-11-29 10:15:00 | 246.95 | 2024-11-29 14:00:00 | 246.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-04 09:30:00 | 256.39 | 2024-12-04 10:10:00 | 255.78 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-12-06 10:05:00 | 259.48 | 2024-12-06 10:20:00 | 260.50 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-12-13 09:55:00 | 254.25 | 2024-12-13 10:00:00 | 254.97 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-12-17 10:20:00 | 256.87 | 2024-12-17 10:40:00 | 257.45 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-18 11:00:00 | 252.30 | 2024-12-18 12:35:00 | 251.13 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-12-18 11:00:00 | 252.30 | 2024-12-18 13:05:00 | 252.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-20 09:50:00 | 246.85 | 2024-12-20 10:35:00 | 247.62 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-12-23 09:50:00 | 240.60 | 2024-12-23 09:55:00 | 239.11 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-12-23 09:50:00 | 240.60 | 2024-12-23 10:00:00 | 240.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-24 10:05:00 | 247.17 | 2024-12-24 10:40:00 | 246.42 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-01-01 10:55:00 | 241.47 | 2025-01-01 11:10:00 | 240.88 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-01-09 10:40:00 | 233.00 | 2025-01-09 10:45:00 | 232.44 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-01-10 09:35:00 | 227.42 | 2025-01-10 09:40:00 | 228.06 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-01-17 10:55:00 | 226.38 | 2025-01-17 11:15:00 | 226.95 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-01-20 10:25:00 | 230.65 | 2025-01-20 11:15:00 | 231.61 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-01-20 10:25:00 | 230.65 | 2025-01-20 15:15:00 | 231.95 | TARGET_HIT | 0.50 | 0.56% |
| SELL | retest1 | 2025-01-22 11:15:00 | 228.46 | 2025-01-22 12:00:00 | 227.43 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-01-22 11:15:00 | 228.46 | 2025-01-22 14:55:00 | 227.90 | TARGET_HIT | 0.50 | 0.25% |
| SELL | retest1 | 2025-01-24 09:55:00 | 226.12 | 2025-01-24 10:25:00 | 226.82 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-02-01 11:10:00 | 216.22 | 2025-02-01 11:20:00 | 215.63 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-02-04 09:35:00 | 213.15 | 2025-02-04 09:45:00 | 214.29 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-02-04 09:35:00 | 213.15 | 2025-02-04 10:05:00 | 213.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-05 09:40:00 | 217.40 | 2025-02-05 09:45:00 | 216.72 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-02-06 11:15:00 | 219.50 | 2025-02-06 11:20:00 | 220.10 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-02-07 10:15:00 | 215.80 | 2025-02-07 10:20:00 | 216.96 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2025-02-10 09:45:00 | 214.21 | 2025-02-10 09:50:00 | 214.91 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-02-13 10:55:00 | 212.26 | 2025-02-13 13:10:00 | 211.39 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-02-13 10:55:00 | 212.26 | 2025-02-13 15:20:00 | 210.80 | TARGET_HIT | 0.50 | 0.69% |
| SELL | retest1 | 2025-02-21 10:05:00 | 211.78 | 2025-02-21 10:15:00 | 210.80 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-02-21 10:05:00 | 211.78 | 2025-02-21 10:20:00 | 211.78 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-25 10:15:00 | 206.93 | 2025-02-25 12:00:00 | 206.09 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-02-25 10:15:00 | 206.93 | 2025-02-25 15:20:00 | 204.51 | TARGET_HIT | 0.50 | 1.17% |
| BUY | retest1 | 2025-03-06 09:35:00 | 209.47 | 2025-03-06 09:50:00 | 208.71 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-03-10 11:10:00 | 203.70 | 2025-03-10 11:20:00 | 204.37 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-03-13 09:50:00 | 205.76 | 2025-03-13 09:55:00 | 206.78 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-03-13 09:50:00 | 205.76 | 2025-03-13 11:00:00 | 205.76 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-18 10:00:00 | 207.64 | 2025-03-18 10:55:00 | 208.45 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-03-18 10:00:00 | 207.64 | 2025-03-18 15:20:00 | 209.38 | TARGET_HIT | 0.50 | 0.84% |
| BUY | retest1 | 2025-03-19 09:30:00 | 211.85 | 2025-03-19 09:35:00 | 211.31 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-03-24 10:55:00 | 221.50 | 2025-03-24 12:20:00 | 222.31 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-03-24 10:55:00 | 221.50 | 2025-03-24 15:20:00 | 224.42 | TARGET_HIT | 0.50 | 1.32% |
| SELL | retest1 | 2025-03-25 10:30:00 | 221.89 | 2025-03-25 11:40:00 | 220.78 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-03-25 10:30:00 | 221.89 | 2025-03-25 11:55:00 | 221.89 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-09 10:45:00 | 230.75 | 2025-04-09 11:05:00 | 229.48 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-04-09 10:45:00 | 230.75 | 2025-04-09 11:20:00 | 230.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-11 11:10:00 | 233.73 | 2025-04-11 11:15:00 | 233.02 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-04-21 09:30:00 | 246.10 | 2025-04-21 10:00:00 | 245.40 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-04-22 10:25:00 | 254.03 | 2025-04-22 10:35:00 | 255.23 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-04-22 10:25:00 | 254.03 | 2025-04-22 13:35:00 | 254.22 | TARGET_HIT | 0.50 | 0.07% |
| SELL | retest1 | 2025-04-23 10:05:00 | 249.48 | 2025-04-23 10:15:00 | 248.44 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-04-23 10:05:00 | 249.48 | 2025-04-23 11:15:00 | 249.35 | TARGET_HIT | 0.50 | 0.05% |
