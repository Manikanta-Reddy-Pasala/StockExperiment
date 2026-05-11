# Biocon Ltd. (BIOCON)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2024-09-02 15:25:00 (24280 bars)
- **Last close:** 362.00
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
| ENTRY1 | 116 |
| ENTRY2 | 0 |
| PARTIAL | 42 |
| TARGET_HIT | 24 |
| STOP_HIT | 92 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 158 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 66 / 92
- **Target hits / Stop hits / Partials:** 24 / 92 / 42
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 24.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 95 | 40 | 42.1% | 14 | 55 | 26 | 0.12% | 11.5% |
| BUY @ 2nd Alert (retest1) | 95 | 40 | 42.1% | 14 | 55 | 26 | 0.12% | 11.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 63 | 26 | 41.3% | 10 | 37 | 16 | 0.20% | 12.8% |
| SELL @ 2nd Alert (retest1) | 63 | 26 | 41.3% | 10 | 37 | 16 | 0.20% | 12.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 158 | 66 | 41.8% | 24 | 92 | 42 | 0.15% | 24.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-15 10:55:00 | 246.70 | 245.44 | 0.00 | ORB-long ORB[243.50,246.25] vol=4.4x ATR=0.65 |
| Stop hit — per-position SL triggered | 2023-05-15 11:05:00 | 246.05 | 245.60 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 09:35:00 | 242.45 | 244.30 | 0.00 | ORB-short ORB[244.15,246.65] vol=3.6x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-19 09:55:00 | 241.22 | 243.58 | 0.00 | T1 1.5R @ 241.22 |
| Stop hit — per-position SL triggered | 2023-05-19 10:10:00 | 242.45 | 243.28 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-05-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-23 11:00:00 | 242.70 | 241.38 | 0.00 | ORB-long ORB[240.40,242.35] vol=1.9x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-23 11:10:00 | 243.68 | 241.99 | 0.00 | T1 1.5R @ 243.68 |
| Stop hit — per-position SL triggered | 2023-05-23 11:40:00 | 242.70 | 242.55 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-25 09:45:00 | 247.90 | 247.00 | 0.00 | ORB-long ORB[243.45,246.50] vol=2.8x ATR=1.02 |
| Stop hit — per-position SL triggered | 2023-05-25 09:50:00 | 246.88 | 247.01 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2023-05-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-26 09:35:00 | 238.55 | 240.30 | 0.00 | ORB-short ORB[239.80,241.80] vol=1.6x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-26 10:15:00 | 237.28 | 239.42 | 0.00 | T1 1.5R @ 237.28 |
| Stop hit — per-position SL triggered | 2023-05-26 10:50:00 | 238.55 | 238.97 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-05-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-29 10:55:00 | 241.85 | 241.58 | 0.00 | ORB-long ORB[238.50,241.10] vol=1.8x ATR=0.80 |
| Stop hit — per-position SL triggered | 2023-05-29 11:05:00 | 241.05 | 241.57 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-05-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-30 09:30:00 | 240.75 | 241.60 | 0.00 | ORB-short ORB[240.85,242.95] vol=1.5x ATR=0.67 |
| Stop hit — per-position SL triggered | 2023-05-30 10:10:00 | 241.42 | 241.28 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-05-31 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-31 10:30:00 | 243.55 | 242.79 | 0.00 | ORB-long ORB[240.55,242.95] vol=1.5x ATR=0.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-31 11:05:00 | 244.41 | 243.56 | 0.00 | T1 1.5R @ 244.41 |
| Target hit | 2023-05-31 11:40:00 | 243.70 | 243.95 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2023-06-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-07 09:55:00 | 246.40 | 244.98 | 0.00 | ORB-long ORB[243.25,245.85] vol=1.8x ATR=0.71 |
| Stop hit — per-position SL triggered | 2023-06-07 10:10:00 | 245.69 | 245.44 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-06-14 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-14 10:30:00 | 238.55 | 239.24 | 0.00 | ORB-short ORB[238.80,240.50] vol=1.7x ATR=0.45 |
| Stop hit — per-position SL triggered | 2023-06-14 11:20:00 | 239.00 | 238.99 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-15 09:50:00 | 241.70 | 241.07 | 0.00 | ORB-long ORB[239.00,241.60] vol=2.8x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-15 10:20:00 | 242.61 | 241.35 | 0.00 | T1 1.5R @ 242.61 |
| Stop hit — per-position SL triggered | 2023-06-15 11:20:00 | 241.70 | 241.87 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-06-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-16 09:35:00 | 247.30 | 245.77 | 0.00 | ORB-long ORB[243.50,246.30] vol=2.9x ATR=0.69 |
| Stop hit — per-position SL triggered | 2023-06-16 09:45:00 | 246.61 | 246.16 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-06-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-22 10:15:00 | 243.45 | 244.65 | 0.00 | ORB-short ORB[244.45,245.90] vol=1.6x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-22 11:30:00 | 242.62 | 243.86 | 0.00 | T1 1.5R @ 242.62 |
| Target hit | 2023-06-22 15:20:00 | 238.70 | 241.18 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2023-06-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-26 09:35:00 | 240.00 | 238.16 | 0.00 | ORB-long ORB[236.10,238.90] vol=1.6x ATR=0.68 |
| Stop hit — per-position SL triggered | 2023-06-26 10:05:00 | 239.32 | 239.06 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-06-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-27 10:00:00 | 244.85 | 243.19 | 0.00 | ORB-long ORB[240.15,243.35] vol=1.5x ATR=0.80 |
| Stop hit — per-position SL triggered | 2023-06-27 10:05:00 | 244.05 | 243.26 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-30 11:15:00 | 251.75 | 248.88 | 0.00 | ORB-long ORB[246.00,249.45] vol=7.7x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-30 11:20:00 | 253.01 | 249.76 | 0.00 | T1 1.5R @ 253.01 |
| Target hit | 2023-06-30 15:20:00 | 265.90 | 258.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — SELL (started 2023-07-05 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-05 10:30:00 | 259.25 | 261.55 | 0.00 | ORB-short ORB[262.05,264.55] vol=2.9x ATR=0.94 |
| Stop hit — per-position SL triggered | 2023-07-05 11:40:00 | 260.19 | 260.98 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-07-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-07 10:50:00 | 254.80 | 257.62 | 0.00 | ORB-short ORB[256.55,259.95] vol=1.8x ATR=0.66 |
| Stop hit — per-position SL triggered | 2023-07-07 11:05:00 | 255.46 | 257.17 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-07-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-11 11:10:00 | 252.95 | 253.81 | 0.00 | ORB-short ORB[253.15,256.40] vol=2.0x ATR=0.52 |
| Stop hit — per-position SL triggered | 2023-07-11 11:30:00 | 253.47 | 253.74 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-07-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 09:35:00 | 258.70 | 256.79 | 0.00 | ORB-long ORB[254.30,256.25] vol=2.3x ATR=0.73 |
| Stop hit — per-position SL triggered | 2023-07-12 09:40:00 | 257.97 | 257.45 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-07-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-13 10:25:00 | 258.40 | 259.81 | 0.00 | ORB-short ORB[259.00,261.65] vol=2.0x ATR=0.68 |
| Stop hit — per-position SL triggered | 2023-07-13 11:35:00 | 259.08 | 259.29 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-07-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 09:50:00 | 266.25 | 264.72 | 0.00 | ORB-long ORB[262.60,265.50] vol=2.7x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-17 10:20:00 | 267.62 | 265.49 | 0.00 | T1 1.5R @ 267.62 |
| Stop hit — per-position SL triggered | 2023-07-17 10:35:00 | 266.25 | 265.71 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-07-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-18 10:05:00 | 268.15 | 266.92 | 0.00 | ORB-long ORB[265.00,267.50] vol=2.7x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-18 10:25:00 | 269.54 | 267.73 | 0.00 | T1 1.5R @ 269.54 |
| Stop hit — per-position SL triggered | 2023-07-18 10:40:00 | 268.15 | 267.72 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-07-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-19 09:45:00 | 267.55 | 266.83 | 0.00 | ORB-long ORB[265.45,267.50] vol=2.0x ATR=0.69 |
| Stop hit — per-position SL triggered | 2023-07-19 10:05:00 | 266.86 | 266.96 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-07-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-20 10:50:00 | 268.20 | 266.79 | 0.00 | ORB-long ORB[264.90,267.85] vol=1.5x ATR=0.57 |
| Stop hit — per-position SL triggered | 2023-07-20 11:15:00 | 267.63 | 267.05 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-07-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-27 09:50:00 | 253.70 | 252.06 | 0.00 | ORB-long ORB[250.60,252.75] vol=3.1x ATR=0.71 |
| Stop hit — per-position SL triggered | 2023-07-27 11:25:00 | 252.99 | 252.97 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-08-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-02 10:35:00 | 252.90 | 254.58 | 0.00 | ORB-short ORB[253.60,256.70] vol=1.8x ATR=0.64 |
| Stop hit — per-position SL triggered | 2023-08-02 11:50:00 | 253.54 | 253.98 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-08-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-04 10:30:00 | 254.60 | 256.07 | 0.00 | ORB-short ORB[255.10,258.20] vol=1.6x ATR=0.75 |
| Stop hit — per-position SL triggered | 2023-08-04 11:10:00 | 255.35 | 255.74 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2023-08-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-24 09:50:00 | 262.35 | 261.72 | 0.00 | ORB-long ORB[260.40,262.25] vol=2.2x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-24 10:05:00 | 263.19 | 261.99 | 0.00 | T1 1.5R @ 263.19 |
| Stop hit — per-position SL triggered | 2023-08-24 11:30:00 | 262.35 | 262.61 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-08-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 10:30:00 | 257.60 | 259.03 | 0.00 | ORB-short ORB[258.20,260.35] vol=2.0x ATR=0.65 |
| Stop hit — per-position SL triggered | 2023-08-25 11:05:00 | 258.25 | 258.76 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2023-09-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 09:35:00 | 263.10 | 262.50 | 0.00 | ORB-long ORB[261.50,262.85] vol=2.8x ATR=0.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-05 10:00:00 | 263.96 | 263.03 | 0.00 | T1 1.5R @ 263.96 |
| Target hit | 2023-09-05 11:05:00 | 265.15 | 265.16 | 0.00 | Trail-exit close<VWAP |

### Cycle 32 — SELL (started 2023-09-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-07 10:05:00 | 269.40 | 270.74 | 0.00 | ORB-short ORB[270.00,271.75] vol=1.5x ATR=0.79 |
| Stop hit — per-position SL triggered | 2023-09-07 10:10:00 | 270.19 | 270.71 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-11 10:15:00 | 269.10 | 267.99 | 0.00 | ORB-long ORB[267.35,268.45] vol=2.0x ATR=0.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-11 10:25:00 | 269.96 | 268.48 | 0.00 | T1 1.5R @ 269.96 |
| Target hit | 2023-09-11 15:20:00 | 275.55 | 273.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — BUY (started 2023-09-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-14 10:45:00 | 274.00 | 272.73 | 0.00 | ORB-long ORB[271.50,273.25] vol=1.7x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-14 11:05:00 | 275.15 | 273.31 | 0.00 | T1 1.5R @ 275.15 |
| Stop hit — per-position SL triggered | 2023-09-14 11:20:00 | 274.00 | 273.41 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-09-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-15 09:40:00 | 277.00 | 275.61 | 0.00 | ORB-long ORB[273.25,276.10] vol=3.5x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-15 09:50:00 | 278.30 | 276.56 | 0.00 | T1 1.5R @ 278.30 |
| Stop hit — per-position SL triggered | 2023-09-15 10:00:00 | 277.00 | 276.66 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-09-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-20 10:45:00 | 268.95 | 272.61 | 0.00 | ORB-short ORB[273.10,275.65] vol=3.6x ATR=0.91 |
| Stop hit — per-position SL triggered | 2023-09-20 10:55:00 | 269.86 | 272.32 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-09-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-21 09:30:00 | 273.20 | 274.62 | 0.00 | ORB-short ORB[273.60,276.00] vol=1.8x ATR=0.87 |
| Stop hit — per-position SL triggered | 2023-09-21 09:40:00 | 274.07 | 274.49 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-09-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 09:55:00 | 262.05 | 264.92 | 0.00 | ORB-short ORB[265.50,269.25] vol=1.8x ATR=1.31 |
| Stop hit — per-position SL triggered | 2023-09-22 10:45:00 | 263.36 | 263.82 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2023-09-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-26 10:40:00 | 263.55 | 264.98 | 0.00 | ORB-short ORB[264.00,265.50] vol=2.4x ATR=0.61 |
| Stop hit — per-position SL triggered | 2023-09-26 11:10:00 | 264.16 | 264.75 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-09-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-27 09:45:00 | 264.60 | 263.58 | 0.00 | ORB-long ORB[261.65,264.00] vol=1.7x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-27 09:50:00 | 265.53 | 263.85 | 0.00 | T1 1.5R @ 265.53 |
| Stop hit — per-position SL triggered | 2023-09-27 09:55:00 | 264.60 | 263.98 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-09-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-28 09:35:00 | 268.10 | 267.07 | 0.00 | ORB-long ORB[264.25,267.70] vol=3.2x ATR=0.72 |
| Stop hit — per-position SL triggered | 2023-09-28 09:40:00 | 267.38 | 267.15 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2023-09-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 10:40:00 | 268.95 | 267.02 | 0.00 | ORB-long ORB[265.10,268.70] vol=2.6x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-29 10:50:00 | 270.33 | 267.63 | 0.00 | T1 1.5R @ 270.33 |
| Target hit | 2023-09-29 15:20:00 | 272.20 | 270.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2023-10-03 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-03 10:35:00 | 273.55 | 271.52 | 0.00 | ORB-long ORB[269.10,271.45] vol=1.7x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-03 11:05:00 | 274.73 | 272.31 | 0.00 | T1 1.5R @ 274.73 |
| Stop hit — per-position SL triggered | 2023-10-03 11:20:00 | 273.55 | 272.47 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2023-10-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-04 09:30:00 | 269.20 | 270.45 | 0.00 | ORB-short ORB[270.35,271.75] vol=1.9x ATR=0.81 |
| Stop hit — per-position SL triggered | 2023-10-04 09:40:00 | 270.01 | 270.10 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2023-10-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-10 09:30:00 | 259.50 | 260.29 | 0.00 | ORB-short ORB[259.75,261.70] vol=2.4x ATR=0.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-10 10:30:00 | 258.59 | 259.23 | 0.00 | T1 1.5R @ 258.59 |
| Target hit | 2023-10-10 11:25:00 | 258.00 | 257.98 | 0.00 | Trail-exit close>VWAP |

### Cycle 46 — SELL (started 2023-10-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-11 10:00:00 | 258.30 | 258.82 | 0.00 | ORB-short ORB[258.65,259.70] vol=2.2x ATR=0.50 |
| Stop hit — per-position SL triggered | 2023-10-11 11:05:00 | 258.80 | 258.64 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2023-10-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 09:50:00 | 259.85 | 258.89 | 0.00 | ORB-long ORB[257.40,259.05] vol=1.6x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-12 10:10:00 | 260.65 | 259.37 | 0.00 | T1 1.5R @ 260.65 |
| Stop hit — per-position SL triggered | 2023-10-12 11:55:00 | 259.85 | 259.89 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-10-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-13 11:05:00 | 257.60 | 258.62 | 0.00 | ORB-short ORB[257.70,259.45] vol=1.8x ATR=0.51 |
| Stop hit — per-position SL triggered | 2023-10-13 11:45:00 | 258.11 | 258.38 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2023-10-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-20 10:30:00 | 234.35 | 235.11 | 0.00 | ORB-short ORB[234.45,236.10] vol=2.0x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-20 11:40:00 | 233.57 | 234.66 | 0.00 | T1 1.5R @ 233.57 |
| Target hit | 2023-10-20 15:20:00 | 230.75 | 233.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — SELL (started 2023-10-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-23 10:20:00 | 228.95 | 231.41 | 0.00 | ORB-short ORB[229.50,232.95] vol=1.9x ATR=0.82 |
| Stop hit — per-position SL triggered | 2023-10-23 10:25:00 | 229.77 | 231.21 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2023-10-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-25 10:50:00 | 228.40 | 227.35 | 0.00 | ORB-long ORB[225.65,227.90] vol=1.6x ATR=0.62 |
| Stop hit — per-position SL triggered | 2023-10-25 11:00:00 | 227.78 | 227.41 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2023-10-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-30 09:30:00 | 220.75 | 221.94 | 0.00 | ORB-short ORB[221.55,224.05] vol=1.6x ATR=0.62 |
| Stop hit — per-position SL triggered | 2023-10-30 09:40:00 | 221.37 | 221.67 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2023-10-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-31 11:05:00 | 219.60 | 221.29 | 0.00 | ORB-short ORB[222.00,223.45] vol=3.4x ATR=0.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-31 11:40:00 | 218.70 | 220.72 | 0.00 | T1 1.5R @ 218.70 |
| Stop hit — per-position SL triggered | 2023-10-31 13:35:00 | 219.60 | 219.87 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-11-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 09:30:00 | 222.30 | 221.57 | 0.00 | ORB-long ORB[220.25,222.20] vol=1.6x ATR=0.57 |
| Stop hit — per-position SL triggered | 2023-11-02 09:35:00 | 221.73 | 221.65 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2023-11-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 09:45:00 | 228.55 | 227.53 | 0.00 | ORB-long ORB[225.75,227.75] vol=3.3x ATR=0.60 |
| Stop hit — per-position SL triggered | 2023-11-08 10:10:00 | 227.95 | 228.03 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2023-11-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-09 09:30:00 | 226.80 | 227.63 | 0.00 | ORB-short ORB[227.05,229.35] vol=1.8x ATR=0.66 |
| Stop hit — per-position SL triggered | 2023-11-09 11:50:00 | 227.46 | 227.14 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2023-11-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-10 09:30:00 | 222.95 | 224.72 | 0.00 | ORB-short ORB[224.00,227.30] vol=2.9x ATR=0.74 |
| Stop hit — per-position SL triggered | 2023-11-10 09:45:00 | 223.69 | 224.30 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2023-11-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 09:40:00 | 229.90 | 229.38 | 0.00 | ORB-long ORB[228.25,229.60] vol=2.2x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-16 09:50:00 | 230.61 | 230.10 | 0.00 | T1 1.5R @ 230.61 |
| Target hit | 2023-11-16 10:20:00 | 231.40 | 231.71 | 0.00 | Trail-exit close<VWAP |

### Cycle 59 — BUY (started 2023-11-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-20 09:35:00 | 234.75 | 233.57 | 0.00 | ORB-long ORB[232.30,233.85] vol=2.4x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-20 09:45:00 | 235.54 | 234.63 | 0.00 | T1 1.5R @ 235.54 |
| Target hit | 2023-11-20 11:10:00 | 236.30 | 236.45 | 0.00 | Trail-exit close<VWAP |

### Cycle 60 — BUY (started 2023-11-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-22 09:30:00 | 235.75 | 234.97 | 0.00 | ORB-long ORB[233.70,235.10] vol=3.7x ATR=0.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-22 09:35:00 | 236.62 | 235.45 | 0.00 | T1 1.5R @ 236.62 |
| Target hit | 2023-11-22 11:00:00 | 235.80 | 236.15 | 0.00 | Trail-exit close<VWAP |

### Cycle 61 — BUY (started 2023-11-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 09:35:00 | 237.25 | 235.51 | 0.00 | ORB-long ORB[233.20,235.45] vol=1.7x ATR=0.73 |
| Stop hit — per-position SL triggered | 2023-11-29 09:40:00 | 236.52 | 235.62 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2023-11-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 10:55:00 | 238.15 | 237.55 | 0.00 | ORB-long ORB[236.50,237.95] vol=10.2x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-30 15:15:00 | 239.03 | 238.00 | 0.00 | T1 1.5R @ 239.03 |
| Target hit | 2023-11-30 15:20:00 | 238.35 | 238.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — SELL (started 2023-12-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-04 09:50:00 | 240.00 | 240.95 | 0.00 | ORB-short ORB[240.30,243.40] vol=1.6x ATR=0.74 |
| Stop hit — per-position SL triggered | 2023-12-04 10:20:00 | 240.74 | 240.53 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2023-12-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-06 09:40:00 | 242.50 | 243.54 | 0.00 | ORB-short ORB[243.20,244.45] vol=1.6x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-06 10:10:00 | 241.68 | 242.95 | 0.00 | T1 1.5R @ 241.68 |
| Target hit | 2023-12-06 15:20:00 | 240.55 | 241.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 65 — BUY (started 2023-12-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-07 10:15:00 | 243.10 | 241.85 | 0.00 | ORB-long ORB[240.55,241.50] vol=1.6x ATR=0.56 |
| Stop hit — per-position SL triggered | 2023-12-07 11:30:00 | 242.54 | 242.35 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2023-12-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 11:05:00 | 240.80 | 242.34 | 0.00 | ORB-short ORB[242.55,243.80] vol=2.8x ATR=0.48 |
| Stop hit — per-position SL triggered | 2023-12-08 11:15:00 | 241.28 | 242.26 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2023-12-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-12 10:05:00 | 242.80 | 241.86 | 0.00 | ORB-long ORB[240.45,242.50] vol=2.2x ATR=0.56 |
| Stop hit — per-position SL triggered | 2023-12-12 10:10:00 | 242.24 | 241.87 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2023-12-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 09:30:00 | 253.50 | 254.78 | 0.00 | ORB-short ORB[254.00,256.90] vol=2.1x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-19 09:40:00 | 252.37 | 254.11 | 0.00 | T1 1.5R @ 252.37 |
| Target hit | 2023-12-19 12:35:00 | 251.90 | 251.56 | 0.00 | Trail-exit close>VWAP |

### Cycle 69 — BUY (started 2023-12-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 11:00:00 | 255.10 | 253.44 | 0.00 | ORB-long ORB[252.55,254.50] vol=4.3x ATR=0.69 |
| Stop hit — per-position SL triggered | 2023-12-20 11:10:00 | 254.41 | 253.66 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2023-12-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 09:30:00 | 247.95 | 246.90 | 0.00 | ORB-long ORB[245.45,247.25] vol=2.1x ATR=0.71 |
| Stop hit — per-position SL triggered | 2023-12-22 09:40:00 | 247.24 | 247.02 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2023-12-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 09:30:00 | 251.30 | 249.64 | 0.00 | ORB-long ORB[247.75,249.80] vol=4.6x ATR=1.01 |
| Stop hit — per-position SL triggered | 2023-12-26 09:40:00 | 250.29 | 250.05 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2023-12-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 10:05:00 | 254.05 | 252.76 | 0.00 | ORB-long ORB[251.00,252.85] vol=1.8x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-27 10:45:00 | 255.31 | 253.57 | 0.00 | T1 1.5R @ 255.31 |
| Stop hit — per-position SL triggered | 2023-12-27 10:55:00 | 254.05 | 253.62 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2024-01-01 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-01 10:05:00 | 252.05 | 251.27 | 0.00 | ORB-long ORB[249.65,251.90] vol=1.9x ATR=0.56 |
| Stop hit — per-position SL triggered | 2024-01-01 10:30:00 | 251.49 | 251.61 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2024-01-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-03 09:45:00 | 269.25 | 268.23 | 0.00 | ORB-long ORB[265.60,269.20] vol=1.9x ATR=1.45 |
| Stop hit — per-position SL triggered | 2024-01-03 09:55:00 | 267.80 | 268.31 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2024-01-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 10:35:00 | 286.80 | 284.47 | 0.00 | ORB-long ORB[282.35,286.60] vol=2.1x ATR=1.44 |
| Stop hit — per-position SL triggered | 2024-01-04 10:40:00 | 285.36 | 284.53 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2024-01-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 09:40:00 | 286.00 | 283.75 | 0.00 | ORB-long ORB[282.35,285.35] vol=2.5x ATR=1.22 |
| Stop hit — per-position SL triggered | 2024-01-05 09:45:00 | 284.78 | 283.95 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2024-01-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-10 09:40:00 | 284.55 | 282.53 | 0.00 | ORB-long ORB[280.05,283.90] vol=2.3x ATR=1.41 |
| Stop hit — per-position SL triggered | 2024-01-10 09:45:00 | 283.14 | 282.66 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2024-01-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 10:10:00 | 285.65 | 283.81 | 0.00 | ORB-long ORB[282.40,284.50] vol=2.8x ATR=0.84 |
| Stop hit — per-position SL triggered | 2024-01-11 10:30:00 | 284.81 | 284.71 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2024-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-15 10:15:00 | 281.90 | 277.59 | 0.00 | ORB-long ORB[273.80,276.40] vol=1.6x ATR=1.59 |
| Stop hit — per-position SL triggered | 2024-01-15 11:20:00 | 280.31 | 279.28 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2024-01-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-19 10:00:00 | 281.00 | 281.56 | 0.00 | ORB-short ORB[281.15,283.45] vol=3.6x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-19 10:50:00 | 279.41 | 281.21 | 0.00 | T1 1.5R @ 279.41 |
| Target hit | 2024-01-19 15:20:00 | 275.65 | 278.76 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 81 — SELL (started 2024-01-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-20 09:55:00 | 273.40 | 276.12 | 0.00 | ORB-short ORB[276.60,278.75] vol=2.2x ATR=1.07 |
| Stop hit — per-position SL triggered | 2024-01-20 10:05:00 | 274.47 | 275.52 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2024-01-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-23 09:45:00 | 271.95 | 273.72 | 0.00 | ORB-short ORB[272.55,276.45] vol=2.3x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-23 10:00:00 | 270.20 | 272.92 | 0.00 | T1 1.5R @ 270.20 |
| Target hit | 2024-01-23 15:20:00 | 263.30 | 268.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 83 — BUY (started 2024-01-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-24 11:10:00 | 264.65 | 264.33 | 0.00 | ORB-long ORB[261.00,263.95] vol=2.1x ATR=1.06 |
| Stop hit — per-position SL triggered | 2024-01-24 11:25:00 | 263.59 | 264.32 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2024-01-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-30 09:35:00 | 263.80 | 262.64 | 0.00 | ORB-long ORB[261.45,263.00] vol=1.9x ATR=0.83 |
| Stop hit — per-position SL triggered | 2024-01-30 09:40:00 | 262.97 | 262.75 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2024-01-31 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 10:45:00 | 264.05 | 262.41 | 0.00 | ORB-long ORB[260.00,263.05] vol=1.7x ATR=0.82 |
| Stop hit — per-position SL triggered | 2024-01-31 11:05:00 | 263.23 | 262.92 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2024-02-02 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 10:25:00 | 274.85 | 272.57 | 0.00 | ORB-long ORB[270.60,272.70] vol=2.0x ATR=0.94 |
| Stop hit — per-position SL triggered | 2024-02-02 10:30:00 | 273.91 | 272.68 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2024-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-05 11:15:00 | 282.45 | 278.72 | 0.00 | ORB-long ORB[275.35,279.35] vol=5.3x ATR=1.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-05 11:30:00 | 284.57 | 279.49 | 0.00 | T1 1.5R @ 284.57 |
| Target hit | 2024-02-05 14:55:00 | 282.95 | 283.68 | 0.00 | Trail-exit close<VWAP |

### Cycle 88 — SELL (started 2024-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-13 09:30:00 | 262.60 | 266.27 | 0.00 | ORB-short ORB[266.25,269.75] vol=3.0x ATR=1.79 |
| Stop hit — per-position SL triggered | 2024-02-13 09:35:00 | 264.39 | 265.76 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2024-02-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-16 10:00:00 | 280.00 | 278.42 | 0.00 | ORB-long ORB[275.55,279.40] vol=3.9x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-16 10:15:00 | 281.76 | 279.54 | 0.00 | T1 1.5R @ 281.76 |
| Target hit | 2024-02-16 11:05:00 | 280.65 | 280.83 | 0.00 | Trail-exit close<VWAP |

### Cycle 90 — SELL (started 2024-02-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-21 09:30:00 | 278.85 | 281.08 | 0.00 | ORB-short ORB[280.50,283.80] vol=2.6x ATR=1.33 |
| Stop hit — per-position SL triggered | 2024-02-21 09:45:00 | 280.18 | 280.63 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2024-02-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-23 09:55:00 | 274.10 | 271.80 | 0.00 | ORB-long ORB[270.10,272.70] vol=2.0x ATR=0.95 |
| Stop hit — per-position SL triggered | 2024-02-23 10:00:00 | 273.15 | 271.91 | 0.00 | SL hit |

### Cycle 92 — SELL (started 2024-02-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-26 10:50:00 | 270.25 | 273.91 | 0.00 | ORB-short ORB[273.80,275.85] vol=2.4x ATR=1.10 |
| Stop hit — per-position SL triggered | 2024-02-26 11:05:00 | 271.35 | 273.56 | 0.00 | SL hit |

### Cycle 93 — BUY (started 2024-02-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-28 09:35:00 | 279.00 | 276.89 | 0.00 | ORB-long ORB[275.00,277.20] vol=2.6x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-28 09:40:00 | 280.38 | 278.16 | 0.00 | T1 1.5R @ 280.38 |
| Stop hit — per-position SL triggered | 2024-02-28 09:55:00 | 279.00 | 278.78 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2024-03-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 09:30:00 | 276.65 | 278.31 | 0.00 | ORB-short ORB[277.70,280.85] vol=2.2x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 09:50:00 | 275.29 | 276.64 | 0.00 | T1 1.5R @ 275.29 |
| Target hit | 2024-03-06 13:40:00 | 273.10 | 272.76 | 0.00 | Trail-exit close>VWAP |

### Cycle 95 — BUY (started 2024-03-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-07 10:35:00 | 281.40 | 279.03 | 0.00 | ORB-long ORB[276.00,279.85] vol=1.7x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-07 10:45:00 | 283.06 | 279.80 | 0.00 | T1 1.5R @ 283.06 |
| Target hit | 2024-03-07 12:10:00 | 281.60 | 281.70 | 0.00 | Trail-exit close<VWAP |

### Cycle 96 — BUY (started 2024-03-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-11 09:50:00 | 285.00 | 283.26 | 0.00 | ORB-long ORB[280.85,284.45] vol=1.6x ATR=0.97 |
| Stop hit — per-position SL triggered | 2024-03-11 09:55:00 | 284.03 | 283.42 | 0.00 | SL hit |

### Cycle 97 — SELL (started 2024-03-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-18 11:05:00 | 250.00 | 251.97 | 0.00 | ORB-short ORB[251.25,254.55] vol=1.9x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-18 11:15:00 | 248.69 | 251.66 | 0.00 | T1 1.5R @ 248.69 |
| Stop hit — per-position SL triggered | 2024-03-18 12:00:00 | 250.00 | 251.24 | 0.00 | SL hit |

### Cycle 98 — SELL (started 2024-03-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 10:05:00 | 249.60 | 250.88 | 0.00 | ORB-short ORB[250.00,252.35] vol=2.3x ATR=0.88 |
| Stop hit — per-position SL triggered | 2024-03-19 10:10:00 | 250.48 | 250.87 | 0.00 | SL hit |

### Cycle 99 — BUY (started 2024-03-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 10:00:00 | 251.00 | 249.62 | 0.00 | ORB-long ORB[248.50,250.00] vol=1.6x ATR=0.87 |
| Stop hit — per-position SL triggered | 2024-03-21 10:30:00 | 250.13 | 249.96 | 0.00 | SL hit |

### Cycle 100 — BUY (started 2024-03-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 11:10:00 | 263.35 | 261.12 | 0.00 | ORB-long ORB[260.00,263.30] vol=4.0x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-28 11:15:00 | 264.64 | 261.75 | 0.00 | T1 1.5R @ 264.64 |
| Target hit | 2024-03-28 15:20:00 | 264.15 | 263.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 101 — BUY (started 2024-04-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 11:05:00 | 269.45 | 267.74 | 0.00 | ORB-long ORB[265.50,268.65] vol=2.1x ATR=0.83 |
| Stop hit — per-position SL triggered | 2024-04-01 11:10:00 | 268.62 | 267.84 | 0.00 | SL hit |

### Cycle 102 — BUY (started 2024-04-02 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 10:20:00 | 270.50 | 269.59 | 0.00 | ORB-long ORB[268.45,270.10] vol=2.1x ATR=0.75 |
| Stop hit — per-position SL triggered | 2024-04-02 10:25:00 | 269.75 | 269.59 | 0.00 | SL hit |

### Cycle 103 — SELL (started 2024-04-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 09:50:00 | 271.30 | 273.22 | 0.00 | ORB-short ORB[273.15,274.85] vol=1.7x ATR=0.85 |
| Stop hit — per-position SL triggered | 2024-04-04 10:05:00 | 272.15 | 272.95 | 0.00 | SL hit |

### Cycle 104 — BUY (started 2024-04-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-05 09:30:00 | 271.25 | 270.36 | 0.00 | ORB-long ORB[269.00,271.00] vol=2.2x ATR=0.71 |
| Stop hit — per-position SL triggered | 2024-04-05 09:35:00 | 270.54 | 270.41 | 0.00 | SL hit |

### Cycle 105 — SELL (started 2024-04-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-08 09:30:00 | 269.50 | 270.88 | 0.00 | ORB-short ORB[270.00,272.80] vol=1.6x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-08 09:45:00 | 268.36 | 270.33 | 0.00 | T1 1.5R @ 268.36 |
| Stop hit — per-position SL triggered | 2024-04-08 11:40:00 | 269.50 | 269.43 | 0.00 | SL hit |

### Cycle 106 — SELL (started 2024-04-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-12 10:45:00 | 278.10 | 279.71 | 0.00 | ORB-short ORB[278.55,282.00] vol=2.1x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-12 10:55:00 | 276.45 | 279.35 | 0.00 | T1 1.5R @ 276.45 |
| Target hit | 2024-04-12 15:20:00 | 273.95 | 277.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 107 — BUY (started 2024-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 09:30:00 | 265.50 | 264.15 | 0.00 | ORB-long ORB[262.25,265.20] vol=2.7x ATR=1.10 |
| Stop hit — per-position SL triggered | 2024-04-16 10:20:00 | 264.40 | 264.73 | 0.00 | SL hit |

### Cycle 108 — BUY (started 2024-04-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-22 10:45:00 | 265.35 | 263.93 | 0.00 | ORB-long ORB[263.50,265.00] vol=1.7x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-22 10:50:00 | 266.73 | 264.23 | 0.00 | T1 1.5R @ 266.73 |
| Target hit | 2024-04-22 15:20:00 | 271.80 | 268.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 109 — BUY (started 2024-04-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 09:35:00 | 286.85 | 284.06 | 0.00 | ORB-long ORB[281.40,284.30] vol=2.2x ATR=1.43 |
| Stop hit — per-position SL triggered | 2024-04-24 09:45:00 | 285.42 | 284.84 | 0.00 | SL hit |

### Cycle 110 — SELL (started 2024-04-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-25 09:50:00 | 289.50 | 291.40 | 0.00 | ORB-short ORB[289.80,293.25] vol=2.8x ATR=1.22 |
| Stop hit — per-position SL triggered | 2024-04-25 09:55:00 | 290.72 | 291.36 | 0.00 | SL hit |

### Cycle 111 — SELL (started 2024-04-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-29 10:50:00 | 303.85 | 308.68 | 0.00 | ORB-short ORB[309.60,313.35] vol=3.1x ATR=1.83 |
| Stop hit — per-position SL triggered | 2024-04-29 11:20:00 | 305.68 | 307.90 | 0.00 | SL hit |

### Cycle 112 — SELL (started 2024-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-30 11:15:00 | 303.95 | 305.33 | 0.00 | ORB-short ORB[304.50,307.00] vol=1.8x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-30 12:05:00 | 302.59 | 304.91 | 0.00 | T1 1.5R @ 302.59 |
| Target hit | 2024-04-30 15:20:00 | 298.40 | 302.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 113 — BUY (started 2024-05-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-02 10:05:00 | 304.10 | 300.27 | 0.00 | ORB-long ORB[296.50,300.00] vol=1.7x ATR=1.34 |
| Stop hit — per-position SL triggered | 2024-05-02 10:15:00 | 302.76 | 300.60 | 0.00 | SL hit |

### Cycle 114 — SELL (started 2024-05-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-06 09:35:00 | 303.60 | 306.33 | 0.00 | ORB-short ORB[305.10,309.00] vol=2.4x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-06 09:40:00 | 301.22 | 305.04 | 0.00 | T1 1.5R @ 301.22 |
| Stop hit — per-position SL triggered | 2024-05-06 10:35:00 | 303.60 | 303.37 | 0.00 | SL hit |

### Cycle 115 — BUY (started 2024-05-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-08 09:35:00 | 302.00 | 299.32 | 0.00 | ORB-long ORB[296.00,300.40] vol=2.2x ATR=1.72 |
| Stop hit — per-position SL triggered | 2024-05-08 10:30:00 | 300.28 | 300.35 | 0.00 | SL hit |

### Cycle 116 — BUY (started 2024-05-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-09 09:30:00 | 311.10 | 309.17 | 0.00 | ORB-long ORB[307.20,311.00] vol=1.7x ATR=1.35 |
| Stop hit — per-position SL triggered | 2024-05-09 09:35:00 | 309.75 | 309.41 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-15 10:55:00 | 246.70 | 2023-05-15 11:05:00 | 246.05 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-05-19 09:35:00 | 242.45 | 2023-05-19 09:55:00 | 241.22 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2023-05-19 09:35:00 | 242.45 | 2023-05-19 10:10:00 | 242.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-23 11:00:00 | 242.70 | 2023-05-23 11:10:00 | 243.68 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-05-23 11:00:00 | 242.70 | 2023-05-23 11:40:00 | 242.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-25 09:45:00 | 247.90 | 2023-05-25 09:50:00 | 246.88 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2023-05-26 09:35:00 | 238.55 | 2023-05-26 10:15:00 | 237.28 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2023-05-26 09:35:00 | 238.55 | 2023-05-26 10:50:00 | 238.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-29 10:55:00 | 241.85 | 2023-05-29 11:05:00 | 241.05 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-05-30 09:30:00 | 240.75 | 2023-05-30 10:10:00 | 241.42 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-05-31 10:30:00 | 243.55 | 2023-05-31 11:05:00 | 244.41 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-05-31 10:30:00 | 243.55 | 2023-05-31 11:40:00 | 243.70 | TARGET_HIT | 0.50 | 0.06% |
| BUY | retest1 | 2023-06-07 09:55:00 | 246.40 | 2023-06-07 10:10:00 | 245.69 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-06-14 10:30:00 | 238.55 | 2023-06-14 11:20:00 | 239.00 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-06-15 09:50:00 | 241.70 | 2023-06-15 10:20:00 | 242.61 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-06-15 09:50:00 | 241.70 | 2023-06-15 11:20:00 | 241.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-16 09:35:00 | 247.30 | 2023-06-16 09:45:00 | 246.61 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-06-22 10:15:00 | 243.45 | 2023-06-22 11:30:00 | 242.62 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-06-22 10:15:00 | 243.45 | 2023-06-22 15:20:00 | 238.70 | TARGET_HIT | 0.50 | 1.95% |
| BUY | retest1 | 2023-06-26 09:35:00 | 240.00 | 2023-06-26 10:05:00 | 239.32 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-06-27 10:00:00 | 244.85 | 2023-06-27 10:05:00 | 244.05 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-06-30 11:15:00 | 251.75 | 2023-06-30 11:20:00 | 253.01 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2023-06-30 11:15:00 | 251.75 | 2023-06-30 15:20:00 | 265.90 | TARGET_HIT | 0.50 | 5.62% |
| SELL | retest1 | 2023-07-05 10:30:00 | 259.25 | 2023-07-05 11:40:00 | 260.19 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2023-07-07 10:50:00 | 254.80 | 2023-07-07 11:05:00 | 255.46 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-07-11 11:10:00 | 252.95 | 2023-07-11 11:30:00 | 253.47 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-07-12 09:35:00 | 258.70 | 2023-07-12 09:40:00 | 257.97 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-07-13 10:25:00 | 258.40 | 2023-07-13 11:35:00 | 259.08 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-07-17 09:50:00 | 266.25 | 2023-07-17 10:20:00 | 267.62 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-07-17 09:50:00 | 266.25 | 2023-07-17 10:35:00 | 266.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-18 10:05:00 | 268.15 | 2023-07-18 10:25:00 | 269.54 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-07-18 10:05:00 | 268.15 | 2023-07-18 10:40:00 | 268.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-19 09:45:00 | 267.55 | 2023-07-19 10:05:00 | 266.86 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-07-20 10:50:00 | 268.20 | 2023-07-20 11:15:00 | 267.63 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-07-27 09:50:00 | 253.70 | 2023-07-27 11:25:00 | 252.99 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-08-02 10:35:00 | 252.90 | 2023-08-02 11:50:00 | 253.54 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-08-04 10:30:00 | 254.60 | 2023-08-04 11:10:00 | 255.35 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-08-24 09:50:00 | 262.35 | 2023-08-24 10:05:00 | 263.19 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-08-24 09:50:00 | 262.35 | 2023-08-24 11:30:00 | 262.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-25 10:30:00 | 257.60 | 2023-08-25 11:05:00 | 258.25 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-09-05 09:35:00 | 263.10 | 2023-09-05 10:00:00 | 263.96 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-09-05 09:35:00 | 263.10 | 2023-09-05 11:05:00 | 265.15 | TARGET_HIT | 0.50 | 0.78% |
| SELL | retest1 | 2023-09-07 10:05:00 | 269.40 | 2023-09-07 10:10:00 | 270.19 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-09-11 10:15:00 | 269.10 | 2023-09-11 10:25:00 | 269.96 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-09-11 10:15:00 | 269.10 | 2023-09-11 15:20:00 | 275.55 | TARGET_HIT | 0.50 | 2.40% |
| BUY | retest1 | 2023-09-14 10:45:00 | 274.00 | 2023-09-14 11:05:00 | 275.15 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2023-09-14 10:45:00 | 274.00 | 2023-09-14 11:20:00 | 274.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-15 09:40:00 | 277.00 | 2023-09-15 09:50:00 | 278.30 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2023-09-15 09:40:00 | 277.00 | 2023-09-15 10:00:00 | 277.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-20 10:45:00 | 268.95 | 2023-09-20 10:55:00 | 269.86 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-09-21 09:30:00 | 273.20 | 2023-09-21 09:40:00 | 274.07 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-09-22 09:55:00 | 262.05 | 2023-09-22 10:45:00 | 263.36 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2023-09-26 10:40:00 | 263.55 | 2023-09-26 11:10:00 | 264.16 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-09-27 09:45:00 | 264.60 | 2023-09-27 09:50:00 | 265.53 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-09-27 09:45:00 | 264.60 | 2023-09-27 09:55:00 | 264.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-28 09:35:00 | 268.10 | 2023-09-28 09:40:00 | 267.38 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-09-29 10:40:00 | 268.95 | 2023-09-29 10:50:00 | 270.33 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2023-09-29 10:40:00 | 268.95 | 2023-09-29 15:20:00 | 272.20 | TARGET_HIT | 0.50 | 1.21% |
| BUY | retest1 | 2023-10-03 10:35:00 | 273.55 | 2023-10-03 11:05:00 | 274.73 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2023-10-03 10:35:00 | 273.55 | 2023-10-03 11:20:00 | 273.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-04 09:30:00 | 269.20 | 2023-10-04 09:40:00 | 270.01 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-10-10 09:30:00 | 259.50 | 2023-10-10 10:30:00 | 258.59 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2023-10-10 09:30:00 | 259.50 | 2023-10-10 11:25:00 | 258.00 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2023-10-11 10:00:00 | 258.30 | 2023-10-11 11:05:00 | 258.80 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-10-12 09:50:00 | 259.85 | 2023-10-12 10:10:00 | 260.65 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-10-12 09:50:00 | 259.85 | 2023-10-12 11:55:00 | 259.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-13 11:05:00 | 257.60 | 2023-10-13 11:45:00 | 258.11 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-10-20 10:30:00 | 234.35 | 2023-10-20 11:40:00 | 233.57 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2023-10-20 10:30:00 | 234.35 | 2023-10-20 15:20:00 | 230.75 | TARGET_HIT | 0.50 | 1.54% |
| SELL | retest1 | 2023-10-23 10:20:00 | 228.95 | 2023-10-23 10:25:00 | 229.77 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-10-25 10:50:00 | 228.40 | 2023-10-25 11:00:00 | 227.78 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-10-30 09:30:00 | 220.75 | 2023-10-30 09:40:00 | 221.37 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-10-31 11:05:00 | 219.60 | 2023-10-31 11:40:00 | 218.70 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2023-10-31 11:05:00 | 219.60 | 2023-10-31 13:35:00 | 219.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-02 09:30:00 | 222.30 | 2023-11-02 09:35:00 | 221.73 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-11-08 09:45:00 | 228.55 | 2023-11-08 10:10:00 | 227.95 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-11-09 09:30:00 | 226.80 | 2023-11-09 11:50:00 | 227.46 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-11-10 09:30:00 | 222.95 | 2023-11-10 09:45:00 | 223.69 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-11-16 09:40:00 | 229.90 | 2023-11-16 09:50:00 | 230.61 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-11-16 09:40:00 | 229.90 | 2023-11-16 10:20:00 | 231.40 | TARGET_HIT | 0.50 | 0.65% |
| BUY | retest1 | 2023-11-20 09:35:00 | 234.75 | 2023-11-20 09:45:00 | 235.54 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2023-11-20 09:35:00 | 234.75 | 2023-11-20 11:10:00 | 236.30 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2023-11-22 09:30:00 | 235.75 | 2023-11-22 09:35:00 | 236.62 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2023-11-22 09:30:00 | 235.75 | 2023-11-22 11:00:00 | 235.80 | TARGET_HIT | 0.50 | 0.02% |
| BUY | retest1 | 2023-11-29 09:35:00 | 237.25 | 2023-11-29 09:40:00 | 236.52 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-11-30 10:55:00 | 238.15 | 2023-11-30 15:15:00 | 239.03 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2023-11-30 10:55:00 | 238.15 | 2023-11-30 15:20:00 | 238.35 | TARGET_HIT | 0.50 | 0.08% |
| SELL | retest1 | 2023-12-04 09:50:00 | 240.00 | 2023-12-04 10:20:00 | 240.74 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-12-06 09:40:00 | 242.50 | 2023-12-06 10:10:00 | 241.68 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-12-06 09:40:00 | 242.50 | 2023-12-06 15:20:00 | 240.55 | TARGET_HIT | 0.50 | 0.80% |
| BUY | retest1 | 2023-12-07 10:15:00 | 243.10 | 2023-12-07 11:30:00 | 242.54 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-12-08 11:05:00 | 240.80 | 2023-12-08 11:15:00 | 241.28 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-12-12 10:05:00 | 242.80 | 2023-12-12 10:10:00 | 242.24 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-12-19 09:30:00 | 253.50 | 2023-12-19 09:40:00 | 252.37 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2023-12-19 09:30:00 | 253.50 | 2023-12-19 12:35:00 | 251.90 | TARGET_HIT | 0.50 | 0.63% |
| BUY | retest1 | 2023-12-20 11:00:00 | 255.10 | 2023-12-20 11:10:00 | 254.41 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-12-22 09:30:00 | 247.95 | 2023-12-22 09:40:00 | 247.24 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-12-26 09:30:00 | 251.30 | 2023-12-26 09:40:00 | 250.29 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2023-12-27 10:05:00 | 254.05 | 2023-12-27 10:45:00 | 255.31 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2023-12-27 10:05:00 | 254.05 | 2023-12-27 10:55:00 | 254.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-01 10:05:00 | 252.05 | 2024-01-01 10:30:00 | 251.49 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-01-03 09:45:00 | 269.25 | 2024-01-03 09:55:00 | 267.80 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2024-01-04 10:35:00 | 286.80 | 2024-01-04 10:40:00 | 285.36 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-01-05 09:40:00 | 286.00 | 2024-01-05 09:45:00 | 284.78 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-01-10 09:40:00 | 284.55 | 2024-01-10 09:45:00 | 283.14 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-01-11 10:10:00 | 285.65 | 2024-01-11 10:30:00 | 284.81 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-01-15 10:15:00 | 281.90 | 2024-01-15 11:20:00 | 280.31 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest1 | 2024-01-19 10:00:00 | 281.00 | 2024-01-19 10:50:00 | 279.41 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-01-19 10:00:00 | 281.00 | 2024-01-19 15:20:00 | 275.65 | TARGET_HIT | 0.50 | 1.90% |
| SELL | retest1 | 2024-01-20 09:55:00 | 273.40 | 2024-01-20 10:05:00 | 274.47 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-01-23 09:45:00 | 271.95 | 2024-01-23 10:00:00 | 270.20 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-01-23 09:45:00 | 271.95 | 2024-01-23 15:20:00 | 263.30 | TARGET_HIT | 0.50 | 3.18% |
| BUY | retest1 | 2024-01-24 11:10:00 | 264.65 | 2024-01-24 11:25:00 | 263.59 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-01-30 09:35:00 | 263.80 | 2024-01-30 09:40:00 | 262.97 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-01-31 10:45:00 | 264.05 | 2024-01-31 11:05:00 | 263.23 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-02-02 10:25:00 | 274.85 | 2024-02-02 10:30:00 | 273.91 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-02-05 11:15:00 | 282.45 | 2024-02-05 11:30:00 | 284.57 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2024-02-05 11:15:00 | 282.45 | 2024-02-05 14:55:00 | 282.95 | TARGET_HIT | 0.50 | 0.18% |
| SELL | retest1 | 2024-02-13 09:30:00 | 262.60 | 2024-02-13 09:35:00 | 264.39 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest1 | 2024-02-16 10:00:00 | 280.00 | 2024-02-16 10:15:00 | 281.76 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-02-16 10:00:00 | 280.00 | 2024-02-16 11:05:00 | 280.65 | TARGET_HIT | 0.50 | 0.23% |
| SELL | retest1 | 2024-02-21 09:30:00 | 278.85 | 2024-02-21 09:45:00 | 280.18 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-02-23 09:55:00 | 274.10 | 2024-02-23 10:00:00 | 273.15 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-02-26 10:50:00 | 270.25 | 2024-02-26 11:05:00 | 271.35 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-02-28 09:35:00 | 279.00 | 2024-02-28 09:40:00 | 280.38 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-02-28 09:35:00 | 279.00 | 2024-02-28 09:55:00 | 279.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-06 09:30:00 | 276.65 | 2024-03-06 09:50:00 | 275.29 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-03-06 09:30:00 | 276.65 | 2024-03-06 13:40:00 | 273.10 | TARGET_HIT | 0.50 | 1.28% |
| BUY | retest1 | 2024-03-07 10:35:00 | 281.40 | 2024-03-07 10:45:00 | 283.06 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-03-07 10:35:00 | 281.40 | 2024-03-07 12:10:00 | 281.60 | TARGET_HIT | 0.50 | 0.07% |
| BUY | retest1 | 2024-03-11 09:50:00 | 285.00 | 2024-03-11 09:55:00 | 284.03 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-03-18 11:05:00 | 250.00 | 2024-03-18 11:15:00 | 248.69 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-03-18 11:05:00 | 250.00 | 2024-03-18 12:00:00 | 250.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-19 10:05:00 | 249.60 | 2024-03-19 10:10:00 | 250.48 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-03-21 10:00:00 | 251.00 | 2024-03-21 10:30:00 | 250.13 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-03-28 11:10:00 | 263.35 | 2024-03-28 11:15:00 | 264.64 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-03-28 11:10:00 | 263.35 | 2024-03-28 15:20:00 | 264.15 | TARGET_HIT | 0.50 | 0.30% |
| BUY | retest1 | 2024-04-01 11:05:00 | 269.45 | 2024-04-01 11:10:00 | 268.62 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-04-02 10:20:00 | 270.50 | 2024-04-02 10:25:00 | 269.75 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-04-04 09:50:00 | 271.30 | 2024-04-04 10:05:00 | 272.15 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-04-05 09:30:00 | 271.25 | 2024-04-05 09:35:00 | 270.54 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-04-08 09:30:00 | 269.50 | 2024-04-08 09:45:00 | 268.36 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-04-08 09:30:00 | 269.50 | 2024-04-08 11:40:00 | 269.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-12 10:45:00 | 278.10 | 2024-04-12 10:55:00 | 276.45 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-04-12 10:45:00 | 278.10 | 2024-04-12 15:20:00 | 273.95 | TARGET_HIT | 0.50 | 1.49% |
| BUY | retest1 | 2024-04-16 09:30:00 | 265.50 | 2024-04-16 10:20:00 | 264.40 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-04-22 10:45:00 | 265.35 | 2024-04-22 10:50:00 | 266.73 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-04-22 10:45:00 | 265.35 | 2024-04-22 15:20:00 | 271.80 | TARGET_HIT | 0.50 | 2.43% |
| BUY | retest1 | 2024-04-24 09:35:00 | 286.85 | 2024-04-24 09:45:00 | 285.42 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-04-25 09:50:00 | 289.50 | 2024-04-25 09:55:00 | 290.72 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-04-29 10:50:00 | 303.85 | 2024-04-29 11:20:00 | 305.68 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest1 | 2024-04-30 11:15:00 | 303.95 | 2024-04-30 12:05:00 | 302.59 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-04-30 11:15:00 | 303.95 | 2024-04-30 15:20:00 | 298.40 | TARGET_HIT | 0.50 | 1.83% |
| BUY | retest1 | 2024-05-02 10:05:00 | 304.10 | 2024-05-02 10:15:00 | 302.76 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-05-06 09:35:00 | 303.60 | 2024-05-06 09:40:00 | 301.22 | PARTIAL | 0.50 | 0.78% |
| SELL | retest1 | 2024-05-06 09:35:00 | 303.60 | 2024-05-06 10:35:00 | 303.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-08 09:35:00 | 302.00 | 2024-05-08 10:30:00 | 300.28 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2024-05-09 09:30:00 | 311.10 | 2024-05-09 09:35:00 | 309.75 | STOP_HIT | 1.00 | -0.44% |
