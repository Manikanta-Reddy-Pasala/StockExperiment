# Oil & Natural Gas Corporation Ltd. (ONGC)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 279.00
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
| ENTRY1 | 79 |
| ENTRY2 | 0 |
| PARTIAL | 32 |
| TARGET_HIT | 14 |
| STOP_HIT | 65 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 111 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 46 / 65
- **Target hits / Stop hits / Partials:** 14 / 65 / 32
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 9.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 60 | 23 | 38.3% | 5 | 37 | 18 | 0.04% | 2.5% |
| BUY @ 2nd Alert (retest1) | 60 | 23 | 38.3% | 5 | 37 | 18 | 0.04% | 2.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 51 | 23 | 45.1% | 9 | 28 | 14 | 0.15% | 7.4% |
| SELL @ 2nd Alert (retest1) | 51 | 23 | 45.1% | 9 | 28 | 14 | 0.15% | 7.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 111 | 46 | 41.4% | 14 | 65 | 32 | 0.09% | 9.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 10:30:00 | 246.37 | 244.25 | 0.00 | ORB-long ORB[241.16,244.10] vol=2.0x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-05-14 10:35:00 | 245.77 | 244.36 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-15 09:40:00 | 243.02 | 244.04 | 0.00 | ORB-short ORB[243.24,245.60] vol=2.1x ATR=0.70 |
| Stop hit — per-position SL triggered | 2025-05-15 09:55:00 | 243.72 | 243.81 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-19 09:45:00 | 247.37 | 248.40 | 0.00 | ORB-short ORB[247.66,249.45] vol=2.1x ATR=0.73 |
| Stop hit — per-position SL triggered | 2025-05-19 10:05:00 | 248.10 | 248.15 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-20 09:35:00 | 249.07 | 248.27 | 0.00 | ORB-long ORB[246.79,248.88] vol=1.8x ATR=0.57 |
| Stop hit — per-position SL triggered | 2025-05-20 09:45:00 | 248.50 | 248.34 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 09:30:00 | 244.12 | 244.99 | 0.00 | ORB-short ORB[244.74,246.85] vol=2.4x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 09:55:00 | 243.38 | 244.62 | 0.00 | T1 1.5R @ 243.38 |
| Target hit | 2025-05-27 11:15:00 | 243.87 | 243.85 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — SELL (started 2025-06-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 11:10:00 | 236.28 | 236.46 | 0.00 | ORB-short ORB[236.45,238.40] vol=2.8x ATR=0.45 |
| Stop hit — per-position SL triggered | 2025-06-04 12:00:00 | 236.73 | 236.40 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 10:30:00 | 243.10 | 241.94 | 0.00 | ORB-long ORB[240.81,242.35] vol=2.6x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-09 11:00:00 | 243.88 | 242.25 | 0.00 | T1 1.5R @ 243.88 |
| Stop hit — per-position SL triggered | 2025-06-09 11:05:00 | 243.10 | 242.27 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-16 09:55:00 | 254.90 | 253.35 | 0.00 | ORB-long ORB[252.19,254.00] vol=2.2x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 10:15:00 | 256.32 | 254.08 | 0.00 | T1 1.5R @ 256.32 |
| Target hit | 2025-06-16 12:05:00 | 255.01 | 255.07 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2025-06-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 11:05:00 | 252.84 | 251.22 | 0.00 | ORB-long ORB[249.35,252.45] vol=1.8x ATR=0.71 |
| Stop hit — per-position SL triggered | 2025-06-20 12:15:00 | 252.13 | 251.72 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-27 10:35:00 | 243.80 | 244.53 | 0.00 | ORB-short ORB[244.50,246.22] vol=1.5x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 10:55:00 | 243.13 | 244.27 | 0.00 | T1 1.5R @ 243.13 |
| Stop hit — per-position SL triggered | 2025-06-27 11:00:00 | 243.80 | 244.25 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-07-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 11:05:00 | 242.64 | 242.80 | 0.00 | ORB-short ORB[242.71,243.75] vol=1.9x ATR=0.38 |
| Stop hit — per-position SL triggered | 2025-07-02 11:20:00 | 243.02 | 242.81 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 10:15:00 | 245.12 | 243.90 | 0.00 | ORB-long ORB[241.80,244.55] vol=1.8x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 11:10:00 | 246.01 | 244.65 | 0.00 | T1 1.5R @ 246.01 |
| Stop hit — per-position SL triggered | 2025-07-03 11:30:00 | 245.12 | 244.70 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-07 11:15:00 | 243.91 | 244.39 | 0.00 | ORB-short ORB[243.99,245.50] vol=1.7x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 11:40:00 | 243.33 | 244.27 | 0.00 | T1 1.5R @ 243.33 |
| Target hit | 2025-07-07 15:20:00 | 241.24 | 242.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2025-07-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 10:50:00 | 243.09 | 243.45 | 0.00 | ORB-short ORB[243.51,244.69] vol=3.2x ATR=0.41 |
| Stop hit — per-position SL triggered | 2025-07-10 11:20:00 | 243.50 | 243.42 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:45:00 | 241.44 | 241.87 | 0.00 | ORB-short ORB[242.00,243.09] vol=1.5x ATR=0.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 11:05:00 | 240.89 | 241.77 | 0.00 | T1 1.5R @ 240.89 |
| Stop hit — per-position SL triggered | 2025-07-11 11:10:00 | 241.44 | 241.74 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 10:40:00 | 243.81 | 243.49 | 0.00 | ORB-long ORB[242.86,243.74] vol=4.7x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 10:55:00 | 244.30 | 243.56 | 0.00 | T1 1.5R @ 244.30 |
| Stop hit — per-position SL triggered | 2025-07-17 12:55:00 | 243.81 | 243.78 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 11:15:00 | 245.24 | 245.62 | 0.00 | ORB-short ORB[245.30,246.20] vol=2.6x ATR=0.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 11:25:00 | 244.76 | 245.54 | 0.00 | T1 1.5R @ 244.76 |
| Target hit | 2025-07-24 14:05:00 | 244.86 | 244.86 | 0.00 | Trail-exit close>VWAP |

### Cycle 18 — SELL (started 2025-07-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 09:40:00 | 243.49 | 244.07 | 0.00 | ORB-short ORB[244.00,244.79] vol=1.8x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 09:55:00 | 242.85 | 243.72 | 0.00 | T1 1.5R @ 242.85 |
| Target hit | 2025-07-25 11:55:00 | 242.30 | 242.28 | 0.00 | Trail-exit close>VWAP |

### Cycle 19 — SELL (started 2025-07-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 10:20:00 | 240.59 | 241.09 | 0.00 | ORB-short ORB[241.75,242.89] vol=5.3x ATR=0.52 |
| Stop hit — per-position SL triggered | 2025-07-30 10:40:00 | 241.11 | 241.06 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-07-31 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 10:25:00 | 241.03 | 240.60 | 0.00 | ORB-long ORB[239.80,240.90] vol=3.5x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 11:05:00 | 241.80 | 240.81 | 0.00 | T1 1.5R @ 241.80 |
| Target hit | 2025-07-31 12:05:00 | 241.08 | 241.11 | 0.00 | Trail-exit close<VWAP |

### Cycle 21 — SELL (started 2025-08-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-01 10:30:00 | 236.04 | 237.22 | 0.00 | ORB-short ORB[238.27,240.71] vol=1.9x ATR=0.65 |
| Stop hit — per-position SL triggered | 2025-08-01 11:05:00 | 236.69 | 236.72 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-08-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-04 10:55:00 | 234.58 | 235.95 | 0.00 | ORB-short ORB[234.92,238.15] vol=2.2x ATR=0.56 |
| Stop hit — per-position SL triggered | 2025-08-04 11:05:00 | 235.14 | 235.85 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-08-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-11 10:35:00 | 232.52 | 232.88 | 0.00 | ORB-short ORB[232.72,234.39] vol=3.4x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 11:40:00 | 231.80 | 232.75 | 0.00 | T1 1.5R @ 231.80 |
| Stop hit — per-position SL triggered | 2025-08-11 12:00:00 | 232.52 | 232.61 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-08-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 11:10:00 | 239.37 | 238.02 | 0.00 | ORB-long ORB[236.30,238.49] vol=2.5x ATR=0.50 |
| Stop hit — per-position SL triggered | 2025-08-13 11:25:00 | 238.87 | 238.17 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-08-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 10:45:00 | 238.20 | 237.74 | 0.00 | ORB-long ORB[237.00,237.79] vol=2.7x ATR=0.32 |
| Stop hit — per-position SL triggered | 2025-08-20 11:20:00 | 237.88 | 237.76 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-08-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:45:00 | 240.40 | 239.22 | 0.00 | ORB-long ORB[237.23,238.94] vol=1.6x ATR=0.49 |
| Stop hit — per-position SL triggered | 2025-08-21 10:25:00 | 239.91 | 239.86 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 11:15:00 | 236.87 | 236.29 | 0.00 | ORB-long ORB[235.84,236.83] vol=2.3x ATR=0.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-25 11:40:00 | 237.37 | 236.52 | 0.00 | T1 1.5R @ 237.37 |
| Stop hit — per-position SL triggered | 2025-08-25 11:45:00 | 236.87 | 236.52 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-08-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 10:55:00 | 234.92 | 235.10 | 0.00 | ORB-short ORB[235.14,237.30] vol=7.5x ATR=0.32 |
| Stop hit — per-position SL triggered | 2025-08-26 11:10:00 | 235.24 | 235.08 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-08-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-28 10:45:00 | 235.12 | 234.48 | 0.00 | ORB-long ORB[233.80,234.90] vol=2.0x ATR=0.37 |
| Stop hit — per-position SL triggered | 2025-08-28 11:05:00 | 234.75 | 234.80 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-09-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 09:35:00 | 240.00 | 239.08 | 0.00 | ORB-long ORB[237.40,239.14] vol=2.3x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 09:55:00 | 240.76 | 239.75 | 0.00 | T1 1.5R @ 240.76 |
| Target hit | 2025-09-02 13:20:00 | 240.77 | 241.24 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — SELL (started 2025-09-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-04 09:55:00 | 236.25 | 237.10 | 0.00 | ORB-short ORB[236.36,239.35] vol=2.9x ATR=0.56 |
| Stop hit — per-position SL triggered | 2025-09-04 10:15:00 | 236.81 | 236.99 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-09-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:45:00 | 234.76 | 235.31 | 0.00 | ORB-short ORB[235.75,237.16] vol=2.5x ATR=0.38 |
| Stop hit — per-position SL triggered | 2025-09-05 11:10:00 | 235.14 | 235.26 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-09-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-08 09:35:00 | 233.85 | 234.16 | 0.00 | ORB-short ORB[233.90,234.98] vol=2.0x ATR=0.41 |
| Stop hit — per-position SL triggered | 2025-09-08 09:50:00 | 234.26 | 234.09 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-09-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-10 10:35:00 | 231.83 | 232.37 | 0.00 | ORB-short ORB[232.21,233.04] vol=1.8x ATR=0.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 11:35:00 | 231.26 | 232.14 | 0.00 | T1 1.5R @ 231.26 |
| Stop hit — per-position SL triggered | 2025-09-10 12:10:00 | 231.83 | 231.95 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-09-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-15 09:35:00 | 232.65 | 233.26 | 0.00 | ORB-short ORB[233.13,234.19] vol=1.6x ATR=0.37 |
| Stop hit — per-position SL triggered | 2025-09-15 09:50:00 | 233.02 | 233.12 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 09:30:00 | 233.98 | 233.21 | 0.00 | ORB-long ORB[232.42,233.30] vol=1.7x ATR=0.30 |
| Stop hit — per-position SL triggered | 2025-09-16 09:35:00 | 233.68 | 233.28 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-09-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 10:25:00 | 237.79 | 236.47 | 0.00 | ORB-long ORB[235.30,236.30] vol=1.8x ATR=0.37 |
| Stop hit — per-position SL triggered | 2025-09-17 10:30:00 | 237.42 | 236.53 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 10:15:00 | 237.73 | 237.19 | 0.00 | ORB-long ORB[236.62,237.56] vol=3.3x ATR=0.47 |
| Stop hit — per-position SL triggered | 2025-09-24 10:40:00 | 237.26 | 237.24 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-09-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-25 09:30:00 | 241.30 | 240.22 | 0.00 | ORB-long ORB[238.52,241.00] vol=2.1x ATR=0.52 |
| Stop hit — per-position SL triggered | 2025-09-25 09:45:00 | 240.78 | 240.62 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-09-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-30 10:55:00 | 239.53 | 239.74 | 0.00 | ORB-short ORB[239.70,240.70] vol=1.5x ATR=0.34 |
| Stop hit — per-position SL triggered | 2025-09-30 11:15:00 | 239.87 | 239.74 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 10:15:00 | 244.85 | 242.43 | 0.00 | ORB-long ORB[238.47,240.12] vol=3.3x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-01 10:20:00 | 245.90 | 243.19 | 0.00 | T1 1.5R @ 245.90 |
| Stop hit — per-position SL triggered | 2025-10-01 10:25:00 | 244.85 | 243.30 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-10-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 11:00:00 | 243.90 | 245.25 | 0.00 | ORB-short ORB[244.74,246.80] vol=2.1x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 12:05:00 | 243.19 | 244.84 | 0.00 | T1 1.5R @ 243.19 |
| Target hit | 2025-10-08 15:20:00 | 241.77 | 243.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — BUY (started 2025-10-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 10:35:00 | 242.90 | 241.96 | 0.00 | ORB-long ORB[241.45,242.60] vol=2.5x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 11:20:00 | 243.79 | 242.65 | 0.00 | T1 1.5R @ 243.79 |
| Stop hit — per-position SL triggered | 2025-10-09 11:25:00 | 242.90 | 242.66 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-10-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:50:00 | 246.69 | 245.89 | 0.00 | ORB-long ORB[244.29,246.31] vol=3.2x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 11:10:00 | 247.35 | 246.26 | 0.00 | T1 1.5R @ 247.35 |
| Target hit | 2025-10-15 14:15:00 | 247.00 | 247.10 | 0.00 | Trail-exit close<VWAP |

### Cycle 45 — SELL (started 2025-10-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 11:00:00 | 246.24 | 246.37 | 0.00 | ORB-short ORB[246.35,248.55] vol=1.6x ATR=0.41 |
| Stop hit — per-position SL triggered | 2025-10-17 11:20:00 | 246.65 | 246.38 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-10-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 10:00:00 | 246.31 | 246.41 | 0.00 | ORB-short ORB[246.93,249.20] vol=6.0x ATR=0.71 |
| Stop hit — per-position SL triggered | 2025-10-20 10:55:00 | 247.02 | 246.37 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-10-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 10:00:00 | 256.03 | 254.61 | 0.00 | ORB-long ORB[252.85,254.70] vol=1.6x ATR=0.65 |
| Stop hit — per-position SL triggered | 2025-10-24 10:15:00 | 255.38 | 254.86 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-10-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 11:00:00 | 254.60 | 253.83 | 0.00 | ORB-long ORB[250.55,252.00] vol=2.4x ATR=0.54 |
| Stop hit — per-position SL triggered | 2025-10-29 11:10:00 | 254.06 | 253.85 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-10-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 11:05:00 | 253.78 | 254.73 | 0.00 | ORB-short ORB[254.20,255.58] vol=1.7x ATR=0.41 |
| Stop hit — per-position SL triggered | 2025-10-31 11:30:00 | 254.19 | 254.61 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-11-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 11:10:00 | 257.40 | 256.89 | 0.00 | ORB-long ORB[254.30,257.15] vol=1.8x ATR=0.56 |
| Stop hit — per-position SL triggered | 2025-11-03 11:55:00 | 256.84 | 256.93 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-11-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 09:30:00 | 255.10 | 254.19 | 0.00 | ORB-long ORB[251.80,254.65] vol=1.8x ATR=0.63 |
| Stop hit — per-position SL triggered | 2025-11-10 10:10:00 | 254.47 | 254.61 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-11-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 09:30:00 | 249.30 | 247.32 | 0.00 | ORB-long ORB[245.10,247.90] vol=1.5x ATR=0.93 |
| Stop hit — per-position SL triggered | 2025-11-14 09:40:00 | 248.37 | 247.81 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-11-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 11:10:00 | 249.00 | 248.76 | 0.00 | ORB-long ORB[247.45,248.90] vol=1.7x ATR=0.45 |
| Stop hit — per-position SL triggered | 2025-11-17 11:30:00 | 248.55 | 248.79 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-11-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 10:25:00 | 250.00 | 248.87 | 0.00 | ORB-long ORB[247.60,249.55] vol=2.2x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 10:40:00 | 250.62 | 249.21 | 0.00 | T1 1.5R @ 250.62 |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 250.00 | 249.63 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-11-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-21 11:10:00 | 248.10 | 247.40 | 0.00 | ORB-long ORB[246.75,247.90] vol=1.9x ATR=0.43 |
| Stop hit — per-position SL triggered | 2025-11-21 11:40:00 | 247.67 | 247.54 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-11-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-25 10:45:00 | 247.05 | 245.94 | 0.00 | ORB-long ORB[244.60,246.25] vol=2.5x ATR=0.47 |
| Stop hit — per-position SL triggered | 2025-11-25 10:55:00 | 246.58 | 246.23 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-12-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 10:50:00 | 241.77 | 240.76 | 0.00 | ORB-long ORB[239.69,240.90] vol=2.1x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 12:00:00 | 242.36 | 241.20 | 0.00 | T1 1.5R @ 242.36 |
| Stop hit — per-position SL triggered | 2025-12-04 13:55:00 | 241.77 | 241.87 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-12-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 10:10:00 | 242.00 | 242.35 | 0.00 | ORB-short ORB[242.11,243.20] vol=1.8x ATR=0.46 |
| Stop hit — per-position SL triggered | 2025-12-05 10:50:00 | 242.46 | 242.06 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-12-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 10:10:00 | 240.20 | 239.54 | 0.00 | ORB-long ORB[239.17,240.00] vol=1.5x ATR=0.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 10:25:00 | 240.81 | 239.75 | 0.00 | T1 1.5R @ 240.81 |
| Stop hit — per-position SL triggered | 2025-12-11 10:40:00 | 240.20 | 240.20 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-12-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 10:55:00 | 234.96 | 235.74 | 0.00 | ORB-short ORB[235.05,236.57] vol=2.0x ATR=0.31 |
| Stop hit — per-position SL triggered | 2025-12-24 11:10:00 | 235.27 | 235.70 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-12-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-29 10:10:00 | 235.26 | 234.64 | 0.00 | ORB-long ORB[234.00,234.99] vol=1.5x ATR=0.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 10:25:00 | 235.84 | 234.93 | 0.00 | T1 1.5R @ 235.84 |
| Target hit | 2025-12-29 11:30:00 | 237.14 | 237.27 | 0.00 | Trail-exit close<VWAP |

### Cycle 62 — BUY (started 2026-01-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:00:00 | 240.55 | 239.72 | 0.00 | ORB-long ORB[237.94,239.97] vol=1.8x ATR=0.48 |
| Stop hit — per-position SL triggered | 2026-01-02 10:05:00 | 240.07 | 239.84 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-01-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 10:05:00 | 241.13 | 240.21 | 0.00 | ORB-long ORB[239.61,240.78] vol=2.5x ATR=0.52 |
| Stop hit — per-position SL triggered | 2026-01-07 11:45:00 | 240.61 | 240.57 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2026-01-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:00:00 | 234.89 | 236.33 | 0.00 | ORB-short ORB[237.07,238.89] vol=2.2x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:15:00 | 234.26 | 236.14 | 0.00 | T1 1.5R @ 234.26 |
| Target hit | 2026-01-08 15:20:00 | 231.36 | 233.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 65 — SELL (started 2026-01-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-12 09:40:00 | 231.96 | 233.02 | 0.00 | ORB-short ORB[232.70,234.71] vol=2.2x ATR=0.62 |
| Stop hit — per-position SL triggered | 2026-01-12 09:45:00 | 232.58 | 232.95 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2026-01-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 11:05:00 | 241.51 | 243.39 | 0.00 | ORB-short ORB[242.63,245.34] vol=1.7x ATR=0.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 12:10:00 | 240.65 | 242.81 | 0.00 | T1 1.5R @ 240.65 |
| Stop hit — per-position SL triggered | 2026-01-20 12:25:00 | 241.51 | 242.74 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-01-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-29 09:30:00 | 272.91 | 271.01 | 0.00 | ORB-long ORB[269.01,272.39] vol=1.7x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 09:40:00 | 274.89 | 271.91 | 0.00 | T1 1.5R @ 274.89 |
| Stop hit — per-position SL triggered | 2026-01-29 09:45:00 | 272.91 | 272.09 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-02-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 11:00:00 | 269.15 | 266.36 | 0.00 | ORB-long ORB[265.00,268.00] vol=4.3x ATR=0.91 |
| Stop hit — per-position SL triggered | 2026-02-06 11:10:00 | 268.24 | 266.81 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-02-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 10:45:00 | 268.75 | 270.04 | 0.00 | ORB-short ORB[269.50,273.00] vol=1.7x ATR=0.60 |
| Stop hit — per-position SL triggered | 2026-02-17 11:15:00 | 269.35 | 269.70 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2026-02-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 11:10:00 | 278.50 | 277.31 | 0.00 | ORB-long ORB[276.10,278.40] vol=2.8x ATR=0.62 |
| Stop hit — per-position SL triggered | 2026-02-26 11:25:00 | 277.88 | 277.39 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2026-02-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 10:45:00 | 280.10 | 278.33 | 0.00 | ORB-long ORB[277.20,279.55] vol=2.1x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:05:00 | 281.15 | 279.24 | 0.00 | T1 1.5R @ 281.15 |
| Stop hit — per-position SL triggered | 2026-02-27 11:50:00 | 280.10 | 280.14 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2026-03-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:25:00 | 267.00 | 268.47 | 0.00 | ORB-short ORB[269.00,271.30] vol=2.1x ATR=0.75 |
| Stop hit — per-position SL triggered | 2026-03-13 10:30:00 | 267.75 | 268.42 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:15:00 | 262.05 | 260.90 | 0.00 | ORB-long ORB[259.15,261.80] vol=3.0x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 10:20:00 | 263.25 | 261.01 | 0.00 | T1 1.5R @ 263.25 |
| Stop hit — per-position SL triggered | 2026-03-17 10:25:00 | 262.05 | 261.04 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-03-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 11:05:00 | 270.65 | 268.83 | 0.00 | ORB-long ORB[266.40,268.80] vol=1.5x ATR=0.70 |
| Stop hit — per-position SL triggered | 2026-03-25 11:25:00 | 269.95 | 268.98 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-04-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 10:35:00 | 284.85 | 286.30 | 0.00 | ORB-short ORB[285.35,288.10] vol=2.0x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 11:10:00 | 283.84 | 285.48 | 0.00 | T1 1.5R @ 283.84 |
| Target hit | 2026-04-16 15:20:00 | 282.95 | 284.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 76 — SELL (started 2026-05-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:10:00 | 288.60 | 290.74 | 0.00 | ORB-short ORB[291.70,295.90] vol=2.4x ATR=0.81 |
| Stop hit — per-position SL triggered | 2026-05-05 11:25:00 | 289.41 | 290.65 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2026-05-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 10:30:00 | 287.15 | 288.31 | 0.00 | ORB-short ORB[287.80,290.25] vol=1.6x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 11:05:00 | 285.93 | 287.62 | 0.00 | T1 1.5R @ 285.93 |
| Target hit | 2026-05-06 15:20:00 | 280.30 | 283.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 78 — BUY (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:15:00 | 283.65 | 281.95 | 0.00 | ORB-long ORB[279.70,282.75] vol=1.6x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 10:35:00 | 285.06 | 282.42 | 0.00 | T1 1.5R @ 285.06 |
| Stop hit — per-position SL triggered | 2026-05-07 12:40:00 | 283.65 | 283.56 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-05-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:50:00 | 281.60 | 281.84 | 0.00 | ORB-short ORB[281.80,284.00] vol=1.8x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 11:25:00 | 280.62 | 281.74 | 0.00 | T1 1.5R @ 280.62 |
| Target hit | 2026-05-08 15:20:00 | 279.15 | 280.35 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 10:30:00 | 246.37 | 2025-05-14 10:35:00 | 245.77 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-05-15 09:40:00 | 243.02 | 2025-05-15 09:55:00 | 243.72 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-05-19 09:45:00 | 247.37 | 2025-05-19 10:05:00 | 248.10 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-05-20 09:35:00 | 249.07 | 2025-05-20 09:45:00 | 248.50 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-05-27 09:30:00 | 244.12 | 2025-05-27 09:55:00 | 243.38 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-05-27 09:30:00 | 244.12 | 2025-05-27 11:15:00 | 243.87 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2025-06-04 11:10:00 | 236.28 | 2025-06-04 12:00:00 | 236.73 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-06-09 10:30:00 | 243.10 | 2025-06-09 11:00:00 | 243.88 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-06-09 10:30:00 | 243.10 | 2025-06-09 11:05:00 | 243.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-16 09:55:00 | 254.90 | 2025-06-16 10:15:00 | 256.32 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-06-16 09:55:00 | 254.90 | 2025-06-16 12:05:00 | 255.01 | TARGET_HIT | 0.50 | 0.04% |
| BUY | retest1 | 2025-06-20 11:05:00 | 252.84 | 2025-06-20 12:15:00 | 252.13 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-06-27 10:35:00 | 243.80 | 2025-06-27 10:55:00 | 243.13 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-06-27 10:35:00 | 243.80 | 2025-06-27 11:00:00 | 243.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-02 11:05:00 | 242.64 | 2025-07-02 11:20:00 | 243.02 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-07-03 10:15:00 | 245.12 | 2025-07-03 11:10:00 | 246.01 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-07-03 10:15:00 | 245.12 | 2025-07-03 11:30:00 | 245.12 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-07 11:15:00 | 243.91 | 2025-07-07 11:40:00 | 243.33 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-07-07 11:15:00 | 243.91 | 2025-07-07 15:20:00 | 241.24 | TARGET_HIT | 0.50 | 1.09% |
| SELL | retest1 | 2025-07-10 10:50:00 | 243.09 | 2025-07-10 11:20:00 | 243.50 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-07-11 10:45:00 | 241.44 | 2025-07-11 11:05:00 | 240.89 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2025-07-11 10:45:00 | 241.44 | 2025-07-11 11:10:00 | 241.44 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-17 10:40:00 | 243.81 | 2025-07-17 10:55:00 | 244.30 | PARTIAL | 0.50 | 0.20% |
| BUY | retest1 | 2025-07-17 10:40:00 | 243.81 | 2025-07-17 12:55:00 | 243.81 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-24 11:15:00 | 245.24 | 2025-07-24 11:25:00 | 244.76 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2025-07-24 11:15:00 | 245.24 | 2025-07-24 14:05:00 | 244.86 | TARGET_HIT | 0.50 | 0.15% |
| SELL | retest1 | 2025-07-25 09:40:00 | 243.49 | 2025-07-25 09:55:00 | 242.85 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2025-07-25 09:40:00 | 243.49 | 2025-07-25 11:55:00 | 242.30 | TARGET_HIT | 0.50 | 0.49% |
| SELL | retest1 | 2025-07-30 10:20:00 | 240.59 | 2025-07-30 10:40:00 | 241.11 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-07-31 10:25:00 | 241.03 | 2025-07-31 11:05:00 | 241.80 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-07-31 10:25:00 | 241.03 | 2025-07-31 12:05:00 | 241.08 | TARGET_HIT | 0.50 | 0.02% |
| SELL | retest1 | 2025-08-01 10:30:00 | 236.04 | 2025-08-01 11:05:00 | 236.69 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-08-04 10:55:00 | 234.58 | 2025-08-04 11:05:00 | 235.14 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-08-11 10:35:00 | 232.52 | 2025-08-11 11:40:00 | 231.80 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-08-11 10:35:00 | 232.52 | 2025-08-11 12:00:00 | 232.52 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-13 11:10:00 | 239.37 | 2025-08-13 11:25:00 | 238.87 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-08-20 10:45:00 | 238.20 | 2025-08-20 11:20:00 | 237.88 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2025-08-21 09:45:00 | 240.40 | 2025-08-21 10:25:00 | 239.91 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-08-25 11:15:00 | 236.87 | 2025-08-25 11:40:00 | 237.37 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2025-08-25 11:15:00 | 236.87 | 2025-08-25 11:45:00 | 236.87 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-26 10:55:00 | 234.92 | 2025-08-26 11:10:00 | 235.24 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2025-08-28 10:45:00 | 235.12 | 2025-08-28 11:05:00 | 234.75 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-09-02 09:35:00 | 240.00 | 2025-09-02 09:55:00 | 240.76 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-09-02 09:35:00 | 240.00 | 2025-09-02 13:20:00 | 240.77 | TARGET_HIT | 0.50 | 0.32% |
| SELL | retest1 | 2025-09-04 09:55:00 | 236.25 | 2025-09-04 10:15:00 | 236.81 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-09-05 10:45:00 | 234.76 | 2025-09-05 11:10:00 | 235.14 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-09-08 09:35:00 | 233.85 | 2025-09-08 09:50:00 | 234.26 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-09-10 10:35:00 | 231.83 | 2025-09-10 11:35:00 | 231.26 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2025-09-10 10:35:00 | 231.83 | 2025-09-10 12:10:00 | 231.83 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-15 09:35:00 | 232.65 | 2025-09-15 09:50:00 | 233.02 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-09-16 09:30:00 | 233.98 | 2025-09-16 09:35:00 | 233.68 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2025-09-17 10:25:00 | 237.79 | 2025-09-17 10:30:00 | 237.42 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-09-24 10:15:00 | 237.73 | 2025-09-24 10:40:00 | 237.26 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-09-25 09:30:00 | 241.30 | 2025-09-25 09:45:00 | 240.78 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-09-30 10:55:00 | 239.53 | 2025-09-30 11:15:00 | 239.87 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-10-01 10:15:00 | 244.85 | 2025-10-01 10:20:00 | 245.90 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-10-01 10:15:00 | 244.85 | 2025-10-01 10:25:00 | 244.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-08 11:00:00 | 243.90 | 2025-10-08 12:05:00 | 243.19 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-10-08 11:00:00 | 243.90 | 2025-10-08 15:20:00 | 241.77 | TARGET_HIT | 0.50 | 0.87% |
| BUY | retest1 | 2025-10-09 10:35:00 | 242.90 | 2025-10-09 11:20:00 | 243.79 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-10-09 10:35:00 | 242.90 | 2025-10-09 11:25:00 | 242.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-15 10:50:00 | 246.69 | 2025-10-15 11:10:00 | 247.35 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2025-10-15 10:50:00 | 246.69 | 2025-10-15 14:15:00 | 247.00 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2025-10-17 11:00:00 | 246.24 | 2025-10-17 11:20:00 | 246.65 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-10-20 10:00:00 | 246.31 | 2025-10-20 10:55:00 | 247.02 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-10-24 10:00:00 | 256.03 | 2025-10-24 10:15:00 | 255.38 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-10-29 11:00:00 | 254.60 | 2025-10-29 11:10:00 | 254.06 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-10-31 11:05:00 | 253.78 | 2025-10-31 11:30:00 | 254.19 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-11-03 11:10:00 | 257.40 | 2025-11-03 11:55:00 | 256.84 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-11-10 09:30:00 | 255.10 | 2025-11-10 10:10:00 | 254.47 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-11-14 09:30:00 | 249.30 | 2025-11-14 09:40:00 | 248.37 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-11-17 11:10:00 | 249.00 | 2025-11-17 11:30:00 | 248.55 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-11-20 10:25:00 | 250.00 | 2025-11-20 10:40:00 | 250.62 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2025-11-20 10:25:00 | 250.00 | 2025-11-20 11:15:00 | 250.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-21 11:10:00 | 248.10 | 2025-11-21 11:40:00 | 247.67 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-11-25 10:45:00 | 247.05 | 2025-11-25 10:55:00 | 246.58 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-12-04 10:50:00 | 241.77 | 2025-12-04 12:00:00 | 242.36 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2025-12-04 10:50:00 | 241.77 | 2025-12-04 13:55:00 | 241.77 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-05 10:10:00 | 242.00 | 2025-12-05 10:50:00 | 242.46 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-12-11 10:10:00 | 240.20 | 2025-12-11 10:25:00 | 240.81 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2025-12-11 10:10:00 | 240.20 | 2025-12-11 10:40:00 | 240.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-24 10:55:00 | 234.96 | 2025-12-24 11:10:00 | 235.27 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2025-12-29 10:10:00 | 235.26 | 2025-12-29 10:25:00 | 235.84 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2025-12-29 10:10:00 | 235.26 | 2025-12-29 11:30:00 | 237.14 | TARGET_HIT | 0.50 | 0.80% |
| BUY | retest1 | 2026-01-02 10:00:00 | 240.55 | 2026-01-02 10:05:00 | 240.07 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-01-07 10:05:00 | 241.13 | 2026-01-07 11:45:00 | 240.61 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-01-08 11:00:00 | 234.89 | 2026-01-08 11:15:00 | 234.26 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2026-01-08 11:00:00 | 234.89 | 2026-01-08 15:20:00 | 231.36 | TARGET_HIT | 0.50 | 1.50% |
| SELL | retest1 | 2026-01-12 09:40:00 | 231.96 | 2026-01-12 09:45:00 | 232.58 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-01-20 11:05:00 | 241.51 | 2026-01-20 12:10:00 | 240.65 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-01-20 11:05:00 | 241.51 | 2026-01-20 12:25:00 | 241.51 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-29 09:30:00 | 272.91 | 2026-01-29 09:40:00 | 274.89 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2026-01-29 09:30:00 | 272.91 | 2026-01-29 09:45:00 | 272.91 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-06 11:00:00 | 269.15 | 2026-02-06 11:10:00 | 268.24 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-17 10:45:00 | 268.75 | 2026-02-17 11:15:00 | 269.35 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-26 11:10:00 | 278.50 | 2026-02-26 11:25:00 | 277.88 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-27 10:45:00 | 280.10 | 2026-02-27 11:05:00 | 281.15 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-02-27 10:45:00 | 280.10 | 2026-02-27 11:50:00 | 280.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-13 10:25:00 | 267.00 | 2026-03-13 10:30:00 | 267.75 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-03-17 10:15:00 | 262.05 | 2026-03-17 10:20:00 | 263.25 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-03-17 10:15:00 | 262.05 | 2026-03-17 10:25:00 | 262.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-25 11:05:00 | 270.65 | 2026-03-25 11:25:00 | 269.95 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-04-16 10:35:00 | 284.85 | 2026-04-16 11:10:00 | 283.84 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-04-16 10:35:00 | 284.85 | 2026-04-16 15:20:00 | 282.95 | TARGET_HIT | 0.50 | 0.67% |
| SELL | retest1 | 2026-05-05 11:10:00 | 288.60 | 2026-05-05 11:25:00 | 289.41 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-05-06 10:30:00 | 287.15 | 2026-05-06 11:05:00 | 285.93 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-05-06 10:30:00 | 287.15 | 2026-05-06 15:20:00 | 280.30 | TARGET_HIT | 0.50 | 2.39% |
| BUY | retest1 | 2026-05-07 10:15:00 | 283.65 | 2026-05-07 10:35:00 | 285.06 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2026-05-07 10:15:00 | 283.65 | 2026-05-07 12:40:00 | 283.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-08 10:50:00 | 281.60 | 2026-05-08 11:25:00 | 280.62 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-05-08 10:50:00 | 281.60 | 2026-05-08 15:20:00 | 279.15 | TARGET_HIT | 0.50 | 0.87% |
