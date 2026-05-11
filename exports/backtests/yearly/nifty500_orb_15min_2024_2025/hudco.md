# Housing & Urban Development Corporation Ltd. (HUDCO)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-05-05 15:25:00 (18108 bars)
- **Last close:** 229.30
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
| ENTRY1 | 44 |
| ENTRY2 | 0 |
| PARTIAL | 18 |
| TARGET_HIT | 7 |
| STOP_HIT | 37 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 62 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 37
- **Target hits / Stop hits / Partials:** 7 / 37 / 18
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 7.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 38 | 11 | 28.9% | 2 | 27 | 9 | -0.04% | -1.7% |
| BUY @ 2nd Alert (retest1) | 38 | 11 | 28.9% | 2 | 27 | 9 | -0.04% | -1.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 24 | 14 | 58.3% | 5 | 10 | 9 | 0.37% | 8.9% |
| SELL @ 2nd Alert (retest1) | 24 | 14 | 58.3% | 5 | 10 | 9 | 0.37% | 8.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 62 | 25 | 40.3% | 7 | 37 | 18 | 0.12% | 7.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-18 09:30:00 | 246.55 | 244.31 | 0.00 | ORB-long ORB[242.35,243.80] vol=2.6x ATR=1.25 |
| Stop hit — per-position SL triggered | 2024-05-18 09:55:00 | 245.30 | 245.81 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-06-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 09:55:00 | 285.50 | 281.15 | 0.00 | ORB-long ORB[278.35,281.90] vol=5.4x ATR=1.61 |
| Stop hit — per-position SL triggered | 2024-06-13 10:00:00 | 283.89 | 281.53 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-06-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 10:30:00 | 280.75 | 278.72 | 0.00 | ORB-long ORB[276.75,279.50] vol=3.9x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 10:45:00 | 282.41 | 279.58 | 0.00 | T1 1.5R @ 282.41 |
| Stop hit — per-position SL triggered | 2024-06-25 11:00:00 | 280.75 | 279.70 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-07-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 10:30:00 | 283.70 | 282.27 | 0.00 | ORB-long ORB[279.50,282.55] vol=2.1x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 10:50:00 | 285.51 | 282.68 | 0.00 | T1 1.5R @ 285.51 |
| Stop hit — per-position SL triggered | 2024-07-01 11:30:00 | 283.70 | 283.03 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-07-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 10:45:00 | 295.30 | 289.76 | 0.00 | ORB-long ORB[280.00,284.40] vol=3.2x ATR=2.14 |
| Stop hit — per-position SL triggered | 2024-07-03 11:10:00 | 293.16 | 291.28 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-08-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 10:55:00 | 293.95 | 291.66 | 0.00 | ORB-long ORB[290.00,293.60] vol=2.2x ATR=1.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 11:05:00 | 296.38 | 292.63 | 0.00 | T1 1.5R @ 296.38 |
| Stop hit — per-position SL triggered | 2024-08-09 11:30:00 | 293.95 | 293.73 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-08-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 09:45:00 | 296.60 | 294.32 | 0.00 | ORB-long ORB[291.20,294.50] vol=2.9x ATR=1.65 |
| Stop hit — per-position SL triggered | 2024-08-19 09:50:00 | 294.95 | 294.43 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-08-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 09:40:00 | 290.80 | 288.58 | 0.00 | ORB-long ORB[286.25,288.90] vol=4.8x ATR=1.15 |
| Stop hit — per-position SL triggered | 2024-08-22 09:45:00 | 289.65 | 288.72 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-08-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 09:50:00 | 282.65 | 284.10 | 0.00 | ORB-short ORB[283.70,286.40] vol=1.8x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 10:25:00 | 281.68 | 283.52 | 0.00 | T1 1.5R @ 281.68 |
| Target hit | 2024-08-26 15:20:00 | 278.70 | 280.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2024-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 10:15:00 | 262.90 | 266.78 | 0.00 | ORB-short ORB[267.05,269.45] vol=2.2x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 10:30:00 | 261.37 | 265.68 | 0.00 | T1 1.5R @ 261.37 |
| Stop hit — per-position SL triggered | 2024-09-05 10:40:00 | 262.90 | 265.50 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-09-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:30:00 | 261.50 | 262.79 | 0.00 | ORB-short ORB[261.60,264.45] vol=1.6x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 09:35:00 | 260.16 | 262.22 | 0.00 | T1 1.5R @ 260.16 |
| Target hit | 2024-09-06 15:20:00 | 254.20 | 257.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2024-09-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 10:40:00 | 251.00 | 248.39 | 0.00 | ORB-long ORB[246.60,249.65] vol=2.4x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 10:55:00 | 252.95 | 249.39 | 0.00 | T1 1.5R @ 252.95 |
| Stop hit — per-position SL triggered | 2024-09-12 11:15:00 | 251.00 | 250.14 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-09-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 10:05:00 | 243.95 | 245.90 | 0.00 | ORB-short ORB[245.25,248.45] vol=2.0x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 11:10:00 | 242.55 | 245.19 | 0.00 | T1 1.5R @ 242.55 |
| Stop hit — per-position SL triggered | 2024-09-17 11:40:00 | 243.95 | 245.06 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-09-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 09:45:00 | 247.75 | 246.42 | 0.00 | ORB-long ORB[244.25,247.50] vol=1.6x ATR=1.19 |
| Stop hit — per-position SL triggered | 2024-09-18 10:00:00 | 246.56 | 246.49 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-09-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:30:00 | 239.55 | 241.05 | 0.00 | ORB-short ORB[240.50,243.50] vol=2.4x ATR=0.78 |
| Stop hit — per-position SL triggered | 2024-09-19 09:35:00 | 240.33 | 240.64 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-09-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 09:35:00 | 234.45 | 237.07 | 0.00 | ORB-short ORB[236.25,239.10] vol=2.3x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 09:55:00 | 233.00 | 235.54 | 0.00 | T1 1.5R @ 233.00 |
| Stop hit — per-position SL triggered | 2024-09-26 10:20:00 | 234.45 | 235.23 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-10-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 09:35:00 | 227.95 | 227.22 | 0.00 | ORB-long ORB[226.51,227.75] vol=1.5x ATR=0.96 |
| Stop hit — per-position SL triggered | 2024-10-10 09:45:00 | 226.99 | 227.25 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-10-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 09:40:00 | 226.92 | 224.76 | 0.00 | ORB-long ORB[223.05,224.73] vol=3.0x ATR=0.97 |
| Stop hit — per-position SL triggered | 2024-10-15 09:45:00 | 225.95 | 225.12 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-10-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 09:30:00 | 228.11 | 226.32 | 0.00 | ORB-long ORB[223.10,226.40] vol=3.2x ATR=1.05 |
| Stop hit — per-position SL triggered | 2024-10-16 09:35:00 | 227.06 | 226.70 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 11:15:00 | 219.83 | 222.68 | 0.00 | ORB-short ORB[222.76,225.70] vol=2.0x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 13:25:00 | 218.83 | 221.70 | 0.00 | T1 1.5R @ 218.83 |
| Target hit | 2024-10-17 15:20:00 | 216.88 | 220.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — SELL (started 2024-10-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:45:00 | 212.17 | 214.45 | 0.00 | ORB-short ORB[214.54,217.55] vol=1.6x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 09:50:00 | 210.96 | 213.82 | 0.00 | T1 1.5R @ 210.96 |
| Stop hit — per-position SL triggered | 2024-10-21 10:00:00 | 212.17 | 213.51 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-11-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 09:30:00 | 218.90 | 217.29 | 0.00 | ORB-long ORB[215.42,218.22] vol=1.7x ATR=1.04 |
| Stop hit — per-position SL triggered | 2024-11-26 09:40:00 | 217.86 | 217.48 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-11-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 09:45:00 | 215.98 | 213.07 | 0.00 | ORB-long ORB[210.85,213.79] vol=3.6x ATR=1.07 |
| Stop hit — per-position SL triggered | 2024-11-27 10:20:00 | 214.91 | 214.16 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-12-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-03 10:40:00 | 236.51 | 238.80 | 0.00 | ORB-short ORB[238.13,240.90] vol=1.7x ATR=0.72 |
| Stop hit — per-position SL triggered | 2024-12-03 11:00:00 | 237.23 | 238.27 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-12-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:35:00 | 239.38 | 237.39 | 0.00 | ORB-long ORB[235.50,238.32] vol=1.6x ATR=0.74 |
| Stop hit — per-position SL triggered | 2024-12-04 09:40:00 | 238.64 | 237.60 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-12-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 10:30:00 | 246.63 | 243.72 | 0.00 | ORB-long ORB[241.26,244.50] vol=1.8x ATR=1.48 |
| Stop hit — per-position SL triggered | 2024-12-06 10:35:00 | 245.15 | 243.88 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-12-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 09:45:00 | 248.00 | 246.39 | 0.00 | ORB-long ORB[244.40,247.90] vol=1.6x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 10:20:00 | 249.73 | 247.34 | 0.00 | T1 1.5R @ 249.73 |
| Stop hit — per-position SL triggered | 2024-12-10 12:05:00 | 248.00 | 247.98 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-12-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:30:00 | 248.75 | 251.24 | 0.00 | ORB-short ORB[250.45,253.75] vol=2.4x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 09:40:00 | 247.34 | 249.25 | 0.00 | T1 1.5R @ 247.34 |
| Target hit | 2024-12-12 11:15:00 | 247.31 | 247.31 | 0.00 | Trail-exit close>VWAP |

### Cycle 29 — BUY (started 2024-12-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 09:35:00 | 249.15 | 246.29 | 0.00 | ORB-long ORB[244.39,247.00] vol=2.3x ATR=1.05 |
| Stop hit — per-position SL triggered | 2024-12-20 09:45:00 | 248.10 | 246.95 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-12-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-24 09:30:00 | 232.91 | 235.57 | 0.00 | ORB-short ORB[235.20,237.90] vol=2.4x ATR=1.15 |
| Stop hit — per-position SL triggered | 2024-12-24 09:35:00 | 234.06 | 235.25 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-12-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:40:00 | 230.38 | 232.73 | 0.00 | ORB-short ORB[232.23,234.68] vol=1.8x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 10:15:00 | 229.05 | 231.11 | 0.00 | T1 1.5R @ 229.05 |
| Target hit | 2024-12-26 15:15:00 | 228.72 | 228.52 | 0.00 | Trail-exit close>VWAP |

### Cycle 32 — BUY (started 2025-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:30:00 | 241.66 | 240.24 | 0.00 | ORB-long ORB[238.00,241.30] vol=3.3x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 09:35:00 | 242.88 | 240.81 | 0.00 | T1 1.5R @ 242.88 |
| Stop hit — per-position SL triggered | 2025-01-02 10:20:00 | 241.66 | 241.69 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-01-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 10:50:00 | 238.50 | 240.89 | 0.00 | ORB-short ORB[239.05,242.30] vol=3.9x ATR=1.64 |
| Stop hit — per-position SL triggered | 2025-01-07 11:15:00 | 240.14 | 240.64 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-01-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-21 09:40:00 | 237.42 | 235.89 | 0.00 | ORB-long ORB[234.10,236.20] vol=1.6x ATR=0.91 |
| Stop hit — per-position SL triggered | 2025-01-21 09:45:00 | 236.51 | 235.80 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-01-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-24 10:25:00 | 224.87 | 222.24 | 0.00 | ORB-long ORB[221.60,223.70] vol=2.7x ATR=1.43 |
| Stop hit — per-position SL triggered | 2025-01-24 10:30:00 | 223.44 | 222.36 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-02-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 10:50:00 | 240.69 | 234.37 | 0.00 | ORB-long ORB[231.00,234.50] vol=4.5x ATR=1.23 |
| Stop hit — per-position SL triggered | 2025-02-01 10:55:00 | 239.46 | 234.95 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-02-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 09:30:00 | 207.63 | 209.07 | 0.00 | ORB-short ORB[208.41,210.49] vol=2.0x ATR=0.80 |
| Stop hit — per-position SL triggered | 2025-02-06 09:40:00 | 208.43 | 208.91 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-02-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 10:10:00 | 207.40 | 205.31 | 0.00 | ORB-long ORB[203.60,205.88] vol=4.2x ATR=1.26 |
| Stop hit — per-position SL triggered | 2025-02-07 10:15:00 | 206.14 | 205.36 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-02-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-27 09:55:00 | 175.40 | 177.51 | 0.00 | ORB-short ORB[178.01,179.99] vol=1.8x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-02-27 10:40:00 | 176.26 | 176.87 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-03-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 10:00:00 | 191.32 | 189.88 | 0.00 | ORB-long ORB[188.00,190.65] vol=1.6x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 10:40:00 | 192.44 | 190.48 | 0.00 | T1 1.5R @ 192.44 |
| Stop hit — per-position SL triggered | 2025-03-19 11:20:00 | 191.32 | 190.90 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-04-15 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 10:10:00 | 220.83 | 218.65 | 0.00 | ORB-long ORB[217.50,219.64] vol=1.9x ATR=1.08 |
| Stop hit — per-position SL triggered | 2025-04-15 10:40:00 | 219.75 | 219.60 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 09:30:00 | 220.66 | 219.43 | 0.00 | ORB-long ORB[218.12,220.00] vol=2.4x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 09:35:00 | 221.68 | 220.27 | 0.00 | T1 1.5R @ 221.68 |
| Target hit | 2025-04-16 13:20:00 | 222.79 | 223.34 | 0.00 | Trail-exit close<VWAP |

### Cycle 43 — BUY (started 2025-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:30:00 | 233.50 | 231.26 | 0.00 | ORB-long ORB[229.32,231.70] vol=2.8x ATR=0.95 |
| Stop hit — per-position SL triggered | 2025-04-21 09:45:00 | 232.55 | 231.91 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-05-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:45:00 | 225.68 | 223.81 | 0.00 | ORB-long ORB[222.50,224.34] vol=2.3x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 12:15:00 | 227.05 | 225.32 | 0.00 | T1 1.5R @ 227.05 |
| Target hit | 2025-05-05 15:20:00 | 229.20 | 226.81 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-18 09:30:00 | 246.55 | 2024-05-18 09:55:00 | 245.30 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-06-13 09:55:00 | 285.50 | 2024-06-13 10:00:00 | 283.89 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2024-06-25 10:30:00 | 280.75 | 2024-06-25 10:45:00 | 282.41 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-06-25 10:30:00 | 280.75 | 2024-06-25 11:00:00 | 280.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-01 10:30:00 | 283.70 | 2024-07-01 10:50:00 | 285.51 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-07-01 10:30:00 | 283.70 | 2024-07-01 11:30:00 | 283.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-03 10:45:00 | 295.30 | 2024-07-03 11:10:00 | 293.16 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest1 | 2024-08-09 10:55:00 | 293.95 | 2024-08-09 11:05:00 | 296.38 | PARTIAL | 0.50 | 0.83% |
| BUY | retest1 | 2024-08-09 10:55:00 | 293.95 | 2024-08-09 11:30:00 | 293.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-19 09:45:00 | 296.60 | 2024-08-19 09:50:00 | 294.95 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2024-08-22 09:40:00 | 290.80 | 2024-08-22 09:45:00 | 289.65 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-08-26 09:50:00 | 282.65 | 2024-08-26 10:25:00 | 281.68 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-08-26 09:50:00 | 282.65 | 2024-08-26 15:20:00 | 278.70 | TARGET_HIT | 0.50 | 1.40% |
| SELL | retest1 | 2024-09-05 10:15:00 | 262.90 | 2024-09-05 10:30:00 | 261.37 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-09-05 10:15:00 | 262.90 | 2024-09-05 10:40:00 | 262.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-06 09:30:00 | 261.50 | 2024-09-06 09:35:00 | 260.16 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-09-06 09:30:00 | 261.50 | 2024-09-06 15:20:00 | 254.20 | TARGET_HIT | 0.50 | 2.79% |
| BUY | retest1 | 2024-09-12 10:40:00 | 251.00 | 2024-09-12 10:55:00 | 252.95 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2024-09-12 10:40:00 | 251.00 | 2024-09-12 11:15:00 | 251.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-17 10:05:00 | 243.95 | 2024-09-17 11:10:00 | 242.55 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-09-17 10:05:00 | 243.95 | 2024-09-17 11:40:00 | 243.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-18 09:45:00 | 247.75 | 2024-09-18 10:00:00 | 246.56 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-09-19 09:30:00 | 239.55 | 2024-09-19 09:35:00 | 240.33 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-09-26 09:35:00 | 234.45 | 2024-09-26 09:55:00 | 233.00 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-09-26 09:35:00 | 234.45 | 2024-09-26 10:20:00 | 234.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-10 09:35:00 | 227.95 | 2024-10-10 09:45:00 | 226.99 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-10-15 09:40:00 | 226.92 | 2024-10-15 09:45:00 | 225.95 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-10-16 09:30:00 | 228.11 | 2024-10-16 09:35:00 | 227.06 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-10-17 11:15:00 | 219.83 | 2024-10-17 13:25:00 | 218.83 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-10-17 11:15:00 | 219.83 | 2024-10-17 15:20:00 | 216.88 | TARGET_HIT | 0.50 | 1.34% |
| SELL | retest1 | 2024-10-21 09:45:00 | 212.17 | 2024-10-21 09:50:00 | 210.96 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-10-21 09:45:00 | 212.17 | 2024-10-21 10:00:00 | 212.17 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-26 09:30:00 | 218.90 | 2024-11-26 09:40:00 | 217.86 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-11-27 09:45:00 | 215.98 | 2024-11-27 10:20:00 | 214.91 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-12-03 10:40:00 | 236.51 | 2024-12-03 11:00:00 | 237.23 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-12-04 09:35:00 | 239.38 | 2024-12-04 09:40:00 | 238.64 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-12-06 10:30:00 | 246.63 | 2024-12-06 10:35:00 | 245.15 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest1 | 2024-12-10 09:45:00 | 248.00 | 2024-12-10 10:20:00 | 249.73 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2024-12-10 09:45:00 | 248.00 | 2024-12-10 12:05:00 | 248.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-12 09:30:00 | 248.75 | 2024-12-12 09:40:00 | 247.34 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-12-12 09:30:00 | 248.75 | 2024-12-12 11:15:00 | 247.31 | TARGET_HIT | 0.50 | 0.58% |
| BUY | retest1 | 2024-12-20 09:35:00 | 249.15 | 2024-12-20 09:45:00 | 248.10 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-12-24 09:30:00 | 232.91 | 2024-12-24 09:35:00 | 234.06 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-12-26 09:40:00 | 230.38 | 2024-12-26 10:15:00 | 229.05 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-12-26 09:40:00 | 230.38 | 2024-12-26 15:15:00 | 228.72 | TARGET_HIT | 0.50 | 0.72% |
| BUY | retest1 | 2025-01-02 09:30:00 | 241.66 | 2025-01-02 09:35:00 | 242.88 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-01-02 09:30:00 | 241.66 | 2025-01-02 10:20:00 | 241.66 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-07 10:50:00 | 238.50 | 2025-01-07 11:15:00 | 240.14 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest1 | 2025-01-21 09:40:00 | 237.42 | 2025-01-21 09:45:00 | 236.51 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-01-24 10:25:00 | 224.87 | 2025-01-24 10:30:00 | 223.44 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest1 | 2025-02-01 10:50:00 | 240.69 | 2025-02-01 10:55:00 | 239.46 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2025-02-06 09:30:00 | 207.63 | 2025-02-06 09:40:00 | 208.43 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-02-07 10:10:00 | 207.40 | 2025-02-07 10:15:00 | 206.14 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest1 | 2025-02-27 09:55:00 | 175.40 | 2025-02-27 10:40:00 | 176.26 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-03-19 10:00:00 | 191.32 | 2025-03-19 10:40:00 | 192.44 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-03-19 10:00:00 | 191.32 | 2025-03-19 11:20:00 | 191.32 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-15 10:10:00 | 220.83 | 2025-04-15 10:40:00 | 219.75 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-04-16 09:30:00 | 220.66 | 2025-04-16 09:35:00 | 221.68 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-04-16 09:30:00 | 220.66 | 2025-04-16 13:20:00 | 222.79 | TARGET_HIT | 0.50 | 0.97% |
| BUY | retest1 | 2025-04-21 09:30:00 | 233.50 | 2025-04-21 09:45:00 | 232.55 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-05-05 09:45:00 | 225.68 | 2025-05-05 12:15:00 | 227.05 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-05-05 09:45:00 | 225.68 | 2025-05-05 15:20:00 | 229.20 | TARGET_HIT | 0.50 | 1.56% |
