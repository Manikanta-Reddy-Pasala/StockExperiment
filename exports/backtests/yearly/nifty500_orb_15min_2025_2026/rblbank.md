# RBL Bank Ltd. (RBLBANK)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 343.65
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
| ENTRY1 | 104 |
| ENTRY2 | 0 |
| PARTIAL | 41 |
| TARGET_HIT | 21 |
| STOP_HIT | 83 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 145 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 62 / 83
- **Target hits / Stop hits / Partials:** 21 / 83 / 41
- **Avg / median % per leg:** 0.16% / 0.00%
- **Sum % (uncompounded):** 22.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 80 | 41 | 51.2% | 16 | 39 | 25 | 0.22% | 17.8% |
| BUY @ 2nd Alert (retest1) | 80 | 41 | 51.2% | 16 | 39 | 25 | 0.22% | 17.8% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 65 | 21 | 32.3% | 5 | 44 | 16 | 0.07% | 4.7% |
| SELL @ 2nd Alert (retest1) | 65 | 21 | 32.3% | 5 | 44 | 16 | 0.07% | 4.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 145 | 62 | 42.8% | 21 | 83 | 41 | 0.16% | 22.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-19 09:35:00 | 212.64 | 213.82 | 0.00 | ORB-short ORB[212.94,214.70] vol=1.5x ATR=0.67 |
| Stop hit — per-position SL triggered | 2025-05-19 09:50:00 | 213.31 | 213.69 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-20 09:30:00 | 210.63 | 211.42 | 0.00 | ORB-short ORB[210.65,213.70] vol=1.8x ATR=0.79 |
| Stop hit — per-position SL triggered | 2025-05-20 09:35:00 | 211.42 | 211.53 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 10:55:00 | 209.52 | 208.81 | 0.00 | ORB-long ORB[206.41,209.35] vol=1.9x ATR=0.62 |
| Stop hit — per-position SL triggered | 2025-05-21 12:00:00 | 208.90 | 209.01 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 10:50:00 | 209.58 | 208.56 | 0.00 | ORB-long ORB[207.99,208.88] vol=3.0x ATR=0.45 |
| Stop hit — per-position SL triggered | 2025-05-23 11:15:00 | 209.13 | 208.78 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-26 10:05:00 | 208.95 | 209.74 | 0.00 | ORB-short ORB[209.50,211.25] vol=5.3x ATR=0.51 |
| Stop hit — per-position SL triggered | 2025-05-26 10:10:00 | 209.46 | 209.69 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-05-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-27 10:10:00 | 207.75 | 205.62 | 0.00 | ORB-long ORB[205.05,206.47] vol=2.2x ATR=0.70 |
| Stop hit — per-position SL triggered | 2025-05-27 10:15:00 | 207.05 | 205.72 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-05-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 10:35:00 | 210.99 | 213.74 | 0.00 | ORB-short ORB[214.70,217.46] vol=1.6x ATR=0.85 |
| Stop hit — per-position SL triggered | 2025-05-30 10:50:00 | 211.84 | 213.57 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 09:35:00 | 212.54 | 213.24 | 0.00 | ORB-short ORB[212.80,215.45] vol=3.8x ATR=0.71 |
| Stop hit — per-position SL triggered | 2025-06-03 10:00:00 | 213.25 | 213.03 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 09:30:00 | 209.96 | 210.74 | 0.00 | ORB-short ORB[210.45,212.04] vol=1.8x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 09:40:00 | 209.03 | 210.53 | 0.00 | T1 1.5R @ 209.03 |
| Stop hit — per-position SL triggered | 2025-06-04 10:00:00 | 209.96 | 209.78 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-05 11:15:00 | 211.46 | 211.94 | 0.00 | ORB-short ORB[211.89,213.61] vol=5.2x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-05 11:35:00 | 210.74 | 211.87 | 0.00 | T1 1.5R @ 210.74 |
| Target hit | 2025-06-05 15:20:00 | 206.65 | 209.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2025-06-06 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 10:25:00 | 210.05 | 207.37 | 0.00 | ORB-long ORB[206.04,208.63] vol=2.6x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 10:30:00 | 211.82 | 208.16 | 0.00 | T1 1.5R @ 211.82 |
| Stop hit — per-position SL triggered | 2025-06-06 10:50:00 | 210.05 | 209.31 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-12 09:50:00 | 223.52 | 222.13 | 0.00 | ORB-long ORB[220.00,222.46] vol=1.7x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-06-12 09:55:00 | 222.66 | 222.21 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:35:00 | 219.00 | 217.26 | 0.00 | ORB-long ORB[215.60,218.17] vol=1.7x ATR=0.82 |
| Stop hit — per-position SL triggered | 2025-06-17 09:40:00 | 218.18 | 217.42 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-18 09:30:00 | 220.00 | 218.58 | 0.00 | ORB-long ORB[216.79,219.60] vol=2.4x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-18 09:35:00 | 221.06 | 218.88 | 0.00 | T1 1.5R @ 221.06 |
| Target hit | 2025-06-18 15:20:00 | 227.95 | 225.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2025-06-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 11:10:00 | 228.61 | 226.98 | 0.00 | ORB-long ORB[225.63,228.50] vol=1.7x ATR=0.91 |
| Stop hit — per-position SL triggered | 2025-06-19 11:25:00 | 227.70 | 227.08 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 11:15:00 | 226.10 | 224.86 | 0.00 | ORB-long ORB[222.80,225.62] vol=3.7x ATR=0.67 |
| Stop hit — per-position SL triggered | 2025-06-20 11:35:00 | 225.43 | 225.00 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-06-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 09:45:00 | 227.84 | 226.31 | 0.00 | ORB-long ORB[224.03,226.88] vol=1.6x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-23 10:05:00 | 229.41 | 226.94 | 0.00 | T1 1.5R @ 229.41 |
| Stop hit — per-position SL triggered | 2025-06-23 10:25:00 | 227.84 | 227.16 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-06-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-27 09:40:00 | 236.03 | 238.23 | 0.00 | ORB-short ORB[237.70,239.94] vol=2.5x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 09:55:00 | 234.51 | 237.60 | 0.00 | T1 1.5R @ 234.51 |
| Stop hit — per-position SL triggered | 2025-06-27 10:05:00 | 236.03 | 237.40 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-03 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 10:35:00 | 253.50 | 250.95 | 0.00 | ORB-long ORB[248.06,250.70] vol=2.5x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 10:50:00 | 255.12 | 252.21 | 0.00 | T1 1.5R @ 255.12 |
| Target hit | 2025-07-03 13:50:00 | 253.94 | 253.96 | 0.00 | Trail-exit close<VWAP |

### Cycle 20 — SELL (started 2025-07-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 11:10:00 | 251.26 | 253.26 | 0.00 | ORB-short ORB[252.70,254.43] vol=2.8x ATR=0.74 |
| Stop hit — per-position SL triggered | 2025-07-08 11:25:00 | 252.00 | 253.09 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-07-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 10:00:00 | 252.81 | 251.68 | 0.00 | ORB-long ORB[249.96,251.80] vol=2.5x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 10:05:00 | 253.93 | 252.11 | 0.00 | T1 1.5R @ 253.93 |
| Target hit | 2025-07-09 15:20:00 | 260.36 | 257.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — SELL (started 2025-07-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 11:10:00 | 257.13 | 259.10 | 0.00 | ORB-short ORB[257.61,259.23] vol=2.0x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 11:30:00 | 255.93 | 258.82 | 0.00 | T1 1.5R @ 255.93 |
| Target hit | 2025-07-10 15:20:00 | 255.25 | 255.72 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2025-07-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 09:40:00 | 257.00 | 259.31 | 0.00 | ORB-short ORB[260.25,262.81] vol=3.3x ATR=0.90 |
| Stop hit — per-position SL triggered | 2025-07-22 09:50:00 | 257.90 | 258.56 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-07-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 10:55:00 | 257.16 | 255.32 | 0.00 | ORB-long ORB[254.50,256.68] vol=2.9x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 11:10:00 | 258.56 | 255.93 | 0.00 | T1 1.5R @ 258.56 |
| Target hit | 2025-07-23 15:20:00 | 258.65 | 257.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — SELL (started 2025-07-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-29 10:55:00 | 251.82 | 254.32 | 0.00 | ORB-short ORB[253.00,256.72] vol=2.6x ATR=0.84 |
| Stop hit — per-position SL triggered | 2025-07-29 11:50:00 | 252.66 | 253.51 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-08-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 10:20:00 | 264.05 | 264.87 | 0.00 | ORB-short ORB[264.55,267.00] vol=3.1x ATR=0.99 |
| Stop hit — per-position SL triggered | 2025-08-05 15:00:00 | 265.04 | 264.12 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-08-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 11:00:00 | 258.90 | 262.89 | 0.00 | ORB-short ORB[263.55,266.70] vol=3.0x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 11:10:00 | 257.35 | 262.35 | 0.00 | T1 1.5R @ 257.35 |
| Stop hit — per-position SL triggered | 2025-08-06 11:50:00 | 258.90 | 260.97 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-08-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-07 09:30:00 | 261.85 | 259.72 | 0.00 | ORB-long ORB[257.15,260.95] vol=1.6x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 09:35:00 | 263.24 | 261.13 | 0.00 | T1 1.5R @ 263.24 |
| Target hit | 2025-08-07 09:55:00 | 262.85 | 262.91 | 0.00 | Trail-exit close<VWAP |

### Cycle 29 — BUY (started 2025-08-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 09:35:00 | 257.45 | 255.76 | 0.00 | ORB-long ORB[253.75,256.25] vol=2.8x ATR=1.10 |
| Stop hit — per-position SL triggered | 2025-08-13 10:05:00 | 256.35 | 256.04 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-08-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-14 09:30:00 | 256.05 | 255.22 | 0.00 | ORB-long ORB[253.60,255.75] vol=4.4x ATR=0.95 |
| Stop hit — per-position SL triggered | 2025-08-14 10:00:00 | 255.10 | 255.40 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-08-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 10:40:00 | 257.25 | 255.38 | 0.00 | ORB-long ORB[253.05,256.20] vol=1.5x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 10:55:00 | 258.48 | 255.59 | 0.00 | T1 1.5R @ 258.48 |
| Target hit | 2025-08-18 15:20:00 | 259.15 | 257.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — SELL (started 2025-08-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 10:10:00 | 255.35 | 256.68 | 0.00 | ORB-short ORB[257.00,260.45] vol=1.5x ATR=0.85 |
| Stop hit — per-position SL triggered | 2025-08-19 10:15:00 | 256.20 | 256.65 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-08-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 09:30:00 | 259.10 | 257.33 | 0.00 | ORB-long ORB[254.50,258.00] vol=3.1x ATR=1.14 |
| Stop hit — per-position SL triggered | 2025-08-25 09:55:00 | 257.96 | 258.20 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 09:45:00 | 270.35 | 267.79 | 0.00 | ORB-long ORB[264.45,268.25] vol=2.5x ATR=1.18 |
| Stop hit — per-position SL triggered | 2025-09-02 09:55:00 | 269.17 | 268.04 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-09-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 10:10:00 | 272.85 | 271.00 | 0.00 | ORB-long ORB[268.60,271.75] vol=1.6x ATR=1.17 |
| Stop hit — per-position SL triggered | 2025-09-03 10:35:00 | 271.68 | 271.38 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 11:05:00 | 274.20 | 273.14 | 0.00 | ORB-long ORB[271.60,274.00] vol=1.8x ATR=0.66 |
| Stop hit — per-position SL triggered | 2025-09-11 11:30:00 | 273.54 | 273.18 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-09-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 09:30:00 | 271.65 | 271.02 | 0.00 | ORB-long ORB[269.70,271.55] vol=1.9x ATR=0.65 |
| Stop hit — per-position SL triggered | 2025-09-12 09:35:00 | 271.00 | 271.02 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-09-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 09:50:00 | 266.30 | 266.77 | 0.00 | ORB-short ORB[266.75,268.75] vol=3.6x ATR=0.78 |
| Stop hit — per-position SL triggered | 2025-09-17 10:55:00 | 267.08 | 266.44 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-09-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 10:20:00 | 266.05 | 264.38 | 0.00 | ORB-long ORB[262.95,265.40] vol=1.7x ATR=0.71 |
| Stop hit — per-position SL triggered | 2025-09-18 11:00:00 | 265.34 | 264.89 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-09-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 10:55:00 | 267.40 | 268.68 | 0.00 | ORB-short ORB[268.20,270.70] vol=2.2x ATR=0.74 |
| Stop hit — per-position SL triggered | 2025-09-23 11:15:00 | 268.14 | 268.56 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-09-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 10:00:00 | 271.45 | 270.23 | 0.00 | ORB-long ORB[268.60,270.50] vol=2.1x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 10:05:00 | 272.70 | 270.82 | 0.00 | T1 1.5R @ 272.70 |
| Stop hit — per-position SL triggered | 2025-09-24 10:30:00 | 271.45 | 270.98 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-09-25 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-25 10:50:00 | 272.60 | 271.39 | 0.00 | ORB-long ORB[269.60,271.85] vol=4.0x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 11:00:00 | 273.68 | 271.64 | 0.00 | T1 1.5R @ 273.68 |
| Target hit | 2025-09-25 13:40:00 | 274.00 | 274.20 | 0.00 | Trail-exit close<VWAP |

### Cycle 43 — BUY (started 2025-09-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 10:10:00 | 276.50 | 275.79 | 0.00 | ORB-long ORB[273.10,275.50] vol=1.6x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 10:15:00 | 277.83 | 276.04 | 0.00 | T1 1.5R @ 277.83 |
| Target hit | 2025-09-29 12:10:00 | 277.25 | 277.61 | 0.00 | Trail-exit close<VWAP |

### Cycle 44 — SELL (started 2025-09-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-30 09:50:00 | 272.85 | 275.11 | 0.00 | ORB-short ORB[275.45,278.80] vol=1.5x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 10:00:00 | 271.27 | 274.54 | 0.00 | T1 1.5R @ 271.27 |
| Stop hit — per-position SL triggered | 2025-09-30 10:30:00 | 272.85 | 273.71 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-10-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 09:30:00 | 275.15 | 276.62 | 0.00 | ORB-short ORB[275.40,278.00] vol=2.3x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 09:45:00 | 273.66 | 275.93 | 0.00 | T1 1.5R @ 273.66 |
| Stop hit — per-position SL triggered | 2025-10-06 11:00:00 | 275.15 | 275.10 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-10-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 10:00:00 | 277.40 | 275.75 | 0.00 | ORB-long ORB[274.15,276.25] vol=3.7x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 10:05:00 | 278.86 | 277.11 | 0.00 | T1 1.5R @ 278.86 |
| Target hit | 2025-10-08 10:55:00 | 280.05 | 280.19 | 0.00 | Trail-exit close<VWAP |

### Cycle 47 — BUY (started 2025-10-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 10:25:00 | 289.20 | 286.69 | 0.00 | ORB-long ORB[284.50,287.60] vol=2.3x ATR=0.99 |
| Stop hit — per-position SL triggered | 2025-10-10 10:35:00 | 288.21 | 287.27 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-10-15 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:30:00 | 298.20 | 295.56 | 0.00 | ORB-long ORB[292.40,296.30] vol=1.7x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 10:50:00 | 300.12 | 296.72 | 0.00 | T1 1.5R @ 300.12 |
| Stop hit — per-position SL triggered | 2025-10-15 10:55:00 | 298.20 | 296.83 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-10-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 11:05:00 | 301.30 | 303.52 | 0.00 | ORB-short ORB[303.50,305.50] vol=1.5x ATR=1.03 |
| Stop hit — per-position SL triggered | 2025-10-17 11:20:00 | 302.33 | 303.34 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-10-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-23 10:00:00 | 318.95 | 323.38 | 0.00 | ORB-short ORB[323.75,327.00] vol=1.8x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-23 10:35:00 | 316.59 | 320.57 | 0.00 | T1 1.5R @ 316.59 |
| Stop hit — per-position SL triggered | 2025-10-23 10:50:00 | 318.95 | 319.97 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-10-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-30 11:10:00 | 325.00 | 323.17 | 0.00 | ORB-long ORB[322.00,324.35] vol=4.2x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 11:30:00 | 326.00 | 323.61 | 0.00 | T1 1.5R @ 326.00 |
| Stop hit — per-position SL triggered | 2025-10-30 12:55:00 | 325.00 | 324.97 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-10-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:35:00 | 328.20 | 327.07 | 0.00 | ORB-long ORB[324.25,326.90] vol=4.7x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 09:40:00 | 329.57 | 328.24 | 0.00 | T1 1.5R @ 329.57 |
| Target hit | 2025-10-31 11:05:00 | 329.50 | 329.53 | 0.00 | Trail-exit close<VWAP |

### Cycle 53 — SELL (started 2025-11-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 10:05:00 | 325.05 | 325.82 | 0.00 | ORB-short ORB[325.10,329.40] vol=3.6x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 10:15:00 | 323.46 | 325.47 | 0.00 | T1 1.5R @ 323.46 |
| Stop hit — per-position SL triggered | 2025-11-04 10:25:00 | 325.05 | 325.18 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-11-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 09:30:00 | 320.60 | 322.00 | 0.00 | ORB-short ORB[321.20,325.25] vol=3.9x ATR=1.08 |
| Stop hit — per-position SL triggered | 2025-11-07 09:35:00 | 321.68 | 321.92 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-11-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 10:05:00 | 324.20 | 327.47 | 0.00 | ORB-short ORB[327.40,331.00] vol=1.6x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 10:25:00 | 322.76 | 326.42 | 0.00 | T1 1.5R @ 322.76 |
| Stop hit — per-position SL triggered | 2025-11-10 10:35:00 | 324.20 | 326.20 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-11-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 09:55:00 | 321.95 | 320.75 | 0.00 | ORB-long ORB[318.05,321.50] vol=2.2x ATR=0.97 |
| Stop hit — per-position SL triggered | 2025-11-13 10:10:00 | 320.98 | 320.70 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-11-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 09:30:00 | 311.55 | 310.33 | 0.00 | ORB-long ORB[309.05,311.00] vol=1.9x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 10:00:00 | 312.96 | 311.04 | 0.00 | T1 1.5R @ 312.96 |
| Target hit | 2025-11-20 14:40:00 | 313.20 | 313.63 | 0.00 | Trail-exit close<VWAP |

### Cycle 58 — SELL (started 2025-11-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 10:05:00 | 314.90 | 315.43 | 0.00 | ORB-short ORB[315.35,318.10] vol=2.6x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 10:15:00 | 313.84 | 314.88 | 0.00 | T1 1.5R @ 313.84 |
| Stop hit — per-position SL triggered | 2025-11-27 10:35:00 | 314.90 | 314.75 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-12-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 10:50:00 | 309.90 | 311.80 | 0.00 | ORB-short ORB[311.85,314.10] vol=1.8x ATR=0.58 |
| Stop hit — per-position SL triggered | 2025-12-01 11:05:00 | 310.48 | 311.58 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-12-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 10:50:00 | 304.60 | 305.53 | 0.00 | ORB-short ORB[305.60,307.25] vol=2.4x ATR=0.65 |
| Stop hit — per-position SL triggered | 2025-12-02 11:00:00 | 305.25 | 305.47 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-12-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 10:05:00 | 300.60 | 302.00 | 0.00 | ORB-short ORB[301.20,303.90] vol=1.5x ATR=0.92 |
| Stop hit — per-position SL triggered | 2025-12-03 11:05:00 | 301.52 | 301.20 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-12-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-04 10:50:00 | 300.70 | 302.25 | 0.00 | ORB-short ORB[301.80,304.30] vol=2.0x ATR=0.75 |
| Stop hit — per-position SL triggered | 2025-12-04 10:55:00 | 301.45 | 302.19 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 11:15:00 | 301.70 | 303.12 | 0.00 | ORB-short ORB[303.45,306.75] vol=2.1x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 12:00:00 | 300.51 | 302.36 | 0.00 | T1 1.5R @ 300.51 |
| Target hit | 2025-12-08 15:20:00 | 300.00 | 300.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — SELL (started 2025-12-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 09:35:00 | 295.20 | 296.68 | 0.00 | ORB-short ORB[295.40,299.60] vol=1.7x ATR=1.03 |
| Stop hit — per-position SL triggered | 2025-12-09 10:00:00 | 296.23 | 295.65 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-12-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 10:20:00 | 306.85 | 303.92 | 0.00 | ORB-long ORB[301.00,303.75] vol=2.2x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 10:25:00 | 308.25 | 304.58 | 0.00 | T1 1.5R @ 308.25 |
| Target hit | 2025-12-11 15:20:00 | 311.25 | 309.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — BUY (started 2025-12-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:30:00 | 313.15 | 311.45 | 0.00 | ORB-long ORB[309.10,311.20] vol=1.7x ATR=1.02 |
| Stop hit — per-position SL triggered | 2025-12-12 10:05:00 | 312.13 | 312.75 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-12-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 09:35:00 | 304.45 | 305.53 | 0.00 | ORB-short ORB[305.40,307.60] vol=2.1x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-15 09:45:00 | 303.38 | 304.98 | 0.00 | T1 1.5R @ 303.38 |
| Stop hit — per-position SL triggered | 2025-12-15 10:00:00 | 304.45 | 304.71 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-12-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-19 11:00:00 | 297.30 | 299.08 | 0.00 | ORB-short ORB[298.20,300.40] vol=2.8x ATR=0.62 |
| Stop hit — per-position SL triggered | 2025-12-19 11:15:00 | 297.92 | 299.01 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-12-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 09:30:00 | 305.70 | 304.51 | 0.00 | ORB-long ORB[303.15,304.80] vol=2.1x ATR=0.88 |
| Stop hit — per-position SL triggered | 2025-12-23 09:40:00 | 304.82 | 304.88 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-12-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 09:30:00 | 311.30 | 308.77 | 0.00 | ORB-long ORB[306.80,309.50] vol=3.0x ATR=1.21 |
| Stop hit — per-position SL triggered | 2025-12-24 09:35:00 | 310.09 | 309.03 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-12-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 09:30:00 | 307.45 | 305.82 | 0.00 | ORB-long ORB[302.50,306.25] vol=3.5x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 10:00:00 | 308.90 | 307.16 | 0.00 | T1 1.5R @ 308.90 |
| Stop hit — per-position SL triggered | 2025-12-30 10:55:00 | 307.45 | 307.84 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-12-31 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:45:00 | 314.00 | 312.26 | 0.00 | ORB-long ORB[308.65,310.00] vol=1.7x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 10:55:00 | 315.31 | 312.68 | 0.00 | T1 1.5R @ 315.31 |
| Stop hit — per-position SL triggered | 2025-12-31 11:00:00 | 314.00 | 312.84 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-01-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 09:55:00 | 317.45 | 316.32 | 0.00 | ORB-long ORB[314.25,317.10] vol=2.0x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 10:40:00 | 318.64 | 317.35 | 0.00 | T1 1.5R @ 318.64 |
| Target hit | 2026-01-02 12:55:00 | 318.05 | 318.48 | 0.00 | Trail-exit close<VWAP |

### Cycle 74 — SELL (started 2026-01-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:55:00 | 313.90 | 315.42 | 0.00 | ORB-short ORB[315.30,319.00] vol=1.5x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:10:00 | 312.50 | 314.91 | 0.00 | T1 1.5R @ 312.50 |
| Target hit | 2026-01-08 15:20:00 | 310.10 | 311.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — BUY (started 2026-01-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-12 09:50:00 | 307.85 | 305.20 | 0.00 | ORB-long ORB[303.30,305.90] vol=2.4x ATR=1.51 |
| Stop hit — per-position SL triggered | 2026-01-12 09:55:00 | 306.34 | 305.30 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2026-01-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-13 09:40:00 | 303.90 | 304.63 | 0.00 | ORB-short ORB[304.00,307.70] vol=1.5x ATR=1.23 |
| Stop hit — per-position SL triggered | 2026-01-13 09:45:00 | 305.13 | 304.63 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-01-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 10:55:00 | 308.45 | 306.74 | 0.00 | ORB-long ORB[304.25,307.30] vol=2.2x ATR=1.02 |
| Stop hit — per-position SL triggered | 2026-01-14 12:10:00 | 307.43 | 307.31 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-01-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 11:00:00 | 295.95 | 299.77 | 0.00 | ORB-short ORB[299.40,303.55] vol=1.9x ATR=1.05 |
| Stop hit — per-position SL triggered | 2026-01-22 11:20:00 | 297.00 | 299.53 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2026-01-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 09:55:00 | 294.90 | 292.18 | 0.00 | ORB-long ORB[288.25,292.15] vol=2.2x ATR=1.34 |
| Stop hit — per-position SL triggered | 2026-01-27 10:30:00 | 293.56 | 293.25 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2026-01-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 09:30:00 | 296.10 | 296.84 | 0.00 | ORB-short ORB[296.25,298.20] vol=1.8x ATR=0.79 |
| Stop hit — per-position SL triggered | 2026-01-29 09:40:00 | 296.89 | 296.64 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2026-01-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 09:45:00 | 299.55 | 296.96 | 0.00 | ORB-long ORB[292.55,296.00] vol=5.7x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 11:00:00 | 301.44 | 298.51 | 0.00 | T1 1.5R @ 301.44 |
| Stop hit — per-position SL triggered | 2026-01-30 13:00:00 | 299.55 | 299.54 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2026-02-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 09:40:00 | 303.60 | 304.77 | 0.00 | ORB-short ORB[304.20,306.50] vol=2.4x ATR=0.72 |
| Stop hit — per-position SL triggered | 2026-02-05 10:10:00 | 304.32 | 304.33 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2026-02-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 10:45:00 | 300.15 | 302.08 | 0.00 | ORB-short ORB[302.90,304.90] vol=1.8x ATR=0.88 |
| Stop hit — per-position SL triggered | 2026-02-06 11:30:00 | 301.03 | 301.82 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2026-02-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 09:35:00 | 306.90 | 305.35 | 0.00 | ORB-long ORB[302.50,305.80] vol=1.5x ATR=0.90 |
| Stop hit — per-position SL triggered | 2026-02-09 09:50:00 | 306.00 | 305.64 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:15:00 | 310.10 | 307.95 | 0.00 | ORB-long ORB[306.35,309.90] vol=2.7x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 10:20:00 | 311.59 | 308.53 | 0.00 | T1 1.5R @ 311.59 |
| Target hit | 2026-02-12 11:30:00 | 311.10 | 311.27 | 0.00 | Trail-exit close<VWAP |

### Cycle 86 — BUY (started 2026-02-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:30:00 | 317.40 | 315.63 | 0.00 | ORB-long ORB[312.25,316.75] vol=3.8x ATR=1.13 |
| Stop hit — per-position SL triggered | 2026-02-17 10:15:00 | 316.27 | 316.58 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2026-02-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:10:00 | 322.15 | 324.19 | 0.00 | ORB-short ORB[322.40,326.85] vol=1.6x ATR=1.19 |
| Stop hit — per-position SL triggered | 2026-02-18 10:25:00 | 323.34 | 324.05 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2026-02-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:55:00 | 318.25 | 321.21 | 0.00 | ORB-short ORB[321.00,324.80] vol=1.7x ATR=1.11 |
| Stop hit — per-position SL triggered | 2026-02-27 11:00:00 | 319.36 | 321.04 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 306.20 | 308.42 | 0.00 | ORB-short ORB[308.35,310.10] vol=1.8x ATR=1.03 |
| Stop hit — per-position SL triggered | 2026-03-06 11:00:00 | 307.23 | 308.31 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2026-03-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:45:00 | 306.10 | 307.52 | 0.00 | ORB-short ORB[307.85,309.85] vol=1.6x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:05:00 | 304.94 | 306.90 | 0.00 | T1 1.5R @ 304.94 |
| Target hit | 2026-03-11 15:20:00 | 297.15 | 301.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 91 — BUY (started 2026-03-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 09:55:00 | 297.40 | 295.21 | 0.00 | ORB-long ORB[292.90,296.00] vol=1.8x ATR=1.18 |
| Stop hit — per-position SL triggered | 2026-03-12 10:15:00 | 296.22 | 295.79 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2026-03-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 11:15:00 | 296.25 | 293.75 | 0.00 | ORB-long ORB[292.00,294.85] vol=2.4x ATR=1.29 |
| Stop hit — per-position SL triggered | 2026-03-16 11:50:00 | 294.96 | 293.94 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2026-03-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 09:45:00 | 293.15 | 293.83 | 0.00 | ORB-short ORB[293.20,297.40] vol=3.4x ATR=1.28 |
| Stop hit — per-position SL triggered | 2026-03-17 10:15:00 | 294.43 | 293.84 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2026-03-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 09:40:00 | 293.25 | 295.15 | 0.00 | ORB-short ORB[294.20,297.25] vol=1.7x ATR=1.25 |
| Stop hit — per-position SL triggered | 2026-03-19 10:10:00 | 294.50 | 294.68 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2026-03-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:40:00 | 308.25 | 303.91 | 0.00 | ORB-long ORB[299.00,303.60] vol=2.5x ATR=1.29 |
| Stop hit — per-position SL triggered | 2026-03-25 10:55:00 | 306.96 | 304.45 | 0.00 | SL hit |

### Cycle 96 — SELL (started 2026-04-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:30:00 | 316.95 | 319.53 | 0.00 | ORB-short ORB[318.10,322.15] vol=1.6x ATR=1.30 |
| Stop hit — per-position SL triggered | 2026-04-09 09:35:00 | 318.25 | 319.06 | 0.00 | SL hit |

### Cycle 97 — BUY (started 2026-04-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:40:00 | 322.05 | 321.01 | 0.00 | ORB-long ORB[318.70,321.80] vol=2.4x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 09:50:00 | 323.82 | 322.44 | 0.00 | T1 1.5R @ 323.82 |
| Target hit | 2026-04-10 11:00:00 | 324.40 | 324.46 | 0.00 | Trail-exit close<VWAP |

### Cycle 98 — SELL (started 2026-04-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 11:05:00 | 318.55 | 321.57 | 0.00 | ORB-short ORB[320.25,324.40] vol=1.6x ATR=0.92 |
| Stop hit — per-position SL triggered | 2026-04-15 11:15:00 | 319.47 | 321.37 | 0.00 | SL hit |

### Cycle 99 — SELL (started 2026-04-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 09:40:00 | 313.40 | 315.97 | 0.00 | ORB-short ORB[314.90,318.15] vol=2.5x ATR=1.03 |
| Stop hit — per-position SL triggered | 2026-04-17 09:55:00 | 314.43 | 315.55 | 0.00 | SL hit |

### Cycle 100 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 322.35 | 320.36 | 0.00 | ORB-long ORB[317.85,320.40] vol=1.7x ATR=1.13 |
| Stop hit — per-position SL triggered | 2026-04-21 10:15:00 | 321.22 | 322.08 | 0.00 | SL hit |

### Cycle 101 — SELL (started 2026-04-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:20:00 | 311.25 | 314.08 | 0.00 | ORB-short ORB[312.80,317.25] vol=2.0x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:30:00 | 309.90 | 313.70 | 0.00 | T1 1.5R @ 309.90 |
| Stop hit — per-position SL triggered | 2026-04-24 11:10:00 | 311.25 | 312.50 | 0.00 | SL hit |

### Cycle 102 — BUY (started 2026-04-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:05:00 | 336.60 | 332.03 | 0.00 | ORB-long ORB[321.80,326.80] vol=3.1x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:55:00 | 339.74 | 334.65 | 0.00 | T1 1.5R @ 339.74 |
| Target hit | 2026-04-29 15:20:00 | 341.10 | 339.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 103 — BUY (started 2026-05-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:40:00 | 343.65 | 341.51 | 0.00 | ORB-long ORB[339.00,342.35] vol=2.5x ATR=1.37 |
| Stop hit — per-position SL triggered | 2026-05-04 10:05:00 | 342.28 | 341.99 | 0.00 | SL hit |

### Cycle 104 — BUY (started 2026-05-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:50:00 | 331.90 | 328.65 | 0.00 | ORB-long ORB[326.15,330.90] vol=1.6x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 09:55:00 | 334.20 | 330.91 | 0.00 | T1 1.5R @ 334.20 |
| Stop hit — per-position SL triggered | 2026-05-05 10:25:00 | 331.90 | 331.37 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-19 09:35:00 | 212.64 | 2025-05-19 09:50:00 | 213.31 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-05-20 09:30:00 | 210.63 | 2025-05-20 09:35:00 | 211.42 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-05-21 10:55:00 | 209.52 | 2025-05-21 12:00:00 | 208.90 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-05-23 10:50:00 | 209.58 | 2025-05-23 11:15:00 | 209.13 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-05-26 10:05:00 | 208.95 | 2025-05-26 10:10:00 | 209.46 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-05-27 10:10:00 | 207.75 | 2025-05-27 10:15:00 | 207.05 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-05-30 10:35:00 | 210.99 | 2025-05-30 10:50:00 | 211.84 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-06-03 09:35:00 | 212.54 | 2025-06-03 10:00:00 | 213.25 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-06-04 09:30:00 | 209.96 | 2025-06-04 09:40:00 | 209.03 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-06-04 09:30:00 | 209.96 | 2025-06-04 10:00:00 | 209.96 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-05 11:15:00 | 211.46 | 2025-06-05 11:35:00 | 210.74 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-06-05 11:15:00 | 211.46 | 2025-06-05 15:20:00 | 206.65 | TARGET_HIT | 0.50 | 2.27% |
| BUY | retest1 | 2025-06-06 10:25:00 | 210.05 | 2025-06-06 10:30:00 | 211.82 | PARTIAL | 0.50 | 0.84% |
| BUY | retest1 | 2025-06-06 10:25:00 | 210.05 | 2025-06-06 10:50:00 | 210.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-12 09:50:00 | 223.52 | 2025-06-12 09:55:00 | 222.66 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-06-17 09:35:00 | 219.00 | 2025-06-17 09:40:00 | 218.18 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-06-18 09:30:00 | 220.00 | 2025-06-18 09:35:00 | 221.06 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-06-18 09:30:00 | 220.00 | 2025-06-18 15:20:00 | 227.95 | TARGET_HIT | 0.50 | 3.61% |
| BUY | retest1 | 2025-06-19 11:10:00 | 228.61 | 2025-06-19 11:25:00 | 227.70 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-06-20 11:15:00 | 226.10 | 2025-06-20 11:35:00 | 225.43 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-06-23 09:45:00 | 227.84 | 2025-06-23 10:05:00 | 229.41 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2025-06-23 09:45:00 | 227.84 | 2025-06-23 10:25:00 | 227.84 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-27 09:40:00 | 236.03 | 2025-06-27 09:55:00 | 234.51 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2025-06-27 09:40:00 | 236.03 | 2025-06-27 10:05:00 | 236.03 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-03 10:35:00 | 253.50 | 2025-07-03 10:50:00 | 255.12 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-07-03 10:35:00 | 253.50 | 2025-07-03 13:50:00 | 253.94 | TARGET_HIT | 0.50 | 0.17% |
| SELL | retest1 | 2025-07-08 11:10:00 | 251.26 | 2025-07-08 11:25:00 | 252.00 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-07-09 10:00:00 | 252.81 | 2025-07-09 10:05:00 | 253.93 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-07-09 10:00:00 | 252.81 | 2025-07-09 15:20:00 | 260.36 | TARGET_HIT | 0.50 | 2.99% |
| SELL | retest1 | 2025-07-10 11:10:00 | 257.13 | 2025-07-10 11:30:00 | 255.93 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-07-10 11:10:00 | 257.13 | 2025-07-10 15:20:00 | 255.25 | TARGET_HIT | 0.50 | 0.73% |
| SELL | retest1 | 2025-07-22 09:40:00 | 257.00 | 2025-07-22 09:50:00 | 257.90 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-07-23 10:55:00 | 257.16 | 2025-07-23 11:10:00 | 258.56 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-07-23 10:55:00 | 257.16 | 2025-07-23 15:20:00 | 258.65 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2025-07-29 10:55:00 | 251.82 | 2025-07-29 11:50:00 | 252.66 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-08-05 10:20:00 | 264.05 | 2025-08-05 15:00:00 | 265.04 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-08-06 11:00:00 | 258.90 | 2025-08-06 11:10:00 | 257.35 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2025-08-06 11:00:00 | 258.90 | 2025-08-06 11:50:00 | 258.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-07 09:30:00 | 261.85 | 2025-08-07 09:35:00 | 263.24 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-08-07 09:30:00 | 261.85 | 2025-08-07 09:55:00 | 262.85 | TARGET_HIT | 0.50 | 0.38% |
| BUY | retest1 | 2025-08-13 09:35:00 | 257.45 | 2025-08-13 10:05:00 | 256.35 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-08-14 09:30:00 | 256.05 | 2025-08-14 10:00:00 | 255.10 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-08-18 10:40:00 | 257.25 | 2025-08-18 10:55:00 | 258.48 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-08-18 10:40:00 | 257.25 | 2025-08-18 15:20:00 | 259.15 | TARGET_HIT | 0.50 | 0.74% |
| SELL | retest1 | 2025-08-19 10:10:00 | 255.35 | 2025-08-19 10:15:00 | 256.20 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-08-25 09:30:00 | 259.10 | 2025-08-25 09:55:00 | 257.96 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-09-02 09:45:00 | 270.35 | 2025-09-02 09:55:00 | 269.17 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-09-03 10:10:00 | 272.85 | 2025-09-03 10:35:00 | 271.68 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-09-11 11:05:00 | 274.20 | 2025-09-11 11:30:00 | 273.54 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-09-12 09:30:00 | 271.65 | 2025-09-12 09:35:00 | 271.00 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-09-17 09:50:00 | 266.30 | 2025-09-17 10:55:00 | 267.08 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-09-18 10:20:00 | 266.05 | 2025-09-18 11:00:00 | 265.34 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-09-23 10:55:00 | 267.40 | 2025-09-23 11:15:00 | 268.14 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-09-24 10:00:00 | 271.45 | 2025-09-24 10:05:00 | 272.70 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-09-24 10:00:00 | 271.45 | 2025-09-24 10:30:00 | 271.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-25 10:50:00 | 272.60 | 2025-09-25 11:00:00 | 273.68 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-09-25 10:50:00 | 272.60 | 2025-09-25 13:40:00 | 274.00 | TARGET_HIT | 0.50 | 0.51% |
| BUY | retest1 | 2025-09-29 10:10:00 | 276.50 | 2025-09-29 10:15:00 | 277.83 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-09-29 10:10:00 | 276.50 | 2025-09-29 12:10:00 | 277.25 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2025-09-30 09:50:00 | 272.85 | 2025-09-30 10:00:00 | 271.27 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-09-30 09:50:00 | 272.85 | 2025-09-30 10:30:00 | 272.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-06 09:30:00 | 275.15 | 2025-10-06 09:45:00 | 273.66 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-10-06 09:30:00 | 275.15 | 2025-10-06 11:00:00 | 275.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-08 10:00:00 | 277.40 | 2025-10-08 10:05:00 | 278.86 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-10-08 10:00:00 | 277.40 | 2025-10-08 10:55:00 | 280.05 | TARGET_HIT | 0.50 | 0.96% |
| BUY | retest1 | 2025-10-10 10:25:00 | 289.20 | 2025-10-10 10:35:00 | 288.21 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-10-15 10:30:00 | 298.20 | 2025-10-15 10:50:00 | 300.12 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-10-15 10:30:00 | 298.20 | 2025-10-15 10:55:00 | 298.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-17 11:05:00 | 301.30 | 2025-10-17 11:20:00 | 302.33 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-10-23 10:00:00 | 318.95 | 2025-10-23 10:35:00 | 316.59 | PARTIAL | 0.50 | 0.74% |
| SELL | retest1 | 2025-10-23 10:00:00 | 318.95 | 2025-10-23 10:50:00 | 318.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-30 11:10:00 | 325.00 | 2025-10-30 11:30:00 | 326.00 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-10-30 11:10:00 | 325.00 | 2025-10-30 12:55:00 | 325.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-31 09:35:00 | 328.20 | 2025-10-31 09:40:00 | 329.57 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-10-31 09:35:00 | 328.20 | 2025-10-31 11:05:00 | 329.50 | TARGET_HIT | 0.50 | 0.40% |
| SELL | retest1 | 2025-11-04 10:05:00 | 325.05 | 2025-11-04 10:15:00 | 323.46 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-11-04 10:05:00 | 325.05 | 2025-11-04 10:25:00 | 325.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-07 09:30:00 | 320.60 | 2025-11-07 09:35:00 | 321.68 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-11-10 10:05:00 | 324.20 | 2025-11-10 10:25:00 | 322.76 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-11-10 10:05:00 | 324.20 | 2025-11-10 10:35:00 | 324.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-13 09:55:00 | 321.95 | 2025-11-13 10:10:00 | 320.98 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-11-20 09:30:00 | 311.55 | 2025-11-20 10:00:00 | 312.96 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-11-20 09:30:00 | 311.55 | 2025-11-20 14:40:00 | 313.20 | TARGET_HIT | 0.50 | 0.53% |
| SELL | retest1 | 2025-11-27 10:05:00 | 314.90 | 2025-11-27 10:15:00 | 313.84 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-11-27 10:05:00 | 314.90 | 2025-11-27 10:35:00 | 314.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-01 10:50:00 | 309.90 | 2025-12-01 11:05:00 | 310.48 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-12-02 10:50:00 | 304.60 | 2025-12-02 11:00:00 | 305.25 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-12-03 10:05:00 | 300.60 | 2025-12-03 11:05:00 | 301.52 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-12-04 10:50:00 | 300.70 | 2025-12-04 10:55:00 | 301.45 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-12-08 11:15:00 | 301.70 | 2025-12-08 12:00:00 | 300.51 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-12-08 11:15:00 | 301.70 | 2025-12-08 15:20:00 | 300.00 | TARGET_HIT | 0.50 | 0.56% |
| SELL | retest1 | 2025-12-09 09:35:00 | 295.20 | 2025-12-09 10:00:00 | 296.23 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-12-11 10:20:00 | 306.85 | 2025-12-11 10:25:00 | 308.25 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-12-11 10:20:00 | 306.85 | 2025-12-11 15:20:00 | 311.25 | TARGET_HIT | 0.50 | 1.43% |
| BUY | retest1 | 2025-12-12 09:30:00 | 313.15 | 2025-12-12 10:05:00 | 312.13 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-12-15 09:35:00 | 304.45 | 2025-12-15 09:45:00 | 303.38 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-12-15 09:35:00 | 304.45 | 2025-12-15 10:00:00 | 304.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-19 11:00:00 | 297.30 | 2025-12-19 11:15:00 | 297.92 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-12-23 09:30:00 | 305.70 | 2025-12-23 09:40:00 | 304.82 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-12-24 09:30:00 | 311.30 | 2025-12-24 09:35:00 | 310.09 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-12-30 09:30:00 | 307.45 | 2025-12-30 10:00:00 | 308.90 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-12-30 09:30:00 | 307.45 | 2025-12-30 10:55:00 | 307.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-31 10:45:00 | 314.00 | 2025-12-31 10:55:00 | 315.31 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-12-31 10:45:00 | 314.00 | 2025-12-31 11:00:00 | 314.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-02 09:55:00 | 317.45 | 2026-01-02 10:40:00 | 318.64 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-01-02 09:55:00 | 317.45 | 2026-01-02 12:55:00 | 318.05 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2026-01-08 10:55:00 | 313.90 | 2026-01-08 11:10:00 | 312.50 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-01-08 10:55:00 | 313.90 | 2026-01-08 15:20:00 | 310.10 | TARGET_HIT | 0.50 | 1.21% |
| BUY | retest1 | 2026-01-12 09:50:00 | 307.85 | 2026-01-12 09:55:00 | 306.34 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-01-13 09:40:00 | 303.90 | 2026-01-13 09:45:00 | 305.13 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-01-14 10:55:00 | 308.45 | 2026-01-14 12:10:00 | 307.43 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-01-22 11:00:00 | 295.95 | 2026-01-22 11:20:00 | 297.00 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-01-27 09:55:00 | 294.90 | 2026-01-27 10:30:00 | 293.56 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-01-29 09:30:00 | 296.10 | 2026-01-29 09:40:00 | 296.89 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-01-30 09:45:00 | 299.55 | 2026-01-30 11:00:00 | 301.44 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-01-30 09:45:00 | 299.55 | 2026-01-30 13:00:00 | 299.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-05 09:40:00 | 303.60 | 2026-02-05 10:10:00 | 304.32 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-06 10:45:00 | 300.15 | 2026-02-06 11:30:00 | 301.03 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-09 09:35:00 | 306.90 | 2026-02-09 09:50:00 | 306.00 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-12 10:15:00 | 310.10 | 2026-02-12 10:20:00 | 311.59 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-02-12 10:15:00 | 310.10 | 2026-02-12 11:30:00 | 311.10 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2026-02-17 09:30:00 | 317.40 | 2026-02-17 10:15:00 | 316.27 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-18 10:10:00 | 322.15 | 2026-02-18 10:25:00 | 323.34 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-27 10:55:00 | 318.25 | 2026-02-27 11:00:00 | 319.36 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-03-06 10:45:00 | 306.20 | 2026-03-06 11:00:00 | 307.23 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-11 10:45:00 | 306.10 | 2026-03-11 11:05:00 | 304.94 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-03-11 10:45:00 | 306.10 | 2026-03-11 15:20:00 | 297.15 | TARGET_HIT | 0.50 | 2.92% |
| BUY | retest1 | 2026-03-12 09:55:00 | 297.40 | 2026-03-12 10:15:00 | 296.22 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-03-16 11:15:00 | 296.25 | 2026-03-16 11:50:00 | 294.96 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-03-17 09:45:00 | 293.15 | 2026-03-17 10:15:00 | 294.43 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-03-19 09:40:00 | 293.25 | 2026-03-19 10:10:00 | 294.50 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-03-25 10:40:00 | 308.25 | 2026-03-25 10:55:00 | 306.96 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-04-09 09:30:00 | 316.95 | 2026-04-09 09:35:00 | 318.25 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-10 09:40:00 | 322.05 | 2026-04-10 09:50:00 | 323.82 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-04-10 09:40:00 | 322.05 | 2026-04-10 11:00:00 | 324.40 | TARGET_HIT | 0.50 | 0.73% |
| SELL | retest1 | 2026-04-15 11:05:00 | 318.55 | 2026-04-15 11:15:00 | 319.47 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-04-17 09:40:00 | 313.40 | 2026-04-17 09:55:00 | 314.43 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-21 09:35:00 | 322.35 | 2026-04-21 10:15:00 | 321.22 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-24 10:20:00 | 311.25 | 2026-04-24 10:30:00 | 309.90 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-04-24 10:20:00 | 311.25 | 2026-04-24 11:10:00 | 311.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 10:05:00 | 336.60 | 2026-04-29 10:55:00 | 339.74 | PARTIAL | 0.50 | 0.93% |
| BUY | retest1 | 2026-04-29 10:05:00 | 336.60 | 2026-04-29 15:20:00 | 341.10 | TARGET_HIT | 0.50 | 1.34% |
| BUY | retest1 | 2026-05-04 09:40:00 | 343.65 | 2026-05-04 10:05:00 | 342.28 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-05-05 09:50:00 | 331.90 | 2026-05-05 09:55:00 | 334.20 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2026-05-05 09:50:00 | 331.90 | 2026-05-05 10:25:00 | 331.90 | STOP_HIT | 0.50 | 0.00% |
