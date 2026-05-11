# Eternal Ltd. (ETERNAL)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (14113 bars)
- **Last close:** 256.15
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
| ENTRY1 | 52 |
| ENTRY2 | 0 |
| PARTIAL | 26 |
| TARGET_HIT | 7 |
| STOP_HIT | 45 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 78 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 45
- **Target hits / Stop hits / Partials:** 7 / 45 / 26
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 13.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 45 | 22 | 48.9% | 6 | 23 | 16 | 0.23% | 10.6% |
| BUY @ 2nd Alert (retest1) | 45 | 22 | 48.9% | 6 | 23 | 16 | 0.23% | 10.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 33 | 11 | 33.3% | 1 | 22 | 10 | 0.08% | 2.6% |
| SELL @ 2nd Alert (retest1) | 33 | 11 | 33.3% | 1 | 22 | 10 | 0.08% | 2.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 78 | 33 | 42.3% | 7 | 45 | 26 | 0.17% | 13.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 10:40:00 | 237.77 | 235.47 | 0.00 | ORB-long ORB[234.32,236.77] vol=1.8x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-15 11:05:00 | 238.90 | 236.22 | 0.00 | T1 1.5R @ 238.90 |
| Stop hit — per-position SL triggered | 2025-05-15 11:10:00 | 237.77 | 236.28 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-20 10:20:00 | 233.73 | 236.44 | 0.00 | ORB-short ORB[237.11,240.00] vol=1.9x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-20 11:05:00 | 232.32 | 235.20 | 0.00 | T1 1.5R @ 232.32 |
| Target hit | 2025-05-20 15:20:00 | 228.11 | 231.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2025-05-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 09:50:00 | 228.25 | 226.05 | 0.00 | ORB-long ORB[224.66,227.82] vol=1.9x ATR=1.09 |
| Stop hit — per-position SL triggered | 2025-05-21 09:55:00 | 227.16 | 226.10 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-29 09:30:00 | 226.65 | 225.46 | 0.00 | ORB-long ORB[224.03,226.37] vol=1.7x ATR=0.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-29 09:35:00 | 227.55 | 225.86 | 0.00 | T1 1.5R @ 227.55 |
| Stop hit — per-position SL triggered | 2025-05-29 09:40:00 | 226.65 | 225.94 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 09:35:00 | 241.70 | 243.63 | 0.00 | ORB-short ORB[242.65,245.83] vol=1.5x ATR=1.06 |
| Stop hit — per-position SL triggered | 2025-06-03 09:45:00 | 242.76 | 243.20 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 11:00:00 | 260.37 | 258.28 | 0.00 | ORB-long ORB[256.01,258.25] vol=2.0x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-06 11:25:00 | 261.82 | 258.65 | 0.00 | T1 1.5R @ 261.82 |
| Stop hit — per-position SL triggered | 2025-06-06 12:10:00 | 260.37 | 259.74 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-10 11:10:00 | 254.06 | 254.95 | 0.00 | ORB-short ORB[254.21,256.80] vol=2.3x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-10 12:25:00 | 253.24 | 254.62 | 0.00 | T1 1.5R @ 253.24 |
| Stop hit — per-position SL triggered | 2025-06-10 13:50:00 | 254.06 | 254.29 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 11:10:00 | 254.05 | 254.50 | 0.00 | ORB-short ORB[254.08,257.10] vol=2.0x ATR=0.58 |
| Stop hit — per-position SL triggered | 2025-06-12 11:25:00 | 254.63 | 254.49 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-18 09:50:00 | 248.71 | 248.29 | 0.00 | ORB-long ORB[247.16,248.61] vol=1.5x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-18 10:05:00 | 249.66 | 248.41 | 0.00 | T1 1.5R @ 249.66 |
| Stop hit — per-position SL triggered | 2025-06-18 10:20:00 | 248.71 | 248.55 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 10:05:00 | 250.75 | 248.91 | 0.00 | ORB-long ORB[247.21,249.86] vol=1.7x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:10:00 | 251.85 | 249.42 | 0.00 | T1 1.5R @ 251.85 |
| Stop hit — per-position SL triggered | 2025-06-19 10:35:00 | 250.75 | 250.99 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 09:40:00 | 252.77 | 251.58 | 0.00 | ORB-long ORB[249.01,252.20] vol=4.7x ATR=0.96 |
| Stop hit — per-position SL triggered | 2025-06-20 09:55:00 | 251.81 | 251.80 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-23 10:45:00 | 255.10 | 252.38 | 0.00 | ORB-long ORB[250.05,252.53] vol=1.8x ATR=0.95 |
| Stop hit — per-position SL triggered | 2025-06-23 11:35:00 | 254.15 | 252.84 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-25 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 10:20:00 | 258.55 | 257.27 | 0.00 | ORB-long ORB[255.10,257.52] vol=2.1x ATR=0.78 |
| Stop hit — per-position SL triggered | 2025-06-25 10:55:00 | 257.77 | 257.46 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 10:00:00 | 262.57 | 261.32 | 0.00 | ORB-long ORB[258.95,261.99] vol=2.9x ATR=0.79 |
| Stop hit — per-position SL triggered | 2025-06-26 11:05:00 | 261.78 | 262.02 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-07 10:20:00 | 258.20 | 258.92 | 0.00 | ORB-short ORB[258.40,261.55] vol=3.1x ATR=0.70 |
| Stop hit — per-position SL triggered | 2025-07-07 10:45:00 | 258.90 | 258.88 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 10:15:00 | 264.95 | 264.05 | 0.00 | ORB-long ORB[261.80,264.70] vol=2.2x ATR=0.48 |
| Stop hit — per-position SL triggered | 2025-07-09 11:00:00 | 264.47 | 264.21 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 11:10:00 | 263.50 | 262.17 | 0.00 | ORB-long ORB[260.00,262.30] vol=2.1x ATR=0.59 |
| Stop hit — per-position SL triggered | 2025-07-11 11:15:00 | 262.91 | 262.20 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 09:40:00 | 265.40 | 263.93 | 0.00 | ORB-long ORB[262.30,264.95] vol=1.5x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 10:20:00 | 266.52 | 264.74 | 0.00 | T1 1.5R @ 266.52 |
| Target hit | 2025-07-14 14:15:00 | 271.20 | 271.29 | 0.00 | Trail-exit close<VWAP |

### Cycle 19 — SELL (started 2025-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 11:15:00 | 261.65 | 263.55 | 0.00 | ORB-short ORB[263.15,266.95] vol=2.7x ATR=0.55 |
| Stop hit — per-position SL triggered | 2025-07-16 11:50:00 | 262.20 | 263.32 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-07-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 11:00:00 | 263.05 | 260.00 | 0.00 | ORB-long ORB[256.65,260.45] vol=1.5x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 11:10:00 | 264.28 | 260.51 | 0.00 | T1 1.5R @ 264.28 |
| Stop hit — per-position SL triggered | 2025-07-21 15:05:00 | 263.05 | 263.95 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-29 09:35:00 | 301.05 | 303.11 | 0.00 | ORB-short ORB[301.80,305.65] vol=1.8x ATR=0.98 |
| Stop hit — per-position SL triggered | 2025-07-29 09:50:00 | 302.03 | 302.60 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-07-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 09:35:00 | 302.50 | 304.65 | 0.00 | ORB-short ORB[303.65,306.85] vol=1.6x ATR=1.12 |
| Stop hit — per-position SL triggered | 2025-07-30 10:05:00 | 303.62 | 303.94 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-08-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 09:35:00 | 303.70 | 304.96 | 0.00 | ORB-short ORB[304.00,307.60] vol=1.7x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-05 10:15:00 | 302.57 | 304.32 | 0.00 | T1 1.5R @ 302.57 |
| Stop hit — per-position SL triggered | 2025-08-05 10:50:00 | 303.70 | 304.15 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 10:50:00 | 297.25 | 298.76 | 0.00 | ORB-short ORB[298.60,302.50] vol=2.4x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 11:00:00 | 296.21 | 298.53 | 0.00 | T1 1.5R @ 296.21 |
| Stop hit — per-position SL triggered | 2025-08-06 11:45:00 | 297.25 | 298.16 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-08-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 09:55:00 | 303.05 | 302.18 | 0.00 | ORB-long ORB[299.50,302.00] vol=5.4x ATR=0.77 |
| Stop hit — per-position SL triggered | 2025-08-11 10:00:00 | 302.28 | 302.20 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 11:15:00 | 306.30 | 306.54 | 0.00 | ORB-short ORB[306.55,310.00] vol=1.6x ATR=0.92 |
| Stop hit — per-position SL triggered | 2025-08-12 12:40:00 | 307.22 | 306.45 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-08-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 10:20:00 | 310.20 | 308.66 | 0.00 | ORB-long ORB[305.55,309.30] vol=2.2x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 11:50:00 | 311.58 | 309.52 | 0.00 | T1 1.5R @ 311.58 |
| Target hit | 2025-08-13 15:20:00 | 312.15 | 311.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — BUY (started 2025-08-14 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-14 10:10:00 | 314.50 | 312.08 | 0.00 | ORB-long ORB[310.65,313.00] vol=2.1x ATR=0.98 |
| Stop hit — per-position SL triggered | 2025-08-14 10:15:00 | 313.52 | 312.31 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-08-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 10:30:00 | 320.80 | 318.86 | 0.00 | ORB-long ORB[316.55,320.50] vol=2.8x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-25 10:45:00 | 322.04 | 319.17 | 0.00 | T1 1.5R @ 322.04 |
| Stop hit — per-position SL triggered | 2025-08-25 13:15:00 | 320.80 | 320.27 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-09-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-05 09:35:00 | 331.80 | 329.18 | 0.00 | ORB-long ORB[326.30,329.00] vol=1.8x ATR=1.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 10:45:00 | 333.84 | 331.26 | 0.00 | T1 1.5R @ 333.84 |
| Target hit | 2025-09-05 12:55:00 | 332.05 | 332.06 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — BUY (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 10:15:00 | 328.80 | 327.61 | 0.00 | ORB-long ORB[325.80,328.25] vol=1.6x ATR=1.05 |
| Stop hit — per-position SL triggered | 2025-10-24 10:35:00 | 327.75 | 327.83 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-10-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 10:00:00 | 327.25 | 326.06 | 0.00 | ORB-long ORB[325.00,326.95] vol=3.5x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 10:15:00 | 328.44 | 326.46 | 0.00 | T1 1.5R @ 328.44 |
| Target hit | 2025-10-27 15:20:00 | 333.45 | 330.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — SELL (started 2025-10-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-29 10:50:00 | 330.25 | 332.27 | 0.00 | ORB-short ORB[333.05,335.50] vol=1.8x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 11:25:00 | 329.17 | 331.26 | 0.00 | T1 1.5R @ 329.17 |
| Stop hit — per-position SL triggered | 2025-10-29 11:50:00 | 330.25 | 331.00 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 10:15:00 | 324.75 | 327.27 | 0.00 | ORB-short ORB[327.15,330.30] vol=2.2x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 10:20:00 | 323.54 | 326.93 | 0.00 | T1 1.5R @ 323.54 |
| Stop hit — per-position SL triggered | 2025-10-31 10:25:00 | 324.75 | 326.82 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-11-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 10:50:00 | 318.00 | 315.59 | 0.00 | ORB-long ORB[314.10,317.30] vol=1.7x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 11:00:00 | 319.52 | 316.22 | 0.00 | T1 1.5R @ 319.52 |
| Target hit | 2025-11-03 15:20:00 | 322.45 | 319.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — SELL (started 2025-11-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 09:35:00 | 317.20 | 318.24 | 0.00 | ORB-short ORB[318.00,321.95] vol=2.1x ATR=0.97 |
| Stop hit — per-position SL triggered | 2025-11-04 10:35:00 | 318.17 | 317.84 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-11-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 10:55:00 | 304.95 | 305.54 | 0.00 | ORB-short ORB[305.05,308.90] vol=5.0x ATR=1.14 |
| Stop hit — per-position SL triggered | 2025-11-07 11:15:00 | 306.09 | 305.48 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-11-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-19 10:05:00 | 306.40 | 305.11 | 0.00 | ORB-long ORB[302.60,306.00] vol=2.2x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 10:30:00 | 307.50 | 305.67 | 0.00 | T1 1.5R @ 307.50 |
| Stop hit — per-position SL triggered | 2025-11-19 13:25:00 | 306.40 | 306.23 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-11-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:00:00 | 303.75 | 304.93 | 0.00 | ORB-short ORB[304.15,305.95] vol=1.8x ATR=0.74 |
| Stop hit — per-position SL triggered | 2025-11-21 11:00:00 | 304.49 | 304.37 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-11-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 09:50:00 | 304.30 | 302.36 | 0.00 | ORB-long ORB[301.50,303.30] vol=1.7x ATR=0.69 |
| Stop hit — per-position SL triggered | 2025-11-28 09:55:00 | 303.61 | 302.44 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-12-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-04 09:40:00 | 294.60 | 295.39 | 0.00 | ORB-short ORB[294.75,297.60] vol=1.6x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 09:45:00 | 293.28 | 294.97 | 0.00 | T1 1.5R @ 293.28 |
| Stop hit — per-position SL triggered | 2025-12-04 10:15:00 | 294.60 | 294.15 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2026-01-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-29 10:20:00 | 270.05 | 267.29 | 0.00 | ORB-long ORB[263.05,266.90] vol=2.4x ATR=1.24 |
| Stop hit — per-position SL triggered | 2026-01-29 10:35:00 | 268.81 | 267.78 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2026-02-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:50:00 | 275.55 | 279.46 | 0.00 | ORB-short ORB[281.20,284.65] vol=2.3x ATR=0.88 |
| Stop hit — per-position SL triggered | 2026-02-18 10:55:00 | 276.43 | 279.32 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2026-02-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:50:00 | 275.70 | 277.52 | 0.00 | ORB-short ORB[276.10,279.80] vol=1.8x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 10:55:00 | 274.29 | 277.23 | 0.00 | T1 1.5R @ 274.29 |
| Stop hit — per-position SL triggered | 2026-02-19 11:00:00 | 275.70 | 277.21 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2026-02-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:45:00 | 253.05 | 254.19 | 0.00 | ORB-short ORB[253.85,256.60] vol=2.3x ATR=1.05 |
| Stop hit — per-position SL triggered | 2026-02-25 10:30:00 | 254.10 | 253.71 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2026-04-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 09:40:00 | 238.94 | 240.08 | 0.00 | ORB-short ORB[239.80,241.48] vol=5.1x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 10:05:00 | 237.55 | 239.63 | 0.00 | T1 1.5R @ 237.55 |
| Stop hit — per-position SL triggered | 2026-04-10 10:30:00 | 238.94 | 239.41 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2026-04-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:25:00 | 239.32 | 236.94 | 0.00 | ORB-long ORB[234.53,237.36] vol=2.8x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 10:45:00 | 240.78 | 237.96 | 0.00 | T1 1.5R @ 240.78 |
| Stop hit — per-position SL triggered | 2026-04-13 11:30:00 | 239.32 | 238.62 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2026-04-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:00:00 | 243.79 | 242.32 | 0.00 | ORB-long ORB[240.05,243.66] vol=2.9x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 10:10:00 | 245.25 | 242.90 | 0.00 | T1 1.5R @ 245.25 |
| Target hit | 2026-04-15 15:20:00 | 246.15 | 245.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — BUY (started 2026-04-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:40:00 | 258.13 | 256.47 | 0.00 | ORB-long ORB[254.13,257.21] vol=1.6x ATR=0.90 |
| Stop hit — per-position SL triggered | 2026-04-21 10:50:00 | 257.23 | 257.54 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2026-04-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-27 09:30:00 | 256.50 | 258.43 | 0.00 | ORB-short ORB[257.50,261.27] vol=1.6x ATR=1.01 |
| Stop hit — per-position SL triggered | 2026-04-27 09:35:00 | 257.51 | 258.29 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2026-05-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:05:00 | 248.39 | 246.62 | 0.00 | ORB-long ORB[245.00,247.40] vol=1.7x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 10:45:00 | 249.79 | 247.18 | 0.00 | T1 1.5R @ 249.79 |
| Stop hit — per-position SL triggered | 2026-05-04 13:25:00 | 248.39 | 248.82 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2026-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:00:00 | 247.19 | 250.13 | 0.00 | ORB-short ORB[249.25,252.49] vol=3.7x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:05:00 | 245.94 | 249.41 | 0.00 | T1 1.5R @ 245.94 |
| Stop hit — per-position SL triggered | 2026-05-05 11:10:00 | 247.19 | 249.33 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-15 10:40:00 | 237.77 | 2025-05-15 11:05:00 | 238.90 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-05-15 10:40:00 | 237.77 | 2025-05-15 11:10:00 | 237.77 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-20 10:20:00 | 233.73 | 2025-05-20 11:05:00 | 232.32 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2025-05-20 10:20:00 | 233.73 | 2025-05-20 15:20:00 | 228.11 | TARGET_HIT | 0.50 | 2.40% |
| BUY | retest1 | 2025-05-21 09:50:00 | 228.25 | 2025-05-21 09:55:00 | 227.16 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-05-29 09:30:00 | 226.65 | 2025-05-29 09:35:00 | 227.55 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-05-29 09:30:00 | 226.65 | 2025-05-29 09:40:00 | 226.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-03 09:35:00 | 241.70 | 2025-06-03 09:45:00 | 242.76 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-06-06 11:00:00 | 260.37 | 2025-06-06 11:25:00 | 261.82 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-06-06 11:00:00 | 260.37 | 2025-06-06 12:10:00 | 260.37 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-10 11:10:00 | 254.06 | 2025-06-10 12:25:00 | 253.24 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-06-10 11:10:00 | 254.06 | 2025-06-10 13:50:00 | 254.06 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-12 11:10:00 | 254.05 | 2025-06-12 11:25:00 | 254.63 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-06-18 09:50:00 | 248.71 | 2025-06-18 10:05:00 | 249.66 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-06-18 09:50:00 | 248.71 | 2025-06-18 10:20:00 | 248.71 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-19 10:05:00 | 250.75 | 2025-06-19 10:10:00 | 251.85 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-06-19 10:05:00 | 250.75 | 2025-06-19 10:35:00 | 250.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-20 09:40:00 | 252.77 | 2025-06-20 09:55:00 | 251.81 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-06-23 10:45:00 | 255.10 | 2025-06-23 11:35:00 | 254.15 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-06-25 10:20:00 | 258.55 | 2025-06-25 10:55:00 | 257.77 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-06-26 10:00:00 | 262.57 | 2025-06-26 11:05:00 | 261.78 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-07-07 10:20:00 | 258.20 | 2025-07-07 10:45:00 | 258.90 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-07-09 10:15:00 | 264.95 | 2025-07-09 11:00:00 | 264.47 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-07-11 11:10:00 | 263.50 | 2025-07-11 11:15:00 | 262.91 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-07-14 09:40:00 | 265.40 | 2025-07-14 10:20:00 | 266.52 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-07-14 09:40:00 | 265.40 | 2025-07-14 14:15:00 | 271.20 | TARGET_HIT | 0.50 | 2.19% |
| SELL | retest1 | 2025-07-16 11:15:00 | 261.65 | 2025-07-16 11:50:00 | 262.20 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-07-21 11:00:00 | 263.05 | 2025-07-21 11:10:00 | 264.28 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-07-21 11:00:00 | 263.05 | 2025-07-21 15:05:00 | 263.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-29 09:35:00 | 301.05 | 2025-07-29 09:50:00 | 302.03 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-07-30 09:35:00 | 302.50 | 2025-07-30 10:05:00 | 303.62 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-08-05 09:35:00 | 303.70 | 2025-08-05 10:15:00 | 302.57 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-08-05 09:35:00 | 303.70 | 2025-08-05 10:50:00 | 303.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-06 10:50:00 | 297.25 | 2025-08-06 11:00:00 | 296.21 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-08-06 10:50:00 | 297.25 | 2025-08-06 11:45:00 | 297.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-11 09:55:00 | 303.05 | 2025-08-11 10:00:00 | 302.28 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-08-12 11:15:00 | 306.30 | 2025-08-12 12:40:00 | 307.22 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-08-13 10:20:00 | 310.20 | 2025-08-13 11:50:00 | 311.58 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-08-13 10:20:00 | 310.20 | 2025-08-13 15:20:00 | 312.15 | TARGET_HIT | 0.50 | 0.63% |
| BUY | retest1 | 2025-08-14 10:10:00 | 314.50 | 2025-08-14 10:15:00 | 313.52 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-08-25 10:30:00 | 320.80 | 2025-08-25 10:45:00 | 322.04 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-08-25 10:30:00 | 320.80 | 2025-08-25 13:15:00 | 320.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-05 09:35:00 | 331.80 | 2025-09-05 10:45:00 | 333.84 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-09-05 09:35:00 | 331.80 | 2025-09-05 12:55:00 | 332.05 | TARGET_HIT | 0.50 | 0.08% |
| BUY | retest1 | 2025-10-24 10:15:00 | 328.80 | 2025-10-24 10:35:00 | 327.75 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-10-27 10:00:00 | 327.25 | 2025-10-27 10:15:00 | 328.44 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-10-27 10:00:00 | 327.25 | 2025-10-27 15:20:00 | 333.45 | TARGET_HIT | 0.50 | 1.89% |
| SELL | retest1 | 2025-10-29 10:50:00 | 330.25 | 2025-10-29 11:25:00 | 329.17 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-10-29 10:50:00 | 330.25 | 2025-10-29 11:50:00 | 330.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-31 10:15:00 | 324.75 | 2025-10-31 10:20:00 | 323.54 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-10-31 10:15:00 | 324.75 | 2025-10-31 10:25:00 | 324.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-03 10:50:00 | 318.00 | 2025-11-03 11:00:00 | 319.52 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-11-03 10:50:00 | 318.00 | 2025-11-03 15:20:00 | 322.45 | TARGET_HIT | 0.50 | 1.40% |
| SELL | retest1 | 2025-11-04 09:35:00 | 317.20 | 2025-11-04 10:35:00 | 318.17 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-11-07 10:55:00 | 304.95 | 2025-11-07 11:15:00 | 306.09 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-11-19 10:05:00 | 306.40 | 2025-11-19 10:30:00 | 307.50 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-11-19 10:05:00 | 306.40 | 2025-11-19 13:25:00 | 306.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-21 10:00:00 | 303.75 | 2025-11-21 11:00:00 | 304.49 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-11-28 09:50:00 | 304.30 | 2025-11-28 09:55:00 | 303.61 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-12-04 09:40:00 | 294.60 | 2025-12-04 09:45:00 | 293.28 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-12-04 09:40:00 | 294.60 | 2025-12-04 10:15:00 | 294.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-29 10:20:00 | 270.05 | 2026-01-29 10:35:00 | 268.81 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2026-02-18 10:50:00 | 275.55 | 2026-02-18 10:55:00 | 276.43 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-02-19 10:50:00 | 275.70 | 2026-02-19 10:55:00 | 274.29 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-02-19 10:50:00 | 275.70 | 2026-02-19 11:00:00 | 275.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-25 09:45:00 | 253.05 | 2026-02-25 10:30:00 | 254.10 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-04-10 09:40:00 | 238.94 | 2026-04-10 10:05:00 | 237.55 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-04-10 09:40:00 | 238.94 | 2026-04-10 10:30:00 | 238.94 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-13 10:25:00 | 239.32 | 2026-04-13 10:45:00 | 240.78 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-04-13 10:25:00 | 239.32 | 2026-04-13 11:30:00 | 239.32 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-15 10:00:00 | 243.79 | 2026-04-15 10:10:00 | 245.25 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-04-15 10:00:00 | 243.79 | 2026-04-15 15:20:00 | 246.15 | TARGET_HIT | 0.50 | 0.97% |
| BUY | retest1 | 2026-04-21 09:40:00 | 258.13 | 2026-04-21 10:50:00 | 257.23 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-27 09:30:00 | 256.50 | 2026-04-27 09:35:00 | 257.51 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-05-04 10:05:00 | 248.39 | 2026-05-04 10:45:00 | 249.79 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-05-04 10:05:00 | 248.39 | 2026-05-04 13:25:00 | 248.39 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-05-05 11:00:00 | 247.19 | 2026-05-05 11:05:00 | 245.94 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2026-05-05 11:00:00 | 247.19 | 2026-05-05 11:10:00 | 247.19 | STOP_HIT | 0.50 | 0.00% |
