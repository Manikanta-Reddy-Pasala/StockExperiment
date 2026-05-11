# Aegis Vopak Terminals Ltd. (AEGISVOPAK)

## Backtest Summary

- **Window:** 2025-06-02 09:40:00 → 2026-05-05 15:25:00 (17108 bars)
- **Last close:** 195.50
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
| ENTRY1 | 46 |
| ENTRY2 | 0 |
| PARTIAL | 19 |
| TARGET_HIT | 12 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 65 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 31 / 34
- **Target hits / Stop hits / Partials:** 12 / 34 / 19
- **Avg / median % per leg:** 0.25% / 0.00%
- **Sum % (uncompounded):** 16.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 16 | 53.3% | 6 | 14 | 10 | 0.18% | 5.5% |
| BUY @ 2nd Alert (retest1) | 30 | 16 | 53.3% | 6 | 14 | 10 | 0.18% | 5.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 35 | 15 | 42.9% | 6 | 20 | 9 | 0.31% | 10.8% |
| SELL @ 2nd Alert (retest1) | 35 | 15 | 42.9% | 6 | 20 | 9 | 0.31% | 10.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 65 | 31 | 47.7% | 12 | 34 | 19 | 0.25% | 16.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-11 10:30:00 | 244.00 | 249.85 | 0.00 | ORB-short ORB[252.01,255.14] vol=1.8x ATR=1.33 |
| Stop hit — per-position SL triggered | 2025-06-11 10:35:00 | 245.33 | 249.29 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-07-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 10:00:00 | 246.83 | 245.09 | 0.00 | ORB-long ORB[243.05,245.82] vol=4.6x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 10:05:00 | 248.40 | 245.44 | 0.00 | T1 1.5R @ 248.40 |
| Stop hit — per-position SL triggered | 2025-07-03 12:10:00 | 246.83 | 247.20 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-07-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:00:00 | 263.51 | 264.59 | 0.00 | ORB-short ORB[264.30,268.00] vol=2.8x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:15:00 | 261.83 | 264.18 | 0.00 | T1 1.5R @ 261.83 |
| Target hit | 2025-07-25 15:20:00 | 255.68 | 259.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2025-07-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-31 11:00:00 | 242.36 | 243.27 | 0.00 | ORB-short ORB[243.00,245.50] vol=3.2x ATR=1.04 |
| Stop hit — per-position SL triggered | 2025-07-31 11:35:00 | 243.40 | 243.25 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-08-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 10:55:00 | 243.99 | 242.59 | 0.00 | ORB-long ORB[241.00,243.12] vol=2.5x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 12:15:00 | 245.04 | 243.17 | 0.00 | T1 1.5R @ 245.04 |
| Stop hit — per-position SL triggered | 2025-08-18 15:10:00 | 243.99 | 244.50 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-08-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-20 10:50:00 | 247.00 | 248.93 | 0.00 | ORB-short ORB[248.25,251.59] vol=1.5x ATR=0.82 |
| Stop hit — per-position SL triggered | 2025-08-20 11:20:00 | 247.82 | 248.79 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-09-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 09:40:00 | 239.83 | 240.32 | 0.00 | ORB-short ORB[241.50,243.64] vol=2.4x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 13:50:00 | 238.34 | 239.49 | 0.00 | T1 1.5R @ 238.34 |
| Target hit | 2025-09-05 15:20:00 | 237.09 | 238.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2025-09-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-09 11:05:00 | 242.31 | 243.99 | 0.00 | ORB-short ORB[243.05,246.50] vol=4.1x ATR=0.59 |
| Stop hit — per-position SL triggered | 2025-09-09 12:05:00 | 242.90 | 243.69 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-09-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 09:40:00 | 251.54 | 250.19 | 0.00 | ORB-long ORB[248.21,250.91] vol=2.0x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 09:50:00 | 252.89 | 251.56 | 0.00 | T1 1.5R @ 252.89 |
| Stop hit — per-position SL triggered | 2025-09-12 09:55:00 | 251.54 | 251.48 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-09-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 09:45:00 | 248.64 | 246.68 | 0.00 | ORB-long ORB[244.66,247.99] vol=2.0x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-15 09:55:00 | 249.94 | 247.13 | 0.00 | T1 1.5R @ 249.94 |
| Stop hit — per-position SL triggered | 2025-09-15 11:55:00 | 248.64 | 248.60 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-09-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 11:10:00 | 245.85 | 248.62 | 0.00 | ORB-short ORB[248.42,250.51] vol=3.8x ATR=0.72 |
| Stop hit — per-position SL triggered | 2025-09-17 11:15:00 | 246.57 | 248.49 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-09-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 10:05:00 | 247.72 | 248.30 | 0.00 | ORB-short ORB[248.97,251.51] vol=5.0x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 10:45:00 | 246.32 | 248.17 | 0.00 | T1 1.5R @ 246.32 |
| Stop hit — per-position SL triggered | 2025-09-23 14:50:00 | 247.72 | 246.68 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-09-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-30 10:55:00 | 254.87 | 252.64 | 0.00 | ORB-long ORB[250.99,253.98] vol=4.5x ATR=1.82 |
| Stop hit — per-position SL triggered | 2025-09-30 11:20:00 | 253.05 | 252.84 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 10:15:00 | 274.45 | 276.54 | 0.00 | ORB-short ORB[276.50,279.55] vol=1.8x ATR=1.28 |
| Stop hit — per-position SL triggered | 2025-10-07 11:45:00 | 275.73 | 275.76 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 11:15:00 | 278.55 | 276.44 | 0.00 | ORB-long ORB[275.10,278.35] vol=4.2x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 11:40:00 | 279.95 | 277.80 | 0.00 | T1 1.5R @ 279.95 |
| Target hit | 2025-10-15 11:55:00 | 280.15 | 280.52 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — BUY (started 2025-10-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 09:45:00 | 287.25 | 284.75 | 0.00 | ORB-long ORB[280.45,283.45] vol=2.8x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 10:05:00 | 290.07 | 286.45 | 0.00 | T1 1.5R @ 290.07 |
| Target hit | 2025-10-16 11:00:00 | 288.65 | 289.25 | 0.00 | Trail-exit close<VWAP |

### Cycle 17 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 11:15:00 | 283.70 | 281.65 | 0.00 | ORB-long ORB[278.85,282.95] vol=1.7x ATR=1.02 |
| Stop hit — per-position SL triggered | 2025-10-20 11:40:00 | 282.68 | 281.88 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-10-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-27 09:45:00 | 275.30 | 277.53 | 0.00 | ORB-short ORB[279.00,282.00] vol=2.2x ATR=1.08 |
| Stop hit — per-position SL triggered | 2025-10-27 10:10:00 | 276.38 | 276.80 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-10-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 10:45:00 | 275.95 | 278.28 | 0.00 | ORB-short ORB[279.15,282.80] vol=4.0x ATR=1.04 |
| Stop hit — per-position SL triggered | 2025-10-28 11:15:00 | 276.99 | 277.03 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-10-31 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 10:35:00 | 276.70 | 276.01 | 0.00 | ORB-long ORB[273.25,275.30] vol=5.2x ATR=0.92 |
| Stop hit — per-position SL triggered | 2025-10-31 11:05:00 | 275.78 | 276.02 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-11-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 10:45:00 | 269.90 | 268.79 | 0.00 | ORB-long ORB[268.00,269.60] vol=2.5x ATR=0.71 |
| Stop hit — per-position SL triggered | 2025-11-17 11:05:00 | 269.19 | 268.95 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-11-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 10:55:00 | 263.65 | 265.04 | 0.00 | ORB-short ORB[264.50,267.00] vol=1.9x ATR=0.61 |
| Stop hit — per-position SL triggered | 2025-11-18 11:25:00 | 264.26 | 264.95 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-11-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-19 09:50:00 | 262.50 | 260.97 | 0.00 | ORB-long ORB[259.35,262.40] vol=1.8x ATR=1.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 11:35:00 | 264.13 | 262.39 | 0.00 | T1 1.5R @ 264.13 |
| Target hit | 2025-11-19 15:15:00 | 263.85 | 263.96 | 0.00 | Trail-exit close<VWAP |

### Cycle 24 — SELL (started 2025-11-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 11:10:00 | 259.05 | 259.76 | 0.00 | ORB-short ORB[259.50,261.70] vol=2.1x ATR=0.54 |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 259.59 | 259.75 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-11-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 09:35:00 | 259.40 | 257.07 | 0.00 | ORB-long ORB[254.80,256.50] vol=4.6x ATR=1.04 |
| Stop hit — per-position SL triggered | 2025-11-26 09:40:00 | 258.36 | 257.50 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-11-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 09:30:00 | 262.55 | 261.79 | 0.00 | ORB-long ORB[259.10,262.05] vol=5.9x ATR=1.29 |
| Stop hit — per-position SL triggered | 2025-11-27 09:40:00 | 261.26 | 261.78 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-11-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-28 11:05:00 | 259.80 | 261.46 | 0.00 | ORB-short ORB[261.85,263.40] vol=5.5x ATR=1.10 |
| Stop hit — per-position SL triggered | 2025-11-28 11:30:00 | 260.90 | 260.91 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-12-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 09:30:00 | 258.50 | 259.71 | 0.00 | ORB-short ORB[259.75,261.30] vol=3.5x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 09:35:00 | 257.36 | 258.80 | 0.00 | T1 1.5R @ 257.36 |
| Target hit | 2025-12-08 15:20:00 | 243.50 | 246.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — BUY (started 2025-12-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 11:05:00 | 244.00 | 242.77 | 0.00 | ORB-long ORB[240.85,243.70] vol=3.8x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 12:00:00 | 244.99 | 243.12 | 0.00 | T1 1.5R @ 244.99 |
| Target hit | 2025-12-11 15:20:00 | 246.65 | 245.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — BUY (started 2025-12-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:40:00 | 250.00 | 248.16 | 0.00 | ORB-long ORB[245.90,248.90] vol=1.6x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-12 09:50:00 | 251.46 | 251.34 | 0.00 | T1 1.5R @ 251.46 |
| Target hit | 2025-12-12 10:30:00 | 252.25 | 252.30 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — SELL (started 2025-12-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 11:10:00 | 245.55 | 247.10 | 0.00 | ORB-short ORB[246.40,250.00] vol=2.2x ATR=0.63 |
| Stop hit — per-position SL triggered | 2025-12-16 12:30:00 | 246.18 | 246.90 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-12-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-22 11:05:00 | 257.70 | 259.92 | 0.00 | ORB-short ORB[260.25,264.00] vol=5.6x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-22 15:05:00 | 256.32 | 258.13 | 0.00 | T1 1.5R @ 256.32 |
| Target hit | 2025-12-22 15:20:00 | 255.80 | 257.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — BUY (started 2026-01-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 11:00:00 | 250.26 | 248.78 | 0.00 | ORB-long ORB[247.73,250.00] vol=12.0x ATR=0.78 |
| Stop hit — per-position SL triggered | 2026-01-01 11:10:00 | 249.48 | 248.83 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2026-01-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-05 10:50:00 | 248.43 | 249.25 | 0.00 | ORB-short ORB[248.80,251.31] vol=2.0x ATR=0.65 |
| Stop hit — per-position SL triggered | 2026-01-05 11:35:00 | 249.08 | 249.10 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2026-01-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 11:05:00 | 240.99 | 241.88 | 0.00 | ORB-short ORB[242.12,244.06] vol=1.5x ATR=0.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 11:55:00 | 240.10 | 241.57 | 0.00 | T1 1.5R @ 240.10 |
| Stop hit — per-position SL triggered | 2026-01-06 12:00:00 | 240.99 | 241.52 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2026-01-07 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-07 10:35:00 | 238.35 | 239.65 | 0.00 | ORB-short ORB[238.40,240.79] vol=2.3x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 11:25:00 | 237.22 | 239.09 | 0.00 | T1 1.5R @ 237.22 |
| Target hit | 2026-01-07 15:20:00 | 237.35 | 238.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — BUY (started 2026-01-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-08 10:35:00 | 241.45 | 239.99 | 0.00 | ORB-long ORB[236.99,238.46] vol=2.8x ATR=0.81 |
| Stop hit — per-position SL triggered | 2026-01-08 11:00:00 | 240.64 | 240.35 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2026-01-14 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 10:20:00 | 226.99 | 229.41 | 0.00 | ORB-short ORB[229.84,232.80] vol=1.7x ATR=0.90 |
| Stop hit — per-position SL triggered | 2026-01-14 10:40:00 | 227.89 | 228.84 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2026-01-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-19 09:30:00 | 211.91 | 212.13 | 0.00 | ORB-short ORB[215.52,218.00] vol=7.4x ATR=1.69 |
| Stop hit — per-position SL triggered | 2026-01-19 10:10:00 | 213.60 | 212.06 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2026-02-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:50:00 | 218.25 | 218.96 | 0.00 | ORB-short ORB[219.40,222.13] vol=3.6x ATR=0.77 |
| Stop hit — per-position SL triggered | 2026-02-11 10:10:00 | 219.02 | 218.90 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2026-02-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:55:00 | 216.00 | 217.20 | 0.00 | ORB-short ORB[216.75,218.95] vol=2.5x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 11:15:00 | 215.19 | 216.77 | 0.00 | T1 1.5R @ 215.19 |
| Target hit | 2026-02-12 15:20:00 | 212.77 | 214.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — SELL (started 2026-02-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:35:00 | 207.80 | 209.35 | 0.00 | ORB-short ORB[209.00,212.00] vol=1.5x ATR=0.89 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 208.69 | 209.23 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2026-02-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 10:45:00 | 208.33 | 209.68 | 0.00 | ORB-short ORB[209.61,212.52] vol=1.7x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 10:55:00 | 206.95 | 209.26 | 0.00 | T1 1.5R @ 206.95 |
| Stop hit — per-position SL triggered | 2026-02-16 14:45:00 | 208.33 | 207.63 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2026-02-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:35:00 | 232.64 | 229.04 | 0.00 | ORB-long ORB[224.52,227.95] vol=3.0x ATR=1.43 |
| Stop hit — per-position SL triggered | 2026-02-25 09:45:00 | 231.21 | 230.04 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 11:15:00 | 191.13 | 190.71 | 0.00 | ORB-long ORB[189.25,190.95] vol=2.7x ATR=0.73 |
| Stop hit — per-position SL triggered | 2026-03-17 11:25:00 | 190.40 | 190.74 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2026-04-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:45:00 | 197.96 | 196.80 | 0.00 | ORB-long ORB[195.45,197.79] vol=2.1x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:15:00 | 199.26 | 197.19 | 0.00 | T1 1.5R @ 199.26 |
| Target hit | 2026-04-28 12:25:00 | 198.58 | 198.98 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-06-11 10:30:00 | 244.00 | 2025-06-11 10:35:00 | 245.33 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2025-07-03 10:00:00 | 246.83 | 2025-07-03 10:05:00 | 248.40 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-07-03 10:00:00 | 246.83 | 2025-07-03 12:10:00 | 246.83 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-25 10:00:00 | 263.51 | 2025-07-25 10:15:00 | 261.83 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2025-07-25 10:00:00 | 263.51 | 2025-07-25 15:20:00 | 255.68 | TARGET_HIT | 0.50 | 2.97% |
| SELL | retest1 | 2025-07-31 11:00:00 | 242.36 | 2025-07-31 11:35:00 | 243.40 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-08-18 10:55:00 | 243.99 | 2025-08-18 12:15:00 | 245.04 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-08-18 10:55:00 | 243.99 | 2025-08-18 15:10:00 | 243.99 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-20 10:50:00 | 247.00 | 2025-08-20 11:20:00 | 247.82 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-09-05 09:40:00 | 239.83 | 2025-09-05 13:50:00 | 238.34 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-09-05 09:40:00 | 239.83 | 2025-09-05 15:20:00 | 237.09 | TARGET_HIT | 0.50 | 1.14% |
| SELL | retest1 | 2025-09-09 11:05:00 | 242.31 | 2025-09-09 12:05:00 | 242.90 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-09-12 09:40:00 | 251.54 | 2025-09-12 09:50:00 | 252.89 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-09-12 09:40:00 | 251.54 | 2025-09-12 09:55:00 | 251.54 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-15 09:45:00 | 248.64 | 2025-09-15 09:55:00 | 249.94 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-09-15 09:45:00 | 248.64 | 2025-09-15 11:55:00 | 248.64 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-17 11:10:00 | 245.85 | 2025-09-17 11:15:00 | 246.57 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-09-23 10:05:00 | 247.72 | 2025-09-23 10:45:00 | 246.32 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-09-23 10:05:00 | 247.72 | 2025-09-23 14:50:00 | 247.72 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-30 10:55:00 | 254.87 | 2025-09-30 11:20:00 | 253.05 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest1 | 2025-10-07 10:15:00 | 274.45 | 2025-10-07 11:45:00 | 275.73 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-10-15 11:15:00 | 278.55 | 2025-10-15 11:40:00 | 279.95 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-10-15 11:15:00 | 278.55 | 2025-10-15 11:55:00 | 280.15 | TARGET_HIT | 0.50 | 0.57% |
| BUY | retest1 | 2025-10-16 09:45:00 | 287.25 | 2025-10-16 10:05:00 | 290.07 | PARTIAL | 0.50 | 0.98% |
| BUY | retest1 | 2025-10-16 09:45:00 | 287.25 | 2025-10-16 11:00:00 | 288.65 | TARGET_HIT | 0.50 | 0.49% |
| BUY | retest1 | 2025-10-20 11:15:00 | 283.70 | 2025-10-20 11:40:00 | 282.68 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-10-27 09:45:00 | 275.30 | 2025-10-27 10:10:00 | 276.38 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-10-28 10:45:00 | 275.95 | 2025-10-28 11:15:00 | 276.99 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-10-31 10:35:00 | 276.70 | 2025-10-31 11:05:00 | 275.78 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-11-17 10:45:00 | 269.90 | 2025-11-17 11:05:00 | 269.19 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-11-18 10:55:00 | 263.65 | 2025-11-18 11:25:00 | 264.26 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-11-19 09:50:00 | 262.50 | 2025-11-19 11:35:00 | 264.13 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-11-19 09:50:00 | 262.50 | 2025-11-19 15:15:00 | 263.85 | TARGET_HIT | 0.50 | 0.51% |
| SELL | retest1 | 2025-11-20 11:10:00 | 259.05 | 2025-11-20 11:15:00 | 259.59 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-11-26 09:35:00 | 259.40 | 2025-11-26 09:40:00 | 258.36 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-11-27 09:30:00 | 262.55 | 2025-11-27 09:40:00 | 261.26 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2025-11-28 11:05:00 | 259.80 | 2025-11-28 11:30:00 | 260.90 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-12-08 09:30:00 | 258.50 | 2025-12-08 09:35:00 | 257.36 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-12-08 09:30:00 | 258.50 | 2025-12-08 15:20:00 | 243.50 | TARGET_HIT | 0.50 | 5.80% |
| BUY | retest1 | 2025-12-11 11:05:00 | 244.00 | 2025-12-11 12:00:00 | 244.99 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-12-11 11:05:00 | 244.00 | 2025-12-11 15:20:00 | 246.65 | TARGET_HIT | 0.50 | 1.09% |
| BUY | retest1 | 2025-12-12 09:40:00 | 250.00 | 2025-12-12 09:50:00 | 251.46 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-12-12 09:40:00 | 250.00 | 2025-12-12 10:30:00 | 252.25 | TARGET_HIT | 0.50 | 0.90% |
| SELL | retest1 | 2025-12-16 11:10:00 | 245.55 | 2025-12-16 12:30:00 | 246.18 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-12-22 11:05:00 | 257.70 | 2025-12-22 15:05:00 | 256.32 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-12-22 11:05:00 | 257.70 | 2025-12-22 15:20:00 | 255.80 | TARGET_HIT | 0.50 | 0.74% |
| BUY | retest1 | 2026-01-01 11:00:00 | 250.26 | 2026-01-01 11:10:00 | 249.48 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-01-05 10:50:00 | 248.43 | 2026-01-05 11:35:00 | 249.08 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-01-06 11:05:00 | 240.99 | 2026-01-06 11:55:00 | 240.10 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-01-06 11:05:00 | 240.99 | 2026-01-06 12:00:00 | 240.99 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-07 10:35:00 | 238.35 | 2026-01-07 11:25:00 | 237.22 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-01-07 10:35:00 | 238.35 | 2026-01-07 15:20:00 | 237.35 | TARGET_HIT | 0.50 | 0.42% |
| BUY | retest1 | 2026-01-08 10:35:00 | 241.45 | 2026-01-08 11:00:00 | 240.64 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-01-14 10:20:00 | 226.99 | 2026-01-14 10:40:00 | 227.89 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-01-19 09:30:00 | 211.91 | 2026-01-19 10:10:00 | 213.60 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest1 | 2026-02-11 09:50:00 | 218.25 | 2026-02-11 10:10:00 | 219.02 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-12 10:55:00 | 216.00 | 2026-02-12 11:15:00 | 215.19 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-02-12 10:55:00 | 216.00 | 2026-02-12 15:20:00 | 212.77 | TARGET_HIT | 0.50 | 1.50% |
| SELL | retest1 | 2026-02-13 09:35:00 | 207.80 | 2026-02-13 09:40:00 | 208.69 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-02-16 10:45:00 | 208.33 | 2026-02-16 10:55:00 | 206.95 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2026-02-16 10:45:00 | 208.33 | 2026-02-16 14:45:00 | 208.33 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 09:35:00 | 232.64 | 2026-02-25 09:45:00 | 231.21 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest1 | 2026-03-17 11:15:00 | 191.13 | 2026-03-17 11:25:00 | 190.40 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-28 10:45:00 | 197.96 | 2026-04-28 11:15:00 | 199.26 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-04-28 10:45:00 | 197.96 | 2026-04-28 12:25:00 | 198.58 | TARGET_HIT | 0.50 | 0.31% |
