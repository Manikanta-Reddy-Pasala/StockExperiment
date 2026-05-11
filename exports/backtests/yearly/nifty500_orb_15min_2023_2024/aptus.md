# Aptus Value Housing Finance India Ltd. (APTUS)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (55356 bars)
- **Last close:** 282.50
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
| ENTRY1 | 63 |
| ENTRY2 | 0 |
| PARTIAL | 33 |
| TARGET_HIT | 12 |
| STOP_HIT | 51 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 96 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 45 / 51
- **Target hits / Stop hits / Partials:** 12 / 51 / 33
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 14.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 10 | 32.3% | 2 | 21 | 8 | 0.04% | 1.2% |
| BUY @ 2nd Alert (retest1) | 31 | 10 | 32.3% | 2 | 21 | 8 | 0.04% | 1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 65 | 35 | 53.8% | 10 | 30 | 25 | 0.21% | 13.5% |
| SELL @ 2nd Alert (retest1) | 65 | 35 | 53.8% | 10 | 30 | 25 | 0.21% | 13.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 96 | 45 | 46.9% | 12 | 51 | 33 | 0.15% | 14.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 09:35:00 | 264.50 | 265.45 | 0.00 | ORB-short ORB[264.55,267.50] vol=1.9x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-19 12:40:00 | 263.52 | 264.84 | 0.00 | T1 1.5R @ 263.52 |
| Stop hit — per-position SL triggered | 2023-05-19 13:15:00 | 264.50 | 264.81 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-22 11:00:00 | 267.20 | 265.27 | 0.00 | ORB-long ORB[262.55,265.65] vol=7.1x ATR=0.82 |
| Stop hit — per-position SL triggered | 2023-05-22 11:05:00 | 266.38 | 265.37 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-24 09:30:00 | 260.60 | 262.26 | 0.00 | ORB-short ORB[261.30,264.35] vol=1.7x ATR=1.00 |
| Stop hit — per-position SL triggered | 2023-05-24 09:40:00 | 261.60 | 262.07 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-06-07 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-07 10:25:00 | 278.10 | 276.49 | 0.00 | ORB-long ORB[275.35,277.35] vol=3.2x ATR=0.74 |
| Stop hit — per-position SL triggered | 2023-06-07 10:30:00 | 277.36 | 276.57 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2023-06-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-14 09:40:00 | 273.25 | 273.86 | 0.00 | ORB-short ORB[273.30,276.70] vol=2.4x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-14 10:05:00 | 272.04 | 273.73 | 0.00 | T1 1.5R @ 272.04 |
| Target hit | 2023-06-14 11:00:00 | 272.90 | 272.74 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — SELL (started 2023-06-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-20 09:50:00 | 266.20 | 266.89 | 0.00 | ORB-short ORB[266.30,269.65] vol=2.0x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-20 13:15:00 | 264.92 | 266.31 | 0.00 | T1 1.5R @ 264.92 |
| Target hit | 2023-06-20 15:20:00 | 266.00 | 266.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2023-06-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-22 10:40:00 | 246.00 | 244.83 | 0.00 | ORB-long ORB[242.65,245.30] vol=2.1x ATR=0.99 |
| Stop hit — per-position SL triggered | 2023-06-22 12:05:00 | 245.01 | 245.01 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-06-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-26 11:00:00 | 245.70 | 243.49 | 0.00 | ORB-long ORB[242.00,245.25] vol=4.3x ATR=0.84 |
| Stop hit — per-position SL triggered | 2023-06-26 12:05:00 | 244.86 | 243.98 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-07-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-04 09:40:00 | 246.85 | 247.85 | 0.00 | ORB-short ORB[247.10,250.00] vol=1.8x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-04 09:55:00 | 245.88 | 247.38 | 0.00 | T1 1.5R @ 245.88 |
| Stop hit — per-position SL triggered | 2023-07-04 10:15:00 | 246.85 | 246.90 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-07-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 09:40:00 | 252.90 | 249.98 | 0.00 | ORB-long ORB[246.85,248.90] vol=6.5x ATR=1.16 |
| Stop hit — per-position SL triggered | 2023-07-06 09:55:00 | 251.74 | 250.98 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-07-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-07 09:35:00 | 250.70 | 252.56 | 0.00 | ORB-short ORB[251.65,254.20] vol=2.1x ATR=1.09 |
| Stop hit — per-position SL triggered | 2023-07-07 09:40:00 | 251.79 | 252.48 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-07-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 11:10:00 | 268.40 | 265.86 | 0.00 | ORB-long ORB[263.95,267.00] vol=2.5x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-14 11:15:00 | 269.91 | 266.43 | 0.00 | T1 1.5R @ 269.91 |
| Stop hit — per-position SL triggered | 2023-07-14 11:30:00 | 268.40 | 266.78 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-07-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-19 09:55:00 | 271.80 | 269.52 | 0.00 | ORB-long ORB[266.65,270.50] vol=2.6x ATR=1.25 |
| Stop hit — per-position SL triggered | 2023-07-19 10:10:00 | 270.55 | 269.67 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2023-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-26 10:15:00 | 274.15 | 276.69 | 0.00 | ORB-short ORB[275.35,277.90] vol=2.5x ATR=1.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-26 13:00:00 | 272.29 | 274.19 | 0.00 | T1 1.5R @ 272.29 |
| Target hit | 2023-07-26 15:20:00 | 271.05 | 273.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — SELL (started 2023-08-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-08 10:40:00 | 264.30 | 265.07 | 0.00 | ORB-short ORB[264.50,267.00] vol=3.5x ATR=0.95 |
| Stop hit — per-position SL triggered | 2023-08-08 11:30:00 | 265.25 | 264.83 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-08-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-11 10:40:00 | 269.00 | 266.01 | 0.00 | ORB-long ORB[264.35,266.85] vol=4.1x ATR=0.95 |
| Stop hit — per-position SL triggered | 2023-08-11 10:45:00 | 268.05 | 266.70 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-08-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-17 10:30:00 | 275.20 | 273.77 | 0.00 | ORB-long ORB[271.45,274.70] vol=2.3x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-17 13:10:00 | 276.69 | 274.74 | 0.00 | T1 1.5R @ 276.69 |
| Target hit | 2023-08-17 15:20:00 | 277.15 | 275.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2023-08-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-28 09:35:00 | 267.55 | 270.51 | 0.00 | ORB-short ORB[269.75,272.45] vol=1.5x ATR=1.03 |
| Stop hit — per-position SL triggered | 2023-08-28 10:45:00 | 268.58 | 268.76 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-08-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-29 10:00:00 | 267.85 | 269.88 | 0.00 | ORB-short ORB[268.40,271.80] vol=1.6x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-29 10:45:00 | 266.32 | 268.81 | 0.00 | T1 1.5R @ 266.32 |
| Target hit | 2023-08-29 15:20:00 | 265.00 | 266.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2023-08-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-30 10:45:00 | 265.50 | 267.27 | 0.00 | ORB-short ORB[265.95,268.35] vol=1.6x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-30 11:00:00 | 264.25 | 266.82 | 0.00 | T1 1.5R @ 264.25 |
| Stop hit — per-position SL triggered | 2023-08-30 15:15:00 | 265.50 | 265.43 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-09-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 11:05:00 | 269.75 | 268.20 | 0.00 | ORB-long ORB[266.80,268.90] vol=2.9x ATR=0.86 |
| Stop hit — per-position SL triggered | 2023-09-01 11:40:00 | 268.89 | 268.30 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-09-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 09:50:00 | 271.45 | 269.82 | 0.00 | ORB-long ORB[268.40,271.10] vol=1.7x ATR=0.82 |
| Stop hit — per-position SL triggered | 2023-09-04 10:00:00 | 270.63 | 269.87 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-09-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 09:40:00 | 275.50 | 278.46 | 0.00 | ORB-short ORB[277.25,280.35] vol=1.6x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 09:45:00 | 274.21 | 277.39 | 0.00 | T1 1.5R @ 274.21 |
| Target hit | 2023-09-12 10:05:00 | 274.50 | 274.45 | 0.00 | Trail-exit close>VWAP |

### Cycle 24 — SELL (started 2023-09-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 10:55:00 | 274.95 | 275.05 | 0.00 | ORB-short ORB[275.00,278.90] vol=2.5x ATR=0.96 |
| Stop hit — per-position SL triggered | 2023-09-22 11:35:00 | 275.91 | 275.11 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-03 10:15:00 | 298.65 | 296.25 | 0.00 | ORB-long ORB[293.50,296.95] vol=3.0x ATR=1.34 |
| Stop hit — per-position SL triggered | 2023-10-03 10:35:00 | 297.31 | 296.50 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-10-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-05 10:40:00 | 293.15 | 295.31 | 0.00 | ORB-short ORB[296.00,298.05] vol=1.9x ATR=1.01 |
| Stop hit — per-position SL triggered | 2023-10-05 10:50:00 | 294.16 | 295.20 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-10-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-06 09:40:00 | 293.60 | 295.29 | 0.00 | ORB-short ORB[294.35,298.10] vol=1.8x ATR=1.29 |
| Target hit | 2023-10-06 15:20:00 | 292.65 | 293.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — BUY (started 2023-10-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-10 09:45:00 | 295.70 | 294.92 | 0.00 | ORB-long ORB[292.45,295.35] vol=1.6x ATR=0.94 |
| Stop hit — per-position SL triggered | 2023-10-10 10:30:00 | 294.76 | 295.36 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-10-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-12 10:10:00 | 293.50 | 295.16 | 0.00 | ORB-short ORB[294.15,298.10] vol=1.7x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-12 10:35:00 | 292.05 | 294.25 | 0.00 | T1 1.5R @ 292.05 |
| Stop hit — per-position SL triggered | 2023-10-12 10:40:00 | 293.50 | 294.12 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-10-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 10:40:00 | 297.50 | 294.44 | 0.00 | ORB-long ORB[290.45,293.75] vol=6.4x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-17 11:00:00 | 298.81 | 296.98 | 0.00 | T1 1.5R @ 298.81 |
| Target hit | 2023-10-17 15:20:00 | 301.50 | 300.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — SELL (started 2023-10-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 10:45:00 | 298.25 | 298.44 | 0.00 | ORB-short ORB[298.45,302.90] vol=1.6x ATR=0.99 |
| Stop hit — per-position SL triggered | 2023-10-18 10:55:00 | 299.24 | 298.49 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-10-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-20 09:55:00 | 291.40 | 292.34 | 0.00 | ORB-short ORB[291.80,294.95] vol=1.7x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-20 10:00:00 | 290.19 | 292.06 | 0.00 | T1 1.5R @ 290.19 |
| Stop hit — per-position SL triggered | 2023-10-20 10:05:00 | 291.40 | 292.03 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2023-10-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-23 10:05:00 | 290.00 | 290.83 | 0.00 | ORB-short ORB[290.10,293.00] vol=3.1x ATR=0.71 |
| Stop hit — per-position SL triggered | 2023-10-23 10:10:00 | 290.71 | 290.61 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2023-10-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-26 09:50:00 | 280.20 | 281.55 | 0.00 | ORB-short ORB[280.60,284.40] vol=2.5x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:55:00 | 278.21 | 281.23 | 0.00 | T1 1.5R @ 278.21 |
| Stop hit — per-position SL triggered | 2023-10-26 10:00:00 | 280.20 | 281.16 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-10-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-30 10:30:00 | 293.10 | 289.28 | 0.00 | ORB-long ORB[287.25,290.10] vol=2.1x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-30 10:45:00 | 294.92 | 290.01 | 0.00 | T1 1.5R @ 294.92 |
| Stop hit — per-position SL triggered | 2023-10-30 10:55:00 | 293.10 | 290.22 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-11-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-09 10:15:00 | 284.85 | 288.70 | 0.00 | ORB-short ORB[289.40,291.15] vol=2.5x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-09 11:00:00 | 283.16 | 285.84 | 0.00 | T1 1.5R @ 283.16 |
| Stop hit — per-position SL triggered | 2023-11-09 11:20:00 | 284.85 | 284.49 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-11-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-20 11:10:00 | 292.65 | 294.05 | 0.00 | ORB-short ORB[293.15,296.15] vol=2.0x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-20 11:45:00 | 291.40 | 293.85 | 0.00 | T1 1.5R @ 291.40 |
| Stop hit — per-position SL triggered | 2023-11-20 14:40:00 | 292.65 | 291.91 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2023-11-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 09:45:00 | 294.40 | 293.20 | 0.00 | ORB-long ORB[290.25,293.55] vol=1.9x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-21 10:05:00 | 296.32 | 293.91 | 0.00 | T1 1.5R @ 296.32 |
| Stop hit — per-position SL triggered | 2023-11-21 11:05:00 | 294.40 | 295.05 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2023-11-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-23 10:05:00 | 283.95 | 285.33 | 0.00 | ORB-short ORB[284.20,288.20] vol=2.0x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-23 10:15:00 | 282.74 | 284.46 | 0.00 | T1 1.5R @ 282.74 |
| Stop hit — per-position SL triggered | 2023-11-23 10:40:00 | 283.95 | 284.11 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-12-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-13 10:05:00 | 333.45 | 329.91 | 0.00 | ORB-long ORB[326.70,330.30] vol=3.9x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-13 10:20:00 | 335.70 | 331.35 | 0.00 | T1 1.5R @ 335.70 |
| Stop hit — per-position SL triggered | 2023-12-13 10:35:00 | 333.45 | 332.75 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-12-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 09:35:00 | 337.00 | 333.97 | 0.00 | ORB-long ORB[331.70,334.50] vol=2.3x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 09:45:00 | 339.27 | 335.97 | 0.00 | T1 1.5R @ 339.27 |
| Stop hit — per-position SL triggered | 2023-12-20 10:15:00 | 337.00 | 337.86 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2023-12-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-27 10:55:00 | 328.10 | 331.03 | 0.00 | ORB-short ORB[330.65,334.85] vol=2.5x ATR=0.82 |
| Stop hit — per-position SL triggered | 2023-12-27 11:50:00 | 328.92 | 330.33 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-01-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 09:40:00 | 327.75 | 326.75 | 0.00 | ORB-long ORB[323.50,327.70] vol=1.5x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-04 10:00:00 | 329.47 | 327.26 | 0.00 | T1 1.5R @ 329.47 |
| Stop hit — per-position SL triggered | 2024-01-04 10:10:00 | 327.75 | 327.31 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-08 11:15:00 | 327.50 | 328.01 | 0.00 | ORB-short ORB[327.90,332.60] vol=1.7x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-08 11:55:00 | 326.05 | 327.76 | 0.00 | T1 1.5R @ 326.05 |
| Stop hit — per-position SL triggered | 2024-01-08 12:05:00 | 327.50 | 327.75 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-10 11:15:00 | 322.75 | 324.81 | 0.00 | ORB-short ORB[324.05,328.65] vol=1.9x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-10 11:35:00 | 321.61 | 324.34 | 0.00 | T1 1.5R @ 321.61 |
| Stop hit — per-position SL triggered | 2024-01-10 11:55:00 | 322.75 | 323.82 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-01-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-25 09:35:00 | 367.85 | 365.37 | 0.00 | ORB-long ORB[363.05,367.00] vol=2.2x ATR=1.58 |
| Stop hit — per-position SL triggered | 2024-01-25 09:40:00 | 366.27 | 366.36 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-02-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-05 11:05:00 | 364.35 | 369.36 | 0.00 | ORB-short ORB[368.35,372.90] vol=1.5x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-05 11:35:00 | 362.38 | 368.51 | 0.00 | T1 1.5R @ 362.38 |
| Stop hit — per-position SL triggered | 2024-02-05 11:45:00 | 364.35 | 368.15 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-02-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-08 09:55:00 | 373.90 | 371.67 | 0.00 | ORB-long ORB[367.55,372.65] vol=2.2x ATR=1.28 |
| Stop hit — per-position SL triggered | 2024-02-08 10:00:00 | 372.62 | 371.81 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-02-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-19 09:40:00 | 348.25 | 351.34 | 0.00 | ORB-short ORB[350.75,355.35] vol=1.9x ATR=1.44 |
| Stop hit — per-position SL triggered | 2024-02-19 10:15:00 | 349.69 | 349.84 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-02-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-23 09:30:00 | 347.00 | 348.75 | 0.00 | ORB-short ORB[347.40,351.85] vol=3.1x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-23 09:50:00 | 344.91 | 347.75 | 0.00 | T1 1.5R @ 344.91 |
| Target hit | 2024-02-23 14:00:00 | 345.35 | 345.28 | 0.00 | Trail-exit close>VWAP |

### Cycle 51 — SELL (started 2024-02-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 11:00:00 | 333.00 | 335.87 | 0.00 | ORB-short ORB[336.50,339.00] vol=3.4x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-28 11:25:00 | 331.54 | 334.86 | 0.00 | T1 1.5R @ 331.54 |
| Stop hit — per-position SL triggered | 2024-02-28 14:10:00 | 333.00 | 332.69 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-03-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-05 10:40:00 | 345.55 | 344.99 | 0.00 | ORB-long ORB[340.65,344.80] vol=9.5x ATR=1.46 |
| Stop hit — per-position SL triggered | 2024-03-05 11:10:00 | 344.09 | 344.96 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-03-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 09:40:00 | 341.50 | 341.82 | 0.00 | ORB-short ORB[343.20,346.95] vol=4.0x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 09:55:00 | 339.17 | 341.35 | 0.00 | T1 1.5R @ 339.17 |
| Target hit | 2024-03-06 13:25:00 | 338.05 | 336.84 | 0.00 | Trail-exit close>VWAP |

### Cycle 54 — SELL (started 2024-03-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-13 09:40:00 | 329.20 | 331.65 | 0.00 | ORB-short ORB[332.05,336.55] vol=2.3x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 09:45:00 | 326.45 | 330.76 | 0.00 | T1 1.5R @ 326.45 |
| Target hit | 2024-03-13 10:45:00 | 328.95 | 327.90 | 0.00 | Trail-exit close>VWAP |

### Cycle 55 — SELL (started 2024-03-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-15 11:00:00 | 311.45 | 315.35 | 0.00 | ORB-short ORB[315.00,319.65] vol=2.4x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-15 12:05:00 | 309.49 | 314.04 | 0.00 | T1 1.5R @ 309.49 |
| Stop hit — per-position SL triggered | 2024-03-15 13:10:00 | 311.45 | 313.00 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-03-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 09:45:00 | 312.60 | 313.70 | 0.00 | ORB-short ORB[313.00,315.35] vol=2.5x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-19 10:00:00 | 310.82 | 312.85 | 0.00 | T1 1.5R @ 310.82 |
| Stop hit — per-position SL triggered | 2024-03-19 11:10:00 | 312.60 | 311.40 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-03-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-20 10:00:00 | 305.60 | 307.49 | 0.00 | ORB-short ORB[307.05,311.00] vol=5.6x ATR=1.48 |
| Stop hit — per-position SL triggered | 2024-03-20 10:55:00 | 307.08 | 306.57 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2024-04-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-04 10:10:00 | 342.35 | 339.51 | 0.00 | ORB-long ORB[335.50,339.25] vol=10.2x ATR=1.39 |
| Stop hit — per-position SL triggered | 2024-04-04 10:30:00 | 340.96 | 339.66 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-04-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-05 10:10:00 | 327.00 | 329.38 | 0.00 | ORB-short ORB[328.45,331.85] vol=3.3x ATR=1.55 |
| Stop hit — per-position SL triggered | 2024-04-05 10:15:00 | 328.55 | 329.33 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-04-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-30 10:35:00 | 330.25 | 331.85 | 0.00 | ORB-short ORB[331.15,334.90] vol=2.2x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-30 10:55:00 | 328.48 | 331.45 | 0.00 | T1 1.5R @ 328.48 |
| Target hit | 2024-04-30 15:20:00 | 326.45 | 328.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — SELL (started 2024-05-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-03 10:50:00 | 326.50 | 330.70 | 0.00 | ORB-short ORB[330.60,334.90] vol=1.6x ATR=1.80 |
| Stop hit — per-position SL triggered | 2024-05-03 12:45:00 | 328.30 | 329.61 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-05-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 09:40:00 | 319.15 | 320.50 | 0.00 | ORB-short ORB[320.20,324.00] vol=3.1x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 10:05:00 | 317.39 | 319.91 | 0.00 | T1 1.5R @ 317.39 |
| Stop hit — per-position SL triggered | 2024-05-09 10:10:00 | 319.15 | 319.52 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-05-10 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-10 10:25:00 | 314.10 | 314.59 | 0.00 | ORB-short ORB[314.20,317.50] vol=1.7x ATR=1.22 |
| Stop hit — per-position SL triggered | 2024-05-10 11:25:00 | 315.32 | 314.48 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-19 09:35:00 | 264.50 | 2023-05-19 12:40:00 | 263.52 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2023-05-19 09:35:00 | 264.50 | 2023-05-19 13:15:00 | 264.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-22 11:00:00 | 267.20 | 2023-05-22 11:05:00 | 266.38 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-05-24 09:30:00 | 260.60 | 2023-05-24 09:40:00 | 261.60 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2023-06-07 10:25:00 | 278.10 | 2023-06-07 10:30:00 | 277.36 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-06-14 09:40:00 | 273.25 | 2023-06-14 10:05:00 | 272.04 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2023-06-14 09:40:00 | 273.25 | 2023-06-14 11:00:00 | 272.90 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2023-06-20 09:50:00 | 266.20 | 2023-06-20 13:15:00 | 264.92 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2023-06-20 09:50:00 | 266.20 | 2023-06-20 15:20:00 | 266.00 | TARGET_HIT | 0.50 | 0.08% |
| BUY | retest1 | 2023-06-22 10:40:00 | 246.00 | 2023-06-22 12:05:00 | 245.01 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2023-06-26 11:00:00 | 245.70 | 2023-06-26 12:05:00 | 244.86 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-07-04 09:40:00 | 246.85 | 2023-07-04 09:55:00 | 245.88 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-07-04 09:40:00 | 246.85 | 2023-07-04 10:15:00 | 246.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-06 09:40:00 | 252.90 | 2023-07-06 09:55:00 | 251.74 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2023-07-07 09:35:00 | 250.70 | 2023-07-07 09:40:00 | 251.79 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2023-07-14 11:10:00 | 268.40 | 2023-07-14 11:15:00 | 269.91 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2023-07-14 11:10:00 | 268.40 | 2023-07-14 11:30:00 | 268.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-19 09:55:00 | 271.80 | 2023-07-19 10:10:00 | 270.55 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2023-07-26 10:15:00 | 274.15 | 2023-07-26 13:00:00 | 272.29 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2023-07-26 10:15:00 | 274.15 | 2023-07-26 15:20:00 | 271.05 | TARGET_HIT | 0.50 | 1.13% |
| SELL | retest1 | 2023-08-08 10:40:00 | 264.30 | 2023-08-08 11:30:00 | 265.25 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-08-11 10:40:00 | 269.00 | 2023-08-11 10:45:00 | 268.05 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-08-17 10:30:00 | 275.20 | 2023-08-17 13:10:00 | 276.69 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2023-08-17 10:30:00 | 275.20 | 2023-08-17 15:20:00 | 277.15 | TARGET_HIT | 0.50 | 0.71% |
| SELL | retest1 | 2023-08-28 09:35:00 | 267.55 | 2023-08-28 10:45:00 | 268.58 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2023-08-29 10:00:00 | 267.85 | 2023-08-29 10:45:00 | 266.32 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2023-08-29 10:00:00 | 267.85 | 2023-08-29 15:20:00 | 265.00 | TARGET_HIT | 0.50 | 1.06% |
| SELL | retest1 | 2023-08-30 10:45:00 | 265.50 | 2023-08-30 11:00:00 | 264.25 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2023-08-30 10:45:00 | 265.50 | 2023-08-30 15:15:00 | 265.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-01 11:05:00 | 269.75 | 2023-09-01 11:40:00 | 268.89 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-09-04 09:50:00 | 271.45 | 2023-09-04 10:00:00 | 270.63 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-09-12 09:40:00 | 275.50 | 2023-09-12 09:45:00 | 274.21 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2023-09-12 09:40:00 | 275.50 | 2023-09-12 10:05:00 | 274.50 | TARGET_HIT | 0.50 | 0.36% |
| SELL | retest1 | 2023-09-22 10:55:00 | 274.95 | 2023-09-22 11:35:00 | 275.91 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-10-03 10:15:00 | 298.65 | 2023-10-03 10:35:00 | 297.31 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2023-10-05 10:40:00 | 293.15 | 2023-10-05 10:50:00 | 294.16 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-10-06 09:40:00 | 293.60 | 2023-10-06 15:20:00 | 292.65 | TARGET_HIT | 1.00 | 0.32% |
| BUY | retest1 | 2023-10-10 09:45:00 | 295.70 | 2023-10-10 10:30:00 | 294.76 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-10-12 10:10:00 | 293.50 | 2023-10-12 10:35:00 | 292.05 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2023-10-12 10:10:00 | 293.50 | 2023-10-12 10:40:00 | 293.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-17 10:40:00 | 297.50 | 2023-10-17 11:00:00 | 298.81 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-10-17 10:40:00 | 297.50 | 2023-10-17 15:20:00 | 301.50 | TARGET_HIT | 0.50 | 1.34% |
| SELL | retest1 | 2023-10-18 10:45:00 | 298.25 | 2023-10-18 10:55:00 | 299.24 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-10-20 09:55:00 | 291.40 | 2023-10-20 10:00:00 | 290.19 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2023-10-20 09:55:00 | 291.40 | 2023-10-20 10:05:00 | 291.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-23 10:05:00 | 290.00 | 2023-10-23 10:10:00 | 290.71 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-10-26 09:50:00 | 280.20 | 2023-10-26 09:55:00 | 278.21 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2023-10-26 09:50:00 | 280.20 | 2023-10-26 10:00:00 | 280.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-30 10:30:00 | 293.10 | 2023-10-30 10:45:00 | 294.92 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2023-10-30 10:30:00 | 293.10 | 2023-10-30 10:55:00 | 293.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-09 10:15:00 | 284.85 | 2023-11-09 11:00:00 | 283.16 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2023-11-09 10:15:00 | 284.85 | 2023-11-09 11:20:00 | 284.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-20 11:10:00 | 292.65 | 2023-11-20 11:45:00 | 291.40 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2023-11-20 11:10:00 | 292.65 | 2023-11-20 14:40:00 | 292.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-21 09:45:00 | 294.40 | 2023-11-21 10:05:00 | 296.32 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2023-11-21 09:45:00 | 294.40 | 2023-11-21 11:05:00 | 294.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-23 10:05:00 | 283.95 | 2023-11-23 10:15:00 | 282.74 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2023-11-23 10:05:00 | 283.95 | 2023-11-23 10:40:00 | 283.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-13 10:05:00 | 333.45 | 2023-12-13 10:20:00 | 335.70 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2023-12-13 10:05:00 | 333.45 | 2023-12-13 10:35:00 | 333.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-20 09:35:00 | 337.00 | 2023-12-20 09:45:00 | 339.27 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2023-12-20 09:35:00 | 337.00 | 2023-12-20 10:15:00 | 337.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-27 10:55:00 | 328.10 | 2023-12-27 11:50:00 | 328.92 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-01-04 09:40:00 | 327.75 | 2024-01-04 10:00:00 | 329.47 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-01-04 09:40:00 | 327.75 | 2024-01-04 10:10:00 | 327.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-08 11:15:00 | 327.50 | 2024-01-08 11:55:00 | 326.05 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-01-08 11:15:00 | 327.50 | 2024-01-08 12:05:00 | 327.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-10 11:15:00 | 322.75 | 2024-01-10 11:35:00 | 321.61 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-01-10 11:15:00 | 322.75 | 2024-01-10 11:55:00 | 322.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-25 09:35:00 | 367.85 | 2024-01-25 09:40:00 | 366.27 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-02-05 11:05:00 | 364.35 | 2024-02-05 11:35:00 | 362.38 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-02-05 11:05:00 | 364.35 | 2024-02-05 11:45:00 | 364.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-08 09:55:00 | 373.90 | 2024-02-08 10:00:00 | 372.62 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-02-19 09:40:00 | 348.25 | 2024-02-19 10:15:00 | 349.69 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-02-23 09:30:00 | 347.00 | 2024-02-23 09:50:00 | 344.91 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-02-23 09:30:00 | 347.00 | 2024-02-23 14:00:00 | 345.35 | TARGET_HIT | 0.50 | 0.48% |
| SELL | retest1 | 2024-02-28 11:00:00 | 333.00 | 2024-02-28 11:25:00 | 331.54 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-02-28 11:00:00 | 333.00 | 2024-02-28 14:10:00 | 333.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-05 10:40:00 | 345.55 | 2024-03-05 11:10:00 | 344.09 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-03-06 09:40:00 | 341.50 | 2024-03-06 09:55:00 | 339.17 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2024-03-06 09:40:00 | 341.50 | 2024-03-06 13:25:00 | 338.05 | TARGET_HIT | 0.50 | 1.01% |
| SELL | retest1 | 2024-03-13 09:40:00 | 329.20 | 2024-03-13 09:45:00 | 326.45 | PARTIAL | 0.50 | 0.84% |
| SELL | retest1 | 2024-03-13 09:40:00 | 329.20 | 2024-03-13 10:45:00 | 328.95 | TARGET_HIT | 0.50 | 0.08% |
| SELL | retest1 | 2024-03-15 11:00:00 | 311.45 | 2024-03-15 12:05:00 | 309.49 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-03-15 11:00:00 | 311.45 | 2024-03-15 13:10:00 | 311.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-19 09:45:00 | 312.60 | 2024-03-19 10:00:00 | 310.82 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-03-19 09:45:00 | 312.60 | 2024-03-19 11:10:00 | 312.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-20 10:00:00 | 305.60 | 2024-03-20 10:55:00 | 307.08 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-04-04 10:10:00 | 342.35 | 2024-04-04 10:30:00 | 340.96 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-04-05 10:10:00 | 327.00 | 2024-04-05 10:15:00 | 328.55 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-04-30 10:35:00 | 330.25 | 2024-04-30 10:55:00 | 328.48 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-04-30 10:35:00 | 330.25 | 2024-04-30 15:20:00 | 326.45 | TARGET_HIT | 0.50 | 1.15% |
| SELL | retest1 | 2024-05-03 10:50:00 | 326.50 | 2024-05-03 12:45:00 | 328.30 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2024-05-09 09:40:00 | 319.15 | 2024-05-09 10:05:00 | 317.39 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-05-09 09:40:00 | 319.15 | 2024-05-09 10:10:00 | 319.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-10 10:25:00 | 314.10 | 2024-05-10 11:25:00 | 315.32 | STOP_HIT | 1.00 | -0.39% |
