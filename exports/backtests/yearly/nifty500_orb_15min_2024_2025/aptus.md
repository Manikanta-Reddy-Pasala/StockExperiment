# Aptus Value Housing Finance India Ltd. (APTUS)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-04-04 15:25:00 (16833 bars)
- **Last close:** 298.85
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
| ENTRY1 | 51 |
| ENTRY2 | 0 |
| PARTIAL | 20 |
| TARGET_HIT | 11 |
| STOP_HIT | 40 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 71 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 31 / 40
- **Target hits / Stop hits / Partials:** 11 / 40 / 20
- **Avg / median % per leg:** 0.20% / 0.00%
- **Sum % (uncompounded):** 14.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 32 | 13 | 40.6% | 5 | 19 | 8 | 0.30% | 9.6% |
| BUY @ 2nd Alert (retest1) | 32 | 13 | 40.6% | 5 | 19 | 8 | 0.30% | 9.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 39 | 18 | 46.2% | 6 | 21 | 12 | 0.12% | 4.5% |
| SELL @ 2nd Alert (retest1) | 39 | 18 | 46.2% | 6 | 21 | 12 | 0.12% | 4.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 71 | 31 | 43.7% | 11 | 40 | 20 | 0.20% | 14.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 10:05:00 | 303.95 | 305.69 | 0.00 | ORB-short ORB[305.10,307.95] vol=2.6x ATR=0.99 |
| Stop hit — per-position SL triggered | 2024-05-24 10:20:00 | 304.94 | 305.32 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 10:45:00 | 306.30 | 304.17 | 0.00 | ORB-long ORB[301.25,305.10] vol=1.8x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-29 11:50:00 | 308.05 | 304.86 | 0.00 | T1 1.5R @ 308.05 |
| Stop hit — per-position SL triggered | 2024-05-29 12:40:00 | 306.30 | 305.20 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-06-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 09:50:00 | 324.75 | 322.21 | 0.00 | ORB-long ORB[319.40,323.05] vol=2.0x ATR=1.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 10:10:00 | 326.88 | 323.84 | 0.00 | T1 1.5R @ 326.88 |
| Target hit | 2024-06-12 15:20:00 | 340.00 | 333.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2024-06-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 09:50:00 | 341.65 | 343.68 | 0.00 | ORB-short ORB[343.25,348.10] vol=3.3x ATR=1.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 10:10:00 | 339.55 | 342.63 | 0.00 | T1 1.5R @ 339.55 |
| Target hit | 2024-06-21 15:20:00 | 337.95 | 340.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2024-06-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 10:50:00 | 331.35 | 332.81 | 0.00 | ORB-short ORB[333.00,336.70] vol=1.7x ATR=1.26 |
| Stop hit — per-position SL triggered | 2024-06-26 11:10:00 | 332.61 | 333.20 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:30:00 | 334.00 | 331.13 | 0.00 | ORB-long ORB[328.20,331.45] vol=2.3x ATR=1.45 |
| Stop hit — per-position SL triggered | 2024-06-27 09:35:00 | 332.55 | 331.28 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-07-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 10:05:00 | 349.30 | 346.43 | 0.00 | ORB-long ORB[343.10,347.20] vol=5.5x ATR=1.51 |
| Stop hit — per-position SL triggered | 2024-07-04 10:15:00 | 347.79 | 346.81 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-07-09 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 10:20:00 | 332.80 | 335.56 | 0.00 | ORB-short ORB[334.60,337.95] vol=3.5x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 10:30:00 | 330.36 | 334.68 | 0.00 | T1 1.5R @ 330.36 |
| Stop hit — per-position SL triggered | 2024-07-09 10:50:00 | 332.80 | 334.41 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-07-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:30:00 | 323.30 | 326.13 | 0.00 | ORB-short ORB[326.50,330.05] vol=2.4x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:35:00 | 321.24 | 325.80 | 0.00 | T1 1.5R @ 321.24 |
| Stop hit — per-position SL triggered | 2024-07-10 10:55:00 | 323.30 | 323.05 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 10:00:00 | 329.80 | 327.56 | 0.00 | ORB-long ORB[325.00,328.00] vol=2.4x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 10:10:00 | 331.71 | 332.79 | 0.00 | T1 1.5R @ 331.71 |
| Target hit | 2024-07-11 10:15:00 | 332.60 | 332.85 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2024-07-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 09:35:00 | 327.40 | 326.24 | 0.00 | ORB-long ORB[324.15,327.15] vol=2.0x ATR=1.10 |
| Stop hit — per-position SL triggered | 2024-07-16 09:50:00 | 326.30 | 326.37 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 10:35:00 | 323.30 | 325.23 | 0.00 | ORB-short ORB[324.30,326.80] vol=3.9x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 10:55:00 | 321.28 | 324.66 | 0.00 | T1 1.5R @ 321.28 |
| Stop hit — per-position SL triggered | 2024-07-18 11:55:00 | 323.30 | 323.92 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 10:45:00 | 324.10 | 322.54 | 0.00 | ORB-long ORB[319.35,322.70] vol=1.5x ATR=1.09 |
| Stop hit — per-position SL triggered | 2024-07-25 11:00:00 | 323.01 | 322.68 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:30:00 | 326.50 | 325.05 | 0.00 | ORB-long ORB[320.65,324.80] vol=8.2x ATR=1.23 |
| Stop hit — per-position SL triggered | 2024-07-26 09:40:00 | 325.27 | 325.17 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-08-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-09 09:50:00 | 309.50 | 310.14 | 0.00 | ORB-short ORB[309.70,313.00] vol=2.3x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 15:05:00 | 308.19 | 309.62 | 0.00 | T1 1.5R @ 308.19 |
| Target hit | 2024-08-09 15:20:00 | 308.40 | 309.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2024-08-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 10:55:00 | 307.00 | 308.43 | 0.00 | ORB-short ORB[307.70,310.65] vol=5.3x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-13 14:15:00 | 305.90 | 307.58 | 0.00 | T1 1.5R @ 305.90 |
| Target hit | 2024-08-13 15:20:00 | 305.25 | 307.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2024-08-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:45:00 | 307.35 | 305.49 | 0.00 | ORB-long ORB[303.70,307.00] vol=2.5x ATR=1.07 |
| Stop hit — per-position SL triggered | 2024-08-20 10:15:00 | 306.28 | 305.84 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-08-22 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 10:05:00 | 312.20 | 314.35 | 0.00 | ORB-short ORB[314.20,316.55] vol=1.9x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 10:10:00 | 310.86 | 314.02 | 0.00 | T1 1.5R @ 310.86 |
| Stop hit — per-position SL triggered | 2024-08-22 10:25:00 | 312.20 | 312.75 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 11:15:00 | 310.65 | 308.97 | 0.00 | ORB-long ORB[306.95,310.15] vol=3.0x ATR=0.93 |
| Stop hit — per-position SL triggered | 2024-08-26 11:55:00 | 309.72 | 309.27 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-09-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 09:30:00 | 324.45 | 321.55 | 0.00 | ORB-long ORB[318.25,322.65] vol=2.9x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 09:40:00 | 325.72 | 322.86 | 0.00 | T1 1.5R @ 325.72 |
| Stop hit — per-position SL triggered | 2024-09-04 09:50:00 | 324.45 | 323.64 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-09-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 09:30:00 | 320.30 | 318.73 | 0.00 | ORB-long ORB[315.60,319.45] vol=2.4x ATR=1.38 |
| Stop hit — per-position SL triggered | 2024-09-10 09:40:00 | 318.92 | 318.83 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-09-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 11:10:00 | 353.35 | 350.11 | 0.00 | ORB-long ORB[347.75,351.45] vol=2.2x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-20 12:45:00 | 355.53 | 350.98 | 0.00 | T1 1.5R @ 355.53 |
| Target hit | 2024-09-20 15:20:00 | 362.65 | 354.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — BUY (started 2024-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 09:30:00 | 371.25 | 369.44 | 0.00 | ORB-long ORB[366.10,371.00] vol=1.6x ATR=1.54 |
| Stop hit — per-position SL triggered | 2024-09-26 09:40:00 | 369.71 | 369.71 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-10-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-17 10:45:00 | 382.50 | 377.87 | 0.00 | ORB-long ORB[377.30,381.05] vol=1.6x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 10:50:00 | 385.03 | 382.58 | 0.00 | T1 1.5R @ 385.03 |
| Target hit | 2024-10-17 11:20:00 | 385.05 | 385.69 | 0.00 | Trail-exit close<VWAP |

### Cycle 25 — SELL (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 11:15:00 | 338.90 | 340.42 | 0.00 | ORB-short ORB[339.05,343.95] vol=1.5x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 11:25:00 | 337.38 | 340.15 | 0.00 | T1 1.5R @ 337.38 |
| Stop hit — per-position SL triggered | 2024-10-29 11:30:00 | 338.90 | 340.10 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-11-22 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-22 10:25:00 | 315.05 | 316.71 | 0.00 | ORB-short ORB[315.95,319.40] vol=2.0x ATR=1.00 |
| Stop hit — per-position SL triggered | 2024-11-22 12:00:00 | 316.05 | 316.33 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-12-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:50:00 | 325.60 | 323.41 | 0.00 | ORB-long ORB[320.40,324.15] vol=1.9x ATR=1.17 |
| Stop hit — per-position SL triggered | 2024-12-04 09:55:00 | 324.43 | 323.56 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-12-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 09:40:00 | 321.90 | 322.89 | 0.00 | ORB-short ORB[322.30,326.00] vol=6.1x ATR=1.22 |
| Target hit | 2024-12-05 15:20:00 | 321.60 | 322.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — BUY (started 2024-12-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 11:00:00 | 326.00 | 324.07 | 0.00 | ORB-long ORB[320.20,323.70] vol=3.6x ATR=1.13 |
| Stop hit — per-position SL triggered | 2024-12-06 12:45:00 | 324.87 | 325.09 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-12-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 09:50:00 | 314.90 | 316.10 | 0.00 | ORB-short ORB[315.80,318.80] vol=1.6x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 11:25:00 | 313.23 | 315.15 | 0.00 | T1 1.5R @ 313.23 |
| Target hit | 2024-12-10 14:50:00 | 314.45 | 314.44 | 0.00 | Trail-exit close>VWAP |

### Cycle 31 — SELL (started 2024-12-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 10:25:00 | 312.80 | 314.82 | 0.00 | ORB-short ORB[313.50,315.50] vol=1.7x ATR=0.64 |
| Stop hit — per-position SL triggered | 2024-12-11 10:30:00 | 313.44 | 314.78 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-12-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 09:50:00 | 312.20 | 310.56 | 0.00 | ORB-long ORB[309.30,311.90] vol=4.4x ATR=1.01 |
| Stop hit — per-position SL triggered | 2024-12-17 10:00:00 | 311.19 | 311.57 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-12-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 10:10:00 | 308.80 | 305.08 | 0.00 | ORB-long ORB[301.25,305.45] vol=1.7x ATR=1.26 |
| Stop hit — per-position SL triggered | 2024-12-19 10:25:00 | 307.54 | 306.68 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-12-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 09:30:00 | 310.50 | 309.76 | 0.00 | ORB-long ORB[308.30,310.25] vol=2.4x ATR=0.72 |
| Stop hit — per-position SL triggered | 2024-12-20 10:00:00 | 309.78 | 310.60 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-24 11:15:00 | 309.05 | 310.20 | 0.00 | ORB-short ORB[309.35,312.70] vol=1.9x ATR=0.88 |
| Stop hit — per-position SL triggered | 2024-12-24 11:20:00 | 309.93 | 309.71 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-12-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:40:00 | 302.90 | 304.16 | 0.00 | ORB-short ORB[303.50,307.45] vol=2.1x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 10:30:00 | 301.19 | 303.35 | 0.00 | T1 1.5R @ 301.19 |
| Stop hit — per-position SL triggered | 2024-12-26 10:50:00 | 302.90 | 302.75 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-01-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 10:45:00 | 297.50 | 298.32 | 0.00 | ORB-short ORB[298.10,300.65] vol=1.6x ATR=0.92 |
| Stop hit — per-position SL triggered | 2025-01-03 10:50:00 | 298.42 | 298.32 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-01-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 11:10:00 | 290.85 | 296.10 | 0.00 | ORB-short ORB[295.05,298.40] vol=1.7x ATR=1.06 |
| Stop hit — per-position SL triggered | 2025-01-06 11:30:00 | 291.91 | 295.35 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-01-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 09:30:00 | 290.25 | 291.77 | 0.00 | ORB-short ORB[291.20,295.50] vol=1.6x ATR=0.89 |
| Stop hit — per-position SL triggered | 2025-01-08 09:35:00 | 291.14 | 291.80 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-01-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 09:35:00 | 275.85 | 277.96 | 0.00 | ORB-short ORB[276.35,280.50] vol=2.0x ATR=1.46 |
| Stop hit — per-position SL triggered | 2025-01-13 09:40:00 | 277.31 | 277.90 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-01-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:35:00 | 290.55 | 291.90 | 0.00 | ORB-short ORB[291.35,294.50] vol=1.8x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 11:00:00 | 289.02 | 291.48 | 0.00 | T1 1.5R @ 289.02 |
| Stop hit — per-position SL triggered | 2025-01-21 11:40:00 | 290.55 | 291.03 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:30:00 | 292.25 | 290.66 | 0.00 | ORB-long ORB[288.70,291.50] vol=1.8x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 09:40:00 | 294.27 | 292.43 | 0.00 | T1 1.5R @ 294.27 |
| Stop hit — per-position SL triggered | 2025-01-30 09:55:00 | 292.25 | 292.64 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-02-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-06 09:55:00 | 325.00 | 322.27 | 0.00 | ORB-long ORB[319.50,324.00] vol=3.8x ATR=1.49 |
| Stop hit — per-position SL triggered | 2025-02-06 12:00:00 | 323.51 | 323.45 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-02-14 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 10:20:00 | 301.85 | 303.45 | 0.00 | ORB-short ORB[303.35,306.80] vol=2.1x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 10:30:00 | 300.23 | 303.01 | 0.00 | T1 1.5R @ 300.23 |
| Stop hit — per-position SL triggered | 2025-02-14 11:20:00 | 301.85 | 302.28 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-03-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-11 10:05:00 | 302.40 | 301.91 | 0.00 | ORB-long ORB[300.00,302.20] vol=1.5x ATR=1.22 |
| Stop hit — per-position SL triggered | 2025-03-11 10:10:00 | 301.18 | 301.77 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-03-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 09:50:00 | 299.80 | 301.53 | 0.00 | ORB-short ORB[301.00,305.20] vol=1.5x ATR=1.20 |
| Stop hit — per-position SL triggered | 2025-03-12 10:30:00 | 301.00 | 300.72 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-03-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-19 10:40:00 | 300.00 | 301.22 | 0.00 | ORB-short ORB[301.10,304.50] vol=1.7x ATR=0.81 |
| Stop hit — per-position SL triggered | 2025-03-19 11:00:00 | 300.81 | 301.32 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-03-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-20 10:45:00 | 296.75 | 300.20 | 0.00 | ORB-short ORB[301.00,303.65] vol=2.0x ATR=1.11 |
| Target hit | 2025-03-20 15:20:00 | 296.00 | 297.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — BUY (started 2025-03-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 10:35:00 | 299.40 | 299.10 | 0.00 | ORB-long ORB[295.30,298.95] vol=9.6x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 11:10:00 | 301.07 | 299.36 | 0.00 | T1 1.5R @ 301.07 |
| Target hit | 2025-03-21 15:20:00 | 305.40 | 301.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — SELL (started 2025-03-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:40:00 | 298.55 | 301.32 | 0.00 | ORB-short ORB[302.10,304.70] vol=3.4x ATR=1.31 |
| Stop hit — per-position SL triggered | 2025-03-26 09:55:00 | 299.86 | 300.83 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-28 11:15:00 | 298.70 | 300.28 | 0.00 | ORB-short ORB[298.85,301.95] vol=1.9x ATR=0.76 |
| Stop hit — per-position SL triggered | 2025-03-28 11:20:00 | 299.46 | 300.27 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-24 10:05:00 | 303.95 | 2024-05-24 10:20:00 | 304.94 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-05-29 10:45:00 | 306.30 | 2024-05-29 11:50:00 | 308.05 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-05-29 10:45:00 | 306.30 | 2024-05-29 12:40:00 | 306.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-12 09:50:00 | 324.75 | 2024-06-12 10:10:00 | 326.88 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2024-06-12 09:50:00 | 324.75 | 2024-06-12 15:20:00 | 340.00 | TARGET_HIT | 0.50 | 4.70% |
| SELL | retest1 | 2024-06-21 09:50:00 | 341.65 | 2024-06-21 10:10:00 | 339.55 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-06-21 09:50:00 | 341.65 | 2024-06-21 15:20:00 | 337.95 | TARGET_HIT | 0.50 | 1.08% |
| SELL | retest1 | 2024-06-26 10:50:00 | 331.35 | 2024-06-26 11:10:00 | 332.61 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-06-27 09:30:00 | 334.00 | 2024-06-27 09:35:00 | 332.55 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-07-04 10:05:00 | 349.30 | 2024-07-04 10:15:00 | 347.79 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-07-09 10:20:00 | 332.80 | 2024-07-09 10:30:00 | 330.36 | PARTIAL | 0.50 | 0.73% |
| SELL | retest1 | 2024-07-09 10:20:00 | 332.80 | 2024-07-09 10:50:00 | 332.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-10 10:30:00 | 323.30 | 2024-07-10 10:35:00 | 321.24 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-07-10 10:30:00 | 323.30 | 2024-07-10 10:55:00 | 323.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-11 10:00:00 | 329.80 | 2024-07-11 10:10:00 | 331.71 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-07-11 10:00:00 | 329.80 | 2024-07-11 10:15:00 | 332.60 | TARGET_HIT | 0.50 | 0.85% |
| BUY | retest1 | 2024-07-16 09:35:00 | 327.40 | 2024-07-16 09:50:00 | 326.30 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-07-18 10:35:00 | 323.30 | 2024-07-18 10:55:00 | 321.28 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-07-18 10:35:00 | 323.30 | 2024-07-18 11:55:00 | 323.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-25 10:45:00 | 324.10 | 2024-07-25 11:00:00 | 323.01 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-07-26 09:30:00 | 326.50 | 2024-07-26 09:40:00 | 325.27 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-08-09 09:50:00 | 309.50 | 2024-08-09 15:05:00 | 308.19 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-08-09 09:50:00 | 309.50 | 2024-08-09 15:20:00 | 308.40 | TARGET_HIT | 0.50 | 0.36% |
| SELL | retest1 | 2024-08-13 10:55:00 | 307.00 | 2024-08-13 14:15:00 | 305.90 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-08-13 10:55:00 | 307.00 | 2024-08-13 15:20:00 | 305.25 | TARGET_HIT | 0.50 | 0.57% |
| BUY | retest1 | 2024-08-20 09:45:00 | 307.35 | 2024-08-20 10:15:00 | 306.28 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-08-22 10:05:00 | 312.20 | 2024-08-22 10:10:00 | 310.86 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-08-22 10:05:00 | 312.20 | 2024-08-22 10:25:00 | 312.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-26 11:15:00 | 310.65 | 2024-08-26 11:55:00 | 309.72 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-09-04 09:30:00 | 324.45 | 2024-09-04 09:40:00 | 325.72 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-09-04 09:30:00 | 324.45 | 2024-09-04 09:50:00 | 324.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-10 09:30:00 | 320.30 | 2024-09-10 09:40:00 | 318.92 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-09-20 11:10:00 | 353.35 | 2024-09-20 12:45:00 | 355.53 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-09-20 11:10:00 | 353.35 | 2024-09-20 15:20:00 | 362.65 | TARGET_HIT | 0.50 | 2.63% |
| BUY | retest1 | 2024-09-26 09:30:00 | 371.25 | 2024-09-26 09:40:00 | 369.71 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-10-17 10:45:00 | 382.50 | 2024-10-17 10:50:00 | 385.03 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2024-10-17 10:45:00 | 382.50 | 2024-10-17 11:20:00 | 385.05 | TARGET_HIT | 0.50 | 0.67% |
| SELL | retest1 | 2024-10-29 11:15:00 | 338.90 | 2024-10-29 11:25:00 | 337.38 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-10-29 11:15:00 | 338.90 | 2024-10-29 11:30:00 | 338.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-22 10:25:00 | 315.05 | 2024-11-22 12:00:00 | 316.05 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-12-04 09:50:00 | 325.60 | 2024-12-04 09:55:00 | 324.43 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-12-05 09:40:00 | 321.90 | 2024-12-05 15:20:00 | 321.60 | TARGET_HIT | 1.00 | 0.09% |
| BUY | retest1 | 2024-12-06 11:00:00 | 326.00 | 2024-12-06 12:45:00 | 324.87 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-12-10 09:50:00 | 314.90 | 2024-12-10 11:25:00 | 313.23 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-12-10 09:50:00 | 314.90 | 2024-12-10 14:50:00 | 314.45 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2024-12-11 10:25:00 | 312.80 | 2024-12-11 10:30:00 | 313.44 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-12-17 09:50:00 | 312.20 | 2024-12-17 10:00:00 | 311.19 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-12-19 10:10:00 | 308.80 | 2024-12-19 10:25:00 | 307.54 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-12-20 09:30:00 | 310.50 | 2024-12-20 10:00:00 | 309.78 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-12-24 11:15:00 | 309.05 | 2024-12-24 11:20:00 | 309.93 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-12-26 09:40:00 | 302.90 | 2024-12-26 10:30:00 | 301.19 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-12-26 09:40:00 | 302.90 | 2024-12-26 10:50:00 | 302.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-03 10:45:00 | 297.50 | 2025-01-03 10:50:00 | 298.42 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-01-06 11:10:00 | 290.85 | 2025-01-06 11:30:00 | 291.91 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-01-08 09:30:00 | 290.25 | 2025-01-08 09:35:00 | 291.14 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-01-13 09:35:00 | 275.85 | 2025-01-13 09:40:00 | 277.31 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2025-01-21 10:35:00 | 290.55 | 2025-01-21 11:00:00 | 289.02 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-01-21 10:35:00 | 290.55 | 2025-01-21 11:40:00 | 290.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-30 09:30:00 | 292.25 | 2025-01-30 09:40:00 | 294.27 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2025-01-30 09:30:00 | 292.25 | 2025-01-30 09:55:00 | 292.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-06 09:55:00 | 325.00 | 2025-02-06 12:00:00 | 323.51 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-02-14 10:20:00 | 301.85 | 2025-02-14 10:30:00 | 300.23 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-02-14 10:20:00 | 301.85 | 2025-02-14 11:20:00 | 301.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-11 10:05:00 | 302.40 | 2025-03-11 10:10:00 | 301.18 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-03-12 09:50:00 | 299.80 | 2025-03-12 10:30:00 | 301.00 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-03-19 10:40:00 | 300.00 | 2025-03-19 11:00:00 | 300.81 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-03-20 10:45:00 | 296.75 | 2025-03-20 15:20:00 | 296.00 | TARGET_HIT | 1.00 | 0.25% |
| BUY | retest1 | 2025-03-21 10:35:00 | 299.40 | 2025-03-21 11:10:00 | 301.07 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-03-21 10:35:00 | 299.40 | 2025-03-21 15:20:00 | 305.40 | TARGET_HIT | 0.50 | 2.00% |
| SELL | retest1 | 2025-03-26 09:40:00 | 298.55 | 2025-03-26 09:55:00 | 299.86 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-03-28 11:15:00 | 298.70 | 2025-03-28 11:20:00 | 299.46 | STOP_HIT | 1.00 | -0.25% |
