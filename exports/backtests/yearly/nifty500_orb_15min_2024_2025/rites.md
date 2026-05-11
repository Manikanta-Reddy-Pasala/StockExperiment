# RITES Ltd. (RITES)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 226.80
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
| ENTRY1 | 40 |
| ENTRY2 | 0 |
| PARTIAL | 13 |
| TARGET_HIT | 6 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 53 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 34
- **Target hits / Stop hits / Partials:** 6 / 34 / 13
- **Avg / median % per leg:** 0.06% / 0.00%
- **Sum % (uncompounded):** 3.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 10 | 38.5% | 3 | 16 | 7 | 0.15% | 4.0% |
| BUY @ 2nd Alert (retest1) | 26 | 10 | 38.5% | 3 | 16 | 7 | 0.15% | 4.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 27 | 9 | 33.3% | 3 | 18 | 6 | -0.03% | -0.9% |
| SELL @ 2nd Alert (retest1) | 27 | 9 | 33.3% | 3 | 18 | 6 | -0.03% | -0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 53 | 19 | 35.8% | 6 | 34 | 13 | 0.06% | 3.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 10:10:00 | 336.85 | 335.06 | 0.00 | ORB-long ORB[333.00,335.83] vol=2.3x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 10:30:00 | 338.74 | 335.95 | 0.00 | T1 1.5R @ 338.74 |
| Stop hit — per-position SL triggered | 2024-05-16 10:35:00 | 336.85 | 335.99 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:35:00 | 366.50 | 371.51 | 0.00 | ORB-short ORB[370.50,375.95] vol=1.6x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-22 09:40:00 | 362.79 | 370.60 | 0.00 | T1 1.5R @ 362.79 |
| Stop hit — per-position SL triggered | 2024-05-22 09:55:00 | 366.50 | 369.56 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 09:45:00 | 366.13 | 371.44 | 0.00 | ORB-short ORB[369.00,374.50] vol=2.1x ATR=2.21 |
| Stop hit — per-position SL triggered | 2024-05-27 10:05:00 | 368.34 | 370.32 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 09:45:00 | 348.58 | 351.08 | 0.00 | ORB-short ORB[349.98,354.65] vol=2.1x ATR=1.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 09:55:00 | 345.98 | 350.52 | 0.00 | T1 1.5R @ 345.98 |
| Target hit | 2024-05-31 12:00:00 | 347.00 | 346.49 | 0.00 | Trail-exit close>VWAP |

### Cycle 5 — BUY (started 2024-06-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 10:00:00 | 344.85 | 341.89 | 0.00 | ORB-long ORB[338.73,341.75] vol=3.8x ATR=1.57 |
| Stop hit — per-position SL triggered | 2024-06-12 10:05:00 | 343.28 | 342.61 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 09:35:00 | 360.98 | 358.42 | 0.00 | ORB-long ORB[355.80,359.75] vol=1.8x ATR=1.86 |
| Stop hit — per-position SL triggered | 2024-06-18 10:05:00 | 359.12 | 359.62 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 09:35:00 | 351.00 | 352.73 | 0.00 | ORB-short ORB[351.55,356.18] vol=1.6x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 10:00:00 | 349.06 | 351.88 | 0.00 | T1 1.5R @ 349.06 |
| Stop hit — per-position SL triggered | 2024-06-25 10:15:00 | 351.00 | 351.65 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 10:05:00 | 349.03 | 347.25 | 0.00 | ORB-long ORB[345.00,347.85] vol=1.8x ATR=1.18 |
| Stop hit — per-position SL triggered | 2024-06-27 10:10:00 | 347.85 | 347.29 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 11:15:00 | 348.00 | 345.35 | 0.00 | ORB-long ORB[343.03,346.53] vol=4.4x ATR=1.20 |
| Stop hit — per-position SL triggered | 2024-07-01 11:20:00 | 346.80 | 345.44 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 09:35:00 | 359.50 | 356.61 | 0.00 | ORB-long ORB[350.03,353.38] vol=14.6x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 09:40:00 | 361.79 | 359.04 | 0.00 | T1 1.5R @ 361.79 |
| Target hit | 2024-07-04 10:30:00 | 361.38 | 362.06 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — SELL (started 2024-07-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 09:30:00 | 362.23 | 364.71 | 0.00 | ORB-short ORB[362.35,367.68] vol=1.6x ATR=1.42 |
| Stop hit — per-position SL triggered | 2024-07-23 09:35:00 | 363.65 | 364.55 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-31 09:40:00 | 376.95 | 379.87 | 0.00 | ORB-short ORB[378.55,383.85] vol=2.8x ATR=1.90 |
| Stop hit — per-position SL triggered | 2024-07-31 13:45:00 | 378.85 | 377.53 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-08-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 09:30:00 | 347.30 | 349.01 | 0.00 | ORB-short ORB[347.60,352.58] vol=2.4x ATR=1.83 |
| Stop hit — per-position SL triggered | 2024-08-06 09:35:00 | 349.13 | 348.98 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-08-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-14 09:30:00 | 329.75 | 331.84 | 0.00 | ORB-short ORB[331.28,335.13] vol=3.1x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 09:35:00 | 327.97 | 330.75 | 0.00 | T1 1.5R @ 327.97 |
| Stop hit — per-position SL triggered | 2024-08-14 09:45:00 | 329.75 | 330.24 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-08-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:35:00 | 332.70 | 331.54 | 0.00 | ORB-long ORB[328.80,332.45] vol=1.8x ATR=1.22 |
| Stop hit — per-position SL triggered | 2024-08-16 09:55:00 | 331.48 | 331.65 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-08-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:30:00 | 332.53 | 330.88 | 0.00 | ORB-long ORB[328.00,332.00] vol=1.7x ATR=1.04 |
| Stop hit — per-position SL triggered | 2024-08-21 09:35:00 | 331.49 | 330.97 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-08-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 09:45:00 | 329.00 | 330.22 | 0.00 | ORB-short ORB[329.40,331.65] vol=2.8x ATR=0.77 |
| Target hit | 2024-08-22 15:20:00 | 327.65 | 329.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2024-08-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 09:40:00 | 331.95 | 329.98 | 0.00 | ORB-long ORB[327.52,331.65] vol=2.0x ATR=1.07 |
| Stop hit — per-position SL triggered | 2024-08-23 09:45:00 | 330.88 | 331.33 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 09:55:00 | 331.53 | 329.10 | 0.00 | ORB-long ORB[326.55,329.00] vol=6.5x ATR=1.13 |
| Stop hit — per-position SL triggered | 2024-08-28 10:00:00 | 330.40 | 329.24 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-08-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 09:55:00 | 325.20 | 326.22 | 0.00 | ORB-short ORB[325.27,329.18] vol=1.6x ATR=0.94 |
| Stop hit — per-position SL triggered | 2024-08-30 10:20:00 | 326.14 | 326.05 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-09-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 09:30:00 | 348.15 | 345.97 | 0.00 | ORB-long ORB[342.03,346.75] vol=4.5x ATR=0.96 |
| Stop hit — per-position SL triggered | 2024-09-13 09:35:00 | 347.19 | 346.51 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-09-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 10:45:00 | 355.75 | 351.10 | 0.00 | ORB-long ORB[349.30,354.15] vol=3.0x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 10:55:00 | 357.92 | 353.00 | 0.00 | T1 1.5R @ 357.92 |
| Target hit | 2024-09-26 15:20:00 | 366.80 | 363.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — BUY (started 2024-10-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:50:00 | 317.00 | 314.41 | 0.00 | ORB-long ORB[311.80,316.50] vol=2.0x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 10:05:00 | 319.17 | 315.72 | 0.00 | T1 1.5R @ 319.17 |
| Stop hit — per-position SL triggered | 2024-10-11 10:20:00 | 317.00 | 316.19 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-10-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 09:40:00 | 312.70 | 310.86 | 0.00 | ORB-long ORB[306.80,311.40] vol=3.0x ATR=1.32 |
| Stop hit — per-position SL triggered | 2024-10-16 10:05:00 | 311.38 | 311.31 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-11-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:30:00 | 273.60 | 275.67 | 0.00 | ORB-short ORB[274.65,278.45] vol=2.9x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:35:00 | 271.76 | 275.15 | 0.00 | T1 1.5R @ 271.76 |
| Target hit | 2024-11-13 10:30:00 | 273.30 | 272.64 | 0.00 | Trail-exit close>VWAP |

### Cycle 26 — SELL (started 2024-11-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 09:30:00 | 269.80 | 272.32 | 0.00 | ORB-short ORB[272.20,274.60] vol=2.3x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:35:00 | 268.21 | 271.46 | 0.00 | T1 1.5R @ 268.21 |
| Stop hit — per-position SL triggered | 2024-11-18 09:50:00 | 269.80 | 270.48 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-12-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 09:55:00 | 301.90 | 298.31 | 0.00 | ORB-long ORB[295.70,298.65] vol=2.1x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-11 10:00:00 | 303.64 | 299.51 | 0.00 | T1 1.5R @ 303.64 |
| Target hit | 2024-12-11 12:30:00 | 303.55 | 303.68 | 0.00 | Trail-exit close<VWAP |

### Cycle 28 — SELL (started 2024-12-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:40:00 | 292.35 | 293.73 | 0.00 | ORB-short ORB[294.15,298.25] vol=1.9x ATR=1.12 |
| Stop hit — per-position SL triggered | 2024-12-13 10:55:00 | 293.47 | 293.59 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-12-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 10:00:00 | 277.75 | 278.54 | 0.00 | ORB-short ORB[278.00,279.80] vol=2.3x ATR=0.73 |
| Stop hit — per-position SL triggered | 2024-12-27 10:30:00 | 278.48 | 278.40 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-12-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-30 09:35:00 | 277.75 | 279.02 | 0.00 | ORB-short ORB[277.90,280.90] vol=2.4x ATR=1.01 |
| Stop hit — per-position SL triggered | 2024-12-30 09:40:00 | 278.76 | 278.96 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-01-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 09:30:00 | 261.15 | 264.50 | 0.00 | ORB-short ORB[263.00,266.80] vol=2.4x ATR=1.42 |
| Stop hit — per-position SL triggered | 2025-01-15 09:35:00 | 262.57 | 264.33 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-01-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 11:00:00 | 269.10 | 273.05 | 0.00 | ORB-short ORB[274.15,276.35] vol=1.7x ATR=0.99 |
| Stop hit — per-position SL triggered | 2025-01-21 11:15:00 | 270.09 | 272.91 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-01-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 09:35:00 | 265.80 | 263.64 | 0.00 | ORB-long ORB[261.20,265.00] vol=2.7x ATR=1.34 |
| Stop hit — per-position SL triggered | 2025-01-23 09:40:00 | 264.46 | 263.75 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-01-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 09:50:00 | 262.70 | 265.14 | 0.00 | ORB-short ORB[264.50,267.65] vol=1.7x ATR=1.20 |
| Stop hit — per-position SL triggered | 2025-01-24 10:25:00 | 263.90 | 264.44 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-01-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 09:35:00 | 261.85 | 259.44 | 0.00 | ORB-long ORB[257.20,260.75] vol=2.3x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 11:05:00 | 263.83 | 261.62 | 0.00 | T1 1.5R @ 263.83 |
| Stop hit — per-position SL triggered | 2025-01-31 11:35:00 | 261.85 | 262.21 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-02-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 10:00:00 | 228.20 | 230.17 | 0.00 | ORB-short ORB[229.30,232.30] vol=1.8x ATR=0.95 |
| Stop hit — per-position SL triggered | 2025-02-10 10:25:00 | 229.15 | 229.83 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 09:30:00 | 240.75 | 242.23 | 0.00 | ORB-short ORB[241.60,244.37] vol=2.2x ATR=0.95 |
| Stop hit — per-position SL triggered | 2025-04-23 09:45:00 | 241.70 | 241.83 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-04-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-24 09:45:00 | 239.82 | 241.95 | 0.00 | ORB-short ORB[241.00,243.99] vol=2.2x ATR=1.20 |
| Stop hit — per-position SL triggered | 2025-04-24 09:50:00 | 241.02 | 241.87 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 09:30:00 | 230.35 | 228.06 | 0.00 | ORB-long ORB[226.20,229.44] vol=1.9x ATR=1.20 |
| Stop hit — per-position SL triggered | 2025-04-28 09:35:00 | 229.15 | 228.34 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-07 11:15:00 | 221.42 | 219.02 | 0.00 | ORB-long ORB[217.61,220.85] vol=2.0x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 11:40:00 | 223.06 | 219.38 | 0.00 | T1 1.5R @ 223.06 |
| Stop hit — per-position SL triggered | 2025-05-07 13:35:00 | 221.42 | 219.98 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-16 10:10:00 | 336.85 | 2024-05-16 10:30:00 | 338.74 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-05-16 10:10:00 | 336.85 | 2024-05-16 10:35:00 | 336.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-22 09:35:00 | 366.50 | 2024-05-22 09:40:00 | 362.79 | PARTIAL | 0.50 | 1.01% |
| SELL | retest1 | 2024-05-22 09:35:00 | 366.50 | 2024-05-22 09:55:00 | 366.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-27 09:45:00 | 366.13 | 2024-05-27 10:05:00 | 368.34 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest1 | 2024-05-31 09:45:00 | 348.58 | 2024-05-31 09:55:00 | 345.98 | PARTIAL | 0.50 | 0.75% |
| SELL | retest1 | 2024-05-31 09:45:00 | 348.58 | 2024-05-31 12:00:00 | 347.00 | TARGET_HIT | 0.50 | 0.45% |
| BUY | retest1 | 2024-06-12 10:00:00 | 344.85 | 2024-06-12 10:05:00 | 343.28 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-06-18 09:35:00 | 360.98 | 2024-06-18 10:05:00 | 359.12 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2024-06-25 09:35:00 | 351.00 | 2024-06-25 10:00:00 | 349.06 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-06-25 09:35:00 | 351.00 | 2024-06-25 10:15:00 | 351.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-27 10:05:00 | 349.03 | 2024-06-27 10:10:00 | 347.85 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-07-01 11:15:00 | 348.00 | 2024-07-01 11:20:00 | 346.80 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-07-04 09:35:00 | 359.50 | 2024-07-04 09:40:00 | 361.79 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-07-04 09:35:00 | 359.50 | 2024-07-04 10:30:00 | 361.38 | TARGET_HIT | 0.50 | 0.52% |
| SELL | retest1 | 2024-07-23 09:30:00 | 362.23 | 2024-07-23 09:35:00 | 363.65 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-07-31 09:40:00 | 376.95 | 2024-07-31 13:45:00 | 378.85 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-08-06 09:30:00 | 347.30 | 2024-08-06 09:35:00 | 349.13 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2024-08-14 09:30:00 | 329.75 | 2024-08-14 09:35:00 | 327.97 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-08-14 09:30:00 | 329.75 | 2024-08-14 09:45:00 | 329.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-16 09:35:00 | 332.70 | 2024-08-16 09:55:00 | 331.48 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-08-21 09:30:00 | 332.53 | 2024-08-21 09:35:00 | 331.49 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-08-22 09:45:00 | 329.00 | 2024-08-22 15:20:00 | 327.65 | TARGET_HIT | 1.00 | 0.41% |
| BUY | retest1 | 2024-08-23 09:40:00 | 331.95 | 2024-08-23 09:45:00 | 330.88 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-08-28 09:55:00 | 331.53 | 2024-08-28 10:00:00 | 330.40 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-08-30 09:55:00 | 325.20 | 2024-08-30 10:20:00 | 326.14 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-09-13 09:30:00 | 348.15 | 2024-09-13 09:35:00 | 347.19 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-09-26 10:45:00 | 355.75 | 2024-09-26 10:55:00 | 357.92 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-09-26 10:45:00 | 355.75 | 2024-09-26 15:20:00 | 366.80 | TARGET_HIT | 0.50 | 3.11% |
| BUY | retest1 | 2024-10-11 09:50:00 | 317.00 | 2024-10-11 10:05:00 | 319.17 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-10-11 09:50:00 | 317.00 | 2024-10-11 10:20:00 | 317.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-16 09:40:00 | 312.70 | 2024-10-16 10:05:00 | 311.38 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-11-13 09:30:00 | 273.60 | 2024-11-13 09:35:00 | 271.76 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2024-11-13 09:30:00 | 273.60 | 2024-11-13 10:30:00 | 273.30 | TARGET_HIT | 0.50 | 0.11% |
| SELL | retest1 | 2024-11-18 09:30:00 | 269.80 | 2024-11-18 09:35:00 | 268.21 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-11-18 09:30:00 | 269.80 | 2024-11-18 09:50:00 | 269.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-11 09:55:00 | 301.90 | 2024-12-11 10:00:00 | 303.64 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-12-11 09:55:00 | 301.90 | 2024-12-11 12:30:00 | 303.55 | TARGET_HIT | 0.50 | 0.55% |
| SELL | retest1 | 2024-12-13 10:40:00 | 292.35 | 2024-12-13 10:55:00 | 293.47 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-12-27 10:00:00 | 277.75 | 2024-12-27 10:30:00 | 278.48 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-12-30 09:35:00 | 277.75 | 2024-12-30 09:40:00 | 278.76 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-01-15 09:30:00 | 261.15 | 2025-01-15 09:35:00 | 262.57 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2025-01-21 11:00:00 | 269.10 | 2025-01-21 11:15:00 | 270.09 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-01-23 09:35:00 | 265.80 | 2025-01-23 09:40:00 | 264.46 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2025-01-24 09:50:00 | 262.70 | 2025-01-24 10:25:00 | 263.90 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-01-31 09:35:00 | 261.85 | 2025-01-31 11:05:00 | 263.83 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2025-01-31 09:35:00 | 261.85 | 2025-01-31 11:35:00 | 261.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-10 10:00:00 | 228.20 | 2025-02-10 10:25:00 | 229.15 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-04-23 09:30:00 | 240.75 | 2025-04-23 09:45:00 | 241.70 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-04-24 09:45:00 | 239.82 | 2025-04-24 09:50:00 | 241.02 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-04-28 09:30:00 | 230.35 | 2025-04-28 09:35:00 | 229.15 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2025-05-07 11:15:00 | 221.42 | 2025-05-07 11:40:00 | 223.06 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2025-05-07 11:15:00 | 221.42 | 2025-05-07 13:35:00 | 221.42 | STOP_HIT | 0.50 | 0.00% |
