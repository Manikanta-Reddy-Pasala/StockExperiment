# Aster DM Healthcare Ltd. (ASTERDM)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (55354 bars)
- **Last close:** 742.00
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
| ENTRY1 | 64 |
| ENTRY2 | 0 |
| PARTIAL | 22 |
| TARGET_HIT | 9 |
| STOP_HIT | 55 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 86 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 31 / 55
- **Target hits / Stop hits / Partials:** 9 / 55 / 22
- **Avg / median % per leg:** 0.03% / 0.00%
- **Sum % (uncompounded):** 2.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 45 | 16 | 35.6% | 5 | 29 | 11 | 0.01% | 0.4% |
| BUY @ 2nd Alert (retest1) | 45 | 16 | 35.6% | 5 | 29 | 11 | 0.01% | 0.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 41 | 15 | 36.6% | 4 | 26 | 11 | 0.05% | 2.0% |
| SELL @ 2nd Alert (retest1) | 41 | 15 | 36.6% | 4 | 26 | 11 | 0.05% | 2.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 86 | 31 | 36.0% | 9 | 55 | 22 | 0.03% | 2.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-12 11:00:00 | 251.30 | 249.76 | 0.00 | ORB-long ORB[247.60,249.90] vol=6.7x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-12 11:10:00 | 252.80 | 250.55 | 0.00 | T1 1.5R @ 252.80 |
| Target hit | 2023-05-12 14:40:00 | 252.25 | 252.40 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2023-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-22 09:40:00 | 259.60 | 257.60 | 0.00 | ORB-long ORB[254.80,257.55] vol=2.6x ATR=0.89 |
| Stop hit — per-position SL triggered | 2023-05-22 10:30:00 | 258.71 | 258.97 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-05-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-25 09:45:00 | 264.95 | 263.67 | 0.00 | ORB-long ORB[261.45,264.80] vol=2.6x ATR=0.92 |
| Stop hit — per-position SL triggered | 2023-05-25 09:50:00 | 264.03 | 263.70 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-05-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-31 09:30:00 | 257.80 | 258.96 | 0.00 | ORB-short ORB[258.00,261.10] vol=1.5x ATR=0.81 |
| Stop hit — per-position SL triggered | 2023-05-31 09:50:00 | 258.61 | 258.57 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2023-06-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-13 09:50:00 | 278.40 | 279.90 | 0.00 | ORB-short ORB[279.00,281.45] vol=4.8x ATR=0.72 |
| Stop hit — per-position SL triggered | 2023-06-13 09:55:00 | 279.12 | 279.63 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-06-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-14 09:40:00 | 286.05 | 284.74 | 0.00 | ORB-long ORB[278.20,282.45] vol=4.0x ATR=1.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-14 10:00:00 | 288.15 | 285.87 | 0.00 | T1 1.5R @ 288.15 |
| Stop hit — per-position SL triggered | 2023-06-14 10:10:00 | 286.05 | 286.09 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-06-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-19 10:10:00 | 295.00 | 293.08 | 0.00 | ORB-long ORB[288.80,293.00] vol=2.5x ATR=1.42 |
| Stop hit — per-position SL triggered | 2023-06-19 10:45:00 | 293.58 | 293.28 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-06-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-20 09:30:00 | 285.35 | 287.08 | 0.00 | ORB-short ORB[286.30,289.55] vol=2.4x ATR=1.13 |
| Stop hit — per-position SL triggered | 2023-06-20 09:40:00 | 286.48 | 286.84 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-06-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-21 10:35:00 | 291.90 | 289.43 | 0.00 | ORB-long ORB[287.00,290.95] vol=2.6x ATR=0.98 |
| Stop hit — per-position SL triggered | 2023-06-21 14:20:00 | 290.92 | 290.92 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-06-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-26 09:45:00 | 282.65 | 280.93 | 0.00 | ORB-long ORB[278.80,281.80] vol=1.7x ATR=1.09 |
| Stop hit — per-position SL triggered | 2023-06-26 09:55:00 | 281.56 | 281.26 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-06-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-30 09:55:00 | 280.60 | 281.89 | 0.00 | ORB-short ORB[281.10,284.05] vol=3.5x ATR=1.10 |
| Stop hit — per-position SL triggered | 2023-06-30 10:15:00 | 281.70 | 281.77 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-07-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 10:00:00 | 284.55 | 282.98 | 0.00 | ORB-long ORB[281.05,282.85] vol=3.1x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-03 10:05:00 | 285.84 | 283.48 | 0.00 | T1 1.5R @ 285.84 |
| Target hit | 2023-07-03 13:50:00 | 286.45 | 286.55 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — SELL (started 2023-07-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-06 10:55:00 | 312.40 | 313.63 | 0.00 | ORB-short ORB[313.85,318.45] vol=2.7x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-06 12:45:00 | 310.43 | 313.02 | 0.00 | T1 1.5R @ 310.43 |
| Target hit | 2023-07-06 15:20:00 | 310.80 | 312.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2023-07-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 09:50:00 | 319.95 | 316.47 | 0.00 | ORB-long ORB[311.90,316.45] vol=6.4x ATR=1.33 |
| Stop hit — per-position SL triggered | 2023-07-11 09:55:00 | 318.62 | 317.12 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-07-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-12 10:25:00 | 311.40 | 312.97 | 0.00 | ORB-short ORB[312.00,315.70] vol=2.4x ATR=0.99 |
| Stop hit — per-position SL triggered | 2023-07-12 10:30:00 | 312.39 | 312.41 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-07-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-18 11:10:00 | 310.50 | 312.22 | 0.00 | ORB-short ORB[311.20,314.90] vol=1.8x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-18 11:45:00 | 309.33 | 311.87 | 0.00 | T1 1.5R @ 309.33 |
| Stop hit — per-position SL triggered | 2023-07-18 11:55:00 | 310.50 | 311.84 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-07-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-20 10:10:00 | 314.00 | 310.65 | 0.00 | ORB-long ORB[305.25,309.65] vol=2.6x ATR=1.33 |
| Stop hit — per-position SL triggered | 2023-07-20 10:25:00 | 312.67 | 310.87 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-07-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 10:50:00 | 312.40 | 311.58 | 0.00 | ORB-long ORB[309.25,312.00] vol=3.9x ATR=0.98 |
| Stop hit — per-position SL triggered | 2023-07-26 11:35:00 | 311.42 | 311.71 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-08-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-04 11:00:00 | 320.25 | 317.67 | 0.00 | ORB-long ORB[317.50,320.05] vol=3.7x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-04 11:50:00 | 321.71 | 318.83 | 0.00 | T1 1.5R @ 321.71 |
| Stop hit — per-position SL triggered | 2023-08-04 12:25:00 | 320.25 | 319.33 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-08-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-07 09:55:00 | 320.30 | 321.69 | 0.00 | ORB-short ORB[320.70,323.80] vol=2.2x ATR=1.41 |
| Stop hit — per-position SL triggered | 2023-08-07 10:20:00 | 321.71 | 321.51 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-08-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-08 10:20:00 | 317.80 | 319.50 | 0.00 | ORB-short ORB[318.30,321.80] vol=1.5x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-08 10:55:00 | 316.03 | 318.96 | 0.00 | T1 1.5R @ 316.03 |
| Target hit | 2023-08-08 15:20:00 | 312.95 | 316.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2023-08-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-10 10:45:00 | 312.70 | 311.71 | 0.00 | ORB-long ORB[307.50,311.15] vol=2.7x ATR=1.07 |
| Stop hit — per-position SL triggered | 2023-08-10 11:05:00 | 311.63 | 311.72 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-08-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-11 10:45:00 | 308.00 | 312.33 | 0.00 | ORB-short ORB[311.40,314.40] vol=2.5x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-11 13:00:00 | 306.33 | 308.88 | 0.00 | T1 1.5R @ 306.33 |
| Stop hit — per-position SL triggered | 2023-08-11 13:15:00 | 308.00 | 308.83 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-08-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-17 10:55:00 | 308.50 | 309.60 | 0.00 | ORB-short ORB[309.25,313.80] vol=3.3x ATR=0.84 |
| Stop hit — per-position SL triggered | 2023-08-17 11:50:00 | 309.34 | 309.35 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-08-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-24 09:30:00 | 313.15 | 314.54 | 0.00 | ORB-short ORB[313.95,316.65] vol=1.6x ATR=0.93 |
| Stop hit — per-position SL triggered | 2023-08-24 09:50:00 | 314.08 | 313.68 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-08-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 09:35:00 | 331.45 | 329.86 | 0.00 | ORB-long ORB[325.65,330.15] vol=3.6x ATR=1.31 |
| Stop hit — per-position SL triggered | 2023-08-30 09:40:00 | 330.14 | 329.72 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-09-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 10:00:00 | 341.00 | 337.58 | 0.00 | ORB-long ORB[331.45,335.95] vol=10.4x ATR=1.80 |
| Stop hit — per-position SL triggered | 2023-09-05 10:15:00 | 339.20 | 338.75 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-09-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-07 09:35:00 | 341.25 | 338.29 | 0.00 | ORB-long ORB[334.95,338.45] vol=4.0x ATR=1.72 |
| Stop hit — per-position SL triggered | 2023-09-07 09:55:00 | 339.53 | 339.68 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2023-09-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 10:05:00 | 343.75 | 342.35 | 0.00 | ORB-long ORB[340.00,342.95] vol=6.1x ATR=1.31 |
| Stop hit — per-position SL triggered | 2023-09-08 11:15:00 | 342.44 | 342.71 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-09-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 09:35:00 | 334.85 | 336.82 | 0.00 | ORB-short ORB[336.60,341.05] vol=8.9x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 09:55:00 | 332.22 | 336.26 | 0.00 | T1 1.5R @ 332.22 |
| Stop hit — per-position SL triggered | 2023-09-12 10:00:00 | 334.85 | 336.17 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2023-09-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-14 09:40:00 | 338.50 | 336.70 | 0.00 | ORB-long ORB[333.35,338.05] vol=6.5x ATR=1.88 |
| Stop hit — per-position SL triggered | 2023-09-14 09:45:00 | 336.62 | 336.70 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-09-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-20 10:35:00 | 327.30 | 329.83 | 0.00 | ORB-short ORB[329.00,332.95] vol=2.5x ATR=1.73 |
| Stop hit — per-position SL triggered | 2023-09-20 10:50:00 | 329.03 | 329.54 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-09-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-22 10:50:00 | 330.75 | 326.97 | 0.00 | ORB-long ORB[325.25,327.50] vol=2.8x ATR=1.16 |
| Stop hit — per-position SL triggered | 2023-09-22 11:25:00 | 329.59 | 327.56 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-09-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-27 11:00:00 | 327.80 | 327.23 | 0.00 | ORB-long ORB[324.55,327.30] vol=1.6x ATR=0.74 |
| Stop hit — per-position SL triggered | 2023-09-27 11:35:00 | 327.06 | 327.26 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-09-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 10:30:00 | 328.90 | 327.54 | 0.00 | ORB-long ORB[324.30,328.00] vol=4.1x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-29 11:25:00 | 330.31 | 328.27 | 0.00 | T1 1.5R @ 330.31 |
| Stop hit — per-position SL triggered | 2023-09-29 14:30:00 | 328.90 | 329.44 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-10-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-05 09:40:00 | 330.05 | 330.76 | 0.00 | ORB-short ORB[330.25,333.05] vol=1.6x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-05 10:10:00 | 328.51 | 330.41 | 0.00 | T1 1.5R @ 328.51 |
| Stop hit — per-position SL triggered | 2023-10-05 11:10:00 | 330.05 | 330.03 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-10-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-27 11:10:00 | 335.30 | 338.17 | 0.00 | ORB-short ORB[335.55,339.00] vol=3.5x ATR=1.75 |
| Stop hit — per-position SL triggered | 2023-10-27 12:00:00 | 337.05 | 337.90 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2023-10-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-30 09:45:00 | 335.95 | 333.15 | 0.00 | ORB-long ORB[330.65,333.70] vol=2.1x ATR=1.57 |
| Stop hit — per-position SL triggered | 2023-10-30 09:50:00 | 334.38 | 333.53 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-11-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 09:30:00 | 336.90 | 336.16 | 0.00 | ORB-long ORB[334.70,336.60] vol=4.2x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-08 09:55:00 | 338.65 | 337.16 | 0.00 | T1 1.5R @ 338.65 |
| Target hit | 2023-11-08 12:10:00 | 338.25 | 338.56 | 0.00 | Trail-exit close<VWAP |

### Cycle 40 — SELL (started 2023-11-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-09 10:35:00 | 334.35 | 335.93 | 0.00 | ORB-short ORB[335.05,337.95] vol=1.7x ATR=0.81 |
| Stop hit — per-position SL triggered | 2023-11-09 10:55:00 | 335.16 | 335.85 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2023-11-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-10 09:55:00 | 330.00 | 332.06 | 0.00 | ORB-short ORB[330.30,333.65] vol=2.3x ATR=1.17 |
| Stop hit — per-position SL triggered | 2023-11-10 10:05:00 | 331.17 | 331.95 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2023-11-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 09:40:00 | 343.50 | 342.07 | 0.00 | ORB-long ORB[338.40,341.00] vol=2.8x ATR=1.40 |
| Stop hit — per-position SL triggered | 2023-11-21 09:55:00 | 342.10 | 342.12 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-11-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-23 11:15:00 | 338.00 | 339.46 | 0.00 | ORB-short ORB[338.60,341.75] vol=1.7x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-23 12:35:00 | 336.78 | 339.17 | 0.00 | T1 1.5R @ 336.78 |
| Stop hit — per-position SL triggered | 2023-11-23 15:15:00 | 338.00 | 338.51 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-11-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-24 09:35:00 | 339.10 | 337.52 | 0.00 | ORB-long ORB[336.30,338.15] vol=2.6x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-24 09:40:00 | 340.99 | 337.88 | 0.00 | T1 1.5R @ 340.99 |
| Stop hit — per-position SL triggered | 2023-11-24 09:50:00 | 339.10 | 338.16 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2023-12-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-07 10:20:00 | 399.35 | 402.32 | 0.00 | ORB-short ORB[399.85,405.80] vol=2.5x ATR=1.59 |
| Stop hit — per-position SL triggered | 2023-12-07 10:25:00 | 400.94 | 402.26 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2023-12-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 10:25:00 | 399.85 | 403.62 | 0.00 | ORB-short ORB[403.65,407.40] vol=1.5x ATR=1.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-08 12:05:00 | 397.75 | 401.87 | 0.00 | T1 1.5R @ 397.75 |
| Stop hit — per-position SL triggered | 2023-12-08 12:20:00 | 399.85 | 401.61 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2023-12-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-13 09:30:00 | 395.90 | 397.93 | 0.00 | ORB-short ORB[396.60,402.30] vol=1.5x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-13 09:35:00 | 393.65 | 396.87 | 0.00 | T1 1.5R @ 393.65 |
| Target hit | 2023-12-13 14:10:00 | 395.55 | 395.06 | 0.00 | Trail-exit close>VWAP |

### Cycle 48 — BUY (started 2023-12-22 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 10:45:00 | 393.60 | 390.67 | 0.00 | ORB-long ORB[388.55,392.65] vol=2.3x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-22 10:50:00 | 395.66 | 391.74 | 0.00 | T1 1.5R @ 395.66 |
| Stop hit — per-position SL triggered | 2023-12-22 12:55:00 | 393.60 | 393.43 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-12-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 09:45:00 | 400.85 | 399.23 | 0.00 | ORB-long ORB[395.25,398.85] vol=2.1x ATR=1.99 |
| Stop hit — per-position SL triggered | 2023-12-26 11:20:00 | 398.86 | 399.93 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-01-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-02 09:35:00 | 410.40 | 408.50 | 0.00 | ORB-long ORB[404.20,408.90] vol=4.7x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-02 09:40:00 | 413.14 | 409.55 | 0.00 | T1 1.5R @ 413.14 |
| Stop hit — per-position SL triggered | 2024-01-02 09:55:00 | 410.40 | 409.90 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-01-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-10 11:00:00 | 406.60 | 408.72 | 0.00 | ORB-short ORB[407.55,410.95] vol=2.5x ATR=0.97 |
| Stop hit — per-position SL triggered | 2024-01-10 11:55:00 | 407.57 | 408.34 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-01-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-15 09:40:00 | 400.40 | 403.65 | 0.00 | ORB-short ORB[406.05,410.20] vol=5.6x ATR=1.96 |
| Stop hit — per-position SL triggered | 2024-01-15 10:05:00 | 402.36 | 402.64 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-01-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-20 11:05:00 | 434.15 | 436.21 | 0.00 | ORB-short ORB[436.10,439.00] vol=1.8x ATR=1.20 |
| Stop hit — per-position SL triggered | 2024-01-20 11:35:00 | 435.35 | 435.94 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-01-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-29 09:40:00 | 426.10 | 429.48 | 0.00 | ORB-short ORB[427.60,433.55] vol=2.0x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-29 10:40:00 | 424.04 | 427.06 | 0.00 | T1 1.5R @ 424.04 |
| Stop hit — per-position SL triggered | 2024-01-29 11:10:00 | 426.10 | 426.89 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-01-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-30 10:20:00 | 431.20 | 427.32 | 0.00 | ORB-long ORB[423.60,428.65] vol=2.4x ATR=1.56 |
| Stop hit — per-position SL triggered | 2024-01-30 11:05:00 | 429.64 | 429.53 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-01-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 09:30:00 | 435.10 | 433.57 | 0.00 | ORB-long ORB[429.10,435.00] vol=1.5x ATR=1.67 |
| Stop hit — per-position SL triggered | 2024-01-31 09:40:00 | 433.43 | 433.78 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-03-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 09:55:00 | 448.10 | 450.85 | 0.00 | ORB-short ORB[449.75,454.95] vol=1.6x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 10:45:00 | 445.33 | 449.40 | 0.00 | T1 1.5R @ 445.33 |
| Target hit | 2024-03-06 15:20:00 | 445.25 | 446.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 58 — SELL (started 2024-03-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 10:50:00 | 427.20 | 428.51 | 0.00 | ORB-short ORB[430.00,432.40] vol=1.6x ATR=1.36 |
| Stop hit — per-position SL triggered | 2024-03-19 11:00:00 | 428.56 | 428.48 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-03-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-22 11:10:00 | 438.10 | 440.56 | 0.00 | ORB-short ORB[438.50,444.00] vol=4.4x ATR=0.95 |
| Stop hit — per-position SL triggered | 2024-03-22 11:15:00 | 439.05 | 440.47 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2024-04-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 09:50:00 | 413.35 | 412.10 | 0.00 | ORB-long ORB[407.00,412.00] vol=1.6x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-03 14:25:00 | 415.58 | 413.34 | 0.00 | T1 1.5R @ 415.58 |
| Target hit | 2024-04-03 15:20:00 | 417.55 | 415.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — BUY (started 2024-05-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-03 09:50:00 | 350.40 | 347.24 | 0.00 | ORB-long ORB[342.90,347.95] vol=2.0x ATR=1.39 |
| Stop hit — per-position SL triggered | 2024-05-03 10:05:00 | 349.01 | 347.79 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2024-05-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-08 11:10:00 | 346.20 | 344.06 | 0.00 | ORB-long ORB[340.20,344.80] vol=2.4x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-08 11:25:00 | 347.56 | 344.54 | 0.00 | T1 1.5R @ 347.56 |
| Target hit | 2024-05-08 15:20:00 | 350.50 | 347.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — BUY (started 2024-05-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-09 09:30:00 | 354.60 | 352.13 | 0.00 | ORB-long ORB[349.70,353.95] vol=2.2x ATR=1.26 |
| Stop hit — per-position SL triggered | 2024-05-09 09:50:00 | 353.34 | 353.61 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2024-05-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-10 09:40:00 | 342.60 | 343.68 | 0.00 | ORB-short ORB[343.10,346.95] vol=2.3x ATR=1.47 |
| Stop hit — per-position SL triggered | 2024-05-10 13:45:00 | 344.07 | 342.74 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-12 11:00:00 | 251.30 | 2023-05-12 11:10:00 | 252.80 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2023-05-12 11:00:00 | 251.30 | 2023-05-12 14:40:00 | 252.25 | TARGET_HIT | 0.50 | 0.38% |
| BUY | retest1 | 2023-05-22 09:40:00 | 259.60 | 2023-05-22 10:30:00 | 258.71 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-05-25 09:45:00 | 264.95 | 2023-05-25 09:50:00 | 264.03 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-05-31 09:30:00 | 257.80 | 2023-05-31 09:50:00 | 258.61 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-06-13 09:50:00 | 278.40 | 2023-06-13 09:55:00 | 279.12 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-06-14 09:40:00 | 286.05 | 2023-06-14 10:00:00 | 288.15 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2023-06-14 09:40:00 | 286.05 | 2023-06-14 10:10:00 | 286.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-19 10:10:00 | 295.00 | 2023-06-19 10:45:00 | 293.58 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2023-06-20 09:30:00 | 285.35 | 2023-06-20 09:40:00 | 286.48 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2023-06-21 10:35:00 | 291.90 | 2023-06-21 14:20:00 | 290.92 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-06-26 09:45:00 | 282.65 | 2023-06-26 09:55:00 | 281.56 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2023-06-30 09:55:00 | 280.60 | 2023-06-30 10:15:00 | 281.70 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2023-07-03 10:00:00 | 284.55 | 2023-07-03 10:05:00 | 285.84 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2023-07-03 10:00:00 | 284.55 | 2023-07-03 13:50:00 | 286.45 | TARGET_HIT | 0.50 | 0.67% |
| SELL | retest1 | 2023-07-06 10:55:00 | 312.40 | 2023-07-06 12:45:00 | 310.43 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2023-07-06 10:55:00 | 312.40 | 2023-07-06 15:20:00 | 310.80 | TARGET_HIT | 0.50 | 0.51% |
| BUY | retest1 | 2023-07-11 09:50:00 | 319.95 | 2023-07-11 09:55:00 | 318.62 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2023-07-12 10:25:00 | 311.40 | 2023-07-12 10:30:00 | 312.39 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-07-18 11:10:00 | 310.50 | 2023-07-18 11:45:00 | 309.33 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2023-07-18 11:10:00 | 310.50 | 2023-07-18 11:55:00 | 310.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-20 10:10:00 | 314.00 | 2023-07-20 10:25:00 | 312.67 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2023-07-26 10:50:00 | 312.40 | 2023-07-26 11:35:00 | 311.42 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-08-04 11:00:00 | 320.25 | 2023-08-04 11:50:00 | 321.71 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2023-08-04 11:00:00 | 320.25 | 2023-08-04 12:25:00 | 320.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-07 09:55:00 | 320.30 | 2023-08-07 10:20:00 | 321.71 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2023-08-08 10:20:00 | 317.80 | 2023-08-08 10:55:00 | 316.03 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2023-08-08 10:20:00 | 317.80 | 2023-08-08 15:20:00 | 312.95 | TARGET_HIT | 0.50 | 1.53% |
| BUY | retest1 | 2023-08-10 10:45:00 | 312.70 | 2023-08-10 11:05:00 | 311.63 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-08-11 10:45:00 | 308.00 | 2023-08-11 13:00:00 | 306.33 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2023-08-11 10:45:00 | 308.00 | 2023-08-11 13:15:00 | 308.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-17 10:55:00 | 308.50 | 2023-08-17 11:50:00 | 309.34 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-08-24 09:30:00 | 313.15 | 2023-08-24 09:50:00 | 314.08 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-08-30 09:35:00 | 331.45 | 2023-08-30 09:40:00 | 330.14 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2023-09-05 10:00:00 | 341.00 | 2023-09-05 10:15:00 | 339.20 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2023-09-07 09:35:00 | 341.25 | 2023-09-07 09:55:00 | 339.53 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2023-09-08 10:05:00 | 343.75 | 2023-09-08 11:15:00 | 342.44 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2023-09-12 09:35:00 | 334.85 | 2023-09-12 09:55:00 | 332.22 | PARTIAL | 0.50 | 0.79% |
| SELL | retest1 | 2023-09-12 09:35:00 | 334.85 | 2023-09-12 10:00:00 | 334.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-14 09:40:00 | 338.50 | 2023-09-14 09:45:00 | 336.62 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2023-09-20 10:35:00 | 327.30 | 2023-09-20 10:50:00 | 329.03 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2023-09-22 10:50:00 | 330.75 | 2023-09-22 11:25:00 | 329.59 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-09-27 11:00:00 | 327.80 | 2023-09-27 11:35:00 | 327.06 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-09-29 10:30:00 | 328.90 | 2023-09-29 11:25:00 | 330.31 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2023-09-29 10:30:00 | 328.90 | 2023-09-29 14:30:00 | 328.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-05 09:40:00 | 330.05 | 2023-10-05 10:10:00 | 328.51 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2023-10-05 09:40:00 | 330.05 | 2023-10-05 11:10:00 | 330.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-27 11:10:00 | 335.30 | 2023-10-27 12:00:00 | 337.05 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2023-10-30 09:45:00 | 335.95 | 2023-10-30 09:50:00 | 334.38 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2023-11-08 09:30:00 | 336.90 | 2023-11-08 09:55:00 | 338.65 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-11-08 09:30:00 | 336.90 | 2023-11-08 12:10:00 | 338.25 | TARGET_HIT | 0.50 | 0.40% |
| SELL | retest1 | 2023-11-09 10:35:00 | 334.35 | 2023-11-09 10:55:00 | 335.16 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-11-10 09:55:00 | 330.00 | 2023-11-10 10:05:00 | 331.17 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-11-21 09:40:00 | 343.50 | 2023-11-21 09:55:00 | 342.10 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2023-11-23 11:15:00 | 338.00 | 2023-11-23 12:35:00 | 336.78 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2023-11-23 11:15:00 | 338.00 | 2023-11-23 15:15:00 | 338.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-24 09:35:00 | 339.10 | 2023-11-24 09:40:00 | 340.99 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2023-11-24 09:35:00 | 339.10 | 2023-11-24 09:50:00 | 339.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-07 10:20:00 | 399.35 | 2023-12-07 10:25:00 | 400.94 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2023-12-08 10:25:00 | 399.85 | 2023-12-08 12:05:00 | 397.75 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2023-12-08 10:25:00 | 399.85 | 2023-12-08 12:20:00 | 399.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-13 09:30:00 | 395.90 | 2023-12-13 09:35:00 | 393.65 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2023-12-13 09:30:00 | 395.90 | 2023-12-13 14:10:00 | 395.55 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2023-12-22 10:45:00 | 393.60 | 2023-12-22 10:50:00 | 395.66 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-12-22 10:45:00 | 393.60 | 2023-12-22 12:55:00 | 393.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-26 09:45:00 | 400.85 | 2023-12-26 11:20:00 | 398.86 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-01-02 09:35:00 | 410.40 | 2024-01-02 09:40:00 | 413.14 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2024-01-02 09:35:00 | 410.40 | 2024-01-02 09:55:00 | 410.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-10 11:00:00 | 406.60 | 2024-01-10 11:55:00 | 407.57 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-01-15 09:40:00 | 400.40 | 2024-01-15 10:05:00 | 402.36 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-01-20 11:05:00 | 434.15 | 2024-01-20 11:35:00 | 435.35 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-01-29 09:40:00 | 426.10 | 2024-01-29 10:40:00 | 424.04 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-01-29 09:40:00 | 426.10 | 2024-01-29 11:10:00 | 426.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-30 10:20:00 | 431.20 | 2024-01-30 11:05:00 | 429.64 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-01-31 09:30:00 | 435.10 | 2024-01-31 09:40:00 | 433.43 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-03-06 09:55:00 | 448.10 | 2024-03-06 10:45:00 | 445.33 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-03-06 09:55:00 | 448.10 | 2024-03-06 15:20:00 | 445.25 | TARGET_HIT | 0.50 | 0.64% |
| SELL | retest1 | 2024-03-19 10:50:00 | 427.20 | 2024-03-19 11:00:00 | 428.56 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-03-22 11:10:00 | 438.10 | 2024-03-22 11:15:00 | 439.05 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-04-03 09:50:00 | 413.35 | 2024-04-03 14:25:00 | 415.58 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-04-03 09:50:00 | 413.35 | 2024-04-03 15:20:00 | 417.55 | TARGET_HIT | 0.50 | 1.02% |
| BUY | retest1 | 2024-05-03 09:50:00 | 350.40 | 2024-05-03 10:05:00 | 349.01 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-05-08 11:10:00 | 346.20 | 2024-05-08 11:25:00 | 347.56 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-05-08 11:10:00 | 346.20 | 2024-05-08 15:20:00 | 350.50 | TARGET_HIT | 0.50 | 1.24% |
| BUY | retest1 | 2024-05-09 09:30:00 | 354.60 | 2024-05-09 09:50:00 | 353.34 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-05-10 09:40:00 | 342.60 | 2024-05-10 13:45:00 | 344.07 | STOP_HIT | 1.00 | -0.43% |
