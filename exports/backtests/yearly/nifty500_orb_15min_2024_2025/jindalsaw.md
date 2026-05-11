# Jindal Saw Ltd. (JINDALSAW)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-01-30 15:25:00 (31996 bars)
- **Last close:** 175.29
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
| ENTRY1 | 39 |
| ENTRY2 | 0 |
| PARTIAL | 19 |
| TARGET_HIT | 7 |
| STOP_HIT | 32 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 58 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 32
- **Target hits / Stop hits / Partials:** 7 / 32 / 19
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 10.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 14 | 46.7% | 4 | 16 | 10 | 0.22% | 6.6% |
| BUY @ 2nd Alert (retest1) | 30 | 14 | 46.7% | 4 | 16 | 10 | 0.22% | 6.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 28 | 12 | 42.9% | 3 | 16 | 9 | 0.14% | 4.0% |
| SELL @ 2nd Alert (retest1) | 28 | 12 | 42.9% | 3 | 16 | 9 | 0.14% | 4.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 58 | 26 | 44.8% | 7 | 32 | 19 | 0.18% | 10.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 11:15:00 | 258.75 | 261.37 | 0.00 | ORB-short ORB[263.75,267.45] vol=1.9x ATR=1.83 |
| Stop hit — per-position SL triggered | 2024-05-13 11:25:00 | 260.58 | 261.31 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 11:00:00 | 271.00 | 273.43 | 0.00 | ORB-short ORB[274.63,276.50] vol=3.6x ATR=1.13 |
| Stop hit — per-position SL triggered | 2024-05-15 11:05:00 | 272.13 | 273.36 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:40:00 | 267.00 | 270.27 | 0.00 | ORB-short ORB[269.50,272.70] vol=2.5x ATR=1.16 |
| Stop hit — per-position SL triggered | 2024-05-22 09:55:00 | 268.16 | 269.36 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 09:35:00 | 270.20 | 271.76 | 0.00 | ORB-short ORB[271.00,274.05] vol=2.3x ATR=1.32 |
| Stop hit — per-position SL triggered | 2024-05-24 09:40:00 | 271.52 | 271.69 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 11:15:00 | 271.90 | 273.32 | 0.00 | ORB-short ORB[272.48,276.08] vol=3.4x ATR=1.25 |
| Stop hit — per-position SL triggered | 2024-05-27 12:30:00 | 273.15 | 273.09 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 09:45:00 | 269.73 | 271.35 | 0.00 | ORB-short ORB[271.50,274.50] vol=2.0x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 09:55:00 | 267.83 | 270.92 | 0.00 | T1 1.5R @ 267.83 |
| Stop hit — per-position SL triggered | 2024-05-31 10:05:00 | 269.73 | 270.19 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 09:35:00 | 276.18 | 278.26 | 0.00 | ORB-short ORB[277.08,281.00] vol=2.8x ATR=1.23 |
| Stop hit — per-position SL triggered | 2024-06-13 10:15:00 | 277.41 | 277.31 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 11:00:00 | 282.48 | 280.93 | 0.00 | ORB-long ORB[278.30,281.70] vol=2.9x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 11:10:00 | 284.08 | 282.57 | 0.00 | T1 1.5R @ 284.08 |
| Target hit | 2024-06-20 12:25:00 | 282.63 | 283.21 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2024-06-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 09:45:00 | 278.15 | 277.10 | 0.00 | ORB-long ORB[274.43,277.88] vol=2.5x ATR=0.99 |
| Stop hit — per-position SL triggered | 2024-06-27 10:20:00 | 277.16 | 277.42 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:30:00 | 274.45 | 273.48 | 0.00 | ORB-long ORB[271.52,274.25] vol=1.5x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 09:40:00 | 275.86 | 273.94 | 0.00 | T1 1.5R @ 275.86 |
| Stop hit — per-position SL triggered | 2024-07-01 09:45:00 | 274.45 | 273.98 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 09:30:00 | 274.50 | 273.65 | 0.00 | ORB-long ORB[272.00,274.40] vol=1.9x ATR=1.03 |
| Stop hit — per-position SL triggered | 2024-07-02 09:40:00 | 273.47 | 273.74 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 11:10:00 | 278.50 | 275.92 | 0.00 | ORB-long ORB[274.20,277.73] vol=4.4x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 12:35:00 | 280.30 | 277.00 | 0.00 | T1 1.5R @ 280.30 |
| Stop hit — per-position SL triggered | 2024-07-05 12:45:00 | 278.50 | 277.04 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 09:30:00 | 286.45 | 284.84 | 0.00 | ORB-long ORB[282.50,285.25] vol=4.0x ATR=1.07 |
| Stop hit — per-position SL triggered | 2024-07-12 09:45:00 | 285.38 | 285.38 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 10:15:00 | 281.13 | 279.55 | 0.00 | ORB-long ORB[276.80,278.95] vol=2.0x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 10:20:00 | 282.67 | 279.96 | 0.00 | T1 1.5R @ 282.67 |
| Target hit | 2024-07-16 12:15:00 | 283.25 | 283.35 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — BUY (started 2024-07-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:40:00 | 282.27 | 280.29 | 0.00 | ORB-long ORB[277.52,280.60] vol=1.7x ATR=1.11 |
| Stop hit — per-position SL triggered | 2024-07-26 09:50:00 | 281.16 | 280.46 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-08-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-13 09:50:00 | 319.90 | 318.12 | 0.00 | ORB-long ORB[314.63,319.08] vol=2.6x ATR=1.46 |
| Stop hit — per-position SL triggered | 2024-08-13 10:00:00 | 318.44 | 318.18 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-08-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 10:45:00 | 326.90 | 324.81 | 0.00 | ORB-long ORB[322.50,326.33] vol=1.6x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 11:05:00 | 328.68 | 325.84 | 0.00 | T1 1.5R @ 328.68 |
| Stop hit — per-position SL triggered | 2024-08-21 11:50:00 | 326.90 | 326.81 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 11:15:00 | 340.78 | 336.41 | 0.00 | ORB-long ORB[331.03,336.13] vol=2.4x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 11:20:00 | 343.42 | 336.84 | 0.00 | T1 1.5R @ 343.42 |
| Target hit | 2024-08-22 15:20:00 | 343.98 | 342.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 343.33 | 345.24 | 0.00 | ORB-short ORB[343.50,348.45] vol=2.9x ATR=1.43 |
| Stop hit — per-position SL triggered | 2024-08-28 09:35:00 | 344.76 | 345.26 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-08-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 10:55:00 | 337.63 | 339.06 | 0.00 | ORB-short ORB[338.03,341.73] vol=1.8x ATR=1.28 |
| Stop hit — per-position SL triggered | 2024-08-30 11:20:00 | 338.91 | 339.01 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-09-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 10:00:00 | 348.10 | 346.70 | 0.00 | ORB-long ORB[344.23,347.63] vol=1.7x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-03 10:10:00 | 350.26 | 347.77 | 0.00 | T1 1.5R @ 350.26 |
| Target hit | 2024-09-03 15:20:00 | 357.83 | 353.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2024-09-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 10:55:00 | 350.28 | 347.54 | 0.00 | ORB-long ORB[342.63,347.73] vol=5.0x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 11:15:00 | 352.26 | 348.09 | 0.00 | T1 1.5R @ 352.26 |
| Stop hit — per-position SL triggered | 2024-09-12 11:25:00 | 350.28 | 348.30 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-09-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 10:45:00 | 357.68 | 354.51 | 0.00 | ORB-long ORB[351.33,356.65] vol=2.9x ATR=1.25 |
| Stop hit — per-position SL triggered | 2024-09-13 11:15:00 | 356.43 | 355.07 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-09-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 10:35:00 | 358.78 | 356.72 | 0.00 | ORB-long ORB[353.73,357.25] vol=1.6x ATR=1.21 |
| Stop hit — per-position SL triggered | 2024-09-18 10:50:00 | 357.57 | 356.99 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 10:15:00 | 354.45 | 355.36 | 0.00 | ORB-short ORB[355.00,359.93] vol=4.5x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:25:00 | 352.51 | 355.13 | 0.00 | T1 1.5R @ 352.51 |
| Stop hit — per-position SL triggered | 2024-09-19 10:30:00 | 354.45 | 355.10 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 09:45:00 | 355.18 | 353.35 | 0.00 | ORB-long ORB[350.60,354.90] vol=1.5x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-20 11:15:00 | 357.59 | 354.95 | 0.00 | T1 1.5R @ 357.59 |
| Stop hit — per-position SL triggered | 2024-09-20 12:10:00 | 355.18 | 355.25 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-09-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 10:40:00 | 367.75 | 364.47 | 0.00 | ORB-long ORB[360.28,365.70] vol=1.8x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 10:50:00 | 369.66 | 364.95 | 0.00 | T1 1.5R @ 369.66 |
| Stop hit — per-position SL triggered | 2024-09-26 11:05:00 | 367.75 | 365.78 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-10-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 11:10:00 | 363.00 | 366.15 | 0.00 | ORB-short ORB[367.05,371.70] vol=1.7x ATR=1.26 |
| Stop hit — per-position SL triggered | 2024-10-10 11:30:00 | 364.26 | 365.85 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-10-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 09:35:00 | 304.50 | 306.88 | 0.00 | ORB-short ORB[305.60,308.75] vol=1.5x ATR=1.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 10:25:00 | 301.96 | 305.38 | 0.00 | T1 1.5R @ 301.96 |
| Stop hit — per-position SL triggered | 2024-10-29 15:05:00 | 304.50 | 302.83 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-11-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-12 10:25:00 | 311.00 | 308.30 | 0.00 | ORB-long ORB[305.15,309.45] vol=1.6x ATR=1.34 |
| Stop hit — per-position SL triggered | 2024-11-12 10:35:00 | 309.66 | 308.40 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-11-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-27 09:35:00 | 305.95 | 308.21 | 0.00 | ORB-short ORB[307.00,310.95] vol=1.7x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-27 10:00:00 | 303.92 | 307.11 | 0.00 | T1 1.5R @ 303.92 |
| Stop hit — per-position SL triggered | 2024-11-27 11:05:00 | 305.95 | 306.64 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-12-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 09:55:00 | 336.85 | 339.36 | 0.00 | ORB-short ORB[337.75,342.80] vol=1.6x ATR=1.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 11:10:00 | 334.29 | 337.69 | 0.00 | T1 1.5R @ 334.29 |
| Target hit | 2024-12-10 15:20:00 | 335.00 | 335.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — BUY (started 2024-12-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 09:30:00 | 330.30 | 327.33 | 0.00 | ORB-long ORB[324.60,328.90] vol=2.4x ATR=1.59 |
| Stop hit — per-position SL triggered | 2024-12-12 09:40:00 | 328.71 | 327.92 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-12-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 09:35:00 | 314.35 | 316.13 | 0.00 | ORB-short ORB[315.40,319.45] vol=2.3x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 10:20:00 | 312.60 | 314.85 | 0.00 | T1 1.5R @ 312.60 |
| Target hit | 2024-12-17 12:30:00 | 312.40 | 310.92 | 0.00 | Trail-exit close>VWAP |

### Cycle 35 — SELL (started 2024-12-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:05:00 | 301.45 | 303.60 | 0.00 | ORB-short ORB[301.80,305.25] vol=1.8x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 10:15:00 | 299.79 | 302.75 | 0.00 | T1 1.5R @ 299.79 |
| Target hit | 2024-12-26 15:20:00 | 297.30 | 298.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — SELL (started 2025-01-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 10:50:00 | 282.00 | 284.85 | 0.00 | ORB-short ORB[283.85,287.80] vol=2.5x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 11:00:00 | 280.18 | 284.06 | 0.00 | T1 1.5R @ 280.18 |
| Stop hit — per-position SL triggered | 2025-01-02 12:10:00 | 282.00 | 282.32 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-01-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 09:45:00 | 252.45 | 255.63 | 0.00 | ORB-short ORB[255.40,258.65] vol=1.5x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 10:00:00 | 250.38 | 253.81 | 0.00 | T1 1.5R @ 250.38 |
| Stop hit — per-position SL triggered | 2025-01-10 10:15:00 | 252.45 | 252.97 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-01-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 11:05:00 | 256.70 | 253.91 | 0.00 | ORB-long ORB[250.00,252.20] vol=1.6x ATR=1.05 |
| Stop hit — per-position SL triggered | 2025-01-23 11:40:00 | 255.65 | 254.14 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 09:35:00 | 268.75 | 271.58 | 0.00 | ORB-short ORB[271.05,273.80] vol=1.8x ATR=1.45 |
| Stop hit — per-position SL triggered | 2025-04-23 09:45:00 | 270.20 | 270.78 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-13 11:15:00 | 258.75 | 2024-05-13 11:25:00 | 260.58 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest1 | 2024-05-15 11:00:00 | 271.00 | 2024-05-15 11:05:00 | 272.13 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-05-22 09:40:00 | 267.00 | 2024-05-22 09:55:00 | 268.16 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-05-24 09:35:00 | 270.20 | 2024-05-24 09:40:00 | 271.52 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-05-27 11:15:00 | 271.90 | 2024-05-27 12:30:00 | 273.15 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-05-31 09:45:00 | 269.73 | 2024-05-31 09:55:00 | 267.83 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2024-05-31 09:45:00 | 269.73 | 2024-05-31 10:05:00 | 269.73 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-13 09:35:00 | 276.18 | 2024-06-13 10:15:00 | 277.41 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-06-20 11:00:00 | 282.48 | 2024-06-20 11:10:00 | 284.08 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-06-20 11:00:00 | 282.48 | 2024-06-20 12:25:00 | 282.63 | TARGET_HIT | 0.50 | 0.05% |
| BUY | retest1 | 2024-06-27 09:45:00 | 278.15 | 2024-06-27 10:20:00 | 277.16 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-07-01 09:30:00 | 274.45 | 2024-07-01 09:40:00 | 275.86 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-07-01 09:30:00 | 274.45 | 2024-07-01 09:45:00 | 274.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-02 09:30:00 | 274.50 | 2024-07-02 09:40:00 | 273.47 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-07-05 11:10:00 | 278.50 | 2024-07-05 12:35:00 | 280.30 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-07-05 11:10:00 | 278.50 | 2024-07-05 12:45:00 | 278.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-12 09:30:00 | 286.45 | 2024-07-12 09:45:00 | 285.38 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-07-16 10:15:00 | 281.13 | 2024-07-16 10:20:00 | 282.67 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-07-16 10:15:00 | 281.13 | 2024-07-16 12:15:00 | 283.25 | TARGET_HIT | 0.50 | 0.75% |
| BUY | retest1 | 2024-07-26 09:40:00 | 282.27 | 2024-07-26 09:50:00 | 281.16 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-08-13 09:50:00 | 319.90 | 2024-08-13 10:00:00 | 318.44 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-08-21 10:45:00 | 326.90 | 2024-08-21 11:05:00 | 328.68 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-08-21 10:45:00 | 326.90 | 2024-08-21 11:50:00 | 326.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-22 11:15:00 | 340.78 | 2024-08-22 11:20:00 | 343.42 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2024-08-22 11:15:00 | 340.78 | 2024-08-22 15:20:00 | 343.98 | TARGET_HIT | 0.50 | 0.94% |
| SELL | retest1 | 2024-08-28 09:30:00 | 343.33 | 2024-08-28 09:35:00 | 344.76 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-08-30 10:55:00 | 337.63 | 2024-08-30 11:20:00 | 338.91 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-09-03 10:00:00 | 348.10 | 2024-09-03 10:10:00 | 350.26 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-09-03 10:00:00 | 348.10 | 2024-09-03 15:20:00 | 357.83 | TARGET_HIT | 0.50 | 2.80% |
| BUY | retest1 | 2024-09-12 10:55:00 | 350.28 | 2024-09-12 11:15:00 | 352.26 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-09-12 10:55:00 | 350.28 | 2024-09-12 11:25:00 | 350.28 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-13 10:45:00 | 357.68 | 2024-09-13 11:15:00 | 356.43 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-09-18 10:35:00 | 358.78 | 2024-09-18 10:50:00 | 357.57 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-09-19 10:15:00 | 354.45 | 2024-09-19 10:25:00 | 352.51 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-09-19 10:15:00 | 354.45 | 2024-09-19 10:30:00 | 354.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-20 09:45:00 | 355.18 | 2024-09-20 11:15:00 | 357.59 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2024-09-20 09:45:00 | 355.18 | 2024-09-20 12:10:00 | 355.18 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-26 10:40:00 | 367.75 | 2024-09-26 10:50:00 | 369.66 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-09-26 10:40:00 | 367.75 | 2024-09-26 11:05:00 | 367.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-10 11:10:00 | 363.00 | 2024-10-10 11:30:00 | 364.26 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-10-29 09:35:00 | 304.50 | 2024-10-29 10:25:00 | 301.96 | PARTIAL | 0.50 | 0.84% |
| SELL | retest1 | 2024-10-29 09:35:00 | 304.50 | 2024-10-29 15:05:00 | 304.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-12 10:25:00 | 311.00 | 2024-11-12 10:35:00 | 309.66 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-11-27 09:35:00 | 305.95 | 2024-11-27 10:00:00 | 303.92 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-11-27 09:35:00 | 305.95 | 2024-11-27 11:05:00 | 305.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-10 09:55:00 | 336.85 | 2024-12-10 11:10:00 | 334.29 | PARTIAL | 0.50 | 0.76% |
| SELL | retest1 | 2024-12-10 09:55:00 | 336.85 | 2024-12-10 15:20:00 | 335.00 | TARGET_HIT | 0.50 | 0.55% |
| BUY | retest1 | 2024-12-12 09:30:00 | 330.30 | 2024-12-12 09:40:00 | 328.71 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-12-17 09:35:00 | 314.35 | 2024-12-17 10:20:00 | 312.60 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-12-17 09:35:00 | 314.35 | 2024-12-17 12:30:00 | 312.40 | TARGET_HIT | 0.50 | 0.62% |
| SELL | retest1 | 2024-12-26 10:05:00 | 301.45 | 2024-12-26 10:15:00 | 299.79 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-12-26 10:05:00 | 301.45 | 2024-12-26 15:20:00 | 297.30 | TARGET_HIT | 0.50 | 1.38% |
| SELL | retest1 | 2025-01-02 10:50:00 | 282.00 | 2025-01-02 11:00:00 | 280.18 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2025-01-02 10:50:00 | 282.00 | 2025-01-02 12:10:00 | 282.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-10 09:45:00 | 252.45 | 2025-01-10 10:00:00 | 250.38 | PARTIAL | 0.50 | 0.82% |
| SELL | retest1 | 2025-01-10 09:45:00 | 252.45 | 2025-01-10 10:15:00 | 252.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-23 11:05:00 | 256.70 | 2025-01-23 11:40:00 | 255.65 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-04-23 09:35:00 | 268.75 | 2025-04-23 09:45:00 | 270.20 | STOP_HIT | 1.00 | -0.54% |
