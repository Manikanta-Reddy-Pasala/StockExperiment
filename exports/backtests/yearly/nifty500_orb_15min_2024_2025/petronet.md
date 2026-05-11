# Petronet LNG Ltd. (PETRONET)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
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
| ENTRY1 | 80 |
| ENTRY2 | 0 |
| PARTIAL | 31 |
| TARGET_HIT | 17 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 111 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 48 / 63
- **Target hits / Stop hits / Partials:** 17 / 63 / 31
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 12.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 57 | 20 | 35.1% | 7 | 37 | 13 | 0.02% | 1.2% |
| BUY @ 2nd Alert (retest1) | 57 | 20 | 35.1% | 7 | 37 | 13 | 0.02% | 1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 54 | 28 | 51.9% | 10 | 26 | 18 | 0.21% | 11.2% |
| SELL @ 2nd Alert (retest1) | 54 | 28 | 51.9% | 10 | 26 | 18 | 0.21% | 11.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 111 | 48 | 43.2% | 17 | 63 | 31 | 0.11% | 12.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 11:05:00 | 293.75 | 294.60 | 0.00 | ORB-short ORB[295.85,299.55] vol=1.9x ATR=1.61 |
| Stop hit — per-position SL triggered | 2024-05-13 11:15:00 | 295.36 | 294.62 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 10:05:00 | 311.00 | 308.51 | 0.00 | ORB-long ORB[307.00,309.65] vol=1.8x ATR=1.08 |
| Stop hit — per-position SL triggered | 2024-05-16 10:15:00 | 309.92 | 308.72 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 09:40:00 | 301.90 | 303.09 | 0.00 | ORB-short ORB[302.00,306.10] vol=1.5x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 10:15:00 | 299.99 | 302.49 | 0.00 | T1 1.5R @ 299.99 |
| Stop hit — per-position SL triggered | 2024-05-27 11:40:00 | 301.90 | 301.66 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 11:10:00 | 299.30 | 300.25 | 0.00 | ORB-short ORB[299.90,302.50] vol=2.7x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 11:35:00 | 298.11 | 299.87 | 0.00 | T1 1.5R @ 298.11 |
| Target hit | 2024-05-28 15:20:00 | 296.05 | 298.27 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2024-05-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 09:50:00 | 294.55 | 296.46 | 0.00 | ORB-short ORB[295.95,299.10] vol=2.4x ATR=1.01 |
| Stop hit — per-position SL triggered | 2024-05-30 10:25:00 | 295.56 | 296.08 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 11:15:00 | 292.10 | 293.74 | 0.00 | ORB-short ORB[293.00,296.85] vol=2.6x ATR=0.93 |
| Stop hit — per-position SL triggered | 2024-05-31 11:40:00 | 293.03 | 293.58 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 09:30:00 | 309.00 | 307.02 | 0.00 | ORB-long ORB[304.15,308.50] vol=3.7x ATR=1.29 |
| Stop hit — per-position SL triggered | 2024-06-11 09:35:00 | 307.71 | 307.33 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 11:15:00 | 320.65 | 323.11 | 0.00 | ORB-short ORB[320.90,325.50] vol=1.7x ATR=0.84 |
| Stop hit — per-position SL triggered | 2024-06-13 11:30:00 | 321.49 | 322.94 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:35:00 | 326.20 | 324.96 | 0.00 | ORB-long ORB[321.55,325.50] vol=5.6x ATR=1.09 |
| Stop hit — per-position SL triggered | 2024-06-14 09:50:00 | 325.11 | 325.22 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 09:50:00 | 318.35 | 316.80 | 0.00 | ORB-long ORB[313.60,317.00] vol=1.9x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-21 10:30:00 | 319.86 | 318.10 | 0.00 | T1 1.5R @ 319.86 |
| Target hit | 2024-06-21 12:45:00 | 319.60 | 319.76 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2024-06-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 09:40:00 | 330.45 | 327.20 | 0.00 | ORB-long ORB[324.50,327.30] vol=3.3x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 09:45:00 | 332.36 | 329.64 | 0.00 | T1 1.5R @ 332.36 |
| Stop hit — per-position SL triggered | 2024-06-25 09:50:00 | 330.45 | 329.58 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-27 10:10:00 | 319.00 | 317.98 | 0.00 | ORB-long ORB[315.40,317.95] vol=4.2x ATR=0.83 |
| Stop hit — per-position SL triggered | 2024-06-27 10:40:00 | 318.17 | 318.64 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 09:30:00 | 337.05 | 335.52 | 0.00 | ORB-long ORB[333.85,336.55] vol=2.2x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 09:50:00 | 338.46 | 336.93 | 0.00 | T1 1.5R @ 338.46 |
| Target hit | 2024-07-02 10:25:00 | 338.00 | 338.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — BUY (started 2024-07-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 09:50:00 | 336.50 | 335.43 | 0.00 | ORB-long ORB[332.10,336.15] vol=1.7x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 10:15:00 | 338.25 | 336.06 | 0.00 | T1 1.5R @ 338.25 |
| Stop hit — per-position SL triggered | 2024-07-05 10:35:00 | 336.50 | 336.24 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 09:55:00 | 334.15 | 335.88 | 0.00 | ORB-short ORB[335.45,338.90] vol=1.9x ATR=1.05 |
| Stop hit — per-position SL triggered | 2024-07-08 10:00:00 | 335.20 | 335.80 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 11:15:00 | 337.20 | 335.45 | 0.00 | ORB-long ORB[332.70,336.40] vol=5.7x ATR=0.93 |
| Stop hit — per-position SL triggered | 2024-07-11 11:30:00 | 336.27 | 335.54 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 10:00:00 | 342.00 | 339.14 | 0.00 | ORB-long ORB[337.35,340.80] vol=1.8x ATR=1.46 |
| Stop hit — per-position SL triggered | 2024-07-12 10:15:00 | 340.54 | 339.51 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 09:55:00 | 343.60 | 341.28 | 0.00 | ORB-long ORB[338.25,342.40] vol=1.9x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 10:05:00 | 345.79 | 342.98 | 0.00 | T1 1.5R @ 345.79 |
| Target hit | 2024-07-15 15:20:00 | 349.90 | 347.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — BUY (started 2024-07-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 11:10:00 | 353.70 | 351.04 | 0.00 | ORB-long ORB[348.65,353.00] vol=2.6x ATR=0.97 |
| Stop hit — per-position SL triggered | 2024-07-16 11:50:00 | 352.73 | 351.28 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 09:35:00 | 347.10 | 345.21 | 0.00 | ORB-long ORB[342.00,345.85] vol=2.9x ATR=1.38 |
| Stop hit — per-position SL triggered | 2024-07-23 09:40:00 | 345.72 | 345.48 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 10:50:00 | 339.15 | 336.84 | 0.00 | ORB-long ORB[333.60,337.55] vol=4.2x ATR=1.56 |
| Stop hit — per-position SL triggered | 2024-07-24 10:55:00 | 337.59 | 336.91 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-07-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-29 09:30:00 | 368.70 | 370.80 | 0.00 | ORB-short ORB[369.55,374.40] vol=1.6x ATR=1.28 |
| Stop hit — per-position SL triggered | 2024-07-29 09:50:00 | 369.98 | 370.28 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-07 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:25:00 | 365.85 | 362.12 | 0.00 | ORB-long ORB[358.65,361.15] vol=2.3x ATR=1.55 |
| Stop hit — per-position SL triggered | 2024-08-07 10:45:00 | 364.30 | 363.06 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 09:45:00 | 369.75 | 368.01 | 0.00 | ORB-long ORB[364.85,366.95] vol=4.0x ATR=1.20 |
| Stop hit — per-position SL triggered | 2024-08-09 10:05:00 | 368.55 | 368.52 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-08-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:45:00 | 371.90 | 370.13 | 0.00 | ORB-long ORB[366.10,369.35] vol=1.7x ATR=1.41 |
| Stop hit — per-position SL triggered | 2024-08-16 09:50:00 | 370.49 | 370.31 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:30:00 | 379.30 | 377.86 | 0.00 | ORB-long ORB[375.85,379.05] vol=2.0x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 09:40:00 | 380.66 | 379.32 | 0.00 | T1 1.5R @ 380.66 |
| Stop hit — per-position SL triggered | 2024-08-20 09:45:00 | 379.30 | 379.46 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 10:15:00 | 375.25 | 377.87 | 0.00 | ORB-short ORB[379.35,381.00] vol=2.1x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 12:10:00 | 373.94 | 376.62 | 0.00 | T1 1.5R @ 373.94 |
| Target hit | 2024-08-23 15:20:00 | 370.55 | 374.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2024-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 10:15:00 | 370.00 | 370.32 | 0.00 | ORB-short ORB[370.20,372.80] vol=1.8x ATR=0.75 |
| Stop hit — per-position SL triggered | 2024-08-26 10:25:00 | 370.75 | 370.53 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-08-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 09:45:00 | 367.75 | 369.23 | 0.00 | ORB-short ORB[368.85,371.40] vol=3.1x ATR=0.72 |
| Stop hit — per-position SL triggered | 2024-08-27 09:50:00 | 368.47 | 368.83 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 10:45:00 | 369.40 | 368.23 | 0.00 | ORB-long ORB[365.00,368.20] vol=3.2x ATR=0.90 |
| Stop hit — per-position SL triggered | 2024-09-05 11:40:00 | 368.50 | 368.60 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:10:00 | 359.90 | 362.98 | 0.00 | ORB-short ORB[365.15,368.45] vol=1.8x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 10:20:00 | 358.18 | 361.83 | 0.00 | T1 1.5R @ 358.18 |
| Stop hit — per-position SL triggered | 2024-09-06 10:25:00 | 359.90 | 361.74 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-09 10:15:00 | 347.05 | 348.87 | 0.00 | ORB-short ORB[348.15,352.40] vol=4.1x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 10:30:00 | 344.79 | 348.39 | 0.00 | T1 1.5R @ 344.79 |
| Target hit | 2024-09-09 15:05:00 | 343.75 | 343.59 | 0.00 | Trail-exit close>VWAP |

### Cycle 33 — SELL (started 2024-09-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 10:00:00 | 342.45 | 343.50 | 0.00 | ORB-short ORB[342.85,345.80] vol=3.3x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 11:30:00 | 340.55 | 342.91 | 0.00 | T1 1.5R @ 340.55 |
| Stop hit — per-position SL triggered | 2024-09-10 12:30:00 | 342.45 | 342.58 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-09-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-11 10:05:00 | 338.20 | 340.64 | 0.00 | ORB-short ORB[340.25,343.10] vol=1.8x ATR=1.11 |
| Stop hit — per-position SL triggered | 2024-09-11 10:15:00 | 339.31 | 340.47 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-09-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-13 11:05:00 | 338.55 | 340.12 | 0.00 | ORB-short ORB[340.30,343.00] vol=2.0x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-13 11:30:00 | 337.53 | 339.72 | 0.00 | T1 1.5R @ 337.53 |
| Target hit | 2024-09-13 15:20:00 | 334.90 | 336.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — SELL (started 2024-09-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 09:55:00 | 333.15 | 334.83 | 0.00 | ORB-short ORB[334.30,338.30] vol=1.5x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 10:05:00 | 331.84 | 334.47 | 0.00 | T1 1.5R @ 331.84 |
| Stop hit — per-position SL triggered | 2024-09-17 11:40:00 | 333.15 | 333.36 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-09-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:50:00 | 329.30 | 331.44 | 0.00 | ORB-short ORB[331.25,333.95] vol=1.8x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:05:00 | 327.74 | 330.38 | 0.00 | T1 1.5R @ 327.74 |
| Target hit | 2024-09-19 15:20:00 | 323.00 | 323.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — BUY (started 2024-09-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 10:20:00 | 330.10 | 329.03 | 0.00 | ORB-long ORB[326.00,327.80] vol=1.6x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 10:35:00 | 331.81 | 329.29 | 0.00 | T1 1.5R @ 331.81 |
| Target hit | 2024-09-23 14:35:00 | 330.75 | 330.83 | 0.00 | Trail-exit close<VWAP |

### Cycle 39 — SELL (started 2024-09-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 10:20:00 | 327.45 | 327.58 | 0.00 | ORB-short ORB[328.45,332.40] vol=9.3x ATR=0.88 |
| Stop hit — per-position SL triggered | 2024-09-26 11:05:00 | 328.33 | 327.57 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-10-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 09:35:00 | 344.70 | 342.58 | 0.00 | ORB-long ORB[340.65,343.40] vol=1.9x ATR=1.11 |
| Stop hit — per-position SL triggered | 2024-10-01 09:50:00 | 343.59 | 343.37 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-10-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:05:00 | 354.45 | 358.44 | 0.00 | ORB-short ORB[358.10,361.80] vol=1.6x ATR=1.50 |
| Stop hit — per-position SL triggered | 2024-10-07 10:10:00 | 355.95 | 357.54 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-10-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 09:50:00 | 349.35 | 346.75 | 0.00 | ORB-long ORB[344.55,347.70] vol=1.8x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 10:25:00 | 351.93 | 348.10 | 0.00 | T1 1.5R @ 351.93 |
| Stop hit — per-position SL triggered | 2024-10-08 12:05:00 | 349.35 | 349.28 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-10-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 10:30:00 | 357.30 | 356.67 | 0.00 | ORB-long ORB[353.05,356.60] vol=1.5x ATR=1.07 |
| Stop hit — per-position SL triggered | 2024-10-10 10:35:00 | 356.23 | 356.66 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-10-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 10:55:00 | 353.25 | 352.55 | 0.00 | ORB-long ORB[350.30,352.90] vol=1.5x ATR=0.71 |
| Stop hit — per-position SL triggered | 2024-10-11 11:05:00 | 352.54 | 352.59 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-10-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-22 09:45:00 | 349.85 | 348.04 | 0.00 | ORB-long ORB[345.60,348.65] vol=3.1x ATR=1.23 |
| Stop hit — per-position SL triggered | 2024-10-22 10:10:00 | 348.62 | 348.48 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-11-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-05 11:10:00 | 329.60 | 330.73 | 0.00 | ORB-short ORB[330.00,333.55] vol=1.8x ATR=0.78 |
| Stop hit — per-position SL triggered | 2024-11-05 11:15:00 | 330.38 | 330.71 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-11-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-07 09:55:00 | 348.55 | 347.25 | 0.00 | ORB-long ORB[343.80,347.95] vol=3.4x ATR=1.25 |
| Stop hit — per-position SL triggered | 2024-11-07 10:20:00 | 347.30 | 347.42 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-11-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 09:45:00 | 310.85 | 310.95 | 0.00 | ORB-short ORB[311.10,315.05] vol=1.5x ATR=1.18 |
| Stop hit — per-position SL triggered | 2024-11-18 09:50:00 | 312.03 | 310.99 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-11-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 09:50:00 | 330.70 | 329.43 | 0.00 | ORB-long ORB[327.00,329.75] vol=1.9x ATR=1.07 |
| Stop hit — per-position SL triggered | 2024-11-29 10:05:00 | 329.63 | 329.76 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-12-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:30:00 | 342.30 | 341.17 | 0.00 | ORB-long ORB[338.45,342.15] vol=1.9x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 10:10:00 | 343.35 | 341.93 | 0.00 | T1 1.5R @ 343.35 |
| Stop hit — per-position SL triggered | 2024-12-04 10:25:00 | 342.30 | 342.26 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-12-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 10:25:00 | 337.05 | 335.61 | 0.00 | ORB-long ORB[333.40,336.30] vol=2.7x ATR=0.82 |
| Stop hit — per-position SL triggered | 2024-12-09 11:00:00 | 336.23 | 335.97 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-12-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 09:30:00 | 344.55 | 342.91 | 0.00 | ORB-long ORB[340.65,344.35] vol=1.7x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 09:50:00 | 345.80 | 344.00 | 0.00 | T1 1.5R @ 345.80 |
| Stop hit — per-position SL triggered | 2024-12-12 10:30:00 | 344.55 | 344.62 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-12-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 09:30:00 | 336.55 | 337.99 | 0.00 | ORB-short ORB[336.85,341.45] vol=1.9x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 09:35:00 | 335.29 | 337.62 | 0.00 | T1 1.5R @ 335.29 |
| Stop hit — per-position SL triggered | 2024-12-16 11:15:00 | 336.55 | 336.04 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-12-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 09:35:00 | 344.85 | 343.11 | 0.00 | ORB-long ORB[338.65,343.20] vol=3.6x ATR=1.21 |
| Stop hit — per-position SL triggered | 2024-12-24 09:55:00 | 343.64 | 343.81 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-12-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 10:20:00 | 343.40 | 341.85 | 0.00 | ORB-long ORB[339.20,341.55] vol=2.1x ATR=0.97 |
| Stop hit — per-position SL triggered | 2024-12-26 12:10:00 | 342.43 | 342.36 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-12-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 11:10:00 | 342.00 | 343.36 | 0.00 | ORB-short ORB[344.45,346.40] vol=4.0x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 11:55:00 | 340.95 | 343.12 | 0.00 | T1 1.5R @ 340.95 |
| Target hit | 2024-12-27 15:20:00 | 341.15 | 341.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — SELL (started 2025-01-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:55:00 | 325.55 | 326.65 | 0.00 | ORB-short ORB[327.50,330.85] vol=1.8x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:40:00 | 324.03 | 326.06 | 0.00 | T1 1.5R @ 324.03 |
| Stop hit — per-position SL triggered | 2025-01-06 12:05:00 | 325.55 | 325.79 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-01-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-08 09:35:00 | 336.00 | 334.42 | 0.00 | ORB-long ORB[330.90,335.50] vol=1.9x ATR=1.07 |
| Stop hit — per-position SL triggered | 2025-01-08 09:50:00 | 334.93 | 334.84 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-01-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 09:40:00 | 334.65 | 333.08 | 0.00 | ORB-long ORB[329.50,333.00] vol=3.3x ATR=1.04 |
| Stop hit — per-position SL triggered | 2025-01-09 09:45:00 | 333.61 | 333.19 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-01-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-22 10:35:00 | 321.35 | 323.24 | 0.00 | ORB-short ORB[322.90,327.20] vol=4.3x ATR=1.17 |
| Stop hit — per-position SL triggered | 2025-01-22 10:40:00 | 322.52 | 323.22 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 11:15:00 | 303.05 | 301.07 | 0.00 | ORB-long ORB[297.75,302.00] vol=1.5x ATR=0.82 |
| Stop hit — per-position SL triggered | 2025-01-30 11:30:00 | 302.23 | 301.30 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-02-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 11:05:00 | 314.05 | 317.10 | 0.00 | ORB-short ORB[315.55,319.40] vol=1.9x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 11:20:00 | 312.89 | 316.74 | 0.00 | T1 1.5R @ 312.89 |
| Target hit | 2025-02-06 15:20:00 | 310.75 | 312.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — SELL (started 2025-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-13 09:30:00 | 299.40 | 300.47 | 0.00 | ORB-short ORB[299.95,302.85] vol=2.5x ATR=1.15 |
| Stop hit — per-position SL triggered | 2025-02-13 09:40:00 | 300.55 | 300.45 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-02-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 10:05:00 | 303.00 | 303.88 | 0.00 | ORB-short ORB[304.25,306.70] vol=2.1x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-21 10:20:00 | 301.55 | 303.52 | 0.00 | T1 1.5R @ 301.55 |
| Target hit | 2025-02-21 12:40:00 | 302.55 | 302.19 | 0.00 | Trail-exit close>VWAP |

### Cycle 65 — SELL (started 2025-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-06 11:15:00 | 286.15 | 287.69 | 0.00 | ORB-short ORB[288.10,292.35] vol=1.9x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-06 11:25:00 | 284.80 | 287.53 | 0.00 | T1 1.5R @ 284.80 |
| Target hit | 2025-03-06 14:30:00 | 285.30 | 284.65 | 0.00 | Trail-exit close>VWAP |

### Cycle 66 — SELL (started 2025-03-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 11:00:00 | 285.20 | 288.00 | 0.00 | ORB-short ORB[286.40,289.60] vol=1.9x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 11:30:00 | 283.73 | 287.57 | 0.00 | T1 1.5R @ 283.73 |
| Stop hit — per-position SL triggered | 2025-03-12 13:20:00 | 285.20 | 286.47 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-03-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 11:10:00 | 294.55 | 291.90 | 0.00 | ORB-long ORB[287.15,290.90] vol=1.9x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 12:10:00 | 295.68 | 292.50 | 0.00 | T1 1.5R @ 295.68 |
| Target hit | 2025-03-19 15:20:00 | 296.95 | 294.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — BUY (started 2025-03-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 10:05:00 | 298.40 | 297.42 | 0.00 | ORB-long ORB[294.60,298.35] vol=1.6x ATR=0.95 |
| Stop hit — per-position SL triggered | 2025-03-21 10:15:00 | 297.45 | 297.44 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-03-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 10:05:00 | 302.80 | 300.67 | 0.00 | ORB-long ORB[298.00,302.20] vol=1.6x ATR=1.20 |
| Stop hit — per-position SL triggered | 2025-03-26 10:30:00 | 301.60 | 300.98 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-04-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 09:40:00 | 299.40 | 297.32 | 0.00 | ORB-long ORB[293.85,298.30] vol=2.5x ATR=1.25 |
| Stop hit — per-position SL triggered | 2025-04-02 09:45:00 | 298.15 | 297.54 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-03 11:15:00 | 293.85 | 294.63 | 0.00 | ORB-short ORB[295.90,298.40] vol=4.2x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-04-03 11:55:00 | 294.71 | 294.59 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-04-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 10:45:00 | 296.75 | 295.38 | 0.00 | ORB-long ORB[293.15,295.75] vol=1.5x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 11:15:00 | 298.09 | 295.86 | 0.00 | T1 1.5R @ 298.09 |
| Target hit | 2025-04-16 15:20:00 | 301.00 | 298.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — BUY (started 2025-04-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:40:00 | 309.20 | 306.93 | 0.00 | ORB-long ORB[302.25,305.65] vol=3.4x ATR=1.07 |
| Stop hit — per-position SL triggered | 2025-04-21 09:45:00 | 308.13 | 307.07 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2025-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 11:15:00 | 316.80 | 316.44 | 0.00 | ORB-long ORB[312.50,315.55] vol=5.0x ATR=1.11 |
| Stop hit — per-position SL triggered | 2025-04-22 13:45:00 | 315.69 | 316.47 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2025-04-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:45:00 | 309.95 | 313.08 | 0.00 | ORB-short ORB[312.45,315.50] vol=2.3x ATR=1.13 |
| Stop hit — per-position SL triggered | 2025-04-23 12:45:00 | 311.08 | 312.04 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2025-04-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-24 09:55:00 | 312.40 | 313.77 | 0.00 | ORB-short ORB[313.20,317.50] vol=1.9x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-24 10:00:00 | 310.88 | 313.08 | 0.00 | T1 1.5R @ 310.88 |
| Stop hit — per-position SL triggered | 2025-04-24 10:10:00 | 312.40 | 312.93 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2025-04-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:30:00 | 312.10 | 313.70 | 0.00 | ORB-short ORB[313.30,315.35] vol=2.2x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:40:00 | 310.43 | 313.03 | 0.00 | T1 1.5R @ 310.43 |
| Target hit | 2025-04-25 10:40:00 | 309.65 | 309.25 | 0.00 | Trail-exit close>VWAP |

### Cycle 78 — SELL (started 2025-04-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 10:50:00 | 309.40 | 312.19 | 0.00 | ORB-short ORB[311.35,314.95] vol=1.5x ATR=0.91 |
| Stop hit — per-position SL triggered | 2025-04-29 11:20:00 | 310.31 | 311.71 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2025-04-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 09:40:00 | 314.05 | 312.34 | 0.00 | ORB-long ORB[308.70,312.35] vol=2.3x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 09:45:00 | 315.62 | 313.10 | 0.00 | T1 1.5R @ 315.62 |
| Target hit | 2025-04-30 11:55:00 | 315.20 | 315.52 | 0.00 | Trail-exit close<VWAP |

### Cycle 80 — BUY (started 2025-05-02 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-02 10:10:00 | 318.35 | 316.99 | 0.00 | ORB-long ORB[312.45,315.45] vol=1.6x ATR=1.17 |
| Stop hit — per-position SL triggered | 2025-05-02 10:20:00 | 317.18 | 317.07 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-13 11:05:00 | 293.75 | 2024-05-13 11:15:00 | 295.36 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2024-05-16 10:05:00 | 311.00 | 2024-05-16 10:15:00 | 309.92 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-05-27 09:40:00 | 301.90 | 2024-05-27 10:15:00 | 299.99 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-05-27 09:40:00 | 301.90 | 2024-05-27 11:40:00 | 301.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-28 11:10:00 | 299.30 | 2024-05-28 11:35:00 | 298.11 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-05-28 11:10:00 | 299.30 | 2024-05-28 15:20:00 | 296.05 | TARGET_HIT | 0.50 | 1.09% |
| SELL | retest1 | 2024-05-30 09:50:00 | 294.55 | 2024-05-30 10:25:00 | 295.56 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-05-31 11:15:00 | 292.10 | 2024-05-31 11:40:00 | 293.03 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-06-11 09:30:00 | 309.00 | 2024-06-11 09:35:00 | 307.71 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-06-13 11:15:00 | 320.65 | 2024-06-13 11:30:00 | 321.49 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-06-14 09:35:00 | 326.20 | 2024-06-14 09:50:00 | 325.11 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-06-21 09:50:00 | 318.35 | 2024-06-21 10:30:00 | 319.86 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-06-21 09:50:00 | 318.35 | 2024-06-21 12:45:00 | 319.60 | TARGET_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2024-06-25 09:40:00 | 330.45 | 2024-06-25 09:45:00 | 332.36 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-06-25 09:40:00 | 330.45 | 2024-06-25 09:50:00 | 330.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-27 10:10:00 | 319.00 | 2024-06-27 10:40:00 | 318.17 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-07-02 09:30:00 | 337.05 | 2024-07-02 09:50:00 | 338.46 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-07-02 09:30:00 | 337.05 | 2024-07-02 10:25:00 | 338.00 | TARGET_HIT | 0.50 | 0.28% |
| BUY | retest1 | 2024-07-05 09:50:00 | 336.50 | 2024-07-05 10:15:00 | 338.25 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-07-05 09:50:00 | 336.50 | 2024-07-05 10:35:00 | 336.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-08 09:55:00 | 334.15 | 2024-07-08 10:00:00 | 335.20 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-07-11 11:15:00 | 337.20 | 2024-07-11 11:30:00 | 336.27 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-12 10:00:00 | 342.00 | 2024-07-12 10:15:00 | 340.54 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-07-15 09:55:00 | 343.60 | 2024-07-15 10:05:00 | 345.79 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-07-15 09:55:00 | 343.60 | 2024-07-15 15:20:00 | 349.90 | TARGET_HIT | 0.50 | 1.83% |
| BUY | retest1 | 2024-07-16 11:10:00 | 353.70 | 2024-07-16 11:50:00 | 352.73 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-07-23 09:35:00 | 347.10 | 2024-07-23 09:40:00 | 345.72 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-07-24 10:50:00 | 339.15 | 2024-07-24 10:55:00 | 337.59 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-07-29 09:30:00 | 368.70 | 2024-07-29 09:50:00 | 369.98 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-08-07 10:25:00 | 365.85 | 2024-08-07 10:45:00 | 364.30 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-08-09 09:45:00 | 369.75 | 2024-08-09 10:05:00 | 368.55 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-08-16 09:45:00 | 371.90 | 2024-08-16 09:50:00 | 370.49 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-08-20 09:30:00 | 379.30 | 2024-08-20 09:40:00 | 380.66 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-08-20 09:30:00 | 379.30 | 2024-08-20 09:45:00 | 379.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-23 10:15:00 | 375.25 | 2024-08-23 12:10:00 | 373.94 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-08-23 10:15:00 | 375.25 | 2024-08-23 15:20:00 | 370.55 | TARGET_HIT | 0.50 | 1.25% |
| SELL | retest1 | 2024-08-26 10:15:00 | 370.00 | 2024-08-26 10:25:00 | 370.75 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-08-27 09:45:00 | 367.75 | 2024-08-27 09:50:00 | 368.47 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-09-05 10:45:00 | 369.40 | 2024-09-05 11:40:00 | 368.50 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-09-06 10:10:00 | 359.90 | 2024-09-06 10:20:00 | 358.18 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-09-06 10:10:00 | 359.90 | 2024-09-06 10:25:00 | 359.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-09 10:15:00 | 347.05 | 2024-09-09 10:30:00 | 344.79 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2024-09-09 10:15:00 | 347.05 | 2024-09-09 15:05:00 | 343.75 | TARGET_HIT | 0.50 | 0.95% |
| SELL | retest1 | 2024-09-10 10:00:00 | 342.45 | 2024-09-10 11:30:00 | 340.55 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-09-10 10:00:00 | 342.45 | 2024-09-10 12:30:00 | 342.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-11 10:05:00 | 338.20 | 2024-09-11 10:15:00 | 339.31 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-09-13 11:05:00 | 338.55 | 2024-09-13 11:30:00 | 337.53 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-09-13 11:05:00 | 338.55 | 2024-09-13 15:20:00 | 334.90 | TARGET_HIT | 0.50 | 1.08% |
| SELL | retest1 | 2024-09-17 09:55:00 | 333.15 | 2024-09-17 10:05:00 | 331.84 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-09-17 09:55:00 | 333.15 | 2024-09-17 11:40:00 | 333.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-19 09:50:00 | 329.30 | 2024-09-19 10:05:00 | 327.74 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-09-19 09:50:00 | 329.30 | 2024-09-19 15:20:00 | 323.00 | TARGET_HIT | 0.50 | 1.91% |
| BUY | retest1 | 2024-09-23 10:20:00 | 330.10 | 2024-09-23 10:35:00 | 331.81 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-09-23 10:20:00 | 330.10 | 2024-09-23 14:35:00 | 330.75 | TARGET_HIT | 0.50 | 0.20% |
| SELL | retest1 | 2024-09-26 10:20:00 | 327.45 | 2024-09-26 11:05:00 | 328.33 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-10-01 09:35:00 | 344.70 | 2024-10-01 09:50:00 | 343.59 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-10-07 10:05:00 | 354.45 | 2024-10-07 10:10:00 | 355.95 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-10-08 09:50:00 | 349.35 | 2024-10-08 10:25:00 | 351.93 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2024-10-08 09:50:00 | 349.35 | 2024-10-08 12:05:00 | 349.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-10 10:30:00 | 357.30 | 2024-10-10 10:35:00 | 356.23 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-10-11 10:55:00 | 353.25 | 2024-10-11 11:05:00 | 352.54 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-10-22 09:45:00 | 349.85 | 2024-10-22 10:10:00 | 348.62 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-11-05 11:10:00 | 329.60 | 2024-11-05 11:15:00 | 330.38 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-11-07 09:55:00 | 348.55 | 2024-11-07 10:20:00 | 347.30 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-11-18 09:45:00 | 310.85 | 2024-11-18 09:50:00 | 312.03 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-11-29 09:50:00 | 330.70 | 2024-11-29 10:05:00 | 329.63 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-12-04 09:30:00 | 342.30 | 2024-12-04 10:10:00 | 343.35 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2024-12-04 09:30:00 | 342.30 | 2024-12-04 10:25:00 | 342.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-09 10:25:00 | 337.05 | 2024-12-09 11:00:00 | 336.23 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-12-12 09:30:00 | 344.55 | 2024-12-12 09:50:00 | 345.80 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-12-12 09:30:00 | 344.55 | 2024-12-12 10:30:00 | 344.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-16 09:30:00 | 336.55 | 2024-12-16 09:35:00 | 335.29 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-12-16 09:30:00 | 336.55 | 2024-12-16 11:15:00 | 336.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-24 09:35:00 | 344.85 | 2024-12-24 09:55:00 | 343.64 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-12-26 10:20:00 | 343.40 | 2024-12-26 12:10:00 | 342.43 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-12-27 11:10:00 | 342.00 | 2024-12-27 11:55:00 | 340.95 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-12-27 11:10:00 | 342.00 | 2024-12-27 15:20:00 | 341.15 | TARGET_HIT | 0.50 | 0.25% |
| SELL | retest1 | 2025-01-06 10:55:00 | 325.55 | 2025-01-06 11:40:00 | 324.03 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-01-06 10:55:00 | 325.55 | 2025-01-06 12:05:00 | 325.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-08 09:35:00 | 336.00 | 2025-01-08 09:50:00 | 334.93 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-01-09 09:40:00 | 334.65 | 2025-01-09 09:45:00 | 333.61 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-01-22 10:35:00 | 321.35 | 2025-01-22 10:40:00 | 322.52 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-01-30 11:15:00 | 303.05 | 2025-01-30 11:30:00 | 302.23 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-02-06 11:05:00 | 314.05 | 2025-02-06 11:20:00 | 312.89 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-02-06 11:05:00 | 314.05 | 2025-02-06 15:20:00 | 310.75 | TARGET_HIT | 0.50 | 1.05% |
| SELL | retest1 | 2025-02-13 09:30:00 | 299.40 | 2025-02-13 09:40:00 | 300.55 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-02-21 10:05:00 | 303.00 | 2025-02-21 10:20:00 | 301.55 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-02-21 10:05:00 | 303.00 | 2025-02-21 12:40:00 | 302.55 | TARGET_HIT | 0.50 | 0.15% |
| SELL | retest1 | 2025-03-06 11:15:00 | 286.15 | 2025-03-06 11:25:00 | 284.80 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-03-06 11:15:00 | 286.15 | 2025-03-06 14:30:00 | 285.30 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2025-03-12 11:00:00 | 285.20 | 2025-03-12 11:30:00 | 283.73 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-03-12 11:00:00 | 285.20 | 2025-03-12 13:20:00 | 285.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-19 11:10:00 | 294.55 | 2025-03-19 12:10:00 | 295.68 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-03-19 11:10:00 | 294.55 | 2025-03-19 15:20:00 | 296.95 | TARGET_HIT | 0.50 | 0.81% |
| BUY | retest1 | 2025-03-21 10:05:00 | 298.40 | 2025-03-21 10:15:00 | 297.45 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-03-26 10:05:00 | 302.80 | 2025-03-26 10:30:00 | 301.60 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-04-02 09:40:00 | 299.40 | 2025-04-02 09:45:00 | 298.15 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-04-03 11:15:00 | 293.85 | 2025-04-03 11:55:00 | 294.71 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-04-16 10:45:00 | 296.75 | 2025-04-16 11:15:00 | 298.09 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-04-16 10:45:00 | 296.75 | 2025-04-16 15:20:00 | 301.00 | TARGET_HIT | 0.50 | 1.43% |
| BUY | retest1 | 2025-04-21 09:40:00 | 309.20 | 2025-04-21 09:45:00 | 308.13 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-04-22 11:15:00 | 316.80 | 2025-04-22 13:45:00 | 315.69 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-04-23 10:45:00 | 309.95 | 2025-04-23 12:45:00 | 311.08 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-04-24 09:55:00 | 312.40 | 2025-04-24 10:00:00 | 310.88 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-04-24 09:55:00 | 312.40 | 2025-04-24 10:10:00 | 312.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-25 09:30:00 | 312.10 | 2025-04-25 09:40:00 | 310.43 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-04-25 09:30:00 | 312.10 | 2025-04-25 10:40:00 | 309.65 | TARGET_HIT | 0.50 | 0.79% |
| SELL | retest1 | 2025-04-29 10:50:00 | 309.40 | 2025-04-29 11:20:00 | 310.31 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-04-30 09:40:00 | 314.05 | 2025-04-30 09:45:00 | 315.62 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-04-30 09:40:00 | 314.05 | 2025-04-30 11:55:00 | 315.20 | TARGET_HIT | 0.50 | 0.37% |
| BUY | retest1 | 2025-05-02 10:10:00 | 318.35 | 2025-05-02 10:20:00 | 317.18 | STOP_HIT | 1.00 | -0.37% |
