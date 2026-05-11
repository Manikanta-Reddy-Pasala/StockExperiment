# Biocon Ltd. (BIOCON)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-02-04 15:25:00 (13888 bars)
- **Last close:** 368.00
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
| ENTRY1 | 67 |
| ENTRY2 | 0 |
| PARTIAL | 27 |
| TARGET_HIT | 12 |
| STOP_HIT | 55 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 94 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 39 / 55
- **Target hits / Stop hits / Partials:** 12 / 55 / 27
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 8.93%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 61 | 29 | 47.5% | 11 | 32 | 18 | 0.16% | 9.7% |
| BUY @ 2nd Alert (retest1) | 61 | 29 | 47.5% | 11 | 32 | 18 | 0.16% | 9.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 33 | 10 | 30.3% | 1 | 23 | 9 | -0.02% | -0.8% |
| SELL @ 2nd Alert (retest1) | 33 | 10 | 30.3% | 1 | 23 | 9 | -0.02% | -0.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 94 | 39 | 41.5% | 12 | 55 | 27 | 0.09% | 8.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-19 10:00:00 | 343.80 | 341.78 | 0.00 | ORB-long ORB[339.15,342.50] vol=2.1x ATR=1.17 |
| Stop hit — per-position SL triggered | 2025-05-19 10:25:00 | 342.63 | 342.13 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 09:30:00 | 338.50 | 336.78 | 0.00 | ORB-long ORB[333.35,337.50] vol=3.9x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-21 10:25:00 | 340.38 | 337.67 | 0.00 | T1 1.5R @ 340.38 |
| Stop hit — per-position SL triggered | 2025-05-21 11:05:00 | 338.50 | 337.92 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-23 10:20:00 | 329.45 | 332.04 | 0.00 | ORB-short ORB[331.05,335.80] vol=1.8x ATR=1.07 |
| Stop hit — per-position SL triggered | 2025-05-23 10:40:00 | 330.52 | 331.69 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 10:15:00 | 334.50 | 333.40 | 0.00 | ORB-long ORB[332.00,334.45] vol=1.8x ATR=0.89 |
| Stop hit — per-position SL triggered | 2025-05-26 10:25:00 | 333.61 | 333.47 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 09:30:00 | 330.95 | 332.17 | 0.00 | ORB-short ORB[332.00,334.55] vol=6.9x ATR=0.82 |
| Stop hit — per-position SL triggered | 2025-05-27 09:40:00 | 331.77 | 331.95 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-05-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 11:10:00 | 332.85 | 334.50 | 0.00 | ORB-short ORB[334.85,336.45] vol=4.6x ATR=0.65 |
| Stop hit — per-position SL triggered | 2025-05-29 11:25:00 | 333.50 | 334.45 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-05-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 09:45:00 | 334.00 | 335.31 | 0.00 | ORB-short ORB[334.90,337.60] vol=2.4x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 10:15:00 | 332.73 | 334.64 | 0.00 | T1 1.5R @ 332.73 |
| Stop hit — per-position SL triggered | 2025-05-30 11:40:00 | 334.00 | 333.96 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 09:35:00 | 334.30 | 335.80 | 0.00 | ORB-short ORB[335.60,338.50] vol=1.5x ATR=0.97 |
| Stop hit — per-position SL triggered | 2025-06-04 10:30:00 | 335.27 | 335.17 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 10:30:00 | 334.70 | 333.29 | 0.00 | ORB-long ORB[330.90,334.25] vol=2.5x ATR=0.82 |
| Stop hit — per-position SL triggered | 2025-06-09 10:40:00 | 333.88 | 333.34 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 09:30:00 | 344.85 | 342.97 | 0.00 | ORB-long ORB[339.55,343.90] vol=5.5x ATR=1.25 |
| Stop hit — per-position SL triggered | 2025-06-10 09:35:00 | 343.60 | 343.75 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 10:55:00 | 344.15 | 341.07 | 0.00 | ORB-long ORB[338.10,342.85] vol=2.3x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-11 11:05:00 | 345.50 | 341.72 | 0.00 | T1 1.5R @ 345.50 |
| Target hit | 2025-06-11 15:20:00 | 354.95 | 350.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2025-06-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 09:45:00 | 348.90 | 345.65 | 0.00 | ORB-long ORB[343.00,346.90] vol=3.3x ATR=1.38 |
| Stop hit — per-position SL triggered | 2025-06-19 09:50:00 | 347.52 | 345.86 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 09:45:00 | 354.55 | 352.67 | 0.00 | ORB-long ORB[351.00,353.90] vol=4.8x ATR=1.19 |
| Stop hit — per-position SL triggered | 2025-06-26 09:55:00 | 353.36 | 352.85 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 09:45:00 | 378.90 | 376.18 | 0.00 | ORB-long ORB[372.70,376.00] vol=1.6x ATR=1.21 |
| Stop hit — per-position SL triggered | 2025-07-04 10:05:00 | 377.69 | 376.73 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 11:05:00 | 367.40 | 370.48 | 0.00 | ORB-short ORB[371.65,373.95] vol=2.3x ATR=0.77 |
| Stop hit — per-position SL triggered | 2025-07-08 11:15:00 | 368.17 | 370.05 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 11:00:00 | 370.70 | 374.38 | 0.00 | ORB-short ORB[373.10,377.10] vol=1.7x ATR=0.89 |
| Stop hit — per-position SL triggered | 2025-07-10 12:50:00 | 371.59 | 373.15 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 10:05:00 | 377.20 | 375.16 | 0.00 | ORB-long ORB[372.90,376.10] vol=2.3x ATR=1.05 |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 376.15 | 375.33 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-14 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 09:50:00 | 373.60 | 371.26 | 0.00 | ORB-long ORB[369.00,372.85] vol=1.5x ATR=1.28 |
| Stop hit — per-position SL triggered | 2025-07-14 10:10:00 | 372.32 | 371.70 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 11:10:00 | 386.15 | 384.57 | 0.00 | ORB-long ORB[380.75,385.80] vol=3.2x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 11:40:00 | 387.88 | 385.25 | 0.00 | T1 1.5R @ 387.88 |
| Target hit | 2025-07-15 15:20:00 | 391.15 | 387.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2025-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-18 09:30:00 | 401.25 | 399.46 | 0.00 | ORB-long ORB[396.35,399.00] vol=4.7x ATR=1.07 |
| Stop hit — per-position SL triggered | 2025-07-18 09:40:00 | 400.18 | 399.76 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 09:30:00 | 389.85 | 391.79 | 0.00 | ORB-short ORB[391.15,394.75] vol=1.6x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 09:45:00 | 388.55 | 390.64 | 0.00 | T1 1.5R @ 388.55 |
| Stop hit — per-position SL triggered | 2025-07-22 09:50:00 | 389.85 | 390.57 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-07-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 09:30:00 | 390.75 | 388.00 | 0.00 | ORB-long ORB[384.60,388.85] vol=1.8x ATR=1.11 |
| Stop hit — per-position SL triggered | 2025-07-23 09:40:00 | 389.64 | 388.23 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:15:00 | 394.10 | 397.20 | 0.00 | ORB-short ORB[395.00,399.60] vol=2.1x ATR=1.04 |
| Stop hit — per-position SL triggered | 2025-07-25 10:25:00 | 395.14 | 396.94 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-07-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 10:30:00 | 393.90 | 391.44 | 0.00 | ORB-long ORB[387.45,390.30] vol=1.9x ATR=1.15 |
| Stop hit — per-position SL triggered | 2025-07-29 10:50:00 | 392.75 | 392.33 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-07-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 10:00:00 | 402.80 | 400.79 | 0.00 | ORB-long ORB[397.60,401.00] vol=4.7x ATR=1.14 |
| Stop hit — per-position SL triggered | 2025-07-30 10:05:00 | 401.66 | 400.83 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-08-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-04 09:50:00 | 378.20 | 378.76 | 0.00 | ORB-short ORB[382.60,386.60] vol=16.0x ATR=2.08 |
| Stop hit — per-position SL triggered | 2025-08-04 09:55:00 | 380.28 | 378.87 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 11:15:00 | 363.85 | 367.41 | 0.00 | ORB-short ORB[368.60,372.90] vol=1.8x ATR=0.94 |
| Stop hit — per-position SL triggered | 2025-08-06 12:30:00 | 364.79 | 366.60 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-08-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-08 09:40:00 | 365.55 | 361.79 | 0.00 | ORB-long ORB[358.25,363.70] vol=2.2x ATR=1.96 |
| Stop hit — per-position SL triggered | 2025-08-08 09:45:00 | 363.59 | 361.87 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-08-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-12 09:50:00 | 344.10 | 342.43 | 0.00 | ORB-long ORB[341.10,343.60] vol=1.9x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 09:55:00 | 345.82 | 343.79 | 0.00 | T1 1.5R @ 345.82 |
| Target hit | 2025-08-12 15:20:00 | 354.25 | 351.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — BUY (started 2025-08-13 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 11:00:00 | 357.10 | 355.80 | 0.00 | ORB-long ORB[354.00,356.90] vol=5.6x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 12:05:00 | 358.53 | 356.59 | 0.00 | T1 1.5R @ 358.53 |
| Target hit | 2025-08-13 15:20:00 | 360.15 | 358.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — SELL (started 2025-08-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-20 09:40:00 | 359.40 | 361.15 | 0.00 | ORB-short ORB[360.80,365.05] vol=1.7x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-08-20 09:45:00 | 360.26 | 361.03 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-08-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 10:20:00 | 350.00 | 352.62 | 0.00 | ORB-short ORB[351.90,356.10] vol=1.8x ATR=1.06 |
| Stop hit — per-position SL triggered | 2025-08-29 10:35:00 | 351.06 | 352.22 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-09-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 11:00:00 | 358.80 | 356.44 | 0.00 | ORB-long ORB[353.85,356.40] vol=2.1x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-03 11:05:00 | 360.22 | 357.47 | 0.00 | T1 1.5R @ 360.22 |
| Target hit | 2025-09-03 15:20:00 | 361.40 | 360.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — BUY (started 2025-09-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 09:30:00 | 366.35 | 364.37 | 0.00 | ORB-long ORB[362.50,365.00] vol=1.6x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 09:45:00 | 367.84 | 365.42 | 0.00 | T1 1.5R @ 367.84 |
| Stop hit — per-position SL triggered | 2025-09-08 09:50:00 | 366.35 | 365.53 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-09-09 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 10:20:00 | 365.40 | 363.55 | 0.00 | ORB-long ORB[362.60,364.65] vol=1.5x ATR=0.93 |
| Stop hit — per-position SL triggered | 2025-09-09 14:45:00 | 364.47 | 364.83 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 10:10:00 | 369.30 | 366.87 | 0.00 | ORB-long ORB[364.85,368.10] vol=2.2x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 10:15:00 | 370.67 | 367.53 | 0.00 | T1 1.5R @ 370.67 |
| Stop hit — per-position SL triggered | 2025-09-10 10:20:00 | 369.30 | 367.76 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-09-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 10:45:00 | 366.10 | 365.31 | 0.00 | ORB-long ORB[363.20,365.50] vol=2.9x ATR=0.62 |
| Stop hit — per-position SL triggered | 2025-09-12 10:55:00 | 365.48 | 365.38 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-09-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-15 11:00:00 | 360.40 | 362.41 | 0.00 | ORB-short ORB[362.05,367.00] vol=1.6x ATR=0.75 |
| Stop hit — per-position SL triggered | 2025-09-15 11:05:00 | 361.15 | 362.36 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-09-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 10:30:00 | 372.70 | 371.08 | 0.00 | ORB-long ORB[369.15,372.60] vol=2.2x ATR=1.33 |
| Stop hit — per-position SL triggered | 2025-09-19 12:10:00 | 371.37 | 371.69 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-09-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 11:00:00 | 359.90 | 361.44 | 0.00 | ORB-short ORB[360.20,364.35] vol=2.8x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 11:10:00 | 358.67 | 361.37 | 0.00 | T1 1.5R @ 358.67 |
| Stop hit — per-position SL triggered | 2025-09-23 11:50:00 | 359.90 | 360.95 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 11:15:00 | 346.00 | 348.15 | 0.00 | ORB-short ORB[347.80,352.80] vol=4.2x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 11:30:00 | 344.93 | 347.86 | 0.00 | T1 1.5R @ 344.93 |
| Stop hit — per-position SL triggered | 2025-10-06 11:35:00 | 346.00 | 347.67 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-10-08 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 10:40:00 | 351.65 | 349.59 | 0.00 | ORB-long ORB[346.90,351.00] vol=2.9x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 10:50:00 | 353.19 | 350.57 | 0.00 | T1 1.5R @ 353.19 |
| Stop hit — per-position SL triggered | 2025-10-08 11:10:00 | 351.65 | 351.78 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-10-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 09:55:00 | 348.10 | 348.94 | 0.00 | ORB-short ORB[348.20,353.00] vol=12.8x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 10:15:00 | 346.66 | 348.85 | 0.00 | T1 1.5R @ 346.66 |
| Target hit | 2025-10-14 12:15:00 | 347.65 | 347.14 | 0.00 | Trail-exit close>VWAP |

### Cycle 44 — BUY (started 2025-10-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 11:10:00 | 353.60 | 352.46 | 0.00 | ORB-long ORB[350.05,352.95] vol=1.7x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 11:45:00 | 354.75 | 353.01 | 0.00 | T1 1.5R @ 354.75 |
| Target hit | 2025-10-15 13:25:00 | 354.05 | 354.07 | 0.00 | Trail-exit close<VWAP |

### Cycle 45 — SELL (started 2025-10-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 10:30:00 | 357.45 | 359.48 | 0.00 | ORB-short ORB[358.25,361.00] vol=1.5x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 10:35:00 | 356.13 | 359.18 | 0.00 | T1 1.5R @ 356.13 |
| Stop hit — per-position SL triggered | 2025-10-17 12:00:00 | 357.45 | 358.22 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-10-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 09:40:00 | 359.60 | 358.36 | 0.00 | ORB-long ORB[356.70,358.85] vol=1.7x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 09:45:00 | 360.98 | 359.18 | 0.00 | T1 1.5R @ 360.98 |
| Target hit | 2025-10-20 13:55:00 | 361.40 | 361.85 | 0.00 | Trail-exit close<VWAP |

### Cycle 47 — BUY (started 2025-10-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 11:00:00 | 369.15 | 367.43 | 0.00 | ORB-long ORB[365.95,368.60] vol=3.0x ATR=0.97 |
| Stop hit — per-position SL triggered | 2025-10-29 11:15:00 | 368.18 | 367.73 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-11-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 10:45:00 | 374.80 | 374.18 | 0.00 | ORB-long ORB[372.00,374.65] vol=2.2x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 10:50:00 | 376.19 | 374.67 | 0.00 | T1 1.5R @ 376.19 |
| Target hit | 2025-11-03 11:10:00 | 374.85 | 375.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 49 — BUY (started 2025-11-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 09:30:00 | 387.15 | 385.31 | 0.00 | ORB-long ORB[381.25,385.35] vol=4.6x ATR=1.28 |
| Stop hit — per-position SL triggered | 2025-11-11 09:45:00 | 385.87 | 385.71 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-11-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 11:10:00 | 418.75 | 416.84 | 0.00 | ORB-long ORB[412.65,416.90] vol=2.7x ATR=1.29 |
| Stop hit — per-position SL triggered | 2025-11-14 11:40:00 | 417.46 | 416.98 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-11-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 09:55:00 | 418.25 | 414.50 | 0.00 | ORB-long ORB[409.60,414.65] vol=2.5x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-17 12:55:00 | 420.40 | 417.09 | 0.00 | T1 1.5R @ 420.40 |
| Stop hit — per-position SL triggered | 2025-11-17 14:00:00 | 418.25 | 417.50 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-11-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 09:35:00 | 397.40 | 395.12 | 0.00 | ORB-long ORB[392.50,395.55] vol=2.0x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-26 09:40:00 | 399.24 | 396.28 | 0.00 | T1 1.5R @ 399.24 |
| Stop hit — per-position SL triggered | 2025-11-26 09:45:00 | 397.40 | 396.33 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-12-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 11:10:00 | 400.65 | 404.53 | 0.00 | ORB-short ORB[402.00,407.70] vol=4.8x ATR=1.48 |
| Stop hit — per-position SL triggered | 2025-12-03 11:25:00 | 402.13 | 404.21 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-12-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 10:50:00 | 380.00 | 381.63 | 0.00 | ORB-short ORB[380.70,384.95] vol=1.8x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 10:55:00 | 378.35 | 381.24 | 0.00 | T1 1.5R @ 378.35 |
| Stop hit — per-position SL triggered | 2025-12-10 12:05:00 | 380.00 | 379.98 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-12-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:45:00 | 385.50 | 384.09 | 0.00 | ORB-long ORB[382.50,384.55] vol=1.6x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-12 09:50:00 | 386.81 | 385.67 | 0.00 | T1 1.5R @ 386.81 |
| Target hit | 2025-12-12 10:10:00 | 385.70 | 385.97 | 0.00 | Trail-exit close<VWAP |

### Cycle 56 — SELL (started 2025-12-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 10:55:00 | 384.20 | 386.12 | 0.00 | ORB-short ORB[384.30,387.95] vol=1.7x ATR=0.76 |
| Stop hit — per-position SL triggered | 2025-12-17 11:00:00 | 384.96 | 385.99 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-12-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 10:10:00 | 390.85 | 388.24 | 0.00 | ORB-long ORB[386.05,389.30] vol=2.0x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 10:20:00 | 392.56 | 388.73 | 0.00 | T1 1.5R @ 392.56 |
| Stop hit — per-position SL triggered | 2025-12-18 14:10:00 | 390.85 | 390.84 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-12-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 09:35:00 | 399.75 | 398.14 | 0.00 | ORB-long ORB[393.40,398.70] vol=3.6x ATR=1.34 |
| Stop hit — per-position SL triggered | 2025-12-19 09:55:00 | 398.41 | 399.10 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-12-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 10:45:00 | 394.45 | 396.44 | 0.00 | ORB-short ORB[395.00,399.20] vol=1.9x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 11:15:00 | 393.25 | 396.09 | 0.00 | T1 1.5R @ 393.25 |
| Stop hit — per-position SL triggered | 2025-12-29 11:50:00 | 394.45 | 395.35 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2026-01-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 10:35:00 | 387.20 | 389.71 | 0.00 | ORB-short ORB[390.60,393.90] vol=3.5x ATR=0.91 |
| Stop hit — per-position SL triggered | 2026-01-01 11:00:00 | 388.11 | 388.85 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2026-01-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 11:10:00 | 391.20 | 388.97 | 0.00 | ORB-long ORB[386.35,389.40] vol=1.9x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 11:40:00 | 392.27 | 389.48 | 0.00 | T1 1.5R @ 392.27 |
| Target hit | 2026-01-02 15:20:00 | 393.05 | 392.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — BUY (started 2026-01-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 10:05:00 | 392.50 | 390.34 | 0.00 | ORB-long ORB[387.05,391.35] vol=2.5x ATR=1.50 |
| Stop hit — per-position SL triggered | 2026-01-07 10:35:00 | 391.00 | 390.80 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2026-01-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:10:00 | 382.55 | 385.83 | 0.00 | ORB-short ORB[383.50,388.25] vol=1.7x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:20:00 | 381.15 | 385.39 | 0.00 | T1 1.5R @ 381.15 |
| Stop hit — per-position SL triggered | 2026-01-08 11:35:00 | 382.55 | 385.06 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-01-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-09 11:00:00 | 382.70 | 377.79 | 0.00 | ORB-long ORB[374.30,378.80] vol=3.3x ATR=1.31 |
| Stop hit — per-position SL triggered | 2026-01-09 11:15:00 | 381.39 | 378.28 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-01-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 09:45:00 | 380.25 | 378.11 | 0.00 | ORB-long ORB[372.95,376.90] vol=3.3x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 09:55:00 | 382.49 | 379.01 | 0.00 | T1 1.5R @ 382.49 |
| Target hit | 2026-01-14 13:45:00 | 380.50 | 381.25 | 0.00 | Trail-exit close<VWAP |

### Cycle 66 — BUY (started 2026-01-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 09:45:00 | 382.80 | 381.11 | 0.00 | ORB-long ORB[378.65,382.30] vol=2.5x ATR=1.18 |
| Stop hit — per-position SL triggered | 2026-01-16 10:00:00 | 381.62 | 381.69 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-02-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 11:10:00 | 375.70 | 371.60 | 0.00 | ORB-long ORB[362.65,367.15] vol=18.4x ATR=1.92 |
| Stop hit — per-position SL triggered | 2026-02-01 11:15:00 | 373.78 | 372.23 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-19 10:00:00 | 343.80 | 2025-05-19 10:25:00 | 342.63 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-05-21 09:30:00 | 338.50 | 2025-05-21 10:25:00 | 340.38 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-05-21 09:30:00 | 338.50 | 2025-05-21 11:05:00 | 338.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-23 10:20:00 | 329.45 | 2025-05-23 10:40:00 | 330.52 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-05-26 10:15:00 | 334.50 | 2025-05-26 10:25:00 | 333.61 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-05-27 09:30:00 | 330.95 | 2025-05-27 09:40:00 | 331.77 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-05-29 11:10:00 | 332.85 | 2025-05-29 11:25:00 | 333.50 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-05-30 09:45:00 | 334.00 | 2025-05-30 10:15:00 | 332.73 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-05-30 09:45:00 | 334.00 | 2025-05-30 11:40:00 | 334.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-04 09:35:00 | 334.30 | 2025-06-04 10:30:00 | 335.27 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-06-09 10:30:00 | 334.70 | 2025-06-09 10:40:00 | 333.88 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-06-10 09:30:00 | 344.85 | 2025-06-10 09:35:00 | 343.60 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-06-11 10:55:00 | 344.15 | 2025-06-11 11:05:00 | 345.50 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-06-11 10:55:00 | 344.15 | 2025-06-11 15:20:00 | 354.95 | TARGET_HIT | 0.50 | 3.14% |
| BUY | retest1 | 2025-06-19 09:45:00 | 348.90 | 2025-06-19 09:50:00 | 347.52 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-06-26 09:45:00 | 354.55 | 2025-06-26 09:55:00 | 353.36 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-07-04 09:45:00 | 378.90 | 2025-07-04 10:05:00 | 377.69 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-07-08 11:05:00 | 367.40 | 2025-07-08 11:15:00 | 368.17 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-07-10 11:00:00 | 370.70 | 2025-07-10 12:50:00 | 371.59 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-07-11 10:05:00 | 377.20 | 2025-07-11 10:15:00 | 376.15 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-07-14 09:50:00 | 373.60 | 2025-07-14 10:10:00 | 372.32 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-07-15 11:10:00 | 386.15 | 2025-07-15 11:40:00 | 387.88 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-07-15 11:10:00 | 386.15 | 2025-07-15 15:20:00 | 391.15 | TARGET_HIT | 0.50 | 1.29% |
| BUY | retest1 | 2025-07-18 09:30:00 | 401.25 | 2025-07-18 09:40:00 | 400.18 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-07-22 09:30:00 | 389.85 | 2025-07-22 09:45:00 | 388.55 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-07-22 09:30:00 | 389.85 | 2025-07-22 09:50:00 | 389.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-23 09:30:00 | 390.75 | 2025-07-23 09:40:00 | 389.64 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-07-25 10:15:00 | 394.10 | 2025-07-25 10:25:00 | 395.14 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-07-29 10:30:00 | 393.90 | 2025-07-29 10:50:00 | 392.75 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-07-30 10:00:00 | 402.80 | 2025-07-30 10:05:00 | 401.66 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-08-04 09:50:00 | 378.20 | 2025-08-04 09:55:00 | 380.28 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2025-08-06 11:15:00 | 363.85 | 2025-08-06 12:30:00 | 364.79 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-08-08 09:40:00 | 365.55 | 2025-08-08 09:45:00 | 363.59 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2025-08-12 09:50:00 | 344.10 | 2025-08-12 09:55:00 | 345.82 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-08-12 09:50:00 | 344.10 | 2025-08-12 15:20:00 | 354.25 | TARGET_HIT | 0.50 | 2.95% |
| BUY | retest1 | 2025-08-13 11:00:00 | 357.10 | 2025-08-13 12:05:00 | 358.53 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-08-13 11:00:00 | 357.10 | 2025-08-13 15:20:00 | 360.15 | TARGET_HIT | 0.50 | 0.85% |
| SELL | retest1 | 2025-08-20 09:40:00 | 359.40 | 2025-08-20 09:45:00 | 360.26 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-08-29 10:20:00 | 350.00 | 2025-08-29 10:35:00 | 351.06 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-09-03 11:00:00 | 358.80 | 2025-09-03 11:05:00 | 360.22 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-09-03 11:00:00 | 358.80 | 2025-09-03 15:20:00 | 361.40 | TARGET_HIT | 0.50 | 0.72% |
| BUY | retest1 | 2025-09-08 09:30:00 | 366.35 | 2025-09-08 09:45:00 | 367.84 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-09-08 09:30:00 | 366.35 | 2025-09-08 09:50:00 | 366.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-09 10:20:00 | 365.40 | 2025-09-09 14:45:00 | 364.47 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-09-10 10:10:00 | 369.30 | 2025-09-10 10:15:00 | 370.67 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-09-10 10:10:00 | 369.30 | 2025-09-10 10:20:00 | 369.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-12 10:45:00 | 366.10 | 2025-09-12 10:55:00 | 365.48 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-09-15 11:00:00 | 360.40 | 2025-09-15 11:05:00 | 361.15 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-09-19 10:30:00 | 372.70 | 2025-09-19 12:10:00 | 371.37 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-09-23 11:00:00 | 359.90 | 2025-09-23 11:10:00 | 358.67 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-09-23 11:00:00 | 359.90 | 2025-09-23 11:50:00 | 359.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-06 11:15:00 | 346.00 | 2025-10-06 11:30:00 | 344.93 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-10-06 11:15:00 | 346.00 | 2025-10-06 11:35:00 | 346.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-08 10:40:00 | 351.65 | 2025-10-08 10:50:00 | 353.19 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-10-08 10:40:00 | 351.65 | 2025-10-08 11:10:00 | 351.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-14 09:55:00 | 348.10 | 2025-10-14 10:15:00 | 346.66 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-10-14 09:55:00 | 348.10 | 2025-10-14 12:15:00 | 347.65 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2025-10-15 11:10:00 | 353.60 | 2025-10-15 11:45:00 | 354.75 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-10-15 11:10:00 | 353.60 | 2025-10-15 13:25:00 | 354.05 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2025-10-17 10:30:00 | 357.45 | 2025-10-17 10:35:00 | 356.13 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-10-17 10:30:00 | 357.45 | 2025-10-17 12:00:00 | 357.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-20 09:40:00 | 359.60 | 2025-10-20 09:45:00 | 360.98 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-10-20 09:40:00 | 359.60 | 2025-10-20 13:55:00 | 361.40 | TARGET_HIT | 0.50 | 0.50% |
| BUY | retest1 | 2025-10-29 11:00:00 | 369.15 | 2025-10-29 11:15:00 | 368.18 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-11-03 10:45:00 | 374.80 | 2025-11-03 10:50:00 | 376.19 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-11-03 10:45:00 | 374.80 | 2025-11-03 11:10:00 | 374.85 | TARGET_HIT | 0.50 | 0.01% |
| BUY | retest1 | 2025-11-11 09:30:00 | 387.15 | 2025-11-11 09:45:00 | 385.87 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-11-14 11:10:00 | 418.75 | 2025-11-14 11:40:00 | 417.46 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-11-17 09:55:00 | 418.25 | 2025-11-17 12:55:00 | 420.40 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-11-17 09:55:00 | 418.25 | 2025-11-17 14:00:00 | 418.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-26 09:35:00 | 397.40 | 2025-11-26 09:40:00 | 399.24 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-11-26 09:35:00 | 397.40 | 2025-11-26 09:45:00 | 397.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-03 11:10:00 | 400.65 | 2025-12-03 11:25:00 | 402.13 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-12-10 10:50:00 | 380.00 | 2025-12-10 10:55:00 | 378.35 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-12-10 10:50:00 | 380.00 | 2025-12-10 12:05:00 | 380.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-12 09:45:00 | 385.50 | 2025-12-12 09:50:00 | 386.81 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-12-12 09:45:00 | 385.50 | 2025-12-12 10:10:00 | 385.70 | TARGET_HIT | 0.50 | 0.05% |
| SELL | retest1 | 2025-12-17 10:55:00 | 384.20 | 2025-12-17 11:00:00 | 384.96 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-12-18 10:10:00 | 390.85 | 2025-12-18 10:20:00 | 392.56 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-12-18 10:10:00 | 390.85 | 2025-12-18 14:10:00 | 390.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-19 09:35:00 | 399.75 | 2025-12-19 09:55:00 | 398.41 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-12-29 10:45:00 | 394.45 | 2025-12-29 11:15:00 | 393.25 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-12-29 10:45:00 | 394.45 | 2025-12-29 11:50:00 | 394.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-01 10:35:00 | 387.20 | 2026-01-01 11:00:00 | 388.11 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-01-02 11:10:00 | 391.20 | 2026-01-02 11:40:00 | 392.27 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2026-01-02 11:10:00 | 391.20 | 2026-01-02 15:20:00 | 393.05 | TARGET_HIT | 0.50 | 0.47% |
| BUY | retest1 | 2026-01-07 10:05:00 | 392.50 | 2026-01-07 10:35:00 | 391.00 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-01-08 11:10:00 | 382.55 | 2026-01-08 11:20:00 | 381.15 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-01-08 11:10:00 | 382.55 | 2026-01-08 11:35:00 | 382.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-09 11:00:00 | 382.70 | 2026-01-09 11:15:00 | 381.39 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-01-14 09:45:00 | 380.25 | 2026-01-14 09:55:00 | 382.49 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2026-01-14 09:45:00 | 380.25 | 2026-01-14 13:45:00 | 380.50 | TARGET_HIT | 0.50 | 0.07% |
| BUY | retest1 | 2026-01-16 09:45:00 | 382.80 | 2026-01-16 10:00:00 | 381.62 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-01 11:10:00 | 375.70 | 2026-02-01 11:15:00 | 373.78 | STOP_HIT | 1.00 | -0.51% |
