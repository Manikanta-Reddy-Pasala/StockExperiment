# Usha Martin Ltd. (USHAMART)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (35221 bars)
- **Last close:** 472.00
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
| ENTRY1 | 50 |
| ENTRY2 | 0 |
| PARTIAL | 17 |
| TARGET_HIT | 8 |
| STOP_HIT | 42 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 67 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 42
- **Target hits / Stop hits / Partials:** 8 / 42 / 17
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 5.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 32 | 10 | 31.2% | 3 | 22 | 7 | -0.07% | -2.1% |
| BUY @ 2nd Alert (retest1) | 32 | 10 | 31.2% | 3 | 22 | 7 | -0.07% | -2.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 35 | 15 | 42.9% | 5 | 20 | 10 | 0.22% | 7.8% |
| SELL @ 2nd Alert (retest1) | 35 | 15 | 42.9% | 5 | 20 | 10 | 0.22% | 7.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 67 | 25 | 37.3% | 8 | 42 | 17 | 0.09% | 5.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 10:45:00 | 331.35 | 335.54 | 0.00 | ORB-short ORB[338.10,342.55] vol=1.6x ATR=2.39 |
| Stop hit — per-position SL triggered | 2024-05-13 11:05:00 | 333.74 | 335.40 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 10:25:00 | 339.20 | 342.97 | 0.00 | ORB-short ORB[345.40,348.70] vol=6.1x ATR=1.72 |
| Stop hit — per-position SL triggered | 2024-05-15 10:45:00 | 340.92 | 342.38 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 11:15:00 | 335.95 | 337.97 | 0.00 | ORB-short ORB[336.70,340.80] vol=1.6x ATR=1.10 |
| Stop hit — per-position SL triggered | 2024-05-16 12:05:00 | 337.05 | 337.86 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 09:35:00 | 355.40 | 356.28 | 0.00 | ORB-short ORB[356.30,359.35] vol=2.5x ATR=1.21 |
| Stop hit — per-position SL triggered | 2024-05-24 09:40:00 | 356.61 | 356.47 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-28 09:30:00 | 363.95 | 359.78 | 0.00 | ORB-long ORB[352.85,357.60] vol=8.0x ATR=2.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 09:35:00 | 367.12 | 366.22 | 0.00 | T1 1.5R @ 367.12 |
| Target hit | 2024-05-28 09:55:00 | 365.70 | 366.76 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2024-05-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-30 10:45:00 | 365.35 | 362.08 | 0.00 | ORB-long ORB[358.80,364.20] vol=4.6x ATR=1.46 |
| Stop hit — per-position SL triggered | 2024-05-30 10:50:00 | 363.89 | 362.14 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 09:30:00 | 358.25 | 356.23 | 0.00 | ORB-long ORB[353.00,358.00] vol=1.6x ATR=1.83 |
| Stop hit — per-position SL triggered | 2024-06-07 09:45:00 | 356.42 | 356.60 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 11:15:00 | 368.40 | 365.52 | 0.00 | ORB-long ORB[363.35,368.25] vol=2.4x ATR=1.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 12:15:00 | 370.51 | 367.34 | 0.00 | T1 1.5R @ 370.51 |
| Stop hit — per-position SL triggered | 2024-06-10 12:30:00 | 368.40 | 368.90 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 09:45:00 | 375.60 | 373.90 | 0.00 | ORB-long ORB[372.00,374.95] vol=2.4x ATR=1.31 |
| Stop hit — per-position SL triggered | 2024-06-12 10:00:00 | 374.29 | 374.04 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-20 09:45:00 | 385.90 | 387.22 | 0.00 | ORB-short ORB[386.20,391.15] vol=2.1x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 10:55:00 | 383.53 | 386.39 | 0.00 | T1 1.5R @ 383.53 |
| Stop hit — per-position SL triggered | 2024-06-20 11:10:00 | 385.90 | 385.95 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 09:40:00 | 402.00 | 403.50 | 0.00 | ORB-short ORB[402.55,407.05] vol=4.3x ATR=1.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 10:00:00 | 399.87 | 400.45 | 0.00 | T1 1.5R @ 399.87 |
| Target hit | 2024-07-09 13:40:00 | 394.00 | 393.58 | 0.00 | Trail-exit close>VWAP |

### Cycle 12 — SELL (started 2024-07-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 09:45:00 | 386.50 | 389.16 | 0.00 | ORB-short ORB[388.00,393.40] vol=1.6x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:00:00 | 384.05 | 388.64 | 0.00 | T1 1.5R @ 384.05 |
| Target hit | 2024-07-10 11:30:00 | 384.90 | 384.87 | 0.00 | Trail-exit close>VWAP |

### Cycle 13 — BUY (started 2024-07-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 10:00:00 | 391.95 | 388.48 | 0.00 | ORB-long ORB[386.55,390.50] vol=1.8x ATR=1.96 |
| Stop hit — per-position SL triggered | 2024-07-16 10:05:00 | 389.99 | 388.84 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 11:00:00 | 366.00 | 366.85 | 0.00 | ORB-short ORB[366.20,369.95] vol=2.4x ATR=1.14 |
| Stop hit — per-position SL triggered | 2024-07-23 11:10:00 | 367.14 | 366.80 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-26 10:05:00 | 359.55 | 361.86 | 0.00 | ORB-short ORB[362.50,367.90] vol=5.1x ATR=1.61 |
| Stop hit — per-position SL triggered | 2024-07-26 10:40:00 | 361.16 | 360.90 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-31 10:00:00 | 374.70 | 376.97 | 0.00 | ORB-short ORB[377.20,381.85] vol=4.5x ATR=1.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-31 10:05:00 | 372.58 | 375.73 | 0.00 | T1 1.5R @ 372.58 |
| Stop hit — per-position SL triggered | 2024-07-31 10:20:00 | 374.70 | 375.42 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-08-01 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 10:10:00 | 385.70 | 383.22 | 0.00 | ORB-long ORB[379.75,383.90] vol=3.5x ATR=1.35 |
| Stop hit — per-position SL triggered | 2024-08-01 10:45:00 | 384.35 | 384.49 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-08-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 09:35:00 | 330.95 | 332.93 | 0.00 | ORB-short ORB[332.15,335.00] vol=1.6x ATR=1.34 |
| Stop hit — per-position SL triggered | 2024-08-22 09:40:00 | 332.29 | 332.79 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-08-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 09:45:00 | 333.00 | 334.17 | 0.00 | ORB-short ORB[333.15,336.60] vol=3.6x ATR=1.41 |
| Stop hit — per-position SL triggered | 2024-08-23 09:55:00 | 334.41 | 334.06 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 10:05:00 | 335.55 | 332.96 | 0.00 | ORB-long ORB[330.45,334.60] vol=3.0x ATR=1.34 |
| Stop hit — per-position SL triggered | 2024-08-26 10:10:00 | 334.21 | 333.15 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:15:00 | 336.75 | 339.77 | 0.00 | ORB-short ORB[338.95,342.00] vol=1.7x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 11:50:00 | 334.74 | 338.68 | 0.00 | T1 1.5R @ 334.74 |
| Stop hit — per-position SL triggered | 2024-08-29 12:25:00 | 336.75 | 337.81 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-09-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 11:05:00 | 337.60 | 340.39 | 0.00 | ORB-short ORB[340.05,344.50] vol=2.3x ATR=0.95 |
| Stop hit — per-position SL triggered | 2024-09-05 11:10:00 | 338.55 | 339.59 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-09-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:40:00 | 370.90 | 368.12 | 0.00 | ORB-long ORB[365.50,370.50] vol=1.5x ATR=1.41 |
| Stop hit — per-position SL triggered | 2024-09-11 09:55:00 | 369.49 | 368.83 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-09-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 10:55:00 | 360.20 | 356.24 | 0.00 | ORB-long ORB[355.00,360.00] vol=7.5x ATR=1.37 |
| Stop hit — per-position SL triggered | 2024-09-16 11:00:00 | 358.83 | 356.31 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-09-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:30:00 | 345.30 | 347.22 | 0.00 | ORB-short ORB[345.60,350.35] vol=2.0x ATR=1.62 |
| Stop hit — per-position SL triggered | 2024-09-19 09:35:00 | 346.92 | 346.96 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 11:00:00 | 348.05 | 346.43 | 0.00 | ORB-long ORB[345.40,347.70] vol=3.8x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 11:20:00 | 350.07 | 347.56 | 0.00 | T1 1.5R @ 350.07 |
| Stop hit — per-position SL triggered | 2024-09-23 12:00:00 | 348.05 | 348.28 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-09-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-27 11:00:00 | 350.60 | 353.34 | 0.00 | ORB-short ORB[352.00,354.15] vol=1.8x ATR=1.18 |
| Stop hit — per-position SL triggered | 2024-09-27 11:20:00 | 351.78 | 352.76 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 11:15:00 | 362.80 | 359.96 | 0.00 | ORB-long ORB[357.55,361.80] vol=4.4x ATR=1.27 |
| Stop hit — per-position SL triggered | 2024-10-01 11:20:00 | 361.53 | 360.48 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 11:15:00 | 343.95 | 348.32 | 0.00 | ORB-short ORB[350.10,354.80] vol=1.5x ATR=2.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 13:00:00 | 340.48 | 346.97 | 0.00 | T1 1.5R @ 340.48 |
| Target hit | 2024-10-07 15:20:00 | 340.60 | 343.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — BUY (started 2024-10-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 10:55:00 | 348.10 | 347.12 | 0.00 | ORB-long ORB[345.85,347.50] vol=8.4x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-09 11:15:00 | 349.95 | 347.85 | 0.00 | T1 1.5R @ 349.95 |
| Target hit | 2024-10-09 12:20:00 | 348.40 | 348.52 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — BUY (started 2024-11-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 09:45:00 | 406.65 | 402.44 | 0.00 | ORB-long ORB[398.40,403.60] vol=3.3x ATR=2.36 |
| Stop hit — per-position SL triggered | 2024-11-25 09:50:00 | 404.29 | 402.58 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-12-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 10:05:00 | 411.45 | 408.64 | 0.00 | ORB-long ORB[407.00,410.00] vol=3.0x ATR=1.45 |
| Stop hit — per-position SL triggered | 2024-12-09 10:10:00 | 410.00 | 408.84 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-12-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 11:05:00 | 411.20 | 408.81 | 0.00 | ORB-long ORB[407.00,411.10] vol=4.4x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 11:20:00 | 413.27 | 411.96 | 0.00 | T1 1.5R @ 413.27 |
| Target hit | 2024-12-10 11:40:00 | 411.90 | 416.56 | 0.00 | Trail-exit close<VWAP |

### Cycle 34 — SELL (started 2024-12-13 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 11:10:00 | 393.20 | 394.99 | 0.00 | ORB-short ORB[394.10,399.55] vol=6.5x ATR=1.74 |
| Stop hit — per-position SL triggered | 2024-12-13 11:50:00 | 394.94 | 394.93 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-12-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 10:45:00 | 395.00 | 397.81 | 0.00 | ORB-short ORB[396.30,401.00] vol=3.0x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 11:05:00 | 392.72 | 396.74 | 0.00 | T1 1.5R @ 392.72 |
| Target hit | 2024-12-16 15:20:00 | 385.00 | 389.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — BUY (started 2024-12-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 11:10:00 | 376.50 | 373.86 | 0.00 | ORB-long ORB[370.05,374.80] vol=2.0x ATR=1.29 |
| Stop hit — per-position SL triggered | 2024-12-19 12:00:00 | 375.21 | 374.23 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-12-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 10:55:00 | 378.90 | 376.46 | 0.00 | ORB-long ORB[372.65,375.85] vol=3.6x ATR=1.00 |
| Stop hit — per-position SL triggered | 2024-12-24 11:35:00 | 377.90 | 378.11 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-01-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 10:55:00 | 375.40 | 377.73 | 0.00 | ORB-short ORB[377.00,382.50] vol=3.2x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 11:05:00 | 373.24 | 376.60 | 0.00 | T1 1.5R @ 373.24 |
| Target hit | 2025-01-03 15:20:00 | 367.00 | 372.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — BUY (started 2025-01-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 10:55:00 | 378.65 | 373.42 | 0.00 | ORB-long ORB[368.10,373.40] vol=1.7x ATR=1.98 |
| Stop hit — per-position SL triggered | 2025-01-07 11:25:00 | 376.67 | 376.81 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-01-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 09:30:00 | 348.10 | 350.43 | 0.00 | ORB-short ORB[349.65,354.70] vol=1.5x ATR=2.03 |
| Stop hit — per-position SL triggered | 2025-01-15 09:35:00 | 350.13 | 350.37 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-01-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 11:05:00 | 353.60 | 357.60 | 0.00 | ORB-short ORB[359.50,363.25] vol=5.2x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 11:25:00 | 351.35 | 356.20 | 0.00 | T1 1.5R @ 351.35 |
| Stop hit — per-position SL triggered | 2025-01-21 11:45:00 | 353.60 | 355.40 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-01-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:40:00 | 350.10 | 346.67 | 0.00 | ORB-long ORB[342.55,345.90] vol=1.9x ATR=1.34 |
| Stop hit — per-position SL triggered | 2025-01-23 10:50:00 | 348.76 | 346.98 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-02-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 10:40:00 | 334.95 | 337.27 | 0.00 | ORB-short ORB[342.45,347.10] vol=2.2x ATR=2.20 |
| Stop hit — per-position SL triggered | 2025-02-01 11:35:00 | 337.15 | 336.43 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-02-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 10:30:00 | 321.70 | 325.63 | 0.00 | ORB-short ORB[324.70,329.35] vol=2.2x ATR=1.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-04 10:50:00 | 319.13 | 324.99 | 0.00 | T1 1.5R @ 319.13 |
| Stop hit — per-position SL triggered | 2025-02-04 11:45:00 | 321.70 | 323.69 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-03-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-03 10:05:00 | 289.35 | 293.70 | 0.00 | ORB-short ORB[296.50,300.35] vol=2.1x ATR=1.71 |
| Stop hit — per-position SL triggered | 2025-03-03 10:25:00 | 291.06 | 292.80 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-03-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 11:00:00 | 334.15 | 330.92 | 0.00 | ORB-long ORB[328.80,333.00] vol=2.4x ATR=1.31 |
| Stop hit — per-position SL triggered | 2025-03-07 11:15:00 | 332.84 | 331.06 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-03-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:50:00 | 310.80 | 308.84 | 0.00 | ORB-long ORB[307.45,309.80] vol=1.6x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 10:10:00 | 312.75 | 309.72 | 0.00 | T1 1.5R @ 312.75 |
| Stop hit — per-position SL triggered | 2025-03-18 10:35:00 | 310.80 | 309.92 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-03-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:35:00 | 330.00 | 328.73 | 0.00 | ORB-long ORB[326.40,329.45] vol=3.0x ATR=1.28 |
| Stop hit — per-position SL triggered | 2025-03-21 09:40:00 | 328.72 | 328.77 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-03-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 10:20:00 | 342.85 | 339.59 | 0.00 | ORB-long ORB[335.30,340.15] vol=2.9x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-26 10:50:00 | 345.04 | 340.53 | 0.00 | T1 1.5R @ 345.04 |
| Stop hit — per-position SL triggered | 2025-03-26 13:10:00 | 342.85 | 343.01 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-04-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-11 11:05:00 | 307.70 | 305.27 | 0.00 | ORB-long ORB[302.20,306.25] vol=2.5x ATR=0.93 |
| Stop hit — per-position SL triggered | 2025-04-11 11:40:00 | 306.77 | 305.53 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-13 10:45:00 | 331.35 | 2024-05-13 11:05:00 | 333.74 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest1 | 2024-05-15 10:25:00 | 339.20 | 2024-05-15 10:45:00 | 340.92 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2024-05-16 11:15:00 | 335.95 | 2024-05-16 12:05:00 | 337.05 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-05-24 09:35:00 | 355.40 | 2024-05-24 09:40:00 | 356.61 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-05-28 09:30:00 | 363.95 | 2024-05-28 09:35:00 | 367.12 | PARTIAL | 0.50 | 0.87% |
| BUY | retest1 | 2024-05-28 09:30:00 | 363.95 | 2024-05-28 09:55:00 | 365.70 | TARGET_HIT | 0.50 | 0.48% |
| BUY | retest1 | 2024-05-30 10:45:00 | 365.35 | 2024-05-30 10:50:00 | 363.89 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-06-07 09:30:00 | 358.25 | 2024-06-07 09:45:00 | 356.42 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-06-10 11:15:00 | 368.40 | 2024-06-10 12:15:00 | 370.51 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-06-10 11:15:00 | 368.40 | 2024-06-10 12:30:00 | 368.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-12 09:45:00 | 375.60 | 2024-06-12 10:00:00 | 374.29 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-06-20 09:45:00 | 385.90 | 2024-06-20 10:55:00 | 383.53 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-06-20 09:45:00 | 385.90 | 2024-06-20 11:10:00 | 385.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-09 09:40:00 | 402.00 | 2024-07-09 10:00:00 | 399.87 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-07-09 09:40:00 | 402.00 | 2024-07-09 13:40:00 | 394.00 | TARGET_HIT | 0.50 | 1.99% |
| SELL | retest1 | 2024-07-10 09:45:00 | 386.50 | 2024-07-10 10:00:00 | 384.05 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-07-10 09:45:00 | 386.50 | 2024-07-10 11:30:00 | 384.90 | TARGET_HIT | 0.50 | 0.41% |
| BUY | retest1 | 2024-07-16 10:00:00 | 391.95 | 2024-07-16 10:05:00 | 389.99 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-07-23 11:00:00 | 366.00 | 2024-07-23 11:10:00 | 367.14 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-07-26 10:05:00 | 359.55 | 2024-07-26 10:40:00 | 361.16 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-07-31 10:00:00 | 374.70 | 2024-07-31 10:05:00 | 372.58 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-07-31 10:00:00 | 374.70 | 2024-07-31 10:20:00 | 374.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-01 10:10:00 | 385.70 | 2024-08-01 10:45:00 | 384.35 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-08-22 09:35:00 | 330.95 | 2024-08-22 09:40:00 | 332.29 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-08-23 09:45:00 | 333.00 | 2024-08-23 09:55:00 | 334.41 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-08-26 10:05:00 | 335.55 | 2024-08-26 10:10:00 | 334.21 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-08-29 10:15:00 | 336.75 | 2024-08-29 11:50:00 | 334.74 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-08-29 10:15:00 | 336.75 | 2024-08-29 12:25:00 | 336.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-05 11:05:00 | 337.60 | 2024-09-05 11:10:00 | 338.55 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-09-11 09:40:00 | 370.90 | 2024-09-11 09:55:00 | 369.49 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-09-16 10:55:00 | 360.20 | 2024-09-16 11:00:00 | 358.83 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-09-19 09:30:00 | 345.30 | 2024-09-19 09:35:00 | 346.92 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-09-23 11:00:00 | 348.05 | 2024-09-23 11:20:00 | 350.07 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-09-23 11:00:00 | 348.05 | 2024-09-23 12:00:00 | 348.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-27 11:00:00 | 350.60 | 2024-09-27 11:20:00 | 351.78 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-10-01 11:15:00 | 362.80 | 2024-10-01 11:20:00 | 361.53 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-10-07 11:15:00 | 343.95 | 2024-10-07 13:00:00 | 340.48 | PARTIAL | 0.50 | 1.01% |
| SELL | retest1 | 2024-10-07 11:15:00 | 343.95 | 2024-10-07 15:20:00 | 340.60 | TARGET_HIT | 0.50 | 0.97% |
| BUY | retest1 | 2024-10-09 10:55:00 | 348.10 | 2024-10-09 11:15:00 | 349.95 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-10-09 10:55:00 | 348.10 | 2024-10-09 12:20:00 | 348.40 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2024-11-25 09:45:00 | 406.65 | 2024-11-25 09:50:00 | 404.29 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2024-12-09 10:05:00 | 411.45 | 2024-12-09 10:10:00 | 410.00 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-12-10 11:05:00 | 411.20 | 2024-12-10 11:20:00 | 413.27 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-12-10 11:05:00 | 411.20 | 2024-12-10 11:40:00 | 411.90 | TARGET_HIT | 0.50 | 0.17% |
| SELL | retest1 | 2024-12-13 11:10:00 | 393.20 | 2024-12-13 11:50:00 | 394.94 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-12-16 10:45:00 | 395.00 | 2024-12-16 11:05:00 | 392.72 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-12-16 10:45:00 | 395.00 | 2024-12-16 15:20:00 | 385.00 | TARGET_HIT | 0.50 | 2.53% |
| BUY | retest1 | 2024-12-19 11:10:00 | 376.50 | 2024-12-19 12:00:00 | 375.21 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-12-24 10:55:00 | 378.90 | 2024-12-24 11:35:00 | 377.90 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-01-03 10:55:00 | 375.40 | 2025-01-03 11:05:00 | 373.24 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-01-03 10:55:00 | 375.40 | 2025-01-03 15:20:00 | 367.00 | TARGET_HIT | 0.50 | 2.24% |
| BUY | retest1 | 2025-01-07 10:55:00 | 378.65 | 2025-01-07 11:25:00 | 376.67 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2025-01-15 09:30:00 | 348.10 | 2025-01-15 09:35:00 | 350.13 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2025-01-21 11:05:00 | 353.60 | 2025-01-21 11:25:00 | 351.35 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2025-01-21 11:05:00 | 353.60 | 2025-01-21 11:45:00 | 353.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-23 10:40:00 | 350.10 | 2025-01-23 10:50:00 | 348.76 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-02-01 10:40:00 | 334.95 | 2025-02-01 11:35:00 | 337.15 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest1 | 2025-02-04 10:30:00 | 321.70 | 2025-02-04 10:50:00 | 319.13 | PARTIAL | 0.50 | 0.80% |
| SELL | retest1 | 2025-02-04 10:30:00 | 321.70 | 2025-02-04 11:45:00 | 321.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-03 10:05:00 | 289.35 | 2025-03-03 10:25:00 | 291.06 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2025-03-07 11:00:00 | 334.15 | 2025-03-07 11:15:00 | 332.84 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-03-18 09:50:00 | 310.80 | 2025-03-18 10:10:00 | 312.75 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-03-18 09:50:00 | 310.80 | 2025-03-18 10:35:00 | 310.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-21 09:35:00 | 330.00 | 2025-03-21 09:40:00 | 328.72 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-03-26 10:20:00 | 342.85 | 2025-03-26 10:50:00 | 345.04 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-03-26 10:20:00 | 342.85 | 2025-03-26 13:10:00 | 342.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-11 11:05:00 | 307.70 | 2025-04-11 11:40:00 | 306.77 | STOP_HIT | 1.00 | -0.30% |
