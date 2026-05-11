# KOTAKBANK (KOTAKBANK)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 381.00
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
| ENTRY1 | 65 |
| ENTRY2 | 0 |
| PARTIAL | 24 |
| TARGET_HIT | 9 |
| STOP_HIT | 56 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 89 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 56
- **Target hits / Stop hits / Partials:** 9 / 56 / 24
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 9.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 56 | 25 | 44.6% | 7 | 31 | 18 | 0.19% | 10.6% |
| BUY @ 2nd Alert (retest1) | 56 | 25 | 44.6% | 7 | 31 | 18 | 0.19% | 10.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 33 | 8 | 24.2% | 2 | 25 | 6 | -0.03% | -0.9% |
| SELL @ 2nd Alert (retest1) | 33 | 8 | 24.2% | 2 | 25 | 6 | -0.03% | -0.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 89 | 33 | 37.1% | 9 | 56 | 24 | 0.11% | 9.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 11:05:00 | 323.86 | 325.20 | 0.00 | ORB-short ORB[324.78,326.85] vol=1.6x ATR=0.87 |
| Stop hit — per-position SL triggered | 2024-05-13 11:30:00 | 324.73 | 325.04 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 10:55:00 | 329.49 | 328.58 | 0.00 | ORB-long ORB[326.50,329.28] vol=2.4x ATR=0.68 |
| Stop hit — per-position SL triggered | 2024-05-14 12:10:00 | 328.81 | 328.73 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 10:30:00 | 337.09 | 335.59 | 0.00 | ORB-long ORB[333.00,335.57] vol=1.6x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-17 11:35:00 | 338.23 | 336.63 | 0.00 | T1 1.5R @ 338.23 |
| Target hit | 2024-05-17 15:20:00 | 339.48 | 338.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2024-05-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 11:00:00 | 336.61 | 337.97 | 0.00 | ORB-short ORB[338.66,340.99] vol=1.7x ATR=0.65 |
| Stop hit — per-position SL triggered | 2024-05-22 11:05:00 | 337.26 | 337.91 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-31 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 10:40:00 | 334.84 | 336.59 | 0.00 | ORB-short ORB[338.44,340.60] vol=1.9x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 11:25:00 | 333.28 | 335.95 | 0.00 | T1 1.5R @ 333.28 |
| Stop hit — per-position SL triggered | 2024-05-31 13:20:00 | 334.84 | 335.14 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-05 09:35:00 | 334.71 | 332.33 | 0.00 | ORB-long ORB[329.00,332.99] vol=1.5x ATR=2.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-05 10:55:00 | 338.44 | 334.28 | 0.00 | T1 1.5R @ 338.44 |
| Target hit | 2024-06-05 15:20:00 | 343.09 | 339.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2024-06-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 11:10:00 | 346.88 | 343.78 | 0.00 | ORB-long ORB[341.44,344.79] vol=3.2x ATR=0.96 |
| Stop hit — per-position SL triggered | 2024-06-06 11:30:00 | 345.92 | 343.90 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 11:05:00 | 348.31 | 346.31 | 0.00 | ORB-long ORB[344.40,346.99] vol=1.5x ATR=1.04 |
| Stop hit — per-position SL triggered | 2024-06-07 11:25:00 | 347.27 | 346.49 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-19 09:50:00 | 348.66 | 346.58 | 0.00 | ORB-long ORB[343.00,346.68] vol=1.5x ATR=0.78 |
| Stop hit — per-position SL triggered | 2024-06-19 10:00:00 | 347.88 | 347.10 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 10:45:00 | 349.52 | 350.58 | 0.00 | ORB-short ORB[349.75,354.18] vol=1.7x ATR=0.76 |
| Stop hit — per-position SL triggered | 2024-06-21 10:55:00 | 350.28 | 350.51 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 11:00:00 | 360.29 | 359.05 | 0.00 | ORB-long ORB[355.22,359.36] vol=2.8x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 11:30:00 | 361.48 | 359.46 | 0.00 | T1 1.5R @ 361.48 |
| Stop hit — per-position SL triggered | 2024-06-26 11:40:00 | 360.29 | 359.49 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 11:00:00 | 369.21 | 371.33 | 0.00 | ORB-short ORB[370.66,374.00] vol=1.6x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 11:05:00 | 368.27 | 370.36 | 0.00 | T1 1.5R @ 368.27 |
| Stop hit — per-position SL triggered | 2024-07-09 13:55:00 | 369.21 | 369.02 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 10:30:00 | 365.93 | 367.29 | 0.00 | ORB-short ORB[367.58,369.78] vol=1.5x ATR=0.94 |
| Stop hit — per-position SL triggered | 2024-07-12 10:55:00 | 366.87 | 367.00 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 10:30:00 | 351.67 | 352.43 | 0.00 | ORB-short ORB[351.84,354.39] vol=1.8x ATR=0.81 |
| Stop hit — per-position SL triggered | 2024-07-23 10:55:00 | 352.48 | 352.37 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-08-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 10:40:00 | 353.64 | 355.50 | 0.00 | ORB-short ORB[355.96,358.44] vol=1.9x ATR=0.76 |
| Stop hit — per-position SL triggered | 2024-08-06 10:50:00 | 354.40 | 355.38 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-08-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-07 11:00:00 | 352.77 | 353.41 | 0.00 | ORB-short ORB[352.92,357.32] vol=1.6x ATR=0.87 |
| Stop hit — per-position SL triggered | 2024-08-07 11:10:00 | 353.64 | 353.37 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 10:15:00 | 354.38 | 352.74 | 0.00 | ORB-long ORB[350.25,353.29] vol=2.3x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 10:30:00 | 355.57 | 353.35 | 0.00 | T1 1.5R @ 355.57 |
| Target hit | 2024-08-12 14:50:00 | 355.70 | 356.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — SELL (started 2024-08-22 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 10:40:00 | 361.51 | 362.29 | 0.00 | ORB-short ORB[361.69,363.27] vol=1.7x ATR=0.65 |
| Stop hit — per-position SL triggered | 2024-08-22 11:25:00 | 362.16 | 362.08 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-09-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 11:10:00 | 355.08 | 356.06 | 0.00 | ORB-short ORB[355.34,356.95] vol=2.0x ATR=0.50 |
| Stop hit — per-position SL triggered | 2024-09-03 12:00:00 | 355.58 | 355.82 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-09-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-09 10:40:00 | 354.89 | 353.30 | 0.00 | ORB-long ORB[351.69,353.79] vol=1.7x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 11:00:00 | 355.94 | 353.70 | 0.00 | T1 1.5R @ 355.94 |
| Stop hit — per-position SL triggered | 2024-09-09 11:35:00 | 354.89 | 353.94 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-09-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 10:35:00 | 360.79 | 358.83 | 0.00 | ORB-long ORB[357.00,359.00] vol=1.6x ATR=0.62 |
| Stop hit — per-position SL triggered | 2024-09-11 10:50:00 | 360.17 | 359.03 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-09-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 10:40:00 | 370.84 | 369.47 | 0.00 | ORB-long ORB[369.02,370.59] vol=1.6x ATR=0.66 |
| Stop hit — per-position SL triggered | 2024-09-18 10:55:00 | 370.18 | 369.66 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-09-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 10:35:00 | 376.97 | 376.53 | 0.00 | ORB-long ORB[374.00,375.86] vol=1.6x ATR=0.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-20 10:45:00 | 378.08 | 376.73 | 0.00 | T1 1.5R @ 378.08 |
| Stop hit — per-position SL triggered | 2024-09-20 13:35:00 | 376.97 | 378.03 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-09-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 11:00:00 | 382.37 | 384.99 | 0.00 | ORB-short ORB[384.90,387.72] vol=4.0x ATR=0.70 |
| Stop hit — per-position SL triggered | 2024-09-24 11:10:00 | 383.07 | 384.84 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-09-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 10:55:00 | 379.86 | 380.97 | 0.00 | ORB-short ORB[380.91,384.54] vol=5.1x ATR=0.81 |
| Stop hit — per-position SL triggered | 2024-09-25 13:20:00 | 380.67 | 380.52 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-09-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-27 11:00:00 | 377.14 | 378.94 | 0.00 | ORB-short ORB[379.33,382.00] vol=2.9x ATR=0.59 |
| Stop hit — per-position SL triggered | 2024-09-27 11:10:00 | 377.73 | 378.78 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-10-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 10:10:00 | 362.66 | 360.00 | 0.00 | ORB-long ORB[358.21,360.66] vol=1.8x ATR=1.11 |
| Stop hit — per-position SL triggered | 2024-10-09 10:30:00 | 361.55 | 360.96 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-10-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 10:35:00 | 365.34 | 364.86 | 0.00 | ORB-long ORB[361.06,362.58] vol=1.6x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-10 10:55:00 | 366.83 | 365.01 | 0.00 | T1 1.5R @ 366.83 |
| Target hit | 2024-10-10 15:20:00 | 374.50 | 372.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — BUY (started 2024-10-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:55:00 | 376.08 | 374.45 | 0.00 | ORB-long ORB[372.82,375.40] vol=1.8x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-11 10:25:00 | 377.59 | 375.80 | 0.00 | T1 1.5R @ 377.59 |
| Stop hit — per-position SL triggered | 2024-10-11 10:55:00 | 376.08 | 376.16 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-10-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 11:00:00 | 382.29 | 380.21 | 0.00 | ORB-long ORB[376.49,378.99] vol=1.9x ATR=0.73 |
| Stop hit — per-position SL triggered | 2024-10-14 11:45:00 | 381.56 | 380.82 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-10-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:40:00 | 373.04 | 374.07 | 0.00 | ORB-short ORB[373.41,375.66] vol=2.0x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 09:55:00 | 371.91 | 373.85 | 0.00 | T1 1.5R @ 371.91 |
| Stop hit — per-position SL triggered | 2024-10-17 10:45:00 | 373.04 | 372.95 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 11:15:00 | 352.39 | 350.49 | 0.00 | ORB-long ORB[347.44,350.05] vol=2.4x ATR=0.90 |
| Stop hit — per-position SL triggered | 2024-11-11 11:55:00 | 351.49 | 350.67 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-11-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 11:05:00 | 347.23 | 345.28 | 0.00 | ORB-long ORB[342.70,346.10] vol=2.1x ATR=0.69 |
| Stop hit — per-position SL triggered | 2024-11-19 11:15:00 | 346.54 | 345.39 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-26 11:15:00 | 355.57 | 357.32 | 0.00 | ORB-short ORB[356.98,359.19] vol=1.9x ATR=0.76 |
| Stop hit — per-position SL triggered | 2024-11-26 11:25:00 | 356.33 | 357.17 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-11-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:35:00 | 354.77 | 355.92 | 0.00 | ORB-short ORB[355.04,357.78] vol=1.9x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 11:15:00 | 353.56 | 355.39 | 0.00 | T1 1.5R @ 353.56 |
| Target hit | 2024-11-28 15:20:00 | 352.42 | 353.04 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — SELL (started 2024-12-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-02 10:50:00 | 350.23 | 352.29 | 0.00 | ORB-short ORB[351.33,354.09] vol=2.6x ATR=0.74 |
| Stop hit — per-position SL triggered | 2024-12-02 11:05:00 | 350.97 | 351.72 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-12-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-03 10:50:00 | 348.55 | 349.21 | 0.00 | ORB-short ORB[348.72,352.24] vol=2.6x ATR=0.65 |
| Stop hit — per-position SL triggered | 2024-12-03 11:00:00 | 349.20 | 349.20 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-12-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 10:55:00 | 352.98 | 351.41 | 0.00 | ORB-long ORB[349.28,351.00] vol=1.7x ATR=0.54 |
| Stop hit — per-position SL triggered | 2024-12-04 11:30:00 | 352.44 | 351.73 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 350.39 | 351.36 | 0.00 | ORB-short ORB[350.50,352.87] vol=2.4x ATR=0.65 |
| Stop hit — per-position SL triggered | 2024-12-05 12:05:00 | 351.04 | 351.11 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 11:15:00 | 359.60 | 357.92 | 0.00 | ORB-long ORB[353.62,358.39] vol=1.7x ATR=0.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 11:25:00 | 360.47 | 358.15 | 0.00 | T1 1.5R @ 360.47 |
| Stop hit — per-position SL triggered | 2024-12-09 12:00:00 | 359.60 | 358.61 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-12-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 11:05:00 | 357.09 | 357.59 | 0.00 | ORB-short ORB[357.45,359.29] vol=2.0x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 11:15:00 | 356.29 | 357.48 | 0.00 | T1 1.5R @ 356.29 |
| Target hit | 2024-12-12 15:20:00 | 354.01 | 354.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — BUY (started 2024-12-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 09:50:00 | 353.48 | 352.26 | 0.00 | ORB-long ORB[351.12,352.71] vol=1.6x ATR=0.78 |
| Stop hit — per-position SL triggered | 2024-12-27 10:05:00 | 352.70 | 352.37 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-12-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-30 09:40:00 | 350.01 | 350.74 | 0.00 | ORB-short ORB[350.02,353.69] vol=2.2x ATR=0.84 |
| Stop hit — per-position SL triggered | 2024-12-30 09:50:00 | 350.85 | 350.68 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-01-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-14 10:50:00 | 348.36 | 349.09 | 0.00 | ORB-short ORB[348.73,350.54] vol=4.6x ATR=0.69 |
| Stop hit — per-position SL triggered | 2025-01-14 10:55:00 | 349.05 | 349.06 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 11:15:00 | 379.20 | 380.82 | 0.00 | ORB-short ORB[380.26,384.19] vol=1.7x ATR=0.81 |
| Stop hit — per-position SL triggered | 2025-01-21 11:40:00 | 380.01 | 380.73 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-01-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-24 11:15:00 | 380.55 | 377.21 | 0.00 | ORB-long ORB[375.84,380.20] vol=1.7x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 11:25:00 | 381.92 | 377.62 | 0.00 | T1 1.5R @ 381.92 |
| Stop hit — per-position SL triggered | 2025-01-24 12:00:00 | 380.55 | 378.30 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:30:00 | 386.46 | 384.98 | 0.00 | ORB-long ORB[382.48,385.19] vol=4.5x ATR=0.98 |
| Stop hit — per-position SL triggered | 2025-01-30 09:40:00 | 385.48 | 385.31 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-02-01 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 10:10:00 | 382.17 | 380.37 | 0.00 | ORB-long ORB[377.80,381.79] vol=1.6x ATR=1.03 |
| Stop hit — per-position SL triggered | 2025-02-01 10:35:00 | 381.14 | 380.61 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-02-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 10:05:00 | 384.51 | 383.60 | 0.00 | ORB-long ORB[382.69,384.20] vol=2.1x ATR=0.63 |
| Stop hit — per-position SL triggered | 2025-02-07 10:10:00 | 383.88 | 383.60 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-02-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-11 10:35:00 | 384.05 | 388.20 | 0.00 | ORB-short ORB[389.44,394.10] vol=2.0x ATR=1.11 |
| Stop hit — per-position SL triggered | 2025-02-11 10:45:00 | 385.16 | 387.96 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-02-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-20 09:45:00 | 393.53 | 394.71 | 0.00 | ORB-short ORB[394.54,396.47] vol=4.5x ATR=0.87 |
| Stop hit — per-position SL triggered | 2025-02-20 10:00:00 | 394.40 | 394.61 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-03-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 10:45:00 | 385.78 | 384.27 | 0.00 | ORB-long ORB[379.08,383.00] vol=2.2x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-05 11:10:00 | 386.92 | 384.73 | 0.00 | T1 1.5R @ 386.92 |
| Target hit | 2025-03-05 15:00:00 | 386.63 | 387.14 | 0.00 | Trail-exit close<VWAP |

### Cycle 53 — BUY (started 2025-03-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-12 09:30:00 | 393.31 | 391.31 | 0.00 | ORB-long ORB[387.60,392.38] vol=2.1x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 09:55:00 | 394.89 | 392.41 | 0.00 | T1 1.5R @ 394.89 |
| Target hit | 2025-03-12 11:35:00 | 395.86 | 395.92 | 0.00 | Trail-exit close<VWAP |

### Cycle 54 — BUY (started 2025-03-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 09:40:00 | 397.41 | 396.22 | 0.00 | ORB-long ORB[393.91,396.67] vol=1.8x ATR=0.75 |
| Stop hit — per-position SL triggered | 2025-03-13 09:50:00 | 396.66 | 396.39 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-03-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:40:00 | 402.30 | 400.34 | 0.00 | ORB-long ORB[398.62,401.93] vol=1.6x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 10:50:00 | 403.47 | 400.71 | 0.00 | T1 1.5R @ 403.47 |
| Target hit | 2025-03-18 15:20:00 | 407.10 | 404.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — BUY (started 2025-03-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 10:30:00 | 409.61 | 408.70 | 0.00 | ORB-long ORB[406.37,409.40] vol=2.7x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-03-19 10:45:00 | 408.75 | 408.75 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-03-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 11:00:00 | 407.06 | 405.39 | 0.00 | ORB-long ORB[405.06,406.72] vol=1.5x ATR=0.69 |
| Stop hit — per-position SL triggered | 2025-03-20 11:45:00 | 406.37 | 405.73 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-03-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:30:00 | 411.29 | 409.68 | 0.00 | ORB-long ORB[407.00,410.82] vol=1.7x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 09:40:00 | 412.42 | 410.52 | 0.00 | T1 1.5R @ 412.42 |
| Stop hit — per-position SL triggered | 2025-03-21 10:00:00 | 411.29 | 410.84 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-03-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 09:35:00 | 439.00 | 437.97 | 0.00 | ORB-long ORB[435.23,438.04] vol=1.6x ATR=1.11 |
| Stop hit — per-position SL triggered | 2025-03-25 09:45:00 | 437.89 | 438.19 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-04-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-04 11:05:00 | 426.89 | 426.13 | 0.00 | ORB-long ORB[423.63,426.52] vol=3.6x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 11:45:00 | 428.46 | 426.39 | 0.00 | T1 1.5R @ 428.46 |
| Stop hit — per-position SL triggered | 2025-04-04 12:15:00 | 426.89 | 426.47 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-04-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-11 10:45:00 | 423.36 | 418.33 | 0.00 | ORB-long ORB[411.70,417.07] vol=2.3x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-11 11:10:00 | 425.17 | 419.72 | 0.00 | T1 1.5R @ 425.17 |
| Stop hit — per-position SL triggered | 2025-04-11 12:35:00 | 423.36 | 422.60 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:35:00 | 441.00 | 439.40 | 0.00 | ORB-long ORB[435.08,440.38] vol=1.8x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 09:40:00 | 442.89 | 439.86 | 0.00 | T1 1.5R @ 442.89 |
| Stop hit — per-position SL triggered | 2025-04-21 09:50:00 | 441.00 | 440.39 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-04-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 09:50:00 | 457.98 | 454.22 | 0.00 | ORB-long ORB[449.08,455.00] vol=1.5x ATR=1.33 |
| Stop hit — per-position SL triggered | 2025-04-22 10:00:00 | 456.65 | 455.49 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-04-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 11:05:00 | 444.28 | 443.48 | 0.00 | ORB-long ORB[439.60,443.78] vol=1.8x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 11:25:00 | 445.82 | 443.65 | 0.00 | T1 1.5R @ 445.82 |
| Stop hit — per-position SL triggered | 2025-04-30 11:35:00 | 444.28 | 443.68 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-05-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-02 10:50:00 | 438.26 | 441.59 | 0.00 | ORB-short ORB[439.76,444.40] vol=1.9x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-02 12:15:00 | 436.27 | 440.14 | 0.00 | T1 1.5R @ 436.27 |
| Stop hit — per-position SL triggered | 2025-05-02 13:20:00 | 438.26 | 439.27 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-13 11:05:00 | 323.86 | 2024-05-13 11:30:00 | 324.73 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-05-14 10:55:00 | 329.49 | 2024-05-14 12:10:00 | 328.81 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-05-17 10:30:00 | 337.09 | 2024-05-17 11:35:00 | 338.23 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-05-17 10:30:00 | 337.09 | 2024-05-17 15:20:00 | 339.48 | TARGET_HIT | 0.50 | 0.71% |
| SELL | retest1 | 2024-05-22 11:00:00 | 336.61 | 2024-05-22 11:05:00 | 337.26 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-05-31 10:40:00 | 334.84 | 2024-05-31 11:25:00 | 333.28 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-05-31 10:40:00 | 334.84 | 2024-05-31 13:20:00 | 334.84 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-05 09:35:00 | 334.71 | 2024-06-05 10:55:00 | 338.44 | PARTIAL | 0.50 | 1.11% |
| BUY | retest1 | 2024-06-05 09:35:00 | 334.71 | 2024-06-05 15:20:00 | 343.09 | TARGET_HIT | 0.50 | 2.50% |
| BUY | retest1 | 2024-06-06 11:10:00 | 346.88 | 2024-06-06 11:30:00 | 345.92 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-06-07 11:05:00 | 348.31 | 2024-06-07 11:25:00 | 347.27 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-06-19 09:50:00 | 348.66 | 2024-06-19 10:00:00 | 347.88 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-06-21 10:45:00 | 349.52 | 2024-06-21 10:55:00 | 350.28 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-06-26 11:00:00 | 360.29 | 2024-06-26 11:30:00 | 361.48 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-06-26 11:00:00 | 360.29 | 2024-06-26 11:40:00 | 360.29 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-09 11:00:00 | 369.21 | 2024-07-09 11:05:00 | 368.27 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2024-07-09 11:00:00 | 369.21 | 2024-07-09 13:55:00 | 369.21 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-12 10:30:00 | 365.93 | 2024-07-12 10:55:00 | 366.87 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-07-23 10:30:00 | 351.67 | 2024-07-23 10:55:00 | 352.48 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-08-06 10:40:00 | 353.64 | 2024-08-06 10:50:00 | 354.40 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-08-07 11:00:00 | 352.77 | 2024-08-07 11:10:00 | 353.64 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-08-12 10:15:00 | 354.38 | 2024-08-12 10:30:00 | 355.57 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-08-12 10:15:00 | 354.38 | 2024-08-12 14:50:00 | 355.70 | TARGET_HIT | 0.50 | 0.37% |
| SELL | retest1 | 2024-08-22 10:40:00 | 361.51 | 2024-08-22 11:25:00 | 362.16 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-09-03 11:10:00 | 355.08 | 2024-09-03 12:00:00 | 355.58 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2024-09-09 10:40:00 | 354.89 | 2024-09-09 11:00:00 | 355.94 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-09-09 10:40:00 | 354.89 | 2024-09-09 11:35:00 | 354.89 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-11 10:35:00 | 360.79 | 2024-09-11 10:50:00 | 360.17 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2024-09-18 10:40:00 | 370.84 | 2024-09-18 10:55:00 | 370.18 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-09-20 10:35:00 | 376.97 | 2024-09-20 10:45:00 | 378.08 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-09-20 10:35:00 | 376.97 | 2024-09-20 13:35:00 | 376.97 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-24 11:00:00 | 382.37 | 2024-09-24 11:10:00 | 383.07 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-09-25 10:55:00 | 379.86 | 2024-09-25 13:20:00 | 380.67 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-09-27 11:00:00 | 377.14 | 2024-09-27 11:10:00 | 377.73 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2024-10-09 10:10:00 | 362.66 | 2024-10-09 10:30:00 | 361.55 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-10-10 10:35:00 | 365.34 | 2024-10-10 10:55:00 | 366.83 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2024-10-10 10:35:00 | 365.34 | 2024-10-10 15:20:00 | 374.50 | TARGET_HIT | 0.50 | 2.51% |
| BUY | retest1 | 2024-10-11 09:55:00 | 376.08 | 2024-10-11 10:25:00 | 377.59 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-10-11 09:55:00 | 376.08 | 2024-10-11 10:55:00 | 376.08 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-14 11:00:00 | 382.29 | 2024-10-14 11:45:00 | 381.56 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-10-17 09:40:00 | 373.04 | 2024-10-17 09:55:00 | 371.91 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2024-10-17 09:40:00 | 373.04 | 2024-10-17 10:45:00 | 373.04 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-11 11:15:00 | 352.39 | 2024-11-11 11:55:00 | 351.49 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-11-19 11:05:00 | 347.23 | 2024-11-19 11:15:00 | 346.54 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-11-26 11:15:00 | 355.57 | 2024-11-26 11:25:00 | 356.33 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-11-28 10:35:00 | 354.77 | 2024-11-28 11:15:00 | 353.56 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-11-28 10:35:00 | 354.77 | 2024-11-28 15:20:00 | 352.42 | TARGET_HIT | 0.50 | 0.66% |
| SELL | retest1 | 2024-12-02 10:50:00 | 350.23 | 2024-12-02 11:05:00 | 350.97 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-12-03 10:50:00 | 348.55 | 2024-12-03 11:00:00 | 349.20 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-12-04 10:55:00 | 352.98 | 2024-12-04 11:30:00 | 352.44 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2024-12-05 10:55:00 | 350.39 | 2024-12-05 12:05:00 | 351.04 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-12-09 11:15:00 | 359.60 | 2024-12-09 11:25:00 | 360.47 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2024-12-09 11:15:00 | 359.60 | 2024-12-09 12:00:00 | 359.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-12 11:05:00 | 357.09 | 2024-12-12 11:15:00 | 356.29 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2024-12-12 11:05:00 | 357.09 | 2024-12-12 15:20:00 | 354.01 | TARGET_HIT | 0.50 | 0.86% |
| BUY | retest1 | 2024-12-27 09:50:00 | 353.48 | 2024-12-27 10:05:00 | 352.70 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-30 09:40:00 | 350.01 | 2024-12-30 09:50:00 | 350.85 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-01-14 10:50:00 | 348.36 | 2025-01-14 10:55:00 | 349.05 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-01-21 11:15:00 | 379.20 | 2025-01-21 11:40:00 | 380.01 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-01-24 11:15:00 | 380.55 | 2025-01-24 11:25:00 | 381.92 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-01-24 11:15:00 | 380.55 | 2025-01-24 12:00:00 | 380.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-30 09:30:00 | 386.46 | 2025-01-30 09:40:00 | 385.48 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-02-01 10:10:00 | 382.17 | 2025-02-01 10:35:00 | 381.14 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-02-07 10:05:00 | 384.51 | 2025-02-07 10:10:00 | 383.88 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-02-11 10:35:00 | 384.05 | 2025-02-11 10:45:00 | 385.16 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-02-20 09:45:00 | 393.53 | 2025-02-20 10:00:00 | 394.40 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-03-05 10:45:00 | 385.78 | 2025-03-05 11:10:00 | 386.92 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-03-05 10:45:00 | 385.78 | 2025-03-05 15:00:00 | 386.63 | TARGET_HIT | 0.50 | 0.22% |
| BUY | retest1 | 2025-03-12 09:30:00 | 393.31 | 2025-03-12 09:55:00 | 394.89 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-03-12 09:30:00 | 393.31 | 2025-03-12 11:35:00 | 395.86 | TARGET_HIT | 0.50 | 0.65% |
| BUY | retest1 | 2025-03-13 09:40:00 | 397.41 | 2025-03-13 09:50:00 | 396.66 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-03-18 10:40:00 | 402.30 | 2025-03-18 10:50:00 | 403.47 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2025-03-18 10:40:00 | 402.30 | 2025-03-18 15:20:00 | 407.10 | TARGET_HIT | 0.50 | 1.19% |
| BUY | retest1 | 2025-03-19 10:30:00 | 409.61 | 2025-03-19 10:45:00 | 408.75 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-03-20 11:00:00 | 407.06 | 2025-03-20 11:45:00 | 406.37 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-03-21 09:30:00 | 411.29 | 2025-03-21 09:40:00 | 412.42 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2025-03-21 09:30:00 | 411.29 | 2025-03-21 10:00:00 | 411.29 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-25 09:35:00 | 439.00 | 2025-03-25 09:45:00 | 437.89 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-04-04 11:05:00 | 426.89 | 2025-04-04 11:45:00 | 428.46 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-04-04 11:05:00 | 426.89 | 2025-04-04 12:15:00 | 426.89 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-11 10:45:00 | 423.36 | 2025-04-11 11:10:00 | 425.17 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-04-11 10:45:00 | 423.36 | 2025-04-11 12:35:00 | 423.36 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-21 09:35:00 | 441.00 | 2025-04-21 09:40:00 | 442.89 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-04-21 09:35:00 | 441.00 | 2025-04-21 09:50:00 | 441.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-22 09:50:00 | 457.98 | 2025-04-22 10:00:00 | 456.65 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-04-30 11:05:00 | 444.28 | 2025-04-30 11:25:00 | 445.82 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-04-30 11:05:00 | 444.28 | 2025-04-30 11:35:00 | 444.28 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-02 10:50:00 | 438.26 | 2025-05-02 12:15:00 | 436.27 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-05-02 10:50:00 | 438.26 | 2025-05-02 13:20:00 | 438.26 | STOP_HIT | 0.50 | 0.00% |
