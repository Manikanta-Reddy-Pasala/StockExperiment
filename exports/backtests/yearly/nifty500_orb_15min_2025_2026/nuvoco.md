# Nuvoco Vistas Corporation Ltd. (NUVOCO)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 328.90
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
| ENTRY1 | 79 |
| ENTRY2 | 0 |
| PARTIAL | 29 |
| TARGET_HIT | 7 |
| STOP_HIT | 72 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 108 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 36 / 72
- **Target hits / Stop hits / Partials:** 7 / 72 / 29
- **Avg / median % per leg:** 0.01% / 0.00%
- **Sum % (uncompounded):** 1.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 52 | 13 | 25.0% | 4 | 39 | 9 | -0.14% | -7.4% |
| BUY @ 2nd Alert (retest1) | 52 | 13 | 25.0% | 4 | 39 | 9 | -0.14% | -7.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 56 | 23 | 41.1% | 3 | 33 | 20 | 0.16% | 8.9% |
| SELL @ 2nd Alert (retest1) | 56 | 23 | 41.1% | 3 | 33 | 20 | 0.16% | 8.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 108 | 36 | 33.3% | 7 | 72 | 29 | 0.01% | 1.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 10:50:00 | 352.90 | 349.76 | 0.00 | ORB-long ORB[343.05,347.60] vol=2.1x ATR=1.55 |
| Stop hit — per-position SL triggered | 2025-05-14 11:00:00 | 351.35 | 350.02 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-19 10:45:00 | 357.50 | 357.34 | 0.00 | ORB-long ORB[355.00,357.40] vol=1.6x ATR=0.82 |
| Stop hit — per-position SL triggered | 2025-05-19 11:30:00 | 356.68 | 357.49 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-20 09:35:00 | 359.45 | 358.55 | 0.00 | ORB-long ORB[356.05,358.70] vol=1.6x ATR=1.06 |
| Stop hit — per-position SL triggered | 2025-05-20 09:40:00 | 358.39 | 358.53 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 09:35:00 | 359.70 | 359.21 | 0.00 | ORB-long ORB[355.60,359.25] vol=3.4x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-23 10:35:00 | 361.60 | 359.85 | 0.00 | T1 1.5R @ 361.60 |
| Target hit | 2025-05-23 12:15:00 | 360.45 | 361.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — SELL (started 2025-05-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-26 11:10:00 | 357.05 | 359.02 | 0.00 | ORB-short ORB[359.00,362.60] vol=2.6x ATR=0.84 |
| Stop hit — per-position SL triggered | 2025-05-26 11:15:00 | 357.89 | 358.98 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-05-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-28 10:25:00 | 354.00 | 355.52 | 0.00 | ORB-short ORB[355.30,357.15] vol=3.4x ATR=0.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 10:40:00 | 352.89 | 355.02 | 0.00 | T1 1.5R @ 352.89 |
| Stop hit — per-position SL triggered | 2025-05-28 11:00:00 | 354.00 | 354.55 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-05-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 09:35:00 | 351.25 | 352.48 | 0.00 | ORB-short ORB[351.60,355.60] vol=3.1x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-29 10:50:00 | 349.59 | 351.77 | 0.00 | T1 1.5R @ 349.59 |
| Stop hit — per-position SL triggered | 2025-05-29 12:55:00 | 351.25 | 351.45 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 09:35:00 | 353.45 | 355.36 | 0.00 | ORB-short ORB[353.80,358.80] vol=2.2x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 09:45:00 | 351.62 | 354.68 | 0.00 | T1 1.5R @ 351.62 |
| Stop hit — per-position SL triggered | 2025-06-04 09:55:00 | 353.45 | 354.32 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 09:40:00 | 360.00 | 358.35 | 0.00 | ORB-long ORB[355.00,359.80] vol=2.9x ATR=1.25 |
| Stop hit — per-position SL triggered | 2025-06-10 09:55:00 | 358.75 | 358.46 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 09:40:00 | 355.45 | 357.48 | 0.00 | ORB-short ORB[356.75,361.75] vol=2.0x ATR=1.00 |
| Stop hit — per-position SL triggered | 2025-06-12 09:55:00 | 356.45 | 356.97 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:30:00 | 346.50 | 349.33 | 0.00 | ORB-short ORB[348.10,353.10] vol=1.9x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 09:40:00 | 345.07 | 348.56 | 0.00 | T1 1.5R @ 345.07 |
| Stop hit — per-position SL triggered | 2025-06-16 10:05:00 | 346.50 | 347.00 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 11:15:00 | 344.25 | 343.59 | 0.00 | ORB-long ORB[338.00,342.70] vol=19.7x ATR=1.20 |
| Stop hit — per-position SL triggered | 2025-06-20 11:20:00 | 343.05 | 343.57 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 10:30:00 | 344.85 | 342.42 | 0.00 | ORB-long ORB[340.65,344.00] vol=1.7x ATR=1.06 |
| Stop hit — per-position SL triggered | 2025-06-24 10:35:00 | 343.79 | 342.52 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 10:00:00 | 345.50 | 343.28 | 0.00 | ORB-long ORB[339.50,343.40] vol=1.7x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-25 10:10:00 | 346.84 | 344.88 | 0.00 | T1 1.5R @ 346.84 |
| Stop hit — per-position SL triggered | 2025-06-25 12:10:00 | 345.50 | 345.95 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-04 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 10:20:00 | 361.60 | 359.01 | 0.00 | ORB-long ORB[358.15,360.30] vol=1.5x ATR=1.07 |
| Stop hit — per-position SL triggered | 2025-07-04 10:35:00 | 360.53 | 359.23 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-07 10:55:00 | 355.30 | 357.62 | 0.00 | ORB-short ORB[357.20,360.60] vol=2.2x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 11:30:00 | 353.92 | 357.15 | 0.00 | T1 1.5R @ 353.92 |
| Stop hit — per-position SL triggered | 2025-07-07 13:00:00 | 355.30 | 356.18 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 09:30:00 | 368.95 | 365.98 | 0.00 | ORB-long ORB[361.65,367.00] vol=3.4x ATR=1.38 |
| Stop hit — per-position SL triggered | 2025-07-11 09:40:00 | 367.57 | 368.36 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-14 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 09:50:00 | 364.40 | 363.08 | 0.00 | ORB-long ORB[360.20,363.55] vol=3.5x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:55:00 | 366.36 | 363.46 | 0.00 | T1 1.5R @ 366.36 |
| Stop hit — per-position SL triggered | 2025-07-14 11:00:00 | 364.40 | 364.22 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 09:30:00 | 367.35 | 366.42 | 0.00 | ORB-long ORB[362.60,365.00] vol=18.8x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-15 09:40:00 | 369.19 | 366.53 | 0.00 | T1 1.5R @ 369.19 |
| Stop hit — per-position SL triggered | 2025-07-15 09:55:00 | 367.35 | 366.66 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-07-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 10:55:00 | 381.05 | 377.85 | 0.00 | ORB-long ORB[374.00,379.45] vol=2.2x ATR=1.89 |
| Stop hit — per-position SL triggered | 2025-07-17 12:00:00 | 379.16 | 378.99 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-07-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-22 09:35:00 | 404.90 | 402.53 | 0.00 | ORB-long ORB[399.00,404.45] vol=3.3x ATR=1.79 |
| Stop hit — per-position SL triggered | 2025-07-22 09:40:00 | 403.11 | 402.97 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-07-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 09:30:00 | 413.80 | 410.99 | 0.00 | ORB-long ORB[408.65,412.00] vol=1.5x ATR=1.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 09:45:00 | 416.00 | 413.09 | 0.00 | T1 1.5R @ 416.00 |
| Target hit | 2025-07-24 10:40:00 | 415.00 | 415.06 | 0.00 | Trail-exit close<VWAP |

### Cycle 23 — BUY (started 2025-07-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-25 09:55:00 | 416.80 | 413.56 | 0.00 | ORB-long ORB[408.80,414.00] vol=1.7x ATR=1.65 |
| Stop hit — per-position SL triggered | 2025-07-25 10:10:00 | 415.15 | 414.20 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-08-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-06 09:30:00 | 440.45 | 437.86 | 0.00 | ORB-long ORB[435.00,439.20] vol=3.2x ATR=1.54 |
| Stop hit — per-position SL triggered | 2025-08-06 09:35:00 | 438.91 | 438.09 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-08-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 11:05:00 | 446.75 | 444.31 | 0.00 | ORB-long ORB[440.55,445.60] vol=2.0x ATR=1.42 |
| Stop hit — per-position SL triggered | 2025-08-11 11:10:00 | 445.33 | 444.35 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-08-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 10:00:00 | 452.70 | 455.49 | 0.00 | ORB-short ORB[455.50,459.90] vol=1.6x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-14 10:10:00 | 450.44 | 454.94 | 0.00 | T1 1.5R @ 450.44 |
| Stop hit — per-position SL triggered | 2025-08-14 10:15:00 | 452.70 | 454.57 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-08-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 09:35:00 | 457.55 | 454.91 | 0.00 | ORB-long ORB[450.95,455.80] vol=2.7x ATR=1.31 |
| Stop hit — per-position SL triggered | 2025-08-19 09:45:00 | 456.24 | 455.29 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-08-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 09:40:00 | 463.25 | 460.34 | 0.00 | ORB-long ORB[456.00,460.55] vol=1.8x ATR=2.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 09:50:00 | 466.52 | 462.80 | 0.00 | T1 1.5R @ 466.52 |
| Target hit | 2025-08-20 12:25:00 | 463.50 | 463.65 | 0.00 | Trail-exit close<VWAP |

### Cycle 29 — BUY (started 2025-08-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:50:00 | 467.90 | 464.49 | 0.00 | ORB-long ORB[459.10,464.85] vol=2.7x ATR=1.60 |
| Stop hit — per-position SL triggered | 2025-08-21 10:00:00 | 466.30 | 464.90 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-08-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 09:35:00 | 464.00 | 459.42 | 0.00 | ORB-long ORB[455.80,462.35] vol=1.7x ATR=2.23 |
| Stop hit — per-position SL triggered | 2025-08-25 10:15:00 | 461.77 | 460.92 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-09-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 11:05:00 | 464.70 | 463.04 | 0.00 | ORB-long ORB[458.80,464.20] vol=2.0x ATR=0.94 |
| Stop hit — per-position SL triggered | 2025-09-02 11:10:00 | 463.76 | 463.06 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-09-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 10:00:00 | 466.70 | 463.44 | 0.00 | ORB-long ORB[458.25,465.20] vol=1.7x ATR=1.78 |
| Stop hit — per-position SL triggered | 2025-09-08 10:20:00 | 464.92 | 463.67 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-09-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-11 11:10:00 | 435.60 | 439.26 | 0.00 | ORB-short ORB[438.40,443.70] vol=1.6x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-11 11:55:00 | 434.05 | 438.52 | 0.00 | T1 1.5R @ 434.05 |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 435.60 | 438.23 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 11:05:00 | 434.05 | 431.24 | 0.00 | ORB-long ORB[426.60,431.35] vol=2.4x ATR=1.61 |
| Stop hit — per-position SL triggered | 2025-09-12 11:20:00 | 432.44 | 431.34 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-09-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 11:10:00 | 441.65 | 437.94 | 0.00 | ORB-long ORB[434.00,440.00] vol=2.7x ATR=1.57 |
| Stop hit — per-position SL triggered | 2025-09-16 11:15:00 | 440.08 | 438.01 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 09:40:00 | 446.85 | 445.60 | 0.00 | ORB-long ORB[442.40,445.70] vol=1.9x ATR=1.32 |
| Stop hit — per-position SL triggered | 2025-09-18 10:20:00 | 445.53 | 446.44 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-09-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 09:40:00 | 441.00 | 443.52 | 0.00 | ORB-short ORB[441.55,447.95] vol=1.6x ATR=1.67 |
| Stop hit — per-position SL triggered | 2025-09-24 10:05:00 | 442.67 | 442.78 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-10-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 11:05:00 | 412.70 | 415.13 | 0.00 | ORB-short ORB[413.40,419.30] vol=3.8x ATR=1.66 |
| Stop hit — per-position SL triggered | 2025-10-01 11:25:00 | 414.36 | 415.04 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-10-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-09 09:50:00 | 427.60 | 429.99 | 0.00 | ORB-short ORB[429.15,433.90] vol=2.2x ATR=1.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 10:10:00 | 425.47 | 428.57 | 0.00 | T1 1.5R @ 425.47 |
| Stop hit — per-position SL triggered | 2025-10-09 10:15:00 | 427.60 | 428.48 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-10-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 10:10:00 | 439.00 | 435.97 | 0.00 | ORB-long ORB[430.00,434.35] vol=3.4x ATR=1.72 |
| Stop hit — per-position SL triggered | 2025-10-10 10:40:00 | 437.28 | 436.41 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-10-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 09:35:00 | 433.80 | 431.74 | 0.00 | ORB-long ORB[428.00,432.75] vol=2.5x ATR=1.51 |
| Stop hit — per-position SL triggered | 2025-10-13 09:40:00 | 432.29 | 431.78 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-10-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 09:40:00 | 423.15 | 426.53 | 0.00 | ORB-short ORB[426.95,430.00] vol=2.4x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 10:40:00 | 421.34 | 424.64 | 0.00 | T1 1.5R @ 421.34 |
| Stop hit — per-position SL triggered | 2025-10-14 12:25:00 | 423.15 | 423.14 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-10-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 09:50:00 | 428.45 | 425.08 | 0.00 | ORB-long ORB[420.40,426.15] vol=3.6x ATR=1.95 |
| Stop hit — per-position SL triggered | 2025-10-28 09:55:00 | 426.50 | 425.03 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-10-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-29 10:00:00 | 418.50 | 420.40 | 0.00 | ORB-short ORB[420.00,423.05] vol=1.8x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 10:40:00 | 416.34 | 419.53 | 0.00 | T1 1.5R @ 416.34 |
| Stop hit — per-position SL triggered | 2025-10-29 11:05:00 | 418.50 | 418.98 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-10-30 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-30 09:55:00 | 422.55 | 421.06 | 0.00 | ORB-long ORB[417.40,422.05] vol=2.5x ATR=1.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 10:35:00 | 425.26 | 422.22 | 0.00 | T1 1.5R @ 425.26 |
| Target hit | 2025-10-30 13:30:00 | 422.75 | 422.85 | 0.00 | Trail-exit close<VWAP |

### Cycle 46 — SELL (started 2025-10-31 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 10:30:00 | 416.25 | 417.87 | 0.00 | ORB-short ORB[418.05,424.25] vol=6.0x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 11:10:00 | 414.35 | 417.39 | 0.00 | T1 1.5R @ 414.35 |
| Stop hit — per-position SL triggered | 2025-10-31 14:55:00 | 416.25 | 416.65 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-11-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 10:55:00 | 406.10 | 407.77 | 0.00 | ORB-short ORB[406.75,411.50] vol=1.8x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 11:35:00 | 404.33 | 407.41 | 0.00 | T1 1.5R @ 404.33 |
| Target hit | 2025-11-04 15:20:00 | 397.75 | 402.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — SELL (started 2025-11-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 09:35:00 | 394.35 | 395.45 | 0.00 | ORB-short ORB[395.00,399.65] vol=2.0x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 09:50:00 | 392.06 | 395.00 | 0.00 | T1 1.5R @ 392.06 |
| Target hit | 2025-11-06 15:20:00 | 382.45 | 384.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2025-11-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 09:30:00 | 377.10 | 379.23 | 0.00 | ORB-short ORB[378.05,382.00] vol=1.8x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:40:00 | 375.12 | 377.37 | 0.00 | T1 1.5R @ 375.12 |
| Stop hit — per-position SL triggered | 2025-11-07 09:55:00 | 377.10 | 377.17 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-11-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 10:00:00 | 375.30 | 376.14 | 0.00 | ORB-short ORB[376.10,380.30] vol=1.8x ATR=1.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:15:00 | 373.44 | 375.70 | 0.00 | T1 1.5R @ 373.44 |
| Stop hit — per-position SL triggered | 2025-11-11 10:25:00 | 375.30 | 375.65 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-11-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-17 09:35:00 | 368.20 | 370.69 | 0.00 | ORB-short ORB[369.05,373.85] vol=2.7x ATR=1.44 |
| Stop hit — per-position SL triggered | 2025-11-17 09:40:00 | 369.64 | 370.55 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 11:15:00 | 356.85 | 358.16 | 0.00 | ORB-short ORB[358.60,363.35] vol=4.9x ATR=0.94 |
| Stop hit — per-position SL triggered | 2025-11-21 11:30:00 | 357.79 | 358.10 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-11-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 11:10:00 | 359.95 | 362.80 | 0.00 | ORB-short ORB[363.55,367.95] vol=2.2x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 12:25:00 | 358.49 | 360.94 | 0.00 | T1 1.5R @ 358.49 |
| Stop hit — per-position SL triggered | 2025-11-24 14:00:00 | 359.95 | 360.48 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-12-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 10:35:00 | 354.45 | 355.94 | 0.00 | ORB-short ORB[355.30,358.75] vol=2.1x ATR=1.41 |
| Stop hit — per-position SL triggered | 2025-12-01 11:00:00 | 355.86 | 356.10 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-12-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 09:45:00 | 358.50 | 357.79 | 0.00 | ORB-long ORB[354.80,358.00] vol=2.7x ATR=0.94 |
| Stop hit — per-position SL triggered | 2025-12-02 09:50:00 | 357.56 | 357.81 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-12-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 11:05:00 | 344.75 | 347.42 | 0.00 | ORB-short ORB[348.10,351.15] vol=4.8x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 11:10:00 | 343.49 | 346.40 | 0.00 | T1 1.5R @ 343.49 |
| Stop hit — per-position SL triggered | 2025-12-08 11:35:00 | 344.75 | 345.38 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-12-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 09:35:00 | 337.05 | 339.76 | 0.00 | ORB-short ORB[339.05,343.00] vol=2.3x ATR=1.36 |
| Stop hit — per-position SL triggered | 2025-12-09 09:40:00 | 338.41 | 339.59 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-12-10 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 10:50:00 | 337.00 | 340.27 | 0.00 | ORB-short ORB[340.25,344.85] vol=1.8x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 12:15:00 | 335.51 | 339.14 | 0.00 | T1 1.5R @ 335.51 |
| Stop hit — per-position SL triggered | 2025-12-10 13:25:00 | 337.00 | 337.82 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-12-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:30:00 | 342.25 | 340.13 | 0.00 | ORB-long ORB[335.70,340.45] vol=4.1x ATR=1.50 |
| Stop hit — per-position SL triggered | 2025-12-12 09:45:00 | 340.75 | 340.40 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-12-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-19 09:35:00 | 350.40 | 351.88 | 0.00 | ORB-short ORB[352.10,354.70] vol=2.1x ATR=0.85 |
| Stop hit — per-position SL triggered | 2025-12-19 09:50:00 | 351.25 | 351.45 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-12-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-22 10:50:00 | 357.90 | 358.80 | 0.00 | ORB-short ORB[358.00,361.00] vol=1.8x ATR=0.99 |
| Stop hit — per-position SL triggered | 2025-12-22 11:35:00 | 358.89 | 358.54 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2026-01-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-05 09:50:00 | 355.25 | 356.34 | 0.00 | ORB-short ORB[356.00,358.80] vol=1.6x ATR=1.13 |
| Stop hit — per-position SL triggered | 2026-01-05 10:05:00 | 356.38 | 356.20 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2026-01-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 09:40:00 | 352.90 | 353.90 | 0.00 | ORB-short ORB[353.05,358.00] vol=6.9x ATR=1.05 |
| Stop hit — per-position SL triggered | 2026-01-06 10:00:00 | 353.95 | 353.84 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-01-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-09 10:30:00 | 350.95 | 348.37 | 0.00 | ORB-long ORB[345.30,349.80] vol=1.6x ATR=1.30 |
| Stop hit — per-position SL triggered | 2026-01-09 11:05:00 | 349.65 | 348.79 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-01-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-13 10:00:00 | 352.50 | 350.12 | 0.00 | ORB-long ORB[346.85,351.80] vol=1.7x ATR=1.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-13 10:10:00 | 354.36 | 351.49 | 0.00 | T1 1.5R @ 354.36 |
| Stop hit — per-position SL triggered | 2026-01-13 13:00:00 | 352.50 | 352.58 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-01-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 10:40:00 | 355.00 | 353.57 | 0.00 | ORB-long ORB[349.85,354.75] vol=3.8x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 11:05:00 | 357.31 | 354.29 | 0.00 | T1 1.5R @ 357.31 |
| Stop hit — per-position SL triggered | 2026-01-14 12:50:00 | 355.00 | 354.70 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-01-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 10:40:00 | 348.40 | 349.77 | 0.00 | ORB-short ORB[350.00,352.80] vol=2.4x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 11:00:00 | 347.00 | 349.59 | 0.00 | T1 1.5R @ 347.00 |
| Target hit | 2026-01-23 14:05:00 | 347.55 | 347.42 | 0.00 | Trail-exit close>VWAP |

### Cycle 68 — BUY (started 2026-02-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-02 09:50:00 | 335.85 | 331.86 | 0.00 | ORB-long ORB[326.50,331.45] vol=2.0x ATR=2.57 |
| Stop hit — per-position SL triggered | 2026-02-02 11:10:00 | 333.28 | 333.40 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2026-02-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 11:00:00 | 343.95 | 345.86 | 0.00 | ORB-short ORB[345.50,348.70] vol=1.8x ATR=0.94 |
| Stop hit — per-position SL triggered | 2026-02-05 11:20:00 | 344.89 | 345.60 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 11:15:00 | 343.20 | 343.65 | 0.00 | ORB-short ORB[344.05,349.00] vol=1.9x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-06 12:25:00 | 342.07 | 343.47 | 0.00 | T1 1.5R @ 342.07 |
| Stop hit — per-position SL triggered | 2026-02-06 13:25:00 | 343.20 | 343.41 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-02-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:55:00 | 345.00 | 345.65 | 0.00 | ORB-short ORB[345.10,350.00] vol=3.8x ATR=1.22 |
| Stop hit — per-position SL triggered | 2026-02-11 11:00:00 | 346.22 | 345.78 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2026-02-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:35:00 | 330.70 | 331.17 | 0.00 | ORB-short ORB[331.00,333.80] vol=5.6x ATR=1.02 |
| Stop hit — per-position SL triggered | 2026-02-24 09:45:00 | 331.72 | 331.15 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-02-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:35:00 | 334.35 | 333.07 | 0.00 | ORB-long ORB[331.10,333.20] vol=4.9x ATR=1.06 |
| Stop hit — per-position SL triggered | 2026-02-26 11:30:00 | 333.29 | 333.60 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-03-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 09:30:00 | 303.15 | 300.64 | 0.00 | ORB-long ORB[298.15,302.35] vol=1.7x ATR=1.78 |
| Stop hit — per-position SL triggered | 2026-03-27 09:35:00 | 301.37 | 300.69 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-04-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 09:30:00 | 305.35 | 309.58 | 0.00 | ORB-short ORB[309.20,313.75] vol=4.0x ATR=1.35 |
| Stop hit — per-position SL triggered | 2026-04-17 09:40:00 | 306.70 | 308.23 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-04-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:30:00 | 295.75 | 294.63 | 0.00 | ORB-long ORB[291.90,295.70] vol=2.7x ATR=0.83 |
| Stop hit — per-position SL triggered | 2026-04-27 09:35:00 | 294.92 | 294.67 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-04-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 10:05:00 | 289.20 | 286.13 | 0.00 | ORB-long ORB[283.40,287.60] vol=2.8x ATR=1.21 |
| Stop hit — per-position SL triggered | 2026-04-30 10:10:00 | 287.99 | 286.19 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2026-05-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:50:00 | 293.15 | 291.74 | 0.00 | ORB-long ORB[288.45,291.95] vol=2.7x ATR=1.32 |
| Stop hit — per-position SL triggered | 2026-05-05 10:10:00 | 291.83 | 291.85 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2026-05-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:00:00 | 306.85 | 301.51 | 0.00 | ORB-long ORB[297.15,301.00] vol=6.5x ATR=1.66 |
| Stop hit — per-position SL triggered | 2026-05-06 10:05:00 | 305.19 | 302.69 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 10:50:00 | 352.90 | 2025-05-14 11:00:00 | 351.35 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-05-19 10:45:00 | 357.50 | 2025-05-19 11:30:00 | 356.68 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-05-20 09:35:00 | 359.45 | 2025-05-20 09:40:00 | 358.39 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-05-23 09:35:00 | 359.70 | 2025-05-23 10:35:00 | 361.60 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-05-23 09:35:00 | 359.70 | 2025-05-23 12:15:00 | 360.45 | TARGET_HIT | 0.50 | 0.21% |
| SELL | retest1 | 2025-05-26 11:10:00 | 357.05 | 2025-05-26 11:15:00 | 357.89 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-05-28 10:25:00 | 354.00 | 2025-05-28 10:40:00 | 352.89 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-05-28 10:25:00 | 354.00 | 2025-05-28 11:00:00 | 354.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-29 09:35:00 | 351.25 | 2025-05-29 10:50:00 | 349.59 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-05-29 09:35:00 | 351.25 | 2025-05-29 12:55:00 | 351.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-04 09:35:00 | 353.45 | 2025-06-04 09:45:00 | 351.62 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-06-04 09:35:00 | 353.45 | 2025-06-04 09:55:00 | 353.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-10 09:40:00 | 360.00 | 2025-06-10 09:55:00 | 358.75 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-06-12 09:40:00 | 355.45 | 2025-06-12 09:55:00 | 356.45 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-06-16 09:30:00 | 346.50 | 2025-06-16 09:40:00 | 345.07 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-06-16 09:30:00 | 346.50 | 2025-06-16 10:05:00 | 346.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-20 11:15:00 | 344.25 | 2025-06-20 11:20:00 | 343.05 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-06-24 10:30:00 | 344.85 | 2025-06-24 10:35:00 | 343.79 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-06-25 10:00:00 | 345.50 | 2025-06-25 10:10:00 | 346.84 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-06-25 10:00:00 | 345.50 | 2025-06-25 12:10:00 | 345.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-04 10:20:00 | 361.60 | 2025-07-04 10:35:00 | 360.53 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-07-07 10:55:00 | 355.30 | 2025-07-07 11:30:00 | 353.92 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-07-07 10:55:00 | 355.30 | 2025-07-07 13:00:00 | 355.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-11 09:30:00 | 368.95 | 2025-07-11 09:40:00 | 367.57 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-07-14 09:50:00 | 364.40 | 2025-07-14 09:55:00 | 366.36 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-07-14 09:50:00 | 364.40 | 2025-07-14 11:00:00 | 364.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-15 09:30:00 | 367.35 | 2025-07-15 09:40:00 | 369.19 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-07-15 09:30:00 | 367.35 | 2025-07-15 09:55:00 | 367.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-17 10:55:00 | 381.05 | 2025-07-17 12:00:00 | 379.16 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-07-22 09:35:00 | 404.90 | 2025-07-22 09:40:00 | 403.11 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-07-24 09:30:00 | 413.80 | 2025-07-24 09:45:00 | 416.00 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-07-24 09:30:00 | 413.80 | 2025-07-24 10:40:00 | 415.00 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2025-07-25 09:55:00 | 416.80 | 2025-07-25 10:10:00 | 415.15 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-08-06 09:30:00 | 440.45 | 2025-08-06 09:35:00 | 438.91 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-08-11 11:05:00 | 446.75 | 2025-08-11 11:10:00 | 445.33 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-08-14 10:00:00 | 452.70 | 2025-08-14 10:10:00 | 450.44 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-08-14 10:00:00 | 452.70 | 2025-08-14 10:15:00 | 452.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-19 09:35:00 | 457.55 | 2025-08-19 09:45:00 | 456.24 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-08-20 09:40:00 | 463.25 | 2025-08-20 09:50:00 | 466.52 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2025-08-20 09:40:00 | 463.25 | 2025-08-20 12:25:00 | 463.50 | TARGET_HIT | 0.50 | 0.05% |
| BUY | retest1 | 2025-08-21 09:50:00 | 467.90 | 2025-08-21 10:00:00 | 466.30 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-08-25 09:35:00 | 464.00 | 2025-08-25 10:15:00 | 461.77 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-09-02 11:05:00 | 464.70 | 2025-09-02 11:10:00 | 463.76 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-09-08 10:00:00 | 466.70 | 2025-09-08 10:20:00 | 464.92 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-09-11 11:10:00 | 435.60 | 2025-09-11 11:55:00 | 434.05 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-09-11 11:10:00 | 435.60 | 2025-09-11 12:15:00 | 435.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-12 11:05:00 | 434.05 | 2025-09-12 11:20:00 | 432.44 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-09-16 11:10:00 | 441.65 | 2025-09-16 11:15:00 | 440.08 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-09-18 09:40:00 | 446.85 | 2025-09-18 10:20:00 | 445.53 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-09-24 09:40:00 | 441.00 | 2025-09-24 10:05:00 | 442.67 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-10-01 11:05:00 | 412.70 | 2025-10-01 11:25:00 | 414.36 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-10-09 09:50:00 | 427.60 | 2025-10-09 10:10:00 | 425.47 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-10-09 09:50:00 | 427.60 | 2025-10-09 10:15:00 | 427.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-10 10:10:00 | 439.00 | 2025-10-10 10:40:00 | 437.28 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-10-13 09:35:00 | 433.80 | 2025-10-13 09:40:00 | 432.29 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-10-14 09:40:00 | 423.15 | 2025-10-14 10:40:00 | 421.34 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-10-14 09:40:00 | 423.15 | 2025-10-14 12:25:00 | 423.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-28 09:50:00 | 428.45 | 2025-10-28 09:55:00 | 426.50 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-10-29 10:00:00 | 418.50 | 2025-10-29 10:40:00 | 416.34 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-10-29 10:00:00 | 418.50 | 2025-10-29 11:05:00 | 418.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-30 09:55:00 | 422.55 | 2025-10-30 10:35:00 | 425.26 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-10-30 09:55:00 | 422.55 | 2025-10-30 13:30:00 | 422.75 | TARGET_HIT | 0.50 | 0.05% |
| SELL | retest1 | 2025-10-31 10:30:00 | 416.25 | 2025-10-31 11:10:00 | 414.35 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-10-31 10:30:00 | 416.25 | 2025-10-31 14:55:00 | 416.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-04 10:55:00 | 406.10 | 2025-11-04 11:35:00 | 404.33 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-11-04 10:55:00 | 406.10 | 2025-11-04 15:20:00 | 397.75 | TARGET_HIT | 0.50 | 2.06% |
| SELL | retest1 | 2025-11-06 09:35:00 | 394.35 | 2025-11-06 09:50:00 | 392.06 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-11-06 09:35:00 | 394.35 | 2025-11-06 15:20:00 | 382.45 | TARGET_HIT | 0.50 | 3.02% |
| SELL | retest1 | 2025-11-07 09:30:00 | 377.10 | 2025-11-07 09:40:00 | 375.12 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-11-07 09:30:00 | 377.10 | 2025-11-07 09:55:00 | 377.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-11 10:00:00 | 375.30 | 2025-11-11 10:15:00 | 373.44 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-11-11 10:00:00 | 375.30 | 2025-11-11 10:25:00 | 375.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-17 09:35:00 | 368.20 | 2025-11-17 09:40:00 | 369.64 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-11-21 11:15:00 | 356.85 | 2025-11-21 11:30:00 | 357.79 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-11-24 11:10:00 | 359.95 | 2025-11-24 12:25:00 | 358.49 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-11-24 11:10:00 | 359.95 | 2025-11-24 14:00:00 | 359.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-01 10:35:00 | 354.45 | 2025-12-01 11:00:00 | 355.86 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-12-02 09:45:00 | 358.50 | 2025-12-02 09:50:00 | 357.56 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-12-08 11:05:00 | 344.75 | 2025-12-08 11:10:00 | 343.49 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-12-08 11:05:00 | 344.75 | 2025-12-08 11:35:00 | 344.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-09 09:35:00 | 337.05 | 2025-12-09 09:40:00 | 338.41 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-12-10 10:50:00 | 337.00 | 2025-12-10 12:15:00 | 335.51 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-12-10 10:50:00 | 337.00 | 2025-12-10 13:25:00 | 337.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-12 09:30:00 | 342.25 | 2025-12-12 09:45:00 | 340.75 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-12-19 09:35:00 | 350.40 | 2025-12-19 09:50:00 | 351.25 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-12-22 10:50:00 | 357.90 | 2025-12-22 11:35:00 | 358.89 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-01-05 09:50:00 | 355.25 | 2026-01-05 10:05:00 | 356.38 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-01-06 09:40:00 | 352.90 | 2026-01-06 10:00:00 | 353.95 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-01-09 10:30:00 | 350.95 | 2026-01-09 11:05:00 | 349.65 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-01-13 10:00:00 | 352.50 | 2026-01-13 10:10:00 | 354.36 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-01-13 10:00:00 | 352.50 | 2026-01-13 13:00:00 | 352.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-14 10:40:00 | 355.00 | 2026-01-14 11:05:00 | 357.31 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-01-14 10:40:00 | 355.00 | 2026-01-14 12:50:00 | 355.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-23 10:40:00 | 348.40 | 2026-01-23 11:00:00 | 347.00 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-01-23 10:40:00 | 348.40 | 2026-01-23 14:05:00 | 347.55 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2026-02-02 09:50:00 | 335.85 | 2026-02-02 11:10:00 | 333.28 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest1 | 2026-02-05 11:00:00 | 343.95 | 2026-02-05 11:20:00 | 344.89 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-02-06 11:15:00 | 343.20 | 2026-02-06 12:25:00 | 342.07 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-02-06 11:15:00 | 343.20 | 2026-02-06 13:25:00 | 343.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-11 10:55:00 | 345.00 | 2026-02-11 11:00:00 | 346.22 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-24 09:35:00 | 330.70 | 2026-02-24 09:45:00 | 331.72 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-26 09:35:00 | 334.35 | 2026-02-26 11:30:00 | 333.29 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-03-27 09:30:00 | 303.15 | 2026-03-27 09:35:00 | 301.37 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest1 | 2026-04-17 09:30:00 | 305.35 | 2026-04-17 09:40:00 | 306.70 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-04-27 09:30:00 | 295.75 | 2026-04-27 09:35:00 | 294.92 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-30 10:05:00 | 289.20 | 2026-04-30 10:10:00 | 287.99 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-05-05 09:50:00 | 293.15 | 2026-05-05 10:10:00 | 291.83 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-05-06 10:00:00 | 306.85 | 2026-05-06 10:05:00 | 305.19 | STOP_HIT | 1.00 | -0.54% |
