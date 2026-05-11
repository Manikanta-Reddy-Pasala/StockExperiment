# Sonata Software Ltd. (SONATSOFTW)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 296.65
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
| ENTRY1 | 47 |
| ENTRY2 | 0 |
| PARTIAL | 14 |
| TARGET_HIT | 7 |
| STOP_HIT | 40 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 61 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 40
- **Target hits / Stop hits / Partials:** 7 / 40 / 14
- **Avg / median % per leg:** 0.01% / -0.26%
- **Sum % (uncompounded):** 0.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 8 | 23.5% | 2 | 26 | 6 | -0.10% | -3.4% |
| BUY @ 2nd Alert (retest1) | 34 | 8 | 23.5% | 2 | 26 | 6 | -0.10% | -3.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 27 | 13 | 48.1% | 5 | 14 | 8 | 0.15% | 4.0% |
| SELL @ 2nd Alert (retest1) | 27 | 13 | 48.1% | 5 | 14 | 8 | 0.15% | 4.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 61 | 21 | 34.4% | 7 | 40 | 14 | 0.01% | 0.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 09:30:00 | 402.15 | 398.12 | 0.00 | ORB-long ORB[393.65,398.85] vol=3.2x ATR=2.08 |
| Stop hit — per-position SL triggered | 2025-05-15 09:40:00 | 400.07 | 399.22 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 10:00:00 | 399.15 | 396.55 | 0.00 | ORB-long ORB[393.55,398.35] vol=2.1x ATR=1.35 |
| Stop hit — per-position SL triggered | 2025-05-26 10:05:00 | 397.80 | 397.90 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-30 09:35:00 | 408.45 | 405.60 | 0.00 | ORB-long ORB[402.70,408.00] vol=2.3x ATR=1.63 |
| Stop hit — per-position SL triggered | 2025-05-30 10:15:00 | 406.82 | 407.31 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-06-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 10:10:00 | 417.60 | 415.42 | 0.00 | ORB-long ORB[412.15,416.30] vol=1.6x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 10:20:00 | 419.65 | 416.02 | 0.00 | T1 1.5R @ 419.65 |
| Stop hit — per-position SL triggered | 2025-06-04 10:35:00 | 417.60 | 416.50 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 09:35:00 | 421.95 | 417.61 | 0.00 | ORB-long ORB[411.00,416.90] vol=6.4x ATR=1.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 09:40:00 | 424.57 | 420.70 | 0.00 | T1 1.5R @ 424.57 |
| Stop hit — per-position SL triggered | 2025-06-27 09:50:00 | 421.95 | 421.82 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-30 10:45:00 | 411.65 | 413.68 | 0.00 | ORB-short ORB[412.90,416.90] vol=2.4x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-30 13:00:00 | 409.73 | 412.76 | 0.00 | T1 1.5R @ 409.73 |
| Stop hit — per-position SL triggered | 2025-06-30 13:55:00 | 411.65 | 412.55 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-07-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 10:40:00 | 405.45 | 409.17 | 0.00 | ORB-short ORB[410.00,413.20] vol=1.7x ATR=1.12 |
| Stop hit — per-position SL triggered | 2025-07-02 15:05:00 | 406.57 | 407.22 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-07-04 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 10:20:00 | 408.25 | 406.82 | 0.00 | ORB-long ORB[404.80,407.90] vol=1.8x ATR=1.17 |
| Stop hit — per-position SL triggered | 2025-07-04 10:30:00 | 407.08 | 406.89 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-07-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 09:55:00 | 413.60 | 411.17 | 0.00 | ORB-long ORB[407.00,412.70] vol=2.7x ATR=1.26 |
| Stop hit — per-position SL triggered | 2025-07-07 10:05:00 | 412.34 | 411.47 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-07-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 10:50:00 | 432.80 | 428.58 | 0.00 | ORB-long ORB[425.30,430.55] vol=3.5x ATR=1.56 |
| Stop hit — per-position SL triggered | 2025-07-14 10:55:00 | 431.24 | 428.70 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-15 11:15:00 | 432.60 | 435.66 | 0.00 | ORB-short ORB[432.90,438.30] vol=1.9x ATR=1.38 |
| Stop hit — per-position SL triggered | 2025-07-15 11:30:00 | 433.98 | 435.49 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-07-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 11:05:00 | 432.90 | 435.90 | 0.00 | ORB-short ORB[435.60,440.00] vol=2.4x ATR=1.13 |
| Stop hit — per-position SL triggered | 2025-07-17 11:10:00 | 434.03 | 435.84 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:05:00 | 434.95 | 437.24 | 0.00 | ORB-short ORB[436.85,439.95] vol=2.0x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:15:00 | 433.29 | 436.31 | 0.00 | T1 1.5R @ 433.29 |
| Stop hit — per-position SL triggered | 2025-07-18 13:05:00 | 434.95 | 435.25 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 10:35:00 | 433.40 | 437.75 | 0.00 | ORB-short ORB[438.10,442.80] vol=1.8x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 11:30:00 | 431.60 | 437.01 | 0.00 | T1 1.5R @ 431.60 |
| Target hit | 2025-07-22 15:20:00 | 430.05 | 433.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — BUY (started 2025-07-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 11:05:00 | 413.55 | 411.42 | 0.00 | ORB-long ORB[407.05,412.80] vol=2.3x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-30 14:20:00 | 415.28 | 412.51 | 0.00 | T1 1.5R @ 415.28 |
| Stop hit — per-position SL triggered | 2025-07-30 15:05:00 | 413.55 | 413.32 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-08-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 09:45:00 | 361.90 | 359.73 | 0.00 | ORB-long ORB[356.95,361.00] vol=1.9x ATR=1.16 |
| Stop hit — per-position SL triggered | 2025-08-20 09:50:00 | 360.74 | 359.91 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-08-21 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 10:25:00 | 379.50 | 375.78 | 0.00 | ORB-long ORB[371.90,377.20] vol=2.0x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 10:30:00 | 381.85 | 376.49 | 0.00 | T1 1.5R @ 381.85 |
| Stop hit — per-position SL triggered | 2025-08-21 10:35:00 | 379.50 | 376.86 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-08-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-29 10:35:00 | 362.40 | 359.23 | 0.00 | ORB-long ORB[358.25,361.90] vol=1.7x ATR=1.27 |
| Stop hit — per-position SL triggered | 2025-08-29 10:40:00 | 361.13 | 359.34 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-09-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 09:45:00 | 386.10 | 382.56 | 0.00 | ORB-long ORB[379.25,384.95] vol=1.8x ATR=1.70 |
| Stop hit — per-position SL triggered | 2025-09-12 10:55:00 | 384.40 | 383.55 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 10:15:00 | 354.50 | 351.49 | 0.00 | ORB-long ORB[348.90,352.00] vol=2.3x ATR=1.40 |
| Stop hit — per-position SL triggered | 2025-10-06 10:40:00 | 353.10 | 352.35 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-10-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 09:30:00 | 358.15 | 355.84 | 0.00 | ORB-long ORB[351.35,356.50] vol=2.7x ATR=1.45 |
| Stop hit — per-position SL triggered | 2025-10-08 09:50:00 | 356.70 | 358.35 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-10-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 09:40:00 | 363.05 | 364.79 | 0.00 | ORB-short ORB[363.25,368.00] vol=2.1x ATR=1.30 |
| Stop hit — per-position SL triggered | 2025-10-13 09:45:00 | 364.35 | 364.75 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-10-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-14 09:30:00 | 368.30 | 366.38 | 0.00 | ORB-long ORB[363.60,366.70] vol=1.9x ATR=1.14 |
| Stop hit — per-position SL triggered | 2025-10-14 09:35:00 | 367.16 | 366.43 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-10-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 11:00:00 | 366.00 | 366.53 | 0.00 | ORB-short ORB[366.30,370.00] vol=1.5x ATR=0.83 |
| Stop hit — per-position SL triggered | 2025-10-17 11:05:00 | 366.83 | 366.53 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-10-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 10:00:00 | 361.65 | 363.61 | 0.00 | ORB-short ORB[363.40,368.70] vol=2.1x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 13:35:00 | 359.86 | 361.80 | 0.00 | T1 1.5R @ 359.86 |
| Target hit | 2025-10-20 15:20:00 | 359.75 | 360.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — BUY (started 2025-10-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:30:00 | 371.95 | 369.80 | 0.00 | ORB-long ORB[368.90,371.00] vol=2.0x ATR=0.78 |
| Stop hit — per-position SL triggered | 2025-10-29 10:35:00 | 371.17 | 369.96 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-10-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 09:30:00 | 373.45 | 374.68 | 0.00 | ORB-short ORB[374.00,376.35] vol=1.7x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 09:40:00 | 372.06 | 374.30 | 0.00 | T1 1.5R @ 372.06 |
| Target hit | 2025-10-30 14:15:00 | 372.50 | 372.46 | 0.00 | Trail-exit close>VWAP |

### Cycle 28 — BUY (started 2025-10-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:45:00 | 374.00 | 372.80 | 0.00 | ORB-long ORB[370.65,373.85] vol=2.1x ATR=0.98 |
| Stop hit — per-position SL triggered | 2025-10-31 09:50:00 | 373.02 | 372.90 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-11-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 09:35:00 | 373.00 | 371.83 | 0.00 | ORB-long ORB[369.05,372.60] vol=1.8x ATR=1.16 |
| Stop hit — per-position SL triggered | 2025-11-03 10:25:00 | 371.84 | 372.65 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-11-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 10:20:00 | 370.00 | 366.55 | 0.00 | ORB-long ORB[365.50,367.75] vol=2.4x ATR=1.14 |
| Stop hit — per-position SL triggered | 2025-11-10 10:25:00 | 368.86 | 366.87 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-11-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-19 10:35:00 | 369.00 | 366.76 | 0.00 | ORB-long ORB[362.60,366.95] vol=2.9x ATR=1.06 |
| Stop hit — per-position SL triggered | 2025-11-19 10:40:00 | 367.94 | 367.12 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-11-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-24 09:30:00 | 362.65 | 360.98 | 0.00 | ORB-long ORB[358.20,361.90] vol=2.7x ATR=1.12 |
| Stop hit — per-position SL triggered | 2025-11-24 09:35:00 | 361.53 | 361.61 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-12-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 11:10:00 | 357.00 | 358.33 | 0.00 | ORB-short ORB[358.00,360.30] vol=3.5x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 11:20:00 | 356.03 | 358.07 | 0.00 | T1 1.5R @ 356.03 |
| Target hit | 2025-12-02 15:20:00 | 350.00 | 352.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — SELL (started 2025-12-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 09:55:00 | 346.55 | 350.71 | 0.00 | ORB-short ORB[351.25,355.90] vol=1.6x ATR=1.49 |
| Stop hit — per-position SL triggered | 2025-12-09 10:05:00 | 348.04 | 350.42 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-12-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 09:35:00 | 355.15 | 351.80 | 0.00 | ORB-long ORB[348.70,352.00] vol=2.3x ATR=1.28 |
| Stop hit — per-position SL triggered | 2025-12-10 09:45:00 | 353.87 | 352.09 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-12-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-11 09:35:00 | 342.45 | 345.98 | 0.00 | ORB-short ORB[345.15,350.00] vol=2.7x ATR=1.46 |
| Stop hit — per-position SL triggered | 2025-12-11 10:15:00 | 343.91 | 344.53 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-12-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 11:05:00 | 357.65 | 359.57 | 0.00 | ORB-short ORB[358.80,362.60] vol=2.1x ATR=0.91 |
| Stop hit — per-position SL triggered | 2025-12-24 12:20:00 | 358.56 | 359.13 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2026-01-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 10:00:00 | 360.10 | 361.53 | 0.00 | ORB-short ORB[360.70,363.50] vol=1.8x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 10:20:00 | 358.70 | 361.19 | 0.00 | T1 1.5R @ 358.70 |
| Stop hit — per-position SL triggered | 2026-01-01 10:30:00 | 360.10 | 361.10 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2026-02-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 10:50:00 | 309.50 | 311.24 | 0.00 | ORB-short ORB[310.60,314.85] vol=1.8x ATR=0.98 |
| Stop hit — per-position SL triggered | 2026-02-05 15:10:00 | 310.48 | 310.13 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2026-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 09:30:00 | 315.45 | 318.29 | 0.00 | ORB-short ORB[317.00,321.65] vol=3.5x ATR=1.06 |
| Stop hit — per-position SL triggered | 2026-02-11 09:50:00 | 316.51 | 317.49 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2026-03-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 09:45:00 | 262.75 | 258.89 | 0.00 | ORB-long ORB[256.35,259.80] vol=1.6x ATR=1.54 |
| Stop hit — per-position SL triggered | 2026-03-04 09:50:00 | 261.21 | 259.13 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 246.05 | 249.07 | 0.00 | ORB-short ORB[249.20,252.50] vol=1.6x ATR=0.92 |
| Stop hit — per-position SL triggered | 2026-03-06 10:50:00 | 246.97 | 249.01 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2026-03-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 09:35:00 | 233.70 | 235.30 | 0.00 | ORB-short ORB[234.80,238.00] vol=2.6x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 14:10:00 | 231.79 | 234.08 | 0.00 | T1 1.5R @ 231.79 |
| Target hit | 2026-03-19 15:20:00 | 232.55 | 233.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — BUY (started 2026-04-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 11:00:00 | 273.75 | 269.69 | 0.00 | ORB-long ORB[267.19,270.45] vol=4.6x ATR=1.10 |
| Stop hit — per-position SL triggered | 2026-04-23 11:05:00 | 272.65 | 270.16 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2026-04-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:00:00 | 260.60 | 258.70 | 0.00 | ORB-long ORB[257.00,259.99] vol=1.9x ATR=1.04 |
| Stop hit — per-position SL triggered | 2026-04-28 10:40:00 | 259.56 | 259.25 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2026-04-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:45:00 | 262.70 | 260.19 | 0.00 | ORB-long ORB[257.15,260.54] vol=3.8x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 09:50:00 | 264.42 | 263.98 | 0.00 | T1 1.5R @ 264.42 |
| Target hit | 2026-04-29 10:00:00 | 264.64 | 264.84 | 0.00 | Trail-exit close<VWAP |

### Cycle 47 — BUY (started 2026-05-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:30:00 | 259.75 | 258.53 | 0.00 | ORB-long ORB[255.50,258.45] vol=2.0x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 10:35:00 | 261.57 | 258.72 | 0.00 | T1 1.5R @ 261.57 |
| Target hit | 2026-05-04 12:10:00 | 260.45 | 260.81 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-15 09:30:00 | 402.15 | 2025-05-15 09:40:00 | 400.07 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2025-05-26 10:00:00 | 399.15 | 2025-05-26 10:05:00 | 397.80 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-05-30 09:35:00 | 408.45 | 2025-05-30 10:15:00 | 406.82 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-06-04 10:10:00 | 417.60 | 2025-06-04 10:20:00 | 419.65 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-06-04 10:10:00 | 417.60 | 2025-06-04 10:35:00 | 417.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-27 09:35:00 | 421.95 | 2025-06-27 09:40:00 | 424.57 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-06-27 09:35:00 | 421.95 | 2025-06-27 09:50:00 | 421.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-30 10:45:00 | 411.65 | 2025-06-30 13:00:00 | 409.73 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-06-30 10:45:00 | 411.65 | 2025-06-30 13:55:00 | 411.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-02 10:40:00 | 405.45 | 2025-07-02 15:05:00 | 406.57 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-07-04 10:20:00 | 408.25 | 2025-07-04 10:30:00 | 407.08 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-07-07 09:55:00 | 413.60 | 2025-07-07 10:05:00 | 412.34 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-07-14 10:50:00 | 432.80 | 2025-07-14 10:55:00 | 431.24 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-07-15 11:15:00 | 432.60 | 2025-07-15 11:30:00 | 433.98 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-07-17 11:05:00 | 432.90 | 2025-07-17 11:10:00 | 434.03 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-07-18 10:05:00 | 434.95 | 2025-07-18 10:15:00 | 433.29 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-07-18 10:05:00 | 434.95 | 2025-07-18 13:05:00 | 434.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-22 10:35:00 | 433.40 | 2025-07-22 11:30:00 | 431.60 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-07-22 10:35:00 | 433.40 | 2025-07-22 15:20:00 | 430.05 | TARGET_HIT | 0.50 | 0.77% |
| BUY | retest1 | 2025-07-30 11:05:00 | 413.55 | 2025-07-30 14:20:00 | 415.28 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-07-30 11:05:00 | 413.55 | 2025-07-30 15:05:00 | 413.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-20 09:45:00 | 361.90 | 2025-08-20 09:50:00 | 360.74 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-08-21 10:25:00 | 379.50 | 2025-08-21 10:30:00 | 381.85 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-08-21 10:25:00 | 379.50 | 2025-08-21 10:35:00 | 379.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-29 10:35:00 | 362.40 | 2025-08-29 10:40:00 | 361.13 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-09-12 09:45:00 | 386.10 | 2025-09-12 10:55:00 | 384.40 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-10-06 10:15:00 | 354.50 | 2025-10-06 10:40:00 | 353.10 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-10-08 09:30:00 | 358.15 | 2025-10-08 09:50:00 | 356.70 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-10-13 09:40:00 | 363.05 | 2025-10-13 09:45:00 | 364.35 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-10-14 09:30:00 | 368.30 | 2025-10-14 09:35:00 | 367.16 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-10-17 11:00:00 | 366.00 | 2025-10-17 11:05:00 | 366.83 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-10-20 10:00:00 | 361.65 | 2025-10-20 13:35:00 | 359.86 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-10-20 10:00:00 | 361.65 | 2025-10-20 15:20:00 | 359.75 | TARGET_HIT | 0.50 | 0.53% |
| BUY | retest1 | 2025-10-29 10:30:00 | 371.95 | 2025-10-29 10:35:00 | 371.17 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-10-30 09:30:00 | 373.45 | 2025-10-30 09:40:00 | 372.06 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-10-30 09:30:00 | 373.45 | 2025-10-30 14:15:00 | 372.50 | TARGET_HIT | 0.50 | 0.25% |
| BUY | retest1 | 2025-10-31 09:45:00 | 374.00 | 2025-10-31 09:50:00 | 373.02 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-11-03 09:35:00 | 373.00 | 2025-11-03 10:25:00 | 371.84 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-11-10 10:20:00 | 370.00 | 2025-11-10 10:25:00 | 368.86 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-11-19 10:35:00 | 369.00 | 2025-11-19 10:40:00 | 367.94 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-11-24 09:30:00 | 362.65 | 2025-11-24 09:35:00 | 361.53 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-12-02 11:10:00 | 357.00 | 2025-12-02 11:20:00 | 356.03 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-12-02 11:10:00 | 357.00 | 2025-12-02 15:20:00 | 350.00 | TARGET_HIT | 0.50 | 1.96% |
| SELL | retest1 | 2025-12-09 09:55:00 | 346.55 | 2025-12-09 10:05:00 | 348.04 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-12-10 09:35:00 | 355.15 | 2025-12-10 09:45:00 | 353.87 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-12-11 09:35:00 | 342.45 | 2025-12-11 10:15:00 | 343.91 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-12-24 11:05:00 | 357.65 | 2025-12-24 12:20:00 | 358.56 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-01-01 10:00:00 | 360.10 | 2026-01-01 10:20:00 | 358.70 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-01-01 10:00:00 | 360.10 | 2026-01-01 10:30:00 | 360.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-05 10:50:00 | 309.50 | 2026-02-05 15:10:00 | 310.48 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-02-11 09:30:00 | 315.45 | 2026-02-11 09:50:00 | 316.51 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-04 09:45:00 | 262.75 | 2026-03-04 09:50:00 | 261.21 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2026-03-06 10:45:00 | 246.05 | 2026-03-06 10:50:00 | 246.97 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-03-19 09:35:00 | 233.70 | 2026-03-19 14:10:00 | 231.79 | PARTIAL | 0.50 | 0.82% |
| SELL | retest1 | 2026-03-19 09:35:00 | 233.70 | 2026-03-19 15:20:00 | 232.55 | TARGET_HIT | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-23 11:00:00 | 273.75 | 2026-04-23 11:05:00 | 272.65 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-28 10:00:00 | 260.60 | 2026-04-28 10:40:00 | 259.56 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-29 09:45:00 | 262.70 | 2026-04-29 09:50:00 | 264.42 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-04-29 09:45:00 | 262.70 | 2026-04-29 10:00:00 | 264.64 | TARGET_HIT | 0.50 | 0.74% |
| BUY | retest1 | 2026-05-04 10:30:00 | 259.75 | 2026-05-04 10:35:00 | 261.57 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-05-04 10:30:00 | 259.75 | 2026-05-04 12:10:00 | 260.45 | TARGET_HIT | 0.50 | 0.27% |
