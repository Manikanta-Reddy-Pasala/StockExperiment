# EIH Ltd. (EIHOTEL)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 336.00
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
| ENTRY1 | 56 |
| ENTRY2 | 0 |
| PARTIAL | 22 |
| TARGET_HIT | 8 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 78 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 48
- **Target hits / Stop hits / Partials:** 8 / 48 / 22
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 11.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 46 | 18 | 39.1% | 4 | 28 | 14 | 0.10% | 4.5% |
| BUY @ 2nd Alert (retest1) | 46 | 18 | 39.1% | 4 | 28 | 14 | 0.10% | 4.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 32 | 12 | 37.5% | 4 | 20 | 8 | 0.23% | 7.3% |
| SELL @ 2nd Alert (retest1) | 32 | 12 | 37.5% | 4 | 20 | 8 | 0.23% | 7.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 78 | 30 | 38.5% | 8 | 48 | 22 | 0.15% | 11.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 09:50:00 | 483.30 | 480.79 | 0.00 | ORB-long ORB[474.80,479.05] vol=3.8x ATR=1.87 |
| Stop hit — per-position SL triggered | 2024-05-14 10:00:00 | 481.43 | 480.97 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 09:30:00 | 484.55 | 480.95 | 0.00 | ORB-long ORB[476.35,483.00] vol=3.0x ATR=2.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-21 10:10:00 | 487.95 | 483.71 | 0.00 | T1 1.5R @ 487.95 |
| Stop hit — per-position SL triggered | 2024-05-21 11:35:00 | 484.55 | 485.49 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 11:10:00 | 476.35 | 481.40 | 0.00 | ORB-short ORB[482.00,487.00] vol=2.0x ATR=1.24 |
| Stop hit — per-position SL triggered | 2024-05-23 11:15:00 | 477.59 | 481.00 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 11:00:00 | 476.65 | 480.93 | 0.00 | ORB-short ORB[480.90,485.75] vol=1.7x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-24 11:05:00 | 474.07 | 480.21 | 0.00 | T1 1.5R @ 474.07 |
| Stop hit — per-position SL triggered | 2024-05-24 11:30:00 | 476.65 | 479.04 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-06-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-12 09:35:00 | 425.00 | 428.03 | 0.00 | ORB-short ORB[428.50,432.80] vol=2.6x ATR=1.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 10:55:00 | 422.20 | 425.86 | 0.00 | T1 1.5R @ 422.20 |
| Stop hit — per-position SL triggered | 2024-06-12 11:20:00 | 425.00 | 425.70 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-14 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:50:00 | 434.60 | 430.03 | 0.00 | ORB-long ORB[426.75,429.80] vol=1.6x ATR=1.70 |
| Stop hit — per-position SL triggered | 2024-06-14 09:55:00 | 432.90 | 430.48 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 10:55:00 | 450.40 | 446.93 | 0.00 | ORB-long ORB[442.05,446.70] vol=3.8x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 11:10:00 | 452.98 | 447.45 | 0.00 | T1 1.5R @ 452.98 |
| Stop hit — per-position SL triggered | 2024-06-20 15:15:00 | 450.40 | 450.12 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-24 11:00:00 | 450.80 | 447.75 | 0.00 | ORB-long ORB[442.25,447.35] vol=2.8x ATR=1.74 |
| Stop hit — per-position SL triggered | 2024-06-24 11:10:00 | 449.06 | 448.05 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 09:40:00 | 455.95 | 453.98 | 0.00 | ORB-long ORB[448.50,455.00] vol=3.4x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 09:50:00 | 458.39 | 455.84 | 0.00 | T1 1.5R @ 458.39 |
| Stop hit — per-position SL triggered | 2024-06-25 10:05:00 | 455.95 | 456.02 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 09:55:00 | 434.60 | 436.58 | 0.00 | ORB-short ORB[435.00,440.00] vol=1.6x ATR=1.61 |
| Stop hit — per-position SL triggered | 2024-06-27 10:00:00 | 436.21 | 436.50 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:35:00 | 428.70 | 430.25 | 0.00 | ORB-short ORB[429.10,434.50] vol=1.6x ATR=1.63 |
| Stop hit — per-position SL triggered | 2024-07-02 09:40:00 | 430.33 | 430.19 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 10:45:00 | 429.60 | 426.43 | 0.00 | ORB-long ORB[422.10,428.20] vol=4.3x ATR=1.71 |
| Stop hit — per-position SL triggered | 2024-07-04 10:50:00 | 427.89 | 426.45 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 10:10:00 | 434.90 | 429.50 | 0.00 | ORB-long ORB[424.00,430.00] vol=1.6x ATR=1.43 |
| Stop hit — per-position SL triggered | 2024-07-05 10:15:00 | 433.47 | 429.66 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 09:45:00 | 425.20 | 427.28 | 0.00 | ORB-short ORB[427.30,432.60] vol=2.0x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:05:00 | 423.02 | 426.62 | 0.00 | T1 1.5R @ 423.02 |
| Target hit | 2024-07-10 11:50:00 | 423.30 | 423.09 | 0.00 | Trail-exit close>VWAP |

### Cycle 15 — BUY (started 2024-07-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:50:00 | 419.40 | 416.66 | 0.00 | ORB-long ORB[413.70,417.70] vol=2.7x ATR=1.42 |
| Stop hit — per-position SL triggered | 2024-07-26 10:55:00 | 417.98 | 416.71 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 10:00:00 | 442.60 | 438.15 | 0.00 | ORB-long ORB[434.95,439.90] vol=4.9x ATR=2.47 |
| Stop hit — per-position SL triggered | 2024-07-31 10:05:00 | 440.13 | 438.53 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-08-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 09:50:00 | 385.75 | 380.64 | 0.00 | ORB-long ORB[374.75,380.50] vol=1.9x ATR=2.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 10:35:00 | 388.78 | 383.70 | 0.00 | T1 1.5R @ 388.78 |
| Target hit | 2024-08-09 15:20:00 | 389.55 | 387.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2024-08-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 10:20:00 | 385.80 | 388.67 | 0.00 | ORB-short ORB[389.05,393.00] vol=1.5x ATR=1.59 |
| Stop hit — per-position SL triggered | 2024-08-13 11:00:00 | 387.39 | 387.44 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-19 10:15:00 | 375.70 | 378.36 | 0.00 | ORB-short ORB[377.90,383.10] vol=1.9x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-19 10:25:00 | 373.32 | 377.73 | 0.00 | T1 1.5R @ 373.32 |
| Stop hit — per-position SL triggered | 2024-08-19 10:35:00 | 375.70 | 377.44 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-08-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 10:10:00 | 365.65 | 368.62 | 0.00 | ORB-short ORB[370.50,374.20] vol=1.6x ATR=1.35 |
| Stop hit — per-position SL triggered | 2024-08-20 10:30:00 | 367.00 | 367.55 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:35:00 | 380.00 | 377.47 | 0.00 | ORB-long ORB[375.10,378.80] vol=2.2x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-22 10:40:00 | 382.24 | 377.87 | 0.00 | T1 1.5R @ 382.24 |
| Stop hit — per-position SL triggered | 2024-08-22 10:45:00 | 380.00 | 377.98 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:55:00 | 387.65 | 383.83 | 0.00 | ORB-long ORB[380.00,384.80] vol=8.6x ATR=2.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 10:55:00 | 391.05 | 386.30 | 0.00 | T1 1.5R @ 391.05 |
| Stop hit — per-position SL triggered | 2024-08-26 12:45:00 | 387.65 | 387.13 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 11:00:00 | 383.25 | 384.89 | 0.00 | ORB-short ORB[386.05,389.70] vol=1.6x ATR=1.00 |
| Stop hit — per-position SL triggered | 2024-08-28 11:40:00 | 384.25 | 384.49 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-30 10:55:00 | 376.85 | 377.12 | 0.00 | ORB-short ORB[377.50,381.40] vol=1.7x ATR=1.15 |
| Stop hit — per-position SL triggered | 2024-08-30 11:35:00 | 378.00 | 377.19 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-09-02 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-02 10:30:00 | 377.35 | 379.53 | 0.00 | ORB-short ORB[379.20,384.55] vol=1.9x ATR=1.21 |
| Stop hit — per-position SL triggered | 2024-09-02 10:55:00 | 378.56 | 379.28 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-09-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-13 09:35:00 | 379.00 | 382.39 | 0.00 | ORB-short ORB[381.30,386.15] vol=3.4x ATR=1.83 |
| Stop hit — per-position SL triggered | 2024-09-13 09:55:00 | 380.83 | 381.15 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-09-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 10:40:00 | 378.10 | 381.34 | 0.00 | ORB-short ORB[381.15,385.00] vol=1.6x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-16 10:55:00 | 376.09 | 379.92 | 0.00 | T1 1.5R @ 376.09 |
| Stop hit — per-position SL triggered | 2024-09-16 11:40:00 | 378.10 | 379.16 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 11:00:00 | 376.25 | 375.34 | 0.00 | ORB-long ORB[373.25,376.20] vol=5.6x ATR=0.90 |
| Stop hit — per-position SL triggered | 2024-09-18 12:10:00 | 375.35 | 375.47 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-09-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 10:35:00 | 372.15 | 375.48 | 0.00 | ORB-short ORB[377.05,382.25] vol=2.4x ATR=1.22 |
| Stop hit — per-position SL triggered | 2024-09-19 10:50:00 | 373.37 | 375.28 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 10:45:00 | 370.70 | 369.17 | 0.00 | ORB-long ORB[366.35,370.00] vol=2.9x ATR=0.95 |
| Stop hit — per-position SL triggered | 2024-09-26 12:15:00 | 369.75 | 369.59 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-10-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 10:50:00 | 374.15 | 375.67 | 0.00 | ORB-short ORB[376.50,380.05] vol=4.5x ATR=1.00 |
| Stop hit — per-position SL triggered | 2024-10-01 11:00:00 | 375.15 | 375.49 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-10-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 10:50:00 | 400.40 | 397.19 | 0.00 | ORB-long ORB[391.00,396.50] vol=2.6x ATR=1.32 |
| Stop hit — per-position SL triggered | 2024-10-09 11:00:00 | 399.08 | 397.37 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-10-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 09:30:00 | 390.00 | 391.11 | 0.00 | ORB-short ORB[390.40,394.95] vol=2.4x ATR=2.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:15:00 | 386.38 | 389.67 | 0.00 | T1 1.5R @ 386.38 |
| Target hit | 2024-10-22 15:20:00 | 374.00 | 378.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — SELL (started 2024-10-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:55:00 | 364.10 | 366.60 | 0.00 | ORB-short ORB[368.00,372.75] vol=2.0x ATR=1.84 |
| Stop hit — per-position SL triggered | 2024-10-25 10:20:00 | 365.94 | 365.75 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-10-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 10:50:00 | 360.75 | 359.26 | 0.00 | ORB-long ORB[354.45,358.40] vol=1.5x ATR=1.47 |
| Stop hit — per-position SL triggered | 2024-10-30 10:55:00 | 359.28 | 359.26 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-10-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 09:45:00 | 364.45 | 359.56 | 0.00 | ORB-long ORB[355.20,359.70] vol=1.6x ATR=2.28 |
| Stop hit — per-position SL triggered | 2024-10-31 10:10:00 | 362.17 | 361.31 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-11-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 10:00:00 | 369.35 | 366.07 | 0.00 | ORB-long ORB[362.30,364.90] vol=4.0x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-27 10:05:00 | 371.05 | 367.25 | 0.00 | T1 1.5R @ 371.05 |
| Stop hit — per-position SL triggered | 2024-11-27 10:10:00 | 369.35 | 367.84 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-11-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:50:00 | 374.45 | 371.86 | 0.00 | ORB-long ORB[367.55,372.60] vol=4.5x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 10:00:00 | 376.32 | 372.93 | 0.00 | T1 1.5R @ 376.32 |
| Target hit | 2024-11-28 10:35:00 | 376.55 | 377.69 | 0.00 | Trail-exit close<VWAP |

### Cycle 39 — BUY (started 2024-11-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 11:10:00 | 376.60 | 373.74 | 0.00 | ORB-long ORB[372.35,376.00] vol=2.1x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 11:30:00 | 378.62 | 374.35 | 0.00 | T1 1.5R @ 378.62 |
| Stop hit — per-position SL triggered | 2024-11-29 11:35:00 | 376.60 | 374.46 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-12-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 09:30:00 | 382.45 | 380.02 | 0.00 | ORB-long ORB[376.15,381.80] vol=1.6x ATR=1.40 |
| Stop hit — per-position SL triggered | 2024-12-03 09:45:00 | 381.05 | 380.26 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-12-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 09:30:00 | 407.25 | 404.47 | 0.00 | ORB-long ORB[400.90,404.95] vol=4.6x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 09:45:00 | 409.94 | 406.67 | 0.00 | T1 1.5R @ 409.94 |
| Target hit | 2024-12-06 11:10:00 | 408.40 | 409.03 | 0.00 | Trail-exit close<VWAP |

### Cycle 42 — BUY (started 2024-12-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 10:45:00 | 428.95 | 425.68 | 0.00 | ORB-long ORB[423.00,427.90] vol=4.6x ATR=2.16 |
| Stop hit — per-position SL triggered | 2024-12-12 10:50:00 | 426.79 | 425.73 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-12-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 09:30:00 | 427.75 | 426.31 | 0.00 | ORB-long ORB[423.05,426.70] vol=2.7x ATR=1.66 |
| Stop hit — per-position SL triggered | 2024-12-17 09:40:00 | 426.09 | 426.39 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-12-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 09:35:00 | 429.55 | 427.71 | 0.00 | ORB-long ORB[423.20,428.45] vol=1.6x ATR=1.89 |
| Stop hit — per-position SL triggered | 2024-12-18 09:45:00 | 427.66 | 427.89 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-01-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-07 11:10:00 | 416.00 | 411.87 | 0.00 | ORB-long ORB[408.25,413.40] vol=2.3x ATR=1.80 |
| Stop hit — per-position SL triggered | 2025-01-07 11:50:00 | 414.20 | 413.91 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-01-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:55:00 | 414.35 | 419.19 | 0.00 | ORB-short ORB[417.20,422.80] vol=1.7x ATR=1.32 |
| Stop hit — per-position SL triggered | 2025-01-09 11:20:00 | 415.67 | 418.61 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-01-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:20:00 | 402.00 | 404.12 | 0.00 | ORB-short ORB[404.50,408.90] vol=2.9x ATR=1.21 |
| Stop hit — per-position SL triggered | 2025-01-21 10:30:00 | 403.21 | 403.96 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-23 10:15:00 | 389.90 | 393.89 | 0.00 | ORB-short ORB[392.10,396.30] vol=2.8x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 10:35:00 | 387.53 | 391.60 | 0.00 | T1 1.5R @ 387.53 |
| Target hit | 2025-01-23 14:00:00 | 384.70 | 384.55 | 0.00 | Trail-exit close>VWAP |

### Cycle 49 — BUY (started 2025-01-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 09:45:00 | 361.00 | 358.94 | 0.00 | ORB-long ORB[355.30,360.00] vol=2.2x ATR=2.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 12:05:00 | 364.81 | 360.82 | 0.00 | T1 1.5R @ 364.81 |
| Stop hit — per-position SL triggered | 2025-01-29 12:30:00 | 361.00 | 361.21 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-02-01 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 10:20:00 | 365.45 | 369.17 | 0.00 | ORB-short ORB[371.05,374.00] vol=5.5x ATR=1.36 |
| Stop hit — per-position SL triggered | 2025-02-01 10:35:00 | 366.81 | 368.32 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-02-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-13 10:55:00 | 342.85 | 345.62 | 0.00 | ORB-short ORB[343.25,347.80] vol=2.1x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-13 11:35:00 | 340.25 | 344.75 | 0.00 | T1 1.5R @ 340.25 |
| Target hit | 2025-02-13 15:20:00 | 336.40 | 338.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — BUY (started 2025-03-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-12 10:20:00 | 345.10 | 342.70 | 0.00 | ORB-long ORB[339.60,343.85] vol=3.3x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 10:35:00 | 347.35 | 343.46 | 0.00 | T1 1.5R @ 347.35 |
| Target hit | 2025-03-12 11:20:00 | 346.20 | 346.41 | 0.00 | Trail-exit close<VWAP |

### Cycle 53 — BUY (started 2025-03-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:20:00 | 369.35 | 365.63 | 0.00 | ORB-long ORB[360.55,365.95] vol=2.4x ATR=2.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 10:25:00 | 372.51 | 367.60 | 0.00 | T1 1.5R @ 372.51 |
| Stop hit — per-position SL triggered | 2025-03-18 10:30:00 | 369.35 | 367.88 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-03-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 09:35:00 | 369.20 | 367.19 | 0.00 | ORB-long ORB[365.60,368.65] vol=2.1x ATR=1.34 |
| Stop hit — per-position SL triggered | 2025-03-19 10:45:00 | 367.86 | 368.45 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-03-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:40:00 | 365.40 | 362.05 | 0.00 | ORB-long ORB[357.75,362.50] vol=4.1x ATR=1.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 09:50:00 | 367.83 | 363.72 | 0.00 | T1 1.5R @ 367.83 |
| Stop hit — per-position SL triggered | 2025-03-21 13:05:00 | 365.40 | 365.32 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 09:35:00 | 380.25 | 382.15 | 0.00 | ORB-short ORB[382.30,386.00] vol=2.3x ATR=1.41 |
| Stop hit — per-position SL triggered | 2025-04-23 09:45:00 | 381.66 | 381.86 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-14 09:50:00 | 483.30 | 2024-05-14 10:00:00 | 481.43 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-05-21 09:30:00 | 484.55 | 2024-05-21 10:10:00 | 487.95 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2024-05-21 09:30:00 | 484.55 | 2024-05-21 11:35:00 | 484.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-23 11:10:00 | 476.35 | 2024-05-23 11:15:00 | 477.59 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-05-24 11:00:00 | 476.65 | 2024-05-24 11:05:00 | 474.07 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-05-24 11:00:00 | 476.65 | 2024-05-24 11:30:00 | 476.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-12 09:35:00 | 425.00 | 2024-06-12 10:55:00 | 422.20 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-06-12 09:35:00 | 425.00 | 2024-06-12 11:20:00 | 425.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-14 09:50:00 | 434.60 | 2024-06-14 09:55:00 | 432.90 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-06-20 10:55:00 | 450.40 | 2024-06-20 11:10:00 | 452.98 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-06-20 10:55:00 | 450.40 | 2024-06-20 15:15:00 | 450.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-24 11:00:00 | 450.80 | 2024-06-24 11:10:00 | 449.06 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-06-25 09:40:00 | 455.95 | 2024-06-25 09:50:00 | 458.39 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-06-25 09:40:00 | 455.95 | 2024-06-25 10:05:00 | 455.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-27 09:55:00 | 434.60 | 2024-06-27 10:00:00 | 436.21 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-07-02 09:35:00 | 428.70 | 2024-07-02 09:40:00 | 430.33 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-07-04 10:45:00 | 429.60 | 2024-07-04 10:50:00 | 427.89 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-07-05 10:10:00 | 434.90 | 2024-07-05 10:15:00 | 433.47 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-07-10 09:45:00 | 425.20 | 2024-07-10 10:05:00 | 423.02 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-07-10 09:45:00 | 425.20 | 2024-07-10 11:50:00 | 423.30 | TARGET_HIT | 0.50 | 0.45% |
| BUY | retest1 | 2024-07-26 10:50:00 | 419.40 | 2024-07-26 10:55:00 | 417.98 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-07-31 10:00:00 | 442.60 | 2024-07-31 10:05:00 | 440.13 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2024-08-09 09:50:00 | 385.75 | 2024-08-09 10:35:00 | 388.78 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2024-08-09 09:50:00 | 385.75 | 2024-08-09 15:20:00 | 389.55 | TARGET_HIT | 0.50 | 0.99% |
| SELL | retest1 | 2024-08-13 10:20:00 | 385.80 | 2024-08-13 11:00:00 | 387.39 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-08-19 10:15:00 | 375.70 | 2024-08-19 10:25:00 | 373.32 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-08-19 10:15:00 | 375.70 | 2024-08-19 10:35:00 | 375.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-20 10:10:00 | 365.65 | 2024-08-20 10:30:00 | 367.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-08-22 10:35:00 | 380.00 | 2024-08-22 10:40:00 | 382.24 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-08-22 10:35:00 | 380.00 | 2024-08-22 10:45:00 | 380.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-26 09:55:00 | 387.65 | 2024-08-26 10:55:00 | 391.05 | PARTIAL | 0.50 | 0.88% |
| BUY | retest1 | 2024-08-26 09:55:00 | 387.65 | 2024-08-26 12:45:00 | 387.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-28 11:00:00 | 383.25 | 2024-08-28 11:40:00 | 384.25 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-08-30 10:55:00 | 376.85 | 2024-08-30 11:35:00 | 378.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-09-02 10:30:00 | 377.35 | 2024-09-02 10:55:00 | 378.56 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-09-13 09:35:00 | 379.00 | 2024-09-13 09:55:00 | 380.83 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-09-16 10:40:00 | 378.10 | 2024-09-16 10:55:00 | 376.09 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-09-16 10:40:00 | 378.10 | 2024-09-16 11:40:00 | 378.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-18 11:00:00 | 376.25 | 2024-09-18 12:10:00 | 375.35 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-09-19 10:35:00 | 372.15 | 2024-09-19 10:50:00 | 373.37 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-09-26 10:45:00 | 370.70 | 2024-09-26 12:15:00 | 369.75 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-10-01 10:50:00 | 374.15 | 2024-10-01 11:00:00 | 375.15 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-10-09 10:50:00 | 400.40 | 2024-10-09 11:00:00 | 399.08 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-10-22 09:30:00 | 390.00 | 2024-10-22 10:15:00 | 386.38 | PARTIAL | 0.50 | 0.93% |
| SELL | retest1 | 2024-10-22 09:30:00 | 390.00 | 2024-10-22 15:20:00 | 374.00 | TARGET_HIT | 0.50 | 4.10% |
| SELL | retest1 | 2024-10-25 09:55:00 | 364.10 | 2024-10-25 10:20:00 | 365.94 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-10-30 10:50:00 | 360.75 | 2024-10-30 10:55:00 | 359.28 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-10-31 09:45:00 | 364.45 | 2024-10-31 10:10:00 | 362.17 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest1 | 2024-11-27 10:00:00 | 369.35 | 2024-11-27 10:05:00 | 371.05 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-11-27 10:00:00 | 369.35 | 2024-11-27 10:10:00 | 369.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-28 09:50:00 | 374.45 | 2024-11-28 10:00:00 | 376.32 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-11-28 09:50:00 | 374.45 | 2024-11-28 10:35:00 | 376.55 | TARGET_HIT | 0.50 | 0.56% |
| BUY | retest1 | 2024-11-29 11:10:00 | 376.60 | 2024-11-29 11:30:00 | 378.62 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-11-29 11:10:00 | 376.60 | 2024-11-29 11:35:00 | 376.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-03 09:30:00 | 382.45 | 2024-12-03 09:45:00 | 381.05 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-12-06 09:30:00 | 407.25 | 2024-12-06 09:45:00 | 409.94 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2024-12-06 09:30:00 | 407.25 | 2024-12-06 11:10:00 | 408.40 | TARGET_HIT | 0.50 | 0.28% |
| BUY | retest1 | 2024-12-12 10:45:00 | 428.95 | 2024-12-12 10:50:00 | 426.79 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-12-17 09:30:00 | 427.75 | 2024-12-17 09:40:00 | 426.09 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-12-18 09:35:00 | 429.55 | 2024-12-18 09:45:00 | 427.66 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-01-07 11:10:00 | 416.00 | 2025-01-07 11:50:00 | 414.20 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-01-09 10:55:00 | 414.35 | 2025-01-09 11:20:00 | 415.67 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-01-21 10:20:00 | 402.00 | 2025-01-21 10:30:00 | 403.21 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-01-23 10:15:00 | 389.90 | 2025-01-23 10:35:00 | 387.53 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-01-23 10:15:00 | 389.90 | 2025-01-23 14:00:00 | 384.70 | TARGET_HIT | 0.50 | 1.33% |
| BUY | retest1 | 2025-01-29 09:45:00 | 361.00 | 2025-01-29 12:05:00 | 364.81 | PARTIAL | 0.50 | 1.05% |
| BUY | retest1 | 2025-01-29 09:45:00 | 361.00 | 2025-01-29 12:30:00 | 361.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-01 10:20:00 | 365.45 | 2025-02-01 10:35:00 | 366.81 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-02-13 10:55:00 | 342.85 | 2025-02-13 11:35:00 | 340.25 | PARTIAL | 0.50 | 0.76% |
| SELL | retest1 | 2025-02-13 10:55:00 | 342.85 | 2025-02-13 15:20:00 | 336.40 | TARGET_HIT | 0.50 | 1.88% |
| BUY | retest1 | 2025-03-12 10:20:00 | 345.10 | 2025-03-12 10:35:00 | 347.35 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2025-03-12 10:20:00 | 345.10 | 2025-03-12 11:20:00 | 346.20 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2025-03-18 10:20:00 | 369.35 | 2025-03-18 10:25:00 | 372.51 | PARTIAL | 0.50 | 0.86% |
| BUY | retest1 | 2025-03-18 10:20:00 | 369.35 | 2025-03-18 10:30:00 | 369.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-19 09:35:00 | 369.20 | 2025-03-19 10:45:00 | 367.86 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-03-21 09:40:00 | 365.40 | 2025-03-21 09:50:00 | 367.83 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2025-03-21 09:40:00 | 365.40 | 2025-03-21 13:05:00 | 365.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-23 09:35:00 | 380.25 | 2025-04-23 09:45:00 | 381.66 | STOP_HIT | 1.00 | -0.37% |
