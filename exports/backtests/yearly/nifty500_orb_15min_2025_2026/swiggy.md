# Swiggy Ltd. (SWIGGY)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 282.80
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
| PARTIAL | 25 |
| TARGET_HIT | 8 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 81 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 48
- **Target hits / Stop hits / Partials:** 8 / 48 / 25
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 8.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 46 | 19 | 41.3% | 6 | 27 | 13 | 0.13% | 6.1% |
| BUY @ 2nd Alert (retest1) | 46 | 19 | 41.3% | 6 | 27 | 13 | 0.13% | 6.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 35 | 14 | 40.0% | 2 | 21 | 12 | 0.07% | 2.5% |
| SELL @ 2nd Alert (retest1) | 35 | 14 | 40.0% | 2 | 21 | 12 | 0.07% | 2.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 81 | 33 | 40.7% | 8 | 48 | 25 | 0.11% | 8.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 10:50:00 | 313.00 | 310.31 | 0.00 | ORB-long ORB[308.80,311.20] vol=5.4x ATR=0.83 |
| Stop hit — per-position SL triggered | 2025-05-15 11:00:00 | 312.17 | 310.81 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-26 09:40:00 | 320.15 | 324.05 | 0.00 | ORB-short ORB[323.00,327.10] vol=2.7x ATR=1.39 |
| Stop hit — per-position SL triggered | 2025-05-26 10:00:00 | 321.54 | 322.94 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 10:15:00 | 344.25 | 340.55 | 0.00 | ORB-long ORB[337.45,341.70] vol=2.0x ATR=1.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 10:25:00 | 346.80 | 341.76 | 0.00 | T1 1.5R @ 346.80 |
| Stop hit — per-position SL triggered | 2025-06-03 10:30:00 | 344.25 | 341.98 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-06-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-10 10:55:00 | 361.75 | 364.30 | 0.00 | ORB-short ORB[362.30,366.90] vol=1.8x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-10 11:05:00 | 359.80 | 363.66 | 0.00 | T1 1.5R @ 359.80 |
| Target hit | 2025-06-10 13:10:00 | 359.65 | 359.63 | 0.00 | Trail-exit close>VWAP |

### Cycle 5 — SELL (started 2025-06-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-11 10:05:00 | 354.50 | 358.38 | 0.00 | ORB-short ORB[357.15,361.60] vol=2.0x ATR=1.82 |
| Stop hit — per-position SL triggered | 2025-06-11 10:15:00 | 356.32 | 357.98 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-16 09:50:00 | 359.00 | 356.10 | 0.00 | ORB-long ORB[352.20,357.10] vol=1.6x ATR=1.74 |
| Stop hit — per-position SL triggered | 2025-06-16 09:55:00 | 357.26 | 356.75 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-27 11:00:00 | 399.95 | 403.84 | 0.00 | ORB-short ORB[402.85,408.50] vol=2.1x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 11:15:00 | 397.80 | 403.37 | 0.00 | T1 1.5R @ 397.80 |
| Stop hit — per-position SL triggered | 2025-06-27 12:05:00 | 399.95 | 402.02 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-30 10:00:00 | 398.75 | 401.43 | 0.00 | ORB-short ORB[402.10,406.95] vol=1.5x ATR=1.61 |
| Stop hit — per-position SL triggered | 2025-06-30 10:05:00 | 400.36 | 401.32 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-07-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:45:00 | 394.20 | 398.22 | 0.00 | ORB-short ORB[396.45,402.10] vol=1.8x ATR=1.25 |
| Stop hit — per-position SL triggered | 2025-07-01 10:55:00 | 395.45 | 397.96 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-07-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 09:30:00 | 381.10 | 379.55 | 0.00 | ORB-long ORB[377.60,380.00] vol=2.6x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 09:40:00 | 383.03 | 381.79 | 0.00 | T1 1.5R @ 383.03 |
| Target hit | 2025-07-10 10:35:00 | 385.05 | 385.45 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — SELL (started 2025-07-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 09:30:00 | 379.70 | 381.03 | 0.00 | ORB-short ORB[380.05,383.85] vol=1.6x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 10:05:00 | 377.92 | 380.10 | 0.00 | T1 1.5R @ 377.92 |
| Stop hit — per-position SL triggered | 2025-07-11 10:20:00 | 379.70 | 379.96 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-14 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 10:50:00 | 387.60 | 385.98 | 0.00 | ORB-long ORB[380.40,384.50] vol=5.8x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 10:55:00 | 389.83 | 386.70 | 0.00 | T1 1.5R @ 389.83 |
| Stop hit — per-position SL triggered | 2025-07-14 11:10:00 | 387.60 | 386.77 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-07-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 09:45:00 | 398.10 | 396.32 | 0.00 | ORB-long ORB[392.25,397.70] vol=1.8x ATR=1.71 |
| Stop hit — per-position SL triggered | 2025-07-15 09:50:00 | 396.39 | 396.37 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-18 09:50:00 | 390.90 | 388.54 | 0.00 | ORB-long ORB[386.05,389.75] vol=1.8x ATR=1.19 |
| Stop hit — per-position SL triggered | 2025-07-18 09:55:00 | 389.71 | 388.61 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 09:55:00 | 423.10 | 420.26 | 0.00 | ORB-long ORB[415.75,421.50] vol=2.7x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 10:15:00 | 425.46 | 422.31 | 0.00 | T1 1.5R @ 425.46 |
| Target hit | 2025-07-24 11:25:00 | 424.40 | 424.62 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — SELL (started 2025-08-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 10:20:00 | 389.75 | 395.04 | 0.00 | ORB-short ORB[395.15,398.90] vol=1.7x ATR=1.58 |
| Stop hit — per-position SL triggered | 2025-08-05 10:30:00 | 391.33 | 394.63 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-08-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 09:55:00 | 398.75 | 401.30 | 0.00 | ORB-short ORB[399.05,403.80] vol=1.5x ATR=1.38 |
| Stop hit — per-position SL triggered | 2025-08-14 10:05:00 | 400.13 | 401.15 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-08-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 10:45:00 | 406.40 | 402.91 | 0.00 | ORB-long ORB[399.50,402.80] vol=2.3x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 10:50:00 | 407.76 | 403.20 | 0.00 | T1 1.5R @ 407.76 |
| Stop hit — per-position SL triggered | 2025-08-19 11:10:00 | 406.40 | 404.10 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-08-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-25 09:40:00 | 420.80 | 423.05 | 0.00 | ORB-short ORB[422.20,428.00] vol=1.5x ATR=1.64 |
| Stop hit — per-position SL triggered | 2025-08-25 09:45:00 | 422.44 | 422.90 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-09-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 09:40:00 | 431.30 | 428.75 | 0.00 | ORB-long ORB[424.00,429.00] vol=2.1x ATR=1.83 |
| Stop hit — per-position SL triggered | 2025-09-02 10:25:00 | 429.47 | 429.61 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-09-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-05 09:35:00 | 427.65 | 424.23 | 0.00 | ORB-long ORB[420.70,425.00] vol=1.8x ATR=1.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 09:40:00 | 430.49 | 425.23 | 0.00 | T1 1.5R @ 430.49 |
| Stop hit — per-position SL triggered | 2025-09-05 10:10:00 | 427.65 | 426.89 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-09-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 09:55:00 | 431.15 | 428.82 | 0.00 | ORB-long ORB[425.30,430.00] vol=2.0x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 10:30:00 | 433.45 | 429.72 | 0.00 | T1 1.5R @ 433.45 |
| Target hit | 2025-09-16 15:20:00 | 436.85 | 434.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — BUY (started 2025-09-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 10:30:00 | 441.55 | 439.79 | 0.00 | ORB-long ORB[436.25,440.85] vol=1.6x ATR=1.38 |
| Stop hit — per-position SL triggered | 2025-09-18 10:40:00 | 440.17 | 439.83 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-10-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-06 10:35:00 | 417.15 | 412.91 | 0.00 | ORB-long ORB[408.15,413.85] vol=1.7x ATR=1.40 |
| Stop hit — per-position SL triggered | 2025-10-06 11:50:00 | 415.75 | 414.24 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-10-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 10:10:00 | 416.25 | 418.40 | 0.00 | ORB-short ORB[418.50,422.65] vol=2.1x ATR=1.33 |
| Stop hit — per-position SL triggered | 2025-10-07 11:00:00 | 417.58 | 418.03 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 11:15:00 | 421.90 | 419.98 | 0.00 | ORB-long ORB[416.50,421.40] vol=1.6x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 12:35:00 | 423.37 | 420.69 | 0.00 | T1 1.5R @ 423.37 |
| Stop hit — per-position SL triggered | 2025-10-08 12:55:00 | 421.90 | 420.81 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-10-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 09:30:00 | 429.10 | 425.87 | 0.00 | ORB-long ORB[421.05,426.85] vol=3.3x ATR=1.35 |
| Stop hit — per-position SL triggered | 2025-10-09 09:45:00 | 427.75 | 426.66 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-10-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:05:00 | 441.50 | 439.95 | 0.00 | ORB-long ORB[437.25,441.20] vol=1.5x ATR=1.46 |
| Stop hit — per-position SL triggered | 2025-10-15 10:15:00 | 440.04 | 439.99 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-10-24 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 10:25:00 | 428.45 | 426.62 | 0.00 | ORB-long ORB[424.60,427.45] vol=1.6x ATR=1.13 |
| Stop hit — per-position SL triggered | 2025-10-24 11:00:00 | 427.32 | 426.90 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-10-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-29 11:05:00 | 421.50 | 423.58 | 0.00 | ORB-short ORB[422.25,426.60] vol=4.2x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 11:10:00 | 420.09 | 423.23 | 0.00 | T1 1.5R @ 420.09 |
| Stop hit — per-position SL triggered | 2025-10-29 11:15:00 | 421.50 | 423.17 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 10:15:00 | 415.40 | 418.49 | 0.00 | ORB-short ORB[419.05,422.50] vol=2.3x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 10:30:00 | 413.69 | 417.86 | 0.00 | T1 1.5R @ 413.69 |
| Stop hit — per-position SL triggered | 2025-10-30 11:25:00 | 415.40 | 416.19 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-11-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 10:10:00 | 408.20 | 405.20 | 0.00 | ORB-long ORB[403.30,407.95] vol=2.2x ATR=1.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 10:20:00 | 410.30 | 406.25 | 0.00 | T1 1.5R @ 410.30 |
| Target hit | 2025-11-04 15:20:00 | 413.00 | 411.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — SELL (started 2025-11-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-17 10:40:00 | 389.55 | 392.30 | 0.00 | ORB-short ORB[390.90,395.00] vol=1.9x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-17 10:55:00 | 387.62 | 390.57 | 0.00 | T1 1.5R @ 387.62 |
| Stop hit — per-position SL triggered | 2025-11-17 11:20:00 | 389.55 | 389.20 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-12-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 10:40:00 | 385.55 | 382.12 | 0.00 | ORB-long ORB[378.50,382.85] vol=3.1x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 10:55:00 | 387.23 | 383.31 | 0.00 | T1 1.5R @ 387.23 |
| Target hit | 2025-12-01 15:20:00 | 387.50 | 387.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — BUY (started 2025-12-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-03 09:40:00 | 400.10 | 399.09 | 0.00 | ORB-long ORB[395.05,399.45] vol=3.8x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 09:45:00 | 402.84 | 400.15 | 0.00 | T1 1.5R @ 402.84 |
| Stop hit — per-position SL triggered | 2025-12-03 10:25:00 | 400.10 | 400.57 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 10:15:00 | 396.25 | 390.04 | 0.00 | ORB-long ORB[380.15,386.05] vol=1.6x ATR=2.23 |
| Stop hit — per-position SL triggered | 2025-12-09 10:55:00 | 394.02 | 392.32 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-12-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-23 10:55:00 | 401.80 | 403.24 | 0.00 | ORB-short ORB[401.90,406.90] vol=1.9x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 13:40:00 | 400.48 | 402.45 | 0.00 | T1 1.5R @ 400.48 |
| Stop hit — per-position SL triggered | 2025-12-23 15:00:00 | 401.80 | 402.51 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 11:15:00 | 401.30 | 403.60 | 0.00 | ORB-short ORB[402.70,406.60] vol=2.8x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 11:35:00 | 399.80 | 402.58 | 0.00 | T1 1.5R @ 399.80 |
| Stop hit — per-position SL triggered | 2025-12-24 12:40:00 | 401.30 | 402.09 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 11:15:00 | 393.40 | 389.17 | 0.00 | ORB-long ORB[386.75,389.15] vol=2.5x ATR=0.95 |
| Stop hit — per-position SL triggered | 2026-01-01 11:20:00 | 392.45 | 389.25 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2026-01-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-09 09:50:00 | 356.10 | 353.08 | 0.00 | ORB-long ORB[349.50,354.35] vol=3.6x ATR=1.67 |
| Stop hit — per-position SL triggered | 2026-01-09 09:55:00 | 354.43 | 353.36 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2026-01-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 09:55:00 | 343.55 | 345.05 | 0.00 | ORB-short ORB[344.50,348.60] vol=1.6x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 10:30:00 | 341.62 | 344.13 | 0.00 | T1 1.5R @ 341.62 |
| Stop hit — per-position SL triggered | 2026-01-14 11:15:00 | 343.55 | 343.82 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2026-01-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 10:35:00 | 322.55 | 325.63 | 0.00 | ORB-short ORB[324.05,328.15] vol=1.8x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:40:00 | 320.37 | 325.09 | 0.00 | T1 1.5R @ 320.37 |
| Stop hit — per-position SL triggered | 2026-01-21 11:00:00 | 322.55 | 324.19 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2026-02-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 11:05:00 | 322.00 | 318.87 | 0.00 | ORB-long ORB[309.35,313.00] vol=2.2x ATR=1.87 |
| Stop hit — per-position SL triggered | 2026-02-01 11:40:00 | 320.13 | 319.76 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2026-02-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:40:00 | 342.20 | 338.50 | 0.00 | ORB-long ORB[333.65,337.65] vol=4.5x ATR=1.63 |
| Stop hit — per-position SL triggered | 2026-02-16 10:55:00 | 340.57 | 339.29 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2026-02-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 09:40:00 | 321.05 | 322.61 | 0.00 | ORB-short ORB[321.35,324.90] vol=1.6x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 10:35:00 | 319.31 | 321.46 | 0.00 | T1 1.5R @ 319.31 |
| Target hit | 2026-02-23 14:15:00 | 320.45 | 320.44 | 0.00 | Trail-exit close>VWAP |

### Cycle 46 — SELL (started 2026-02-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:35:00 | 311.80 | 314.30 | 0.00 | ORB-short ORB[314.20,317.60] vol=1.6x ATR=1.12 |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 312.92 | 313.24 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:15:00 | 305.20 | 306.17 | 0.00 | ORB-short ORB[305.50,308.00] vol=1.5x ATR=0.83 |
| Stop hit — per-position SL triggered | 2026-02-27 10:20:00 | 306.03 | 306.13 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:15:00 | 298.60 | 296.12 | 0.00 | ORB-long ORB[292.80,297.20] vol=1.6x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 10:55:00 | 300.57 | 297.13 | 0.00 | T1 1.5R @ 300.57 |
| Stop hit — per-position SL triggered | 2026-03-18 11:45:00 | 298.60 | 297.64 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2026-03-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 10:40:00 | 290.25 | 286.46 | 0.00 | ORB-long ORB[284.05,287.30] vol=2.3x ATR=1.43 |
| Stop hit — per-position SL triggered | 2026-03-20 10:55:00 | 288.82 | 287.12 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2026-04-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 11:00:00 | 278.35 | 276.10 | 0.00 | ORB-long ORB[273.40,277.40] vol=1.8x ATR=0.90 |
| Stop hit — per-position SL triggered | 2026-04-10 11:05:00 | 277.45 | 276.15 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2026-04-15 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:10:00 | 273.35 | 271.77 | 0.00 | ORB-long ORB[268.80,272.70] vol=3.1x ATR=1.34 |
| Stop hit — per-position SL triggered | 2026-04-15 10:40:00 | 272.01 | 271.98 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2026-04-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 09:35:00 | 279.40 | 281.28 | 0.00 | ORB-short ORB[280.00,283.35] vol=2.4x ATR=1.20 |
| Stop hit — per-position SL triggered | 2026-04-17 09:45:00 | 280.60 | 281.17 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 280.40 | 279.52 | 0.00 | ORB-long ORB[277.45,280.25] vol=2.5x ATR=0.85 |
| Stop hit — per-position SL triggered | 2026-04-21 09:40:00 | 279.55 | 279.76 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 11:15:00 | 287.40 | 284.68 | 0.00 | ORB-long ORB[282.45,286.25] vol=6.7x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 11:30:00 | 288.79 | 285.76 | 0.00 | T1 1.5R @ 288.79 |
| Target hit | 2026-04-22 15:20:00 | 294.85 | 290.36 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — SELL (started 2026-04-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:50:00 | 282.00 | 284.28 | 0.00 | ORB-short ORB[284.00,287.55] vol=2.0x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:30:00 | 280.30 | 282.97 | 0.00 | T1 1.5R @ 280.30 |
| Stop hit — per-position SL triggered | 2026-04-28 11:05:00 | 282.00 | 282.20 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2026-05-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:45:00 | 276.40 | 273.49 | 0.00 | ORB-long ORB[270.70,273.75] vol=4.2x ATR=1.21 |
| Stop hit — per-position SL triggered | 2026-05-04 12:10:00 | 275.19 | 274.82 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-15 10:50:00 | 313.00 | 2025-05-15 11:00:00 | 312.17 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-05-26 09:40:00 | 320.15 | 2025-05-26 10:00:00 | 321.54 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-06-03 10:15:00 | 344.25 | 2025-06-03 10:25:00 | 346.80 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2025-06-03 10:15:00 | 344.25 | 2025-06-03 10:30:00 | 344.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-10 10:55:00 | 361.75 | 2025-06-10 11:05:00 | 359.80 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-06-10 10:55:00 | 361.75 | 2025-06-10 13:10:00 | 359.65 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2025-06-11 10:05:00 | 354.50 | 2025-06-11 10:15:00 | 356.32 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2025-06-16 09:50:00 | 359.00 | 2025-06-16 09:55:00 | 357.26 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2025-06-27 11:00:00 | 399.95 | 2025-06-27 11:15:00 | 397.80 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2025-06-27 11:00:00 | 399.95 | 2025-06-27 12:05:00 | 399.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-30 10:00:00 | 398.75 | 2025-06-30 10:05:00 | 400.36 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-07-01 10:45:00 | 394.20 | 2025-07-01 10:55:00 | 395.45 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-07-10 09:30:00 | 381.10 | 2025-07-10 09:40:00 | 383.03 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-07-10 09:30:00 | 381.10 | 2025-07-10 10:35:00 | 385.05 | TARGET_HIT | 0.50 | 1.04% |
| SELL | retest1 | 2025-07-11 09:30:00 | 379.70 | 2025-07-11 10:05:00 | 377.92 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-07-11 09:30:00 | 379.70 | 2025-07-11 10:20:00 | 379.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-14 10:50:00 | 387.60 | 2025-07-14 10:55:00 | 389.83 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-07-14 10:50:00 | 387.60 | 2025-07-14 11:10:00 | 387.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-15 09:45:00 | 398.10 | 2025-07-15 09:50:00 | 396.39 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-07-18 09:50:00 | 390.90 | 2025-07-18 09:55:00 | 389.71 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-07-24 09:55:00 | 423.10 | 2025-07-24 10:15:00 | 425.46 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-07-24 09:55:00 | 423.10 | 2025-07-24 11:25:00 | 424.40 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2025-08-05 10:20:00 | 389.75 | 2025-08-05 10:30:00 | 391.33 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-08-14 09:55:00 | 398.75 | 2025-08-14 10:05:00 | 400.13 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-08-19 10:45:00 | 406.40 | 2025-08-19 10:50:00 | 407.76 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-08-19 10:45:00 | 406.40 | 2025-08-19 11:10:00 | 406.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-25 09:40:00 | 420.80 | 2025-08-25 09:45:00 | 422.44 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-09-02 09:40:00 | 431.30 | 2025-09-02 10:25:00 | 429.47 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-09-05 09:35:00 | 427.65 | 2025-09-05 09:40:00 | 430.49 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-09-05 09:35:00 | 427.65 | 2025-09-05 10:10:00 | 427.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-16 09:55:00 | 431.15 | 2025-09-16 10:30:00 | 433.45 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-09-16 09:55:00 | 431.15 | 2025-09-16 15:20:00 | 436.85 | TARGET_HIT | 0.50 | 1.32% |
| BUY | retest1 | 2025-09-18 10:30:00 | 441.55 | 2025-09-18 10:40:00 | 440.17 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-10-06 10:35:00 | 417.15 | 2025-10-06 11:50:00 | 415.75 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-10-07 10:10:00 | 416.25 | 2025-10-07 11:00:00 | 417.58 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-10-08 11:15:00 | 421.90 | 2025-10-08 12:35:00 | 423.37 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-10-08 11:15:00 | 421.90 | 2025-10-08 12:55:00 | 421.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-09 09:30:00 | 429.10 | 2025-10-09 09:45:00 | 427.75 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-10-15 10:05:00 | 441.50 | 2025-10-15 10:15:00 | 440.04 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-10-24 10:25:00 | 428.45 | 2025-10-24 11:00:00 | 427.32 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-10-29 11:05:00 | 421.50 | 2025-10-29 11:10:00 | 420.09 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-10-29 11:05:00 | 421.50 | 2025-10-29 11:15:00 | 421.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-30 10:15:00 | 415.40 | 2025-10-30 10:30:00 | 413.69 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-10-30 10:15:00 | 415.40 | 2025-10-30 11:25:00 | 415.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-04 10:10:00 | 408.20 | 2025-11-04 10:20:00 | 410.30 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-11-04 10:10:00 | 408.20 | 2025-11-04 15:20:00 | 413.00 | TARGET_HIT | 0.50 | 1.18% |
| SELL | retest1 | 2025-11-17 10:40:00 | 389.55 | 2025-11-17 10:55:00 | 387.62 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-11-17 10:40:00 | 389.55 | 2025-11-17 11:20:00 | 389.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-01 10:40:00 | 385.55 | 2025-12-01 10:55:00 | 387.23 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-12-01 10:40:00 | 385.55 | 2025-12-01 15:20:00 | 387.50 | TARGET_HIT | 0.50 | 0.51% |
| BUY | retest1 | 2025-12-03 09:40:00 | 400.10 | 2025-12-03 09:45:00 | 402.84 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2025-12-03 09:40:00 | 400.10 | 2025-12-03 10:25:00 | 400.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-09 10:15:00 | 396.25 | 2025-12-09 10:55:00 | 394.02 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2025-12-23 10:55:00 | 401.80 | 2025-12-23 13:40:00 | 400.48 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-12-23 10:55:00 | 401.80 | 2025-12-23 15:00:00 | 401.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-24 11:15:00 | 401.30 | 2025-12-24 11:35:00 | 399.80 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-12-24 11:15:00 | 401.30 | 2025-12-24 12:40:00 | 401.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-01 11:15:00 | 393.40 | 2026-01-01 11:20:00 | 392.45 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-01-09 09:50:00 | 356.10 | 2026-01-09 09:55:00 | 354.43 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-01-14 09:55:00 | 343.55 | 2026-01-14 10:30:00 | 341.62 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2026-01-14 09:55:00 | 343.55 | 2026-01-14 11:15:00 | 343.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-21 10:35:00 | 322.55 | 2026-01-21 10:40:00 | 320.37 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2026-01-21 10:35:00 | 322.55 | 2026-01-21 11:00:00 | 322.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-01 11:05:00 | 322.00 | 2026-02-01 11:40:00 | 320.13 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2026-02-16 10:40:00 | 342.20 | 2026-02-16 10:55:00 | 340.57 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-02-23 09:40:00 | 321.05 | 2026-02-23 10:35:00 | 319.31 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2026-02-23 09:40:00 | 321.05 | 2026-02-23 14:15:00 | 320.45 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2026-02-25 09:35:00 | 311.80 | 2026-02-25 10:15:00 | 312.92 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-27 10:15:00 | 305.20 | 2026-02-27 10:20:00 | 306.03 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-03-18 10:15:00 | 298.60 | 2026-03-18 10:55:00 | 300.57 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-03-18 10:15:00 | 298.60 | 2026-03-18 11:45:00 | 298.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-20 10:40:00 | 290.25 | 2026-03-20 10:55:00 | 288.82 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2026-04-10 11:00:00 | 278.35 | 2026-04-10 11:05:00 | 277.45 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-15 10:10:00 | 273.35 | 2026-04-15 10:40:00 | 272.01 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-04-17 09:35:00 | 279.40 | 2026-04-17 09:45:00 | 280.60 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2026-04-21 09:35:00 | 280.40 | 2026-04-21 09:40:00 | 279.55 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-22 11:15:00 | 287.40 | 2026-04-22 11:30:00 | 288.79 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-04-22 11:15:00 | 287.40 | 2026-04-22 15:20:00 | 294.85 | TARGET_HIT | 0.50 | 2.59% |
| SELL | retest1 | 2026-04-28 09:50:00 | 282.00 | 2026-04-28 10:30:00 | 280.30 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-04-28 09:50:00 | 282.00 | 2026-04-28 11:05:00 | 282.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-04 10:45:00 | 276.40 | 2026-05-04 12:10:00 | 275.19 | STOP_HIT | 1.00 | -0.44% |
