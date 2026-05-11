# REC Ltd. (RECLTD)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 359.30
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
| ENTRY1 | 95 |
| ENTRY2 | 0 |
| PARTIAL | 41 |
| TARGET_HIT | 20 |
| STOP_HIT | 75 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 136 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 61 / 75
- **Target hits / Stop hits / Partials:** 20 / 75 / 41
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 20.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 68 | 32 | 47.1% | 11 | 36 | 21 | 0.16% | 10.6% |
| BUY @ 2nd Alert (retest1) | 68 | 32 | 47.1% | 11 | 36 | 21 | 0.16% | 10.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 68 | 29 | 42.6% | 9 | 39 | 20 | 0.15% | 10.0% |
| SELL @ 2nd Alert (retest1) | 68 | 29 | 42.6% | 9 | 39 | 20 | 0.15% | 10.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 136 | 61 | 44.9% | 20 | 75 | 41 | 0.15% | 20.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-19 10:00:00 | 408.00 | 409.45 | 0.00 | ORB-short ORB[408.55,412.00] vol=2.7x ATR=1.33 |
| Stop hit — per-position SL triggered | 2025-05-19 10:10:00 | 409.33 | 409.36 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 09:45:00 | 403.30 | 405.53 | 0.00 | ORB-short ORB[404.25,408.50] vol=1.8x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 10:00:00 | 401.91 | 404.67 | 0.00 | T1 1.5R @ 401.91 |
| Stop hit — per-position SL triggered | 2025-05-27 10:10:00 | 403.30 | 404.12 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 09:35:00 | 402.40 | 404.04 | 0.00 | ORB-short ORB[403.30,406.20] vol=1.8x ATR=0.88 |
| Stop hit — per-position SL triggered | 2025-05-29 10:00:00 | 403.28 | 403.72 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-06-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 09:30:00 | 399.00 | 400.35 | 0.00 | ORB-short ORB[399.10,403.90] vol=3.1x ATR=0.91 |
| Stop hit — per-position SL triggered | 2025-06-06 09:45:00 | 399.91 | 399.99 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 09:50:00 | 422.50 | 419.57 | 0.00 | ORB-long ORB[417.00,422.30] vol=1.6x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-09 10:30:00 | 424.69 | 421.31 | 0.00 | T1 1.5R @ 424.69 |
| Target hit | 2025-06-09 15:20:00 | 426.15 | 424.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2025-06-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 10:00:00 | 426.80 | 424.25 | 0.00 | ORB-long ORB[421.50,424.75] vol=4.2x ATR=1.00 |
| Stop hit — per-position SL triggered | 2025-06-11 10:05:00 | 425.80 | 424.37 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 09:40:00 | 414.30 | 416.49 | 0.00 | ORB-short ORB[415.30,419.00] vol=1.9x ATR=1.34 |
| Stop hit — per-position SL triggered | 2025-06-12 09:55:00 | 415.64 | 415.86 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:30:00 | 396.65 | 399.63 | 0.00 | ORB-short ORB[398.70,403.45] vol=1.6x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 09:45:00 | 394.88 | 397.74 | 0.00 | T1 1.5R @ 394.88 |
| Stop hit — per-position SL triggered | 2025-06-16 10:05:00 | 396.65 | 397.30 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-18 10:50:00 | 394.85 | 397.56 | 0.00 | ORB-short ORB[395.35,399.90] vol=1.9x ATR=1.04 |
| Stop hit — per-position SL triggered | 2025-06-18 10:55:00 | 395.89 | 397.53 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 10:00:00 | 387.75 | 389.56 | 0.00 | ORB-short ORB[388.50,392.65] vol=2.0x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:20:00 | 385.86 | 389.06 | 0.00 | T1 1.5R @ 385.86 |
| Target hit | 2025-06-19 15:20:00 | 383.55 | 385.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2025-06-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 09:30:00 | 403.40 | 402.01 | 0.00 | ORB-long ORB[400.25,402.95] vol=2.3x ATR=1.39 |
| Stop hit — per-position SL triggered | 2025-06-24 09:55:00 | 402.01 | 402.37 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-07-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:50:00 | 399.45 | 399.86 | 0.00 | ORB-short ORB[402.30,404.65] vol=1.6x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 11:00:00 | 397.84 | 399.65 | 0.00 | T1 1.5R @ 397.84 |
| Stop hit — per-position SL triggered | 2025-07-01 11:10:00 | 399.45 | 399.62 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 10:00:00 | 399.25 | 401.16 | 0.00 | ORB-short ORB[400.55,403.35] vol=1.5x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 10:05:00 | 398.05 | 400.93 | 0.00 | T1 1.5R @ 398.05 |
| Stop hit — per-position SL triggered | 2025-07-02 11:05:00 | 399.25 | 400.02 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-08 09:45:00 | 396.70 | 394.19 | 0.00 | ORB-long ORB[391.45,394.65] vol=1.7x ATR=0.93 |
| Stop hit — per-position SL triggered | 2025-07-08 09:50:00 | 395.77 | 394.35 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-09 10:30:00 | 393.10 | 394.21 | 0.00 | ORB-short ORB[394.20,395.85] vol=1.6x ATR=0.69 |
| Stop hit — per-position SL triggered | 2025-07-09 11:15:00 | 393.79 | 393.94 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-07-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 09:50:00 | 401.05 | 398.68 | 0.00 | ORB-long ORB[395.70,399.65] vol=2.3x ATR=1.07 |
| Stop hit — per-position SL triggered | 2025-07-11 10:10:00 | 399.98 | 399.69 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 10:20:00 | 402.30 | 401.25 | 0.00 | ORB-long ORB[400.40,402.25] vol=2.3x ATR=0.79 |
| Stop hit — per-position SL triggered | 2025-07-17 10:30:00 | 401.51 | 401.42 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-18 11:10:00 | 404.45 | 400.96 | 0.00 | ORB-long ORB[398.40,403.00] vol=5.4x ATR=1.11 |
| Stop hit — per-position SL triggered | 2025-07-18 11:50:00 | 403.34 | 401.87 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-31 09:45:00 | 394.45 | 396.19 | 0.00 | ORB-short ORB[395.00,398.45] vol=1.5x ATR=1.04 |
| Stop hit — per-position SL triggered | 2025-07-31 09:50:00 | 395.49 | 396.13 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-08-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 10:50:00 | 392.50 | 394.76 | 0.00 | ORB-short ORB[393.55,397.80] vol=2.7x ATR=0.93 |
| Stop hit — per-position SL triggered | 2025-08-05 12:15:00 | 393.43 | 394.30 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-08-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 11:00:00 | 391.20 | 391.75 | 0.00 | ORB-short ORB[392.55,395.50] vol=2.1x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 11:30:00 | 389.93 | 391.42 | 0.00 | T1 1.5R @ 389.93 |
| Stop hit — per-position SL triggered | 2025-08-06 12:00:00 | 391.20 | 391.26 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-08-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 11:10:00 | 386.05 | 387.46 | 0.00 | ORB-short ORB[386.10,389.00] vol=3.6x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 11:30:00 | 384.70 | 387.08 | 0.00 | T1 1.5R @ 384.70 |
| Target hit | 2025-08-07 14:50:00 | 384.60 | 383.92 | 0.00 | Trail-exit close>VWAP |

### Cycle 23 — SELL (started 2025-08-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 09:30:00 | 382.85 | 384.27 | 0.00 | ORB-short ORB[384.10,386.00] vol=3.5x ATR=0.82 |
| Stop hit — per-position SL triggered | 2025-08-14 09:45:00 | 383.67 | 383.97 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 09:30:00 | 378.00 | 379.38 | 0.00 | ORB-short ORB[379.30,381.40] vol=2.4x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 10:00:00 | 376.86 | 378.66 | 0.00 | T1 1.5R @ 376.86 |
| Stop hit — per-position SL triggered | 2025-08-22 11:20:00 | 378.00 | 378.17 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-08-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:30:00 | 369.40 | 371.02 | 0.00 | ORB-short ORB[370.50,375.50] vol=4.4x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 09:35:00 | 368.19 | 370.67 | 0.00 | T1 1.5R @ 368.19 |
| Stop hit — per-position SL triggered | 2025-08-26 09:40:00 | 369.40 | 370.53 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-08-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 09:50:00 | 349.60 | 352.03 | 0.00 | ORB-short ORB[351.70,355.80] vol=1.5x ATR=1.08 |
| Stop hit — per-position SL triggered | 2025-08-29 10:20:00 | 350.68 | 351.34 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-09-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 11:00:00 | 355.25 | 353.45 | 0.00 | ORB-long ORB[351.00,353.45] vol=1.5x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 11:35:00 | 356.38 | 353.79 | 0.00 | T1 1.5R @ 356.38 |
| Target hit | 2025-09-01 15:20:00 | 361.10 | 356.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — BUY (started 2025-09-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 09:30:00 | 362.85 | 361.74 | 0.00 | ORB-long ORB[360.15,362.70] vol=1.7x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 09:35:00 | 364.12 | 362.14 | 0.00 | T1 1.5R @ 364.12 |
| Target hit | 2025-09-02 13:20:00 | 366.35 | 366.63 | 0.00 | Trail-exit close<VWAP |

### Cycle 29 — BUY (started 2025-09-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 09:50:00 | 369.65 | 368.70 | 0.00 | ORB-long ORB[366.75,368.70] vol=1.6x ATR=0.82 |
| Stop hit — per-position SL triggered | 2025-09-08 10:30:00 | 368.83 | 368.86 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-09-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 09:40:00 | 375.40 | 373.76 | 0.00 | ORB-long ORB[371.00,374.50] vol=3.0x ATR=0.94 |
| Stop hit — per-position SL triggered | 2025-09-11 09:45:00 | 374.46 | 373.84 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-09-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 09:50:00 | 372.05 | 373.36 | 0.00 | ORB-short ORB[373.05,375.45] vol=3.3x ATR=0.69 |
| Stop hit — per-position SL triggered | 2025-09-12 10:50:00 | 372.74 | 372.81 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-09-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 09:30:00 | 380.75 | 379.67 | 0.00 | ORB-long ORB[377.80,380.45] vol=1.8x ATR=0.79 |
| Stop hit — per-position SL triggered | 2025-09-16 09:40:00 | 379.96 | 379.77 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-09-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 10:25:00 | 384.00 | 381.70 | 0.00 | ORB-long ORB[380.65,381.90] vol=2.3x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 10:30:00 | 385.09 | 382.31 | 0.00 | T1 1.5R @ 385.09 |
| Stop hit — per-position SL triggered | 2025-09-17 10:40:00 | 384.00 | 382.47 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 09:30:00 | 388.70 | 386.02 | 0.00 | ORB-long ORB[383.50,387.50] vol=1.7x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-18 09:35:00 | 390.37 | 387.11 | 0.00 | T1 1.5R @ 390.37 |
| Stop hit — per-position SL triggered | 2025-09-18 09:40:00 | 388.70 | 387.28 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 11:15:00 | 387.05 | 385.38 | 0.00 | ORB-long ORB[383.20,385.90] vol=3.2x ATR=0.66 |
| Stop hit — per-position SL triggered | 2025-09-19 11:50:00 | 386.39 | 385.58 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-23 09:40:00 | 387.00 | 386.32 | 0.00 | ORB-long ORB[383.95,386.95] vol=2.8x ATR=0.88 |
| Stop hit — per-position SL triggered | 2025-09-23 09:45:00 | 386.12 | 386.33 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-09-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 11:00:00 | 371.60 | 370.30 | 0.00 | ORB-long ORB[367.25,370.00] vol=1.9x ATR=0.81 |
| Stop hit — per-position SL triggered | 2025-09-29 11:20:00 | 370.79 | 370.37 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-10-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 10:45:00 | 377.05 | 378.51 | 0.00 | ORB-short ORB[378.45,381.95] vol=2.9x ATR=0.84 |
| Stop hit — per-position SL triggered | 2025-10-06 14:35:00 | 377.89 | 377.78 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-10-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 11:10:00 | 378.00 | 378.97 | 0.00 | ORB-short ORB[378.15,380.25] vol=5.0x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 11:20:00 | 376.99 | 378.85 | 0.00 | T1 1.5R @ 376.99 |
| Stop hit — per-position SL triggered | 2025-10-07 13:00:00 | 378.00 | 378.08 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-10-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 11:00:00 | 374.55 | 376.30 | 0.00 | ORB-short ORB[376.25,379.30] vol=3.1x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 11:20:00 | 373.32 | 375.81 | 0.00 | T1 1.5R @ 373.32 |
| Stop hit — per-position SL triggered | 2025-10-08 13:30:00 | 374.55 | 374.94 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-10-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-09 10:55:00 | 370.85 | 371.99 | 0.00 | ORB-short ORB[371.55,374.25] vol=1.5x ATR=0.74 |
| Stop hit — per-position SL triggered | 2025-10-09 15:15:00 | 371.59 | 371.19 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-10-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 09:55:00 | 370.65 | 372.30 | 0.00 | ORB-short ORB[372.20,374.00] vol=2.4x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 11:15:00 | 369.38 | 371.16 | 0.00 | T1 1.5R @ 369.38 |
| Stop hit — per-position SL triggered | 2025-10-14 13:00:00 | 370.65 | 370.28 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-10-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:45:00 | 374.60 | 373.29 | 0.00 | ORB-long ORB[370.70,373.60] vol=1.6x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 10:55:00 | 375.84 | 373.69 | 0.00 | T1 1.5R @ 375.84 |
| Target hit | 2025-10-15 15:00:00 | 375.35 | 375.91 | 0.00 | Trail-exit close<VWAP |

### Cycle 44 — BUY (started 2025-10-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 10:45:00 | 379.50 | 377.67 | 0.00 | ORB-long ORB[376.50,378.55] vol=2.5x ATR=0.80 |
| Stop hit — per-position SL triggered | 2025-10-16 10:55:00 | 378.70 | 377.80 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-10-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 09:45:00 | 372.00 | 373.74 | 0.00 | ORB-short ORB[373.00,378.00] vol=1.6x ATR=1.00 |
| Stop hit — per-position SL triggered | 2025-10-17 14:20:00 | 373.00 | 372.43 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-10-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 11:10:00 | 370.00 | 372.96 | 0.00 | ORB-short ORB[373.95,376.90] vol=3.8x ATR=0.65 |
| Stop hit — per-position SL triggered | 2025-10-28 11:20:00 | 370.65 | 372.75 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-10-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:25:00 | 375.40 | 373.19 | 0.00 | ORB-long ORB[369.70,372.00] vol=3.1x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 10:30:00 | 376.83 | 373.72 | 0.00 | T1 1.5R @ 376.83 |
| Stop hit — per-position SL triggered | 2025-10-29 10:35:00 | 375.40 | 373.82 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-10-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 11:00:00 | 376.20 | 378.35 | 0.00 | ORB-short ORB[377.00,380.60] vol=2.8x ATR=0.81 |
| Stop hit — per-position SL triggered | 2025-10-31 11:20:00 | 377.01 | 378.25 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-11-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 10:45:00 | 377.50 | 376.47 | 0.00 | ORB-long ORB[374.90,377.45] vol=4.6x ATR=0.92 |
| Stop hit — per-position SL triggered | 2025-11-03 11:25:00 | 376.58 | 376.60 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 10:15:00 | 376.85 | 378.01 | 0.00 | ORB-short ORB[377.20,379.00] vol=2.6x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 10:45:00 | 375.72 | 377.77 | 0.00 | T1 1.5R @ 375.72 |
| Target hit | 2025-11-04 15:20:00 | 370.70 | 373.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — SELL (started 2025-11-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 09:45:00 | 368.25 | 370.33 | 0.00 | ORB-short ORB[369.60,372.70] vol=1.8x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 10:10:00 | 366.82 | 369.14 | 0.00 | T1 1.5R @ 366.82 |
| Target hit | 2025-11-06 15:20:00 | 362.60 | 365.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — BUY (started 2025-11-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 10:05:00 | 362.00 | 359.84 | 0.00 | ORB-long ORB[358.35,361.80] vol=2.0x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 10:10:00 | 363.69 | 360.38 | 0.00 | T1 1.5R @ 363.69 |
| Stop hit — per-position SL triggered | 2025-11-07 10:25:00 | 362.00 | 360.62 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-11-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-10 10:35:00 | 363.30 | 365.19 | 0.00 | ORB-short ORB[364.00,366.70] vol=2.2x ATR=0.93 |
| Stop hit — per-position SL triggered | 2025-11-10 11:05:00 | 364.23 | 364.93 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-11-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 09:45:00 | 358.65 | 360.12 | 0.00 | ORB-short ORB[358.85,363.50] vol=1.7x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:30:00 | 357.38 | 359.28 | 0.00 | T1 1.5R @ 357.38 |
| Stop hit — per-position SL triggered | 2025-11-11 11:50:00 | 358.65 | 358.80 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-11-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 09:45:00 | 367.55 | 365.81 | 0.00 | ORB-long ORB[363.05,366.90] vol=1.7x ATR=0.80 |
| Stop hit — per-position SL triggered | 2025-11-12 09:50:00 | 366.75 | 365.92 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-11-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 10:50:00 | 363.85 | 362.97 | 0.00 | ORB-long ORB[361.60,363.45] vol=2.7x ATR=0.61 |
| Stop hit — per-position SL triggered | 2025-11-13 11:10:00 | 363.24 | 363.12 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-11-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 09:55:00 | 361.75 | 361.11 | 0.00 | ORB-long ORB[358.30,361.35] vol=1.6x ATR=0.82 |
| Stop hit — per-position SL triggered | 2025-11-14 10:00:00 | 360.93 | 361.12 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-11-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 09:40:00 | 361.90 | 360.84 | 0.00 | ORB-long ORB[359.90,361.25] vol=1.9x ATR=0.57 |
| Stop hit — per-position SL triggered | 2025-11-20 09:45:00 | 361.33 | 360.96 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:15:00 | 358.20 | 359.53 | 0.00 | ORB-short ORB[358.70,361.20] vol=1.8x ATR=0.65 |
| Stop hit — per-position SL triggered | 2025-11-21 11:10:00 | 358.85 | 359.06 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-11-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 10:50:00 | 361.70 | 360.62 | 0.00 | ORB-long ORB[356.00,360.00] vol=2.1x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 12:40:00 | 362.88 | 361.13 | 0.00 | T1 1.5R @ 362.88 |
| Target hit | 2025-11-27 15:10:00 | 362.05 | 362.11 | 0.00 | Trail-exit close<VWAP |

### Cycle 61 — SELL (started 2025-12-01 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 09:55:00 | 360.00 | 361.17 | 0.00 | ORB-short ORB[361.10,362.80] vol=1.5x ATR=0.63 |
| Stop hit — per-position SL triggered | 2025-12-01 10:25:00 | 360.63 | 361.00 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-12-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 11:10:00 | 357.95 | 356.76 | 0.00 | ORB-long ORB[355.15,357.30] vol=1.6x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 11:55:00 | 359.16 | 357.03 | 0.00 | T1 1.5R @ 359.16 |
| Stop hit — per-position SL triggered | 2025-12-02 13:05:00 | 357.95 | 357.32 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-12-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 11:10:00 | 350.00 | 353.70 | 0.00 | ORB-short ORB[355.90,359.20] vol=2.1x ATR=0.69 |
| Stop hit — per-position SL triggered | 2025-12-03 11:35:00 | 350.69 | 353.18 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-12-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 10:25:00 | 354.40 | 352.42 | 0.00 | ORB-long ORB[350.10,352.50] vol=1.5x ATR=0.81 |
| Stop hit — per-position SL triggered | 2025-12-04 10:35:00 | 353.59 | 352.51 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-12-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 09:35:00 | 349.35 | 350.92 | 0.00 | ORB-short ORB[350.25,353.85] vol=1.5x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 09:45:00 | 348.00 | 350.01 | 0.00 | T1 1.5R @ 348.00 |
| Target hit | 2025-12-08 15:20:00 | 342.50 | 344.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — BUY (started 2025-12-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 09:30:00 | 347.00 | 345.55 | 0.00 | ORB-long ORB[342.85,346.00] vol=2.1x ATR=0.92 |
| Stop hit — per-position SL triggered | 2025-12-10 09:55:00 | 346.08 | 346.21 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-12-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 10:40:00 | 338.60 | 339.91 | 0.00 | ORB-short ORB[339.95,343.00] vol=2.6x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 11:15:00 | 337.84 | 339.56 | 0.00 | T1 1.5R @ 337.84 |
| Target hit | 2025-12-16 15:20:00 | 335.35 | 337.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — SELL (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 11:15:00 | 335.00 | 335.30 | 0.00 | ORB-short ORB[335.20,337.30] vol=1.7x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 11:25:00 | 334.16 | 335.27 | 0.00 | T1 1.5R @ 334.16 |
| Target hit | 2025-12-17 15:10:00 | 334.60 | 334.58 | 0.00 | Trail-exit close>VWAP |

### Cycle 69 — SELL (started 2025-12-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 09:55:00 | 331.30 | 331.82 | 0.00 | ORB-short ORB[331.40,334.90] vol=4.6x ATR=0.70 |
| Stop hit — per-position SL triggered | 2025-12-18 10:05:00 | 332.00 | 331.80 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-12-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 09:30:00 | 347.30 | 346.27 | 0.00 | ORB-long ORB[343.35,347.05] vol=3.3x ATR=0.80 |
| Stop hit — per-position SL triggered | 2025-12-23 09:35:00 | 346.50 | 346.31 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-12-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 09:35:00 | 357.35 | 356.08 | 0.00 | ORB-long ORB[353.55,356.40] vol=1.6x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 09:40:00 | 358.41 | 356.64 | 0.00 | T1 1.5R @ 358.41 |
| Target hit | 2025-12-26 11:15:00 | 358.20 | 358.56 | 0.00 | Trail-exit close<VWAP |

### Cycle 72 — SELL (started 2025-12-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 09:55:00 | 356.70 | 357.83 | 0.00 | ORB-short ORB[357.10,359.40] vol=1.6x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 10:10:00 | 355.55 | 357.32 | 0.00 | T1 1.5R @ 355.55 |
| Target hit | 2025-12-29 13:45:00 | 355.25 | 355.23 | 0.00 | Trail-exit close>VWAP |

### Cycle 73 — BUY (started 2026-01-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 09:45:00 | 385.05 | 382.19 | 0.00 | ORB-long ORB[378.60,383.00] vol=2.2x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 09:55:00 | 387.43 | 383.61 | 0.00 | T1 1.5R @ 387.43 |
| Stop hit — per-position SL triggered | 2026-01-06 10:05:00 | 385.05 | 383.87 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-01-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 09:50:00 | 386.25 | 384.30 | 0.00 | ORB-long ORB[382.70,385.90] vol=3.1x ATR=1.38 |
| Stop hit — per-position SL triggered | 2026-01-07 10:20:00 | 384.87 | 385.08 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2026-01-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:05:00 | 378.35 | 382.94 | 0.00 | ORB-short ORB[383.20,387.45] vol=2.4x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:20:00 | 376.51 | 382.25 | 0.00 | T1 1.5R @ 376.51 |
| Target hit | 2026-01-08 15:20:00 | 371.35 | 377.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 76 — BUY (started 2026-01-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 11:05:00 | 370.50 | 368.82 | 0.00 | ORB-long ORB[364.30,367.85] vol=4.6x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 11:45:00 | 372.07 | 369.52 | 0.00 | T1 1.5R @ 372.07 |
| Stop hit — per-position SL triggered | 2026-01-14 13:25:00 | 370.50 | 370.09 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:15:00 | 374.00 | 372.32 | 0.00 | ORB-long ORB[370.00,373.50] vol=1.8x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 10:30:00 | 375.53 | 373.16 | 0.00 | T1 1.5R @ 375.53 |
| Stop hit — per-position SL triggered | 2026-01-16 11:10:00 | 374.00 | 373.91 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-01-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-21 09:40:00 | 354.85 | 356.97 | 0.00 | ORB-short ORB[355.45,359.65] vol=1.6x ATR=1.60 |
| Stop hit — per-position SL triggered | 2026-01-21 10:05:00 | 356.45 | 356.07 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2026-01-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 11:00:00 | 369.70 | 366.54 | 0.00 | ORB-long ORB[364.50,367.85] vol=3.7x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 11:30:00 | 371.36 | 367.12 | 0.00 | T1 1.5R @ 371.36 |
| Target hit | 2026-01-28 15:20:00 | 377.10 | 372.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 80 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 342.15 | 344.75 | 0.00 | ORB-short ORB[344.00,348.50] vol=1.8x ATR=1.12 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 343.27 | 344.23 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2026-02-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:40:00 | 347.40 | 347.26 | 0.00 | ORB-long ORB[343.20,346.60] vol=2.5x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:05:00 | 348.96 | 347.48 | 0.00 | T1 1.5R @ 348.96 |
| Target hit | 2026-02-16 15:20:00 | 353.60 | 349.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 82 — BUY (started 2026-02-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:55:00 | 356.40 | 355.07 | 0.00 | ORB-long ORB[352.70,355.10] vol=1.8x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 12:15:00 | 357.55 | 355.83 | 0.00 | T1 1.5R @ 357.55 |
| Stop hit — per-position SL triggered | 2026-02-17 14:15:00 | 356.40 | 356.22 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2026-02-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:55:00 | 363.75 | 362.07 | 0.00 | ORB-long ORB[358.40,363.00] vol=2.2x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:15:00 | 365.11 | 362.58 | 0.00 | T1 1.5R @ 365.11 |
| Stop hit — per-position SL triggered | 2026-02-18 13:50:00 | 363.75 | 363.39 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 348.00 | 349.12 | 0.00 | ORB-short ORB[348.30,351.40] vol=1.7x ATR=0.88 |
| Stop hit — per-position SL triggered | 2026-02-24 09:45:00 | 348.88 | 348.78 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2026-02-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:05:00 | 354.05 | 355.34 | 0.00 | ORB-short ORB[354.75,357.50] vol=1.8x ATR=0.77 |
| Stop hit — per-position SL triggered | 2026-02-25 11:50:00 | 354.82 | 355.01 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:15:00 | 350.10 | 351.76 | 0.00 | ORB-short ORB[352.05,354.20] vol=1.7x ATR=0.79 |
| Stop hit — per-position SL triggered | 2026-02-27 10:20:00 | 350.89 | 351.71 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2026-03-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 10:45:00 | 336.70 | 332.89 | 0.00 | ORB-long ORB[330.35,333.95] vol=2.4x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:00:00 | 338.58 | 334.02 | 0.00 | T1 1.5R @ 338.58 |
| Target hit | 2026-03-05 14:15:00 | 337.10 | 337.23 | 0.00 | Trail-exit close<VWAP |

### Cycle 88 — BUY (started 2026-03-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:35:00 | 343.15 | 341.98 | 0.00 | ORB-long ORB[339.40,342.90] vol=1.7x ATR=1.23 |
| Stop hit — per-position SL triggered | 2026-03-18 09:55:00 | 341.92 | 342.46 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2026-03-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 11:05:00 | 328.45 | 326.52 | 0.00 | ORB-long ORB[323.25,327.65] vol=1.6x ATR=0.84 |
| Stop hit — per-position SL triggered | 2026-03-25 11:25:00 | 327.61 | 326.81 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2026-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:45:00 | 358.50 | 356.45 | 0.00 | ORB-long ORB[354.00,357.40] vol=2.0x ATR=1.06 |
| Stop hit — per-position SL triggered | 2026-04-16 09:50:00 | 357.44 | 356.75 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 11:15:00 | 372.00 | 368.51 | 0.00 | ORB-long ORB[362.70,366.70] vol=2.7x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-17 12:20:00 | 373.67 | 370.49 | 0.00 | T1 1.5R @ 373.67 |
| Target hit | 2026-04-17 15:20:00 | 373.00 | 372.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 92 — SELL (started 2026-04-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:00:00 | 370.10 | 373.56 | 0.00 | ORB-short ORB[373.35,378.75] vol=1.9x ATR=0.92 |
| Stop hit — per-position SL triggered | 2026-04-24 11:30:00 | 371.02 | 373.28 | 0.00 | SL hit |

### Cycle 93 — BUY (started 2026-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:30:00 | 380.50 | 377.86 | 0.00 | ORB-long ORB[374.20,379.25] vol=3.5x ATR=1.41 |
| Stop hit — per-position SL triggered | 2026-04-28 09:50:00 | 379.09 | 378.61 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 11:15:00 | 354.80 | 356.26 | 0.00 | ORB-short ORB[356.30,361.10] vol=2.3x ATR=0.98 |
| Stop hit — per-position SL triggered | 2026-04-30 11:45:00 | 355.78 | 356.08 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2026-05-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:40:00 | 361.70 | 360.15 | 0.00 | ORB-long ORB[357.60,361.45] vol=1.6x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 10:00:00 | 363.42 | 360.97 | 0.00 | T1 1.5R @ 363.42 |
| Target hit | 2026-05-07 10:55:00 | 362.90 | 363.00 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-19 10:00:00 | 408.00 | 2025-05-19 10:10:00 | 409.33 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-05-27 09:45:00 | 403.30 | 2025-05-27 10:00:00 | 401.91 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-05-27 09:45:00 | 403.30 | 2025-05-27 10:10:00 | 403.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-29 09:35:00 | 402.40 | 2025-05-29 10:00:00 | 403.28 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-06-06 09:30:00 | 399.00 | 2025-06-06 09:45:00 | 399.91 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-06-09 09:50:00 | 422.50 | 2025-06-09 10:30:00 | 424.69 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-06-09 09:50:00 | 422.50 | 2025-06-09 15:20:00 | 426.15 | TARGET_HIT | 0.50 | 0.86% |
| BUY | retest1 | 2025-06-11 10:00:00 | 426.80 | 2025-06-11 10:05:00 | 425.80 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-06-12 09:40:00 | 414.30 | 2025-06-12 09:55:00 | 415.64 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-06-16 09:30:00 | 396.65 | 2025-06-16 09:45:00 | 394.88 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-06-16 09:30:00 | 396.65 | 2025-06-16 10:05:00 | 396.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-18 10:50:00 | 394.85 | 2025-06-18 10:55:00 | 395.89 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-06-19 10:00:00 | 387.75 | 2025-06-19 10:20:00 | 385.86 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-06-19 10:00:00 | 387.75 | 2025-06-19 15:20:00 | 383.55 | TARGET_HIT | 0.50 | 1.08% |
| BUY | retest1 | 2025-06-24 09:30:00 | 403.40 | 2025-06-24 09:55:00 | 402.01 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-07-01 10:50:00 | 399.45 | 2025-07-01 11:00:00 | 397.84 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-07-01 10:50:00 | 399.45 | 2025-07-01 11:10:00 | 399.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-02 10:00:00 | 399.25 | 2025-07-02 10:05:00 | 398.05 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-07-02 10:00:00 | 399.25 | 2025-07-02 11:05:00 | 399.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-08 09:45:00 | 396.70 | 2025-07-08 09:50:00 | 395.77 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-07-09 10:30:00 | 393.10 | 2025-07-09 11:15:00 | 393.79 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-07-11 09:50:00 | 401.05 | 2025-07-11 10:10:00 | 399.98 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-07-17 10:20:00 | 402.30 | 2025-07-17 10:30:00 | 401.51 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-07-18 11:10:00 | 404.45 | 2025-07-18 11:50:00 | 403.34 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-07-31 09:45:00 | 394.45 | 2025-07-31 09:50:00 | 395.49 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-08-05 10:50:00 | 392.50 | 2025-08-05 12:15:00 | 393.43 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-08-06 11:00:00 | 391.20 | 2025-08-06 11:30:00 | 389.93 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-08-06 11:00:00 | 391.20 | 2025-08-06 12:00:00 | 391.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-07 11:10:00 | 386.05 | 2025-08-07 11:30:00 | 384.70 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-08-07 11:10:00 | 386.05 | 2025-08-07 14:50:00 | 384.60 | TARGET_HIT | 0.50 | 0.38% |
| SELL | retest1 | 2025-08-14 09:30:00 | 382.85 | 2025-08-14 09:45:00 | 383.67 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-08-22 09:30:00 | 378.00 | 2025-08-22 10:00:00 | 376.86 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-08-22 09:30:00 | 378.00 | 2025-08-22 11:20:00 | 378.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-26 09:30:00 | 369.40 | 2025-08-26 09:35:00 | 368.19 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-08-26 09:30:00 | 369.40 | 2025-08-26 09:40:00 | 369.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-29 09:50:00 | 349.60 | 2025-08-29 10:20:00 | 350.68 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-09-01 11:00:00 | 355.25 | 2025-09-01 11:35:00 | 356.38 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-09-01 11:00:00 | 355.25 | 2025-09-01 15:20:00 | 361.10 | TARGET_HIT | 0.50 | 1.65% |
| BUY | retest1 | 2025-09-02 09:30:00 | 362.85 | 2025-09-02 09:35:00 | 364.12 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-09-02 09:30:00 | 362.85 | 2025-09-02 13:20:00 | 366.35 | TARGET_HIT | 0.50 | 0.96% |
| BUY | retest1 | 2025-09-08 09:50:00 | 369.65 | 2025-09-08 10:30:00 | 368.83 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-09-11 09:40:00 | 375.40 | 2025-09-11 09:45:00 | 374.46 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-09-12 09:50:00 | 372.05 | 2025-09-12 10:50:00 | 372.74 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-09-16 09:30:00 | 380.75 | 2025-09-16 09:40:00 | 379.96 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-09-17 10:25:00 | 384.00 | 2025-09-17 10:30:00 | 385.09 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-09-17 10:25:00 | 384.00 | 2025-09-17 10:40:00 | 384.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-18 09:30:00 | 388.70 | 2025-09-18 09:35:00 | 390.37 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-09-18 09:30:00 | 388.70 | 2025-09-18 09:40:00 | 388.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-19 11:15:00 | 387.05 | 2025-09-19 11:50:00 | 386.39 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-09-23 09:40:00 | 387.00 | 2025-09-23 09:45:00 | 386.12 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-09-29 11:00:00 | 371.60 | 2025-09-29 11:20:00 | 370.79 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-10-06 10:45:00 | 377.05 | 2025-10-06 14:35:00 | 377.89 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-10-07 11:10:00 | 378.00 | 2025-10-07 11:20:00 | 376.99 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-10-07 11:10:00 | 378.00 | 2025-10-07 13:00:00 | 378.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-08 11:00:00 | 374.55 | 2025-10-08 11:20:00 | 373.32 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-10-08 11:00:00 | 374.55 | 2025-10-08 13:30:00 | 374.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-09 10:55:00 | 370.85 | 2025-10-09 15:15:00 | 371.59 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-10-14 09:55:00 | 370.65 | 2025-10-14 11:15:00 | 369.38 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-10-14 09:55:00 | 370.65 | 2025-10-14 13:00:00 | 370.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-15 10:45:00 | 374.60 | 2025-10-15 10:55:00 | 375.84 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-10-15 10:45:00 | 374.60 | 2025-10-15 15:00:00 | 375.35 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2025-10-16 10:45:00 | 379.50 | 2025-10-16 10:55:00 | 378.70 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-10-17 09:45:00 | 372.00 | 2025-10-17 14:20:00 | 373.00 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-10-28 11:10:00 | 370.00 | 2025-10-28 11:20:00 | 370.65 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-10-29 10:25:00 | 375.40 | 2025-10-29 10:30:00 | 376.83 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-10-29 10:25:00 | 375.40 | 2025-10-29 10:35:00 | 375.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-31 11:00:00 | 376.20 | 2025-10-31 11:20:00 | 377.01 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-11-03 10:45:00 | 377.50 | 2025-11-03 11:25:00 | 376.58 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-11-04 10:15:00 | 376.85 | 2025-11-04 10:45:00 | 375.72 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-11-04 10:15:00 | 376.85 | 2025-11-04 15:20:00 | 370.70 | TARGET_HIT | 0.50 | 1.63% |
| SELL | retest1 | 2025-11-06 09:45:00 | 368.25 | 2025-11-06 10:10:00 | 366.82 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-11-06 09:45:00 | 368.25 | 2025-11-06 15:20:00 | 362.60 | TARGET_HIT | 0.50 | 1.53% |
| BUY | retest1 | 2025-11-07 10:05:00 | 362.00 | 2025-11-07 10:10:00 | 363.69 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-11-07 10:05:00 | 362.00 | 2025-11-07 10:25:00 | 362.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-10 10:35:00 | 363.30 | 2025-11-10 11:05:00 | 364.23 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-11-11 09:45:00 | 358.65 | 2025-11-11 10:30:00 | 357.38 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-11-11 09:45:00 | 358.65 | 2025-11-11 11:50:00 | 358.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-12 09:45:00 | 367.55 | 2025-11-12 09:50:00 | 366.75 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-11-13 10:50:00 | 363.85 | 2025-11-13 11:10:00 | 363.24 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-11-14 09:55:00 | 361.75 | 2025-11-14 10:00:00 | 360.93 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-11-20 09:40:00 | 361.90 | 2025-11-20 09:45:00 | 361.33 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-11-21 10:15:00 | 358.20 | 2025-11-21 11:10:00 | 358.85 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-11-27 10:50:00 | 361.70 | 2025-11-27 12:40:00 | 362.88 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-11-27 10:50:00 | 361.70 | 2025-11-27 15:10:00 | 362.05 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2025-12-01 09:55:00 | 360.00 | 2025-12-01 10:25:00 | 360.63 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-12-02 11:10:00 | 357.95 | 2025-12-02 11:55:00 | 359.16 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-12-02 11:10:00 | 357.95 | 2025-12-02 13:05:00 | 357.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-03 11:10:00 | 350.00 | 2025-12-03 11:35:00 | 350.69 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-12-04 10:25:00 | 354.40 | 2025-12-04 10:35:00 | 353.59 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-12-08 09:35:00 | 349.35 | 2025-12-08 09:45:00 | 348.00 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-12-08 09:35:00 | 349.35 | 2025-12-08 15:20:00 | 342.50 | TARGET_HIT | 0.50 | 1.96% |
| BUY | retest1 | 2025-12-10 09:30:00 | 347.00 | 2025-12-10 09:55:00 | 346.08 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-12-16 10:40:00 | 338.60 | 2025-12-16 11:15:00 | 337.84 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2025-12-16 10:40:00 | 338.60 | 2025-12-16 15:20:00 | 335.35 | TARGET_HIT | 0.50 | 0.96% |
| SELL | retest1 | 2025-12-17 11:15:00 | 335.00 | 2025-12-17 11:25:00 | 334.16 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-12-17 11:15:00 | 335.00 | 2025-12-17 15:10:00 | 334.60 | TARGET_HIT | 0.50 | 0.12% |
| SELL | retest1 | 2025-12-18 09:55:00 | 331.30 | 2025-12-18 10:05:00 | 332.00 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-12-23 09:30:00 | 347.30 | 2025-12-23 09:35:00 | 346.50 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-12-26 09:35:00 | 357.35 | 2025-12-26 09:40:00 | 358.41 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-12-26 09:35:00 | 357.35 | 2025-12-26 11:15:00 | 358.20 | TARGET_HIT | 0.50 | 0.24% |
| SELL | retest1 | 2025-12-29 09:55:00 | 356.70 | 2025-12-29 10:10:00 | 355.55 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-12-29 09:55:00 | 356.70 | 2025-12-29 13:45:00 | 355.25 | TARGET_HIT | 0.50 | 0.41% |
| BUY | retest1 | 2026-01-06 09:45:00 | 385.05 | 2026-01-06 09:55:00 | 387.43 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-01-06 09:45:00 | 385.05 | 2026-01-06 10:05:00 | 385.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-07 09:50:00 | 386.25 | 2026-01-07 10:20:00 | 384.87 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-01-08 11:05:00 | 378.35 | 2026-01-08 11:20:00 | 376.51 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2026-01-08 11:05:00 | 378.35 | 2026-01-08 15:20:00 | 371.35 | TARGET_HIT | 0.50 | 1.85% |
| BUY | retest1 | 2026-01-14 11:05:00 | 370.50 | 2026-01-14 11:45:00 | 372.07 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-01-14 11:05:00 | 370.50 | 2026-01-14 13:25:00 | 370.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-16 10:15:00 | 374.00 | 2026-01-16 10:30:00 | 375.53 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-01-16 10:15:00 | 374.00 | 2026-01-16 11:10:00 | 374.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-21 09:40:00 | 354.85 | 2026-01-21 10:05:00 | 356.45 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-01-28 11:00:00 | 369.70 | 2026-01-28 11:30:00 | 371.36 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-01-28 11:00:00 | 369.70 | 2026-01-28 15:20:00 | 377.10 | TARGET_HIT | 0.50 | 2.00% |
| SELL | retest1 | 2026-02-13 09:30:00 | 342.15 | 2026-02-13 09:40:00 | 343.27 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-16 10:40:00 | 347.40 | 2026-02-16 11:05:00 | 348.96 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-02-16 10:40:00 | 347.40 | 2026-02-16 15:20:00 | 353.60 | TARGET_HIT | 0.50 | 1.78% |
| BUY | retest1 | 2026-02-17 10:55:00 | 356.40 | 2026-02-17 12:15:00 | 357.55 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2026-02-17 10:55:00 | 356.40 | 2026-02-17 14:15:00 | 356.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-18 10:55:00 | 363.75 | 2026-02-18 11:15:00 | 365.11 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-02-18 10:55:00 | 363.75 | 2026-02-18 13:50:00 | 363.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-24 09:30:00 | 348.00 | 2026-02-24 09:45:00 | 348.88 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-02-25 11:05:00 | 354.05 | 2026-02-25 11:50:00 | 354.82 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-27 10:15:00 | 350.10 | 2026-02-27 10:20:00 | 350.89 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-03-05 10:45:00 | 336.70 | 2026-03-05 11:00:00 | 338.58 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-03-05 10:45:00 | 336.70 | 2026-03-05 14:15:00 | 337.10 | TARGET_HIT | 0.50 | 0.12% |
| BUY | retest1 | 2026-03-18 09:35:00 | 343.15 | 2026-03-18 09:55:00 | 341.92 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-03-25 11:05:00 | 328.45 | 2026-03-25 11:25:00 | 327.61 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-04-16 09:45:00 | 358.50 | 2026-04-16 09:50:00 | 357.44 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-17 11:15:00 | 372.00 | 2026-04-17 12:20:00 | 373.67 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-04-17 11:15:00 | 372.00 | 2026-04-17 15:20:00 | 373.00 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2026-04-24 11:00:00 | 370.10 | 2026-04-24 11:30:00 | 371.02 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-04-28 09:30:00 | 380.50 | 2026-04-28 09:50:00 | 379.09 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-04-30 11:15:00 | 354.80 | 2026-04-30 11:45:00 | 355.78 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2026-05-07 09:40:00 | 361.70 | 2026-05-07 10:00:00 | 363.42 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-05-07 09:40:00 | 361.70 | 2026-05-07 10:55:00 | 362.90 | TARGET_HIT | 0.50 | 0.33% |
