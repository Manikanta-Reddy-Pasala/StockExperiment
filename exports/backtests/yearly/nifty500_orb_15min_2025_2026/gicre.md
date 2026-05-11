# General Insurance Corporation of India (GICRE)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2025-12-05 15:25:00 (10813 bars)
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
| ENTRY1 | 64 |
| ENTRY2 | 0 |
| PARTIAL | 31 |
| TARGET_HIT | 10 |
| STOP_HIT | 54 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 95 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 54
- **Target hits / Stop hits / Partials:** 10 / 54 / 31
- **Avg / median % per leg:** 0.13% / 0.00%
- **Sum % (uncompounded):** 12.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 10 | 34.5% | 2 | 19 | 8 | 0.03% | 0.9% |
| BUY @ 2nd Alert (retest1) | 29 | 10 | 34.5% | 2 | 19 | 8 | 0.03% | 0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 66 | 31 | 47.0% | 8 | 35 | 23 | 0.18% | 11.6% |
| SELL @ 2nd Alert (retest1) | 66 | 31 | 47.0% | 8 | 35 | 23 | 0.18% | 11.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 95 | 41 | 43.2% | 10 | 54 | 31 | 0.13% | 12.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-19 10:15:00 | 440.45 | 434.93 | 0.00 | ORB-long ORB[432.05,437.05] vol=3.0x ATR=2.04 |
| Stop hit — per-position SL triggered | 2025-05-19 10:25:00 | 438.41 | 435.90 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-22 09:30:00 | 428.80 | 431.36 | 0.00 | ORB-short ORB[430.05,436.05] vol=2.2x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-22 09:35:00 | 426.10 | 429.39 | 0.00 | T1 1.5R @ 426.10 |
| Stop hit — per-position SL triggered | 2025-05-22 10:20:00 | 428.80 | 428.59 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 09:45:00 | 420.50 | 417.68 | 0.00 | ORB-long ORB[415.35,419.75] vol=1.7x ATR=1.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 09:50:00 | 423.05 | 419.09 | 0.00 | T1 1.5R @ 423.05 |
| Stop hit — per-position SL triggered | 2025-05-28 10:10:00 | 420.50 | 421.18 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 10:00:00 | 401.70 | 403.60 | 0.00 | ORB-short ORB[402.60,408.35] vol=2.1x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 10:15:00 | 399.68 | 403.05 | 0.00 | T1 1.5R @ 399.68 |
| Stop hit — per-position SL triggered | 2025-05-30 10:25:00 | 401.70 | 402.75 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 09:35:00 | 407.30 | 409.04 | 0.00 | ORB-short ORB[409.00,411.40] vol=2.7x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 09:50:00 | 405.65 | 408.27 | 0.00 | T1 1.5R @ 405.65 |
| Stop hit — per-position SL triggered | 2025-06-03 10:00:00 | 407.30 | 408.02 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 09:30:00 | 402.65 | 404.90 | 0.00 | ORB-short ORB[404.00,407.90] vol=2.2x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 09:45:00 | 400.71 | 403.52 | 0.00 | T1 1.5R @ 400.71 |
| Stop hit — per-position SL triggered | 2025-06-04 10:40:00 | 402.65 | 402.58 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 11:05:00 | 406.80 | 404.31 | 0.00 | ORB-long ORB[401.60,406.75] vol=3.5x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-09 11:20:00 | 408.35 | 405.80 | 0.00 | T1 1.5R @ 408.35 |
| Stop hit — per-position SL triggered | 2025-06-09 11:55:00 | 406.80 | 406.36 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-17 10:30:00 | 385.65 | 387.12 | 0.00 | ORB-short ORB[385.85,390.95] vol=1.7x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-17 10:55:00 | 384.07 | 386.66 | 0.00 | T1 1.5R @ 384.07 |
| Target hit | 2025-06-17 15:20:00 | 379.00 | 382.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2025-06-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-18 10:55:00 | 375.90 | 379.45 | 0.00 | ORB-short ORB[378.20,382.50] vol=1.6x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-18 11:20:00 | 374.22 | 378.65 | 0.00 | T1 1.5R @ 374.22 |
| Target hit | 2025-06-18 15:20:00 | 371.40 | 374.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2025-06-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 09:45:00 | 376.50 | 372.99 | 0.00 | ORB-long ORB[369.25,374.90] vol=2.3x ATR=1.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 11:20:00 | 379.21 | 375.24 | 0.00 | T1 1.5R @ 379.21 |
| Stop hit — per-position SL triggered | 2025-06-20 11:50:00 | 376.50 | 375.48 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-30 09:30:00 | 384.95 | 386.12 | 0.00 | ORB-short ORB[385.45,388.85] vol=1.8x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-30 09:55:00 | 383.50 | 385.36 | 0.00 | T1 1.5R @ 383.50 |
| Stop hit — per-position SL triggered | 2025-06-30 10:55:00 | 384.95 | 384.30 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-07-01 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:10:00 | 381.80 | 384.76 | 0.00 | ORB-short ORB[385.50,389.65] vol=1.9x ATR=1.29 |
| Stop hit — per-position SL triggered | 2025-07-01 10:30:00 | 383.09 | 384.55 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 11:05:00 | 375.00 | 377.28 | 0.00 | ORB-short ORB[377.05,379.95] vol=2.0x ATR=0.58 |
| Stop hit — per-position SL triggered | 2025-07-08 12:35:00 | 375.58 | 376.94 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:45:00 | 373.50 | 375.12 | 0.00 | ORB-short ORB[374.25,377.55] vol=1.9x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 11:05:00 | 372.36 | 374.85 | 0.00 | T1 1.5R @ 372.36 |
| Stop hit — per-position SL triggered | 2025-07-11 11:10:00 | 373.50 | 374.63 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 09:30:00 | 389.55 | 391.57 | 0.00 | ORB-short ORB[390.00,394.95] vol=1.6x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 10:40:00 | 387.72 | 390.27 | 0.00 | T1 1.5R @ 387.72 |
| Target hit | 2025-07-17 15:20:00 | 387.00 | 387.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2025-07-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:20:00 | 382.15 | 385.17 | 0.00 | ORB-short ORB[386.75,389.35] vol=1.6x ATR=0.82 |
| Stop hit — per-position SL triggered | 2025-07-18 12:55:00 | 382.97 | 383.59 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-21 09:30:00 | 378.50 | 380.05 | 0.00 | ORB-short ORB[378.90,382.35] vol=1.7x ATR=1.02 |
| Stop hit — per-position SL triggered | 2025-07-21 09:50:00 | 379.52 | 379.76 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 09:55:00 | 380.05 | 381.35 | 0.00 | ORB-short ORB[381.35,383.25] vol=2.5x ATR=0.78 |
| Stop hit — per-position SL triggered | 2025-07-22 10:40:00 | 380.83 | 380.99 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 11:10:00 | 380.75 | 381.71 | 0.00 | ORB-short ORB[382.10,384.15] vol=1.5x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 12:50:00 | 379.82 | 381.23 | 0.00 | T1 1.5R @ 379.82 |
| Stop hit — per-position SL triggered | 2025-07-23 13:40:00 | 380.75 | 380.90 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 09:40:00 | 379.00 | 379.92 | 0.00 | ORB-short ORB[379.15,381.50] vol=2.7x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:15:00 | 377.75 | 379.27 | 0.00 | T1 1.5R @ 377.75 |
| Target hit | 2025-07-25 15:20:00 | 373.35 | 376.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — SELL (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-29 10:15:00 | 370.00 | 371.26 | 0.00 | ORB-short ORB[371.00,374.50] vol=2.5x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 10:55:00 | 368.12 | 370.76 | 0.00 | T1 1.5R @ 368.12 |
| Stop hit — per-position SL triggered | 2025-07-29 11:20:00 | 370.00 | 370.52 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-08-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-04 10:50:00 | 380.25 | 381.98 | 0.00 | ORB-short ORB[381.00,385.00] vol=1.6x ATR=1.00 |
| Stop hit — per-position SL triggered | 2025-08-04 11:05:00 | 381.25 | 381.94 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-08-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 11:05:00 | 379.65 | 384.18 | 0.00 | ORB-short ORB[384.15,389.20] vol=1.6x ATR=1.09 |
| Stop hit — per-position SL triggered | 2025-08-06 12:20:00 | 380.74 | 383.54 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 10:45:00 | 392.50 | 392.98 | 0.00 | ORB-short ORB[393.00,396.05] vol=1.6x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-14 12:10:00 | 391.23 | 392.29 | 0.00 | T1 1.5R @ 391.23 |
| Target hit | 2025-08-14 15:20:00 | 386.25 | 389.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — SELL (started 2025-08-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-20 11:10:00 | 389.70 | 390.16 | 0.00 | ORB-short ORB[389.95,393.30] vol=4.3x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 11:35:00 | 388.74 | 390.03 | 0.00 | T1 1.5R @ 388.74 |
| Target hit | 2025-08-20 15:20:00 | 386.95 | 388.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — SELL (started 2025-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 09:30:00 | 380.50 | 382.64 | 0.00 | ORB-short ORB[381.70,385.50] vol=1.9x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 09:35:00 | 378.97 | 381.01 | 0.00 | T1 1.5R @ 378.97 |
| Stop hit — per-position SL triggered | 2025-08-22 09:40:00 | 380.50 | 380.99 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-09-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-03 09:30:00 | 370.60 | 372.97 | 0.00 | ORB-short ORB[371.85,376.45] vol=3.0x ATR=1.06 |
| Stop hit — per-position SL triggered | 2025-09-03 09:50:00 | 371.66 | 372.55 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-09-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 10:30:00 | 370.00 | 368.23 | 0.00 | ORB-long ORB[366.85,369.10] vol=3.1x ATR=0.79 |
| Stop hit — per-position SL triggered | 2025-09-11 10:35:00 | 369.21 | 368.27 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-09-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 09:45:00 | 373.40 | 371.87 | 0.00 | ORB-long ORB[368.55,373.20] vol=2.1x ATR=1.11 |
| Stop hit — per-position SL triggered | 2025-09-15 11:00:00 | 372.29 | 372.32 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-16 11:15:00 | 371.15 | 372.43 | 0.00 | ORB-short ORB[371.80,374.95] vol=3.7x ATR=0.54 |
| Stop hit — per-position SL triggered | 2025-09-16 11:30:00 | 371.69 | 372.37 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-09-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 10:35:00 | 368.95 | 370.09 | 0.00 | ORB-short ORB[369.20,372.10] vol=2.5x ATR=0.68 |
| Stop hit — per-position SL triggered | 2025-09-17 10:40:00 | 369.63 | 369.90 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-09-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 10:10:00 | 365.00 | 365.96 | 0.00 | ORB-short ORB[365.40,368.50] vol=1.6x ATR=0.51 |
| Stop hit — per-position SL triggered | 2025-09-18 10:55:00 | 365.51 | 365.70 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-09-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 11:10:00 | 363.50 | 364.07 | 0.00 | ORB-short ORB[364.05,365.90] vol=2.5x ATR=0.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 14:30:00 | 362.80 | 363.57 | 0.00 | T1 1.5R @ 362.80 |
| Stop hit — per-position SL triggered | 2025-09-19 15:00:00 | 363.50 | 363.55 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-23 10:05:00 | 368.00 | 365.63 | 0.00 | ORB-long ORB[364.05,366.80] vol=3.3x ATR=1.05 |
| Stop hit — per-position SL triggered | 2025-09-23 10:10:00 | 366.95 | 365.92 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-09-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 11:10:00 | 364.30 | 363.43 | 0.00 | ORB-long ORB[359.15,364.00] vol=9.0x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 11:20:00 | 365.64 | 363.60 | 0.00 | T1 1.5R @ 365.64 |
| Stop hit — per-position SL triggered | 2025-09-29 11:35:00 | 364.30 | 363.65 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-30 09:45:00 | 366.95 | 365.90 | 0.00 | ORB-long ORB[362.05,366.60] vol=3.2x ATR=0.90 |
| Stop hit — per-position SL triggered | 2025-09-30 10:30:00 | 366.05 | 366.12 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-10-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-03 10:10:00 | 364.80 | 366.85 | 0.00 | ORB-short ORB[366.05,368.00] vol=1.9x ATR=0.82 |
| Stop hit — per-position SL triggered | 2025-10-03 11:00:00 | 365.62 | 365.74 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-10-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-09 09:40:00 | 370.35 | 372.40 | 0.00 | ORB-short ORB[371.90,375.30] vol=1.8x ATR=1.09 |
| Stop hit — per-position SL triggered | 2025-10-09 09:45:00 | 371.44 | 372.25 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-10-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:40:00 | 381.40 | 379.58 | 0.00 | ORB-long ORB[376.10,379.45] vol=1.6x ATR=1.27 |
| Stop hit — per-position SL triggered | 2025-10-10 09:55:00 | 380.13 | 379.76 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-10-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 10:55:00 | 383.10 | 379.78 | 0.00 | ORB-long ORB[377.60,380.35] vol=3.1x ATR=1.08 |
| Stop hit — per-position SL triggered | 2025-10-13 11:25:00 | 382.02 | 380.40 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-10-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 09:35:00 | 380.40 | 379.55 | 0.00 | ORB-long ORB[377.20,380.20] vol=2.2x ATR=1.20 |
| Stop hit — per-position SL triggered | 2025-10-15 09:55:00 | 379.20 | 379.76 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-10-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-16 10:50:00 | 379.70 | 381.13 | 0.00 | ORB-short ORB[380.40,383.85] vol=1.7x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 10:55:00 | 378.46 | 380.79 | 0.00 | T1 1.5R @ 378.46 |
| Stop hit — per-position SL triggered | 2025-10-16 11:05:00 | 379.70 | 380.35 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 10:15:00 | 387.25 | 386.23 | 0.00 | ORB-long ORB[382.55,386.10] vol=2.2x ATR=1.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 10:30:00 | 389.36 | 387.61 | 0.00 | T1 1.5R @ 389.36 |
| Target hit | 2025-10-17 11:20:00 | 388.70 | 388.74 | 0.00 | Trail-exit close<VWAP |

### Cycle 44 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 11:15:00 | 385.00 | 382.71 | 0.00 | ORB-long ORB[381.45,384.90] vol=5.5x ATR=0.85 |
| Stop hit — per-position SL triggered | 2025-10-20 11:20:00 | 384.15 | 382.78 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-10-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 10:20:00 | 389.10 | 387.32 | 0.00 | ORB-long ORB[385.40,387.35] vol=5.3x ATR=0.92 |
| Stop hit — per-position SL triggered | 2025-10-23 10:30:00 | 388.18 | 388.06 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-10-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-27 09:50:00 | 383.35 | 384.72 | 0.00 | ORB-short ORB[384.05,385.65] vol=2.1x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 10:50:00 | 381.79 | 384.07 | 0.00 | T1 1.5R @ 381.79 |
| Stop hit — per-position SL triggered | 2025-10-27 11:00:00 | 383.35 | 383.81 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-10-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 11:05:00 | 385.00 | 384.32 | 0.00 | ORB-long ORB[383.20,384.90] vol=3.1x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 11:25:00 | 385.91 | 384.43 | 0.00 | T1 1.5R @ 385.91 |
| Stop hit — per-position SL triggered | 2025-10-29 11:30:00 | 385.00 | 384.45 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 11:15:00 | 383.45 | 383.91 | 0.00 | ORB-short ORB[384.30,386.50] vol=2.2x ATR=0.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 12:25:00 | 382.70 | 383.71 | 0.00 | T1 1.5R @ 382.70 |
| Target hit | 2025-10-30 15:00:00 | 383.40 | 383.08 | 0.00 | Trail-exit close>VWAP |

### Cycle 49 — SELL (started 2025-10-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 09:30:00 | 381.50 | 382.02 | 0.00 | ORB-short ORB[381.60,383.60] vol=2.6x ATR=0.64 |
| Stop hit — per-position SL triggered | 2025-10-31 09:35:00 | 382.14 | 382.00 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-11-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 11:05:00 | 380.65 | 378.45 | 0.00 | ORB-long ORB[378.20,380.50] vol=1.8x ATR=0.74 |
| Stop hit — per-position SL triggered | 2025-11-04 11:40:00 | 379.91 | 378.90 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-11-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 10:10:00 | 374.55 | 375.14 | 0.00 | ORB-short ORB[375.20,376.80] vol=2.1x ATR=0.83 |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 375.38 | 374.96 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-11-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 09:30:00 | 368.20 | 368.97 | 0.00 | ORB-short ORB[369.00,374.10] vol=4.3x ATR=1.17 |
| Stop hit — per-position SL triggered | 2025-11-07 10:30:00 | 369.37 | 368.54 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-11-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 09:30:00 | 395.85 | 393.33 | 0.00 | ORB-long ORB[389.80,393.00] vol=4.6x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-14 09:40:00 | 398.09 | 396.06 | 0.00 | T1 1.5R @ 398.09 |
| Stop hit — per-position SL triggered | 2025-11-14 09:50:00 | 395.85 | 396.22 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-11-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-18 10:50:00 | 395.60 | 393.04 | 0.00 | ORB-long ORB[391.35,394.15] vol=3.4x ATR=0.92 |
| Stop hit — per-position SL triggered | 2025-11-18 11:00:00 | 394.68 | 393.19 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-11-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 09:40:00 | 391.70 | 392.17 | 0.00 | ORB-short ORB[392.05,394.05] vol=1.9x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 09:50:00 | 390.22 | 391.97 | 0.00 | T1 1.5R @ 390.22 |
| Stop hit — per-position SL triggered | 2025-11-19 10:15:00 | 391.70 | 391.78 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 11:15:00 | 385.95 | 388.40 | 0.00 | ORB-short ORB[389.30,392.00] vol=2.7x ATR=0.72 |
| Stop hit — per-position SL triggered | 2025-11-20 11:35:00 | 386.67 | 388.25 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-11-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 09:50:00 | 380.70 | 382.86 | 0.00 | ORB-short ORB[383.60,385.90] vol=1.7x ATR=0.94 |
| Stop hit — per-position SL triggered | 2025-11-21 09:55:00 | 381.64 | 382.50 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-11-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 09:30:00 | 388.00 | 385.09 | 0.00 | ORB-long ORB[380.20,385.45] vol=1.6x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-26 09:35:00 | 389.91 | 387.62 | 0.00 | T1 1.5R @ 389.91 |
| Target hit | 2025-11-26 11:45:00 | 388.85 | 389.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 59 — SELL (started 2025-11-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 11:10:00 | 390.20 | 391.53 | 0.00 | ORB-short ORB[391.30,394.50] vol=2.2x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 11:25:00 | 389.12 | 391.37 | 0.00 | T1 1.5R @ 389.12 |
| Target hit | 2025-11-27 12:15:00 | 388.75 | 388.38 | 0.00 | Trail-exit close>VWAP |

### Cycle 60 — BUY (started 2025-12-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 10:50:00 | 391.00 | 389.08 | 0.00 | ORB-long ORB[387.85,390.00] vol=4.9x ATR=0.97 |
| Stop hit — per-position SL triggered | 2025-12-01 11:20:00 | 390.03 | 389.52 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-12-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 09:50:00 | 384.80 | 386.62 | 0.00 | ORB-short ORB[386.20,389.90] vol=1.5x ATR=0.97 |
| Stop hit — per-position SL triggered | 2025-12-02 10:25:00 | 385.77 | 386.23 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-12-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 09:30:00 | 385.10 | 386.08 | 0.00 | ORB-short ORB[385.15,388.20] vol=2.0x ATR=0.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 10:05:00 | 383.78 | 385.26 | 0.00 | T1 1.5R @ 383.78 |
| Stop hit — per-position SL triggered | 2025-12-03 12:20:00 | 385.10 | 384.57 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-12-04 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-04 10:55:00 | 383.95 | 385.02 | 0.00 | ORB-short ORB[384.25,388.15] vol=3.5x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 12:50:00 | 382.72 | 384.14 | 0.00 | T1 1.5R @ 382.72 |
| Stop hit — per-position SL triggered | 2025-12-04 13:30:00 | 383.95 | 384.11 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-12-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 09:40:00 | 381.50 | 382.79 | 0.00 | ORB-short ORB[383.00,386.55] vol=7.0x ATR=1.18 |
| Stop hit — per-position SL triggered | 2025-12-05 10:00:00 | 382.68 | 382.73 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-19 10:15:00 | 440.45 | 2025-05-19 10:25:00 | 438.41 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-05-22 09:30:00 | 428.80 | 2025-05-22 09:35:00 | 426.10 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2025-05-22 09:30:00 | 428.80 | 2025-05-22 10:20:00 | 428.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-28 09:45:00 | 420.50 | 2025-05-28 09:50:00 | 423.05 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-05-28 09:45:00 | 420.50 | 2025-05-28 10:10:00 | 420.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-30 10:00:00 | 401.70 | 2025-05-30 10:15:00 | 399.68 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-05-30 10:00:00 | 401.70 | 2025-05-30 10:25:00 | 401.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-03 09:35:00 | 407.30 | 2025-06-03 09:50:00 | 405.65 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-06-03 09:35:00 | 407.30 | 2025-06-03 10:00:00 | 407.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-04 09:30:00 | 402.65 | 2025-06-04 09:45:00 | 400.71 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-06-04 09:30:00 | 402.65 | 2025-06-04 10:40:00 | 402.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-09 11:05:00 | 406.80 | 2025-06-09 11:20:00 | 408.35 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-06-09 11:05:00 | 406.80 | 2025-06-09 11:55:00 | 406.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-17 10:30:00 | 385.65 | 2025-06-17 10:55:00 | 384.07 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-06-17 10:30:00 | 385.65 | 2025-06-17 15:20:00 | 379.00 | TARGET_HIT | 0.50 | 1.72% |
| SELL | retest1 | 2025-06-18 10:55:00 | 375.90 | 2025-06-18 11:20:00 | 374.22 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-06-18 10:55:00 | 375.90 | 2025-06-18 15:20:00 | 371.40 | TARGET_HIT | 0.50 | 1.20% |
| BUY | retest1 | 2025-06-20 09:45:00 | 376.50 | 2025-06-20 11:20:00 | 379.21 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2025-06-20 09:45:00 | 376.50 | 2025-06-20 11:50:00 | 376.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-30 09:30:00 | 384.95 | 2025-06-30 09:55:00 | 383.50 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-06-30 09:30:00 | 384.95 | 2025-06-30 10:55:00 | 384.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-01 10:10:00 | 381.80 | 2025-07-01 10:30:00 | 383.09 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-07-08 11:05:00 | 375.00 | 2025-07-08 12:35:00 | 375.58 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-07-11 10:45:00 | 373.50 | 2025-07-11 11:05:00 | 372.36 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-07-11 10:45:00 | 373.50 | 2025-07-11 11:10:00 | 373.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-17 09:30:00 | 389.55 | 2025-07-17 10:40:00 | 387.72 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-07-17 09:30:00 | 389.55 | 2025-07-17 15:20:00 | 387.00 | TARGET_HIT | 0.50 | 0.65% |
| SELL | retest1 | 2025-07-18 10:20:00 | 382.15 | 2025-07-18 12:55:00 | 382.97 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-21 09:30:00 | 378.50 | 2025-07-21 09:50:00 | 379.52 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-07-22 09:55:00 | 380.05 | 2025-07-22 10:40:00 | 380.83 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-07-23 11:10:00 | 380.75 | 2025-07-23 12:50:00 | 379.82 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-07-23 11:10:00 | 380.75 | 2025-07-23 13:40:00 | 380.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-25 09:40:00 | 379.00 | 2025-07-25 10:15:00 | 377.75 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-07-25 09:40:00 | 379.00 | 2025-07-25 15:20:00 | 373.35 | TARGET_HIT | 0.50 | 1.49% |
| SELL | retest1 | 2025-07-29 10:15:00 | 370.00 | 2025-07-29 10:55:00 | 368.12 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-07-29 10:15:00 | 370.00 | 2025-07-29 11:20:00 | 370.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-04 10:50:00 | 380.25 | 2025-08-04 11:05:00 | 381.25 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-08-06 11:05:00 | 379.65 | 2025-08-06 12:20:00 | 380.74 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-08-14 10:45:00 | 392.50 | 2025-08-14 12:10:00 | 391.23 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-08-14 10:45:00 | 392.50 | 2025-08-14 15:20:00 | 386.25 | TARGET_HIT | 0.50 | 1.59% |
| SELL | retest1 | 2025-08-20 11:10:00 | 389.70 | 2025-08-20 11:35:00 | 388.74 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-08-20 11:10:00 | 389.70 | 2025-08-20 15:20:00 | 386.95 | TARGET_HIT | 0.50 | 0.71% |
| SELL | retest1 | 2025-08-22 09:30:00 | 380.50 | 2025-08-22 09:35:00 | 378.97 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-08-22 09:30:00 | 380.50 | 2025-08-22 09:40:00 | 380.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-03 09:30:00 | 370.60 | 2025-09-03 09:50:00 | 371.66 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-09-11 10:30:00 | 370.00 | 2025-09-11 10:35:00 | 369.21 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-09-15 09:45:00 | 373.40 | 2025-09-15 11:00:00 | 372.29 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-09-16 11:15:00 | 371.15 | 2025-09-16 11:30:00 | 371.69 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-09-17 10:35:00 | 368.95 | 2025-09-17 10:40:00 | 369.63 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-09-18 10:10:00 | 365.00 | 2025-09-18 10:55:00 | 365.51 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-09-19 11:10:00 | 363.50 | 2025-09-19 14:30:00 | 362.80 | PARTIAL | 0.50 | 0.19% |
| SELL | retest1 | 2025-09-19 11:10:00 | 363.50 | 2025-09-19 15:00:00 | 363.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-23 10:05:00 | 368.00 | 2025-09-23 10:10:00 | 366.95 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-09-29 11:10:00 | 364.30 | 2025-09-29 11:20:00 | 365.64 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-09-29 11:10:00 | 364.30 | 2025-09-29 11:35:00 | 364.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-30 09:45:00 | 366.95 | 2025-09-30 10:30:00 | 366.05 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-10-03 10:10:00 | 364.80 | 2025-10-03 11:00:00 | 365.62 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-10-09 09:40:00 | 370.35 | 2025-10-09 09:45:00 | 371.44 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-10-10 09:40:00 | 381.40 | 2025-10-10 09:55:00 | 380.13 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-10-13 10:55:00 | 383.10 | 2025-10-13 11:25:00 | 382.02 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-10-15 09:35:00 | 380.40 | 2025-10-15 09:55:00 | 379.20 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-10-16 10:50:00 | 379.70 | 2025-10-16 10:55:00 | 378.46 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-10-16 10:50:00 | 379.70 | 2025-10-16 11:05:00 | 379.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-17 10:15:00 | 387.25 | 2025-10-17 10:30:00 | 389.36 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-10-17 10:15:00 | 387.25 | 2025-10-17 11:20:00 | 388.70 | TARGET_HIT | 0.50 | 0.37% |
| BUY | retest1 | 2025-10-20 11:15:00 | 385.00 | 2025-10-20 11:20:00 | 384.15 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-10-23 10:20:00 | 389.10 | 2025-10-23 10:30:00 | 388.18 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-10-27 09:50:00 | 383.35 | 2025-10-27 10:50:00 | 381.79 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-10-27 09:50:00 | 383.35 | 2025-10-27 11:00:00 | 383.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-29 11:05:00 | 385.00 | 2025-10-29 11:25:00 | 385.91 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2025-10-29 11:05:00 | 385.00 | 2025-10-29 11:30:00 | 385.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-30 11:15:00 | 383.45 | 2025-10-30 12:25:00 | 382.70 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2025-10-30 11:15:00 | 383.45 | 2025-10-30 15:00:00 | 383.40 | TARGET_HIT | 0.50 | 0.01% |
| SELL | retest1 | 2025-10-31 09:30:00 | 381.50 | 2025-10-31 09:35:00 | 382.14 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-11-04 11:05:00 | 380.65 | 2025-11-04 11:40:00 | 379.91 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-11-06 10:10:00 | 374.55 | 2025-11-06 11:15:00 | 375.38 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-11-07 09:30:00 | 368.20 | 2025-11-07 10:30:00 | 369.37 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-11-14 09:30:00 | 395.85 | 2025-11-14 09:40:00 | 398.09 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-11-14 09:30:00 | 395.85 | 2025-11-14 09:50:00 | 395.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-18 10:50:00 | 395.60 | 2025-11-18 11:00:00 | 394.68 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-11-19 09:40:00 | 391.70 | 2025-11-19 09:50:00 | 390.22 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-11-19 09:40:00 | 391.70 | 2025-11-19 10:15:00 | 391.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-20 11:15:00 | 385.95 | 2025-11-20 11:35:00 | 386.67 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-11-21 09:50:00 | 380.70 | 2025-11-21 09:55:00 | 381.64 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-11-26 09:30:00 | 388.00 | 2025-11-26 09:35:00 | 389.91 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-11-26 09:30:00 | 388.00 | 2025-11-26 11:45:00 | 388.85 | TARGET_HIT | 0.50 | 0.22% |
| SELL | retest1 | 2025-11-27 11:10:00 | 390.20 | 2025-11-27 11:25:00 | 389.12 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2025-11-27 11:10:00 | 390.20 | 2025-11-27 12:15:00 | 388.75 | TARGET_HIT | 0.50 | 0.37% |
| BUY | retest1 | 2025-12-01 10:50:00 | 391.00 | 2025-12-01 11:20:00 | 390.03 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-12-02 09:50:00 | 384.80 | 2025-12-02 10:25:00 | 385.77 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-12-03 09:30:00 | 385.10 | 2025-12-03 10:05:00 | 383.78 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-12-03 09:30:00 | 385.10 | 2025-12-03 12:20:00 | 385.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-04 10:55:00 | 383.95 | 2025-12-04 12:50:00 | 382.72 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-12-04 10:55:00 | 383.95 | 2025-12-04 13:30:00 | 383.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-05 09:40:00 | 381.50 | 2025-12-05 10:00:00 | 382.68 | STOP_HIT | 1.00 | -0.31% |
