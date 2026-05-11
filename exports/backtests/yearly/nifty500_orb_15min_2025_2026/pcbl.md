# PCBL Chemical Ltd. (PCBL)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-05 15:25:00 (18238 bars)
- **Last close:** 306.00
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
| ENTRY1 | 75 |
| ENTRY2 | 0 |
| PARTIAL | 29 |
| TARGET_HIT | 16 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 104 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 45 / 59
- **Target hits / Stop hits / Partials:** 16 / 59 / 29
- **Avg / median % per leg:** 0.24% / 0.00%
- **Sum % (uncompounded):** 25.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 46 | 16 | 34.8% | 6 | 30 | 10 | 0.22% | 10.1% |
| BUY @ 2nd Alert (retest1) | 46 | 16 | 34.8% | 6 | 30 | 10 | 0.22% | 10.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 58 | 29 | 50.0% | 10 | 29 | 19 | 0.26% | 15.0% |
| SELL @ 2nd Alert (retest1) | 58 | 29 | 50.0% | 10 | 29 | 19 | 0.26% | 15.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 104 | 45 | 43.3% | 16 | 59 | 29 | 0.24% | 25.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 09:40:00 | 376.75 | 374.72 | 0.00 | ORB-long ORB[371.60,376.00] vol=1.5x ATR=1.42 |
| Stop hit — per-position SL triggered | 2025-05-13 09:50:00 | 375.33 | 374.82 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-16 10:35:00 | 395.00 | 392.39 | 0.00 | ORB-long ORB[389.00,393.90] vol=1.9x ATR=1.77 |
| Stop hit — per-position SL triggered | 2025-05-16 11:05:00 | 393.23 | 392.80 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-19 09:30:00 | 390.55 | 392.93 | 0.00 | ORB-short ORB[390.90,396.45] vol=1.9x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-19 09:35:00 | 388.49 | 392.18 | 0.00 | T1 1.5R @ 388.49 |
| Stop hit — per-position SL triggered | 2025-05-19 09:45:00 | 390.55 | 391.60 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-28 10:55:00 | 405.40 | 405.78 | 0.00 | ORB-short ORB[406.00,408.90] vol=1.9x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 11:45:00 | 403.86 | 405.56 | 0.00 | T1 1.5R @ 403.86 |
| Stop hit — per-position SL triggered | 2025-05-28 13:05:00 | 405.40 | 405.46 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 10:05:00 | 402.15 | 403.84 | 0.00 | ORB-short ORB[403.70,407.50] vol=2.1x ATR=1.16 |
| Stop hit — per-position SL triggered | 2025-05-29 10:20:00 | 403.31 | 403.73 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-05-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 10:25:00 | 401.95 | 403.49 | 0.00 | ORB-short ORB[403.55,407.00] vol=3.1x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-30 10:40:00 | 400.74 | 403.09 | 0.00 | T1 1.5R @ 400.74 |
| Stop hit — per-position SL triggered | 2025-05-30 11:30:00 | 401.95 | 402.00 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 09:30:00 | 425.65 | 421.68 | 0.00 | ORB-long ORB[418.20,422.20] vol=5.6x ATR=1.90 |
| Stop hit — per-position SL triggered | 2025-06-05 09:35:00 | 423.75 | 422.98 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-06-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:30:00 | 395.25 | 400.44 | 0.00 | ORB-short ORB[399.20,405.00] vol=2.5x ATR=2.09 |
| Stop hit — per-position SL triggered | 2025-06-16 09:50:00 | 397.34 | 397.94 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-19 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 10:00:00 | 385.65 | 388.84 | 0.00 | ORB-short ORB[387.00,391.35] vol=1.8x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 10:15:00 | 383.34 | 387.17 | 0.00 | T1 1.5R @ 383.34 |
| Stop hit — per-position SL triggered | 2025-06-19 11:05:00 | 385.65 | 385.48 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-06-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-23 11:00:00 | 385.45 | 387.54 | 0.00 | ORB-short ORB[386.00,390.00] vol=4.2x ATR=1.27 |
| Stop hit — per-position SL triggered | 2025-06-23 11:25:00 | 386.72 | 387.10 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 09:35:00 | 397.25 | 395.73 | 0.00 | ORB-long ORB[392.70,397.00] vol=1.9x ATR=1.84 |
| Stop hit — per-position SL triggered | 2025-06-24 10:10:00 | 395.41 | 395.83 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 10:05:00 | 397.70 | 396.28 | 0.00 | ORB-long ORB[393.30,397.45] vol=4.0x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-25 10:20:00 | 399.67 | 396.83 | 0.00 | T1 1.5R @ 399.67 |
| Target hit | 2025-06-25 15:20:00 | 415.75 | 407.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2025-06-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 10:50:00 | 422.55 | 419.72 | 0.00 | ORB-long ORB[416.65,420.00] vol=6.4x ATR=1.60 |
| Stop hit — per-position SL triggered | 2025-06-27 10:55:00 | 420.95 | 419.79 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-06-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-30 11:05:00 | 415.00 | 418.22 | 0.00 | ORB-short ORB[417.20,422.00] vol=2.5x ATR=1.33 |
| Stop hit — per-position SL triggered | 2025-06-30 11:40:00 | 416.33 | 417.69 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:40:00 | 413.05 | 415.57 | 0.00 | ORB-short ORB[414.00,417.75] vol=1.8x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 10:15:00 | 410.97 | 414.23 | 0.00 | T1 1.5R @ 410.97 |
| Target hit | 2025-07-02 12:00:00 | 412.35 | 411.74 | 0.00 | Trail-exit close>VWAP |

### Cycle 16 — SELL (started 2025-07-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-07 11:10:00 | 405.65 | 407.89 | 0.00 | ORB-short ORB[407.15,412.60] vol=3.3x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 11:20:00 | 403.88 | 407.63 | 0.00 | T1 1.5R @ 403.88 |
| Target hit | 2025-07-07 15:20:00 | 401.85 | 405.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — SELL (started 2025-07-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 11:10:00 | 399.70 | 403.08 | 0.00 | ORB-short ORB[400.90,405.85] vol=2.9x ATR=1.20 |
| Stop hit — per-position SL triggered | 2025-07-08 14:15:00 | 400.90 | 401.73 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 09:30:00 | 413.00 | 408.99 | 0.00 | ORB-long ORB[402.40,407.10] vol=5.1x ATR=1.61 |
| Stop hit — per-position SL triggered | 2025-07-09 09:35:00 | 411.39 | 413.13 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 10:45:00 | 424.80 | 421.85 | 0.00 | ORB-long ORB[420.70,424.50] vol=1.8x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-16 11:30:00 | 427.03 | 423.26 | 0.00 | T1 1.5R @ 427.03 |
| Target hit | 2025-07-16 15:20:00 | 427.75 | 424.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2025-07-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 09:50:00 | 424.85 | 427.46 | 0.00 | ORB-short ORB[425.55,429.95] vol=3.3x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 11:55:00 | 422.51 | 426.21 | 0.00 | T1 1.5R @ 422.51 |
| Target hit | 2025-07-17 15:20:00 | 422.80 | 424.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — BUY (started 2025-07-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 09:40:00 | 425.95 | 421.28 | 0.00 | ORB-long ORB[416.05,421.95] vol=3.6x ATR=2.28 |
| Stop hit — per-position SL triggered | 2025-07-21 09:50:00 | 423.67 | 421.83 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 10:15:00 | 394.45 | 391.73 | 0.00 | ORB-long ORB[389.25,393.10] vol=2.1x ATR=1.43 |
| Stop hit — per-position SL triggered | 2025-07-30 10:20:00 | 393.02 | 391.94 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-08-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 11:00:00 | 380.70 | 384.79 | 0.00 | ORB-short ORB[383.10,387.75] vol=2.7x ATR=1.32 |
| Stop hit — per-position SL triggered | 2025-08-22 13:00:00 | 382.02 | 382.94 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:30:00 | 375.30 | 376.87 | 0.00 | ORB-short ORB[376.00,380.00] vol=2.5x ATR=1.06 |
| Stop hit — per-position SL triggered | 2025-08-26 09:45:00 | 376.36 | 376.46 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-08-28 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-28 10:30:00 | 379.60 | 376.74 | 0.00 | ORB-long ORB[373.75,378.25] vol=1.7x ATR=1.47 |
| Stop hit — per-position SL triggered | 2025-08-28 11:00:00 | 378.13 | 377.13 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-09-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 09:35:00 | 393.60 | 391.38 | 0.00 | ORB-long ORB[388.65,392.60] vol=1.6x ATR=1.39 |
| Stop hit — per-position SL triggered | 2025-09-11 09:40:00 | 392.21 | 391.48 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-09-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 11:00:00 | 390.90 | 387.83 | 0.00 | ORB-long ORB[385.35,389.90] vol=2.0x ATR=1.01 |
| Stop hit — per-position SL triggered | 2025-09-15 11:10:00 | 389.89 | 387.90 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-09-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-16 11:00:00 | 390.00 | 393.04 | 0.00 | ORB-short ORB[391.40,394.65] vol=3.0x ATR=1.07 |
| Stop hit — per-position SL triggered | 2025-09-16 11:50:00 | 391.07 | 392.60 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-09-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 10:00:00 | 397.60 | 400.39 | 0.00 | ORB-short ORB[399.00,403.65] vol=2.5x ATR=1.37 |
| Stop hit — per-position SL triggered | 2025-09-23 10:25:00 | 398.97 | 399.98 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-10-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 10:50:00 | 376.60 | 375.68 | 0.00 | ORB-long ORB[373.15,376.50] vol=2.6x ATR=0.96 |
| Stop hit — per-position SL triggered | 2025-10-01 11:15:00 | 375.64 | 375.64 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-10-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 10:35:00 | 381.10 | 382.77 | 0.00 | ORB-short ORB[382.20,384.60] vol=1.6x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 10:55:00 | 379.96 | 382.37 | 0.00 | T1 1.5R @ 379.96 |
| Target hit | 2025-10-06 15:20:00 | 377.65 | 379.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — BUY (started 2025-10-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 10:55:00 | 383.65 | 381.13 | 0.00 | ORB-long ORB[377.10,380.75] vol=8.7x ATR=1.12 |
| Stop hit — per-position SL triggered | 2025-10-08 11:00:00 | 382.53 | 381.31 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-10-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 09:45:00 | 381.80 | 383.27 | 0.00 | ORB-short ORB[382.65,385.00] vol=1.8x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 09:55:00 | 380.47 | 382.48 | 0.00 | T1 1.5R @ 380.47 |
| Target hit | 2025-10-14 14:15:00 | 380.60 | 379.80 | 0.00 | Trail-exit close>VWAP |

### Cycle 34 — SELL (started 2025-10-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 10:20:00 | 367.45 | 368.66 | 0.00 | ORB-short ORB[368.05,371.90] vol=2.7x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 11:30:00 | 366.07 | 368.01 | 0.00 | T1 1.5R @ 366.07 |
| Stop hit — per-position SL triggered | 2025-10-24 15:00:00 | 367.45 | 367.18 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-10-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 10:25:00 | 363.30 | 364.04 | 0.00 | ORB-short ORB[363.50,365.60] vol=3.2x ATR=0.68 |
| Stop hit — per-position SL triggered | 2025-10-28 10:30:00 | 363.98 | 364.02 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:15:00 | 368.30 | 366.40 | 0.00 | ORB-long ORB[364.50,366.90] vol=4.7x ATR=0.96 |
| Stop hit — per-position SL triggered | 2025-10-29 10:20:00 | 367.34 | 366.44 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-10-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-30 09:35:00 | 373.15 | 371.97 | 0.00 | ORB-long ORB[370.40,373.00] vol=1.7x ATR=1.04 |
| Stop hit — per-position SL triggered | 2025-10-30 09:40:00 | 372.11 | 371.97 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-10-31 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 10:40:00 | 367.20 | 368.70 | 0.00 | ORB-short ORB[368.75,370.75] vol=3.0x ATR=0.68 |
| Stop hit — per-position SL triggered | 2025-10-31 10:45:00 | 367.88 | 368.67 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-11-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-03 11:10:00 | 359.95 | 361.26 | 0.00 | ORB-short ORB[361.95,364.80] vol=9.2x ATR=0.87 |
| Stop hit — per-position SL triggered | 2025-11-03 11:15:00 | 360.82 | 361.89 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-11-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 10:05:00 | 349.25 | 352.57 | 0.00 | ORB-short ORB[353.10,356.10] vol=2.5x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 10:20:00 | 347.83 | 350.98 | 0.00 | T1 1.5R @ 347.83 |
| Stop hit — per-position SL triggered | 2025-11-06 10:25:00 | 349.25 | 350.83 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-11-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 11:05:00 | 353.40 | 351.10 | 0.00 | ORB-long ORB[350.40,353.25] vol=1.8x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 11:35:00 | 354.82 | 351.47 | 0.00 | T1 1.5R @ 354.82 |
| Stop hit — per-position SL triggered | 2025-11-07 11:45:00 | 353.40 | 351.58 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-11-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 10:45:00 | 350.30 | 348.26 | 0.00 | ORB-long ORB[346.30,349.50] vol=2.3x ATR=1.01 |
| Stop hit — per-position SL triggered | 2025-11-10 11:05:00 | 349.29 | 348.36 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-11-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-11 10:05:00 | 352.55 | 349.58 | 0.00 | ORB-long ORB[346.80,349.85] vol=2.6x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 10:20:00 | 354.16 | 350.57 | 0.00 | T1 1.5R @ 354.16 |
| Stop hit — per-position SL triggered | 2025-11-11 11:35:00 | 352.55 | 352.41 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-11-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 10:00:00 | 352.55 | 352.00 | 0.00 | ORB-long ORB[349.40,352.20] vol=2.5x ATR=0.78 |
| Stop hit — per-position SL triggered | 2025-11-12 10:05:00 | 351.77 | 351.90 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-11-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 10:30:00 | 351.30 | 350.13 | 0.00 | ORB-long ORB[348.80,351.00] vol=2.2x ATR=0.81 |
| Stop hit — per-position SL triggered | 2025-11-13 10:55:00 | 350.49 | 350.18 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-11-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 10:55:00 | 350.50 | 347.87 | 0.00 | ORB-long ORB[345.20,347.40] vol=3.4x ATR=0.84 |
| Stop hit — per-position SL triggered | 2025-11-14 11:00:00 | 349.66 | 347.91 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-11-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-18 10:50:00 | 343.35 | 343.96 | 0.00 | ORB-short ORB[344.90,348.55] vol=1.8x ATR=0.69 |
| Stop hit — per-position SL triggered | 2025-11-18 12:00:00 | 344.04 | 343.73 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-11-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 09:35:00 | 333.90 | 335.16 | 0.00 | ORB-short ORB[334.20,338.35] vol=2.1x ATR=1.01 |
| Stop hit — per-position SL triggered | 2025-11-20 09:40:00 | 334.91 | 335.73 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-11-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 09:30:00 | 328.20 | 330.07 | 0.00 | ORB-short ORB[329.45,332.80] vol=1.8x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 09:40:00 | 326.68 | 329.31 | 0.00 | T1 1.5R @ 326.68 |
| Stop hit — per-position SL triggered | 2025-11-24 11:35:00 | 328.20 | 326.91 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-11-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 11:10:00 | 329.30 | 329.89 | 0.00 | ORB-short ORB[329.50,331.50] vol=1.6x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 11:20:00 | 328.13 | 329.80 | 0.00 | T1 1.5R @ 328.13 |
| Stop hit — per-position SL triggered | 2025-11-27 12:55:00 | 329.30 | 329.41 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-12-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 10:35:00 | 323.20 | 325.70 | 0.00 | ORB-short ORB[324.70,327.90] vol=2.8x ATR=0.82 |
| Stop hit — per-position SL triggered | 2025-12-01 10:55:00 | 324.02 | 325.41 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-12-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 09:30:00 | 320.20 | 321.64 | 0.00 | ORB-short ORB[321.00,323.75] vol=1.6x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-12-02 10:00:00 | 321.06 | 320.96 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-12-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 11:05:00 | 315.00 | 316.51 | 0.00 | ORB-short ORB[316.00,319.95] vol=1.7x ATR=0.75 |
| Stop hit — per-position SL triggered | 2025-12-03 11:45:00 | 315.75 | 316.33 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-12-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 10:50:00 | 310.20 | 309.29 | 0.00 | ORB-long ORB[307.30,310.00] vol=2.4x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 14:10:00 | 311.49 | 310.08 | 0.00 | T1 1.5R @ 311.49 |
| Target hit | 2025-12-09 15:20:00 | 312.30 | 310.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — BUY (started 2025-12-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 09:40:00 | 311.35 | 310.46 | 0.00 | ORB-long ORB[309.00,310.80] vol=2.0x ATR=0.70 |
| Stop hit — per-position SL triggered | 2025-12-17 10:00:00 | 310.65 | 310.56 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-12-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 10:25:00 | 313.50 | 311.98 | 0.00 | ORB-long ORB[310.10,312.25] vol=1.7x ATR=0.81 |
| Stop hit — per-position SL triggered | 2025-12-19 10:35:00 | 312.69 | 312.04 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-12-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 10:30:00 | 307.10 | 306.25 | 0.00 | ORB-long ORB[304.10,306.60] vol=1.5x ATR=0.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 11:05:00 | 308.21 | 306.75 | 0.00 | T1 1.5R @ 308.21 |
| Target hit | 2025-12-26 14:15:00 | 307.45 | 307.47 | 0.00 | Trail-exit close<VWAP |

### Cycle 58 — BUY (started 2025-12-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-29 09:40:00 | 305.25 | 303.30 | 0.00 | ORB-long ORB[301.40,304.25] vol=1.5x ATR=1.03 |
| Stop hit — per-position SL triggered | 2025-12-29 09:45:00 | 304.22 | 303.49 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2026-01-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:00:00 | 301.30 | 298.00 | 0.00 | ORB-long ORB[296.00,298.40] vol=2.0x ATR=0.97 |
| Stop hit — per-position SL triggered | 2026-01-02 10:10:00 | 300.33 | 298.71 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2026-01-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-05 11:00:00 | 298.70 | 300.07 | 0.00 | ORB-short ORB[299.80,303.15] vol=1.9x ATR=0.72 |
| Stop hit — per-position SL triggered | 2026-01-05 12:10:00 | 299.42 | 299.77 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2026-01-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 09:35:00 | 293.90 | 295.22 | 0.00 | ORB-short ORB[294.50,298.65] vol=2.2x ATR=0.92 |
| Stop hit — per-position SL triggered | 2026-01-06 09:50:00 | 294.82 | 294.90 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2026-01-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:00:00 | 291.55 | 293.02 | 0.00 | ORB-short ORB[292.00,294.90] vol=1.7x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:15:00 | 290.78 | 292.92 | 0.00 | T1 1.5R @ 290.78 |
| Target hit | 2026-01-08 15:20:00 | 285.50 | 287.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — BUY (started 2026-01-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-09 09:55:00 | 285.75 | 283.45 | 0.00 | ORB-long ORB[281.00,284.60] vol=1.7x ATR=1.28 |
| Stop hit — per-position SL triggered | 2026-01-09 11:00:00 | 284.47 | 284.37 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-01-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-13 10:20:00 | 283.20 | 282.28 | 0.00 | ORB-long ORB[280.40,282.85] vol=4.2x ATR=1.09 |
| Stop hit — per-position SL triggered | 2026-01-13 10:40:00 | 282.11 | 282.51 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-01-14 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 10:20:00 | 281.70 | 280.68 | 0.00 | ORB-long ORB[279.50,281.60] vol=2.0x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 10:35:00 | 283.08 | 281.29 | 0.00 | T1 1.5R @ 283.08 |
| Stop hit — per-position SL triggered | 2026-01-14 14:40:00 | 281.70 | 283.02 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2026-01-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-16 11:10:00 | 279.20 | 279.72 | 0.00 | ORB-short ORB[279.55,282.75] vol=8.0x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 11:55:00 | 278.21 | 279.58 | 0.00 | T1 1.5R @ 278.21 |
| Target hit | 2026-01-16 15:20:00 | 274.35 | 277.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 67 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 10:15:00 | 273.20 | 276.12 | 0.00 | ORB-short ORB[276.30,280.05] vol=1.9x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 10:35:00 | 271.40 | 275.46 | 0.00 | T1 1.5R @ 271.40 |
| Target hit | 2026-01-23 15:20:00 | 263.65 | 269.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — SELL (started 2026-01-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 09:55:00 | 267.25 | 269.07 | 0.00 | ORB-short ORB[268.50,271.85] vol=2.7x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 10:20:00 | 265.52 | 268.57 | 0.00 | T1 1.5R @ 265.52 |
| Target hit | 2026-01-29 15:20:00 | 265.70 | 266.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — BUY (started 2026-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 09:30:00 | 264.20 | 262.81 | 0.00 | ORB-long ORB[261.50,264.00] vol=2.8x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 10:05:00 | 265.79 | 263.70 | 0.00 | T1 1.5R @ 265.79 |
| Target hit | 2026-01-30 15:20:00 | 266.15 | 266.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 70 — BUY (started 2026-02-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 11:00:00 | 267.60 | 265.51 | 0.00 | ORB-long ORB[263.00,264.55] vol=2.6x ATR=0.78 |
| Stop hit — per-position SL triggered | 2026-02-01 11:10:00 | 266.82 | 265.63 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2026-02-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:40:00 | 278.40 | 275.33 | 0.00 | ORB-long ORB[272.75,275.75] vol=1.9x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 10:45:00 | 279.86 | 276.46 | 0.00 | T1 1.5R @ 279.86 |
| Target hit | 2026-02-09 15:20:00 | 298.45 | 297.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 72 — SELL (started 2026-03-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:00:00 | 266.80 | 268.89 | 0.00 | ORB-short ORB[269.00,271.35] vol=1.8x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:10:00 | 265.25 | 268.38 | 0.00 | T1 1.5R @ 265.25 |
| Stop hit — per-position SL triggered | 2026-03-13 10:25:00 | 266.80 | 268.14 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2026-03-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 09:50:00 | 248.80 | 251.07 | 0.00 | ORB-short ORB[249.40,253.00] vol=1.9x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 10:05:00 | 246.51 | 250.01 | 0.00 | T1 1.5R @ 246.51 |
| Target hit | 2026-03-19 15:20:00 | 246.00 | 246.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 74 — SELL (started 2026-04-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 11:05:00 | 284.01 | 286.07 | 0.00 | ORB-short ORB[284.50,287.99] vol=1.6x ATR=1.11 |
| Stop hit — per-position SL triggered | 2026-04-17 13:45:00 | 285.12 | 285.38 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2026-04-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:55:00 | 296.99 | 293.23 | 0.00 | ORB-long ORB[290.65,294.95] vol=4.5x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:00:00 | 298.70 | 295.84 | 0.00 | T1 1.5R @ 298.70 |
| Stop hit — per-position SL triggered | 2026-04-28 11:05:00 | 296.99 | 296.12 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-13 09:40:00 | 376.75 | 2025-05-13 09:50:00 | 375.33 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-05-16 10:35:00 | 395.00 | 2025-05-16 11:05:00 | 393.23 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-05-19 09:30:00 | 390.55 | 2025-05-19 09:35:00 | 388.49 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-05-19 09:30:00 | 390.55 | 2025-05-19 09:45:00 | 390.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-28 10:55:00 | 405.40 | 2025-05-28 11:45:00 | 403.86 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-05-28 10:55:00 | 405.40 | 2025-05-28 13:05:00 | 405.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-29 10:05:00 | 402.15 | 2025-05-29 10:20:00 | 403.31 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-05-30 10:25:00 | 401.95 | 2025-05-30 10:40:00 | 400.74 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-05-30 10:25:00 | 401.95 | 2025-05-30 11:30:00 | 401.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-05 09:30:00 | 425.65 | 2025-06-05 09:35:00 | 423.75 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-06-16 09:30:00 | 395.25 | 2025-06-16 09:50:00 | 397.34 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2025-06-19 10:00:00 | 385.65 | 2025-06-19 10:15:00 | 383.34 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2025-06-19 10:00:00 | 385.65 | 2025-06-19 11:05:00 | 385.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-23 11:00:00 | 385.45 | 2025-06-23 11:25:00 | 386.72 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-06-24 09:35:00 | 397.25 | 2025-06-24 10:10:00 | 395.41 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-06-25 10:05:00 | 397.70 | 2025-06-25 10:20:00 | 399.67 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-06-25 10:05:00 | 397.70 | 2025-06-25 15:20:00 | 415.75 | TARGET_HIT | 0.50 | 4.54% |
| BUY | retest1 | 2025-06-27 10:50:00 | 422.55 | 2025-06-27 10:55:00 | 420.95 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-06-30 11:05:00 | 415.00 | 2025-06-30 11:40:00 | 416.33 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-07-02 09:40:00 | 413.05 | 2025-07-02 10:15:00 | 410.97 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-07-02 09:40:00 | 413.05 | 2025-07-02 12:00:00 | 412.35 | TARGET_HIT | 0.50 | 0.17% |
| SELL | retest1 | 2025-07-07 11:10:00 | 405.65 | 2025-07-07 11:20:00 | 403.88 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-07-07 11:10:00 | 405.65 | 2025-07-07 15:20:00 | 401.85 | TARGET_HIT | 0.50 | 0.94% |
| SELL | retest1 | 2025-07-08 11:10:00 | 399.70 | 2025-07-08 14:15:00 | 400.90 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-07-09 09:30:00 | 413.00 | 2025-07-09 09:35:00 | 411.39 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-07-16 10:45:00 | 424.80 | 2025-07-16 11:30:00 | 427.03 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-07-16 10:45:00 | 424.80 | 2025-07-16 15:20:00 | 427.75 | TARGET_HIT | 0.50 | 0.69% |
| SELL | retest1 | 2025-07-17 09:50:00 | 424.85 | 2025-07-17 11:55:00 | 422.51 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-07-17 09:50:00 | 424.85 | 2025-07-17 15:20:00 | 422.80 | TARGET_HIT | 0.50 | 0.48% |
| BUY | retest1 | 2025-07-21 09:40:00 | 425.95 | 2025-07-21 09:50:00 | 423.67 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2025-07-30 10:15:00 | 394.45 | 2025-07-30 10:20:00 | 393.02 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-08-22 11:00:00 | 380.70 | 2025-08-22 13:00:00 | 382.02 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-08-26 09:30:00 | 375.30 | 2025-08-26 09:45:00 | 376.36 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-08-28 10:30:00 | 379.60 | 2025-08-28 11:00:00 | 378.13 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-09-11 09:35:00 | 393.60 | 2025-09-11 09:40:00 | 392.21 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-09-15 11:00:00 | 390.90 | 2025-09-15 11:10:00 | 389.89 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-09-16 11:00:00 | 390.00 | 2025-09-16 11:50:00 | 391.07 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-09-23 10:00:00 | 397.60 | 2025-09-23 10:25:00 | 398.97 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-10-01 10:50:00 | 376.60 | 2025-10-01 11:15:00 | 375.64 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-10-06 10:35:00 | 381.10 | 2025-10-06 10:55:00 | 379.96 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-10-06 10:35:00 | 381.10 | 2025-10-06 15:20:00 | 377.65 | TARGET_HIT | 0.50 | 0.91% |
| BUY | retest1 | 2025-10-08 10:55:00 | 383.65 | 2025-10-08 11:00:00 | 382.53 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-10-14 09:45:00 | 381.80 | 2025-10-14 09:55:00 | 380.47 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-10-14 09:45:00 | 381.80 | 2025-10-14 14:15:00 | 380.60 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2025-10-24 10:20:00 | 367.45 | 2025-10-24 11:30:00 | 366.07 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-10-24 10:20:00 | 367.45 | 2025-10-24 15:00:00 | 367.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-28 10:25:00 | 363.30 | 2025-10-28 10:30:00 | 363.98 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-10-29 10:15:00 | 368.30 | 2025-10-29 10:20:00 | 367.34 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-10-30 09:35:00 | 373.15 | 2025-10-30 09:40:00 | 372.11 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-10-31 10:40:00 | 367.20 | 2025-10-31 10:45:00 | 367.88 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-11-03 11:10:00 | 359.95 | 2025-11-03 11:15:00 | 360.82 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-11-06 10:05:00 | 349.25 | 2025-11-06 10:20:00 | 347.83 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-11-06 10:05:00 | 349.25 | 2025-11-06 10:25:00 | 349.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-07 11:05:00 | 353.40 | 2025-11-07 11:35:00 | 354.82 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-11-07 11:05:00 | 353.40 | 2025-11-07 11:45:00 | 353.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-10 10:45:00 | 350.30 | 2025-11-10 11:05:00 | 349.29 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-11-11 10:05:00 | 352.55 | 2025-11-11 10:20:00 | 354.16 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-11-11 10:05:00 | 352.55 | 2025-11-11 11:35:00 | 352.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-12 10:00:00 | 352.55 | 2025-11-12 10:05:00 | 351.77 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-11-13 10:30:00 | 351.30 | 2025-11-13 10:55:00 | 350.49 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-11-14 10:55:00 | 350.50 | 2025-11-14 11:00:00 | 349.66 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-11-18 10:50:00 | 343.35 | 2025-11-18 12:00:00 | 344.04 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-11-20 09:35:00 | 333.90 | 2025-11-20 09:40:00 | 334.91 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-11-24 09:30:00 | 328.20 | 2025-11-24 09:40:00 | 326.68 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-11-24 09:30:00 | 328.20 | 2025-11-24 11:35:00 | 328.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-27 11:10:00 | 329.30 | 2025-11-27 11:20:00 | 328.13 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-11-27 11:10:00 | 329.30 | 2025-11-27 12:55:00 | 329.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-01 10:35:00 | 323.20 | 2025-12-01 10:55:00 | 324.02 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-12-02 09:30:00 | 320.20 | 2025-12-02 10:00:00 | 321.06 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-12-03 11:05:00 | 315.00 | 2025-12-03 11:45:00 | 315.75 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-12-09 10:50:00 | 310.20 | 2025-12-09 14:10:00 | 311.49 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-12-09 10:50:00 | 310.20 | 2025-12-09 15:20:00 | 312.30 | TARGET_HIT | 0.50 | 0.68% |
| BUY | retest1 | 2025-12-17 09:40:00 | 311.35 | 2025-12-17 10:00:00 | 310.65 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-12-19 10:25:00 | 313.50 | 2025-12-19 10:35:00 | 312.69 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-12-26 10:30:00 | 307.10 | 2025-12-26 11:05:00 | 308.21 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-12-26 10:30:00 | 307.10 | 2025-12-26 14:15:00 | 307.45 | TARGET_HIT | 0.50 | 0.11% |
| BUY | retest1 | 2025-12-29 09:40:00 | 305.25 | 2025-12-29 09:45:00 | 304.22 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-01-02 10:00:00 | 301.30 | 2026-01-02 10:10:00 | 300.33 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-01-05 11:00:00 | 298.70 | 2026-01-05 12:10:00 | 299.42 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-01-06 09:35:00 | 293.90 | 2026-01-06 09:50:00 | 294.82 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-01-08 11:00:00 | 291.55 | 2026-01-08 11:15:00 | 290.78 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2026-01-08 11:00:00 | 291.55 | 2026-01-08 15:20:00 | 285.50 | TARGET_HIT | 0.50 | 2.08% |
| BUY | retest1 | 2026-01-09 09:55:00 | 285.75 | 2026-01-09 11:00:00 | 284.47 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-01-13 10:20:00 | 283.20 | 2026-01-13 10:40:00 | 282.11 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-01-14 10:20:00 | 281.70 | 2026-01-14 10:35:00 | 283.08 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-01-14 10:20:00 | 281.70 | 2026-01-14 14:40:00 | 281.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-16 11:10:00 | 279.20 | 2026-01-16 11:55:00 | 278.21 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-01-16 11:10:00 | 279.20 | 2026-01-16 15:20:00 | 274.35 | TARGET_HIT | 0.50 | 1.74% |
| SELL | retest1 | 2026-01-23 10:15:00 | 273.20 | 2026-01-23 10:35:00 | 271.40 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2026-01-23 10:15:00 | 273.20 | 2026-01-23 15:20:00 | 263.65 | TARGET_HIT | 0.50 | 3.50% |
| SELL | retest1 | 2026-01-29 09:55:00 | 267.25 | 2026-01-29 10:20:00 | 265.52 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2026-01-29 09:55:00 | 267.25 | 2026-01-29 15:20:00 | 265.70 | TARGET_HIT | 0.50 | 0.58% |
| BUY | retest1 | 2026-01-30 09:30:00 | 264.20 | 2026-01-30 10:05:00 | 265.79 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-01-30 09:30:00 | 264.20 | 2026-01-30 15:20:00 | 266.15 | TARGET_HIT | 0.50 | 0.74% |
| BUY | retest1 | 2026-02-01 11:00:00 | 267.60 | 2026-02-01 11:10:00 | 266.82 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-09 10:40:00 | 278.40 | 2026-02-09 10:45:00 | 279.86 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-02-09 10:40:00 | 278.40 | 2026-02-09 15:20:00 | 298.45 | TARGET_HIT | 0.50 | 7.20% |
| SELL | retest1 | 2026-03-13 10:00:00 | 266.80 | 2026-03-13 10:10:00 | 265.25 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-03-13 10:00:00 | 266.80 | 2026-03-13 10:25:00 | 266.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-19 09:50:00 | 248.80 | 2026-03-19 10:05:00 | 246.51 | PARTIAL | 0.50 | 0.92% |
| SELL | retest1 | 2026-03-19 09:50:00 | 248.80 | 2026-03-19 15:20:00 | 246.00 | TARGET_HIT | 0.50 | 1.13% |
| SELL | retest1 | 2026-04-17 11:05:00 | 284.01 | 2026-04-17 13:45:00 | 285.12 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-28 10:55:00 | 296.99 | 2026-04-28 11:00:00 | 298.70 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-04-28 10:55:00 | 296.99 | 2026-04-28 11:05:00 | 296.99 | STOP_HIT | 0.50 | 0.00% |
