# KOTAKBANK (KOTAKBANK)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (55354 bars)
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
| ENTRY1 | 104 |
| ENTRY2 | 0 |
| PARTIAL | 41 |
| TARGET_HIT | 13 |
| STOP_HIT | 91 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 145 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 54 / 91
- **Target hits / Stop hits / Partials:** 13 / 91 / 41
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 11.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 60 | 17 | 28.3% | 3 | 43 | 14 | -0.02% | -1.2% |
| BUY @ 2nd Alert (retest1) | 60 | 17 | 28.3% | 3 | 43 | 14 | -0.02% | -1.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 85 | 37 | 43.5% | 10 | 48 | 27 | 0.15% | 12.7% |
| SELL @ 2nd Alert (retest1) | 85 | 37 | 43.5% | 10 | 48 | 27 | 0.15% | 12.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 145 | 54 | 37.2% | 13 | 91 | 41 | 0.08% | 11.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-16 11:10:00 | 392.48 | 393.79 | 0.00 | ORB-short ORB[392.96,395.60] vol=1.6x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-16 11:20:00 | 391.23 | 393.67 | 0.00 | T1 1.5R @ 391.23 |
| Target hit | 2023-05-16 15:20:00 | 388.64 | 391.97 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2023-05-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-17 10:30:00 | 384.01 | 387.23 | 0.00 | ORB-short ORB[388.00,390.18] vol=1.6x ATR=1.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-17 11:30:00 | 382.38 | 385.96 | 0.00 | T1 1.5R @ 382.38 |
| Target hit | 2023-05-17 15:20:00 | 381.98 | 382.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2023-05-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-18 11:05:00 | 385.88 | 384.53 | 0.00 | ORB-long ORB[382.24,385.55] vol=1.9x ATR=0.65 |
| Stop hit — per-position SL triggered | 2023-05-18 11:40:00 | 385.23 | 384.76 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-05-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-23 10:50:00 | 384.46 | 385.62 | 0.00 | ORB-short ORB[385.40,388.04] vol=1.6x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-23 11:25:00 | 383.53 | 385.17 | 0.00 | T1 1.5R @ 383.53 |
| Stop hit — per-position SL triggered | 2023-05-23 12:45:00 | 384.46 | 384.79 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-24 11:00:00 | 387.22 | 386.92 | 0.00 | ORB-long ORB[383.80,387.07] vol=1.6x ATR=0.62 |
| Stop hit — per-position SL triggered | 2023-05-24 11:15:00 | 386.60 | 386.92 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-06-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-06 09:30:00 | 389.03 | 387.81 | 0.00 | ORB-long ORB[386.03,388.60] vol=1.6x ATR=0.77 |
| Stop hit — per-position SL triggered | 2023-06-06 10:30:00 | 388.26 | 388.62 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-06-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-13 09:30:00 | 374.17 | 374.60 | 0.00 | ORB-short ORB[374.26,376.60] vol=5.2x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-13 10:30:00 | 373.20 | 374.08 | 0.00 | T1 1.5R @ 373.20 |
| Target hit | 2023-06-13 15:20:00 | 370.55 | 372.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2023-06-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-15 10:55:00 | 370.63 | 371.41 | 0.00 | ORB-short ORB[371.72,373.80] vol=12.6x ATR=0.55 |
| Stop hit — per-position SL triggered | 2023-06-15 11:15:00 | 371.18 | 371.38 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-06-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-19 09:45:00 | 370.20 | 371.32 | 0.00 | ORB-short ORB[370.88,373.69] vol=3.2x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-19 10:05:00 | 368.80 | 370.93 | 0.00 | T1 1.5R @ 368.80 |
| Target hit | 2023-06-19 15:20:00 | 364.97 | 365.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2023-06-28 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-28 10:25:00 | 366.94 | 367.30 | 0.00 | ORB-short ORB[367.40,369.50] vol=2.6x ATR=0.57 |
| Stop hit — per-position SL triggered | 2023-06-28 10:45:00 | 367.51 | 367.20 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-06-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-30 10:40:00 | 367.00 | 368.37 | 0.00 | ORB-short ORB[368.13,369.58] vol=2.1x ATR=0.67 |
| Stop hit — per-position SL triggered | 2023-06-30 10:50:00 | 367.67 | 368.17 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-07-03 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-03 10:50:00 | 368.21 | 369.99 | 0.00 | ORB-short ORB[368.29,371.20] vol=1.8x ATR=0.79 |
| Stop hit — per-position SL triggered | 2023-07-03 11:05:00 | 369.00 | 369.91 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-07-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 09:55:00 | 374.73 | 373.97 | 0.00 | ORB-long ORB[371.81,374.68] vol=2.6x ATR=0.67 |
| Stop hit — per-position SL triggered | 2023-07-06 10:40:00 | 374.06 | 374.10 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2023-07-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-07 10:45:00 | 372.88 | 373.85 | 0.00 | ORB-short ORB[373.01,374.62] vol=2.2x ATR=0.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-07 10:50:00 | 371.88 | 373.74 | 0.00 | T1 1.5R @ 371.88 |
| Stop hit — per-position SL triggered | 2023-07-07 11:05:00 | 372.88 | 373.69 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-07-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 10:10:00 | 378.25 | 376.66 | 0.00 | ORB-long ORB[374.28,376.71] vol=2.4x ATR=0.85 |
| Stop hit — per-position SL triggered | 2023-07-11 10:40:00 | 377.40 | 377.11 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-07-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 10:50:00 | 375.21 | 374.60 | 0.00 | ORB-long ORB[373.33,374.88] vol=1.9x ATR=0.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-17 11:05:00 | 376.10 | 374.72 | 0.00 | T1 1.5R @ 376.10 |
| Stop hit — per-position SL triggered | 2023-07-17 11:15:00 | 375.21 | 374.78 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-07-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-27 09:55:00 | 379.97 | 379.45 | 0.00 | ORB-long ORB[377.80,379.78] vol=2.1x ATR=0.65 |
| Stop hit — per-position SL triggered | 2023-07-27 10:10:00 | 379.32 | 379.56 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-31 11:15:00 | 371.45 | 372.94 | 0.00 | ORB-short ORB[372.52,375.00] vol=3.1x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-31 12:35:00 | 370.46 | 371.93 | 0.00 | T1 1.5R @ 370.46 |
| Stop hit — per-position SL triggered | 2023-07-31 15:10:00 | 371.45 | 371.46 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-08-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-01 10:35:00 | 369.97 | 371.00 | 0.00 | ORB-short ORB[370.00,373.15] vol=1.6x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-01 11:00:00 | 369.14 | 370.75 | 0.00 | T1 1.5R @ 369.14 |
| Stop hit — per-position SL triggered | 2023-08-01 11:10:00 | 369.97 | 370.71 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-08-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-02 11:00:00 | 365.91 | 367.64 | 0.00 | ORB-short ORB[367.40,369.58] vol=1.7x ATR=0.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-02 11:05:00 | 365.12 | 367.48 | 0.00 | T1 1.5R @ 365.12 |
| Stop hit — per-position SL triggered | 2023-08-02 11:10:00 | 365.91 | 367.38 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-08-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-04 09:50:00 | 363.96 | 364.49 | 0.00 | ORB-short ORB[364.23,366.00] vol=1.5x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-04 10:00:00 | 362.45 | 364.04 | 0.00 | T1 1.5R @ 362.45 |
| Stop hit — per-position SL triggered | 2023-08-04 10:05:00 | 363.96 | 364.06 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-08-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-07 10:55:00 | 365.75 | 366.64 | 0.00 | ORB-short ORB[366.05,369.00] vol=2.5x ATR=0.66 |
| Stop hit — per-position SL triggered | 2023-08-07 11:30:00 | 366.41 | 366.51 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-08-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-09 10:50:00 | 364.56 | 364.81 | 0.00 | ORB-short ORB[365.04,366.80] vol=1.7x ATR=0.60 |
| Stop hit — per-position SL triggered | 2023-08-09 12:00:00 | 365.16 | 364.72 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-08-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-10 10:30:00 | 363.39 | 364.84 | 0.00 | ORB-short ORB[364.00,366.20] vol=2.1x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-10 12:10:00 | 362.10 | 363.86 | 0.00 | T1 1.5R @ 362.10 |
| Target hit | 2023-08-10 15:20:00 | 360.00 | 362.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — BUY (started 2023-08-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-14 11:00:00 | 359.97 | 358.73 | 0.00 | ORB-long ORB[357.00,359.77] vol=2.0x ATR=0.64 |
| Stop hit — per-position SL triggered | 2023-08-14 11:10:00 | 359.33 | 358.79 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-08-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-17 11:15:00 | 355.00 | 356.27 | 0.00 | ORB-short ORB[355.58,357.48] vol=1.9x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-17 12:10:00 | 354.05 | 355.89 | 0.00 | T1 1.5R @ 354.05 |
| Target hit | 2023-08-17 15:20:00 | 353.31 | 354.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2023-08-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-24 09:40:00 | 359.30 | 357.98 | 0.00 | ORB-long ORB[356.30,358.25] vol=1.5x ATR=0.66 |
| Stop hit — per-position SL triggered | 2023-08-24 10:00:00 | 358.64 | 358.25 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-08-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-30 10:45:00 | 356.93 | 357.11 | 0.00 | ORB-short ORB[357.01,358.59] vol=2.0x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-30 13:00:00 | 356.30 | 356.95 | 0.00 | T1 1.5R @ 356.30 |
| Stop hit — per-position SL triggered | 2023-08-30 13:10:00 | 356.93 | 356.93 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2023-08-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-31 09:35:00 | 356.35 | 354.66 | 0.00 | ORB-long ORB[353.03,355.23] vol=1.6x ATR=0.85 |
| Stop hit — per-position SL triggered | 2023-08-31 10:00:00 | 355.50 | 355.05 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-09-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-04 11:10:00 | 351.99 | 353.85 | 0.00 | ORB-short ORB[353.40,357.80] vol=1.8x ATR=0.55 |
| Stop hit — per-position SL triggered | 2023-09-04 11:25:00 | 352.54 | 353.39 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2023-09-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 10:00:00 | 355.73 | 354.34 | 0.00 | ORB-long ORB[352.62,354.00] vol=2.0x ATR=0.55 |
| Stop hit — per-position SL triggered | 2023-09-05 10:30:00 | 355.18 | 355.00 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-09-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 10:45:00 | 354.72 | 354.18 | 0.00 | ORB-long ORB[352.95,354.60] vol=2.6x ATR=0.59 |
| Stop hit — per-position SL triggered | 2023-09-06 11:45:00 | 354.13 | 354.41 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-09-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-11 10:45:00 | 361.82 | 360.63 | 0.00 | ORB-long ORB[359.05,360.40] vol=1.5x ATR=0.49 |
| Stop hit — per-position SL triggered | 2023-09-11 11:05:00 | 361.33 | 360.88 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-09-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-13 10:50:00 | 364.76 | 363.40 | 0.00 | ORB-long ORB[360.12,362.72] vol=2.6x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-13 11:10:00 | 365.86 | 363.72 | 0.00 | T1 1.5R @ 365.86 |
| Stop hit — per-position SL triggered | 2023-09-13 12:15:00 | 364.76 | 364.14 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-09-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-15 11:00:00 | 364.40 | 363.27 | 0.00 | ORB-long ORB[362.60,364.26] vol=1.8x ATR=0.46 |
| Stop hit — per-position SL triggered | 2023-09-15 11:40:00 | 363.94 | 363.62 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-09-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-18 11:10:00 | 359.91 | 360.78 | 0.00 | ORB-short ORB[361.29,362.40] vol=3.7x ATR=0.36 |
| Stop hit — per-position SL triggered | 2023-09-18 11:15:00 | 360.27 | 360.77 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-20 11:15:00 | 355.57 | 358.03 | 0.00 | ORB-short ORB[356.05,359.64] vol=1.6x ATR=0.71 |
| Stop hit — per-position SL triggered | 2023-09-20 11:20:00 | 356.28 | 358.01 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-26 09:30:00 | 353.68 | 354.79 | 0.00 | ORB-short ORB[354.79,356.36] vol=2.4x ATR=0.77 |
| Stop hit — per-position SL triggered | 2023-09-26 09:50:00 | 354.45 | 354.38 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2023-09-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-28 09:40:00 | 351.10 | 351.75 | 0.00 | ORB-short ORB[351.28,354.38] vol=6.1x ATR=0.66 |
| Stop hit — per-position SL triggered | 2023-09-28 09:50:00 | 351.76 | 351.72 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-10-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-05 10:05:00 | 347.25 | 346.61 | 0.00 | ORB-long ORB[344.23,346.68] vol=2.3x ATR=0.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-05 10:50:00 | 348.21 | 346.91 | 0.00 | T1 1.5R @ 348.21 |
| Target hit | 2023-10-05 13:15:00 | 347.59 | 347.63 | 0.00 | Trail-exit close<VWAP |

### Cycle 41 — BUY (started 2023-10-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-09 11:00:00 | 346.51 | 346.07 | 0.00 | ORB-long ORB[344.78,346.38] vol=1.8x ATR=0.55 |
| Stop hit — per-position SL triggered | 2023-10-09 11:20:00 | 345.96 | 346.11 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2023-10-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 09:45:00 | 355.15 | 353.98 | 0.00 | ORB-long ORB[352.01,354.40] vol=2.5x ATR=0.71 |
| Stop hit — per-position SL triggered | 2023-10-11 09:55:00 | 354.44 | 354.09 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-10-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-12 11:10:00 | 352.97 | 353.52 | 0.00 | ORB-short ORB[354.11,355.38] vol=2.0x ATR=0.51 |
| Stop hit — per-position SL triggered | 2023-10-12 11:30:00 | 353.48 | 353.50 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-10-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 11:10:00 | 351.76 | 351.36 | 0.00 | ORB-long ORB[350.16,351.60] vol=4.0x ATR=0.37 |
| Stop hit — per-position SL triggered | 2023-10-17 11:40:00 | 351.39 | 351.38 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2023-10-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 09:45:00 | 353.05 | 353.77 | 0.00 | ORB-short ORB[353.81,355.04] vol=1.9x ATR=0.64 |
| Stop hit — per-position SL triggered | 2023-10-18 09:55:00 | 353.69 | 353.72 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2023-10-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-20 10:30:00 | 350.67 | 348.97 | 0.00 | ORB-long ORB[346.53,348.33] vol=4.7x ATR=0.66 |
| Stop hit — per-position SL triggered | 2023-10-20 10:50:00 | 350.01 | 349.09 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2023-10-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-26 09:40:00 | 341.57 | 343.63 | 0.00 | ORB-short ORB[343.01,345.54] vol=1.6x ATR=0.98 |
| Stop hit — per-position SL triggered | 2023-10-26 10:00:00 | 342.55 | 343.15 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-11-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-01 11:10:00 | 344.40 | 344.88 | 0.00 | ORB-short ORB[345.20,347.19] vol=2.8x ATR=0.56 |
| Stop hit — per-position SL triggered | 2023-11-01 11:55:00 | 344.96 | 344.82 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2023-11-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-06 10:50:00 | 347.06 | 348.18 | 0.00 | ORB-short ORB[348.02,349.40] vol=1.8x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-06 11:05:00 | 346.42 | 347.94 | 0.00 | T1 1.5R @ 346.42 |
| Stop hit — per-position SL triggered | 2023-11-06 11:10:00 | 347.06 | 347.91 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2023-11-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-07 10:00:00 | 346.14 | 346.83 | 0.00 | ORB-short ORB[346.24,347.95] vol=1.8x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-07 10:05:00 | 345.41 | 346.74 | 0.00 | T1 1.5R @ 345.41 |
| Stop hit — per-position SL triggered | 2023-11-07 10:30:00 | 346.14 | 346.56 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2023-11-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-08 10:30:00 | 348.49 | 349.21 | 0.00 | ORB-short ORB[349.27,352.19] vol=1.5x ATR=0.71 |
| Stop hit — per-position SL triggered | 2023-11-08 10:40:00 | 349.20 | 349.16 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-11-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-09 11:05:00 | 349.80 | 348.76 | 0.00 | ORB-long ORB[347.82,349.76] vol=2.5x ATR=0.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-09 11:20:00 | 350.56 | 349.05 | 0.00 | T1 1.5R @ 350.56 |
| Stop hit — per-position SL triggered | 2023-11-09 14:00:00 | 349.80 | 349.83 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2023-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-13 10:15:00 | 347.34 | 348.19 | 0.00 | ORB-short ORB[347.77,350.19] vol=4.8x ATR=0.65 |
| Stop hit — per-position SL triggered | 2023-11-13 11:15:00 | 347.99 | 347.73 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-11-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 11:00:00 | 355.53 | 354.52 | 0.00 | ORB-long ORB[353.00,354.77] vol=2.6x ATR=0.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-16 11:10:00 | 356.21 | 354.75 | 0.00 | T1 1.5R @ 356.21 |
| Stop hit — per-position SL triggered | 2023-11-16 11:25:00 | 355.53 | 354.88 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2023-11-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-30 10:05:00 | 350.17 | 351.28 | 0.00 | ORB-short ORB[351.29,353.20] vol=1.9x ATR=0.75 |
| Stop hit — per-position SL triggered | 2023-11-30 11:00:00 | 350.92 | 350.93 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2023-12-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-06 09:30:00 | 365.16 | 366.14 | 0.00 | ORB-short ORB[365.39,367.87] vol=1.7x ATR=0.81 |
| Stop hit — per-position SL triggered | 2023-12-06 09:40:00 | 365.97 | 365.93 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2023-12-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-11 09:30:00 | 371.63 | 370.02 | 0.00 | ORB-long ORB[368.06,370.91] vol=1.9x ATR=1.01 |
| Stop hit — per-position SL triggered | 2023-12-11 09:50:00 | 370.62 | 370.49 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2023-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-12 10:15:00 | 368.00 | 369.52 | 0.00 | ORB-short ORB[368.86,370.60] vol=1.5x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-12 13:00:00 | 367.07 | 368.52 | 0.00 | T1 1.5R @ 367.07 |
| Target hit | 2023-12-12 15:20:00 | 364.24 | 367.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — BUY (started 2023-12-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-14 09:30:00 | 370.10 | 369.00 | 0.00 | ORB-long ORB[366.49,369.89] vol=2.1x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-14 09:50:00 | 371.49 | 370.09 | 0.00 | T1 1.5R @ 371.49 |
| Target hit | 2023-12-14 10:35:00 | 370.44 | 370.60 | 0.00 | Trail-exit close<VWAP |

### Cycle 60 — BUY (started 2023-12-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-18 10:20:00 | 370.90 | 370.12 | 0.00 | ORB-long ORB[368.03,370.12] vol=6.5x ATR=0.63 |
| Stop hit — per-position SL triggered | 2023-12-18 10:25:00 | 370.27 | 370.14 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2023-12-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 10:40:00 | 372.44 | 371.05 | 0.00 | ORB-long ORB[369.80,371.98] vol=2.7x ATR=0.79 |
| Stop hit — per-position SL triggered | 2023-12-20 10:55:00 | 371.65 | 371.17 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2023-12-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 09:40:00 | 374.30 | 372.91 | 0.00 | ORB-long ORB[370.41,371.80] vol=5.7x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-22 10:05:00 | 375.80 | 373.56 | 0.00 | T1 1.5R @ 375.80 |
| Stop hit — per-position SL triggered | 2023-12-22 10:10:00 | 374.30 | 373.62 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2023-12-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 11:05:00 | 375.26 | 373.51 | 0.00 | ORB-long ORB[371.60,373.40] vol=1.6x ATR=0.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-26 11:40:00 | 376.39 | 374.16 | 0.00 | T1 1.5R @ 376.39 |
| Target hit | 2023-12-26 15:20:00 | 377.35 | 376.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — BUY (started 2023-12-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 09:55:00 | 379.65 | 378.88 | 0.00 | ORB-long ORB[377.46,379.20] vol=2.4x ATR=0.77 |
| Stop hit — per-position SL triggered | 2023-12-27 10:00:00 | 378.88 | 378.91 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2023-12-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 09:45:00 | 383.58 | 382.52 | 0.00 | ORB-long ORB[380.62,383.00] vol=1.5x ATR=0.82 |
| Stop hit — per-position SL triggered | 2023-12-28 09:50:00 | 382.76 | 382.56 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2023-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-29 11:15:00 | 379.41 | 380.27 | 0.00 | ORB-short ORB[379.67,383.55] vol=1.7x ATR=0.65 |
| Stop hit — per-position SL triggered | 2023-12-29 11:55:00 | 380.06 | 380.15 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2024-01-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 09:55:00 | 377.03 | 379.18 | 0.00 | ORB-short ORB[379.40,381.60] vol=2.9x ATR=1.01 |
| Stop hit — per-position SL triggered | 2024-01-02 10:15:00 | 378.04 | 378.89 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2024-01-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-05 10:50:00 | 370.54 | 370.87 | 0.00 | ORB-short ORB[371.79,373.98] vol=4.1x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-05 11:15:00 | 369.31 | 370.85 | 0.00 | T1 1.5R @ 369.31 |
| Stop hit — per-position SL triggered | 2024-01-05 11:55:00 | 370.54 | 370.82 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2024-01-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-08 11:05:00 | 364.92 | 365.69 | 0.00 | ORB-short ORB[367.20,370.29] vol=18.3x ATR=0.68 |
| Stop hit — per-position SL triggered | 2024-01-08 11:10:00 | 365.60 | 365.68 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2024-01-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 10:30:00 | 369.92 | 368.68 | 0.00 | ORB-long ORB[365.80,368.20] vol=4.4x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-09 10:55:00 | 370.98 | 369.15 | 0.00 | T1 1.5R @ 370.98 |
| Stop hit — per-position SL triggered | 2024-01-09 12:00:00 | 369.92 | 369.41 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2024-01-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-16 10:45:00 | 370.96 | 370.58 | 0.00 | ORB-long ORB[368.49,370.20] vol=9.7x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-16 11:10:00 | 371.90 | 370.67 | 0.00 | T1 1.5R @ 371.90 |
| Stop hit — per-position SL triggered | 2024-01-16 11:20:00 | 370.96 | 370.69 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2024-01-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-17 10:20:00 | 362.08 | 363.98 | 0.00 | ORB-short ORB[362.86,366.93] vol=2.3x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-17 10:35:00 | 360.66 | 363.33 | 0.00 | T1 1.5R @ 360.66 |
| Target hit | 2024-01-17 15:20:00 | 355.80 | 359.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 73 — SELL (started 2024-01-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-24 10:45:00 | 355.99 | 358.26 | 0.00 | ORB-short ORB[356.16,360.30] vol=3.9x ATR=1.11 |
| Stop hit — per-position SL triggered | 2024-01-24 10:55:00 | 357.10 | 357.71 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2024-01-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-25 09:30:00 | 356.23 | 357.76 | 0.00 | ORB-short ORB[356.35,359.52] vol=1.6x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-25 10:15:00 | 354.17 | 356.94 | 0.00 | T1 1.5R @ 354.17 |
| Target hit | 2024-01-25 15:10:00 | 353.53 | 353.42 | 0.00 | Trail-exit close>VWAP |

### Cycle 75 — BUY (started 2024-01-31 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 10:05:00 | 365.61 | 361.44 | 0.00 | ORB-long ORB[360.13,363.80] vol=1.6x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-31 10:30:00 | 367.59 | 362.27 | 0.00 | T1 1.5R @ 367.59 |
| Stop hit — per-position SL triggered | 2024-01-31 10:50:00 | 365.61 | 362.73 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-02-02 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-02 10:20:00 | 365.60 | 366.54 | 0.00 | ORB-short ORB[366.30,368.37] vol=2.4x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 11:05:00 | 364.42 | 366.32 | 0.00 | T1 1.5R @ 364.42 |
| Stop hit — per-position SL triggered | 2024-02-02 11:10:00 | 365.60 | 366.30 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2024-02-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-06 10:30:00 | 358.03 | 359.00 | 0.00 | ORB-short ORB[359.02,362.80] vol=12.0x ATR=0.77 |
| Stop hit — per-position SL triggered | 2024-02-06 10:35:00 | 358.80 | 359.00 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2024-02-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-07 09:30:00 | 361.25 | 360.48 | 0.00 | ORB-long ORB[358.20,360.97] vol=2.1x ATR=0.89 |
| Stop hit — per-position SL triggered | 2024-02-07 09:45:00 | 360.36 | 360.67 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2024-02-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 11:00:00 | 351.95 | 357.37 | 0.00 | ORB-short ORB[358.00,360.76] vol=3.0x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 12:30:00 | 350.46 | 354.71 | 0.00 | T1 1.5R @ 350.46 |
| Stop hit — per-position SL triggered | 2024-02-08 13:15:00 | 351.95 | 354.33 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2024-02-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-13 09:35:00 | 344.93 | 343.83 | 0.00 | ORB-long ORB[341.80,344.17] vol=1.6x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-13 10:10:00 | 346.37 | 344.55 | 0.00 | T1 1.5R @ 346.37 |
| Stop hit — per-position SL triggered | 2024-02-13 11:05:00 | 344.93 | 345.03 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2024-02-15 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-15 10:50:00 | 345.62 | 345.89 | 0.00 | ORB-short ORB[346.34,351.49] vol=3.8x ATR=0.99 |
| Stop hit — per-position SL triggered | 2024-02-15 13:40:00 | 346.61 | 345.87 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2024-02-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-21 10:20:00 | 350.35 | 351.32 | 0.00 | ORB-short ORB[351.31,353.80] vol=1.7x ATR=0.88 |
| Stop hit — per-position SL triggered | 2024-02-21 11:10:00 | 351.23 | 351.19 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2024-02-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-22 09:35:00 | 345.58 | 346.62 | 0.00 | ORB-short ORB[347.00,349.00] vol=3.3x ATR=0.82 |
| Stop hit — per-position SL triggered | 2024-02-22 09:40:00 | 346.40 | 346.56 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2024-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-26 10:15:00 | 341.57 | 343.00 | 0.00 | ORB-short ORB[343.18,345.08] vol=4.1x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-26 10:40:00 | 340.53 | 342.05 | 0.00 | T1 1.5R @ 340.53 |
| Stop hit — per-position SL triggered | 2024-02-26 10:50:00 | 341.57 | 342.03 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2024-02-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-27 10:20:00 | 340.70 | 341.18 | 0.00 | ORB-short ORB[340.80,342.99] vol=3.0x ATR=0.62 |
| Stop hit — per-position SL triggered | 2024-02-27 11:20:00 | 341.32 | 341.11 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2024-02-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:55:00 | 338.66 | 341.51 | 0.00 | ORB-short ORB[341.26,342.57] vol=2.4x ATR=0.70 |
| Stop hit — per-position SL triggered | 2024-02-28 11:00:00 | 339.36 | 341.32 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2024-02-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-29 09:35:00 | 335.01 | 335.78 | 0.00 | ORB-short ORB[335.59,337.59] vol=5.5x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-29 09:40:00 | 333.54 | 335.37 | 0.00 | T1 1.5R @ 333.54 |
| Stop hit — per-position SL triggered | 2024-02-29 09:45:00 | 335.01 | 335.48 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2024-03-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-04 10:05:00 | 347.79 | 346.92 | 0.00 | ORB-long ORB[345.00,347.71] vol=1.7x ATR=0.74 |
| Stop hit — per-position SL triggered | 2024-03-04 10:35:00 | 347.05 | 347.29 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2024-03-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-05 09:35:00 | 343.78 | 344.44 | 0.00 | ORB-short ORB[344.27,345.44] vol=2.4x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-05 10:00:00 | 342.71 | 344.15 | 0.00 | T1 1.5R @ 342.71 |
| Stop hit — per-position SL triggered | 2024-03-05 11:30:00 | 343.78 | 343.54 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2024-03-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-06 09:40:00 | 348.76 | 345.89 | 0.00 | ORB-long ORB[343.68,345.37] vol=2.6x ATR=0.89 |
| Stop hit — per-position SL triggered | 2024-03-06 09:50:00 | 347.87 | 346.67 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2024-03-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-12 10:40:00 | 343.98 | 346.28 | 0.00 | ORB-short ORB[345.20,347.95] vol=1.6x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 10:45:00 | 342.14 | 345.79 | 0.00 | T1 1.5R @ 342.14 |
| Stop hit — per-position SL triggered | 2024-03-12 10:55:00 | 343.98 | 345.60 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2024-03-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-13 10:20:00 | 347.44 | 346.21 | 0.00 | ORB-long ORB[344.07,345.97] vol=1.8x ATR=0.86 |
| Stop hit — per-position SL triggered | 2024-03-13 10:35:00 | 346.58 | 346.30 | 0.00 | SL hit |

### Cycle 93 — BUY (started 2024-03-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-14 11:15:00 | 350.74 | 346.35 | 0.00 | ORB-long ORB[344.33,347.40] vol=1.8x ATR=1.06 |
| Stop hit — per-position SL triggered | 2024-03-14 11:25:00 | 349.68 | 346.52 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2024-03-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-15 10:55:00 | 345.63 | 346.27 | 0.00 | ORB-short ORB[346.91,350.02] vol=1.8x ATR=0.91 |
| Stop hit — per-position SL triggered | 2024-03-15 11:10:00 | 346.54 | 346.21 | 0.00 | SL hit |

### Cycle 95 — SELL (started 2024-03-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-18 11:10:00 | 344.95 | 346.15 | 0.00 | ORB-short ORB[346.02,347.60] vol=2.2x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-18 11:15:00 | 343.66 | 345.56 | 0.00 | T1 1.5R @ 343.66 |
| Stop hit — per-position SL triggered | 2024-03-18 11:30:00 | 344.95 | 345.44 | 0.00 | SL hit |

### Cycle 96 — BUY (started 2024-03-19 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-19 10:45:00 | 348.14 | 347.30 | 0.00 | ORB-long ORB[346.11,347.68] vol=1.5x ATR=0.73 |
| Stop hit — per-position SL triggered | 2024-03-19 11:05:00 | 347.41 | 347.51 | 0.00 | SL hit |

### Cycle 97 — SELL (started 2024-03-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-20 10:40:00 | 346.98 | 348.68 | 0.00 | ORB-short ORB[349.80,352.29] vol=2.0x ATR=0.84 |
| Stop hit — per-position SL triggered | 2024-03-20 10:50:00 | 347.82 | 348.38 | 0.00 | SL hit |

### Cycle 98 — BUY (started 2024-03-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 10:10:00 | 355.56 | 354.90 | 0.00 | ORB-long ORB[353.00,355.31] vol=2.0x ATR=0.88 |
| Stop hit — per-position SL triggered | 2024-03-22 13:05:00 | 354.68 | 355.55 | 0.00 | SL hit |

### Cycle 99 — BUY (started 2024-03-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 10:45:00 | 353.79 | 353.01 | 0.00 | ORB-long ORB[350.78,352.40] vol=1.8x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-27 13:30:00 | 354.83 | 353.49 | 0.00 | T1 1.5R @ 354.83 |
| Stop hit — per-position SL triggered | 2024-03-27 14:20:00 | 353.79 | 353.67 | 0.00 | SL hit |

### Cycle 100 — SELL (started 2024-04-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-02 10:00:00 | 356.18 | 356.76 | 0.00 | ORB-short ORB[356.40,359.60] vol=1.7x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-02 10:15:00 | 354.94 | 356.45 | 0.00 | T1 1.5R @ 354.94 |
| Target hit | 2024-04-02 15:20:00 | 351.06 | 353.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 101 — BUY (started 2024-04-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-05 10:40:00 | 351.81 | 351.16 | 0.00 | ORB-long ORB[348.02,351.77] vol=1.7x ATR=0.78 |
| Stop hit — per-position SL triggered | 2024-04-05 10:45:00 | 351.03 | 351.16 | 0.00 | SL hit |

### Cycle 102 — BUY (started 2024-04-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 11:05:00 | 359.12 | 357.02 | 0.00 | ORB-long ORB[355.13,357.33] vol=2.7x ATR=0.64 |
| Stop hit — per-position SL triggered | 2024-04-16 11:20:00 | 358.48 | 357.25 | 0.00 | SL hit |

### Cycle 103 — BUY (started 2024-04-23 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 10:20:00 | 364.51 | 363.96 | 0.00 | ORB-long ORB[360.62,364.40] vol=1.6x ATR=0.76 |
| Stop hit — per-position SL triggered | 2024-04-23 10:40:00 | 363.75 | 363.96 | 0.00 | SL hit |

### Cycle 104 — BUY (started 2024-05-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-07 10:55:00 | 326.17 | 324.99 | 0.00 | ORB-long ORB[324.48,326.00] vol=2.2x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 11:00:00 | 327.19 | 325.11 | 0.00 | T1 1.5R @ 327.19 |
| Stop hit — per-position SL triggered | 2024-05-07 11:15:00 | 326.17 | 325.29 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-16 11:10:00 | 392.48 | 2023-05-16 11:20:00 | 391.23 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2023-05-16 11:10:00 | 392.48 | 2023-05-16 15:20:00 | 388.64 | TARGET_HIT | 0.50 | 0.98% |
| SELL | retest1 | 2023-05-17 10:30:00 | 384.01 | 2023-05-17 11:30:00 | 382.38 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2023-05-17 10:30:00 | 384.01 | 2023-05-17 15:20:00 | 381.98 | TARGET_HIT | 0.50 | 0.53% |
| BUY | retest1 | 2023-05-18 11:05:00 | 385.88 | 2023-05-18 11:40:00 | 385.23 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-05-23 10:50:00 | 384.46 | 2023-05-23 11:25:00 | 383.53 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2023-05-23 10:50:00 | 384.46 | 2023-05-23 12:45:00 | 384.46 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-24 11:00:00 | 387.22 | 2023-05-24 11:15:00 | 386.60 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-06-06 09:30:00 | 389.03 | 2023-06-06 10:30:00 | 388.26 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-06-13 09:30:00 | 374.17 | 2023-06-13 10:30:00 | 373.20 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2023-06-13 09:30:00 | 374.17 | 2023-06-13 15:20:00 | 370.55 | TARGET_HIT | 0.50 | 0.97% |
| SELL | retest1 | 2023-06-15 10:55:00 | 370.63 | 2023-06-15 11:15:00 | 371.18 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2023-06-19 09:45:00 | 370.20 | 2023-06-19 10:05:00 | 368.80 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2023-06-19 09:45:00 | 370.20 | 2023-06-19 15:20:00 | 364.97 | TARGET_HIT | 0.50 | 1.41% |
| SELL | retest1 | 2023-06-28 10:25:00 | 366.94 | 2023-06-28 10:45:00 | 367.51 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2023-06-30 10:40:00 | 367.00 | 2023-06-30 10:50:00 | 367.67 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-07-03 10:50:00 | 368.21 | 2023-07-03 11:05:00 | 369.00 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-07-06 09:55:00 | 374.73 | 2023-07-06 10:40:00 | 374.06 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-07-07 10:45:00 | 372.88 | 2023-07-07 10:50:00 | 371.88 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-07-07 10:45:00 | 372.88 | 2023-07-07 11:05:00 | 372.88 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-11 10:10:00 | 378.25 | 2023-07-11 10:40:00 | 377.40 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-07-17 10:50:00 | 375.21 | 2023-07-17 11:05:00 | 376.10 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2023-07-17 10:50:00 | 375.21 | 2023-07-17 11:15:00 | 375.21 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-27 09:55:00 | 379.97 | 2023-07-27 10:10:00 | 379.32 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-07-31 11:15:00 | 371.45 | 2023-07-31 12:35:00 | 370.46 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-07-31 11:15:00 | 371.45 | 2023-07-31 15:10:00 | 371.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-01 10:35:00 | 369.97 | 2023-08-01 11:00:00 | 369.14 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2023-08-01 10:35:00 | 369.97 | 2023-08-01 11:10:00 | 369.97 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-02 11:00:00 | 365.91 | 2023-08-02 11:05:00 | 365.12 | PARTIAL | 0.50 | 0.22% |
| SELL | retest1 | 2023-08-02 11:00:00 | 365.91 | 2023-08-02 11:10:00 | 365.91 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-04 09:50:00 | 363.96 | 2023-08-04 10:00:00 | 362.45 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2023-08-04 09:50:00 | 363.96 | 2023-08-04 10:05:00 | 363.96 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-07 10:55:00 | 365.75 | 2023-08-07 11:30:00 | 366.41 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-08-09 10:50:00 | 364.56 | 2023-08-09 12:00:00 | 365.16 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-08-10 10:30:00 | 363.39 | 2023-08-10 12:10:00 | 362.10 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2023-08-10 10:30:00 | 363.39 | 2023-08-10 15:20:00 | 360.00 | TARGET_HIT | 0.50 | 0.93% |
| BUY | retest1 | 2023-08-14 11:00:00 | 359.97 | 2023-08-14 11:10:00 | 359.33 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-08-17 11:15:00 | 355.00 | 2023-08-17 12:10:00 | 354.05 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-08-17 11:15:00 | 355.00 | 2023-08-17 15:20:00 | 353.31 | TARGET_HIT | 0.50 | 0.48% |
| BUY | retest1 | 2023-08-24 09:40:00 | 359.30 | 2023-08-24 10:00:00 | 358.64 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-08-30 10:45:00 | 356.93 | 2023-08-30 13:00:00 | 356.30 | PARTIAL | 0.50 | 0.18% |
| SELL | retest1 | 2023-08-30 10:45:00 | 356.93 | 2023-08-30 13:10:00 | 356.93 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-31 09:35:00 | 356.35 | 2023-08-31 10:00:00 | 355.50 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-09-04 11:10:00 | 351.99 | 2023-09-04 11:25:00 | 352.54 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-09-05 10:00:00 | 355.73 | 2023-09-05 10:30:00 | 355.18 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2023-09-06 10:45:00 | 354.72 | 2023-09-06 11:45:00 | 354.13 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-09-11 10:45:00 | 361.82 | 2023-09-11 11:05:00 | 361.33 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2023-09-13 10:50:00 | 364.76 | 2023-09-13 11:10:00 | 365.86 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-09-13 10:50:00 | 364.76 | 2023-09-13 12:15:00 | 364.76 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-15 11:00:00 | 364.40 | 2023-09-15 11:40:00 | 363.94 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2023-09-18 11:10:00 | 359.91 | 2023-09-18 11:15:00 | 360.27 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest1 | 2023-09-20 11:15:00 | 355.57 | 2023-09-20 11:20:00 | 356.28 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-09-26 09:30:00 | 353.68 | 2023-09-26 09:50:00 | 354.45 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-09-28 09:40:00 | 351.10 | 2023-09-28 09:50:00 | 351.76 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-10-05 10:05:00 | 347.25 | 2023-10-05 10:50:00 | 348.21 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2023-10-05 10:05:00 | 347.25 | 2023-10-05 13:15:00 | 347.59 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2023-10-09 11:00:00 | 346.51 | 2023-10-09 11:20:00 | 345.96 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-10-11 09:45:00 | 355.15 | 2023-10-11 09:55:00 | 354.44 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-10-12 11:10:00 | 352.97 | 2023-10-12 11:30:00 | 353.48 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2023-10-17 11:10:00 | 351.76 | 2023-10-17 11:40:00 | 351.39 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest1 | 2023-10-18 09:45:00 | 353.05 | 2023-10-18 09:55:00 | 353.69 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-10-20 10:30:00 | 350.67 | 2023-10-20 10:50:00 | 350.01 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-10-26 09:40:00 | 341.57 | 2023-10-26 10:00:00 | 342.55 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-11-01 11:10:00 | 344.40 | 2023-11-01 11:55:00 | 344.96 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-11-06 10:50:00 | 347.06 | 2023-11-06 11:05:00 | 346.42 | PARTIAL | 0.50 | 0.18% |
| SELL | retest1 | 2023-11-06 10:50:00 | 347.06 | 2023-11-06 11:10:00 | 347.06 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-07 10:00:00 | 346.14 | 2023-11-07 10:05:00 | 345.41 | PARTIAL | 0.50 | 0.21% |
| SELL | retest1 | 2023-11-07 10:00:00 | 346.14 | 2023-11-07 10:30:00 | 346.14 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-08 10:30:00 | 348.49 | 2023-11-08 10:40:00 | 349.20 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-11-09 11:05:00 | 349.80 | 2023-11-09 11:20:00 | 350.56 | PARTIAL | 0.50 | 0.22% |
| BUY | retest1 | 2023-11-09 11:05:00 | 349.80 | 2023-11-09 14:00:00 | 349.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-13 10:15:00 | 347.34 | 2023-11-13 11:15:00 | 347.99 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-11-16 11:00:00 | 355.53 | 2023-11-16 11:10:00 | 356.21 | PARTIAL | 0.50 | 0.19% |
| BUY | retest1 | 2023-11-16 11:00:00 | 355.53 | 2023-11-16 11:25:00 | 355.53 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-30 10:05:00 | 350.17 | 2023-11-30 11:00:00 | 350.92 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-12-06 09:30:00 | 365.16 | 2023-12-06 09:40:00 | 365.97 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-12-11 09:30:00 | 371.63 | 2023-12-11 09:50:00 | 370.62 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-12-12 10:15:00 | 368.00 | 2023-12-12 13:00:00 | 367.07 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2023-12-12 10:15:00 | 368.00 | 2023-12-12 15:20:00 | 364.24 | TARGET_HIT | 0.50 | 1.02% |
| BUY | retest1 | 2023-12-14 09:30:00 | 370.10 | 2023-12-14 09:50:00 | 371.49 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2023-12-14 09:30:00 | 370.10 | 2023-12-14 10:35:00 | 370.44 | TARGET_HIT | 0.50 | 0.09% |
| BUY | retest1 | 2023-12-18 10:20:00 | 370.90 | 2023-12-18 10:25:00 | 370.27 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-12-20 10:40:00 | 372.44 | 2023-12-20 10:55:00 | 371.65 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-12-22 09:40:00 | 374.30 | 2023-12-22 10:05:00 | 375.80 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-12-22 09:40:00 | 374.30 | 2023-12-22 10:10:00 | 374.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-26 11:05:00 | 375.26 | 2023-12-26 11:40:00 | 376.39 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-12-26 11:05:00 | 375.26 | 2023-12-26 15:20:00 | 377.35 | TARGET_HIT | 0.50 | 0.56% |
| BUY | retest1 | 2023-12-27 09:55:00 | 379.65 | 2023-12-27 10:00:00 | 378.88 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-12-28 09:45:00 | 383.58 | 2023-12-28 09:50:00 | 382.76 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-12-29 11:15:00 | 379.41 | 2023-12-29 11:55:00 | 380.06 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-01-02 09:55:00 | 377.03 | 2024-01-02 10:15:00 | 378.04 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-01-05 10:50:00 | 370.54 | 2024-01-05 11:15:00 | 369.31 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-01-05 10:50:00 | 370.54 | 2024-01-05 11:55:00 | 370.54 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-08 11:05:00 | 364.92 | 2024-01-08 11:10:00 | 365.60 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2024-01-09 10:30:00 | 369.92 | 2024-01-09 10:55:00 | 370.98 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-01-09 10:30:00 | 369.92 | 2024-01-09 12:00:00 | 369.92 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-16 10:45:00 | 370.96 | 2024-01-16 11:10:00 | 371.90 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2024-01-16 10:45:00 | 370.96 | 2024-01-16 11:20:00 | 370.96 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-17 10:20:00 | 362.08 | 2024-01-17 10:35:00 | 360.66 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-01-17 10:20:00 | 362.08 | 2024-01-17 15:20:00 | 355.80 | TARGET_HIT | 0.50 | 1.73% |
| SELL | retest1 | 2024-01-24 10:45:00 | 355.99 | 2024-01-24 10:55:00 | 357.10 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-01-25 09:30:00 | 356.23 | 2024-01-25 10:15:00 | 354.17 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-01-25 09:30:00 | 356.23 | 2024-01-25 15:10:00 | 353.53 | TARGET_HIT | 0.50 | 0.76% |
| BUY | retest1 | 2024-01-31 10:05:00 | 365.61 | 2024-01-31 10:30:00 | 367.59 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-01-31 10:05:00 | 365.61 | 2024-01-31 10:50:00 | 365.61 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-02 10:20:00 | 365.60 | 2024-02-02 11:05:00 | 364.42 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-02-02 10:20:00 | 365.60 | 2024-02-02 11:10:00 | 365.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-06 10:30:00 | 358.03 | 2024-02-06 10:35:00 | 358.80 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-02-07 09:30:00 | 361.25 | 2024-02-07 09:45:00 | 360.36 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-02-08 11:00:00 | 351.95 | 2024-02-08 12:30:00 | 350.46 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-02-08 11:00:00 | 351.95 | 2024-02-08 13:15:00 | 351.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-13 09:35:00 | 344.93 | 2024-02-13 10:10:00 | 346.37 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-02-13 09:35:00 | 344.93 | 2024-02-13 11:05:00 | 344.93 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-15 10:50:00 | 345.62 | 2024-02-15 13:40:00 | 346.61 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-02-21 10:20:00 | 350.35 | 2024-02-21 11:10:00 | 351.23 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-02-22 09:35:00 | 345.58 | 2024-02-22 09:40:00 | 346.40 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-02-26 10:15:00 | 341.57 | 2024-02-26 10:40:00 | 340.53 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-02-26 10:15:00 | 341.57 | 2024-02-26 10:50:00 | 341.57 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-27 10:20:00 | 340.70 | 2024-02-27 11:20:00 | 341.32 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-02-28 10:55:00 | 338.66 | 2024-02-28 11:00:00 | 339.36 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-02-29 09:35:00 | 335.01 | 2024-02-29 09:40:00 | 333.54 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-02-29 09:35:00 | 335.01 | 2024-02-29 09:45:00 | 335.01 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-04 10:05:00 | 347.79 | 2024-03-04 10:35:00 | 347.05 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-03-05 09:35:00 | 343.78 | 2024-03-05 10:00:00 | 342.71 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-03-05 09:35:00 | 343.78 | 2024-03-05 11:30:00 | 343.78 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-06 09:40:00 | 348.76 | 2024-03-06 09:50:00 | 347.87 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-03-12 10:40:00 | 343.98 | 2024-03-12 10:45:00 | 342.14 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-03-12 10:40:00 | 343.98 | 2024-03-12 10:55:00 | 343.98 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-13 10:20:00 | 347.44 | 2024-03-13 10:35:00 | 346.58 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-03-14 11:15:00 | 350.74 | 2024-03-14 11:25:00 | 349.68 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-03-15 10:55:00 | 345.63 | 2024-03-15 11:10:00 | 346.54 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-03-18 11:10:00 | 344.95 | 2024-03-18 11:15:00 | 343.66 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-03-18 11:10:00 | 344.95 | 2024-03-18 11:30:00 | 344.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-19 10:45:00 | 348.14 | 2024-03-19 11:05:00 | 347.41 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-03-20 10:40:00 | 346.98 | 2024-03-20 10:50:00 | 347.82 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-03-22 10:10:00 | 355.56 | 2024-03-22 13:05:00 | 354.68 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-03-27 10:45:00 | 353.79 | 2024-03-27 13:30:00 | 354.83 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-03-27 10:45:00 | 353.79 | 2024-03-27 14:20:00 | 353.79 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-02 10:00:00 | 356.18 | 2024-04-02 10:15:00 | 354.94 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-04-02 10:00:00 | 356.18 | 2024-04-02 15:20:00 | 351.06 | TARGET_HIT | 0.50 | 1.44% |
| BUY | retest1 | 2024-04-05 10:40:00 | 351.81 | 2024-04-05 10:45:00 | 351.03 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-04-16 11:05:00 | 359.12 | 2024-04-16 11:20:00 | 358.48 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-04-23 10:20:00 | 364.51 | 2024-04-23 10:40:00 | 363.75 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-05-07 10:55:00 | 326.17 | 2024-05-07 11:00:00 | 327.19 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2024-05-07 10:55:00 | 326.17 | 2024-05-07 11:15:00 | 326.17 | STOP_HIT | 0.50 | 0.00% |
