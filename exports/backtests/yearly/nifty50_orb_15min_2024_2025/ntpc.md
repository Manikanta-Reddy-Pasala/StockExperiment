# NTPC (NTPC)

## Backtest Summary

- **Window:** 2024-09-09 09:15:00 → 2026-05-08 15:25:00 (30775 bars)
- **Last close:** 402.10
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
| ENTRY1 | 37 |
| ENTRY2 | 0 |
| PARTIAL | 14 |
| TARGET_HIT | 10 |
| STOP_HIT | 27 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 51 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 24 / 27
- **Target hits / Stop hits / Partials:** 10 / 27 / 14
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 9.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 6 | 35.3% | 3 | 11 | 3 | -0.00% | -0.0% |
| BUY @ 2nd Alert (retest1) | 17 | 6 | 35.3% | 3 | 11 | 3 | -0.00% | -0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 34 | 18 | 52.9% | 7 | 16 | 11 | 0.27% | 9.1% |
| SELL @ 2nd Alert (retest1) | 34 | 18 | 52.9% | 7 | 16 | 11 | 0.27% | 9.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 51 | 24 | 47.1% | 10 | 27 | 14 | 0.18% | 9.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 10:20:00 | 395.45 | 392.19 | 0.00 | ORB-long ORB[390.00,393.95] vol=1.5x ATR=1.47 |
| Stop hit — per-position SL triggered | 2024-09-10 10:40:00 | 393.98 | 392.98 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-09-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-16 09:50:00 | 407.60 | 406.24 | 0.00 | ORB-long ORB[402.00,406.30] vol=1.5x ATR=1.25 |
| Stop hit — per-position SL triggered | 2024-09-16 09:55:00 | 406.35 | 406.30 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-10-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:40:00 | 419.65 | 425.11 | 0.00 | ORB-short ORB[427.55,433.00] vol=3.0x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:50:00 | 417.12 | 424.26 | 0.00 | T1 1.5R @ 417.12 |
| Stop hit — per-position SL triggered | 2024-10-07 11:25:00 | 419.65 | 422.13 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-10-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 10:05:00 | 427.90 | 423.78 | 0.00 | ORB-long ORB[420.50,424.45] vol=1.9x ATR=1.51 |
| Stop hit — per-position SL triggered | 2024-10-10 12:00:00 | 426.39 | 426.89 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-10-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 10:05:00 | 426.85 | 425.54 | 0.00 | ORB-long ORB[423.55,426.65] vol=1.6x ATR=1.02 |
| Stop hit — per-position SL triggered | 2024-10-15 10:10:00 | 425.83 | 425.55 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 11:15:00 | 423.90 | 427.00 | 0.00 | ORB-short ORB[424.50,428.85] vol=1.8x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 11:45:00 | 422.73 | 426.49 | 0.00 | T1 1.5R @ 422.73 |
| Stop hit — per-position SL triggered | 2024-10-16 12:20:00 | 423.90 | 426.10 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-10-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 10:40:00 | 421.85 | 418.25 | 0.00 | ORB-long ORB[414.45,419.00] vol=1.8x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 11:55:00 | 424.00 | 419.60 | 0.00 | T1 1.5R @ 424.00 |
| Target hit | 2024-10-18 15:20:00 | 425.80 | 422.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2024-10-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:35:00 | 419.60 | 421.70 | 0.00 | ORB-short ORB[422.10,425.50] vol=4.1x ATR=1.67 |
| Stop hit — per-position SL triggered | 2024-10-22 11:30:00 | 421.27 | 421.45 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-10-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-23 09:45:00 | 404.60 | 408.25 | 0.00 | ORB-short ORB[410.30,414.75] vol=2.9x ATR=1.92 |
| Stop hit — per-position SL triggered | 2024-10-23 09:50:00 | 406.52 | 408.11 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-11-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 09:50:00 | 367.90 | 364.35 | 0.00 | ORB-long ORB[361.35,364.65] vol=1.7x ATR=1.22 |
| Stop hit — per-position SL triggered | 2024-11-27 09:55:00 | 366.68 | 364.57 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-11-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:35:00 | 365.50 | 368.34 | 0.00 | ORB-short ORB[366.40,371.85] vol=1.8x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 12:35:00 | 363.82 | 366.86 | 0.00 | T1 1.5R @ 363.82 |
| Target hit | 2024-11-28 15:20:00 | 363.20 | 364.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — SELL (started 2024-12-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 10:45:00 | 360.20 | 362.41 | 0.00 | ORB-short ORB[363.70,367.00] vol=3.2x ATR=0.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 11:45:00 | 359.14 | 361.41 | 0.00 | T1 1.5R @ 359.14 |
| Target hit | 2024-12-12 15:20:00 | 355.75 | 356.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2024-12-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 10:10:00 | 350.55 | 351.39 | 0.00 | ORB-short ORB[351.80,353.85] vol=3.5x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 10:20:00 | 349.63 | 351.08 | 0.00 | T1 1.5R @ 349.63 |
| Stop hit — per-position SL triggered | 2024-12-17 10:55:00 | 350.55 | 350.28 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-12-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 10:55:00 | 344.45 | 346.49 | 0.00 | ORB-short ORB[347.00,351.00] vol=3.2x ATR=0.93 |
| Stop hit — per-position SL triggered | 2024-12-18 11:10:00 | 345.38 | 346.29 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-01-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 10:45:00 | 327.30 | 327.10 | 0.00 | ORB-long ORB[323.35,326.55] vol=1.7x ATR=0.91 |
| Stop hit — per-position SL triggered | 2025-01-17 10:50:00 | 326.39 | 327.06 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-01-22 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-22 11:10:00 | 319.75 | 323.91 | 0.00 | ORB-short ORB[322.25,327.00] vol=1.8x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 12:05:00 | 317.98 | 323.23 | 0.00 | T1 1.5R @ 317.98 |
| Stop hit — per-position SL triggered | 2025-01-22 12:10:00 | 319.75 | 323.05 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-02-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 10:55:00 | 328.90 | 327.89 | 0.00 | ORB-long ORB[324.75,327.85] vol=1.5x ATR=0.93 |
| Stop hit — per-position SL triggered | 2025-02-01 11:00:00 | 327.97 | 327.90 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-02-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 11:10:00 | 316.65 | 318.19 | 0.00 | ORB-short ORB[319.00,322.70] vol=1.6x ATR=0.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 11:50:00 | 315.54 | 317.82 | 0.00 | T1 1.5R @ 315.54 |
| Target hit | 2025-02-06 15:20:00 | 313.65 | 314.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — SELL (started 2025-02-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-14 11:00:00 | 300.65 | 304.19 | 0.00 | ORB-short ORB[303.80,308.25] vol=1.6x ATR=0.95 |
| Stop hit — per-position SL triggered | 2025-02-14 11:10:00 | 301.60 | 303.97 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-03-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-07 11:05:00 | 333.60 | 334.52 | 0.00 | ORB-short ORB[334.00,338.75] vol=1.7x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-07 11:45:00 | 332.12 | 334.28 | 0.00 | T1 1.5R @ 332.12 |
| Target hit | 2025-03-07 15:20:00 | 329.55 | 331.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — BUY (started 2025-03-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 10:00:00 | 333.35 | 331.97 | 0.00 | ORB-long ORB[329.80,333.00] vol=1.5x ATR=1.01 |
| Stop hit — per-position SL triggered | 2025-03-13 10:10:00 | 332.34 | 332.04 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-03-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:55:00 | 334.25 | 333.25 | 0.00 | ORB-long ORB[331.35,333.80] vol=2.4x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 10:00:00 | 335.39 | 333.63 | 0.00 | T1 1.5R @ 335.39 |
| Target hit | 2025-03-18 12:40:00 | 335.40 | 335.44 | 0.00 | Trail-exit close<VWAP |

### Cycle 23 — BUY (started 2025-03-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 09:30:00 | 370.40 | 368.73 | 0.00 | ORB-long ORB[366.80,369.65] vol=2.8x ATR=1.17 |
| Stop hit — per-position SL triggered | 2025-03-25 10:10:00 | 369.23 | 369.40 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-03-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 10:35:00 | 362.15 | 363.61 | 0.00 | ORB-short ORB[364.50,369.00] vol=3.1x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-26 11:00:00 | 360.30 | 363.20 | 0.00 | T1 1.5R @ 360.30 |
| Target hit | 2025-03-26 15:20:00 | 353.90 | 359.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — SELL (started 2025-03-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-28 11:10:00 | 360.80 | 361.44 | 0.00 | ORB-short ORB[361.05,364.30] vol=2.8x ATR=1.16 |
| Stop hit — per-position SL triggered | 2025-03-28 11:50:00 | 361.96 | 361.35 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 11:15:00 | 361.40 | 359.90 | 0.00 | ORB-long ORB[357.35,360.95] vol=2.1x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-17 11:40:00 | 362.87 | 360.20 | 0.00 | T1 1.5R @ 362.87 |
| Target hit | 2025-04-17 15:20:00 | 364.10 | 362.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2025-04-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 10:35:00 | 367.55 | 365.86 | 0.00 | ORB-long ORB[361.80,365.80] vol=2.2x ATR=0.82 |
| Stop hit — per-position SL triggered | 2025-04-21 11:05:00 | 366.73 | 366.11 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-04-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-22 09:55:00 | 361.00 | 362.36 | 0.00 | ORB-short ORB[362.05,367.05] vol=3.4x ATR=1.13 |
| Stop hit — per-position SL triggered | 2025-04-22 10:05:00 | 362.13 | 362.29 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-04-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:25:00 | 360.15 | 361.97 | 0.00 | ORB-short ORB[362.05,363.70] vol=2.5x ATR=0.83 |
| Stop hit — per-position SL triggered | 2025-04-23 10:30:00 | 360.98 | 361.82 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-04-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 10:40:00 | 354.10 | 357.86 | 0.00 | ORB-short ORB[362.55,365.80] vol=1.5x ATR=1.41 |
| Stop hit — per-position SL triggered | 2025-04-25 10:45:00 | 355.51 | 357.50 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-04-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 10:40:00 | 360.80 | 361.74 | 0.00 | ORB-short ORB[361.00,364.35] vol=1.7x ATR=0.96 |
| Stop hit — per-position SL triggered | 2025-04-29 11:00:00 | 361.76 | 361.67 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-04-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 09:35:00 | 360.60 | 358.99 | 0.00 | ORB-long ORB[355.10,359.60] vol=2.1x ATR=1.09 |
| Stop hit — per-position SL triggered | 2025-04-30 10:05:00 | 359.51 | 359.24 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-05-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-02 10:55:00 | 352.95 | 356.04 | 0.00 | ORB-short ORB[353.30,357.85] vol=2.4x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-02 11:50:00 | 351.18 | 354.98 | 0.00 | T1 1.5R @ 351.18 |
| Target hit | 2025-05-02 15:20:00 | 348.40 | 350.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — SELL (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-05 10:15:00 | 345.90 | 347.80 | 0.00 | ORB-short ORB[348.05,352.00] vol=1.5x ATR=1.07 |
| Stop hit — per-position SL triggered | 2025-05-05 10:25:00 | 346.97 | 347.67 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-05-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-06 10:35:00 | 344.95 | 345.77 | 0.00 | ORB-short ORB[346.20,349.50] vol=3.1x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 11:25:00 | 343.75 | 345.32 | 0.00 | T1 1.5R @ 343.75 |
| Target hit | 2025-05-06 15:20:00 | 342.10 | 342.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — SELL (started 2025-05-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-08 11:05:00 | 339.60 | 340.58 | 0.00 | ORB-short ORB[341.00,343.10] vol=2.4x ATR=0.73 |
| Stop hit — per-position SL triggered | 2025-05-08 11:45:00 | 340.33 | 340.45 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-05-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-09 11:00:00 | 331.30 | 334.01 | 0.00 | ORB-short ORB[334.20,338.75] vol=1.8x ATR=1.20 |
| Stop hit — per-position SL triggered | 2025-05-09 11:25:00 | 332.50 | 333.69 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-09-10 10:20:00 | 395.45 | 2024-09-10 10:40:00 | 393.98 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-09-16 09:50:00 | 407.60 | 2024-09-16 09:55:00 | 406.35 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-10-07 10:40:00 | 419.65 | 2024-10-07 10:50:00 | 417.12 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-10-07 10:40:00 | 419.65 | 2024-10-07 11:25:00 | 419.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-10 10:05:00 | 427.90 | 2024-10-10 12:00:00 | 426.39 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-10-15 10:05:00 | 426.85 | 2024-10-15 10:10:00 | 425.83 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-10-16 11:15:00 | 423.90 | 2024-10-16 11:45:00 | 422.73 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2024-10-16 11:15:00 | 423.90 | 2024-10-16 12:20:00 | 423.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-18 10:40:00 | 421.85 | 2024-10-18 11:55:00 | 424.00 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-10-18 10:40:00 | 421.85 | 2024-10-18 15:20:00 | 425.80 | TARGET_HIT | 0.50 | 0.94% |
| SELL | retest1 | 2024-10-22 10:35:00 | 419.60 | 2024-10-22 11:30:00 | 421.27 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-10-23 09:45:00 | 404.60 | 2024-10-23 09:50:00 | 406.52 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-11-27 09:50:00 | 367.90 | 2024-11-27 09:55:00 | 366.68 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-11-28 10:35:00 | 365.50 | 2024-11-28 12:35:00 | 363.82 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-11-28 10:35:00 | 365.50 | 2024-11-28 15:20:00 | 363.20 | TARGET_HIT | 0.50 | 0.63% |
| SELL | retest1 | 2024-12-12 10:45:00 | 360.20 | 2024-12-12 11:45:00 | 359.14 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-12-12 10:45:00 | 360.20 | 2024-12-12 15:20:00 | 355.75 | TARGET_HIT | 0.50 | 1.24% |
| SELL | retest1 | 2024-12-17 10:10:00 | 350.55 | 2024-12-17 10:20:00 | 349.63 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2024-12-17 10:10:00 | 350.55 | 2024-12-17 10:55:00 | 350.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-18 10:55:00 | 344.45 | 2024-12-18 11:10:00 | 345.38 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-01-17 10:45:00 | 327.30 | 2025-01-17 10:50:00 | 326.39 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-01-22 11:10:00 | 319.75 | 2025-01-22 12:05:00 | 317.98 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-01-22 11:10:00 | 319.75 | 2025-01-22 12:10:00 | 319.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-01 10:55:00 | 328.90 | 2025-02-01 11:00:00 | 327.97 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-02-06 11:10:00 | 316.65 | 2025-02-06 11:50:00 | 315.54 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-02-06 11:10:00 | 316.65 | 2025-02-06 15:20:00 | 313.65 | TARGET_HIT | 0.50 | 0.95% |
| SELL | retest1 | 2025-02-14 11:00:00 | 300.65 | 2025-02-14 11:10:00 | 301.60 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-03-07 11:05:00 | 333.60 | 2025-03-07 11:45:00 | 332.12 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-03-07 11:05:00 | 333.60 | 2025-03-07 15:20:00 | 329.55 | TARGET_HIT | 0.50 | 1.21% |
| BUY | retest1 | 2025-03-13 10:00:00 | 333.35 | 2025-03-13 10:10:00 | 332.34 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-03-18 09:55:00 | 334.25 | 2025-03-18 10:00:00 | 335.39 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-03-18 09:55:00 | 334.25 | 2025-03-18 12:40:00 | 335.40 | TARGET_HIT | 0.50 | 0.34% |
| BUY | retest1 | 2025-03-25 09:30:00 | 370.40 | 2025-03-25 10:10:00 | 369.23 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-03-26 10:35:00 | 362.15 | 2025-03-26 11:00:00 | 360.30 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-03-26 10:35:00 | 362.15 | 2025-03-26 15:20:00 | 353.90 | TARGET_HIT | 0.50 | 2.28% |
| SELL | retest1 | 2025-03-28 11:10:00 | 360.80 | 2025-03-28 11:50:00 | 361.96 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-04-17 11:15:00 | 361.40 | 2025-04-17 11:40:00 | 362.87 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-04-17 11:15:00 | 361.40 | 2025-04-17 15:20:00 | 364.10 | TARGET_HIT | 0.50 | 0.75% |
| BUY | retest1 | 2025-04-21 10:35:00 | 367.55 | 2025-04-21 11:05:00 | 366.73 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-04-22 09:55:00 | 361.00 | 2025-04-22 10:05:00 | 362.13 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-04-23 10:25:00 | 360.15 | 2025-04-23 10:30:00 | 360.98 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-04-25 10:40:00 | 354.10 | 2025-04-25 10:45:00 | 355.51 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-04-29 10:40:00 | 360.80 | 2025-04-29 11:00:00 | 361.76 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-04-30 09:35:00 | 360.60 | 2025-04-30 10:05:00 | 359.51 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-05-02 10:55:00 | 352.95 | 2025-05-02 11:50:00 | 351.18 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-05-02 10:55:00 | 352.95 | 2025-05-02 15:20:00 | 348.40 | TARGET_HIT | 0.50 | 1.29% |
| SELL | retest1 | 2025-05-05 10:15:00 | 345.90 | 2025-05-05 10:25:00 | 346.97 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-05-06 10:35:00 | 344.95 | 2025-05-06 11:25:00 | 343.75 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-05-06 10:35:00 | 344.95 | 2025-05-06 15:20:00 | 342.10 | TARGET_HIT | 0.50 | 0.83% |
| SELL | retest1 | 2025-05-08 11:05:00 | 339.60 | 2025-05-08 11:45:00 | 340.33 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-05-09 11:00:00 | 331.30 | 2025-05-09 11:25:00 | 332.50 | STOP_HIT | 1.00 | -0.36% |
