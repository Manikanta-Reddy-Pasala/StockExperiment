# Aster DM Healthcare Ltd. (ASTERDM)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 742.00
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
| ENTRY1 | 62 |
| ENTRY2 | 0 |
| PARTIAL | 22 |
| TARGET_HIT | 11 |
| STOP_HIT | 51 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 84 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 51
- **Target hits / Stop hits / Partials:** 11 / 51 / 22
- **Avg / median % per leg:** 0.09% / 0.00%
- **Sum % (uncompounded):** 7.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 42 | 14 | 33.3% | 5 | 28 | 9 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 42 | 14 | 33.3% | 5 | 28 | 9 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 42 | 19 | 45.2% | 6 | 23 | 13 | 0.18% | 7.5% |
| SELL @ 2nd Alert (retest1) | 42 | 19 | 45.2% | 6 | 23 | 13 | 0.18% | 7.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 84 | 33 | 39.3% | 11 | 51 | 22 | 0.09% | 7.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 10:35:00 | 338.65 | 341.29 | 0.00 | ORB-short ORB[342.50,345.05] vol=1.5x ATR=1.55 |
| Stop hit — per-position SL triggered | 2024-05-13 11:25:00 | 340.20 | 340.88 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 09:30:00 | 345.50 | 343.80 | 0.00 | ORB-long ORB[341.95,344.90] vol=1.9x ATR=1.14 |
| Stop hit — per-position SL triggered | 2024-05-14 09:45:00 | 344.36 | 344.13 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 09:30:00 | 353.15 | 351.17 | 0.00 | ORB-long ORB[348.90,352.30] vol=3.1x ATR=1.12 |
| Stop hit — per-position SL triggered | 2024-05-21 09:35:00 | 352.03 | 351.36 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 09:30:00 | 362.55 | 364.11 | 0.00 | ORB-short ORB[363.15,368.15] vol=2.1x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 09:35:00 | 360.22 | 363.78 | 0.00 | T1 1.5R @ 360.22 |
| Stop hit — per-position SL triggered | 2024-05-30 10:40:00 | 362.55 | 361.80 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-06-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 09:40:00 | 361.90 | 363.42 | 0.00 | ORB-short ORB[362.40,364.90] vol=2.0x ATR=1.12 |
| Stop hit — per-position SL triggered | 2024-06-13 10:15:00 | 363.02 | 363.17 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-14 09:30:00 | 358.70 | 361.28 | 0.00 | ORB-short ORB[359.00,364.15] vol=2.1x ATR=0.99 |
| Stop hit — per-position SL triggered | 2024-06-14 09:40:00 | 359.69 | 360.82 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 11:05:00 | 354.20 | 352.53 | 0.00 | ORB-long ORB[350.55,353.70] vol=1.7x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-20 11:10:00 | 355.38 | 352.72 | 0.00 | T1 1.5R @ 355.38 |
| Stop hit — per-position SL triggered | 2024-06-20 12:00:00 | 354.20 | 353.21 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 09:35:00 | 354.70 | 353.17 | 0.00 | ORB-long ORB[351.25,354.60] vol=1.6x ATR=0.99 |
| Stop hit — per-position SL triggered | 2024-06-26 10:00:00 | 353.71 | 353.63 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 09:30:00 | 348.00 | 350.45 | 0.00 | ORB-short ORB[350.05,353.95] vol=3.3x ATR=0.98 |
| Stop hit — per-position SL triggered | 2024-06-27 09:45:00 | 348.98 | 349.62 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-07-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:35:00 | 348.85 | 350.78 | 0.00 | ORB-short ORB[350.10,353.65] vol=1.6x ATR=1.06 |
| Stop hit — per-position SL triggered | 2024-07-02 09:55:00 | 349.91 | 350.19 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 10:25:00 | 345.80 | 347.04 | 0.00 | ORB-short ORB[346.25,349.90] vol=4.3x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 10:35:00 | 344.07 | 346.72 | 0.00 | T1 1.5R @ 344.07 |
| Stop hit — per-position SL triggered | 2024-07-04 11:00:00 | 345.80 | 346.59 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 09:40:00 | 342.20 | 343.75 | 0.00 | ORB-short ORB[343.05,348.00] vol=3.3x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 09:45:00 | 340.85 | 343.28 | 0.00 | T1 1.5R @ 340.85 |
| Target hit | 2024-07-05 15:20:00 | 339.70 | 341.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2024-07-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 09:50:00 | 337.20 | 338.08 | 0.00 | ORB-short ORB[338.05,340.50] vol=1.9x ATR=0.63 |
| Stop hit — per-position SL triggered | 2024-07-12 10:55:00 | 337.83 | 337.69 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-15 09:30:00 | 333.75 | 335.86 | 0.00 | ORB-short ORB[335.35,339.60] vol=3.4x ATR=1.01 |
| Stop hit — per-position SL triggered | 2024-07-15 12:30:00 | 334.76 | 334.22 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:30:00 | 331.00 | 332.17 | 0.00 | ORB-short ORB[332.10,334.95] vol=2.5x ATR=0.88 |
| Stop hit — per-position SL triggered | 2024-07-18 09:35:00 | 331.88 | 332.09 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 09:50:00 | 328.90 | 329.39 | 0.00 | ORB-short ORB[328.95,333.00] vol=1.5x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 10:00:00 | 327.37 | 327.96 | 0.00 | T1 1.5R @ 327.37 |
| Target hit | 2024-07-19 15:05:00 | 325.40 | 323.03 | 0.00 | Trail-exit close>VWAP |

### Cycle 17 — BUY (started 2024-07-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 09:55:00 | 323.80 | 322.76 | 0.00 | ORB-long ORB[319.50,323.75] vol=1.7x ATR=1.26 |
| Stop hit — per-position SL triggered | 2024-07-25 10:05:00 | 322.54 | 322.83 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-08-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 10:25:00 | 390.90 | 392.68 | 0.00 | ORB-short ORB[393.10,398.40] vol=2.9x ATR=1.34 |
| Stop hit — per-position SL triggered | 2024-08-20 10:35:00 | 392.24 | 392.60 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:55:00 | 398.40 | 394.63 | 0.00 | ORB-long ORB[388.80,394.40] vol=1.9x ATR=1.51 |
| Stop hit — per-position SL triggered | 2024-08-21 10:15:00 | 396.89 | 395.93 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 09:45:00 | 400.50 | 399.35 | 0.00 | ORB-long ORB[397.00,400.00] vol=14.6x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 10:05:00 | 402.49 | 399.46 | 0.00 | T1 1.5R @ 402.49 |
| Stop hit — per-position SL triggered | 2024-08-23 10:45:00 | 400.50 | 400.07 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 10:40:00 | 402.55 | 400.88 | 0.00 | ORB-long ORB[397.60,402.00] vol=2.1x ATR=0.98 |
| Stop hit — per-position SL triggered | 2024-08-26 10:50:00 | 401.57 | 401.00 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 09:30:00 | 407.95 | 409.26 | 0.00 | ORB-short ORB[408.10,412.30] vol=1.8x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 09:45:00 | 406.49 | 408.73 | 0.00 | T1 1.5R @ 406.49 |
| Stop hit — per-position SL triggered | 2024-08-29 11:15:00 | 407.95 | 406.23 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:50:00 | 405.90 | 403.27 | 0.00 | ORB-long ORB[401.55,405.75] vol=1.6x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 11:50:00 | 407.74 | 406.86 | 0.00 | T1 1.5R @ 407.74 |
| Target hit | 2024-08-30 13:45:00 | 407.15 | 407.31 | 0.00 | Trail-exit close<VWAP |

### Cycle 24 — BUY (started 2024-09-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 11:00:00 | 407.25 | 405.72 | 0.00 | ORB-long ORB[401.35,407.15] vol=3.8x ATR=1.16 |
| Stop hit — per-position SL triggered | 2024-09-03 12:50:00 | 406.09 | 406.14 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-09-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-09 09:30:00 | 408.15 | 410.35 | 0.00 | ORB-short ORB[410.00,415.00] vol=1.5x ATR=1.65 |
| Stop hit — per-position SL triggered | 2024-09-09 09:35:00 | 409.80 | 410.48 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 11:10:00 | 413.00 | 409.60 | 0.00 | ORB-long ORB[408.25,412.85] vol=4.4x ATR=1.21 |
| Stop hit — per-position SL triggered | 2024-09-10 11:20:00 | 411.79 | 409.89 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-09-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-12 09:30:00 | 416.95 | 415.36 | 0.00 | ORB-long ORB[409.75,413.10] vol=9.0x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-12 09:35:00 | 419.59 | 417.38 | 0.00 | T1 1.5R @ 419.59 |
| Stop hit — per-position SL triggered | 2024-09-12 09:45:00 | 416.95 | 417.86 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-09-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-13 11:05:00 | 417.75 | 419.18 | 0.00 | ORB-short ORB[418.15,420.75] vol=2.2x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-13 12:30:00 | 416.33 | 418.87 | 0.00 | T1 1.5R @ 416.33 |
| Stop hit — per-position SL triggered | 2024-09-13 12:50:00 | 417.75 | 418.48 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-09-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 11:00:00 | 409.25 | 413.95 | 0.00 | ORB-short ORB[412.70,418.00] vol=2.3x ATR=1.31 |
| Stop hit — per-position SL triggered | 2024-09-17 11:15:00 | 410.56 | 413.53 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 10:10:00 | 429.90 | 427.79 | 0.00 | ORB-long ORB[423.85,429.70] vol=2.3x ATR=1.69 |
| Stop hit — per-position SL triggered | 2024-09-23 10:25:00 | 428.21 | 428.26 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 09:40:00 | 421.55 | 422.57 | 0.00 | ORB-short ORB[422.15,424.85] vol=2.3x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 11:05:00 | 419.85 | 421.70 | 0.00 | T1 1.5R @ 419.85 |
| Target hit | 2024-09-24 13:05:00 | 421.30 | 420.71 | 0.00 | Trail-exit close>VWAP |

### Cycle 32 — BUY (started 2024-09-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 09:40:00 | 423.05 | 421.51 | 0.00 | ORB-long ORB[419.40,422.95] vol=1.6x ATR=1.48 |
| Stop hit — per-position SL triggered | 2024-09-25 09:45:00 | 421.57 | 421.54 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 11:15:00 | 414.70 | 416.61 | 0.00 | ORB-short ORB[416.80,421.05] vol=2.6x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-26 11:25:00 | 413.43 | 415.08 | 0.00 | T1 1.5R @ 413.43 |
| Target hit | 2024-09-26 15:20:00 | 411.25 | 412.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — BUY (started 2024-10-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 11:00:00 | 418.10 | 415.46 | 0.00 | ORB-long ORB[411.80,416.10] vol=2.8x ATR=1.10 |
| Stop hit — per-position SL triggered | 2024-10-01 11:15:00 | 417.00 | 415.59 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-10-03 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-03 09:45:00 | 420.70 | 416.19 | 0.00 | ORB-long ORB[411.00,416.75] vol=2.4x ATR=1.86 |
| Stop hit — per-position SL triggered | 2024-10-03 13:25:00 | 418.84 | 419.15 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-10-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 09:30:00 | 409.00 | 406.42 | 0.00 | ORB-long ORB[402.55,407.85] vol=2.3x ATR=2.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 09:55:00 | 412.48 | 408.77 | 0.00 | T1 1.5R @ 412.48 |
| Target hit | 2024-10-08 15:20:00 | 418.70 | 415.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — BUY (started 2024-10-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 10:10:00 | 424.60 | 423.60 | 0.00 | ORB-long ORB[419.80,424.55] vol=2.9x ATR=1.37 |
| Stop hit — per-position SL triggered | 2024-10-10 10:25:00 | 423.23 | 423.63 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-10-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 09:35:00 | 415.85 | 416.62 | 0.00 | ORB-short ORB[416.30,419.20] vol=3.3x ATR=1.23 |
| Stop hit — per-position SL triggered | 2024-10-11 09:45:00 | 417.08 | 417.03 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-10-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 09:40:00 | 429.05 | 426.36 | 0.00 | ORB-long ORB[420.80,427.00] vol=2.4x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 09:50:00 | 431.32 | 430.25 | 0.00 | T1 1.5R @ 431.32 |
| Target hit | 2024-10-16 10:05:00 | 429.80 | 430.90 | 0.00 | Trail-exit close<VWAP |

### Cycle 40 — SELL (started 2024-10-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:55:00 | 418.50 | 422.45 | 0.00 | ORB-short ORB[423.15,428.80] vol=2.0x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 11:25:00 | 415.86 | 421.15 | 0.00 | T1 1.5R @ 415.86 |
| Target hit | 2024-10-22 15:20:00 | 403.95 | 411.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — SELL (started 2024-10-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 09:45:00 | 435.35 | 437.70 | 0.00 | ORB-short ORB[435.75,441.70] vol=2.2x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 10:20:00 | 432.60 | 436.30 | 0.00 | T1 1.5R @ 432.60 |
| Target hit | 2024-10-29 15:20:00 | 431.40 | 432.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — SELL (started 2024-11-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 10:50:00 | 438.90 | 441.75 | 0.00 | ORB-short ORB[440.40,445.50] vol=1.7x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-07 11:05:00 | 436.88 | 440.73 | 0.00 | T1 1.5R @ 436.88 |
| Stop hit — per-position SL triggered | 2024-11-07 11:10:00 | 438.90 | 440.71 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 10:15:00 | 439.30 | 438.40 | 0.00 | ORB-long ORB[434.00,438.90] vol=3.2x ATR=1.04 |
| Stop hit — per-position SL triggered | 2024-11-08 10:25:00 | 438.26 | 438.44 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-11-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-12 09:35:00 | 442.25 | 440.62 | 0.00 | ORB-long ORB[437.75,441.20] vol=2.0x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 09:40:00 | 444.01 | 442.68 | 0.00 | T1 1.5R @ 444.01 |
| Stop hit — per-position SL triggered | 2024-11-12 09:45:00 | 442.25 | 442.65 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-11-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:35:00 | 432.90 | 436.61 | 0.00 | ORB-short ORB[435.80,441.00] vol=2.2x ATR=2.15 |
| Stop hit — per-position SL triggered | 2024-11-13 10:30:00 | 435.05 | 434.20 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-11-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 10:10:00 | 439.10 | 436.43 | 0.00 | ORB-long ORB[432.00,436.50] vol=4.6x ATR=1.80 |
| Stop hit — per-position SL triggered | 2024-11-19 11:00:00 | 437.30 | 437.00 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-12-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-11 10:50:00 | 488.50 | 486.77 | 0.00 | ORB-long ORB[483.25,488.05] vol=3.6x ATR=1.10 |
| Stop hit — per-position SL triggered | 2024-12-11 10:55:00 | 487.40 | 486.80 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-12-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 09:35:00 | 476.40 | 478.28 | 0.00 | ORB-short ORB[478.85,482.45] vol=3.3x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 09:45:00 | 474.48 | 477.14 | 0.00 | T1 1.5R @ 474.48 |
| Stop hit — per-position SL triggered | 2024-12-13 09:50:00 | 476.40 | 477.37 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-12-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 11:10:00 | 477.35 | 481.98 | 0.00 | ORB-short ORB[478.50,484.60] vol=1.5x ATR=1.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 11:40:00 | 475.49 | 480.51 | 0.00 | T1 1.5R @ 475.49 |
| Stop hit — per-position SL triggered | 2024-12-16 14:40:00 | 477.35 | 477.96 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-12-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 10:20:00 | 497.75 | 491.82 | 0.00 | ORB-long ORB[487.00,490.45] vol=3.7x ATR=1.87 |
| Stop hit — per-position SL triggered | 2024-12-18 10:25:00 | 495.88 | 492.37 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-12-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 09:45:00 | 509.65 | 507.02 | 0.00 | ORB-long ORB[503.35,509.00] vol=1.7x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 10:05:00 | 512.06 | 509.05 | 0.00 | T1 1.5R @ 512.06 |
| Target hit | 2024-12-27 10:20:00 | 510.15 | 511.06 | 0.00 | Trail-exit close<VWAP |

### Cycle 52 — BUY (started 2024-12-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 10:00:00 | 515.45 | 510.97 | 0.00 | ORB-long ORB[507.05,513.85] vol=1.8x ATR=1.97 |
| Stop hit — per-position SL triggered | 2024-12-30 10:05:00 | 513.48 | 511.60 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-01-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 11:10:00 | 502.45 | 497.02 | 0.00 | ORB-long ORB[492.00,497.20] vol=6.1x ATR=1.42 |
| Stop hit — per-position SL triggered | 2025-01-16 11:15:00 | 501.03 | 498.94 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-17 11:15:00 | 492.55 | 494.90 | 0.00 | ORB-short ORB[493.25,497.75] vol=2.1x ATR=1.40 |
| Stop hit — per-position SL triggered | 2025-01-17 11:25:00 | 493.95 | 494.79 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-01-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-21 10:55:00 | 498.55 | 496.90 | 0.00 | ORB-long ORB[493.15,498.30] vol=1.5x ATR=1.10 |
| Stop hit — per-position SL triggered | 2025-01-21 11:00:00 | 497.45 | 496.95 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-01-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:30:00 | 501.00 | 500.59 | 0.00 | ORB-long ORB[494.00,500.05] vol=14.9x ATR=1.57 |
| Stop hit — per-position SL triggered | 2025-01-23 10:50:00 | 499.43 | 500.55 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-02-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 10:10:00 | 480.45 | 478.00 | 0.00 | ORB-long ORB[474.90,479.85] vol=1.5x ATR=1.68 |
| Stop hit — per-position SL triggered | 2025-02-05 10:15:00 | 478.77 | 478.22 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-02-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 09:55:00 | 420.25 | 417.01 | 0.00 | ORB-long ORB[414.80,418.10] vol=1.6x ATR=1.96 |
| Stop hit — per-position SL triggered | 2025-02-20 10:05:00 | 418.29 | 417.18 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2025-03-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-04 10:50:00 | 395.95 | 394.32 | 0.00 | ORB-long ORB[387.10,392.95] vol=3.0x ATR=1.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-04 11:40:00 | 398.73 | 394.72 | 0.00 | T1 1.5R @ 398.73 |
| Target hit | 2025-03-04 13:35:00 | 396.65 | 396.84 | 0.00 | Trail-exit close<VWAP |

### Cycle 60 — BUY (started 2025-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 10:15:00 | 512.45 | 507.05 | 0.00 | ORB-long ORB[500.10,507.15] vol=1.7x ATR=2.12 |
| Stop hit — per-position SL triggered | 2025-04-24 10:25:00 | 510.33 | 507.63 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-04-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 10:00:00 | 501.00 | 506.44 | 0.00 | ORB-short ORB[510.00,515.50] vol=1.6x ATR=2.50 |
| Stop hit — per-position SL triggered | 2025-04-25 10:10:00 | 503.50 | 506.07 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-04-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-30 10:00:00 | 503.20 | 505.36 | 0.00 | ORB-short ORB[504.05,511.15] vol=1.7x ATR=2.23 |
| Stop hit — per-position SL triggered | 2025-04-30 10:20:00 | 505.43 | 505.04 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-13 10:35:00 | 338.65 | 2024-05-13 11:25:00 | 340.20 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-05-14 09:30:00 | 345.50 | 2024-05-14 09:45:00 | 344.36 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-05-21 09:30:00 | 353.15 | 2024-05-21 09:35:00 | 352.03 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-05-30 09:30:00 | 362.55 | 2024-05-30 09:35:00 | 360.22 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-05-30 09:30:00 | 362.55 | 2024-05-30 10:40:00 | 362.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-13 09:40:00 | 361.90 | 2024-06-13 10:15:00 | 363.02 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-06-14 09:30:00 | 358.70 | 2024-06-14 09:40:00 | 359.69 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-06-20 11:05:00 | 354.20 | 2024-06-20 11:10:00 | 355.38 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2024-06-20 11:05:00 | 354.20 | 2024-06-20 12:00:00 | 354.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-26 09:35:00 | 354.70 | 2024-06-26 10:00:00 | 353.71 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-06-27 09:30:00 | 348.00 | 2024-06-27 09:45:00 | 348.98 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-07-02 09:35:00 | 348.85 | 2024-07-02 09:55:00 | 349.91 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-07-04 10:25:00 | 345.80 | 2024-07-04 10:35:00 | 344.07 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-07-04 10:25:00 | 345.80 | 2024-07-04 11:00:00 | 345.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-05 09:40:00 | 342.20 | 2024-07-05 09:45:00 | 340.85 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-07-05 09:40:00 | 342.20 | 2024-07-05 15:20:00 | 339.70 | TARGET_HIT | 0.50 | 0.73% |
| SELL | retest1 | 2024-07-12 09:50:00 | 337.20 | 2024-07-12 10:55:00 | 337.83 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2024-07-15 09:30:00 | 333.75 | 2024-07-15 12:30:00 | 334.76 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-07-18 09:30:00 | 331.00 | 2024-07-18 09:35:00 | 331.88 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-07-19 09:50:00 | 328.90 | 2024-07-19 10:00:00 | 327.37 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-07-19 09:50:00 | 328.90 | 2024-07-19 15:05:00 | 325.40 | TARGET_HIT | 0.50 | 1.06% |
| BUY | retest1 | 2024-07-25 09:55:00 | 323.80 | 2024-07-25 10:05:00 | 322.54 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-08-20 10:25:00 | 390.90 | 2024-08-20 10:35:00 | 392.24 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-08-21 09:55:00 | 398.40 | 2024-08-21 10:15:00 | 396.89 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-08-23 09:45:00 | 400.50 | 2024-08-23 10:05:00 | 402.49 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-08-23 09:45:00 | 400.50 | 2024-08-23 10:45:00 | 400.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-26 10:40:00 | 402.55 | 2024-08-26 10:50:00 | 401.57 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-08-29 09:30:00 | 407.95 | 2024-08-29 09:45:00 | 406.49 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2024-08-29 09:30:00 | 407.95 | 2024-08-29 11:15:00 | 407.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-30 10:50:00 | 405.90 | 2024-08-30 11:50:00 | 407.74 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-08-30 10:50:00 | 405.90 | 2024-08-30 13:45:00 | 407.15 | TARGET_HIT | 0.50 | 0.31% |
| BUY | retest1 | 2024-09-03 11:00:00 | 407.25 | 2024-09-03 12:50:00 | 406.09 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-09-09 09:30:00 | 408.15 | 2024-09-09 09:35:00 | 409.80 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-09-10 11:10:00 | 413.00 | 2024-09-10 11:20:00 | 411.79 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-09-12 09:30:00 | 416.95 | 2024-09-12 09:35:00 | 419.59 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-09-12 09:30:00 | 416.95 | 2024-09-12 09:45:00 | 416.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-13 11:05:00 | 417.75 | 2024-09-13 12:30:00 | 416.33 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-09-13 11:05:00 | 417.75 | 2024-09-13 12:50:00 | 417.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-17 11:00:00 | 409.25 | 2024-09-17 11:15:00 | 410.56 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-09-23 10:10:00 | 429.90 | 2024-09-23 10:25:00 | 428.21 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-09-24 09:40:00 | 421.55 | 2024-09-24 11:05:00 | 419.85 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-09-24 09:40:00 | 421.55 | 2024-09-24 13:05:00 | 421.30 | TARGET_HIT | 0.50 | 0.06% |
| BUY | retest1 | 2024-09-25 09:40:00 | 423.05 | 2024-09-25 09:45:00 | 421.57 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-09-26 11:15:00 | 414.70 | 2024-09-26 11:25:00 | 413.43 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2024-09-26 11:15:00 | 414.70 | 2024-09-26 15:20:00 | 411.25 | TARGET_HIT | 0.50 | 0.83% |
| BUY | retest1 | 2024-10-01 11:00:00 | 418.10 | 2024-10-01 11:15:00 | 417.00 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-10-03 09:45:00 | 420.70 | 2024-10-03 13:25:00 | 418.84 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-10-08 09:30:00 | 409.00 | 2024-10-08 09:55:00 | 412.48 | PARTIAL | 0.50 | 0.85% |
| BUY | retest1 | 2024-10-08 09:30:00 | 409.00 | 2024-10-08 15:20:00 | 418.70 | TARGET_HIT | 0.50 | 2.37% |
| BUY | retest1 | 2024-10-10 10:10:00 | 424.60 | 2024-10-10 10:25:00 | 423.23 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-10-11 09:35:00 | 415.85 | 2024-10-11 09:45:00 | 417.08 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-10-16 09:40:00 | 429.05 | 2024-10-16 09:50:00 | 431.32 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-10-16 09:40:00 | 429.05 | 2024-10-16 10:05:00 | 429.80 | TARGET_HIT | 0.50 | 0.17% |
| SELL | retest1 | 2024-10-22 10:55:00 | 418.50 | 2024-10-22 11:25:00 | 415.86 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-10-22 10:55:00 | 418.50 | 2024-10-22 15:20:00 | 403.95 | TARGET_HIT | 0.50 | 3.48% |
| SELL | retest1 | 2024-10-29 09:45:00 | 435.35 | 2024-10-29 10:20:00 | 432.60 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-10-29 09:45:00 | 435.35 | 2024-10-29 15:20:00 | 431.40 | TARGET_HIT | 0.50 | 0.91% |
| SELL | retest1 | 2024-11-07 10:50:00 | 438.90 | 2024-11-07 11:05:00 | 436.88 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-11-07 10:50:00 | 438.90 | 2024-11-07 11:10:00 | 438.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-08 10:15:00 | 439.30 | 2024-11-08 10:25:00 | 438.26 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-11-12 09:35:00 | 442.25 | 2024-11-12 09:40:00 | 444.01 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-11-12 09:35:00 | 442.25 | 2024-11-12 09:45:00 | 442.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-13 09:35:00 | 432.90 | 2024-11-13 10:30:00 | 435.05 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-11-19 10:10:00 | 439.10 | 2024-11-19 11:00:00 | 437.30 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-12-11 10:50:00 | 488.50 | 2024-12-11 10:55:00 | 487.40 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-13 09:35:00 | 476.40 | 2024-12-13 09:45:00 | 474.48 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-12-13 09:35:00 | 476.40 | 2024-12-13 09:50:00 | 476.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-16 11:10:00 | 477.35 | 2024-12-16 11:40:00 | 475.49 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-12-16 11:10:00 | 477.35 | 2024-12-16 14:40:00 | 477.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-18 10:20:00 | 497.75 | 2024-12-18 10:25:00 | 495.88 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-12-27 09:45:00 | 509.65 | 2024-12-27 10:05:00 | 512.06 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-12-27 09:45:00 | 509.65 | 2024-12-27 10:20:00 | 510.15 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2024-12-30 10:00:00 | 515.45 | 2024-12-30 10:05:00 | 513.48 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-01-16 11:10:00 | 502.45 | 2025-01-16 11:15:00 | 501.03 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-01-17 11:15:00 | 492.55 | 2025-01-17 11:25:00 | 493.95 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-01-21 10:55:00 | 498.55 | 2025-01-21 11:00:00 | 497.45 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-01-23 10:30:00 | 501.00 | 2025-01-23 10:50:00 | 499.43 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-02-05 10:10:00 | 480.45 | 2025-02-05 10:15:00 | 478.77 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-02-20 09:55:00 | 420.25 | 2025-02-20 10:05:00 | 418.29 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-03-04 10:50:00 | 395.95 | 2025-03-04 11:40:00 | 398.73 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2025-03-04 10:50:00 | 395.95 | 2025-03-04 13:35:00 | 396.65 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2025-04-24 10:15:00 | 512.45 | 2025-04-24 10:25:00 | 510.33 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-04-25 10:00:00 | 501.00 | 2025-04-25 10:10:00 | 503.50 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2025-04-30 10:00:00 | 503.20 | 2025-04-30 10:20:00 | 505.43 | STOP_HIT | 1.00 | -0.44% |
