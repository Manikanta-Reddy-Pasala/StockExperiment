# Zydus Wellness Ltd. (ZYDUSWELL)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-05-05 15:25:00 (18108 bars)
- **Last close:** 344.00
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
| ENTRY1 | 65 |
| ENTRY2 | 0 |
| PARTIAL | 25 |
| TARGET_HIT | 16 |
| STOP_HIT | 49 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 90 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 49
- **Target hits / Stop hits / Partials:** 16 / 49 / 25
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 15.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 42 | 19 | 45.2% | 9 | 23 | 10 | 0.19% | 7.9% |
| BUY @ 2nd Alert (retest1) | 42 | 19 | 45.2% | 9 | 23 | 10 | 0.19% | 7.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 48 | 22 | 45.8% | 7 | 26 | 15 | 0.15% | 7.1% |
| SELL @ 2nd Alert (retest1) | 48 | 22 | 45.8% | 7 | 26 | 15 | 0.15% | 7.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 90 | 41 | 45.6% | 16 | 49 | 25 | 0.17% | 15.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 09:40:00 | 339.99 | 338.30 | 0.00 | ORB-long ORB[334.61,339.27] vol=1.5x ATR=1.02 |
| Stop hit — per-position SL triggered | 2024-05-14 09:55:00 | 338.97 | 338.45 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 10:35:00 | 349.15 | 351.07 | 0.00 | ORB-short ORB[351.00,353.36] vol=1.7x ATR=1.09 |
| Stop hit — per-position SL triggered | 2024-05-16 11:55:00 | 350.24 | 350.71 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 09:30:00 | 345.03 | 345.50 | 0.00 | ORB-short ORB[345.07,348.18] vol=1.7x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 09:50:00 | 343.76 | 344.81 | 0.00 | T1 1.5R @ 343.76 |
| Stop hit — per-position SL triggered | 2024-05-23 10:45:00 | 345.03 | 344.45 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 09:50:00 | 354.97 | 352.33 | 0.00 | ORB-long ORB[348.91,352.71] vol=5.7x ATR=1.45 |
| Stop hit — per-position SL triggered | 2024-05-29 10:05:00 | 353.52 | 352.56 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 10:00:00 | 345.61 | 345.64 | 0.00 | ORB-short ORB[345.89,348.96] vol=3.0x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 10:40:00 | 343.43 | 345.34 | 0.00 | T1 1.5R @ 343.43 |
| Stop hit — per-position SL triggered | 2024-05-31 11:15:00 | 345.61 | 345.28 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-05 09:30:00 | 351.58 | 347.27 | 0.00 | ORB-long ORB[344.05,347.40] vol=1.8x ATR=2.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-05 09:40:00 | 355.04 | 353.60 | 0.00 | T1 1.5R @ 355.04 |
| Target hit | 2024-06-05 10:10:00 | 360.01 | 360.38 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — SELL (started 2024-06-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 10:20:00 | 368.00 | 370.23 | 0.00 | ORB-short ORB[369.06,372.98] vol=1.6x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-19 11:00:00 | 365.97 | 369.06 | 0.00 | T1 1.5R @ 365.97 |
| Target hit | 2024-06-19 12:05:00 | 367.93 | 367.92 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — SELL (started 2024-06-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 10:40:00 | 370.57 | 373.06 | 0.00 | ORB-short ORB[374.41,379.80] vol=3.1x ATR=1.26 |
| Stop hit — per-position SL triggered | 2024-06-21 11:15:00 | 371.83 | 372.74 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 09:30:00 | 373.35 | 374.60 | 0.00 | ORB-short ORB[373.52,375.81] vol=2.2x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 09:45:00 | 371.61 | 374.01 | 0.00 | T1 1.5R @ 371.61 |
| Stop hit — per-position SL triggered | 2024-06-25 09:55:00 | 373.35 | 373.94 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 11:00:00 | 365.00 | 366.85 | 0.00 | ORB-short ORB[366.00,370.75] vol=3.0x ATR=0.90 |
| Stop hit — per-position SL triggered | 2024-06-26 11:10:00 | 365.90 | 366.84 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 09:30:00 | 418.59 | 415.57 | 0.00 | ORB-long ORB[410.00,415.14] vol=3.7x ATR=1.87 |
| Stop hit — per-position SL triggered | 2024-07-09 09:35:00 | 416.72 | 415.79 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 11:10:00 | 433.48 | 429.58 | 0.00 | ORB-long ORB[428.00,432.61] vol=3.1x ATR=1.62 |
| Stop hit — per-position SL triggered | 2024-07-15 14:15:00 | 431.86 | 430.59 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-18 09:40:00 | 432.69 | 431.31 | 0.00 | ORB-long ORB[426.00,432.00] vol=2.4x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 09:45:00 | 435.14 | 434.50 | 0.00 | T1 1.5R @ 435.14 |
| Target hit | 2024-07-18 09:55:00 | 434.91 | 437.08 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — BUY (started 2024-07-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 10:10:00 | 480.00 | 476.79 | 0.00 | ORB-long ORB[474.23,477.57] vol=1.8x ATR=1.34 |
| Stop hit — per-position SL triggered | 2024-07-29 10:15:00 | 478.66 | 477.23 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 10:15:00 | 485.00 | 481.52 | 0.00 | ORB-long ORB[478.92,484.42] vol=3.4x ATR=2.38 |
| Stop hit — per-position SL triggered | 2024-08-01 10:20:00 | 482.62 | 481.58 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-08-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-09 10:00:00 | 460.75 | 461.96 | 0.00 | ORB-short ORB[461.00,463.80] vol=2.9x ATR=1.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-09 10:25:00 | 458.64 | 461.60 | 0.00 | T1 1.5R @ 458.64 |
| Stop hit — per-position SL triggered | 2024-08-09 10:50:00 | 460.75 | 460.20 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-08-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:35:00 | 446.82 | 445.63 | 0.00 | ORB-long ORB[444.01,446.13] vol=4.7x ATR=1.38 |
| Stop hit — per-position SL triggered | 2024-08-16 09:40:00 | 445.44 | 445.64 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-08-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-20 09:45:00 | 441.61 | 442.94 | 0.00 | ORB-short ORB[442.00,444.57] vol=1.9x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 09:50:00 | 439.39 | 442.53 | 0.00 | T1 1.5R @ 439.39 |
| Stop hit — per-position SL triggered | 2024-08-20 10:20:00 | 441.61 | 441.67 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-08-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-21 10:10:00 | 454.94 | 455.49 | 0.00 | ORB-short ORB[456.01,460.13] vol=6.4x ATR=2.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 11:35:00 | 451.73 | 454.73 | 0.00 | T1 1.5R @ 451.73 |
| Stop hit — per-position SL triggered | 2024-08-21 11:45:00 | 454.94 | 454.79 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-08-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 09:35:00 | 453.60 | 456.39 | 0.00 | ORB-short ORB[455.20,459.00] vol=2.8x ATR=1.56 |
| Stop hit — per-position SL triggered | 2024-08-26 10:00:00 | 455.16 | 455.42 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 448.94 | 452.40 | 0.00 | ORB-short ORB[452.07,456.20] vol=3.1x ATR=1.88 |
| Stop hit — per-position SL triggered | 2024-08-28 09:40:00 | 450.82 | 451.06 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:25:00 | 453.82 | 456.49 | 0.00 | ORB-short ORB[456.80,463.59] vol=2.6x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 10:55:00 | 451.73 | 455.57 | 0.00 | T1 1.5R @ 451.73 |
| Stop hit — per-position SL triggered | 2024-08-29 12:25:00 | 453.82 | 454.25 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 09:35:00 | 455.39 | 453.23 | 0.00 | ORB-long ORB[450.61,455.37] vol=1.7x ATR=1.80 |
| Stop hit — per-position SL triggered | 2024-08-30 09:45:00 | 453.59 | 453.35 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-09-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-04 09:30:00 | 440.74 | 442.40 | 0.00 | ORB-short ORB[441.00,446.55] vol=1.8x ATR=1.48 |
| Stop hit — per-position SL triggered | 2024-09-04 09:35:00 | 442.22 | 442.46 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-09-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-06 09:35:00 | 455.05 | 453.42 | 0.00 | ORB-long ORB[450.00,453.79] vol=3.7x ATR=1.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 10:25:00 | 457.66 | 454.53 | 0.00 | T1 1.5R @ 457.66 |
| Target hit | 2024-09-06 12:15:00 | 455.21 | 455.62 | 0.00 | Trail-exit close<VWAP |

### Cycle 26 — SELL (started 2024-09-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 09:30:00 | 439.55 | 440.72 | 0.00 | ORB-short ORB[440.20,443.78] vol=2.3x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 10:10:00 | 437.79 | 439.68 | 0.00 | T1 1.5R @ 437.79 |
| Target hit | 2024-09-17 15:20:00 | 434.60 | 435.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — SELL (started 2024-09-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 11:00:00 | 429.77 | 432.19 | 0.00 | ORB-short ORB[431.20,436.00] vol=3.8x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 11:05:00 | 427.85 | 431.01 | 0.00 | T1 1.5R @ 427.85 |
| Stop hit — per-position SL triggered | 2024-09-18 11:10:00 | 429.77 | 431.00 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-09-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:50:00 | 426.00 | 427.95 | 0.00 | ORB-short ORB[427.65,430.31] vol=4.0x ATR=1.27 |
| Stop hit — per-position SL triggered | 2024-09-19 09:55:00 | 427.27 | 428.32 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-09-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-20 10:30:00 | 413.49 | 416.04 | 0.00 | ORB-short ORB[415.26,419.94] vol=2.1x ATR=1.88 |
| Stop hit — per-position SL triggered | 2024-09-20 11:00:00 | 415.37 | 415.37 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-09-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 09:40:00 | 399.17 | 401.00 | 0.00 | ORB-short ORB[400.08,404.71] vol=2.8x ATR=1.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 10:05:00 | 397.06 | 399.96 | 0.00 | T1 1.5R @ 397.06 |
| Target hit | 2024-09-25 13:30:00 | 397.59 | 396.93 | 0.00 | Trail-exit close>VWAP |

### Cycle 31 — BUY (started 2024-09-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 09:30:00 | 400.42 | 399.24 | 0.00 | ORB-long ORB[396.38,399.33] vol=3.3x ATR=1.28 |
| Stop hit — per-position SL triggered | 2024-09-27 09:40:00 | 399.14 | 399.36 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-10-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 10:10:00 | 389.61 | 388.47 | 0.00 | ORB-long ORB[386.22,388.79] vol=3.3x ATR=1.50 |
| Target hit | 2024-10-09 15:20:00 | 390.41 | 389.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — BUY (started 2024-10-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 10:05:00 | 403.20 | 400.56 | 0.00 | ORB-long ORB[398.12,403.00] vol=1.5x ATR=2.29 |
| Stop hit — per-position SL triggered | 2024-10-16 10:30:00 | 400.91 | 400.71 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-10-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 10:30:00 | 397.69 | 398.82 | 0.00 | ORB-short ORB[398.50,401.79] vol=19.3x ATR=1.60 |
| Stop hit — per-position SL triggered | 2024-10-17 10:35:00 | 399.29 | 398.82 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-10-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:30:00 | 363.09 | 364.87 | 0.00 | ORB-short ORB[364.77,368.36] vol=1.8x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:40:00 | 360.74 | 363.72 | 0.00 | T1 1.5R @ 360.74 |
| Target hit | 2024-10-25 14:25:00 | 359.31 | 359.12 | 0.00 | Trail-exit close>VWAP |

### Cycle 36 — SELL (started 2024-11-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-21 10:40:00 | 382.60 | 385.77 | 0.00 | ORB-short ORB[384.00,388.99] vol=2.6x ATR=1.69 |
| Stop hit — per-position SL triggered | 2024-11-21 11:25:00 | 384.29 | 385.28 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-11-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 10:40:00 | 402.00 | 401.87 | 0.00 | ORB-long ORB[397.26,401.52] vol=2.4x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 11:00:00 | 403.90 | 402.02 | 0.00 | T1 1.5R @ 403.90 |
| Stop hit — per-position SL triggered | 2024-11-28 12:40:00 | 402.00 | 403.58 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-12-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 09:50:00 | 415.23 | 413.11 | 0.00 | ORB-long ORB[409.21,414.06] vol=2.5x ATR=2.03 |
| Stop hit — per-position SL triggered | 2024-12-03 10:00:00 | 413.20 | 413.27 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-12-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-06 09:40:00 | 409.73 | 411.25 | 0.00 | ORB-short ORB[410.25,414.24] vol=2.1x ATR=1.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 09:55:00 | 407.69 | 410.73 | 0.00 | T1 1.5R @ 407.69 |
| Target hit | 2024-12-06 11:55:00 | 409.15 | 409.06 | 0.00 | Trail-exit close>VWAP |

### Cycle 40 — BUY (started 2024-12-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 09:35:00 | 412.64 | 409.29 | 0.00 | ORB-long ORB[405.53,409.78] vol=2.3x ATR=1.28 |
| Stop hit — per-position SL triggered | 2024-12-10 09:45:00 | 411.36 | 410.02 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-12-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:30:00 | 400.02 | 402.77 | 0.00 | ORB-short ORB[402.64,406.49] vol=3.8x ATR=1.27 |
| Stop hit — per-position SL triggered | 2024-12-13 10:50:00 | 401.29 | 402.35 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-12-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:55:00 | 380.31 | 382.22 | 0.00 | ORB-short ORB[382.00,385.79] vol=1.7x ATR=1.09 |
| Stop hit — per-position SL triggered | 2024-12-26 11:15:00 | 381.40 | 381.08 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-12-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 10:45:00 | 378.79 | 379.44 | 0.00 | ORB-short ORB[379.09,380.56] vol=2.0x ATR=0.47 |
| Stop hit — per-position SL triggered | 2024-12-27 11:10:00 | 379.26 | 379.51 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-12-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 10:30:00 | 379.93 | 378.46 | 0.00 | ORB-long ORB[375.13,378.71] vol=2.3x ATR=1.07 |
| Stop hit — per-position SL triggered | 2024-12-30 10:35:00 | 378.86 | 378.48 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-12-31 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-31 10:40:00 | 382.26 | 379.46 | 0.00 | ORB-long ORB[378.41,381.80] vol=2.2x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-31 10:50:00 | 384.33 | 380.50 | 0.00 | T1 1.5R @ 384.33 |
| Target hit | 2024-12-31 15:20:00 | 390.54 | 389.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — BUY (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 11:15:00 | 397.34 | 395.00 | 0.00 | ORB-long ORB[394.08,397.30] vol=8.5x ATR=1.08 |
| Stop hit — per-position SL triggered | 2025-01-02 11:20:00 | 396.26 | 395.19 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-01-15 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-15 10:40:00 | 378.28 | 375.04 | 0.00 | ORB-long ORB[371.28,374.92] vol=4.9x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-15 12:50:00 | 380.53 | 377.34 | 0.00 | T1 1.5R @ 380.53 |
| Target hit | 2025-01-15 13:25:00 | 380.00 | 380.14 | 0.00 | Trail-exit close<VWAP |

### Cycle 48 — BUY (started 2025-01-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 10:40:00 | 383.59 | 381.57 | 0.00 | ORB-long ORB[379.53,381.98] vol=1.5x ATR=1.01 |
| Stop hit — per-position SL triggered | 2025-01-16 10:45:00 | 382.58 | 381.64 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-01-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-17 09:45:00 | 379.10 | 379.78 | 0.00 | ORB-short ORB[379.63,384.43] vol=7.4x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 11:05:00 | 377.44 | 379.25 | 0.00 | T1 1.5R @ 377.44 |
| Target hit | 2025-01-17 15:20:00 | 375.00 | 377.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — SELL (started 2025-01-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-20 10:00:00 | 371.21 | 372.42 | 0.00 | ORB-short ORB[372.36,377.70] vol=1.9x ATR=1.11 |
| Stop hit — per-position SL triggered | 2025-01-20 10:30:00 | 372.32 | 372.28 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-01-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 11:10:00 | 372.22 | 377.28 | 0.00 | ORB-short ORB[378.18,383.44] vol=1.8x ATR=1.08 |
| Stop hit — per-position SL triggered | 2025-01-21 11:35:00 | 373.30 | 376.99 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-01-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-22 11:00:00 | 365.00 | 368.30 | 0.00 | ORB-short ORB[367.94,372.59] vol=15.1x ATR=1.67 |
| Stop hit — per-position SL triggered | 2025-01-22 11:05:00 | 366.67 | 367.99 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-01-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 10:50:00 | 364.67 | 359.94 | 0.00 | ORB-long ORB[351.44,356.27] vol=2.6x ATR=1.25 |
| Stop hit — per-position SL triggered | 2025-01-29 10:55:00 | 363.42 | 360.16 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-02-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-17 10:55:00 | 334.29 | 331.49 | 0.00 | ORB-long ORB[329.00,333.57] vol=3.4x ATR=1.24 |
| Stop hit — per-position SL triggered | 2025-02-17 11:10:00 | 333.05 | 332.12 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-19 09:30:00 | 330.04 | 327.62 | 0.00 | ORB-long ORB[325.68,328.48] vol=1.9x ATR=1.19 |
| Stop hit — per-position SL triggered | 2025-02-19 09:35:00 | 328.85 | 328.70 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 11:15:00 | 322.22 | 320.43 | 0.00 | ORB-long ORB[317.95,321.99] vol=3.9x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-03-13 11:40:00 | 321.36 | 320.54 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-03-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-17 10:25:00 | 311.71 | 317.14 | 0.00 | ORB-short ORB[316.77,320.26] vol=2.3x ATR=1.40 |
| Stop hit — per-position SL triggered | 2025-03-17 10:30:00 | 313.11 | 316.57 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-03-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 11:05:00 | 325.13 | 323.74 | 0.00 | ORB-long ORB[319.27,323.87] vol=2.0x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 11:50:00 | 326.80 | 324.25 | 0.00 | T1 1.5R @ 326.80 |
| Target hit | 2025-03-18 15:20:00 | 328.01 | 326.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — BUY (started 2025-03-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 09:35:00 | 333.37 | 332.87 | 0.00 | ORB-long ORB[330.09,332.37] vol=4.8x ATR=1.00 |
| Stop hit — per-position SL triggered | 2025-03-20 09:55:00 | 332.37 | 332.78 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-03-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-24 09:35:00 | 331.07 | 333.44 | 0.00 | ORB-short ORB[332.03,336.99] vol=1.8x ATR=1.33 |
| Stop hit — per-position SL triggered | 2025-03-24 10:55:00 | 332.40 | 332.33 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-03-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 10:55:00 | 329.59 | 330.60 | 0.00 | ORB-short ORB[330.40,334.97] vol=3.8x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-26 11:15:00 | 328.17 | 330.35 | 0.00 | T1 1.5R @ 328.17 |
| Target hit | 2025-03-26 15:20:00 | 323.03 | 327.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — BUY (started 2025-04-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 10:00:00 | 336.78 | 334.47 | 0.00 | ORB-long ORB[332.00,335.39] vol=1.8x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 12:00:00 | 338.78 | 335.53 | 0.00 | T1 1.5R @ 338.78 |
| Target hit | 2025-04-02 15:20:00 | 341.00 | 338.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — BUY (started 2025-04-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-08 10:00:00 | 344.37 | 340.44 | 0.00 | ORB-long ORB[336.81,340.39] vol=2.2x ATR=1.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-08 10:20:00 | 347.09 | 342.44 | 0.00 | T1 1.5R @ 347.09 |
| Stop hit — per-position SL triggered | 2025-04-08 10:35:00 | 344.37 | 342.95 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-04-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 09:40:00 | 352.64 | 351.53 | 0.00 | ORB-long ORB[349.84,351.04] vol=2.1x ATR=1.09 |
| Stop hit — per-position SL triggered | 2025-04-16 10:15:00 | 351.55 | 351.63 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 09:35:00 | 349.62 | 348.29 | 0.00 | ORB-long ORB[344.88,349.24] vol=1.5x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 09:45:00 | 351.32 | 349.45 | 0.00 | T1 1.5R @ 351.32 |
| Target hit | 2025-04-22 15:00:00 | 354.40 | 355.13 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-14 09:40:00 | 339.99 | 2024-05-14 09:55:00 | 338.97 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-05-16 10:35:00 | 349.15 | 2024-05-16 11:55:00 | 350.24 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-05-23 09:30:00 | 345.03 | 2024-05-23 09:50:00 | 343.76 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-05-23 09:30:00 | 345.03 | 2024-05-23 10:45:00 | 345.03 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-29 09:50:00 | 354.97 | 2024-05-29 10:05:00 | 353.52 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-05-31 10:00:00 | 345.61 | 2024-05-31 10:40:00 | 343.43 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-05-31 10:00:00 | 345.61 | 2024-05-31 11:15:00 | 345.61 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-05 09:30:00 | 351.58 | 2024-06-05 09:40:00 | 355.04 | PARTIAL | 0.50 | 0.98% |
| BUY | retest1 | 2024-06-05 09:30:00 | 351.58 | 2024-06-05 10:10:00 | 360.01 | TARGET_HIT | 0.50 | 2.40% |
| SELL | retest1 | 2024-06-19 10:20:00 | 368.00 | 2024-06-19 11:00:00 | 365.97 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-06-19 10:20:00 | 368.00 | 2024-06-19 12:05:00 | 367.93 | TARGET_HIT | 0.50 | 0.02% |
| SELL | retest1 | 2024-06-21 10:40:00 | 370.57 | 2024-06-21 11:15:00 | 371.83 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-06-25 09:30:00 | 373.35 | 2024-06-25 09:45:00 | 371.61 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-06-25 09:30:00 | 373.35 | 2024-06-25 09:55:00 | 373.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-26 11:00:00 | 365.00 | 2024-06-26 11:10:00 | 365.90 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-07-09 09:30:00 | 418.59 | 2024-07-09 09:35:00 | 416.72 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-07-15 11:10:00 | 433.48 | 2024-07-15 14:15:00 | 431.86 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-07-18 09:40:00 | 432.69 | 2024-07-18 09:45:00 | 435.14 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-07-18 09:40:00 | 432.69 | 2024-07-18 09:55:00 | 434.91 | TARGET_HIT | 0.50 | 0.51% |
| BUY | retest1 | 2024-07-29 10:10:00 | 480.00 | 2024-07-29 10:15:00 | 478.66 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-08-01 10:15:00 | 485.00 | 2024-08-01 10:20:00 | 482.62 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-08-09 10:00:00 | 460.75 | 2024-08-09 10:25:00 | 458.64 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-08-09 10:00:00 | 460.75 | 2024-08-09 10:50:00 | 460.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-16 09:35:00 | 446.82 | 2024-08-16 09:40:00 | 445.44 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-08-20 09:45:00 | 441.61 | 2024-08-20 09:50:00 | 439.39 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-08-20 09:45:00 | 441.61 | 2024-08-20 10:20:00 | 441.61 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-21 10:10:00 | 454.94 | 2024-08-21 11:35:00 | 451.73 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2024-08-21 10:10:00 | 454.94 | 2024-08-21 11:45:00 | 454.94 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-26 09:35:00 | 453.60 | 2024-08-26 10:00:00 | 455.16 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-08-28 09:30:00 | 448.94 | 2024-08-28 09:40:00 | 450.82 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-08-29 10:25:00 | 453.82 | 2024-08-29 10:55:00 | 451.73 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-08-29 10:25:00 | 453.82 | 2024-08-29 12:25:00 | 453.82 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-30 09:35:00 | 455.39 | 2024-08-30 09:45:00 | 453.59 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-09-04 09:30:00 | 440.74 | 2024-09-04 09:35:00 | 442.22 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-09-06 09:35:00 | 455.05 | 2024-09-06 10:25:00 | 457.66 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-09-06 09:35:00 | 455.05 | 2024-09-06 12:15:00 | 455.21 | TARGET_HIT | 0.50 | 0.04% |
| SELL | retest1 | 2024-09-17 09:30:00 | 439.55 | 2024-09-17 10:10:00 | 437.79 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-09-17 09:30:00 | 439.55 | 2024-09-17 15:20:00 | 434.60 | TARGET_HIT | 0.50 | 1.13% |
| SELL | retest1 | 2024-09-18 11:00:00 | 429.77 | 2024-09-18 11:05:00 | 427.85 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-09-18 11:00:00 | 429.77 | 2024-09-18 11:10:00 | 429.77 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-19 09:50:00 | 426.00 | 2024-09-19 09:55:00 | 427.27 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-09-20 10:30:00 | 413.49 | 2024-09-20 11:00:00 | 415.37 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-09-25 09:40:00 | 399.17 | 2024-09-25 10:05:00 | 397.06 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-09-25 09:40:00 | 399.17 | 2024-09-25 13:30:00 | 397.59 | TARGET_HIT | 0.50 | 0.40% |
| BUY | retest1 | 2024-09-27 09:30:00 | 400.42 | 2024-09-27 09:40:00 | 399.14 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-10-09 10:10:00 | 389.61 | 2024-10-09 15:20:00 | 390.41 | TARGET_HIT | 1.00 | 0.21% |
| BUY | retest1 | 2024-10-16 10:05:00 | 403.20 | 2024-10-16 10:30:00 | 400.91 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest1 | 2024-10-17 10:30:00 | 397.69 | 2024-10-17 10:35:00 | 399.29 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-10-25 09:30:00 | 363.09 | 2024-10-25 09:40:00 | 360.74 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2024-10-25 09:30:00 | 363.09 | 2024-10-25 14:25:00 | 359.31 | TARGET_HIT | 0.50 | 1.04% |
| SELL | retest1 | 2024-11-21 10:40:00 | 382.60 | 2024-11-21 11:25:00 | 384.29 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-11-28 10:40:00 | 402.00 | 2024-11-28 11:00:00 | 403.90 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-11-28 10:40:00 | 402.00 | 2024-11-28 12:40:00 | 402.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-03 09:50:00 | 415.23 | 2024-12-03 10:00:00 | 413.20 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-12-06 09:40:00 | 409.73 | 2024-12-06 09:55:00 | 407.69 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-12-06 09:40:00 | 409.73 | 2024-12-06 11:55:00 | 409.15 | TARGET_HIT | 0.50 | 0.14% |
| BUY | retest1 | 2024-12-10 09:35:00 | 412.64 | 2024-12-10 09:45:00 | 411.36 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-12-13 10:30:00 | 400.02 | 2024-12-13 10:50:00 | 401.29 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-12-26 09:55:00 | 380.31 | 2024-12-26 11:15:00 | 381.40 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-12-27 10:45:00 | 378.79 | 2024-12-27 11:10:00 | 379.26 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest1 | 2024-12-30 10:30:00 | 379.93 | 2024-12-30 10:35:00 | 378.86 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-12-31 10:40:00 | 382.26 | 2024-12-31 10:50:00 | 384.33 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-12-31 10:40:00 | 382.26 | 2024-12-31 15:20:00 | 390.54 | TARGET_HIT | 0.50 | 2.17% |
| BUY | retest1 | 2025-01-02 11:15:00 | 397.34 | 2025-01-02 11:20:00 | 396.26 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-01-15 10:40:00 | 378.28 | 2025-01-15 12:50:00 | 380.53 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-01-15 10:40:00 | 378.28 | 2025-01-15 13:25:00 | 380.00 | TARGET_HIT | 0.50 | 0.45% |
| BUY | retest1 | 2025-01-16 10:40:00 | 383.59 | 2025-01-16 10:45:00 | 382.58 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-01-17 09:45:00 | 379.10 | 2025-01-17 11:05:00 | 377.44 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-01-17 09:45:00 | 379.10 | 2025-01-17 15:20:00 | 375.00 | TARGET_HIT | 0.50 | 1.08% |
| SELL | retest1 | 2025-01-20 10:00:00 | 371.21 | 2025-01-20 10:30:00 | 372.32 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-01-21 11:10:00 | 372.22 | 2025-01-21 11:35:00 | 373.30 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-22 11:00:00 | 365.00 | 2025-01-22 11:05:00 | 366.67 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-01-29 10:50:00 | 364.67 | 2025-01-29 10:55:00 | 363.42 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-02-17 10:55:00 | 334.29 | 2025-02-17 11:10:00 | 333.05 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-02-19 09:30:00 | 330.04 | 2025-02-19 09:35:00 | 328.85 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-03-13 11:15:00 | 322.22 | 2025-03-13 11:40:00 | 321.36 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-03-17 10:25:00 | 311.71 | 2025-03-17 10:30:00 | 313.11 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-03-18 11:05:00 | 325.13 | 2025-03-18 11:50:00 | 326.80 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-03-18 11:05:00 | 325.13 | 2025-03-18 15:20:00 | 328.01 | TARGET_HIT | 0.50 | 0.89% |
| BUY | retest1 | 2025-03-20 09:35:00 | 333.37 | 2025-03-20 09:55:00 | 332.37 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-03-24 09:35:00 | 331.07 | 2025-03-24 10:55:00 | 332.40 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-03-26 10:55:00 | 329.59 | 2025-03-26 11:15:00 | 328.17 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-03-26 10:55:00 | 329.59 | 2025-03-26 15:20:00 | 323.03 | TARGET_HIT | 0.50 | 1.99% |
| BUY | retest1 | 2025-04-02 10:00:00 | 336.78 | 2025-04-02 12:00:00 | 338.78 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-04-02 10:00:00 | 336.78 | 2025-04-02 15:20:00 | 341.00 | TARGET_HIT | 0.50 | 1.25% |
| BUY | retest1 | 2025-04-08 10:00:00 | 344.37 | 2025-04-08 10:20:00 | 347.09 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2025-04-08 10:00:00 | 344.37 | 2025-04-08 10:35:00 | 344.37 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-16 09:40:00 | 352.64 | 2025-04-16 10:15:00 | 351.55 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-04-22 09:35:00 | 349.62 | 2025-04-22 09:45:00 | 351.32 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-04-22 09:35:00 | 349.62 | 2025-04-22 15:00:00 | 354.40 | TARGET_HIT | 0.50 | 1.37% |
