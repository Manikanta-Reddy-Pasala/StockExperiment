# Bikaji Foods International Ltd. (BIKAJI)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (55355 bars)
- **Last close:** 670.20
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
| ENTRY1 | 84 |
| ENTRY2 | 0 |
| PARTIAL | 38 |
| TARGET_HIT | 12 |
| STOP_HIT | 72 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 122 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 50 / 72
- **Target hits / Stop hits / Partials:** 12 / 72 / 38
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 14.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 54 | 21 | 38.9% | 4 | 33 | 17 | 0.12% | 6.3% |
| BUY @ 2nd Alert (retest1) | 54 | 21 | 38.9% | 4 | 33 | 17 | 0.12% | 6.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 68 | 29 | 42.6% | 8 | 39 | 21 | 0.12% | 8.2% |
| SELL @ 2nd Alert (retest1) | 68 | 29 | 42.6% | 8 | 39 | 21 | 0.12% | 8.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 122 | 50 | 41.0% | 12 | 72 | 38 | 0.12% | 14.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-15 09:50:00 | 373.00 | 368.60 | 0.00 | ORB-long ORB[364.05,369.00] vol=1.5x ATR=2.24 |
| Stop hit — per-position SL triggered | 2023-05-15 10:10:00 | 370.76 | 369.68 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 09:30:00 | 378.00 | 380.31 | 0.00 | ORB-short ORB[379.30,384.65] vol=2.0x ATR=1.43 |
| Stop hit — per-position SL triggered | 2023-05-19 09:50:00 | 379.43 | 380.05 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-05-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-31 09:30:00 | 379.50 | 377.07 | 0.00 | ORB-long ORB[374.05,379.40] vol=1.9x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-31 09:35:00 | 381.37 | 377.75 | 0.00 | T1 1.5R @ 381.37 |
| Stop hit — per-position SL triggered | 2023-05-31 10:15:00 | 379.50 | 379.70 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-06-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-02 11:00:00 | 380.50 | 382.86 | 0.00 | ORB-short ORB[381.40,384.20] vol=3.0x ATR=0.90 |
| Stop hit — per-position SL triggered | 2023-06-02 11:05:00 | 381.40 | 382.82 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2023-06-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-05 09:30:00 | 378.50 | 381.84 | 0.00 | ORB-short ORB[381.45,385.00] vol=3.1x ATR=1.29 |
| Stop hit — per-position SL triggered | 2023-06-05 09:35:00 | 379.79 | 381.41 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2023-06-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-06 09:55:00 | 378.80 | 380.21 | 0.00 | ORB-short ORB[379.15,383.00] vol=2.1x ATR=1.74 |
| Stop hit — per-position SL triggered | 2023-06-06 11:25:00 | 380.54 | 379.94 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-06-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-12 11:05:00 | 395.65 | 396.79 | 0.00 | ORB-short ORB[396.00,401.90] vol=3.0x ATR=1.38 |
| Stop hit — per-position SL triggered | 2023-06-12 11:30:00 | 397.03 | 396.75 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-06-15 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-15 10:40:00 | 416.10 | 413.43 | 0.00 | ORB-long ORB[412.00,415.70] vol=1.8x ATR=1.12 |
| Stop hit — per-position SL triggered | 2023-06-15 10:45:00 | 414.98 | 413.57 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-06-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-16 10:55:00 | 412.50 | 414.39 | 0.00 | ORB-short ORB[412.60,416.70] vol=1.6x ATR=1.01 |
| Stop hit — per-position SL triggered | 2023-06-16 11:40:00 | 413.51 | 413.96 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-06-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-19 09:50:00 | 410.30 | 408.36 | 0.00 | ORB-long ORB[406.50,410.20] vol=1.6x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-19 10:10:00 | 412.62 | 409.09 | 0.00 | T1 1.5R @ 412.62 |
| Target hit | 2023-06-19 15:20:00 | 415.00 | 412.87 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2023-06-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-20 09:30:00 | 411.60 | 414.08 | 0.00 | ORB-short ORB[413.50,416.95] vol=2.3x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-20 09:35:00 | 410.23 | 413.18 | 0.00 | T1 1.5R @ 410.23 |
| Stop hit — per-position SL triggered | 2023-06-20 09:40:00 | 411.60 | 413.14 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-06-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-21 11:05:00 | 413.40 | 416.97 | 0.00 | ORB-short ORB[417.05,419.95] vol=2.4x ATR=0.98 |
| Stop hit — per-position SL triggered | 2023-06-21 11:50:00 | 414.38 | 416.63 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-06-22 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-22 10:05:00 | 411.40 | 409.33 | 0.00 | ORB-long ORB[406.40,410.95] vol=2.1x ATR=1.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-22 11:15:00 | 413.61 | 410.28 | 0.00 | T1 1.5R @ 413.61 |
| Stop hit — per-position SL triggered | 2023-06-22 11:20:00 | 411.40 | 410.30 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2023-07-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-04 09:40:00 | 406.60 | 408.25 | 0.00 | ORB-short ORB[408.05,411.60] vol=3.5x ATR=1.56 |
| Stop hit — per-position SL triggered | 2023-07-04 10:05:00 | 408.16 | 407.46 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-05 10:15:00 | 407.00 | 408.60 | 0.00 | ORB-short ORB[407.55,411.85] vol=4.2x ATR=1.42 |
| Stop hit — per-position SL triggered | 2023-07-05 10:30:00 | 408.42 | 408.46 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2023-07-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-11 10:45:00 | 406.00 | 408.06 | 0.00 | ORB-short ORB[406.40,409.70] vol=1.8x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-11 11:50:00 | 404.08 | 406.68 | 0.00 | T1 1.5R @ 404.08 |
| Stop hit — per-position SL triggered | 2023-07-11 12:45:00 | 406.00 | 406.46 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-07-14 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 10:20:00 | 417.75 | 414.38 | 0.00 | ORB-long ORB[412.10,417.00] vol=1.6x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-14 10:30:00 | 420.69 | 416.00 | 0.00 | T1 1.5R @ 420.69 |
| Stop hit — per-position SL triggered | 2023-07-14 10:45:00 | 417.75 | 416.43 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-07-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-19 09:30:00 | 432.00 | 430.87 | 0.00 | ORB-long ORB[426.60,431.80] vol=3.4x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-19 09:40:00 | 434.75 | 441.68 | 0.00 | T1 1.5R @ 434.75 |
| Target hit | 2023-07-19 09:45:00 | 441.30 | 441.80 | 0.00 | Trail-exit close<VWAP |

### Cycle 19 — BUY (started 2023-07-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-31 09:35:00 | 484.00 | 479.92 | 0.00 | ORB-long ORB[473.50,479.80] vol=4.0x ATR=2.34 |
| Stop hit — per-position SL triggered | 2023-07-31 10:40:00 | 481.66 | 482.92 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-08-04 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-04 11:10:00 | 477.70 | 479.89 | 0.00 | ORB-short ORB[480.15,484.50] vol=2.1x ATR=1.54 |
| Stop hit — per-position SL triggered | 2023-08-04 11:25:00 | 479.24 | 479.77 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-08-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-07 09:30:00 | 484.10 | 481.94 | 0.00 | ORB-long ORB[477.00,483.80] vol=1.6x ATR=1.27 |
| Stop hit — per-position SL triggered | 2023-08-07 09:40:00 | 482.83 | 482.86 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-08-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-09 09:35:00 | 492.65 | 487.61 | 0.00 | ORB-long ORB[482.50,489.00] vol=4.3x ATR=2.21 |
| Stop hit — per-position SL triggered | 2023-08-09 09:55:00 | 490.44 | 489.85 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-08-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-11 09:35:00 | 484.60 | 486.84 | 0.00 | ORB-short ORB[486.00,493.00] vol=1.9x ATR=2.12 |
| Stop hit — per-position SL triggered | 2023-08-11 09:45:00 | 486.72 | 486.76 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-08-17 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-17 10:00:00 | 476.80 | 478.01 | 0.00 | ORB-short ORB[478.00,483.00] vol=2.7x ATR=1.64 |
| Stop hit — per-position SL triggered | 2023-08-17 10:30:00 | 478.44 | 477.63 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-08-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-18 09:50:00 | 482.10 | 479.07 | 0.00 | ORB-long ORB[475.00,479.90] vol=2.3x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-18 10:00:00 | 484.59 | 481.29 | 0.00 | T1 1.5R @ 484.59 |
| Target hit | 2023-08-18 10:45:00 | 483.45 | 483.48 | 0.00 | Trail-exit close<VWAP |

### Cycle 26 — SELL (started 2023-08-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 10:30:00 | 474.30 | 479.74 | 0.00 | ORB-short ORB[476.10,482.15] vol=1.8x ATR=1.67 |
| Stop hit — per-position SL triggered | 2023-08-25 10:40:00 | 475.97 | 479.11 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-08-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-28 09:40:00 | 483.50 | 481.31 | 0.00 | ORB-long ORB[476.75,482.80] vol=2.7x ATR=1.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-28 09:45:00 | 485.62 | 482.21 | 0.00 | T1 1.5R @ 485.62 |
| Stop hit — per-position SL triggered | 2023-08-28 09:55:00 | 483.50 | 482.49 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-08-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-29 10:30:00 | 481.50 | 484.89 | 0.00 | ORB-short ORB[484.10,488.40] vol=2.7x ATR=1.48 |
| Stop hit — per-position SL triggered | 2023-08-29 10:35:00 | 482.98 | 484.75 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-09-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-04 10:45:00 | 482.00 | 485.45 | 0.00 | ORB-short ORB[486.20,490.00] vol=1.9x ATR=1.57 |
| Stop hit — per-position SL triggered | 2023-09-04 11:20:00 | 483.57 | 484.62 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-09-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 11:00:00 | 487.05 | 483.41 | 0.00 | ORB-long ORB[481.20,484.80] vol=2.7x ATR=1.26 |
| Stop hit — per-position SL triggered | 2023-09-05 11:05:00 | 485.79 | 483.94 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-09-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-20 09:50:00 | 490.40 | 492.23 | 0.00 | ORB-short ORB[490.50,495.00] vol=2.5x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-20 10:00:00 | 487.83 | 491.43 | 0.00 | T1 1.5R @ 487.83 |
| Stop hit — per-position SL triggered | 2023-09-20 10:10:00 | 490.40 | 491.17 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-09-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-26 11:10:00 | 489.20 | 492.47 | 0.00 | ORB-short ORB[490.55,496.40] vol=4.2x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-26 11:15:00 | 487.25 | 490.84 | 0.00 | T1 1.5R @ 487.25 |
| Stop hit — per-position SL triggered | 2023-09-26 11:55:00 | 489.20 | 488.24 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2023-09-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-28 11:00:00 | 490.45 | 491.29 | 0.00 | ORB-short ORB[490.75,493.90] vol=2.3x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-28 11:20:00 | 488.55 | 491.15 | 0.00 | T1 1.5R @ 488.55 |
| Stop hit — per-position SL triggered | 2023-09-28 15:00:00 | 490.45 | 489.57 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2023-09-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-29 10:55:00 | 488.05 | 490.03 | 0.00 | ORB-short ORB[488.85,494.40] vol=2.5x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-29 11:15:00 | 485.91 | 488.91 | 0.00 | T1 1.5R @ 485.91 |
| Target hit | 2023-09-29 12:25:00 | 487.25 | 486.98 | 0.00 | Trail-exit close>VWAP |

### Cycle 35 — BUY (started 2023-10-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-03 09:30:00 | 491.00 | 488.53 | 0.00 | ORB-long ORB[484.75,490.00] vol=1.9x ATR=1.72 |
| Stop hit — per-position SL triggered | 2023-10-03 10:05:00 | 489.28 | 490.01 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-10-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-04 10:35:00 | 490.10 | 488.38 | 0.00 | ORB-long ORB[485.00,488.95] vol=3.4x ATR=1.52 |
| Stop hit — per-position SL triggered | 2023-10-04 11:20:00 | 488.58 | 489.21 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-10-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-06 10:45:00 | 480.30 | 482.23 | 0.00 | ORB-short ORB[481.15,483.70] vol=2.1x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-06 11:10:00 | 478.50 | 481.14 | 0.00 | T1 1.5R @ 478.50 |
| Target hit | 2023-10-06 15:20:00 | 471.75 | 473.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — BUY (started 2023-10-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-11 09:35:00 | 477.00 | 475.80 | 0.00 | ORB-long ORB[472.55,476.95] vol=2.4x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-11 10:35:00 | 478.92 | 476.77 | 0.00 | T1 1.5R @ 478.92 |
| Target hit | 2023-10-11 15:20:00 | 482.75 | 483.55 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 39 — BUY (started 2023-10-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-13 10:00:00 | 484.80 | 483.69 | 0.00 | ORB-long ORB[480.55,484.50] vol=2.2x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-13 11:10:00 | 486.61 | 484.61 | 0.00 | T1 1.5R @ 486.61 |
| Stop hit — per-position SL triggered | 2023-10-13 12:15:00 | 484.80 | 484.77 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2023-10-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-16 09:35:00 | 484.95 | 483.61 | 0.00 | ORB-long ORB[479.00,484.90] vol=1.5x ATR=1.41 |
| Stop hit — per-position SL triggered | 2023-10-16 09:55:00 | 483.54 | 484.13 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2023-10-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-17 10:25:00 | 489.10 | 492.72 | 0.00 | ORB-short ORB[491.35,496.40] vol=1.7x ATR=1.85 |
| Target hit | 2023-10-17 15:20:00 | 487.25 | 490.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — SELL (started 2023-10-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 10:35:00 | 485.10 | 488.02 | 0.00 | ORB-short ORB[486.05,490.50] vol=2.0x ATR=1.63 |
| Stop hit — per-position SL triggered | 2023-10-18 10:55:00 | 486.73 | 487.34 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2023-10-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-30 10:30:00 | 472.80 | 471.34 | 0.00 | ORB-long ORB[463.65,470.00] vol=2.1x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-30 10:50:00 | 475.57 | 472.11 | 0.00 | T1 1.5R @ 475.57 |
| Stop hit — per-position SL triggered | 2023-10-30 12:15:00 | 472.80 | 472.61 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-11-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-01 09:40:00 | 482.00 | 478.66 | 0.00 | ORB-long ORB[474.00,479.45] vol=1.5x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-01 10:05:00 | 484.51 | 480.51 | 0.00 | T1 1.5R @ 484.51 |
| Stop hit — per-position SL triggered | 2023-11-01 11:00:00 | 482.00 | 481.47 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-11-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-15 10:45:00 | 531.65 | 524.59 | 0.00 | ORB-long ORB[518.60,524.90] vol=8.7x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-15 10:50:00 | 534.90 | 527.17 | 0.00 | T1 1.5R @ 534.90 |
| Stop hit — per-position SL triggered | 2023-11-15 10:55:00 | 531.65 | 527.55 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2023-11-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-22 10:55:00 | 540.15 | 544.77 | 0.00 | ORB-short ORB[545.55,551.05] vol=1.6x ATR=2.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-22 11:45:00 | 537.12 | 543.68 | 0.00 | T1 1.5R @ 537.12 |
| Target hit | 2023-11-22 15:20:00 | 534.65 | 538.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — BUY (started 2023-11-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-23 11:00:00 | 541.10 | 540.02 | 0.00 | ORB-long ORB[534.80,540.60] vol=2.7x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-23 13:00:00 | 543.46 | 540.47 | 0.00 | T1 1.5R @ 543.46 |
| Stop hit — per-position SL triggered | 2023-11-23 13:20:00 | 541.10 | 540.97 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2023-11-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-24 09:45:00 | 548.90 | 543.61 | 0.00 | ORB-long ORB[537.10,542.00] vol=4.8x ATR=2.62 |
| Stop hit — per-position SL triggered | 2023-11-24 09:50:00 | 546.28 | 545.15 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2023-11-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-28 11:10:00 | 547.05 | 550.54 | 0.00 | ORB-short ORB[547.55,554.90] vol=2.7x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-28 11:30:00 | 544.41 | 549.93 | 0.00 | T1 1.5R @ 544.41 |
| Target hit | 2023-11-28 15:20:00 | 542.85 | 546.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — BUY (started 2023-11-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 09:40:00 | 546.00 | 543.96 | 0.00 | ORB-long ORB[540.25,544.00] vol=4.4x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-29 09:45:00 | 548.76 | 544.23 | 0.00 | T1 1.5R @ 548.76 |
| Stop hit — per-position SL triggered | 2023-11-29 09:50:00 | 546.00 | 544.33 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2023-11-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 09:40:00 | 553.00 | 550.06 | 0.00 | ORB-long ORB[544.50,551.00] vol=4.3x ATR=1.58 |
| Stop hit — per-position SL triggered | 2023-11-30 09:45:00 | 551.42 | 550.47 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2023-12-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 11:00:00 | 523.45 | 528.44 | 0.00 | ORB-short ORB[526.05,531.95] vol=3.0x ATR=1.84 |
| Stop hit — per-position SL triggered | 2023-12-08 11:15:00 | 525.29 | 528.00 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-12-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-14 10:15:00 | 539.00 | 535.51 | 0.00 | ORB-long ORB[533.00,537.45] vol=2.3x ATR=1.84 |
| Stop hit — per-position SL triggered | 2023-12-14 10:20:00 | 537.16 | 536.00 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-12-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-19 09:50:00 | 544.60 | 540.85 | 0.00 | ORB-long ORB[534.05,539.60] vol=1.7x ATR=2.34 |
| Stop hit — per-position SL triggered | 2023-12-19 12:25:00 | 542.26 | 543.67 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2023-12-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-20 10:55:00 | 539.55 | 542.13 | 0.00 | ORB-short ORB[542.05,547.65] vol=11.9x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 11:50:00 | 536.60 | 541.19 | 0.00 | T1 1.5R @ 536.60 |
| Stop hit — per-position SL triggered | 2023-12-20 12:00:00 | 539.55 | 540.72 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2023-12-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 09:40:00 | 557.25 | 555.40 | 0.00 | ORB-long ORB[549.75,557.00] vol=1.9x ATR=2.22 |
| Stop hit — per-position SL triggered | 2023-12-27 10:00:00 | 555.03 | 555.60 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2023-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-29 10:15:00 | 552.25 | 554.59 | 0.00 | ORB-short ORB[553.00,558.00] vol=2.6x ATR=1.93 |
| Stop hit — per-position SL triggered | 2023-12-29 10:20:00 | 554.18 | 554.58 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2024-01-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-03 09:30:00 | 556.90 | 554.53 | 0.00 | ORB-long ORB[549.65,554.45] vol=3.6x ATR=1.89 |
| Stop hit — per-position SL triggered | 2024-01-03 09:40:00 | 555.01 | 553.64 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2024-01-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 09:30:00 | 576.00 | 573.99 | 0.00 | ORB-long ORB[568.90,574.90] vol=4.2x ATR=2.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-05 10:10:00 | 580.29 | 576.81 | 0.00 | T1 1.5R @ 580.29 |
| Stop hit — per-position SL triggered | 2024-01-05 10:45:00 | 576.00 | 577.42 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-01-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-10 11:05:00 | 571.15 | 573.82 | 0.00 | ORB-short ORB[572.10,578.35] vol=2.8x ATR=1.33 |
| Stop hit — per-position SL triggered | 2024-01-10 11:10:00 | 572.48 | 573.79 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2024-01-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-11 10:10:00 | 569.45 | 572.95 | 0.00 | ORB-short ORB[572.30,580.00] vol=1.9x ATR=1.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-11 10:55:00 | 566.61 | 571.90 | 0.00 | T1 1.5R @ 566.61 |
| Stop hit — per-position SL triggered | 2024-01-11 11:00:00 | 569.45 | 571.76 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-01-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-15 09:40:00 | 580.50 | 584.70 | 0.00 | ORB-short ORB[583.05,587.45] vol=2.1x ATR=2.11 |
| Stop hit — per-position SL triggered | 2024-01-15 10:00:00 | 582.61 | 582.47 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-01-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-17 10:55:00 | 574.45 | 579.11 | 0.00 | ORB-short ORB[575.20,583.60] vol=4.1x ATR=2.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-17 12:05:00 | 570.99 | 576.87 | 0.00 | T1 1.5R @ 570.99 |
| Stop hit — per-position SL triggered | 2024-01-17 12:50:00 | 574.45 | 572.61 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2024-01-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-19 11:05:00 | 575.05 | 578.16 | 0.00 | ORB-short ORB[578.00,584.80] vol=1.7x ATR=2.03 |
| Stop hit — per-position SL triggered | 2024-01-19 13:00:00 | 577.08 | 577.01 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-01-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-25 10:25:00 | 574.25 | 578.86 | 0.00 | ORB-short ORB[576.40,582.15] vol=1.6x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-25 10:55:00 | 571.21 | 576.97 | 0.00 | T1 1.5R @ 571.21 |
| Stop hit — per-position SL triggered | 2024-01-25 13:55:00 | 574.25 | 574.89 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2024-01-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 09:30:00 | 580.25 | 578.64 | 0.00 | ORB-long ORB[575.50,579.85] vol=2.4x ATR=2.29 |
| Stop hit — per-position SL triggered | 2024-01-29 09:40:00 | 577.96 | 578.76 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2024-02-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-12 09:40:00 | 544.50 | 547.85 | 0.00 | ORB-short ORB[545.10,552.85] vol=1.5x ATR=2.35 |
| Stop hit — per-position SL triggered | 2024-02-12 11:10:00 | 546.85 | 546.10 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2024-02-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-20 10:10:00 | 536.20 | 536.67 | 0.00 | ORB-short ORB[538.10,543.95] vol=10.1x ATR=2.31 |
| Stop hit — per-position SL triggered | 2024-02-20 11:15:00 | 538.51 | 536.65 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2024-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-27 11:15:00 | 527.50 | 530.26 | 0.00 | ORB-short ORB[528.25,533.90] vol=3.1x ATR=1.18 |
| Stop hit — per-position SL triggered | 2024-02-27 11:25:00 | 528.68 | 530.19 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2024-02-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:35:00 | 520.10 | 522.29 | 0.00 | ORB-short ORB[521.00,526.00] vol=3.0x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-28 10:40:00 | 518.08 | 521.63 | 0.00 | T1 1.5R @ 518.08 |
| Stop hit — per-position SL triggered | 2024-02-28 10:45:00 | 520.10 | 521.54 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2024-03-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-07 11:15:00 | 514.05 | 516.62 | 0.00 | ORB-short ORB[514.30,521.60] vol=2.3x ATR=2.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-07 11:20:00 | 510.93 | 515.87 | 0.00 | T1 1.5R @ 510.93 |
| Stop hit — per-position SL triggered | 2024-03-07 11:25:00 | 514.05 | 515.86 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2024-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-11 10:15:00 | 537.00 | 531.19 | 0.00 | ORB-long ORB[526.75,533.85] vol=1.8x ATR=2.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 10:30:00 | 540.72 | 534.09 | 0.00 | T1 1.5R @ 540.72 |
| Stop hit — per-position SL triggered | 2024-03-11 11:10:00 | 537.00 | 536.05 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2024-03-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-13 09:40:00 | 503.60 | 506.29 | 0.00 | ORB-short ORB[505.00,512.40] vol=1.7x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 09:45:00 | 500.97 | 505.18 | 0.00 | T1 1.5R @ 500.97 |
| Target hit | 2024-03-13 10:15:00 | 500.75 | 498.73 | 0.00 | Trail-exit close>VWAP |

### Cycle 74 — SELL (started 2024-03-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-15 10:55:00 | 489.90 | 491.20 | 0.00 | ORB-short ORB[490.65,496.55] vol=1.6x ATR=1.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-15 11:05:00 | 487.27 | 490.82 | 0.00 | T1 1.5R @ 487.27 |
| Target hit | 2024-03-15 15:20:00 | 486.90 | 487.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — SELL (started 2024-03-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-20 10:50:00 | 482.65 | 483.58 | 0.00 | ORB-short ORB[483.00,490.00] vol=1.6x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 11:10:00 | 480.59 | 483.12 | 0.00 | T1 1.5R @ 480.59 |
| Stop hit — per-position SL triggered | 2024-03-20 11:15:00 | 482.65 | 482.96 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-03-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-26 09:30:00 | 486.05 | 488.60 | 0.00 | ORB-short ORB[487.50,491.95] vol=1.6x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-26 09:35:00 | 483.79 | 487.27 | 0.00 | T1 1.5R @ 483.79 |
| Stop hit — per-position SL triggered | 2024-03-26 09:40:00 | 486.05 | 486.74 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2024-03-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-27 10:00:00 | 489.35 | 486.83 | 0.00 | ORB-long ORB[482.45,487.95] vol=8.3x ATR=2.05 |
| Stop hit — per-position SL triggered | 2024-03-27 10:30:00 | 487.30 | 486.88 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2024-04-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 09:40:00 | 503.80 | 500.59 | 0.00 | ORB-long ORB[496.50,502.90] vol=6.6x ATR=2.74 |
| Stop hit — per-position SL triggered | 2024-04-02 09:45:00 | 501.06 | 500.73 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2024-04-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-22 11:00:00 | 522.00 | 523.53 | 0.00 | ORB-short ORB[522.15,527.85] vol=2.0x ATR=1.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-22 12:55:00 | 519.61 | 522.76 | 0.00 | T1 1.5R @ 519.61 |
| Target hit | 2024-04-22 15:20:00 | 515.00 | 518.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 80 — BUY (started 2024-04-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 11:10:00 | 521.80 | 518.91 | 0.00 | ORB-long ORB[515.10,520.00] vol=3.7x ATR=1.49 |
| Stop hit — per-position SL triggered | 2024-04-23 11:15:00 | 520.31 | 518.94 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2024-04-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 10:20:00 | 525.40 | 523.13 | 0.00 | ORB-long ORB[521.20,524.30] vol=2.3x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-24 10:30:00 | 527.58 | 523.61 | 0.00 | T1 1.5R @ 527.58 |
| Stop hit — per-position SL triggered | 2024-04-24 11:05:00 | 525.40 | 525.25 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2024-04-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-26 09:45:00 | 529.30 | 531.64 | 0.00 | ORB-short ORB[531.00,537.60] vol=1.9x ATR=2.20 |
| Stop hit — per-position SL triggered | 2024-04-26 09:50:00 | 531.50 | 531.59 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2024-04-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 09:40:00 | 536.00 | 532.79 | 0.00 | ORB-long ORB[525.55,533.45] vol=3.7x ATR=2.46 |
| Stop hit — per-position SL triggered | 2024-04-30 09:50:00 | 533.54 | 533.00 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2024-05-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 11:10:00 | 520.00 | 524.59 | 0.00 | ORB-short ORB[523.05,528.95] vol=2.9x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 11:40:00 | 516.83 | 522.01 | 0.00 | T1 1.5R @ 516.83 |
| Stop hit — per-position SL triggered | 2024-05-09 11:45:00 | 520.00 | 521.95 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-15 09:50:00 | 373.00 | 2023-05-15 10:10:00 | 370.76 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest1 | 2023-05-19 09:30:00 | 378.00 | 2023-05-19 09:50:00 | 379.43 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2023-05-31 09:30:00 | 379.50 | 2023-05-31 09:35:00 | 381.37 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2023-05-31 09:30:00 | 379.50 | 2023-05-31 10:15:00 | 379.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-02 11:00:00 | 380.50 | 2023-06-02 11:05:00 | 381.40 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-06-05 09:30:00 | 378.50 | 2023-06-05 09:35:00 | 379.79 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-06-06 09:55:00 | 378.80 | 2023-06-06 11:25:00 | 380.54 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2023-06-12 11:05:00 | 395.65 | 2023-06-12 11:30:00 | 397.03 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-06-15 10:40:00 | 416.10 | 2023-06-15 10:45:00 | 414.98 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-06-16 10:55:00 | 412.50 | 2023-06-16 11:40:00 | 413.51 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-06-19 09:50:00 | 410.30 | 2023-06-19 10:10:00 | 412.62 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2023-06-19 09:50:00 | 410.30 | 2023-06-19 15:20:00 | 415.00 | TARGET_HIT | 0.50 | 1.15% |
| SELL | retest1 | 2023-06-20 09:30:00 | 411.60 | 2023-06-20 09:35:00 | 410.23 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2023-06-20 09:30:00 | 411.60 | 2023-06-20 09:40:00 | 411.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-21 11:05:00 | 413.40 | 2023-06-21 11:50:00 | 414.38 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-06-22 10:05:00 | 411.40 | 2023-06-22 11:15:00 | 413.61 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2023-06-22 10:05:00 | 411.40 | 2023-06-22 11:20:00 | 411.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-04 09:40:00 | 406.60 | 2023-07-04 10:05:00 | 408.16 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2023-07-05 10:15:00 | 407.00 | 2023-07-05 10:30:00 | 408.42 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-07-11 10:45:00 | 406.00 | 2023-07-11 11:50:00 | 404.08 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2023-07-11 10:45:00 | 406.00 | 2023-07-11 12:45:00 | 406.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-14 10:20:00 | 417.75 | 2023-07-14 10:30:00 | 420.69 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2023-07-14 10:20:00 | 417.75 | 2023-07-14 10:45:00 | 417.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-19 09:30:00 | 432.00 | 2023-07-19 09:40:00 | 434.75 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2023-07-19 09:30:00 | 432.00 | 2023-07-19 09:45:00 | 441.30 | TARGET_HIT | 0.50 | 2.15% |
| BUY | retest1 | 2023-07-31 09:35:00 | 484.00 | 2023-07-31 10:40:00 | 481.66 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2023-08-04 11:10:00 | 477.70 | 2023-08-04 11:25:00 | 479.24 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-08-07 09:30:00 | 484.10 | 2023-08-07 09:40:00 | 482.83 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-08-09 09:35:00 | 492.65 | 2023-08-09 09:55:00 | 490.44 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2023-08-11 09:35:00 | 484.60 | 2023-08-11 09:45:00 | 486.72 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2023-08-17 10:00:00 | 476.80 | 2023-08-17 10:30:00 | 478.44 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-08-18 09:50:00 | 482.10 | 2023-08-18 10:00:00 | 484.59 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-08-18 09:50:00 | 482.10 | 2023-08-18 10:45:00 | 483.45 | TARGET_HIT | 0.50 | 0.28% |
| SELL | retest1 | 2023-08-25 10:30:00 | 474.30 | 2023-08-25 10:40:00 | 475.97 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-08-28 09:40:00 | 483.50 | 2023-08-28 09:45:00 | 485.62 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-08-28 09:40:00 | 483.50 | 2023-08-28 09:55:00 | 483.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-29 10:30:00 | 481.50 | 2023-08-29 10:35:00 | 482.98 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-09-04 10:45:00 | 482.00 | 2023-09-04 11:20:00 | 483.57 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-09-05 11:00:00 | 487.05 | 2023-09-05 11:05:00 | 485.79 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-09-20 09:50:00 | 490.40 | 2023-09-20 10:00:00 | 487.83 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2023-09-20 09:50:00 | 490.40 | 2023-09-20 10:10:00 | 490.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-26 11:10:00 | 489.20 | 2023-09-26 11:15:00 | 487.25 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2023-09-26 11:10:00 | 489.20 | 2023-09-26 11:55:00 | 489.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-28 11:00:00 | 490.45 | 2023-09-28 11:20:00 | 488.55 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-09-28 11:00:00 | 490.45 | 2023-09-28 15:00:00 | 490.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-29 10:55:00 | 488.05 | 2023-09-29 11:15:00 | 485.91 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2023-09-29 10:55:00 | 488.05 | 2023-09-29 12:25:00 | 487.25 | TARGET_HIT | 0.50 | 0.16% |
| BUY | retest1 | 2023-10-03 09:30:00 | 491.00 | 2023-10-03 10:05:00 | 489.28 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-10-04 10:35:00 | 490.10 | 2023-10-04 11:20:00 | 488.58 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-10-06 10:45:00 | 480.30 | 2023-10-06 11:10:00 | 478.50 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2023-10-06 10:45:00 | 480.30 | 2023-10-06 15:20:00 | 471.75 | TARGET_HIT | 0.50 | 1.78% |
| BUY | retest1 | 2023-10-11 09:35:00 | 477.00 | 2023-10-11 10:35:00 | 478.92 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-10-11 09:35:00 | 477.00 | 2023-10-11 15:20:00 | 482.75 | TARGET_HIT | 0.50 | 1.21% |
| BUY | retest1 | 2023-10-13 10:00:00 | 484.80 | 2023-10-13 11:10:00 | 486.61 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2023-10-13 10:00:00 | 484.80 | 2023-10-13 12:15:00 | 484.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-16 09:35:00 | 484.95 | 2023-10-16 09:55:00 | 483.54 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-10-17 10:25:00 | 489.10 | 2023-10-17 15:20:00 | 487.25 | TARGET_HIT | 1.00 | 0.38% |
| SELL | retest1 | 2023-10-18 10:35:00 | 485.10 | 2023-10-18 10:55:00 | 486.73 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-10-30 10:30:00 | 472.80 | 2023-10-30 10:50:00 | 475.57 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2023-10-30 10:30:00 | 472.80 | 2023-10-30 12:15:00 | 472.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-01 09:40:00 | 482.00 | 2023-11-01 10:05:00 | 484.51 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-11-01 09:40:00 | 482.00 | 2023-11-01 11:00:00 | 482.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-15 10:45:00 | 531.65 | 2023-11-15 10:50:00 | 534.90 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2023-11-15 10:45:00 | 531.65 | 2023-11-15 10:55:00 | 531.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-22 10:55:00 | 540.15 | 2023-11-22 11:45:00 | 537.12 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2023-11-22 10:55:00 | 540.15 | 2023-11-22 15:20:00 | 534.65 | TARGET_HIT | 0.50 | 1.02% |
| BUY | retest1 | 2023-11-23 11:00:00 | 541.10 | 2023-11-23 13:00:00 | 543.46 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-11-23 11:00:00 | 541.10 | 2023-11-23 13:20:00 | 541.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-24 09:45:00 | 548.90 | 2023-11-24 09:50:00 | 546.28 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2023-11-28 11:10:00 | 547.05 | 2023-11-28 11:30:00 | 544.41 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2023-11-28 11:10:00 | 547.05 | 2023-11-28 15:20:00 | 542.85 | TARGET_HIT | 0.50 | 0.77% |
| BUY | retest1 | 2023-11-29 09:40:00 | 546.00 | 2023-11-29 09:45:00 | 548.76 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2023-11-29 09:40:00 | 546.00 | 2023-11-29 09:50:00 | 546.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-30 09:40:00 | 553.00 | 2023-11-30 09:45:00 | 551.42 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-12-08 11:00:00 | 523.45 | 2023-12-08 11:15:00 | 525.29 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-12-14 10:15:00 | 539.00 | 2023-12-14 10:20:00 | 537.16 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2023-12-19 09:50:00 | 544.60 | 2023-12-19 12:25:00 | 542.26 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2023-12-20 10:55:00 | 539.55 | 2023-12-20 11:50:00 | 536.60 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2023-12-20 10:55:00 | 539.55 | 2023-12-20 12:00:00 | 539.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-27 09:40:00 | 557.25 | 2023-12-27 10:00:00 | 555.03 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2023-12-29 10:15:00 | 552.25 | 2023-12-29 10:20:00 | 554.18 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-01-03 09:30:00 | 556.90 | 2024-01-03 09:40:00 | 555.01 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-01-05 09:30:00 | 576.00 | 2024-01-05 10:10:00 | 580.29 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2024-01-05 09:30:00 | 576.00 | 2024-01-05 10:45:00 | 576.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-10 11:05:00 | 571.15 | 2024-01-10 11:10:00 | 572.48 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-01-11 10:10:00 | 569.45 | 2024-01-11 10:55:00 | 566.61 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-01-11 10:10:00 | 569.45 | 2024-01-11 11:00:00 | 569.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-15 09:40:00 | 580.50 | 2024-01-15 10:00:00 | 582.61 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-01-17 10:55:00 | 574.45 | 2024-01-17 12:05:00 | 570.99 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2024-01-17 10:55:00 | 574.45 | 2024-01-17 12:50:00 | 574.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-19 11:05:00 | 575.05 | 2024-01-19 13:00:00 | 577.08 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-01-25 10:25:00 | 574.25 | 2024-01-25 10:55:00 | 571.21 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-01-25 10:25:00 | 574.25 | 2024-01-25 13:55:00 | 574.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-29 09:30:00 | 580.25 | 2024-01-29 09:40:00 | 577.96 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-02-12 09:40:00 | 544.50 | 2024-02-12 11:10:00 | 546.85 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-02-20 10:10:00 | 536.20 | 2024-02-20 11:15:00 | 538.51 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-02-27 11:15:00 | 527.50 | 2024-02-27 11:25:00 | 528.68 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-02-28 10:35:00 | 520.10 | 2024-02-28 10:40:00 | 518.08 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-02-28 10:35:00 | 520.10 | 2024-02-28 10:45:00 | 520.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-07 11:15:00 | 514.05 | 2024-03-07 11:20:00 | 510.93 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-03-07 11:15:00 | 514.05 | 2024-03-07 11:25:00 | 514.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-11 10:15:00 | 537.00 | 2024-03-11 10:30:00 | 540.72 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-03-11 10:15:00 | 537.00 | 2024-03-11 11:10:00 | 537.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-13 09:40:00 | 503.60 | 2024-03-13 09:45:00 | 500.97 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-03-13 09:40:00 | 503.60 | 2024-03-13 10:15:00 | 500.75 | TARGET_HIT | 0.50 | 0.57% |
| SELL | retest1 | 2024-03-15 10:55:00 | 489.90 | 2024-03-15 11:05:00 | 487.27 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-03-15 10:55:00 | 489.90 | 2024-03-15 15:20:00 | 486.90 | TARGET_HIT | 0.50 | 0.61% |
| SELL | retest1 | 2024-03-20 10:50:00 | 482.65 | 2024-03-20 11:10:00 | 480.59 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-03-20 10:50:00 | 482.65 | 2024-03-20 11:15:00 | 482.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-26 09:30:00 | 486.05 | 2024-03-26 09:35:00 | 483.79 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-03-26 09:30:00 | 486.05 | 2024-03-26 09:40:00 | 486.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-27 10:00:00 | 489.35 | 2024-03-27 10:30:00 | 487.30 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-04-02 09:40:00 | 503.80 | 2024-04-02 09:45:00 | 501.06 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2024-04-22 11:00:00 | 522.00 | 2024-04-22 12:55:00 | 519.61 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-04-22 11:00:00 | 522.00 | 2024-04-22 15:20:00 | 515.00 | TARGET_HIT | 0.50 | 1.34% |
| BUY | retest1 | 2024-04-23 11:10:00 | 521.80 | 2024-04-23 11:15:00 | 520.31 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-04-24 10:20:00 | 525.40 | 2024-04-24 10:30:00 | 527.58 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-04-24 10:20:00 | 525.40 | 2024-04-24 11:05:00 | 525.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-26 09:45:00 | 529.30 | 2024-04-26 09:50:00 | 531.50 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-04-30 09:40:00 | 536.00 | 2024-04-30 09:50:00 | 533.54 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-05-09 11:10:00 | 520.00 | 2024-05-09 11:40:00 | 516.83 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-05-09 11:10:00 | 520.00 | 2024-05-09 11:45:00 | 520.00 | STOP_HIT | 0.50 | 0.00% |
