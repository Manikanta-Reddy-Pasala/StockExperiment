# Computer Age Management Services Ltd. (CAMS)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2023-11-07 15:25:00 (9226 bars)
- **Last close:** 481.56
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
| ENTRY1 | 42 |
| ENTRY2 | 0 |
| PARTIAL | 16 |
| TARGET_HIT | 4 |
| STOP_HIT | 38 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 58 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 38
- **Target hits / Stop hits / Partials:** 4 / 38 / 16
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 6.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 6 | 27.3% | 1 | 16 | 5 | 0.16% | 3.6% |
| BUY @ 2nd Alert (retest1) | 22 | 6 | 27.3% | 1 | 16 | 5 | 0.16% | 3.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 36 | 14 | 38.9% | 3 | 22 | 11 | 0.07% | 2.5% |
| SELL @ 2nd Alert (retest1) | 36 | 14 | 38.9% | 3 | 22 | 11 | 0.07% | 2.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 58 | 20 | 34.5% | 4 | 38 | 16 | 0.10% | 6.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-12 10:25:00 | 413.78 | 414.67 | 0.00 | ORB-short ORB[415.60,418.81] vol=1.9x ATR=1.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-12 14:55:00 | 411.38 | 413.16 | 0.00 | T1 1.5R @ 411.38 |
| Target hit | 2023-05-12 15:20:00 | 412.18 | 413.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2023-05-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 09:30:00 | 412.19 | 413.46 | 0.00 | ORB-short ORB[413.24,415.60] vol=2.5x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-19 09:35:00 | 411.02 | 412.81 | 0.00 | T1 1.5R @ 411.02 |
| Stop hit — per-position SL triggered | 2023-05-19 10:15:00 | 412.19 | 411.65 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-25 09:55:00 | 428.90 | 430.81 | 0.00 | ORB-short ORB[429.68,434.54] vol=1.6x ATR=1.07 |
| Stop hit — per-position SL triggered | 2023-05-25 12:50:00 | 429.97 | 429.65 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-05-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-26 09:30:00 | 426.76 | 428.66 | 0.00 | ORB-short ORB[428.10,430.94] vol=2.1x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-26 09:45:00 | 425.38 | 427.88 | 0.00 | T1 1.5R @ 425.38 |
| Stop hit — per-position SL triggered | 2023-05-26 09:55:00 | 426.76 | 427.67 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-30 09:40:00 | 429.00 | 428.47 | 0.00 | ORB-long ORB[427.21,428.92] vol=1.5x ATR=0.76 |
| Stop hit — per-position SL triggered | 2023-05-30 09:50:00 | 428.24 | 428.51 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2023-06-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-06 10:50:00 | 435.64 | 436.08 | 0.00 | ORB-short ORB[436.24,439.39] vol=2.4x ATR=0.70 |
| Stop hit — per-position SL triggered | 2023-06-06 12:00:00 | 436.34 | 435.97 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-06-07 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-07 10:20:00 | 438.48 | 436.64 | 0.00 | ORB-long ORB[433.61,437.19] vol=2.9x ATR=0.79 |
| Stop hit — per-position SL triggered | 2023-06-07 11:50:00 | 437.69 | 437.33 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-06-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-09 09:40:00 | 431.17 | 432.04 | 0.00 | ORB-short ORB[431.42,435.76] vol=1.6x ATR=0.86 |
| Stop hit — per-position SL triggered | 2023-06-09 12:40:00 | 432.03 | 431.24 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-06-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-16 09:30:00 | 439.71 | 442.26 | 0.00 | ORB-short ORB[440.64,445.76] vol=1.9x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-16 10:00:00 | 437.69 | 440.95 | 0.00 | T1 1.5R @ 437.69 |
| Stop hit — per-position SL triggered | 2023-06-16 12:35:00 | 439.71 | 440.10 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-06-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-21 11:10:00 | 436.99 | 438.77 | 0.00 | ORB-short ORB[437.02,441.32] vol=2.8x ATR=0.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-21 11:20:00 | 435.79 | 438.35 | 0.00 | T1 1.5R @ 435.79 |
| Stop hit — per-position SL triggered | 2023-06-21 11:55:00 | 436.99 | 438.07 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-06-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-22 10:20:00 | 437.30 | 438.70 | 0.00 | ORB-short ORB[438.03,441.35] vol=1.6x ATR=0.86 |
| Stop hit — per-position SL triggered | 2023-06-22 10:30:00 | 438.16 | 438.61 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-06-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-27 09:55:00 | 429.14 | 429.99 | 0.00 | ORB-short ORB[429.51,432.51] vol=1.8x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-27 10:05:00 | 427.77 | 429.50 | 0.00 | T1 1.5R @ 427.77 |
| Stop hit — per-position SL triggered | 2023-06-27 10:15:00 | 429.14 | 429.36 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-06-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 10:35:00 | 432.85 | 431.74 | 0.00 | ORB-long ORB[428.01,432.76] vol=2.0x ATR=1.07 |
| Stop hit — per-position SL triggered | 2023-06-28 10:40:00 | 431.78 | 431.75 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-07-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 09:30:00 | 460.80 | 458.93 | 0.00 | ORB-long ORB[455.52,460.79] vol=1.9x ATR=1.91 |
| Stop hit — per-position SL triggered | 2023-07-05 09:35:00 | 458.89 | 458.93 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-07-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-12 09:45:00 | 449.82 | 450.36 | 0.00 | ORB-short ORB[450.61,453.46] vol=3.3x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-12 10:15:00 | 448.00 | 449.83 | 0.00 | T1 1.5R @ 448.00 |
| Target hit | 2023-07-12 14:50:00 | 447.64 | 447.38 | 0.00 | Trail-exit close>VWAP |

### Cycle 16 — BUY (started 2023-07-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 09:40:00 | 459.89 | 458.29 | 0.00 | ORB-long ORB[456.08,458.84] vol=1.7x ATR=1.21 |
| Stop hit — per-position SL triggered | 2023-07-17 09:50:00 | 458.68 | 458.58 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-07-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-19 09:30:00 | 454.71 | 456.06 | 0.00 | ORB-short ORB[454.88,459.41] vol=1.6x ATR=1.17 |
| Stop hit — per-position SL triggered | 2023-07-19 09:45:00 | 455.88 | 455.65 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-07-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-25 11:00:00 | 463.90 | 465.34 | 0.00 | ORB-short ORB[465.42,469.96] vol=1.5x ATR=1.08 |
| Stop hit — per-position SL triggered | 2023-07-25 11:25:00 | 464.98 | 464.88 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-07-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 09:50:00 | 468.70 | 467.52 | 0.00 | ORB-long ORB[464.41,468.29] vol=1.8x ATR=1.08 |
| Stop hit — per-position SL triggered | 2023-07-28 10:10:00 | 467.62 | 467.65 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-08-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-01 09:35:00 | 480.20 | 478.42 | 0.00 | ORB-long ORB[473.00,479.75] vol=3.5x ATR=1.66 |
| Stop hit — per-position SL triggered | 2023-08-01 10:05:00 | 478.54 | 479.21 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-08-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-02 09:35:00 | 480.20 | 478.78 | 0.00 | ORB-long ORB[474.41,479.41] vol=3.5x ATR=1.36 |
| Stop hit — per-position SL triggered | 2023-08-02 09:45:00 | 478.84 | 478.90 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-08-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-08 09:35:00 | 472.27 | 475.23 | 0.00 | ORB-short ORB[473.69,480.00] vol=1.9x ATR=1.77 |
| Stop hit — per-position SL triggered | 2023-08-08 09:40:00 | 474.04 | 475.07 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-08-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-11 09:45:00 | 481.83 | 483.22 | 0.00 | ORB-short ORB[482.00,485.92] vol=2.6x ATR=1.17 |
| Stop hit — per-position SL triggered | 2023-08-11 09:55:00 | 483.00 | 483.19 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-08-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-18 09:35:00 | 501.70 | 505.40 | 0.00 | ORB-short ORB[503.71,509.04] vol=1.6x ATR=1.34 |
| Stop hit — per-position SL triggered | 2023-08-18 09:40:00 | 503.04 | 504.97 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-08-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-21 10:10:00 | 500.00 | 503.04 | 0.00 | ORB-short ORB[503.10,508.99] vol=1.7x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-21 10:30:00 | 497.53 | 502.16 | 0.00 | T1 1.5R @ 497.53 |
| Stop hit — per-position SL triggered | 2023-08-21 13:15:00 | 500.00 | 499.52 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-08-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-24 10:55:00 | 495.84 | 498.21 | 0.00 | ORB-short ORB[497.28,503.36] vol=2.7x ATR=1.05 |
| Stop hit — per-position SL triggered | 2023-08-24 11:00:00 | 496.89 | 498.19 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2023-08-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-29 11:05:00 | 482.23 | 480.52 | 0.00 | ORB-long ORB[475.99,481.00] vol=6.2x ATR=1.03 |
| Stop hit — per-position SL triggered | 2023-08-29 11:10:00 | 481.20 | 480.58 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-08-31 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-31 10:05:00 | 475.88 | 477.49 | 0.00 | ORB-short ORB[476.80,479.40] vol=1.9x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-31 12:30:00 | 474.27 | 476.33 | 0.00 | T1 1.5R @ 474.27 |
| Stop hit — per-position SL triggered | 2023-08-31 15:00:00 | 475.88 | 475.61 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-09-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-01 10:35:00 | 473.37 | 474.78 | 0.00 | ORB-short ORB[474.44,477.97] vol=2.0x ATR=0.92 |
| Stop hit — per-position SL triggered | 2023-09-01 10:50:00 | 474.29 | 474.68 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-09-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 10:30:00 | 479.53 | 474.87 | 0.00 | ORB-long ORB[473.00,476.48] vol=2.3x ATR=1.19 |
| Stop hit — per-position SL triggered | 2023-09-04 10:35:00 | 478.34 | 474.92 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2023-09-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 10:40:00 | 497.58 | 493.84 | 0.00 | ORB-long ORB[489.60,495.60] vol=2.4x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-06 10:45:00 | 499.91 | 495.12 | 0.00 | T1 1.5R @ 499.91 |
| Stop hit — per-position SL triggered | 2023-09-06 10:50:00 | 497.58 | 495.22 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-09-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-07 09:30:00 | 497.76 | 495.15 | 0.00 | ORB-long ORB[492.25,496.75] vol=2.4x ATR=1.40 |
| Stop hit — per-position SL triggered | 2023-09-07 09:40:00 | 496.36 | 495.52 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2023-09-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-20 09:55:00 | 502.00 | 503.35 | 0.00 | ORB-short ORB[502.47,509.59] vol=2.4x ATR=1.67 |
| Stop hit — per-position SL triggered | 2023-09-20 10:00:00 | 503.67 | 503.47 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2023-09-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 09:30:00 | 496.90 | 500.72 | 0.00 | ORB-short ORB[498.20,505.00] vol=1.7x ATR=2.18 |
| Stop hit — per-position SL triggered | 2023-09-22 09:35:00 | 499.08 | 500.54 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2023-09-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 11:00:00 | 494.78 | 491.57 | 0.00 | ORB-long ORB[487.39,493.73] vol=4.3x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-29 11:05:00 | 496.80 | 492.13 | 0.00 | T1 1.5R @ 496.80 |
| Stop hit — per-position SL triggered | 2023-09-29 11:50:00 | 494.78 | 492.62 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-10-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 09:30:00 | 490.21 | 488.39 | 0.00 | ORB-long ORB[485.34,490.00] vol=3.3x ATR=1.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-12 09:35:00 | 492.41 | 489.52 | 0.00 | T1 1.5R @ 492.41 |
| Target hit | 2023-10-12 15:00:00 | 511.40 | 511.92 | 0.00 | Trail-exit close<VWAP |

### Cycle 37 — BUY (started 2023-10-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-18 09:35:00 | 522.20 | 519.11 | 0.00 | ORB-long ORB[515.82,520.56] vol=4.4x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 09:40:00 | 525.30 | 521.80 | 0.00 | T1 1.5R @ 525.30 |
| Stop hit — per-position SL triggered | 2023-10-18 10:00:00 | 522.20 | 522.74 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2023-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-23 10:15:00 | 492.71 | 498.00 | 0.00 | ORB-short ORB[498.01,503.39] vol=3.1x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 10:20:00 | 490.11 | 497.60 | 0.00 | T1 1.5R @ 490.11 |
| Stop hit — per-position SL triggered | 2023-10-23 10:25:00 | 492.71 | 497.30 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2023-10-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-30 11:00:00 | 457.02 | 462.19 | 0.00 | ORB-short ORB[462.38,467.70] vol=2.1x ATR=1.13 |
| Stop hit — per-position SL triggered | 2023-10-30 11:25:00 | 458.15 | 461.59 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2023-10-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-31 09:35:00 | 454.19 | 455.59 | 0.00 | ORB-short ORB[454.37,458.79] vol=2.3x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-31 10:05:00 | 452.43 | 454.82 | 0.00 | T1 1.5R @ 452.43 |
| Target hit | 2023-10-31 15:20:00 | 450.41 | 451.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — BUY (started 2023-11-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 09:45:00 | 457.20 | 455.34 | 0.00 | ORB-long ORB[451.51,455.79] vol=2.4x ATR=1.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-02 10:10:00 | 459.76 | 456.45 | 0.00 | T1 1.5R @ 459.76 |
| Stop hit — per-position SL triggered | 2023-11-02 10:30:00 | 457.20 | 456.86 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2023-11-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 09:40:00 | 459.01 | 456.77 | 0.00 | ORB-long ORB[453.33,458.96] vol=1.8x ATR=1.46 |
| Stop hit — per-position SL triggered | 2023-11-03 09:55:00 | 457.55 | 457.16 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-12 10:25:00 | 413.78 | 2023-05-12 14:55:00 | 411.38 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2023-05-12 10:25:00 | 413.78 | 2023-05-12 15:20:00 | 412.18 | TARGET_HIT | 0.50 | 0.39% |
| SELL | retest1 | 2023-05-19 09:30:00 | 412.19 | 2023-05-19 09:35:00 | 411.02 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2023-05-19 09:30:00 | 412.19 | 2023-05-19 10:15:00 | 412.19 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-05-25 09:55:00 | 428.90 | 2023-05-25 12:50:00 | 429.97 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-05-26 09:30:00 | 426.76 | 2023-05-26 09:45:00 | 425.38 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2023-05-26 09:30:00 | 426.76 | 2023-05-26 09:55:00 | 426.76 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-30 09:40:00 | 429.00 | 2023-05-30 09:50:00 | 428.24 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-06-06 10:50:00 | 435.64 | 2023-06-06 12:00:00 | 436.34 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-06-07 10:20:00 | 438.48 | 2023-06-07 11:50:00 | 437.69 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-06-09 09:40:00 | 431.17 | 2023-06-09 12:40:00 | 432.03 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-06-16 09:30:00 | 439.71 | 2023-06-16 10:00:00 | 437.69 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2023-06-16 09:30:00 | 439.71 | 2023-06-16 12:35:00 | 439.71 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-21 11:10:00 | 436.99 | 2023-06-21 11:20:00 | 435.79 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-06-21 11:10:00 | 436.99 | 2023-06-21 11:55:00 | 436.99 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-22 10:20:00 | 437.30 | 2023-06-22 10:30:00 | 438.16 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-06-27 09:55:00 | 429.14 | 2023-06-27 10:05:00 | 427.77 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2023-06-27 09:55:00 | 429.14 | 2023-06-27 10:15:00 | 429.14 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-28 10:35:00 | 432.85 | 2023-06-28 10:40:00 | 431.78 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-07-05 09:30:00 | 460.80 | 2023-07-05 09:35:00 | 458.89 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2023-07-12 09:45:00 | 449.82 | 2023-07-12 10:15:00 | 448.00 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2023-07-12 09:45:00 | 449.82 | 2023-07-12 14:50:00 | 447.64 | TARGET_HIT | 0.50 | 0.48% |
| BUY | retest1 | 2023-07-17 09:40:00 | 459.89 | 2023-07-17 09:50:00 | 458.68 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-07-19 09:30:00 | 454.71 | 2023-07-19 09:45:00 | 455.88 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-07-25 11:00:00 | 463.90 | 2023-07-25 11:25:00 | 464.98 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-07-28 09:50:00 | 468.70 | 2023-07-28 10:10:00 | 467.62 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-08-01 09:35:00 | 480.20 | 2023-08-01 10:05:00 | 478.54 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-08-02 09:35:00 | 480.20 | 2023-08-02 09:45:00 | 478.84 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-08-08 09:35:00 | 472.27 | 2023-08-08 09:40:00 | 474.04 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2023-08-11 09:45:00 | 481.83 | 2023-08-11 09:55:00 | 483.00 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-08-18 09:35:00 | 501.70 | 2023-08-18 09:40:00 | 503.04 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-08-21 10:10:00 | 500.00 | 2023-08-21 10:30:00 | 497.53 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2023-08-21 10:10:00 | 500.00 | 2023-08-21 13:15:00 | 500.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-24 10:55:00 | 495.84 | 2023-08-24 11:00:00 | 496.89 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-08-29 11:05:00 | 482.23 | 2023-08-29 11:10:00 | 481.20 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-08-31 10:05:00 | 475.88 | 2023-08-31 12:30:00 | 474.27 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-08-31 10:05:00 | 475.88 | 2023-08-31 15:00:00 | 475.88 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-01 10:35:00 | 473.37 | 2023-09-01 10:50:00 | 474.29 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-09-04 10:30:00 | 479.53 | 2023-09-04 10:35:00 | 478.34 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-09-06 10:40:00 | 497.58 | 2023-09-06 10:45:00 | 499.91 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2023-09-06 10:40:00 | 497.58 | 2023-09-06 10:50:00 | 497.58 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-07 09:30:00 | 497.76 | 2023-09-07 09:40:00 | 496.36 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-09-20 09:55:00 | 502.00 | 2023-09-20 10:00:00 | 503.67 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-09-22 09:30:00 | 496.90 | 2023-09-22 09:35:00 | 499.08 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2023-09-29 11:00:00 | 494.78 | 2023-09-29 11:05:00 | 496.80 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-09-29 11:00:00 | 494.78 | 2023-09-29 11:50:00 | 494.78 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-12 09:30:00 | 490.21 | 2023-10-12 09:35:00 | 492.41 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2023-10-12 09:30:00 | 490.21 | 2023-10-12 15:00:00 | 511.40 | TARGET_HIT | 0.50 | 4.32% |
| BUY | retest1 | 2023-10-18 09:35:00 | 522.20 | 2023-10-18 09:40:00 | 525.30 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2023-10-18 09:35:00 | 522.20 | 2023-10-18 10:00:00 | 522.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-23 10:15:00 | 492.71 | 2023-10-23 10:20:00 | 490.11 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2023-10-23 10:15:00 | 492.71 | 2023-10-23 10:25:00 | 492.71 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-30 11:00:00 | 457.02 | 2023-10-30 11:25:00 | 458.15 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-10-31 09:35:00 | 454.19 | 2023-10-31 10:05:00 | 452.43 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-10-31 09:35:00 | 454.19 | 2023-10-31 15:20:00 | 450.41 | TARGET_HIT | 0.50 | 0.83% |
| BUY | retest1 | 2023-11-02 09:45:00 | 457.20 | 2023-11-02 10:10:00 | 459.76 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2023-11-02 09:45:00 | 457.20 | 2023-11-02 10:30:00 | 457.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-03 09:40:00 | 459.01 | 2023-11-03 09:55:00 | 457.55 | STOP_HIT | 1.00 | -0.32% |
