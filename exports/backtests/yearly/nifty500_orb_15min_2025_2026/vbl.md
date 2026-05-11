# Varun Beverages Ltd. (VBL)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 508.35
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
| ENTRY1 | 87 |
| ENTRY2 | 0 |
| PARTIAL | 31 |
| TARGET_HIT | 11 |
| STOP_HIT | 76 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 118 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 42 / 76
- **Target hits / Stop hits / Partials:** 11 / 76 / 31
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 9.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 61 | 19 | 31.1% | 4 | 42 | 15 | 0.06% | 3.9% |
| BUY @ 2nd Alert (retest1) | 61 | 19 | 31.1% | 4 | 42 | 15 | 0.06% | 3.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 57 | 23 | 40.4% | 7 | 34 | 16 | 0.10% | 5.4% |
| SELL @ 2nd Alert (retest1) | 57 | 23 | 40.4% | 7 | 34 | 16 | 0.10% | 5.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 118 | 42 | 35.6% | 11 | 76 | 31 | 0.08% | 9.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-13 10:55:00 | 509.95 | 513.20 | 0.00 | ORB-short ORB[513.00,518.00] vol=4.4x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-13 11:40:00 | 507.96 | 512.08 | 0.00 | T1 1.5R @ 507.96 |
| Stop hit — per-position SL triggered | 2025-05-13 13:45:00 | 509.95 | 511.46 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 09:55:00 | 514.40 | 512.30 | 0.00 | ORB-long ORB[508.50,513.45] vol=1.7x ATR=1.21 |
| Stop hit — per-position SL triggered | 2025-05-14 10:25:00 | 513.19 | 512.80 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 09:30:00 | 517.00 | 514.17 | 0.00 | ORB-long ORB[510.75,515.30] vol=1.8x ATR=1.22 |
| Stop hit — per-position SL triggered | 2025-05-15 09:35:00 | 515.78 | 514.25 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-16 10:35:00 | 500.95 | 504.30 | 0.00 | ORB-short ORB[503.10,507.30] vol=1.6x ATR=1.40 |
| Stop hit — per-position SL triggered | 2025-05-16 10:45:00 | 502.35 | 503.88 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 11:15:00 | 479.70 | 481.37 | 0.00 | ORB-short ORB[480.70,485.75] vol=1.6x ATR=0.66 |
| Stop hit — per-position SL triggered | 2025-05-29 11:20:00 | 480.36 | 481.33 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 11:15:00 | 477.85 | 480.51 | 0.00 | ORB-short ORB[479.50,484.80] vol=1.6x ATR=0.77 |
| Stop hit — per-position SL triggered | 2025-06-03 11:20:00 | 478.62 | 480.47 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 09:35:00 | 470.50 | 472.56 | 0.00 | ORB-short ORB[472.10,477.00] vol=1.8x ATR=1.18 |
| Stop hit — per-position SL triggered | 2025-06-04 09:45:00 | 471.68 | 471.89 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 10:20:00 | 469.85 | 469.70 | 0.00 | ORB-long ORB[466.95,469.80] vol=1.9x ATR=1.03 |
| Stop hit — per-position SL triggered | 2025-06-05 10:30:00 | 468.82 | 469.66 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 09:45:00 | 477.20 | 474.95 | 0.00 | ORB-long ORB[468.25,474.80] vol=2.4x ATR=1.40 |
| Stop hit — per-position SL triggered | 2025-06-06 10:00:00 | 475.80 | 475.44 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 09:50:00 | 482.80 | 481.39 | 0.00 | ORB-long ORB[477.60,482.60] vol=2.0x ATR=1.45 |
| Stop hit — per-position SL triggered | 2025-06-10 10:10:00 | 481.35 | 481.52 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-13 09:30:00 | 466.75 | 469.08 | 0.00 | ORB-short ORB[468.30,471.75] vol=1.7x ATR=1.57 |
| Stop hit — per-position SL triggered | 2025-06-13 09:35:00 | 468.32 | 468.94 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-06-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:30:00 | 461.80 | 465.60 | 0.00 | ORB-short ORB[463.80,469.00] vol=1.7x ATR=1.45 |
| Stop hit — per-position SL triggered | 2025-06-16 09:35:00 | 463.25 | 465.30 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:55:00 | 454.95 | 456.91 | 0.00 | ORB-short ORB[457.40,459.85] vol=2.0x ATR=0.77 |
| Stop hit — per-position SL triggered | 2025-07-01 11:05:00 | 455.72 | 456.76 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 10:25:00 | 455.00 | 451.99 | 0.00 | ORB-long ORB[449.30,452.25] vol=2.5x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-04 11:25:00 | 456.36 | 453.54 | 0.00 | T1 1.5R @ 456.36 |
| Stop hit — per-position SL triggered | 2025-07-04 11:35:00 | 455.00 | 453.70 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 10:15:00 | 458.50 | 454.25 | 0.00 | ORB-long ORB[452.30,456.60] vol=2.0x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 11:35:00 | 460.06 | 456.94 | 0.00 | T1 1.5R @ 460.06 |
| Target hit | 2025-07-07 15:20:00 | 463.85 | 460.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2025-07-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 10:50:00 | 459.70 | 462.47 | 0.00 | ORB-short ORB[462.70,466.55] vol=1.6x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 11:00:00 | 458.10 | 462.13 | 0.00 | T1 1.5R @ 458.10 |
| Stop hit — per-position SL triggered | 2025-07-08 11:05:00 | 459.70 | 461.62 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:45:00 | 453.60 | 458.54 | 0.00 | ORB-short ORB[457.00,461.35] vol=2.1x ATR=1.27 |
| Stop hit — per-position SL triggered | 2025-07-11 10:50:00 | 454.87 | 458.36 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 09:40:00 | 455.70 | 453.83 | 0.00 | ORB-long ORB[450.00,454.45] vol=2.0x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 10:40:00 | 457.38 | 455.69 | 0.00 | T1 1.5R @ 457.38 |
| Stop hit — per-position SL triggered | 2025-07-14 11:10:00 | 455.70 | 455.75 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 10:35:00 | 466.90 | 465.58 | 0.00 | ORB-long ORB[462.40,465.35] vol=7.5x ATR=1.03 |
| Stop hit — per-position SL triggered | 2025-07-15 10:40:00 | 465.87 | 465.58 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-16 09:40:00 | 465.20 | 466.38 | 0.00 | ORB-short ORB[466.65,470.45] vol=10.6x ATR=0.99 |
| Stop hit — per-position SL triggered | 2025-07-16 09:55:00 | 466.19 | 466.36 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-07-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-17 10:40:00 | 489.25 | 485.07 | 0.00 | ORB-long ORB[481.30,486.50] vol=1.8x ATR=1.45 |
| Stop hit — per-position SL triggered | 2025-07-17 11:10:00 | 487.80 | 486.31 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-07-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 11:05:00 | 488.00 | 486.27 | 0.00 | ORB-long ORB[483.65,487.85] vol=1.8x ATR=1.00 |
| Stop hit — per-position SL triggered | 2025-07-21 11:15:00 | 487.00 | 486.40 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 09:40:00 | 484.30 | 485.64 | 0.00 | ORB-short ORB[484.80,489.65] vol=1.6x ATR=0.97 |
| Stop hit — per-position SL triggered | 2025-07-24 09:45:00 | 485.27 | 485.62 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-07-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 09:50:00 | 485.45 | 482.09 | 0.00 | ORB-long ORB[473.15,479.00] vol=3.6x ATR=1.52 |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 483.93 | 483.23 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-07-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-29 09:40:00 | 490.00 | 487.42 | 0.00 | ORB-long ORB[483.25,489.40] vol=1.7x ATR=1.66 |
| Stop hit — per-position SL triggered | 2025-07-29 09:50:00 | 488.34 | 487.89 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-08-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-01 10:30:00 | 514.65 | 517.80 | 0.00 | ORB-short ORB[517.80,524.20] vol=1.9x ATR=1.49 |
| Stop hit — per-position SL triggered | 2025-08-01 11:05:00 | 516.14 | 516.98 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-08-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-04 10:00:00 | 505.05 | 506.87 | 0.00 | ORB-short ORB[505.50,511.55] vol=1.9x ATR=1.75 |
| Stop hit — per-position SL triggered | 2025-08-04 10:40:00 | 506.80 | 506.22 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-08-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 11:00:00 | 500.90 | 504.24 | 0.00 | ORB-short ORB[505.05,507.95] vol=3.4x ATR=1.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 11:15:00 | 499.30 | 503.67 | 0.00 | T1 1.5R @ 499.30 |
| Stop hit — per-position SL triggered | 2025-08-06 11:20:00 | 500.90 | 503.54 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-08-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-11 09:55:00 | 499.50 | 496.71 | 0.00 | ORB-long ORB[493.60,499.20] vol=4.1x ATR=2.09 |
| Stop hit — per-position SL triggered | 2025-08-11 10:05:00 | 497.41 | 496.89 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-08-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 09:40:00 | 517.25 | 515.80 | 0.00 | ORB-long ORB[513.15,517.10] vol=2.2x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 09:55:00 | 519.26 | 516.45 | 0.00 | T1 1.5R @ 519.26 |
| Stop hit — per-position SL triggered | 2025-08-13 10:05:00 | 517.25 | 516.52 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-08-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 10:40:00 | 501.00 | 503.81 | 0.00 | ORB-short ORB[506.75,510.85] vol=1.6x ATR=1.37 |
| Stop hit — per-position SL triggered | 2025-08-14 12:00:00 | 502.37 | 502.06 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-08-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 09:50:00 | 512.00 | 509.20 | 0.00 | ORB-long ORB[504.80,508.95] vol=1.6x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 10:00:00 | 514.09 | 510.38 | 0.00 | T1 1.5R @ 514.09 |
| Stop hit — per-position SL triggered | 2025-08-20 10:05:00 | 512.00 | 510.58 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-08-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:40:00 | 508.95 | 510.74 | 0.00 | ORB-short ORB[510.00,516.20] vol=1.5x ATR=1.23 |
| Stop hit — per-position SL triggered | 2025-08-26 09:45:00 | 510.18 | 510.67 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-09-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-01 10:30:00 | 482.90 | 485.90 | 0.00 | ORB-short ORB[483.10,486.90] vol=1.7x ATR=1.45 |
| Stop hit — per-position SL triggered | 2025-09-01 12:25:00 | 484.35 | 484.56 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-09-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 11:00:00 | 470.40 | 472.92 | 0.00 | ORB-short ORB[472.55,476.35] vol=3.6x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 12:55:00 | 469.15 | 471.61 | 0.00 | T1 1.5R @ 469.15 |
| Stop hit — per-position SL triggered | 2025-09-12 13:20:00 | 470.40 | 471.48 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-16 11:15:00 | 467.20 | 468.23 | 0.00 | ORB-short ORB[469.00,472.70] vol=2.2x ATR=0.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 13:00:00 | 466.40 | 467.89 | 0.00 | T1 1.5R @ 466.40 |
| Target hit | 2025-09-16 15:20:00 | 463.60 | 465.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — BUY (started 2025-09-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 10:05:00 | 468.45 | 466.97 | 0.00 | ORB-long ORB[463.15,467.40] vol=1.6x ATR=1.36 |
| Stop hit — per-position SL triggered | 2025-09-17 10:10:00 | 467.09 | 466.98 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-09-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 11:10:00 | 468.35 | 468.70 | 0.00 | ORB-short ORB[468.45,473.50] vol=1.5x ATR=0.84 |
| Stop hit — per-position SL triggered | 2025-09-18 11:25:00 | 469.19 | 468.73 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-09-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 11:10:00 | 461.35 | 462.53 | 0.00 | ORB-short ORB[463.70,466.40] vol=4.0x ATR=0.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 12:45:00 | 459.93 | 462.03 | 0.00 | T1 1.5R @ 459.93 |
| Target hit | 2025-09-23 15:20:00 | 458.55 | 460.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — SELL (started 2025-09-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 09:40:00 | 454.70 | 456.47 | 0.00 | ORB-short ORB[454.80,460.95] vol=2.0x ATR=1.18 |
| Stop hit — per-position SL triggered | 2025-09-24 09:50:00 | 455.88 | 456.37 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-30 11:15:00 | 446.20 | 446.84 | 0.00 | ORB-short ORB[447.45,454.00] vol=1.9x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 12:45:00 | 444.51 | 446.53 | 0.00 | T1 1.5R @ 444.51 |
| Target hit | 2025-09-30 15:20:00 | 443.30 | 444.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — SELL (started 2025-10-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 09:35:00 | 437.50 | 438.84 | 0.00 | ORB-short ORB[438.10,440.85] vol=1.8x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 09:50:00 | 435.99 | 438.17 | 0.00 | T1 1.5R @ 435.99 |
| Target hit | 2025-10-08 15:05:00 | 435.00 | 434.83 | 0.00 | Trail-exit close>VWAP |

### Cycle 43 — BUY (started 2025-10-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 11:00:00 | 449.50 | 447.11 | 0.00 | ORB-long ORB[443.35,447.10] vol=2.8x ATR=1.42 |
| Stop hit — per-position SL triggered | 2025-10-10 11:30:00 | 448.08 | 447.47 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-10-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 10:55:00 | 439.20 | 441.71 | 0.00 | ORB-short ORB[442.45,444.90] vol=1.7x ATR=0.80 |
| Stop hit — per-position SL triggered | 2025-10-14 11:20:00 | 440.00 | 441.36 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-10-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 09:30:00 | 451.75 | 449.91 | 0.00 | ORB-long ORB[445.35,450.85] vol=2.7x ATR=1.14 |
| Stop hit — per-position SL triggered | 2025-10-16 09:35:00 | 450.61 | 450.16 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-10-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 09:35:00 | 463.85 | 461.20 | 0.00 | ORB-long ORB[458.45,462.70] vol=2.6x ATR=1.39 |
| Stop hit — per-position SL triggered | 2025-10-17 09:40:00 | 462.46 | 462.02 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-10-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 09:35:00 | 455.00 | 457.24 | 0.00 | ORB-short ORB[456.30,461.30] vol=2.4x ATR=1.32 |
| Stop hit — per-position SL triggered | 2025-10-28 09:45:00 | 456.32 | 456.49 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-10-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 09:55:00 | 463.00 | 460.53 | 0.00 | ORB-long ORB[455.00,461.25] vol=1.5x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 10:00:00 | 465.36 | 461.62 | 0.00 | T1 1.5R @ 465.36 |
| Stop hit — per-position SL triggered | 2025-10-29 11:55:00 | 463.00 | 464.47 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-11-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 11:00:00 | 470.45 | 472.08 | 0.00 | ORB-short ORB[471.60,475.00] vol=1.6x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 11:20:00 | 469.18 | 471.56 | 0.00 | T1 1.5R @ 469.18 |
| Stop hit — per-position SL triggered | 2025-11-04 11:25:00 | 470.45 | 471.52 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-11-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 09:30:00 | 464.45 | 465.89 | 0.00 | ORB-short ORB[464.50,470.60] vol=1.5x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 10:15:00 | 462.26 | 464.64 | 0.00 | T1 1.5R @ 462.26 |
| Target hit | 2025-11-12 15:20:00 | 459.20 | 461.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — SELL (started 2025-11-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-13 09:40:00 | 456.20 | 457.59 | 0.00 | ORB-short ORB[456.40,460.40] vol=1.7x ATR=1.01 |
| Stop hit — per-position SL triggered | 2025-11-13 11:05:00 | 457.21 | 456.60 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-11-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 09:45:00 | 452.60 | 453.42 | 0.00 | ORB-short ORB[453.45,455.25] vol=2.6x ATR=0.78 |
| Stop hit — per-position SL triggered | 2025-11-20 10:00:00 | 453.38 | 453.36 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-24 11:15:00 | 447.95 | 446.31 | 0.00 | ORB-long ORB[442.15,447.65] vol=1.9x ATR=0.88 |
| Stop hit — per-position SL triggered | 2025-11-24 11:30:00 | 447.07 | 446.39 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-11-26 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 09:55:00 | 453.50 | 451.22 | 0.00 | ORB-long ORB[446.75,450.15] vol=1.7x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-26 10:15:00 | 455.04 | 452.71 | 0.00 | T1 1.5R @ 455.04 |
| Target hit | 2025-11-26 15:20:00 | 465.40 | 462.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — BUY (started 2025-12-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 11:05:00 | 485.80 | 483.64 | 0.00 | ORB-long ORB[480.45,485.45] vol=1.5x ATR=1.07 |
| Stop hit — per-position SL triggered | 2025-12-01 11:10:00 | 484.73 | 483.71 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-12-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 09:30:00 | 475.30 | 476.78 | 0.00 | ORB-short ORB[475.80,481.40] vol=1.6x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 10:00:00 | 473.60 | 476.11 | 0.00 | T1 1.5R @ 473.60 |
| Stop hit — per-position SL triggered | 2025-12-03 10:20:00 | 475.30 | 475.59 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-12-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-16 09:40:00 | 481.45 | 477.97 | 0.00 | ORB-long ORB[473.50,478.90] vol=1.5x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 09:45:00 | 483.89 | 479.31 | 0.00 | T1 1.5R @ 483.89 |
| Stop hit — per-position SL triggered | 2025-12-16 10:00:00 | 481.45 | 480.19 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-12-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-19 11:00:00 | 469.40 | 472.95 | 0.00 | ORB-short ORB[470.70,476.85] vol=2.2x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-19 11:40:00 | 467.56 | 471.79 | 0.00 | T1 1.5R @ 467.56 |
| Stop hit — per-position SL triggered | 2025-12-19 12:00:00 | 469.40 | 471.28 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-12-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 11:00:00 | 475.00 | 477.78 | 0.00 | ORB-short ORB[476.60,482.00] vol=1.8x ATR=1.16 |
| Stop hit — per-position SL triggered | 2025-12-24 11:15:00 | 476.16 | 477.65 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-12-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 09:45:00 | 479.15 | 478.04 | 0.00 | ORB-long ORB[475.10,479.10] vol=10.6x ATR=1.24 |
| Stop hit — per-position SL triggered | 2025-12-26 10:00:00 | 477.91 | 478.14 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-12-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-29 09:30:00 | 487.75 | 485.90 | 0.00 | ORB-long ORB[483.05,486.40] vol=2.5x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 09:35:00 | 489.71 | 486.89 | 0.00 | T1 1.5R @ 489.71 |
| Target hit | 2025-12-29 10:10:00 | 489.65 | 490.29 | 0.00 | Trail-exit close<VWAP |

### Cycle 62 — BUY (started 2025-12-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 10:05:00 | 485.00 | 482.07 | 0.00 | ORB-long ORB[480.25,483.65] vol=1.8x ATR=1.44 |
| Stop hit — per-position SL triggered | 2025-12-30 13:55:00 | 483.56 | 484.40 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-12-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-31 11:05:00 | 484.85 | 485.49 | 0.00 | ORB-short ORB[485.05,487.25] vol=5.7x ATR=1.00 |
| Stop hit — per-position SL triggered | 2025-12-31 11:10:00 | 485.85 | 485.45 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2026-01-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:10:00 | 502.05 | 506.16 | 0.00 | ORB-short ORB[505.50,511.65] vol=1.9x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:55:00 | 500.06 | 505.51 | 0.00 | T1 1.5R @ 500.06 |
| Stop hit — per-position SL triggered | 2026-01-08 12:10:00 | 502.05 | 505.20 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-01-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-12 09:30:00 | 494.95 | 491.21 | 0.00 | ORB-long ORB[486.25,491.80] vol=2.7x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:40:00 | 498.01 | 492.85 | 0.00 | T1 1.5R @ 498.01 |
| Stop hit — per-position SL triggered | 2026-01-12 09:45:00 | 494.95 | 493.41 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-01-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-13 10:30:00 | 496.80 | 494.08 | 0.00 | ORB-long ORB[492.05,496.00] vol=2.7x ATR=1.61 |
| Stop hit — per-position SL triggered | 2026-01-13 10:35:00 | 495.19 | 494.18 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2026-01-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 09:35:00 | 506.65 | 503.80 | 0.00 | ORB-long ORB[497.85,505.30] vol=1.7x ATR=1.68 |
| Stop hit — per-position SL triggered | 2026-01-14 09:45:00 | 504.97 | 505.01 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-19 10:15:00 | 498.35 | 494.65 | 0.00 | ORB-long ORB[489.40,495.35] vol=3.5x ATR=1.98 |
| Stop hit — per-position SL triggered | 2026-01-19 10:25:00 | 496.37 | 494.71 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2026-01-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-22 09:35:00 | 485.60 | 483.39 | 0.00 | ORB-long ORB[480.25,483.55] vol=1.7x ATR=1.63 |
| Stop hit — per-position SL triggered | 2026-01-22 10:00:00 | 483.97 | 484.15 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-01-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 09:45:00 | 469.30 | 470.20 | 0.00 | ORB-short ORB[469.40,473.45] vol=1.7x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-28 10:05:00 | 466.71 | 469.72 | 0.00 | T1 1.5R @ 466.71 |
| Target hit | 2026-01-28 14:05:00 | 466.40 | 465.41 | 0.00 | Trail-exit close>VWAP |

### Cycle 71 — BUY (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 11:15:00 | 473.45 | 470.45 | 0.00 | ORB-long ORB[467.05,471.30] vol=10.0x ATR=1.44 |
| Stop hit — per-position SL triggered | 2026-02-01 11:40:00 | 472.01 | 470.68 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2026-02-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 10:55:00 | 438.85 | 443.14 | 0.00 | ORB-short ORB[444.00,448.90] vol=2.3x ATR=1.27 |
| Stop hit — per-position SL triggered | 2026-02-05 11:15:00 | 440.12 | 442.67 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-02-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 11:05:00 | 438.15 | 434.59 | 0.00 | ORB-long ORB[433.50,438.00] vol=2.1x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-06 11:15:00 | 439.84 | 435.43 | 0.00 | T1 1.5R @ 439.84 |
| Stop hit — per-position SL triggered | 2026-02-06 13:25:00 | 438.15 | 438.12 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2026-02-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:40:00 | 445.00 | 442.62 | 0.00 | ORB-long ORB[439.10,443.45] vol=1.7x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:05:00 | 446.58 | 443.30 | 0.00 | T1 1.5R @ 446.58 |
| Target hit | 2026-02-09 15:20:00 | 457.05 | 452.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — BUY (started 2026-02-10 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:20:00 | 460.35 | 458.35 | 0.00 | ORB-long ORB[454.70,460.30] vol=3.0x ATR=1.52 |
| Stop hit — per-position SL triggered | 2026-02-10 10:40:00 | 458.83 | 458.56 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2026-02-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:45:00 | 451.70 | 454.27 | 0.00 | ORB-short ORB[455.00,458.20] vol=3.6x ATR=0.91 |
| Stop hit — per-position SL triggered | 2026-02-12 11:00:00 | 452.61 | 453.73 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 450.05 | 452.20 | 0.00 | ORB-short ORB[450.70,453.75] vol=1.5x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 09:40:00 | 448.11 | 451.42 | 0.00 | T1 1.5R @ 448.11 |
| Stop hit — per-position SL triggered | 2026-02-13 09:45:00 | 450.05 | 451.34 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2026-02-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 11:00:00 | 457.50 | 456.64 | 0.00 | ORB-long ORB[454.35,456.80] vol=2.0x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 12:50:00 | 458.94 | 457.46 | 0.00 | T1 1.5R @ 458.94 |
| Stop hit — per-position SL triggered | 2026-02-17 13:00:00 | 457.50 | 457.54 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:15:00 | 458.60 | 460.66 | 0.00 | ORB-short ORB[460.75,464.00] vol=2.9x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 12:05:00 | 457.09 | 459.86 | 0.00 | T1 1.5R @ 457.09 |
| Target hit | 2026-02-19 15:20:00 | 452.20 | 457.00 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 80 — BUY (started 2026-02-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:20:00 | 460.65 | 458.05 | 0.00 | ORB-long ORB[455.75,458.80] vol=1.9x ATR=1.12 |
| Stop hit — per-position SL triggered | 2026-02-26 10:30:00 | 459.53 | 458.32 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2026-03-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-05 11:05:00 | 438.35 | 436.20 | 0.00 | ORB-long ORB[432.60,437.60] vol=1.5x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:20:00 | 440.60 | 437.40 | 0.00 | T1 1.5R @ 440.60 |
| Stop hit — per-position SL triggered | 2026-03-05 11:40:00 | 438.35 | 437.58 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2026-03-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:35:00 | 439.35 | 438.01 | 0.00 | ORB-long ORB[434.30,438.80] vol=9.6x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 10:00:00 | 441.58 | 438.62 | 0.00 | T1 1.5R @ 441.58 |
| Stop hit — per-position SL triggered | 2026-03-11 10:05:00 | 439.35 | 438.71 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2026-03-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 09:30:00 | 409.20 | 407.89 | 0.00 | ORB-long ORB[405.80,409.10] vol=1.6x ATR=1.18 |
| Stop hit — per-position SL triggered | 2026-03-20 09:35:00 | 408.02 | 407.90 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2026-03-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:30:00 | 397.30 | 395.80 | 0.00 | ORB-long ORB[391.10,397.00] vol=1.5x ATR=1.56 |
| Stop hit — per-position SL triggered | 2026-03-25 09:35:00 | 395.74 | 395.92 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2026-04-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 11:10:00 | 455.70 | 451.15 | 0.00 | ORB-long ORB[446.55,453.00] vol=1.9x ATR=1.39 |
| Stop hit — per-position SL triggered | 2026-04-16 11:25:00 | 454.31 | 451.72 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2026-04-22 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:00:00 | 492.90 | 490.13 | 0.00 | ORB-long ORB[484.35,491.45] vol=2.0x ATR=2.04 |
| Stop hit — per-position SL triggered | 2026-04-22 10:10:00 | 490.86 | 490.26 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2026-04-29 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:20:00 | 527.35 | 523.60 | 0.00 | ORB-long ORB[520.05,524.80] vol=3.2x ATR=1.86 |
| Stop hit — per-position SL triggered | 2026-04-29 10:30:00 | 525.49 | 524.32 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-13 10:55:00 | 509.95 | 2025-05-13 11:40:00 | 507.96 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-05-13 10:55:00 | 509.95 | 2025-05-13 13:45:00 | 509.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-14 09:55:00 | 514.40 | 2025-05-14 10:25:00 | 513.19 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-05-15 09:30:00 | 517.00 | 2025-05-15 09:35:00 | 515.78 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-05-16 10:35:00 | 500.95 | 2025-05-16 10:45:00 | 502.35 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-05-29 11:15:00 | 479.70 | 2025-05-29 11:20:00 | 480.36 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-06-03 11:15:00 | 477.85 | 2025-06-03 11:20:00 | 478.62 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-06-04 09:35:00 | 470.50 | 2025-06-04 09:45:00 | 471.68 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-06-05 10:20:00 | 469.85 | 2025-06-05 10:30:00 | 468.82 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-06-06 09:45:00 | 477.20 | 2025-06-06 10:00:00 | 475.80 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-06-10 09:50:00 | 482.80 | 2025-06-10 10:10:00 | 481.35 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-06-13 09:30:00 | 466.75 | 2025-06-13 09:35:00 | 468.32 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-06-16 09:30:00 | 461.80 | 2025-06-16 09:35:00 | 463.25 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-07-01 10:55:00 | 454.95 | 2025-07-01 11:05:00 | 455.72 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-07-04 10:25:00 | 455.00 | 2025-07-04 11:25:00 | 456.36 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-07-04 10:25:00 | 455.00 | 2025-07-04 11:35:00 | 455.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-07 10:15:00 | 458.50 | 2025-07-07 11:35:00 | 460.06 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-07-07 10:15:00 | 458.50 | 2025-07-07 15:20:00 | 463.85 | TARGET_HIT | 0.50 | 1.17% |
| SELL | retest1 | 2025-07-08 10:50:00 | 459.70 | 2025-07-08 11:00:00 | 458.10 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-07-08 10:50:00 | 459.70 | 2025-07-08 11:05:00 | 459.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-11 10:45:00 | 453.60 | 2025-07-11 10:50:00 | 454.87 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-07-14 09:40:00 | 455.70 | 2025-07-14 10:40:00 | 457.38 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-07-14 09:40:00 | 455.70 | 2025-07-14 11:10:00 | 455.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-15 10:35:00 | 466.90 | 2025-07-15 10:40:00 | 465.87 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-16 09:40:00 | 465.20 | 2025-07-16 09:55:00 | 466.19 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-07-17 10:40:00 | 489.25 | 2025-07-17 11:10:00 | 487.80 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-07-21 11:05:00 | 488.00 | 2025-07-21 11:15:00 | 487.00 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-07-24 09:40:00 | 484.30 | 2025-07-24 09:45:00 | 485.27 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-07-28 09:50:00 | 485.45 | 2025-07-28 10:15:00 | 483.93 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-07-29 09:40:00 | 490.00 | 2025-07-29 09:50:00 | 488.34 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-08-01 10:30:00 | 514.65 | 2025-08-01 11:05:00 | 516.14 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-08-04 10:00:00 | 505.05 | 2025-08-04 10:40:00 | 506.80 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-08-06 11:00:00 | 500.90 | 2025-08-06 11:15:00 | 499.30 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-08-06 11:00:00 | 500.90 | 2025-08-06 11:20:00 | 500.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-11 09:55:00 | 499.50 | 2025-08-11 10:05:00 | 497.41 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-08-13 09:40:00 | 517.25 | 2025-08-13 09:55:00 | 519.26 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-08-13 09:40:00 | 517.25 | 2025-08-13 10:05:00 | 517.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-14 10:40:00 | 501.00 | 2025-08-14 12:00:00 | 502.37 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-08-20 09:50:00 | 512.00 | 2025-08-20 10:00:00 | 514.09 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-08-20 09:50:00 | 512.00 | 2025-08-20 10:05:00 | 512.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-26 09:40:00 | 508.95 | 2025-08-26 09:45:00 | 510.18 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-09-01 10:30:00 | 482.90 | 2025-09-01 12:25:00 | 484.35 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-09-12 11:00:00 | 470.40 | 2025-09-12 12:55:00 | 469.15 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-09-12 11:00:00 | 470.40 | 2025-09-12 13:20:00 | 470.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-16 11:15:00 | 467.20 | 2025-09-16 13:00:00 | 466.40 | PARTIAL | 0.50 | 0.17% |
| SELL | retest1 | 2025-09-16 11:15:00 | 467.20 | 2025-09-16 15:20:00 | 463.60 | TARGET_HIT | 0.50 | 0.77% |
| BUY | retest1 | 2025-09-17 10:05:00 | 468.45 | 2025-09-17 10:10:00 | 467.09 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-09-18 11:10:00 | 468.35 | 2025-09-18 11:25:00 | 469.19 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-09-23 11:10:00 | 461.35 | 2025-09-23 12:45:00 | 459.93 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-09-23 11:10:00 | 461.35 | 2025-09-23 15:20:00 | 458.55 | TARGET_HIT | 0.50 | 0.61% |
| SELL | retest1 | 2025-09-24 09:40:00 | 454.70 | 2025-09-24 09:50:00 | 455.88 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-09-30 11:15:00 | 446.20 | 2025-09-30 12:45:00 | 444.51 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-09-30 11:15:00 | 446.20 | 2025-09-30 15:20:00 | 443.30 | TARGET_HIT | 0.50 | 0.65% |
| SELL | retest1 | 2025-10-08 09:35:00 | 437.50 | 2025-10-08 09:50:00 | 435.99 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-10-08 09:35:00 | 437.50 | 2025-10-08 15:05:00 | 435.00 | TARGET_HIT | 0.50 | 0.57% |
| BUY | retest1 | 2025-10-10 11:00:00 | 449.50 | 2025-10-10 11:30:00 | 448.08 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-10-14 10:55:00 | 439.20 | 2025-10-14 11:20:00 | 440.00 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-10-16 09:30:00 | 451.75 | 2025-10-16 09:35:00 | 450.61 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-10-17 09:35:00 | 463.85 | 2025-10-17 09:40:00 | 462.46 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-10-28 09:35:00 | 455.00 | 2025-10-28 09:45:00 | 456.32 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-10-29 09:55:00 | 463.00 | 2025-10-29 10:00:00 | 465.36 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-10-29 09:55:00 | 463.00 | 2025-10-29 11:55:00 | 463.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-04 11:00:00 | 470.45 | 2025-11-04 11:20:00 | 469.18 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-11-04 11:00:00 | 470.45 | 2025-11-04 11:25:00 | 470.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-12 09:30:00 | 464.45 | 2025-11-12 10:15:00 | 462.26 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-11-12 09:30:00 | 464.45 | 2025-11-12 15:20:00 | 459.20 | TARGET_HIT | 0.50 | 1.13% |
| SELL | retest1 | 2025-11-13 09:40:00 | 456.20 | 2025-11-13 11:05:00 | 457.21 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-11-20 09:45:00 | 452.60 | 2025-11-20 10:00:00 | 453.38 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-11-24 11:15:00 | 447.95 | 2025-11-24 11:30:00 | 447.07 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-11-26 09:55:00 | 453.50 | 2025-11-26 10:15:00 | 455.04 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-11-26 09:55:00 | 453.50 | 2025-11-26 15:20:00 | 465.40 | TARGET_HIT | 0.50 | 2.62% |
| BUY | retest1 | 2025-12-01 11:05:00 | 485.80 | 2025-12-01 11:10:00 | 484.73 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-12-03 09:30:00 | 475.30 | 2025-12-03 10:00:00 | 473.60 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-12-03 09:30:00 | 475.30 | 2025-12-03 10:20:00 | 475.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-16 09:40:00 | 481.45 | 2025-12-16 09:45:00 | 483.89 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-12-16 09:40:00 | 481.45 | 2025-12-16 10:00:00 | 481.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-19 11:00:00 | 469.40 | 2025-12-19 11:40:00 | 467.56 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-12-19 11:00:00 | 469.40 | 2025-12-19 12:00:00 | 469.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-24 11:00:00 | 475.00 | 2025-12-24 11:15:00 | 476.16 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-12-26 09:45:00 | 479.15 | 2025-12-26 10:00:00 | 477.91 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-12-29 09:30:00 | 487.75 | 2025-12-29 09:35:00 | 489.71 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-12-29 09:30:00 | 487.75 | 2025-12-29 10:10:00 | 489.65 | TARGET_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2025-12-30 10:05:00 | 485.00 | 2025-12-30 13:55:00 | 483.56 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-12-31 11:05:00 | 484.85 | 2025-12-31 11:10:00 | 485.85 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-01-08 11:10:00 | 502.05 | 2026-01-08 11:55:00 | 500.06 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-01-08 11:10:00 | 502.05 | 2026-01-08 12:10:00 | 502.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-12 09:30:00 | 494.95 | 2026-01-12 09:40:00 | 498.01 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-01-12 09:30:00 | 494.95 | 2026-01-12 09:45:00 | 494.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-13 10:30:00 | 496.80 | 2026-01-13 10:35:00 | 495.19 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-01-14 09:35:00 | 506.65 | 2026-01-14 09:45:00 | 504.97 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-01-19 10:15:00 | 498.35 | 2026-01-19 10:25:00 | 496.37 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-01-22 09:35:00 | 485.60 | 2026-01-22 10:00:00 | 483.97 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-01-28 09:45:00 | 469.30 | 2026-01-28 10:05:00 | 466.71 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-01-28 09:45:00 | 469.30 | 2026-01-28 14:05:00 | 466.40 | TARGET_HIT | 0.50 | 0.62% |
| BUY | retest1 | 2026-02-01 11:15:00 | 473.45 | 2026-02-01 11:40:00 | 472.01 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-05 10:55:00 | 438.85 | 2026-02-05 11:15:00 | 440.12 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-02-06 11:05:00 | 438.15 | 2026-02-06 11:15:00 | 439.84 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-02-06 11:05:00 | 438.15 | 2026-02-06 13:25:00 | 438.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-09 10:40:00 | 445.00 | 2026-02-09 11:05:00 | 446.58 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-02-09 10:40:00 | 445.00 | 2026-02-09 15:20:00 | 457.05 | TARGET_HIT | 0.50 | 2.71% |
| BUY | retest1 | 2026-02-10 10:20:00 | 460.35 | 2026-02-10 10:40:00 | 458.83 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-12 10:45:00 | 451.70 | 2026-02-12 11:00:00 | 452.61 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-02-13 09:30:00 | 450.05 | 2026-02-13 09:40:00 | 448.11 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-02-13 09:30:00 | 450.05 | 2026-02-13 09:45:00 | 450.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 11:00:00 | 457.50 | 2026-02-17 12:50:00 | 458.94 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2026-02-17 11:00:00 | 457.50 | 2026-02-17 13:00:00 | 457.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-19 11:15:00 | 458.60 | 2026-02-19 12:05:00 | 457.09 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-02-19 11:15:00 | 458.60 | 2026-02-19 15:20:00 | 452.20 | TARGET_HIT | 0.50 | 1.40% |
| BUY | retest1 | 2026-02-26 10:20:00 | 460.65 | 2026-02-26 10:30:00 | 459.53 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-03-05 11:05:00 | 438.35 | 2026-03-05 11:20:00 | 440.60 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-03-05 11:05:00 | 438.35 | 2026-03-05 11:40:00 | 438.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-11 09:35:00 | 439.35 | 2026-03-11 10:00:00 | 441.58 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2026-03-11 09:35:00 | 439.35 | 2026-03-11 10:05:00 | 439.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-20 09:30:00 | 409.20 | 2026-03-20 09:35:00 | 408.02 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-03-25 09:30:00 | 397.30 | 2026-03-25 09:35:00 | 395.74 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-16 11:10:00 | 455.70 | 2026-04-16 11:25:00 | 454.31 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-04-22 10:00:00 | 492.90 | 2026-04-22 10:10:00 | 490.86 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-29 10:20:00 | 527.35 | 2026-04-29 10:30:00 | 525.49 | STOP_HIT | 1.00 | -0.35% |
