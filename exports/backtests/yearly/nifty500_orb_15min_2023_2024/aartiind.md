# Aarti Industries Ltd. (AARTIIND)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2025-07-29 15:25:00 (39667 bars)
- **Last close:** 442.00
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
| ENTRY1 | 103 |
| ENTRY2 | 0 |
| PARTIAL | 38 |
| TARGET_HIT | 23 |
| STOP_HIT | 80 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 141 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 61 / 80
- **Target hits / Stop hits / Partials:** 23 / 80 / 38
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 25.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 69 | 25 | 36.2% | 8 | 44 | 17 | 0.08% | 5.6% |
| BUY @ 2nd Alert (retest1) | 69 | 25 | 36.2% | 8 | 44 | 17 | 0.08% | 5.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 72 | 36 | 50.0% | 15 | 36 | 21 | 0.28% | 19.8% |
| SELL @ 2nd Alert (retest1) | 72 | 36 | 50.0% | 15 | 36 | 21 | 0.28% | 19.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 141 | 61 | 43.3% | 23 | 80 | 38 | 0.18% | 25.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-16 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-16 10:05:00 | 511.70 | 509.54 | 0.00 | ORB-long ORB[504.00,509.80] vol=1.9x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-16 10:15:00 | 513.92 | 510.43 | 0.00 | T1 1.5R @ 513.92 |
| Stop hit — per-position SL triggered | 2023-05-16 10:35:00 | 511.70 | 510.99 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-18 10:35:00 | 500.00 | 502.59 | 0.00 | ORB-short ORB[502.60,505.55] vol=2.9x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-18 12:15:00 | 498.32 | 501.45 | 0.00 | T1 1.5R @ 498.32 |
| Target hit | 2023-05-18 15:20:00 | 495.55 | 498.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2023-05-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 09:30:00 | 493.10 | 496.03 | 0.00 | ORB-short ORB[496.15,498.75] vol=2.0x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-19 09:35:00 | 491.04 | 493.74 | 0.00 | T1 1.5R @ 491.04 |
| Stop hit — per-position SL triggered | 2023-05-19 09:45:00 | 493.10 | 493.24 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-22 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-22 10:10:00 | 501.00 | 496.48 | 0.00 | ORB-long ORB[491.00,497.30] vol=2.0x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-22 11:10:00 | 503.39 | 498.08 | 0.00 | T1 1.5R @ 503.39 |
| Stop hit — per-position SL triggered | 2023-05-22 11:45:00 | 501.00 | 498.99 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-23 09:30:00 | 511.95 | 509.39 | 0.00 | ORB-long ORB[506.50,510.75] vol=2.9x ATR=1.56 |
| Stop hit — per-position SL triggered | 2023-05-23 09:45:00 | 510.39 | 510.23 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-05-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-24 09:35:00 | 511.15 | 509.30 | 0.00 | ORB-long ORB[505.90,510.60] vol=2.4x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-24 10:55:00 | 513.17 | 511.70 | 0.00 | T1 1.5R @ 513.17 |
| Target hit | 2023-05-24 12:10:00 | 514.00 | 514.88 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — SELL (started 2023-05-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-25 10:45:00 | 507.50 | 509.32 | 0.00 | ORB-short ORB[509.00,514.05] vol=2.8x ATR=1.22 |
| Stop hit — per-position SL triggered | 2023-05-25 10:55:00 | 508.72 | 509.30 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-05-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-26 09:35:00 | 505.30 | 506.98 | 0.00 | ORB-short ORB[505.55,509.40] vol=1.5x ATR=1.43 |
| Stop hit — per-position SL triggered | 2023-05-26 09:45:00 | 506.73 | 506.84 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-05-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-29 10:10:00 | 515.90 | 513.37 | 0.00 | ORB-long ORB[510.25,515.00] vol=1.7x ATR=1.44 |
| Stop hit — per-position SL triggered | 2023-05-29 10:35:00 | 514.46 | 514.10 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-30 11:15:00 | 511.00 | 512.55 | 0.00 | ORB-short ORB[512.05,516.60] vol=1.9x ATR=1.03 |
| Stop hit — per-position SL triggered | 2023-05-30 12:00:00 | 512.03 | 512.31 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-06-02 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-02 11:05:00 | 513.85 | 516.30 | 0.00 | ORB-short ORB[515.35,520.55] vol=2.8x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-02 11:10:00 | 512.48 | 516.02 | 0.00 | T1 1.5R @ 512.48 |
| Stop hit — per-position SL triggered | 2023-06-02 11:20:00 | 513.85 | 515.93 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-06-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-06 09:45:00 | 522.70 | 519.25 | 0.00 | ORB-long ORB[514.80,519.60] vol=4.8x ATR=1.44 |
| Stop hit — per-position SL triggered | 2023-06-06 09:50:00 | 521.26 | 519.62 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-06-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-07 10:50:00 | 516.50 | 515.66 | 0.00 | ORB-long ORB[511.70,516.00] vol=1.6x ATR=0.91 |
| Stop hit — per-position SL triggered | 2023-06-07 10:55:00 | 515.59 | 515.72 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2023-06-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-09 09:35:00 | 509.45 | 511.63 | 0.00 | ORB-short ORB[510.10,515.80] vol=2.8x ATR=1.56 |
| Stop hit — per-position SL triggered | 2023-06-09 09:45:00 | 511.01 | 511.54 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-06-14 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-14 10:10:00 | 516.85 | 514.51 | 0.00 | ORB-long ORB[512.80,516.80] vol=1.6x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-14 10:40:00 | 519.22 | 515.72 | 0.00 | T1 1.5R @ 519.22 |
| Target hit | 2023-06-14 15:05:00 | 518.60 | 518.74 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — BUY (started 2023-06-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-15 09:30:00 | 525.35 | 523.41 | 0.00 | ORB-long ORB[520.05,524.35] vol=4.6x ATR=1.39 |
| Stop hit — per-position SL triggered | 2023-06-15 09:35:00 | 523.96 | 523.48 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-06-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-19 11:00:00 | 517.60 | 519.61 | 0.00 | ORB-short ORB[518.85,522.05] vol=3.2x ATR=1.18 |
| Stop hit — per-position SL triggered | 2023-06-19 11:20:00 | 518.78 | 519.44 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-06-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-20 09:30:00 | 515.50 | 518.95 | 0.00 | ORB-short ORB[517.80,522.00] vol=4.3x ATR=1.45 |
| Stop hit — per-position SL triggered | 2023-06-20 09:35:00 | 516.95 | 518.71 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-06-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-21 10:00:00 | 522.80 | 521.14 | 0.00 | ORB-long ORB[518.00,522.45] vol=1.6x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-21 10:25:00 | 524.55 | 521.91 | 0.00 | T1 1.5R @ 524.55 |
| Target hit | 2023-06-21 15:20:00 | 532.20 | 530.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — SELL (started 2023-06-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-22 09:40:00 | 531.70 | 536.72 | 0.00 | ORB-short ORB[533.50,539.25] vol=1.5x ATR=2.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-22 09:45:00 | 528.42 | 536.02 | 0.00 | T1 1.5R @ 528.42 |
| Stop hit — per-position SL triggered | 2023-06-22 09:50:00 | 531.70 | 535.89 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-06-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-27 09:35:00 | 512.75 | 510.71 | 0.00 | ORB-long ORB[508.20,512.00] vol=1.7x ATR=1.39 |
| Stop hit — per-position SL triggered | 2023-06-27 09:55:00 | 511.36 | 511.16 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-06-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-28 11:10:00 | 508.70 | 509.13 | 0.00 | ORB-short ORB[508.80,510.70] vol=1.6x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-28 12:25:00 | 506.89 | 508.88 | 0.00 | T1 1.5R @ 506.89 |
| Target hit | 2023-06-28 15:20:00 | 506.25 | 507.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2023-06-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-30 10:25:00 | 505.80 | 507.31 | 0.00 | ORB-short ORB[506.00,509.35] vol=2.0x ATR=1.33 |
| Stop hit — per-position SL triggered | 2023-06-30 10:50:00 | 507.13 | 507.20 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-07-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-04 09:40:00 | 497.85 | 501.51 | 0.00 | ORB-short ORB[501.80,506.50] vol=4.8x ATR=1.47 |
| Stop hit — per-position SL triggered | 2023-07-04 09:50:00 | 499.32 | 500.68 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-07-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-06 09:45:00 | 485.35 | 487.48 | 0.00 | ORB-short ORB[486.10,489.20] vol=2.3x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-06 09:50:00 | 483.37 | 486.48 | 0.00 | T1 1.5R @ 483.37 |
| Target hit | 2023-07-06 11:30:00 | 484.35 | 483.66 | 0.00 | Trail-exit close>VWAP |

### Cycle 26 — SELL (started 2023-07-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-07 11:05:00 | 481.50 | 485.18 | 0.00 | ORB-short ORB[483.65,488.00] vol=1.7x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-07 11:30:00 | 479.53 | 483.19 | 0.00 | T1 1.5R @ 479.53 |
| Target hit | 2023-07-07 15:20:00 | 472.15 | 476.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — SELL (started 2023-07-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-13 10:25:00 | 449.80 | 453.05 | 0.00 | ORB-short ORB[452.45,456.65] vol=1.7x ATR=1.10 |
| Stop hit — per-position SL triggered | 2023-07-13 10:45:00 | 450.90 | 452.62 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-07-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 09:30:00 | 451.45 | 448.77 | 0.00 | ORB-long ORB[445.90,451.05] vol=1.8x ATR=1.45 |
| Stop hit — per-position SL triggered | 2023-07-14 10:05:00 | 450.00 | 449.76 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2023-07-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-17 09:30:00 | 459.50 | 457.60 | 0.00 | ORB-long ORB[455.30,458.80] vol=2.5x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-17 09:40:00 | 461.40 | 458.77 | 0.00 | T1 1.5R @ 461.40 |
| Stop hit — per-position SL triggered | 2023-07-17 10:20:00 | 459.50 | 460.04 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-07-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-20 10:30:00 | 471.90 | 469.25 | 0.00 | ORB-long ORB[466.30,469.75] vol=4.3x ATR=1.03 |
| Stop hit — per-position SL triggered | 2023-07-20 10:40:00 | 470.87 | 469.49 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2023-07-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-31 09:35:00 | 462.05 | 460.55 | 0.00 | ORB-long ORB[459.00,461.25] vol=1.6x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-31 10:10:00 | 463.50 | 461.69 | 0.00 | T1 1.5R @ 463.50 |
| Target hit | 2023-07-31 15:20:00 | 468.00 | 465.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — BUY (started 2023-08-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-01 09:30:00 | 472.15 | 471.15 | 0.00 | ORB-long ORB[468.15,471.95] vol=1.5x ATR=1.14 |
| Stop hit — per-position SL triggered | 2023-08-01 09:35:00 | 471.01 | 471.16 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-08-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-03 11:15:00 | 473.60 | 471.28 | 0.00 | ORB-long ORB[468.35,473.20] vol=2.5x ATR=1.20 |
| Stop hit — per-position SL triggered | 2023-08-03 12:15:00 | 472.40 | 471.68 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2023-08-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-18 09:55:00 | 454.60 | 456.66 | 0.00 | ORB-short ORB[455.10,459.45] vol=1.7x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-18 10:00:00 | 453.07 | 456.06 | 0.00 | T1 1.5R @ 453.07 |
| Target hit | 2023-08-18 15:20:00 | 449.30 | 451.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — SELL (started 2023-08-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-21 09:45:00 | 447.55 | 450.17 | 0.00 | ORB-short ORB[449.20,452.55] vol=1.8x ATR=1.00 |
| Stop hit — per-position SL triggered | 2023-08-21 09:55:00 | 448.55 | 449.84 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-23 10:15:00 | 455.90 | 453.98 | 0.00 | ORB-long ORB[451.30,454.25] vol=1.9x ATR=0.85 |
| Stop hit — per-position SL triggered | 2023-08-23 10:25:00 | 455.05 | 454.18 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-08-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 11:00:00 | 456.25 | 458.72 | 0.00 | ORB-short ORB[456.30,461.85] vol=1.6x ATR=1.19 |
| Stop hit — per-position SL triggered | 2023-08-25 11:20:00 | 457.44 | 458.60 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2023-08-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-28 11:00:00 | 462.55 | 460.17 | 0.00 | ORB-long ORB[458.00,460.85] vol=2.0x ATR=0.92 |
| Stop hit — per-position SL triggered | 2023-08-28 11:20:00 | 461.63 | 460.46 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-08-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-29 10:05:00 | 468.60 | 465.35 | 0.00 | ORB-long ORB[460.95,466.20] vol=4.1x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-29 11:20:00 | 470.66 | 467.59 | 0.00 | T1 1.5R @ 470.66 |
| Target hit | 2023-08-29 15:20:00 | 483.20 | 474.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — BUY (started 2023-09-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 10:50:00 | 499.30 | 495.29 | 0.00 | ORB-long ORB[491.10,496.15] vol=2.0x ATR=1.36 |
| Stop hit — per-position SL triggered | 2023-09-04 10:55:00 | 497.94 | 495.39 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-09-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 09:30:00 | 507.55 | 505.86 | 0.00 | ORB-long ORB[502.45,507.30] vol=2.7x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-05 09:50:00 | 509.92 | 507.62 | 0.00 | T1 1.5R @ 509.92 |
| Target hit | 2023-09-05 10:20:00 | 509.70 | 509.97 | 0.00 | Trail-exit close<VWAP |

### Cycle 42 — BUY (started 2023-09-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 10:20:00 | 511.50 | 509.34 | 0.00 | ORB-long ORB[506.25,511.20] vol=1.6x ATR=1.80 |
| Stop hit — per-position SL triggered | 2023-09-06 10:40:00 | 509.70 | 509.63 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2023-09-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-07 11:05:00 | 519.80 | 517.25 | 0.00 | ORB-long ORB[514.10,518.00] vol=3.5x ATR=1.53 |
| Stop hit — per-position SL triggered | 2023-09-07 11:15:00 | 518.27 | 517.41 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2023-09-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 09:30:00 | 515.00 | 517.11 | 0.00 | ORB-short ORB[516.15,519.45] vol=2.3x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 09:35:00 | 513.17 | 515.41 | 0.00 | T1 1.5R @ 513.17 |
| Target hit | 2023-09-12 10:05:00 | 511.05 | 509.08 | 0.00 | Trail-exit close>VWAP |

### Cycle 45 — SELL (started 2023-09-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-14 11:05:00 | 525.60 | 530.48 | 0.00 | ORB-short ORB[528.45,534.90] vol=1.7x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-14 11:35:00 | 522.85 | 529.85 | 0.00 | T1 1.5R @ 522.85 |
| Target hit | 2023-09-14 15:20:00 | 522.50 | 525.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — SELL (started 2023-09-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-20 11:00:00 | 504.90 | 509.58 | 0.00 | ORB-short ORB[506.05,513.20] vol=1.8x ATR=1.55 |
| Stop hit — per-position SL triggered | 2023-09-20 11:30:00 | 506.45 | 509.20 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2023-09-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-21 09:35:00 | 510.25 | 507.44 | 0.00 | ORB-long ORB[504.00,508.60] vol=2.7x ATR=1.45 |
| Stop hit — per-position SL triggered | 2023-09-21 09:50:00 | 508.80 | 507.97 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-09-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-22 09:30:00 | 500.80 | 503.20 | 0.00 | ORB-short ORB[502.10,506.20] vol=2.8x ATR=1.47 |
| Stop hit — per-position SL triggered | 2023-09-22 09:35:00 | 502.27 | 503.09 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-09-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-26 10:10:00 | 500.85 | 498.81 | 0.00 | ORB-long ORB[496.50,499.45] vol=3.1x ATR=1.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-26 10:20:00 | 502.79 | 499.94 | 0.00 | T1 1.5R @ 502.79 |
| Stop hit — per-position SL triggered | 2023-09-26 10:25:00 | 500.85 | 499.93 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2023-09-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-27 09:35:00 | 497.20 | 495.45 | 0.00 | ORB-long ORB[493.30,496.30] vol=1.9x ATR=1.36 |
| Stop hit — per-position SL triggered | 2023-09-27 10:05:00 | 495.84 | 496.15 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2023-09-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-28 09:50:00 | 494.95 | 496.35 | 0.00 | ORB-short ORB[495.60,499.40] vol=1.7x ATR=0.96 |
| Stop hit — per-position SL triggered | 2023-09-28 10:00:00 | 495.91 | 496.26 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2023-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-06 11:15:00 | 484.45 | 486.45 | 0.00 | ORB-short ORB[486.65,490.50] vol=2.4x ATR=0.88 |
| Stop hit — per-position SL triggered | 2023-10-06 11:30:00 | 485.33 | 486.35 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2023-10-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-09 10:50:00 | 476.00 | 478.91 | 0.00 | ORB-short ORB[476.05,481.50] vol=1.8x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-09 12:20:00 | 473.91 | 477.68 | 0.00 | T1 1.5R @ 473.91 |
| Target hit | 2023-10-09 15:20:00 | 471.30 | 475.62 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — SELL (started 2023-10-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-12 10:35:00 | 483.40 | 483.99 | 0.00 | ORB-short ORB[484.00,487.10] vol=2.1x ATR=1.11 |
| Stop hit — per-position SL triggered | 2023-10-12 11:30:00 | 484.51 | 483.98 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2023-10-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-16 10:25:00 | 490.70 | 488.74 | 0.00 | ORB-long ORB[484.65,487.50] vol=2.3x ATR=1.25 |
| Stop hit — per-position SL triggered | 2023-10-16 10:45:00 | 489.45 | 489.12 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2023-10-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-18 10:00:00 | 491.35 | 489.18 | 0.00 | ORB-long ORB[487.15,490.05] vol=3.3x ATR=1.28 |
| Stop hit — per-position SL triggered | 2023-10-18 10:25:00 | 490.07 | 489.54 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2023-10-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-23 10:10:00 | 468.40 | 472.20 | 0.00 | ORB-short ORB[470.00,474.55] vol=1.7x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-23 10:25:00 | 466.07 | 471.33 | 0.00 | T1 1.5R @ 466.07 |
| Stop hit — per-position SL triggered | 2023-10-23 10:30:00 | 468.40 | 471.14 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2023-10-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-30 09:35:00 | 455.15 | 453.02 | 0.00 | ORB-long ORB[450.40,455.00] vol=2.9x ATR=1.51 |
| Stop hit — per-position SL triggered | 2023-10-30 09:40:00 | 453.64 | 453.25 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2023-11-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 09:40:00 | 522.00 | 520.34 | 0.00 | ORB-long ORB[518.20,520.20] vol=3.3x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-21 09:45:00 | 523.94 | 521.57 | 0.00 | T1 1.5R @ 523.94 |
| Stop hit — per-position SL triggered | 2023-11-21 10:00:00 | 522.00 | 522.07 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2023-11-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-22 09:35:00 | 523.05 | 521.20 | 0.00 | ORB-long ORB[519.35,522.00] vol=2.0x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-22 09:45:00 | 524.54 | 522.20 | 0.00 | T1 1.5R @ 524.54 |
| Stop hit — per-position SL triggered | 2023-11-22 09:50:00 | 523.05 | 522.24 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2023-11-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-23 09:35:00 | 526.80 | 524.55 | 0.00 | ORB-long ORB[520.55,525.85] vol=2.8x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-23 09:45:00 | 528.61 | 525.95 | 0.00 | T1 1.5R @ 528.61 |
| Stop hit — per-position SL triggered | 2023-11-23 09:55:00 | 526.80 | 526.09 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2023-11-28 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-28 09:55:00 | 528.75 | 532.12 | 0.00 | ORB-short ORB[531.50,535.25] vol=1.6x ATR=1.45 |
| Stop hit — per-position SL triggered | 2023-11-28 10:10:00 | 530.20 | 531.78 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2023-11-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 09:55:00 | 533.90 | 531.87 | 0.00 | ORB-long ORB[529.20,532.40] vol=1.7x ATR=1.23 |
| Stop hit — per-position SL triggered | 2023-11-29 10:15:00 | 532.67 | 532.19 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2023-12-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-01 10:40:00 | 549.20 | 554.69 | 0.00 | ORB-short ORB[557.00,561.00] vol=3.6x ATR=1.99 |
| Stop hit — per-position SL triggered | 2023-12-01 11:15:00 | 551.19 | 553.71 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2023-12-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 10:35:00 | 568.95 | 561.15 | 0.00 | ORB-long ORB[558.00,564.00] vol=2.8x ATR=2.24 |
| Stop hit — per-position SL triggered | 2023-12-04 10:40:00 | 566.71 | 561.75 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2023-12-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-06 10:00:00 | 583.35 | 577.88 | 0.00 | ORB-long ORB[571.50,580.00] vol=4.8x ATR=3.02 |
| Stop hit — per-position SL triggered | 2023-12-06 10:10:00 | 580.33 | 578.83 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2023-12-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 10:55:00 | 573.10 | 575.16 | 0.00 | ORB-short ORB[573.30,579.40] vol=1.9x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-08 11:00:00 | 570.83 | 574.56 | 0.00 | T1 1.5R @ 570.83 |
| Target hit | 2023-12-08 14:40:00 | 570.30 | 569.18 | 0.00 | Trail-exit close>VWAP |

### Cycle 68 — SELL (started 2023-12-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-13 09:30:00 | 564.60 | 567.55 | 0.00 | ORB-short ORB[565.25,571.25] vol=1.5x ATR=2.22 |
| Stop hit — per-position SL triggered | 2023-12-13 09:40:00 | 566.82 | 567.53 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2023-12-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-14 09:35:00 | 570.70 | 572.72 | 0.00 | ORB-short ORB[570.80,577.00] vol=3.8x ATR=2.05 |
| Stop hit — per-position SL triggered | 2023-12-14 09:40:00 | 572.75 | 572.77 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2023-12-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-15 09:35:00 | 585.00 | 582.42 | 0.00 | ORB-long ORB[576.20,583.95] vol=4.1x ATR=1.89 |
| Stop hit — per-position SL triggered | 2023-12-15 09:40:00 | 583.11 | 582.60 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2023-12-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-18 09:35:00 | 598.55 | 594.11 | 0.00 | ORB-long ORB[590.65,596.40] vol=2.2x ATR=2.27 |
| Stop hit — per-position SL triggered | 2023-12-18 09:45:00 | 596.28 | 594.86 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2023-12-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-19 09:40:00 | 610.00 | 607.33 | 0.00 | ORB-long ORB[603.05,608.55] vol=7.0x ATR=2.65 |
| Stop hit — per-position SL triggered | 2023-12-19 09:45:00 | 607.35 | 607.42 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2023-12-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-20 09:40:00 | 609.00 | 612.23 | 0.00 | ORB-short ORB[610.05,615.90] vol=2.8x ATR=2.26 |
| Stop hit — per-position SL triggered | 2023-12-20 10:00:00 | 611.26 | 611.73 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2023-12-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 10:50:00 | 600.75 | 597.41 | 0.00 | ORB-long ORB[594.70,600.15] vol=2.3x ATR=2.25 |
| Stop hit — per-position SL triggered | 2023-12-22 11:20:00 | 598.50 | 597.92 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2023-12-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 10:35:00 | 646.20 | 641.89 | 0.00 | ORB-long ORB[640.00,645.00] vol=1.7x ATR=2.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-28 11:05:00 | 649.57 | 643.05 | 0.00 | T1 1.5R @ 649.57 |
| Stop hit — per-position SL triggered | 2023-12-28 11:55:00 | 646.20 | 644.14 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2024-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-02 09:30:00 | 658.10 | 656.31 | 0.00 | ORB-long ORB[652.10,657.95] vol=1.9x ATR=1.91 |
| Stop hit — per-position SL triggered | 2024-01-02 09:45:00 | 656.19 | 656.55 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2024-01-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-03 09:40:00 | 646.00 | 649.41 | 0.00 | ORB-short ORB[649.05,656.80] vol=1.6x ATR=2.40 |
| Stop hit — per-position SL triggered | 2024-01-03 09:50:00 | 648.40 | 649.09 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2024-01-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-04 10:00:00 | 649.80 | 650.41 | 0.00 | ORB-short ORB[649.85,656.20] vol=10.7x ATR=2.05 |
| Stop hit — per-position SL triggered | 2024-01-04 10:10:00 | 651.85 | 650.46 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2024-01-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-05 09:30:00 | 644.35 | 645.79 | 0.00 | ORB-short ORB[644.45,649.05] vol=1.8x ATR=1.50 |
| Stop hit — per-position SL triggered | 2024-01-05 09:35:00 | 645.85 | 645.71 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2024-01-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-09 09:45:00 | 601.75 | 604.34 | 0.00 | ORB-short ORB[602.35,608.45] vol=1.6x ATR=2.49 |
| Stop hit — per-position SL triggered | 2024-01-09 11:05:00 | 604.24 | 603.72 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2024-01-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 09:40:00 | 612.25 | 608.78 | 0.00 | ORB-long ORB[606.00,609.30] vol=3.5x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-11 10:10:00 | 614.89 | 611.05 | 0.00 | T1 1.5R @ 614.89 |
| Stop hit — per-position SL triggered | 2024-01-11 10:30:00 | 612.25 | 612.48 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2024-01-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 10:00:00 | 641.25 | 637.00 | 0.00 | ORB-long ORB[631.00,638.00] vol=1.5x ATR=2.13 |
| Stop hit — per-position SL triggered | 2024-01-31 10:40:00 | 639.12 | 638.18 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2024-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-01 11:15:00 | 644.30 | 650.57 | 0.00 | ORB-short ORB[650.25,659.60] vol=2.9x ATR=1.87 |
| Stop hit — per-position SL triggered | 2024-02-01 11:35:00 | 646.17 | 650.03 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2024-02-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-07 09:35:00 | 659.55 | 663.17 | 0.00 | ORB-short ORB[661.20,667.75] vol=1.6x ATR=2.21 |
| Stop hit — per-position SL triggered | 2024-02-07 09:50:00 | 661.76 | 662.76 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2024-02-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 09:55:00 | 661.95 | 666.73 | 0.00 | ORB-short ORB[667.10,672.25] vol=2.9x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 10:05:00 | 658.15 | 665.54 | 0.00 | T1 1.5R @ 658.15 |
| Target hit | 2024-02-08 15:20:00 | 634.70 | 649.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 86 — SELL (started 2024-02-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-23 10:10:00 | 679.05 | 682.39 | 0.00 | ORB-short ORB[681.70,689.50] vol=1.6x ATR=2.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-23 10:25:00 | 675.53 | 681.79 | 0.00 | T1 1.5R @ 675.53 |
| Stop hit — per-position SL triggered | 2024-02-23 11:00:00 | 679.05 | 680.65 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2024-03-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-04 09:30:00 | 669.20 | 672.68 | 0.00 | ORB-short ORB[670.00,678.00] vol=2.7x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-04 12:10:00 | 665.94 | 669.70 | 0.00 | T1 1.5R @ 665.94 |
| Target hit | 2024-03-04 15:20:00 | 661.95 | 666.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 88 — SELL (started 2024-03-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 09:35:00 | 653.25 | 656.90 | 0.00 | ORB-short ORB[654.75,662.00] vol=2.1x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 09:50:00 | 650.07 | 654.48 | 0.00 | T1 1.5R @ 650.07 |
| Target hit | 2024-03-06 13:45:00 | 649.30 | 646.73 | 0.00 | Trail-exit close>VWAP |

### Cycle 89 — SELL (started 2024-03-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 09:55:00 | 635.80 | 640.66 | 0.00 | ORB-short ORB[640.10,646.35] vol=2.0x ATR=2.10 |
| Stop hit — per-position SL triggered | 2024-03-19 10:30:00 | 637.90 | 638.81 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2024-03-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 10:55:00 | 649.50 | 644.45 | 0.00 | ORB-long ORB[640.00,646.80] vol=2.8x ATR=1.99 |
| Stop hit — per-position SL triggered | 2024-03-21 11:05:00 | 647.51 | 645.17 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2024-04-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-02 09:35:00 | 692.45 | 689.13 | 0.00 | ORB-long ORB[682.05,691.95] vol=1.6x ATR=2.21 |
| Stop hit — per-position SL triggered | 2024-04-02 11:05:00 | 690.24 | 691.35 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2024-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 11:15:00 | 687.40 | 682.87 | 0.00 | ORB-long ORB[678.30,686.10] vol=1.8x ATR=1.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-03 11:20:00 | 689.83 | 683.62 | 0.00 | T1 1.5R @ 689.83 |
| Target hit | 2024-04-03 14:05:00 | 692.60 | 692.72 | 0.00 | Trail-exit close<VWAP |

### Cycle 93 — SELL (started 2024-04-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-05 09:40:00 | 684.75 | 687.33 | 0.00 | ORB-short ORB[686.30,696.10] vol=2.0x ATR=2.26 |
| Stop hit — per-position SL triggered | 2024-04-05 09:45:00 | 687.01 | 687.29 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2024-04-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-08 09:40:00 | 687.85 | 693.51 | 0.00 | ORB-short ORB[693.10,699.90] vol=1.5x ATR=2.00 |
| Stop hit — per-position SL triggered | 2024-04-08 09:50:00 | 689.85 | 692.32 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2024-04-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-09 09:30:00 | 715.55 | 709.92 | 0.00 | ORB-long ORB[704.45,711.70] vol=3.7x ATR=2.43 |
| Stop hit — per-position SL triggered | 2024-04-09 09:35:00 | 713.12 | 711.77 | 0.00 | SL hit |

### Cycle 96 — BUY (started 2024-04-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 09:50:00 | 741.00 | 733.94 | 0.00 | ORB-long ORB[725.10,735.00] vol=1.5x ATR=2.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-16 10:30:00 | 744.96 | 737.06 | 0.00 | T1 1.5R @ 744.96 |
| Target hit | 2024-04-16 15:20:00 | 748.20 | 748.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 97 — SELL (started 2024-04-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-22 11:00:00 | 724.65 | 726.36 | 0.00 | ORB-short ORB[725.10,735.15] vol=1.8x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-22 15:00:00 | 721.47 | 725.29 | 0.00 | T1 1.5R @ 721.47 |
| Target hit | 2024-04-22 15:20:00 | 721.00 | 724.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 98 — BUY (started 2024-04-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 10:00:00 | 737.70 | 730.60 | 0.00 | ORB-long ORB[722.05,731.00] vol=2.9x ATR=2.48 |
| Stop hit — per-position SL triggered | 2024-04-23 10:40:00 | 735.22 | 733.65 | 0.00 | SL hit |

### Cycle 99 — BUY (started 2024-04-24 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 10:40:00 | 754.75 | 749.47 | 0.00 | ORB-long ORB[741.00,748.25] vol=1.9x ATR=2.44 |
| Stop hit — per-position SL triggered | 2024-04-24 10:45:00 | 752.31 | 749.68 | 0.00 | SL hit |

### Cycle 100 — SELL (started 2024-04-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-30 10:55:00 | 747.35 | 751.65 | 0.00 | ORB-short ORB[748.80,759.60] vol=3.0x ATR=1.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-30 11:05:00 | 744.40 | 750.86 | 0.00 | T1 1.5R @ 744.40 |
| Stop hit — per-position SL triggered | 2024-04-30 11:20:00 | 747.35 | 749.81 | 0.00 | SL hit |

### Cycle 101 — BUY (started 2024-05-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-03 09:40:00 | 754.50 | 751.31 | 0.00 | ORB-long ORB[745.00,753.35] vol=1.7x ATR=2.76 |
| Stop hit — per-position SL triggered | 2024-05-03 09:50:00 | 751.74 | 751.50 | 0.00 | SL hit |

### Cycle 102 — SELL (started 2024-05-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 10:30:00 | 729.55 | 738.77 | 0.00 | ORB-short ORB[739.50,747.60] vol=1.5x ATR=2.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 10:45:00 | 725.51 | 735.81 | 0.00 | T1 1.5R @ 725.51 |
| Target hit | 2024-05-07 14:10:00 | 725.15 | 724.39 | 0.00 | Trail-exit close>VWAP |

### Cycle 103 — SELL (started 2024-05-09 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 10:20:00 | 698.15 | 702.30 | 0.00 | ORB-short ORB[700.75,709.95] vol=2.5x ATR=2.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 10:35:00 | 694.73 | 700.96 | 0.00 | T1 1.5R @ 694.73 |
| Target hit | 2024-05-09 15:20:00 | 664.70 | 679.90 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-16 10:05:00 | 511.70 | 2023-05-16 10:15:00 | 513.92 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2023-05-16 10:05:00 | 511.70 | 2023-05-16 10:35:00 | 511.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-05-18 10:35:00 | 500.00 | 2023-05-18 12:15:00 | 498.32 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-05-18 10:35:00 | 500.00 | 2023-05-18 15:20:00 | 495.55 | TARGET_HIT | 0.50 | 0.89% |
| SELL | retest1 | 2023-05-19 09:30:00 | 493.10 | 2023-05-19 09:35:00 | 491.04 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2023-05-19 09:30:00 | 493.10 | 2023-05-19 09:45:00 | 493.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-22 10:10:00 | 501.00 | 2023-05-22 11:10:00 | 503.39 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2023-05-22 10:10:00 | 501.00 | 2023-05-22 11:45:00 | 501.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-23 09:30:00 | 511.95 | 2023-05-23 09:45:00 | 510.39 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-05-24 09:35:00 | 511.15 | 2023-05-24 10:55:00 | 513.17 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-05-24 09:35:00 | 511.15 | 2023-05-24 12:10:00 | 514.00 | TARGET_HIT | 0.50 | 0.56% |
| SELL | retest1 | 2023-05-25 10:45:00 | 507.50 | 2023-05-25 10:55:00 | 508.72 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-05-26 09:35:00 | 505.30 | 2023-05-26 09:45:00 | 506.73 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-05-29 10:10:00 | 515.90 | 2023-05-29 10:35:00 | 514.46 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-05-30 11:15:00 | 511.00 | 2023-05-30 12:00:00 | 512.03 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-06-02 11:05:00 | 513.85 | 2023-06-02 11:10:00 | 512.48 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-06-02 11:05:00 | 513.85 | 2023-06-02 11:20:00 | 513.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-06 09:45:00 | 522.70 | 2023-06-06 09:50:00 | 521.26 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-06-07 10:50:00 | 516.50 | 2023-06-07 10:55:00 | 515.59 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-06-09 09:35:00 | 509.45 | 2023-06-09 09:45:00 | 511.01 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-06-14 10:10:00 | 516.85 | 2023-06-14 10:40:00 | 519.22 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2023-06-14 10:10:00 | 516.85 | 2023-06-14 15:05:00 | 518.60 | TARGET_HIT | 0.50 | 0.34% |
| BUY | retest1 | 2023-06-15 09:30:00 | 525.35 | 2023-06-15 09:35:00 | 523.96 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-06-19 11:00:00 | 517.60 | 2023-06-19 11:20:00 | 518.78 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-06-20 09:30:00 | 515.50 | 2023-06-20 09:35:00 | 516.95 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-06-21 10:00:00 | 522.80 | 2023-06-21 10:25:00 | 524.55 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-06-21 10:00:00 | 522.80 | 2023-06-21 15:20:00 | 532.20 | TARGET_HIT | 0.50 | 1.80% |
| SELL | retest1 | 2023-06-22 09:40:00 | 531.70 | 2023-06-22 09:45:00 | 528.42 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2023-06-22 09:40:00 | 531.70 | 2023-06-22 09:50:00 | 531.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-27 09:35:00 | 512.75 | 2023-06-27 09:55:00 | 511.36 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-06-28 11:10:00 | 508.70 | 2023-06-28 12:25:00 | 506.89 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2023-06-28 11:10:00 | 508.70 | 2023-06-28 15:20:00 | 506.25 | TARGET_HIT | 0.50 | 0.48% |
| SELL | retest1 | 2023-06-30 10:25:00 | 505.80 | 2023-06-30 10:50:00 | 507.13 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-07-04 09:40:00 | 497.85 | 2023-07-04 09:50:00 | 499.32 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-07-06 09:45:00 | 485.35 | 2023-07-06 09:50:00 | 483.37 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2023-07-06 09:45:00 | 485.35 | 2023-07-06 11:30:00 | 484.35 | TARGET_HIT | 0.50 | 0.21% |
| SELL | retest1 | 2023-07-07 11:05:00 | 481.50 | 2023-07-07 11:30:00 | 479.53 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2023-07-07 11:05:00 | 481.50 | 2023-07-07 15:20:00 | 472.15 | TARGET_HIT | 0.50 | 1.94% |
| SELL | retest1 | 2023-07-13 10:25:00 | 449.80 | 2023-07-13 10:45:00 | 450.90 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-07-14 09:30:00 | 451.45 | 2023-07-14 10:05:00 | 450.00 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-07-17 09:30:00 | 459.50 | 2023-07-17 09:40:00 | 461.40 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-07-17 09:30:00 | 459.50 | 2023-07-17 10:20:00 | 459.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-20 10:30:00 | 471.90 | 2023-07-20 10:40:00 | 470.87 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-07-31 09:35:00 | 462.05 | 2023-07-31 10:10:00 | 463.50 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-07-31 09:35:00 | 462.05 | 2023-07-31 15:20:00 | 468.00 | TARGET_HIT | 0.50 | 1.29% |
| BUY | retest1 | 2023-08-01 09:30:00 | 472.15 | 2023-08-01 09:35:00 | 471.01 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-08-03 11:15:00 | 473.60 | 2023-08-03 12:15:00 | 472.40 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-08-18 09:55:00 | 454.60 | 2023-08-18 10:00:00 | 453.07 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-08-18 09:55:00 | 454.60 | 2023-08-18 15:20:00 | 449.30 | TARGET_HIT | 0.50 | 1.17% |
| SELL | retest1 | 2023-08-21 09:45:00 | 447.55 | 2023-08-21 09:55:00 | 448.55 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-08-23 10:15:00 | 455.90 | 2023-08-23 10:25:00 | 455.05 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-08-25 11:00:00 | 456.25 | 2023-08-25 11:20:00 | 457.44 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-08-28 11:00:00 | 462.55 | 2023-08-28 11:20:00 | 461.63 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-08-29 10:05:00 | 468.60 | 2023-08-29 11:20:00 | 470.66 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-08-29 10:05:00 | 468.60 | 2023-08-29 15:20:00 | 483.20 | TARGET_HIT | 0.50 | 3.12% |
| BUY | retest1 | 2023-09-04 10:50:00 | 499.30 | 2023-09-04 10:55:00 | 497.94 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-09-05 09:30:00 | 507.55 | 2023-09-05 09:50:00 | 509.92 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2023-09-05 09:30:00 | 507.55 | 2023-09-05 10:20:00 | 509.70 | TARGET_HIT | 0.50 | 0.42% |
| BUY | retest1 | 2023-09-06 10:20:00 | 511.50 | 2023-09-06 10:40:00 | 509.70 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-09-07 11:05:00 | 519.80 | 2023-09-07 11:15:00 | 518.27 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-09-12 09:30:00 | 515.00 | 2023-09-12 09:35:00 | 513.17 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2023-09-12 09:30:00 | 515.00 | 2023-09-12 10:05:00 | 511.05 | TARGET_HIT | 0.50 | 0.77% |
| SELL | retest1 | 2023-09-14 11:05:00 | 525.60 | 2023-09-14 11:35:00 | 522.85 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2023-09-14 11:05:00 | 525.60 | 2023-09-14 15:20:00 | 522.50 | TARGET_HIT | 0.50 | 0.59% |
| SELL | retest1 | 2023-09-20 11:00:00 | 504.90 | 2023-09-20 11:30:00 | 506.45 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-09-21 09:35:00 | 510.25 | 2023-09-21 09:50:00 | 508.80 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-09-22 09:30:00 | 500.80 | 2023-09-22 09:35:00 | 502.27 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-09-26 10:10:00 | 500.85 | 2023-09-26 10:20:00 | 502.79 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-09-26 10:10:00 | 500.85 | 2023-09-26 10:25:00 | 500.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-27 09:35:00 | 497.20 | 2023-09-27 10:05:00 | 495.84 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-09-28 09:50:00 | 494.95 | 2023-09-28 10:00:00 | 495.91 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-10-06 11:15:00 | 484.45 | 2023-10-06 11:30:00 | 485.33 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-10-09 10:50:00 | 476.00 | 2023-10-09 12:20:00 | 473.91 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2023-10-09 10:50:00 | 476.00 | 2023-10-09 15:20:00 | 471.30 | TARGET_HIT | 0.50 | 0.99% |
| SELL | retest1 | 2023-10-12 10:35:00 | 483.40 | 2023-10-12 11:30:00 | 484.51 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-10-16 10:25:00 | 490.70 | 2023-10-16 10:45:00 | 489.45 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-10-18 10:00:00 | 491.35 | 2023-10-18 10:25:00 | 490.07 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-10-23 10:10:00 | 468.40 | 2023-10-23 10:25:00 | 466.07 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2023-10-23 10:10:00 | 468.40 | 2023-10-23 10:30:00 | 468.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-30 09:35:00 | 455.15 | 2023-10-30 09:40:00 | 453.64 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-11-21 09:40:00 | 522.00 | 2023-11-21 09:45:00 | 523.94 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2023-11-21 09:40:00 | 522.00 | 2023-11-21 10:00:00 | 522.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-22 09:35:00 | 523.05 | 2023-11-22 09:45:00 | 524.54 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2023-11-22 09:35:00 | 523.05 | 2023-11-22 09:50:00 | 523.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-23 09:35:00 | 526.80 | 2023-11-23 09:45:00 | 528.61 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2023-11-23 09:35:00 | 526.80 | 2023-11-23 09:55:00 | 526.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-28 09:55:00 | 528.75 | 2023-11-28 10:10:00 | 530.20 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-11-29 09:55:00 | 533.90 | 2023-11-29 10:15:00 | 532.67 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-12-01 10:40:00 | 549.20 | 2023-12-01 11:15:00 | 551.19 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-12-04 10:35:00 | 568.95 | 2023-12-04 10:40:00 | 566.71 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2023-12-06 10:00:00 | 583.35 | 2023-12-06 10:10:00 | 580.33 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2023-12-08 10:55:00 | 573.10 | 2023-12-08 11:00:00 | 570.83 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2023-12-08 10:55:00 | 573.10 | 2023-12-08 14:40:00 | 570.30 | TARGET_HIT | 0.50 | 0.49% |
| SELL | retest1 | 2023-12-13 09:30:00 | 564.60 | 2023-12-13 09:40:00 | 566.82 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2023-12-14 09:35:00 | 570.70 | 2023-12-14 09:40:00 | 572.75 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-12-15 09:35:00 | 585.00 | 2023-12-15 09:40:00 | 583.11 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-12-18 09:35:00 | 598.55 | 2023-12-18 09:45:00 | 596.28 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2023-12-19 09:40:00 | 610.00 | 2023-12-19 09:45:00 | 607.35 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2023-12-20 09:40:00 | 609.00 | 2023-12-20 10:00:00 | 611.26 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-12-22 10:50:00 | 600.75 | 2023-12-22 11:20:00 | 598.50 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-12-28 10:35:00 | 646.20 | 2023-12-28 11:05:00 | 649.57 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-12-28 10:35:00 | 646.20 | 2023-12-28 11:55:00 | 646.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-02 09:30:00 | 658.10 | 2024-01-02 09:45:00 | 656.19 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-01-03 09:40:00 | 646.00 | 2024-01-03 09:50:00 | 648.40 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-01-04 10:00:00 | 649.80 | 2024-01-04 10:10:00 | 651.85 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-01-05 09:30:00 | 644.35 | 2024-01-05 09:35:00 | 645.85 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-01-09 09:45:00 | 601.75 | 2024-01-09 11:05:00 | 604.24 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-01-11 09:40:00 | 612.25 | 2024-01-11 10:10:00 | 614.89 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-01-11 09:40:00 | 612.25 | 2024-01-11 10:30:00 | 612.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-31 10:00:00 | 641.25 | 2024-01-31 10:40:00 | 639.12 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-02-01 11:15:00 | 644.30 | 2024-02-01 11:35:00 | 646.17 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-02-07 09:35:00 | 659.55 | 2024-02-07 09:50:00 | 661.76 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-02-08 09:55:00 | 661.95 | 2024-02-08 10:05:00 | 658.15 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-02-08 09:55:00 | 661.95 | 2024-02-08 15:20:00 | 634.70 | TARGET_HIT | 0.50 | 4.12% |
| SELL | retest1 | 2024-02-23 10:10:00 | 679.05 | 2024-02-23 10:25:00 | 675.53 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-02-23 10:10:00 | 679.05 | 2024-02-23 11:00:00 | 679.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-04 09:30:00 | 669.20 | 2024-03-04 12:10:00 | 665.94 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-03-04 09:30:00 | 669.20 | 2024-03-04 15:20:00 | 661.95 | TARGET_HIT | 0.50 | 1.08% |
| SELL | retest1 | 2024-03-06 09:35:00 | 653.25 | 2024-03-06 09:50:00 | 650.07 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-03-06 09:35:00 | 653.25 | 2024-03-06 13:45:00 | 649.30 | TARGET_HIT | 0.50 | 0.60% |
| SELL | retest1 | 2024-03-19 09:55:00 | 635.80 | 2024-03-19 10:30:00 | 637.90 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-03-21 10:55:00 | 649.50 | 2024-03-21 11:05:00 | 647.51 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-04-02 09:35:00 | 692.45 | 2024-04-02 11:05:00 | 690.24 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-04-03 11:15:00 | 687.40 | 2024-04-03 11:20:00 | 689.83 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-04-03 11:15:00 | 687.40 | 2024-04-03 14:05:00 | 692.60 | TARGET_HIT | 0.50 | 0.76% |
| SELL | retest1 | 2024-04-05 09:40:00 | 684.75 | 2024-04-05 09:45:00 | 687.01 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-04-08 09:40:00 | 687.85 | 2024-04-08 09:50:00 | 689.85 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-04-09 09:30:00 | 715.55 | 2024-04-09 09:35:00 | 713.12 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-04-16 09:50:00 | 741.00 | 2024-04-16 10:30:00 | 744.96 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-04-16 09:50:00 | 741.00 | 2024-04-16 15:20:00 | 748.20 | TARGET_HIT | 0.50 | 0.97% |
| SELL | retest1 | 2024-04-22 11:00:00 | 724.65 | 2024-04-22 15:00:00 | 721.47 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-04-22 11:00:00 | 724.65 | 2024-04-22 15:20:00 | 721.00 | TARGET_HIT | 0.50 | 0.50% |
| BUY | retest1 | 2024-04-23 10:00:00 | 737.70 | 2024-04-23 10:40:00 | 735.22 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-04-24 10:40:00 | 754.75 | 2024-04-24 10:45:00 | 752.31 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-04-30 10:55:00 | 747.35 | 2024-04-30 11:05:00 | 744.40 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-04-30 10:55:00 | 747.35 | 2024-04-30 11:20:00 | 747.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-03 09:40:00 | 754.50 | 2024-05-03 09:50:00 | 751.74 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-05-07 10:30:00 | 729.55 | 2024-05-07 10:45:00 | 725.51 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-05-07 10:30:00 | 729.55 | 2024-05-07 14:10:00 | 725.15 | TARGET_HIT | 0.50 | 0.60% |
| SELL | retest1 | 2024-05-09 10:20:00 | 698.15 | 2024-05-09 10:35:00 | 694.73 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-05-09 10:20:00 | 698.15 | 2024-05-09 15:20:00 | 664.70 | TARGET_HIT | 0.50 | 4.79% |
