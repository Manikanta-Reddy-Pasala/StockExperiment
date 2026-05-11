# UPL Ltd. (UPL)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (35371 bars)
- **Last close:** 644.40
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
| ENTRY1 | 90 |
| ENTRY2 | 0 |
| PARTIAL | 27 |
| TARGET_HIT | 11 |
| STOP_HIT | 79 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 117 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 38 / 79
- **Target hits / Stop hits / Partials:** 11 / 79 / 27
- **Avg / median % per leg:** 0.02% / -0.23%
- **Sum % (uncompounded):** 2.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 64 | 18 | 28.1% | 4 | 46 | 14 | -0.01% | -0.9% |
| BUY @ 2nd Alert (retest1) | 64 | 18 | 28.1% | 4 | 46 | 14 | -0.01% | -0.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 53 | 20 | 37.7% | 7 | 33 | 13 | 0.06% | 3.2% |
| SELL @ 2nd Alert (retest1) | 53 | 20 | 37.7% | 7 | 33 | 13 | 0.06% | 3.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 117 | 38 | 32.5% | 11 | 79 | 27 | 0.02% | 2.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:30:00 | 492.57 | 495.23 | 0.00 | ORB-short ORB[494.01,498.61] vol=1.7x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-16 09:35:00 | 490.42 | 494.54 | 0.00 | T1 1.5R @ 490.42 |
| Stop hit — per-position SL triggered | 2024-05-16 09:55:00 | 492.57 | 493.56 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:45:00 | 490.75 | 493.26 | 0.00 | ORB-short ORB[491.76,495.88] vol=1.5x ATR=1.49 |
| Stop hit — per-position SL triggered | 2024-05-22 09:55:00 | 492.24 | 493.04 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 10:40:00 | 490.99 | 492.74 | 0.00 | ORB-short ORB[491.61,496.60] vol=2.0x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 13:00:00 | 489.29 | 491.80 | 0.00 | T1 1.5R @ 489.29 |
| Target hit | 2024-05-23 15:05:00 | 490.41 | 490.35 | 0.00 | Trail-exit close>VWAP |

### Cycle 4 — SELL (started 2024-05-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 11:05:00 | 502.07 | 503.87 | 0.00 | ORB-short ORB[504.56,507.44] vol=1.6x ATR=1.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 11:30:00 | 499.96 | 503.51 | 0.00 | T1 1.5R @ 499.96 |
| Target hit | 2024-05-28 15:20:00 | 496.12 | 500.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2024-06-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-10 09:30:00 | 513.87 | 516.02 | 0.00 | ORB-short ORB[514.59,520.58] vol=2.9x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 09:35:00 | 511.57 | 515.21 | 0.00 | T1 1.5R @ 511.57 |
| Stop hit — per-position SL triggered | 2024-06-10 09:40:00 | 513.87 | 515.32 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 09:55:00 | 530.37 | 525.74 | 0.00 | ORB-long ORB[521.83,527.78] vol=1.7x ATR=2.52 |
| Stop hit — per-position SL triggered | 2024-06-11 10:05:00 | 527.85 | 526.10 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 11:15:00 | 526.19 | 529.50 | 0.00 | ORB-short ORB[527.10,531.37] vol=1.7x ATR=1.11 |
| Stop hit — per-position SL triggered | 2024-06-13 11:25:00 | 527.30 | 529.38 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 09:30:00 | 541.01 | 538.28 | 0.00 | ORB-long ORB[532.00,538.76] vol=4.1x ATR=1.87 |
| Stop hit — per-position SL triggered | 2024-06-20 09:35:00 | 539.14 | 538.62 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:40:00 | 549.36 | 545.01 | 0.00 | ORB-long ORB[541.64,548.21] vol=2.3x ATR=1.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 10:50:00 | 551.98 | 546.28 | 0.00 | T1 1.5R @ 551.98 |
| Stop hit — per-position SL triggered | 2024-06-26 11:00:00 | 549.36 | 546.59 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-01 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:50:00 | 550.08 | 548.11 | 0.00 | ORB-long ORB[544.18,549.60] vol=2.0x ATR=1.59 |
| Stop hit — per-position SL triggered | 2024-07-01 10:10:00 | 548.49 | 548.75 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:50:00 | 549.98 | 551.53 | 0.00 | ORB-short ORB[550.08,556.65] vol=4.7x ATR=1.55 |
| Stop hit — per-position SL triggered | 2024-07-02 09:55:00 | 551.53 | 551.53 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 11:10:00 | 548.97 | 547.69 | 0.00 | ORB-long ORB[544.08,548.35] vol=1.5x ATR=1.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 11:30:00 | 551.10 | 548.33 | 0.00 | T1 1.5R @ 551.10 |
| Stop hit — per-position SL triggered | 2024-07-03 14:30:00 | 548.97 | 549.47 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 09:40:00 | 545.09 | 548.35 | 0.00 | ORB-short ORB[547.82,551.37] vol=1.8x ATR=1.44 |
| Stop hit — per-position SL triggered | 2024-07-08 09:45:00 | 546.53 | 548.13 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 10:00:00 | 542.64 | 540.74 | 0.00 | ORB-long ORB[535.16,541.92] vol=1.6x ATR=1.63 |
| Stop hit — per-position SL triggered | 2024-07-11 10:15:00 | 541.01 | 541.07 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 09:30:00 | 545.23 | 538.75 | 0.00 | ORB-long ORB[535.35,540.53] vol=3.0x ATR=1.51 |
| Stop hit — per-position SL triggered | 2024-07-12 09:35:00 | 543.72 | 540.58 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 10:30:00 | 529.45 | 520.30 | 0.00 | ORB-long ORB[511.13,518.85] vol=1.5x ATR=2.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 10:45:00 | 532.61 | 522.20 | 0.00 | T1 1.5R @ 532.61 |
| Stop hit — per-position SL triggered | 2024-07-22 11:00:00 | 529.45 | 523.16 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 11:10:00 | 529.65 | 524.62 | 0.00 | ORB-long ORB[522.36,528.54] vol=9.5x ATR=2.07 |
| Stop hit — per-position SL triggered | 2024-07-23 11:15:00 | 527.58 | 524.58 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-25 09:55:00 | 504.71 | 508.45 | 0.00 | ORB-short ORB[506.00,513.00] vol=1.7x ATR=1.54 |
| Stop hit — per-position SL triggered | 2024-07-25 10:10:00 | 506.25 | 507.91 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:30:00 | 515.88 | 513.20 | 0.00 | ORB-long ORB[507.54,515.07] vol=1.7x ATR=1.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 09:40:00 | 518.31 | 514.72 | 0.00 | T1 1.5R @ 518.31 |
| Target hit | 2024-07-26 15:20:00 | 521.97 | 519.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 20 — BUY (started 2024-07-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 09:30:00 | 532.57 | 529.15 | 0.00 | ORB-long ORB[523.27,531.18] vol=2.8x ATR=1.60 |
| Stop hit — per-position SL triggered | 2024-07-29 09:35:00 | 530.97 | 530.06 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 11:10:00 | 541.30 | 537.85 | 0.00 | ORB-long ORB[529.98,537.18] vol=1.9x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 11:40:00 | 543.75 | 538.64 | 0.00 | T1 1.5R @ 543.75 |
| Stop hit — per-position SL triggered | 2024-07-30 12:35:00 | 541.30 | 539.18 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-02 09:35:00 | 529.21 | 532.42 | 0.00 | ORB-short ORB[529.50,535.31] vol=1.9x ATR=2.38 |
| Stop hit — per-position SL triggered | 2024-08-02 09:40:00 | 531.59 | 532.00 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 09:45:00 | 537.08 | 533.05 | 0.00 | ORB-long ORB[528.49,535.74] vol=2.1x ATR=2.15 |
| Stop hit — per-position SL triggered | 2024-08-09 10:00:00 | 534.93 | 534.15 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 09:30:00 | 534.68 | 537.17 | 0.00 | ORB-short ORB[536.36,543.03] vol=2.0x ATR=1.92 |
| Stop hit — per-position SL triggered | 2024-08-13 09:40:00 | 536.60 | 536.50 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-16 09:30:00 | 520.39 | 522.29 | 0.00 | ORB-short ORB[520.73,525.67] vol=1.6x ATR=1.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-16 10:00:00 | 518.29 | 521.08 | 0.00 | T1 1.5R @ 518.29 |
| Stop hit — per-position SL triggered | 2024-08-16 10:10:00 | 520.39 | 521.00 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 09:30:00 | 535.35 | 532.68 | 0.00 | ORB-long ORB[529.17,534.11] vol=3.0x ATR=1.80 |
| Stop hit — per-position SL triggered | 2024-08-19 09:50:00 | 533.55 | 533.46 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:55:00 | 547.78 | 545.26 | 0.00 | ORB-long ORB[540.05,547.54] vol=2.2x ATR=1.76 |
| Stop hit — per-position SL triggered | 2024-08-21 10:45:00 | 546.02 | 546.19 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-08-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 11:05:00 | 549.65 | 550.93 | 0.00 | ORB-short ORB[550.32,555.31] vol=2.0x ATR=1.00 |
| Stop hit — per-position SL triggered | 2024-08-26 11:20:00 | 550.65 | 550.91 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-08-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 09:30:00 | 561.35 | 559.11 | 0.00 | ORB-long ORB[555.79,560.53] vol=3.3x ATR=1.70 |
| Stop hit — per-position SL triggered | 2024-08-27 09:35:00 | 559.65 | 559.38 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 553.05 | 556.16 | 0.00 | ORB-short ORB[554.44,562.02] vol=2.2x ATR=1.41 |
| Stop hit — per-position SL triggered | 2024-08-28 09:40:00 | 554.46 | 555.78 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-08-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:50:00 | 565.81 | 562.82 | 0.00 | ORB-long ORB[558.28,565.76] vol=4.2x ATR=1.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 11:05:00 | 568.68 | 563.76 | 0.00 | T1 1.5R @ 568.68 |
| Target hit | 2024-08-30 15:20:00 | 573.48 | 571.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — BUY (started 2024-09-03 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:50:00 | 579.57 | 576.41 | 0.00 | ORB-long ORB[571.71,577.32] vol=2.6x ATR=1.73 |
| Stop hit — per-position SL triggered | 2024-09-03 09:55:00 | 577.84 | 576.55 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-09-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 10:00:00 | 580.05 | 575.81 | 0.00 | ORB-long ORB[568.88,576.98] vol=2.3x ATR=1.63 |
| Stop hit — per-position SL triggered | 2024-09-04 10:15:00 | 578.42 | 576.56 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-09-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 09:30:00 | 587.39 | 585.00 | 0.00 | ORB-long ORB[582.07,587.10] vol=3.0x ATR=1.83 |
| Stop hit — per-position SL triggered | 2024-09-05 09:50:00 | 585.56 | 585.50 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-09-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-06 09:30:00 | 594.73 | 591.24 | 0.00 | ORB-long ORB[588.93,591.85] vol=3.2x ATR=1.90 |
| Stop hit — per-position SL triggered | 2024-09-06 09:35:00 | 592.83 | 591.46 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-09-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 11:05:00 | 592.14 | 588.24 | 0.00 | ORB-long ORB[580.58,587.01] vol=2.7x ATR=1.77 |
| Stop hit — per-position SL triggered | 2024-09-10 11:15:00 | 590.37 | 588.37 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-09-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-11 10:50:00 | 589.22 | 591.14 | 0.00 | ORB-short ORB[589.45,593.87] vol=1.9x ATR=1.17 |
| Stop hit — per-position SL triggered | 2024-09-11 11:05:00 | 590.39 | 591.04 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-09-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 09:45:00 | 586.39 | 587.48 | 0.00 | ORB-short ORB[586.72,590.70] vol=2.0x ATR=1.44 |
| Stop hit — per-position SL triggered | 2024-09-17 10:00:00 | 587.83 | 587.28 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-09-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 10:25:00 | 582.16 | 584.74 | 0.00 | ORB-short ORB[582.93,589.69] vol=1.5x ATR=1.51 |
| Stop hit — per-position SL triggered | 2024-09-18 10:40:00 | 583.67 | 584.41 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-09-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 10:05:00 | 575.74 | 579.97 | 0.00 | ORB-short ORB[580.44,587.78] vol=5.1x ATR=2.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:15:00 | 572.42 | 578.83 | 0.00 | T1 1.5R @ 572.42 |
| Target hit | 2024-09-19 15:00:00 | 569.41 | 568.74 | 0.00 | Trail-exit close>VWAP |

### Cycle 41 — SELL (started 2024-09-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 10:30:00 | 570.46 | 570.91 | 0.00 | ORB-short ORB[571.56,577.80] vol=8.0x ATR=1.31 |
| Stop hit — per-position SL triggered | 2024-09-26 11:20:00 | 571.77 | 570.84 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-10-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 09:40:00 | 593.77 | 592.45 | 0.00 | ORB-long ORB[585.91,593.72] vol=2.3x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-01 10:10:00 | 596.59 | 593.41 | 0.00 | T1 1.5R @ 596.59 |
| Stop hit — per-position SL triggered | 2024-10-01 10:30:00 | 593.77 | 593.92 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-10-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-03 09:55:00 | 597.37 | 592.62 | 0.00 | ORB-long ORB[586.34,594.63] vol=2.2x ATR=2.09 |
| Stop hit — per-position SL triggered | 2024-10-03 10:00:00 | 595.28 | 592.88 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-10-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 09:35:00 | 569.79 | 572.89 | 0.00 | ORB-short ORB[573.34,580.82] vol=2.7x ATR=2.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:50:00 | 565.98 | 570.76 | 0.00 | T1 1.5R @ 565.98 |
| Target hit | 2024-10-07 11:15:00 | 561.92 | 561.07 | 0.00 | Trail-exit close>VWAP |

### Cycle 45 — BUY (started 2024-10-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 11:10:00 | 560.44 | 556.47 | 0.00 | ORB-long ORB[552.09,557.90] vol=1.6x ATR=2.43 |
| Stop hit — per-position SL triggered | 2024-10-08 11:25:00 | 558.01 | 557.19 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-10-14 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 10:10:00 | 554.59 | 558.28 | 0.00 | ORB-short ORB[556.79,562.98] vol=2.2x ATR=1.59 |
| Stop hit — per-position SL triggered | 2024-10-14 10:30:00 | 556.18 | 557.11 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-10-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 10:00:00 | 548.83 | 551.84 | 0.00 | ORB-short ORB[552.09,557.18] vol=1.5x ATR=1.51 |
| Stop hit — per-position SL triggered | 2024-10-15 10:25:00 | 550.34 | 550.62 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-10-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 10:45:00 | 543.70 | 547.12 | 0.00 | ORB-short ORB[546.77,552.91] vol=1.6x ATR=1.18 |
| Stop hit — per-position SL triggered | 2024-10-16 10:55:00 | 544.88 | 547.01 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-10-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 10:20:00 | 539.33 | 540.62 | 0.00 | ORB-short ORB[541.97,547.25] vol=1.6x ATR=1.51 |
| Stop hit — per-position SL triggered | 2024-10-17 10:25:00 | 540.84 | 540.60 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-10-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-21 10:00:00 | 538.81 | 535.42 | 0.00 | ORB-long ORB[532.86,538.57] vol=1.5x ATR=2.07 |
| Stop hit — per-position SL triggered | 2024-10-21 10:10:00 | 536.74 | 536.01 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-10-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-23 09:50:00 | 503.36 | 503.88 | 0.00 | ORB-short ORB[504.37,511.23] vol=2.1x ATR=2.68 |
| Stop hit — per-position SL triggered | 2024-10-23 10:05:00 | 506.04 | 503.97 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-10-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 10:45:00 | 503.60 | 509.97 | 0.00 | ORB-short ORB[510.13,516.26] vol=1.6x ATR=1.56 |
| Stop hit — per-position SL triggered | 2024-10-29 11:35:00 | 505.16 | 508.96 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-10-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 09:50:00 | 517.51 | 510.57 | 0.00 | ORB-long ORB[506.14,512.14] vol=1.6x ATR=2.03 |
| Stop hit — per-position SL triggered | 2024-10-30 09:55:00 | 515.48 | 510.93 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-10-31 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 09:55:00 | 528.11 | 523.60 | 0.00 | ORB-long ORB[520.39,526.34] vol=1.5x ATR=2.08 |
| Stop hit — per-position SL triggered | 2024-10-31 12:25:00 | 526.03 | 526.61 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-12-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-09 10:00:00 | 555.85 | 558.15 | 0.00 | ORB-short ORB[558.00,562.25] vol=4.5x ATR=2.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 10:10:00 | 552.59 | 557.32 | 0.00 | T1 1.5R @ 552.59 |
| Stop hit — per-position SL triggered | 2024-12-09 10:25:00 | 555.85 | 557.16 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:15:00 | 536.90 | 540.49 | 0.00 | ORB-short ORB[540.00,548.00] vol=1.6x ATR=1.68 |
| Stop hit — per-position SL triggered | 2024-12-13 10:50:00 | 538.58 | 539.39 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2024-12-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 09:55:00 | 550.65 | 547.51 | 0.00 | ORB-long ORB[544.10,549.80] vol=2.0x ATR=1.63 |
| Stop hit — per-position SL triggered | 2024-12-17 10:20:00 | 549.02 | 548.24 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-12-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 09:50:00 | 516.50 | 519.47 | 0.00 | ORB-short ORB[517.70,522.60] vol=1.9x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 09:55:00 | 514.17 | 518.62 | 0.00 | T1 1.5R @ 514.17 |
| Target hit | 2024-12-20 11:10:00 | 515.85 | 515.76 | 0.00 | Trail-exit close>VWAP |

### Cycle 59 — SELL (started 2024-12-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-23 11:00:00 | 499.05 | 500.14 | 0.00 | ORB-short ORB[500.50,507.65] vol=1.6x ATR=1.85 |
| Stop hit — per-position SL triggered | 2024-12-23 11:10:00 | 500.90 | 500.22 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2024-12-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 10:45:00 | 510.15 | 508.58 | 0.00 | ORB-long ORB[502.95,508.90] vol=1.8x ATR=1.57 |
| Stop hit — per-position SL triggered | 2024-12-24 10:55:00 | 508.58 | 508.65 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2024-12-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 09:30:00 | 499.80 | 502.50 | 0.00 | ORB-short ORB[501.20,507.05] vol=1.6x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 10:20:00 | 497.56 | 500.74 | 0.00 | T1 1.5R @ 497.56 |
| Stop hit — per-position SL triggered | 2024-12-26 11:00:00 | 499.80 | 500.14 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2024-12-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 10:00:00 | 504.50 | 501.12 | 0.00 | ORB-long ORB[497.00,503.20] vol=2.2x ATR=1.46 |
| Stop hit — per-position SL triggered | 2024-12-30 10:05:00 | 503.04 | 501.31 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-01-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:30:00 | 507.50 | 505.16 | 0.00 | ORB-long ORB[499.85,506.75] vol=3.0x ATR=1.41 |
| Stop hit — per-position SL triggered | 2025-01-02 10:05:00 | 506.09 | 506.68 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-01-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 10:00:00 | 513.75 | 511.91 | 0.00 | ORB-long ORB[508.50,513.30] vol=1.9x ATR=1.35 |
| Stop hit — per-position SL triggered | 2025-01-03 10:05:00 | 512.40 | 511.95 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-08 11:15:00 | 536.15 | 538.96 | 0.00 | ORB-short ORB[536.40,543.40] vol=1.6x ATR=1.51 |
| Stop hit — per-position SL triggered | 2025-01-08 11:25:00 | 537.66 | 538.92 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-01-15 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-15 10:20:00 | 548.30 | 545.81 | 0.00 | ORB-long ORB[543.30,547.00] vol=1.8x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-15 11:20:00 | 550.77 | 547.08 | 0.00 | T1 1.5R @ 550.77 |
| Stop hit — per-position SL triggered | 2025-01-15 11:30:00 | 548.30 | 547.20 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-01-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 10:25:00 | 555.80 | 550.72 | 0.00 | ORB-long ORB[543.50,550.35] vol=2.6x ATR=1.46 |
| Stop hit — per-position SL triggered | 2025-01-17 10:45:00 | 554.34 | 553.03 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-01-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-20 09:30:00 | 545.75 | 549.08 | 0.00 | ORB-short ORB[548.10,554.15] vol=1.8x ATR=1.52 |
| Stop hit — per-position SL triggered | 2025-01-20 09:35:00 | 547.27 | 548.89 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-01-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 09:35:00 | 549.05 | 545.30 | 0.00 | ORB-long ORB[542.05,546.70] vol=1.6x ATR=1.90 |
| Stop hit — per-position SL triggered | 2025-01-23 09:40:00 | 547.15 | 545.48 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-01-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 09:45:00 | 553.40 | 556.11 | 0.00 | ORB-short ORB[554.40,560.45] vol=1.7x ATR=1.51 |
| Stop hit — per-position SL triggered | 2025-01-24 10:10:00 | 554.91 | 555.41 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-01-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-27 09:50:00 | 542.05 | 544.19 | 0.00 | ORB-short ORB[544.30,550.40] vol=1.9x ATR=1.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:00:00 | 539.37 | 543.60 | 0.00 | T1 1.5R @ 539.37 |
| Target hit | 2025-01-27 13:30:00 | 541.05 | 540.77 | 0.00 | Trail-exit close>VWAP |

### Cycle 72 — BUY (started 2025-01-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 09:55:00 | 547.00 | 544.34 | 0.00 | ORB-long ORB[539.30,544.10] vol=2.3x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 10:40:00 | 549.90 | 546.25 | 0.00 | T1 1.5R @ 549.90 |
| Stop hit — per-position SL triggered | 2025-01-29 11:00:00 | 547.00 | 546.53 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2025-01-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-30 09:40:00 | 566.10 | 561.43 | 0.00 | ORB-long ORB[554.60,562.60] vol=2.2x ATR=1.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-30 09:45:00 | 569.06 | 563.08 | 0.00 | T1 1.5R @ 569.06 |
| Target hit | 2025-01-30 13:30:00 | 571.15 | 571.79 | 0.00 | Trail-exit close<VWAP |

### Cycle 74 — BUY (started 2025-02-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 09:30:00 | 639.40 | 635.75 | 0.00 | ORB-long ORB[632.00,636.90] vol=2.5x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-05 09:35:00 | 642.22 | 636.88 | 0.00 | T1 1.5R @ 642.22 |
| Target hit | 2025-02-05 15:20:00 | 645.55 | 643.60 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — SELL (started 2025-02-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 09:35:00 | 626.55 | 629.55 | 0.00 | ORB-short ORB[628.20,634.35] vol=1.6x ATR=2.02 |
| Stop hit — per-position SL triggered | 2025-02-18 09:50:00 | 628.57 | 628.88 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2025-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 09:30:00 | 654.85 | 651.82 | 0.00 | ORB-long ORB[645.80,654.45] vol=2.6x ATR=2.45 |
| Stop hit — per-position SL triggered | 2025-02-25 09:40:00 | 652.40 | 652.15 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2025-03-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-03 11:05:00 | 619.35 | 623.01 | 0.00 | ORB-short ORB[625.00,634.15] vol=4.7x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 11:35:00 | 614.65 | 622.28 | 0.00 | T1 1.5R @ 614.65 |
| Stop hit — per-position SL triggered | 2025-03-03 12:20:00 | 619.35 | 621.45 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2025-03-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 11:10:00 | 639.85 | 630.91 | 0.00 | ORB-long ORB[620.40,627.45] vol=3.0x ATR=1.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-07 11:20:00 | 642.64 | 632.50 | 0.00 | T1 1.5R @ 642.64 |
| Stop hit — per-position SL triggered | 2025-03-07 11:45:00 | 639.85 | 633.60 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2025-03-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 10:55:00 | 606.55 | 613.39 | 0.00 | ORB-short ORB[613.60,619.75] vol=1.9x ATR=2.09 |
| Stop hit — per-position SL triggered | 2025-03-12 11:05:00 | 608.64 | 612.42 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2025-03-19 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 10:50:00 | 645.15 | 641.73 | 0.00 | ORB-long ORB[636.90,641.45] vol=2.2x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 12:45:00 | 647.57 | 642.96 | 0.00 | T1 1.5R @ 647.57 |
| Stop hit — per-position SL triggered | 2025-03-19 13:35:00 | 645.15 | 643.34 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2025-03-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:35:00 | 654.65 | 651.52 | 0.00 | ORB-long ORB[645.65,653.55] vol=1.6x ATR=1.63 |
| Stop hit — per-position SL triggered | 2025-03-21 09:50:00 | 653.02 | 652.08 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2025-04-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 09:30:00 | 641.10 | 634.72 | 0.00 | ORB-long ORB[630.85,636.85] vol=1.6x ATR=2.14 |
| Stop hit — per-position SL triggered | 2025-04-02 09:45:00 | 638.96 | 637.48 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2025-04-04 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-04 11:00:00 | 653.05 | 644.65 | 0.00 | ORB-long ORB[642.25,651.65] vol=2.1x ATR=2.68 |
| Stop hit — per-position SL triggered | 2025-04-04 11:40:00 | 650.37 | 646.38 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2025-04-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-09 09:50:00 | 604.30 | 609.76 | 0.00 | ORB-short ORB[610.05,615.75] vol=1.6x ATR=2.19 |
| Stop hit — per-position SL triggered | 2025-04-09 10:05:00 | 606.49 | 608.30 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2025-04-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 09:35:00 | 665.00 | 661.56 | 0.00 | ORB-long ORB[655.10,663.50] vol=2.0x ATR=2.17 |
| Stop hit — per-position SL triggered | 2025-04-16 09:50:00 | 662.83 | 662.15 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2025-04-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 10:35:00 | 662.05 | 655.14 | 0.00 | ORB-long ORB[650.75,659.95] vol=4.2x ATR=2.33 |
| Stop hit — per-position SL triggered | 2025-04-17 12:15:00 | 659.72 | 658.12 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2025-04-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:30:00 | 663.90 | 658.30 | 0.00 | ORB-long ORB[653.70,661.20] vol=1.6x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 09:40:00 | 666.84 | 662.69 | 0.00 | T1 1.5R @ 666.84 |
| Stop hit — per-position SL triggered | 2025-04-21 09:45:00 | 663.90 | 663.37 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2025-04-22 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 10:30:00 | 680.35 | 674.69 | 0.00 | ORB-long ORB[666.10,674.25] vol=1.7x ATR=2.40 |
| Stop hit — per-position SL triggered | 2025-04-22 11:05:00 | 677.95 | 675.71 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2025-04-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:55:00 | 674.15 | 681.43 | 0.00 | ORB-short ORB[682.25,692.20] vol=2.4x ATR=2.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 10:00:00 | 670.33 | 680.07 | 0.00 | T1 1.5R @ 670.33 |
| Target hit | 2025-04-25 13:55:00 | 670.80 | 668.56 | 0.00 | Trail-exit close>VWAP |

### Cycle 90 — BUY (started 2025-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 11:15:00 | 677.00 | 673.77 | 0.00 | ORB-long ORB[666.00,672.65] vol=1.8x ATR=1.61 |
| Stop hit — per-position SL triggered | 2025-04-30 12:25:00 | 675.39 | 674.64 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-16 09:30:00 | 492.57 | 2024-05-16 09:35:00 | 490.42 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-05-16 09:30:00 | 492.57 | 2024-05-16 09:55:00 | 492.57 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-22 09:45:00 | 490.75 | 2024-05-22 09:55:00 | 492.24 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-05-23 10:40:00 | 490.99 | 2024-05-23 13:00:00 | 489.29 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-05-23 10:40:00 | 490.99 | 2024-05-23 15:05:00 | 490.41 | TARGET_HIT | 0.50 | 0.12% |
| SELL | retest1 | 2024-05-28 11:05:00 | 502.07 | 2024-05-28 11:30:00 | 499.96 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-05-28 11:05:00 | 502.07 | 2024-05-28 15:20:00 | 496.12 | TARGET_HIT | 0.50 | 1.19% |
| SELL | retest1 | 2024-06-10 09:30:00 | 513.87 | 2024-06-10 09:35:00 | 511.57 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-06-10 09:30:00 | 513.87 | 2024-06-10 09:40:00 | 513.87 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-11 09:55:00 | 530.37 | 2024-06-11 10:05:00 | 527.85 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2024-06-13 11:15:00 | 526.19 | 2024-06-13 11:25:00 | 527.30 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-06-20 09:30:00 | 541.01 | 2024-06-20 09:35:00 | 539.14 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-06-26 10:40:00 | 549.36 | 2024-06-26 10:50:00 | 551.98 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-06-26 10:40:00 | 549.36 | 2024-06-26 11:00:00 | 549.36 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-01 09:50:00 | 550.08 | 2024-07-01 10:10:00 | 548.49 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-07-02 09:50:00 | 549.98 | 2024-07-02 09:55:00 | 551.53 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-03 11:10:00 | 548.97 | 2024-07-03 11:30:00 | 551.10 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-07-03 11:10:00 | 548.97 | 2024-07-03 14:30:00 | 548.97 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-08 09:40:00 | 545.09 | 2024-07-08 09:45:00 | 546.53 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-07-11 10:00:00 | 542.64 | 2024-07-11 10:15:00 | 541.01 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-12 09:30:00 | 545.23 | 2024-07-12 09:35:00 | 543.72 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-22 10:30:00 | 529.45 | 2024-07-22 10:45:00 | 532.61 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-07-22 10:30:00 | 529.45 | 2024-07-22 11:00:00 | 529.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-23 11:10:00 | 529.65 | 2024-07-23 11:15:00 | 527.58 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-07-25 09:55:00 | 504.71 | 2024-07-25 10:10:00 | 506.25 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-07-26 09:30:00 | 515.88 | 2024-07-26 09:40:00 | 518.31 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-07-26 09:30:00 | 515.88 | 2024-07-26 15:20:00 | 521.97 | TARGET_HIT | 0.50 | 1.18% |
| BUY | retest1 | 2024-07-29 09:30:00 | 532.57 | 2024-07-29 09:35:00 | 530.97 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-07-30 11:10:00 | 541.30 | 2024-07-30 11:40:00 | 543.75 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-07-30 11:10:00 | 541.30 | 2024-07-30 12:35:00 | 541.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-02 09:35:00 | 529.21 | 2024-08-02 09:40:00 | 531.59 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-08-09 09:45:00 | 537.08 | 2024-08-09 10:00:00 | 534.93 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-08-13 09:30:00 | 534.68 | 2024-08-13 09:40:00 | 536.60 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-08-16 09:30:00 | 520.39 | 2024-08-16 10:00:00 | 518.29 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-08-16 09:30:00 | 520.39 | 2024-08-16 10:10:00 | 520.39 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-19 09:30:00 | 535.35 | 2024-08-19 09:50:00 | 533.55 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-08-21 09:55:00 | 547.78 | 2024-08-21 10:45:00 | 546.02 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-08-26 11:05:00 | 549.65 | 2024-08-26 11:20:00 | 550.65 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-08-27 09:30:00 | 561.35 | 2024-08-27 09:35:00 | 559.65 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-08-28 09:30:00 | 553.05 | 2024-08-28 09:40:00 | 554.46 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-08-30 10:50:00 | 565.81 | 2024-08-30 11:05:00 | 568.68 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-08-30 10:50:00 | 565.81 | 2024-08-30 15:20:00 | 573.48 | TARGET_HIT | 0.50 | 1.36% |
| BUY | retest1 | 2024-09-03 09:50:00 | 579.57 | 2024-09-03 09:55:00 | 577.84 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-09-04 10:00:00 | 580.05 | 2024-09-04 10:15:00 | 578.42 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-09-05 09:30:00 | 587.39 | 2024-09-05 09:50:00 | 585.56 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-09-06 09:30:00 | 594.73 | 2024-09-06 09:35:00 | 592.83 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-09-10 11:05:00 | 592.14 | 2024-09-10 11:15:00 | 590.37 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-09-11 10:50:00 | 589.22 | 2024-09-11 11:05:00 | 590.39 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-09-17 09:45:00 | 586.39 | 2024-09-17 10:00:00 | 587.83 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-09-18 10:25:00 | 582.16 | 2024-09-18 10:40:00 | 583.67 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-09-19 10:05:00 | 575.74 | 2024-09-19 10:15:00 | 572.42 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-09-19 10:05:00 | 575.74 | 2024-09-19 15:00:00 | 569.41 | TARGET_HIT | 0.50 | 1.10% |
| SELL | retest1 | 2024-09-26 10:30:00 | 570.46 | 2024-09-26 11:20:00 | 571.77 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-10-01 09:40:00 | 593.77 | 2024-10-01 10:10:00 | 596.59 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-10-01 09:40:00 | 593.77 | 2024-10-01 10:30:00 | 593.77 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-03 09:55:00 | 597.37 | 2024-10-03 10:00:00 | 595.28 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-10-07 09:35:00 | 569.79 | 2024-10-07 09:50:00 | 565.98 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2024-10-07 09:35:00 | 569.79 | 2024-10-07 11:15:00 | 561.92 | TARGET_HIT | 0.50 | 1.38% |
| BUY | retest1 | 2024-10-08 11:10:00 | 560.44 | 2024-10-08 11:25:00 | 558.01 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-10-14 10:10:00 | 554.59 | 2024-10-14 10:30:00 | 556.18 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-10-15 10:00:00 | 548.83 | 2024-10-15 10:25:00 | 550.34 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-10-16 10:45:00 | 543.70 | 2024-10-16 10:55:00 | 544.88 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-10-17 10:20:00 | 539.33 | 2024-10-17 10:25:00 | 540.84 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-10-21 10:00:00 | 538.81 | 2024-10-21 10:10:00 | 536.74 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-10-23 09:50:00 | 503.36 | 2024-10-23 10:05:00 | 506.04 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2024-10-29 10:45:00 | 503.60 | 2024-10-29 11:35:00 | 505.16 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-10-30 09:50:00 | 517.51 | 2024-10-30 09:55:00 | 515.48 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-10-31 09:55:00 | 528.11 | 2024-10-31 12:25:00 | 526.03 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-12-09 10:00:00 | 555.85 | 2024-12-09 10:10:00 | 552.59 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-12-09 10:00:00 | 555.85 | 2024-12-09 10:25:00 | 555.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-13 10:15:00 | 536.90 | 2024-12-13 10:50:00 | 538.58 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-12-17 09:55:00 | 550.65 | 2024-12-17 10:20:00 | 549.02 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-12-20 09:50:00 | 516.50 | 2024-12-20 09:55:00 | 514.17 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-12-20 09:50:00 | 516.50 | 2024-12-20 11:10:00 | 515.85 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2024-12-23 11:00:00 | 499.05 | 2024-12-23 11:10:00 | 500.90 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-12-24 10:45:00 | 510.15 | 2024-12-24 10:55:00 | 508.58 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-12-26 09:30:00 | 499.80 | 2024-12-26 10:20:00 | 497.56 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-12-26 09:30:00 | 499.80 | 2024-12-26 11:00:00 | 499.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-30 10:00:00 | 504.50 | 2024-12-30 10:05:00 | 503.04 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-01-02 09:30:00 | 507.50 | 2025-01-02 10:05:00 | 506.09 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-01-03 10:00:00 | 513.75 | 2025-01-03 10:05:00 | 512.40 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-01-08 11:15:00 | 536.15 | 2025-01-08 11:25:00 | 537.66 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-01-15 10:20:00 | 548.30 | 2025-01-15 11:20:00 | 550.77 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-01-15 10:20:00 | 548.30 | 2025-01-15 11:30:00 | 548.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-17 10:25:00 | 555.80 | 2025-01-17 10:45:00 | 554.34 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-01-20 09:30:00 | 545.75 | 2025-01-20 09:35:00 | 547.27 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-01-23 09:35:00 | 549.05 | 2025-01-23 09:40:00 | 547.15 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-01-24 09:45:00 | 553.40 | 2025-01-24 10:10:00 | 554.91 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-01-27 09:50:00 | 542.05 | 2025-01-27 10:00:00 | 539.37 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-01-27 09:50:00 | 542.05 | 2025-01-27 13:30:00 | 541.05 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2025-01-29 09:55:00 | 547.00 | 2025-01-29 10:40:00 | 549.90 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-01-29 09:55:00 | 547.00 | 2025-01-29 11:00:00 | 547.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-30 09:40:00 | 566.10 | 2025-01-30 09:45:00 | 569.06 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-01-30 09:40:00 | 566.10 | 2025-01-30 13:30:00 | 571.15 | TARGET_HIT | 0.50 | 0.89% |
| BUY | retest1 | 2025-02-05 09:30:00 | 639.40 | 2025-02-05 09:35:00 | 642.22 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-02-05 09:30:00 | 639.40 | 2025-02-05 15:20:00 | 645.55 | TARGET_HIT | 0.50 | 0.96% |
| SELL | retest1 | 2025-02-18 09:35:00 | 626.55 | 2025-02-18 09:50:00 | 628.57 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-02-25 09:30:00 | 654.85 | 2025-02-25 09:40:00 | 652.40 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-03-03 11:05:00 | 619.35 | 2025-03-03 11:35:00 | 614.65 | PARTIAL | 0.50 | 0.76% |
| SELL | retest1 | 2025-03-03 11:05:00 | 619.35 | 2025-03-03 12:20:00 | 619.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-07 11:10:00 | 639.85 | 2025-03-07 11:20:00 | 642.64 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-03-07 11:10:00 | 639.85 | 2025-03-07 11:45:00 | 639.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-12 10:55:00 | 606.55 | 2025-03-12 11:05:00 | 608.64 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-03-19 10:50:00 | 645.15 | 2025-03-19 12:45:00 | 647.57 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-03-19 10:50:00 | 645.15 | 2025-03-19 13:35:00 | 645.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-21 09:35:00 | 654.65 | 2025-03-21 09:50:00 | 653.02 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-04-02 09:30:00 | 641.10 | 2025-04-02 09:45:00 | 638.96 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-04-04 11:00:00 | 653.05 | 2025-04-04 11:40:00 | 650.37 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-04-09 09:50:00 | 604.30 | 2025-04-09 10:05:00 | 606.49 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-04-16 09:35:00 | 665.00 | 2025-04-16 09:50:00 | 662.83 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-04-17 10:35:00 | 662.05 | 2025-04-17 12:15:00 | 659.72 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-04-21 09:30:00 | 663.90 | 2025-04-21 09:40:00 | 666.84 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-04-21 09:30:00 | 663.90 | 2025-04-21 09:45:00 | 663.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-22 10:30:00 | 680.35 | 2025-04-22 11:05:00 | 677.95 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-04-25 09:55:00 | 674.15 | 2025-04-25 10:00:00 | 670.33 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-04-25 09:55:00 | 674.15 | 2025-04-25 13:55:00 | 670.80 | TARGET_HIT | 0.50 | 0.50% |
| BUY | retest1 | 2025-04-30 11:15:00 | 677.00 | 2025-04-30 12:25:00 | 675.39 | STOP_HIT | 1.00 | -0.24% |
