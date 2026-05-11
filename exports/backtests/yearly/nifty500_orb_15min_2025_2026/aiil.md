# Authum Investment & Infrastructure Ltd. (AIIL)

## Backtest Summary

- **Window:** 2025-07-10 09:15:00 → 2026-05-08 15:25:00 (12163 bars)
- **Last close:** 494.80
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
| PARTIAL | 12 |
| TARGET_HIT | 7 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 54 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 35
- **Target hits / Stop hits / Partials:** 7 / 35 / 12
- **Avg / median % per leg:** 0.07% / -0.26%
- **Sum % (uncompounded):** 3.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 27 | 10 | 37.0% | 4 | 17 | 6 | 0.05% | 1.4% |
| BUY @ 2nd Alert (retest1) | 27 | 10 | 37.0% | 4 | 17 | 6 | 0.05% | 1.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 27 | 9 | 33.3% | 3 | 18 | 6 | 0.08% | 2.1% |
| SELL @ 2nd Alert (retest1) | 27 | 9 | 33.3% | 3 | 18 | 6 | 0.08% | 2.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 54 | 19 | 35.2% | 7 | 35 | 12 | 0.07% | 3.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:55:00 | 521.06 | 530.02 | 0.00 | ORB-short ORB[533.08,538.20] vol=1.9x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 12:20:00 | 517.88 | 524.43 | 0.00 | T1 1.5R @ 517.88 |
| Stop hit — per-position SL triggered | 2025-07-11 14:20:00 | 521.06 | 521.39 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-07-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 10:05:00 | 578.42 | 576.66 | 0.00 | ORB-long ORB[573.00,578.00] vol=1.5x ATR=2.21 |
| Stop hit — per-position SL triggered | 2025-07-23 11:15:00 | 576.21 | 577.19 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-08-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-05 10:10:00 | 564.32 | 561.58 | 0.00 | ORB-long ORB[556.02,563.00] vol=6.0x ATR=2.76 |
| Stop hit — per-position SL triggered | 2025-08-05 10:55:00 | 561.56 | 561.97 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-08-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 10:55:00 | 563.60 | 571.95 | 0.00 | ORB-short ORB[574.10,579.20] vol=2.5x ATR=2.26 |
| Stop hit — per-position SL triggered | 2025-08-06 11:00:00 | 565.86 | 571.57 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-08-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-08 09:50:00 | 578.78 | 575.22 | 0.00 | ORB-long ORB[568.86,576.00] vol=3.8x ATR=3.11 |
| Stop hit — per-position SL triggered | 2025-08-08 10:35:00 | 575.67 | 576.23 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-08-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 10:30:00 | 571.82 | 574.28 | 0.00 | ORB-short ORB[572.50,579.00] vol=2.8x ATR=2.06 |
| Stop hit — per-position SL triggered | 2025-08-12 10:35:00 | 573.88 | 574.26 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-08-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 09:55:00 | 584.44 | 579.66 | 0.00 | ORB-long ORB[575.62,580.00] vol=2.8x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 10:00:00 | 587.17 | 583.48 | 0.00 | T1 1.5R @ 587.17 |
| Target hit | 2025-08-13 11:45:00 | 589.18 | 590.16 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — BUY (started 2025-08-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:30:00 | 580.88 | 578.81 | 0.00 | ORB-long ORB[573.72,579.76] vol=4.3x ATR=3.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 10:25:00 | 585.54 | 580.21 | 0.00 | T1 1.5R @ 585.54 |
| Target hit | 2025-08-21 12:05:00 | 581.30 | 582.32 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2025-08-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 09:45:00 | 609.40 | 603.36 | 0.00 | ORB-long ORB[594.00,601.34] vol=1.6x ATR=2.36 |
| Stop hit — per-position SL triggered | 2025-08-25 12:30:00 | 607.04 | 607.36 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-08-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-26 10:10:00 | 615.88 | 609.91 | 0.00 | ORB-long ORB[602.70,610.60] vol=3.0x ATR=2.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 10:20:00 | 619.11 | 611.25 | 0.00 | T1 1.5R @ 619.11 |
| Stop hit — per-position SL triggered | 2025-08-26 10:25:00 | 615.88 | 611.31 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-09-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 09:55:00 | 625.78 | 621.75 | 0.00 | ORB-long ORB[617.42,624.00] vol=3.0x ATR=2.23 |
| Stop hit — per-position SL triggered | 2025-09-12 11:35:00 | 623.55 | 624.26 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-09-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-15 09:30:00 | 623.32 | 625.95 | 0.00 | ORB-short ORB[624.54,630.00] vol=1.6x ATR=1.73 |
| Stop hit — per-position SL triggered | 2025-09-15 09:35:00 | 625.05 | 625.87 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 10:15:00 | 642.96 | 644.61 | 0.00 | ORB-short ORB[648.00,656.20] vol=1.6x ATR=2.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-18 11:35:00 | 639.26 | 643.87 | 0.00 | T1 1.5R @ 639.26 |
| Target hit | 2025-09-18 14:25:00 | 636.86 | 636.71 | 0.00 | Trail-exit close>VWAP |

### Cycle 14 — SELL (started 2025-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 09:30:00 | 598.56 | 601.23 | 0.00 | ORB-short ORB[598.80,607.70] vol=2.4x ATR=1.98 |
| Stop hit — per-position SL triggered | 2025-09-26 09:40:00 | 600.54 | 600.86 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-10-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 09:30:00 | 644.34 | 641.80 | 0.00 | ORB-long ORB[637.04,642.80] vol=2.0x ATR=2.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 09:55:00 | 647.61 | 644.50 | 0.00 | T1 1.5R @ 647.61 |
| Target hit | 2025-10-07 12:20:00 | 646.32 | 646.64 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — BUY (started 2025-10-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 09:50:00 | 660.42 | 659.22 | 0.00 | ORB-long ORB[653.66,660.00] vol=2.1x ATR=1.95 |
| Stop hit — per-position SL triggered | 2025-10-10 10:40:00 | 658.47 | 659.33 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-10-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-23 10:10:00 | 628.20 | 631.66 | 0.00 | ORB-short ORB[630.42,637.80] vol=2.3x ATR=1.70 |
| Stop hit — per-position SL triggered | 2025-10-23 10:25:00 | 629.90 | 631.30 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-10-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 11:05:00 | 615.60 | 619.02 | 0.00 | ORB-short ORB[616.82,623.28] vol=2.6x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 11:25:00 | 613.72 | 618.38 | 0.00 | T1 1.5R @ 613.72 |
| Stop hit — per-position SL triggered | 2025-10-24 12:00:00 | 615.60 | 617.93 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-29 10:15:00 | 594.38 | 597.88 | 0.00 | ORB-short ORB[597.00,604.60] vol=1.9x ATR=1.33 |
| Stop hit — per-position SL triggered | 2025-10-29 10:25:00 | 595.71 | 597.55 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-10-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 10:55:00 | 586.76 | 590.08 | 0.00 | ORB-short ORB[592.16,596.16] vol=1.7x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-30 11:30:00 | 584.26 | 589.07 | 0.00 | T1 1.5R @ 584.26 |
| Target hit | 2025-10-30 15:20:00 | 583.26 | 586.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — SELL (started 2025-11-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 11:05:00 | 557.22 | 563.39 | 0.00 | ORB-short ORB[570.36,577.98] vol=4.2x ATR=1.63 |
| Stop hit — per-position SL triggered | 2025-11-06 11:10:00 | 558.85 | 563.08 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-11-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 10:45:00 | 560.14 | 558.43 | 0.00 | ORB-long ORB[550.42,555.86] vol=6.8x ATR=2.13 |
| Stop hit — per-position SL triggered | 2025-11-07 11:00:00 | 558.01 | 559.11 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-11-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 11:00:00 | 549.36 | 544.37 | 0.00 | ORB-long ORB[543.00,547.10] vol=1.5x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 11:10:00 | 552.19 | 544.92 | 0.00 | T1 1.5R @ 552.19 |
| Stop hit — per-position SL triggered | 2025-11-12 11:25:00 | 549.36 | 545.69 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-11-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 10:55:00 | 567.06 | 566.35 | 0.00 | ORB-long ORB[560.00,567.02] vol=1.9x ATR=1.81 |
| Stop hit — per-position SL triggered | 2025-11-13 11:40:00 | 565.25 | 566.55 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-11-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-14 09:55:00 | 562.60 | 566.15 | 0.00 | ORB-short ORB[564.00,569.00] vol=2.1x ATR=1.94 |
| Stop hit — per-position SL triggered | 2025-11-14 10:55:00 | 564.54 | 564.97 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-11-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-17 09:35:00 | 546.00 | 549.47 | 0.00 | ORB-short ORB[548.00,554.36] vol=2.1x ATR=1.90 |
| Stop hit — per-position SL triggered | 2025-11-17 09:40:00 | 547.90 | 549.36 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-11-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 09:45:00 | 556.62 | 558.73 | 0.00 | ORB-short ORB[558.00,562.12] vol=3.1x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 09:55:00 | 554.79 | 558.32 | 0.00 | T1 1.5R @ 554.79 |
| Target hit | 2025-11-19 15:20:00 | 541.72 | 546.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2025-11-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 10:50:00 | 538.88 | 540.29 | 0.00 | ORB-short ORB[540.00,548.12] vol=1.8x ATR=2.14 |
| Stop hit — per-position SL triggered | 2025-11-20 10:55:00 | 541.02 | 540.30 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-11-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-21 09:45:00 | 542.98 | 540.52 | 0.00 | ORB-long ORB[536.00,542.34] vol=2.9x ATR=1.86 |
| Stop hit — per-position SL triggered | 2025-11-21 10:05:00 | 541.12 | 541.06 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-12-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 11:00:00 | 526.20 | 530.10 | 0.00 | ORB-short ORB[527.72,535.04] vol=3.1x ATR=1.34 |
| Stop hit — per-position SL triggered | 2025-12-02 11:05:00 | 527.54 | 529.51 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-12-05 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 10:05:00 | 533.52 | 529.52 | 0.00 | ORB-long ORB[520.00,527.20] vol=2.0x ATR=1.41 |
| Stop hit — per-position SL triggered | 2025-12-05 10:10:00 | 532.11 | 529.91 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-12-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 11:00:00 | 528.98 | 527.66 | 0.00 | ORB-long ORB[523.96,527.52] vol=1.5x ATR=1.40 |
| Stop hit — per-position SL triggered | 2025-12-12 11:35:00 | 527.58 | 527.95 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-12-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 09:50:00 | 515.20 | 516.35 | 0.00 | ORB-short ORB[517.00,523.46] vol=2.6x ATR=2.12 |
| Stop hit — per-position SL triggered | 2025-12-15 10:45:00 | 517.32 | 516.26 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-12-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 09:30:00 | 548.56 | 544.72 | 0.00 | ORB-long ORB[540.00,547.52] vol=1.7x ATR=1.80 |
| Stop hit — per-position SL triggered | 2025-12-19 09:40:00 | 546.76 | 544.97 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-12-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-23 10:55:00 | 570.92 | 573.49 | 0.00 | ORB-short ORB[574.38,580.56] vol=1.6x ATR=1.49 |
| Stop hit — per-position SL triggered | 2025-12-23 11:00:00 | 572.41 | 573.43 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-12-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-29 11:05:00 | 581.42 | 579.24 | 0.00 | ORB-long ORB[574.00,579.82] vol=1.6x ATR=1.93 |
| Stop hit — per-position SL triggered | 2025-12-29 11:10:00 | 579.49 | 579.28 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-12-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 10:40:00 | 579.46 | 574.60 | 0.00 | ORB-long ORB[570.40,576.18] vol=6.6x ATR=2.25 |
| Stop hit — per-position SL triggered | 2025-12-30 10:50:00 | 577.21 | 575.59 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-02 11:15:00 | 614.60 | 618.57 | 0.00 | ORB-short ORB[618.00,626.24] vol=2.1x ATR=2.09 |
| Stop hit — per-position SL triggered | 2026-01-02 11:20:00 | 616.69 | 618.47 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2026-03-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:50:00 | 429.00 | 431.73 | 0.00 | ORB-short ORB[429.45,435.80] vol=2.1x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:20:00 | 426.49 | 430.96 | 0.00 | T1 1.5R @ 426.49 |
| Stop hit — per-position SL triggered | 2026-03-05 11:25:00 | 429.00 | 430.95 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2026-03-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 10:00:00 | 454.05 | 448.59 | 0.00 | ORB-long ORB[443.20,449.00] vol=2.4x ATR=2.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:05:00 | 457.79 | 463.33 | 0.00 | T1 1.5R @ 457.79 |
| Target hit | 2026-03-13 10:15:00 | 463.95 | 469.91 | 0.00 | Trail-exit close<VWAP |

### Cycle 41 — SELL (started 2026-03-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 09:35:00 | 480.10 | 484.12 | 0.00 | ORB-short ORB[483.10,488.40] vol=2.3x ATR=1.90 |
| Stop hit — per-position SL triggered | 2026-03-20 09:50:00 | 482.00 | 483.51 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2026-05-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 11:05:00 | 468.45 | 461.17 | 0.00 | ORB-long ORB[456.00,462.20] vol=5.1x ATR=2.67 |
| Stop hit — per-position SL triggered | 2026-05-06 11:40:00 | 465.78 | 462.33 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-07-11 10:55:00 | 521.06 | 2025-07-11 12:20:00 | 517.88 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-07-11 10:55:00 | 521.06 | 2025-07-11 14:20:00 | 521.06 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-23 10:05:00 | 578.42 | 2025-07-23 11:15:00 | 576.21 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-08-05 10:10:00 | 564.32 | 2025-08-05 10:55:00 | 561.56 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2025-08-06 10:55:00 | 563.60 | 2025-08-06 11:00:00 | 565.86 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-08-08 09:50:00 | 578.78 | 2025-08-08 10:35:00 | 575.67 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2025-08-12 10:30:00 | 571.82 | 2025-08-12 10:35:00 | 573.88 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-08-13 09:55:00 | 584.44 | 2025-08-13 10:00:00 | 587.17 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-08-13 09:55:00 | 584.44 | 2025-08-13 11:45:00 | 589.18 | TARGET_HIT | 0.50 | 0.81% |
| BUY | retest1 | 2025-08-21 09:30:00 | 580.88 | 2025-08-21 10:25:00 | 585.54 | PARTIAL | 0.50 | 0.80% |
| BUY | retest1 | 2025-08-21 09:30:00 | 580.88 | 2025-08-21 12:05:00 | 581.30 | TARGET_HIT | 0.50 | 0.07% |
| BUY | retest1 | 2025-08-25 09:45:00 | 609.40 | 2025-08-25 12:30:00 | 607.04 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-08-26 10:10:00 | 615.88 | 2025-08-26 10:20:00 | 619.11 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-08-26 10:10:00 | 615.88 | 2025-08-26 10:25:00 | 615.88 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-12 09:55:00 | 625.78 | 2025-09-12 11:35:00 | 623.55 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-09-15 09:30:00 | 623.32 | 2025-09-15 09:35:00 | 625.05 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-09-18 10:15:00 | 642.96 | 2025-09-18 11:35:00 | 639.26 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2025-09-18 10:15:00 | 642.96 | 2025-09-18 14:25:00 | 636.86 | TARGET_HIT | 0.50 | 0.95% |
| SELL | retest1 | 2025-09-26 09:30:00 | 598.56 | 2025-09-26 09:40:00 | 600.54 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-10-07 09:30:00 | 644.34 | 2025-10-07 09:55:00 | 647.61 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-10-07 09:30:00 | 644.34 | 2025-10-07 12:20:00 | 646.32 | TARGET_HIT | 0.50 | 0.31% |
| BUY | retest1 | 2025-10-10 09:50:00 | 660.42 | 2025-10-10 10:40:00 | 658.47 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-10-23 10:10:00 | 628.20 | 2025-10-23 10:25:00 | 629.90 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-10-24 11:05:00 | 615.60 | 2025-10-24 11:25:00 | 613.72 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-10-24 11:05:00 | 615.60 | 2025-10-24 12:00:00 | 615.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-29 10:15:00 | 594.38 | 2025-10-29 10:25:00 | 595.71 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-10-30 10:55:00 | 586.76 | 2025-10-30 11:30:00 | 584.26 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-10-30 10:55:00 | 586.76 | 2025-10-30 15:20:00 | 583.26 | TARGET_HIT | 0.50 | 0.60% |
| SELL | retest1 | 2025-11-06 11:05:00 | 557.22 | 2025-11-06 11:10:00 | 558.85 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-11-07 10:45:00 | 560.14 | 2025-11-07 11:00:00 | 558.01 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-11-12 11:00:00 | 549.36 | 2025-11-12 11:10:00 | 552.19 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-11-12 11:00:00 | 549.36 | 2025-11-12 11:25:00 | 549.36 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-13 10:55:00 | 567.06 | 2025-11-13 11:40:00 | 565.25 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-11-14 09:55:00 | 562.60 | 2025-11-14 10:55:00 | 564.54 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-11-17 09:35:00 | 546.00 | 2025-11-17 09:40:00 | 547.90 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-11-19 09:45:00 | 556.62 | 2025-11-19 09:55:00 | 554.79 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-11-19 09:45:00 | 556.62 | 2025-11-19 15:20:00 | 541.72 | TARGET_HIT | 0.50 | 2.68% |
| SELL | retest1 | 2025-11-20 10:50:00 | 538.88 | 2025-11-20 10:55:00 | 541.02 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-11-21 09:45:00 | 542.98 | 2025-11-21 10:05:00 | 541.12 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-12-02 11:00:00 | 526.20 | 2025-12-02 11:05:00 | 527.54 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-12-05 10:05:00 | 533.52 | 2025-12-05 10:10:00 | 532.11 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-12-12 11:00:00 | 528.98 | 2025-12-12 11:35:00 | 527.58 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-12-15 09:50:00 | 515.20 | 2025-12-15 10:45:00 | 517.32 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-12-19 09:30:00 | 548.56 | 2025-12-19 09:40:00 | 546.76 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-12-23 10:55:00 | 570.92 | 2025-12-23 11:00:00 | 572.41 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-12-29 11:05:00 | 581.42 | 2025-12-29 11:10:00 | 579.49 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-12-30 10:40:00 | 579.46 | 2025-12-30 10:50:00 | 577.21 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-01-02 11:15:00 | 614.60 | 2026-01-02 11:20:00 | 616.69 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-05 10:50:00 | 429.00 | 2026-03-05 11:20:00 | 426.49 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-03-05 10:50:00 | 429.00 | 2026-03-05 11:25:00 | 429.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-13 10:00:00 | 454.05 | 2026-03-13 10:05:00 | 457.79 | PARTIAL | 0.50 | 0.82% |
| BUY | retest1 | 2026-03-13 10:00:00 | 454.05 | 2026-03-13 10:15:00 | 463.95 | TARGET_HIT | 0.50 | 2.18% |
| SELL | retest1 | 2026-03-20 09:35:00 | 480.10 | 2026-03-20 09:50:00 | 482.00 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-05-06 11:05:00 | 468.45 | 2026-05-06 11:40:00 | 465.78 | STOP_HIT | 1.00 | -0.57% |
