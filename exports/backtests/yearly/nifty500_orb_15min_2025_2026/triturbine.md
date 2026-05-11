# Triveni Turbine Ltd. (TRITURBINE)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 598.20
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
| ENTRY1 | 66 |
| ENTRY2 | 0 |
| PARTIAL | 24 |
| TARGET_HIT | 6 |
| STOP_HIT | 60 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 90 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 60
- **Target hits / Stop hits / Partials:** 6 / 60 / 24
- **Avg / median % per leg:** 0.07% / 0.00%
- **Sum % (uncompounded):** 6.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 14 | 35.0% | 3 | 26 | 11 | 0.02% | 0.7% |
| BUY @ 2nd Alert (retest1) | 40 | 14 | 35.0% | 3 | 26 | 11 | 0.02% | 0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 50 | 16 | 32.0% | 3 | 34 | 13 | 0.11% | 5.3% |
| SELL @ 2nd Alert (retest1) | 50 | 16 | 32.0% | 3 | 34 | 13 | 0.11% | 5.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 90 | 30 | 33.3% | 6 | 60 | 24 | 0.07% | 6.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-15 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 10:10:00 | 604.95 | 596.34 | 0.00 | ORB-long ORB[572.95,582.00] vol=2.4x ATR=4.97 |
| Stop hit — per-position SL triggered | 2025-05-15 10:20:00 | 599.98 | 597.65 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-21 09:40:00 | 575.80 | 581.39 | 0.00 | ORB-short ORB[579.05,586.80] vol=1.9x ATR=2.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-21 10:00:00 | 571.93 | 579.30 | 0.00 | T1 1.5R @ 571.93 |
| Stop hit — per-position SL triggered | 2025-05-21 10:25:00 | 575.80 | 578.30 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 10:20:00 | 592.60 | 585.33 | 0.00 | ORB-long ORB[577.70,586.05] vol=2.1x ATR=1.95 |
| Stop hit — per-position SL triggered | 2025-05-26 10:30:00 | 590.65 | 586.64 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-27 09:45:00 | 601.05 | 595.70 | 0.00 | ORB-long ORB[588.15,596.75] vol=4.4x ATR=2.43 |
| Stop hit — per-position SL triggered | 2025-05-27 09:55:00 | 598.62 | 596.75 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-05 09:35:00 | 586.40 | 590.93 | 0.00 | ORB-short ORB[590.00,596.70] vol=2.4x ATR=2.48 |
| Stop hit — per-position SL triggered | 2025-06-05 09:45:00 | 588.88 | 590.25 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:35:00 | 607.40 | 601.38 | 0.00 | ORB-long ORB[594.35,602.30] vol=5.1x ATR=2.69 |
| Stop hit — per-position SL triggered | 2025-06-17 09:40:00 | 604.71 | 601.86 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 10:30:00 | 610.35 | 605.82 | 0.00 | ORB-long ORB[602.00,609.90] vol=2.0x ATR=2.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-24 10:40:00 | 613.62 | 607.14 | 0.00 | T1 1.5R @ 613.62 |
| Target hit | 2025-06-24 12:45:00 | 611.95 | 612.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — SELL (started 2025-06-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 11:10:00 | 610.55 | 612.34 | 0.00 | ORB-short ORB[611.80,618.80] vol=1.9x ATR=1.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 12:10:00 | 608.03 | 611.88 | 0.00 | T1 1.5R @ 608.03 |
| Stop hit — per-position SL triggered | 2025-06-26 12:25:00 | 610.55 | 611.71 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-07-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:30:00 | 608.20 | 611.55 | 0.00 | ORB-short ORB[612.45,616.05] vol=1.6x ATR=1.58 |
| Stop hit — per-position SL triggered | 2025-07-01 10:35:00 | 609.78 | 611.49 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-07-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:40:00 | 604.00 | 606.36 | 0.00 | ORB-short ORB[606.00,609.00] vol=1.6x ATR=1.63 |
| Stop hit — per-position SL triggered | 2025-07-02 09:45:00 | 605.63 | 606.17 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-07-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 10:00:00 | 627.40 | 622.06 | 0.00 | ORB-long ORB[616.50,625.50] vol=1.8x ATR=2.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 10:05:00 | 631.27 | 624.76 | 0.00 | T1 1.5R @ 631.27 |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 627.40 | 625.71 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 09:30:00 | 648.70 | 645.00 | 0.00 | ORB-long ORB[639.05,647.50] vol=2.2x ATR=2.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 10:30:00 | 652.42 | 647.57 | 0.00 | T1 1.5R @ 652.42 |
| Stop hit — per-position SL triggered | 2025-07-09 11:35:00 | 648.70 | 648.30 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 11:00:00 | 644.10 | 646.74 | 0.00 | ORB-short ORB[644.25,650.00] vol=2.0x ATR=1.64 |
| Stop hit — per-position SL triggered | 2025-07-10 11:20:00 | 645.74 | 646.59 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:00:00 | 639.90 | 641.28 | 0.00 | ORB-short ORB[640.30,648.00] vol=1.6x ATR=2.16 |
| Stop hit — per-position SL triggered | 2025-07-11 11:35:00 | 642.06 | 641.05 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-18 09:30:00 | 656.95 | 653.36 | 0.00 | ORB-long ORB[650.45,654.90] vol=1.8x ATR=2.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 09:35:00 | 660.33 | 655.63 | 0.00 | T1 1.5R @ 660.33 |
| Stop hit — per-position SL triggered | 2025-07-18 09:40:00 | 656.95 | 656.00 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 11:05:00 | 660.65 | 667.66 | 0.00 | ORB-short ORB[665.50,672.80] vol=3.2x ATR=2.17 |
| Stop hit — per-position SL triggered | 2025-07-22 11:15:00 | 662.82 | 667.45 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 09:30:00 | 651.10 | 655.49 | 0.00 | ORB-short ORB[652.45,661.90] vol=3.4x ATR=3.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 09:55:00 | 646.57 | 652.54 | 0.00 | T1 1.5R @ 646.57 |
| Target hit | 2025-07-23 15:20:00 | 629.15 | 637.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2025-08-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 09:30:00 | 517.05 | 519.02 | 0.00 | ORB-short ORB[517.10,523.60] vol=1.7x ATR=1.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 09:55:00 | 514.48 | 517.57 | 0.00 | T1 1.5R @ 514.48 |
| Stop hit — per-position SL triggered | 2025-08-12 10:00:00 | 517.05 | 517.49 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-08-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-21 09:35:00 | 526.70 | 528.71 | 0.00 | ORB-short ORB[527.30,535.00] vol=2.1x ATR=1.25 |
| Stop hit — per-position SL triggered | 2025-08-21 09:55:00 | 527.95 | 528.40 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-08-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 10:35:00 | 534.50 | 531.28 | 0.00 | ORB-long ORB[529.00,533.95] vol=2.0x ATR=1.65 |
| Stop hit — per-position SL triggered | 2025-08-22 10:55:00 | 532.85 | 532.20 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-28 09:30:00 | 523.00 | 524.33 | 0.00 | ORB-short ORB[523.45,527.95] vol=2.3x ATR=1.72 |
| Stop hit — per-position SL triggered | 2025-08-28 09:40:00 | 524.72 | 524.26 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-09-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:55:00 | 508.30 | 511.91 | 0.00 | ORB-short ORB[511.50,517.75] vol=1.8x ATR=1.36 |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 509.66 | 511.07 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-09-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-09 09:55:00 | 512.60 | 509.35 | 0.00 | ORB-long ORB[506.00,512.00] vol=1.5x ATR=1.58 |
| Stop hit — per-position SL triggered | 2025-09-09 10:05:00 | 511.02 | 509.51 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-09-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-11 11:05:00 | 514.50 | 516.18 | 0.00 | ORB-short ORB[514.75,518.85] vol=1.6x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-11 11:50:00 | 512.74 | 515.79 | 0.00 | T1 1.5R @ 512.74 |
| Target hit | 2025-09-11 15:20:00 | 513.20 | 513.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — BUY (started 2025-09-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 10:35:00 | 520.70 | 517.28 | 0.00 | ORB-long ORB[513.65,519.10] vol=4.5x ATR=1.24 |
| Stop hit — per-position SL triggered | 2025-09-12 10:45:00 | 519.46 | 517.55 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-09-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 11:05:00 | 535.80 | 539.61 | 0.00 | ORB-short ORB[539.20,544.40] vol=1.5x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-18 11:45:00 | 533.99 | 538.88 | 0.00 | T1 1.5R @ 533.99 |
| Stop hit — per-position SL triggered | 2025-09-18 14:55:00 | 535.80 | 536.40 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-09-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 11:00:00 | 545.50 | 540.77 | 0.00 | ORB-long ORB[534.30,541.40] vol=4.6x ATR=1.86 |
| Stop hit — per-position SL triggered | 2025-09-22 11:10:00 | 543.64 | 540.99 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-09-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 10:40:00 | 536.00 | 538.55 | 0.00 | ORB-short ORB[538.50,542.30] vol=2.4x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 10:45:00 | 533.95 | 537.21 | 0.00 | T1 1.5R @ 533.95 |
| Stop hit — per-position SL triggered | 2025-09-23 10:50:00 | 536.00 | 537.03 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2025-09-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 09:45:00 | 523.50 | 525.46 | 0.00 | ORB-short ORB[525.10,531.05] vol=1.5x ATR=1.93 |
| Stop hit — per-position SL triggered | 2025-09-24 10:05:00 | 525.43 | 526.48 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-09-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 11:00:00 | 516.35 | 521.46 | 0.00 | ORB-short ORB[520.95,526.90] vol=1.6x ATR=1.64 |
| Stop hit — per-position SL triggered | 2025-09-25 11:10:00 | 517.99 | 521.22 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-10-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-01 11:00:00 | 523.75 | 522.02 | 0.00 | ORB-long ORB[517.25,522.90] vol=2.4x ATR=1.31 |
| Stop hit — per-position SL triggered | 2025-10-01 11:55:00 | 522.44 | 522.19 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-10-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-03 11:05:00 | 520.55 | 524.09 | 0.00 | ORB-short ORB[523.80,528.45] vol=1.9x ATR=1.06 |
| Stop hit — per-position SL triggered | 2025-10-03 11:45:00 | 521.61 | 523.42 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-10-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 10:55:00 | 521.15 | 523.57 | 0.00 | ORB-short ORB[521.30,526.15] vol=2.5x ATR=1.23 |
| Stop hit — per-position SL triggered | 2025-10-07 11:00:00 | 522.38 | 523.10 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-10-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 09:40:00 | 523.60 | 525.87 | 0.00 | ORB-short ORB[524.00,527.80] vol=1.6x ATR=1.10 |
| Stop hit — per-position SL triggered | 2025-10-14 09:45:00 | 524.70 | 525.68 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2025-10-16 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-16 09:55:00 | 520.35 | 522.43 | 0.00 | ORB-short ORB[523.50,525.50] vol=1.9x ATR=1.25 |
| Stop hit — per-position SL triggered | 2025-10-16 10:30:00 | 521.60 | 521.87 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-10-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 09:30:00 | 520.70 | 522.60 | 0.00 | ORB-short ORB[521.40,525.45] vol=1.7x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 09:45:00 | 518.96 | 521.39 | 0.00 | T1 1.5R @ 518.96 |
| Stop hit — per-position SL triggered | 2025-10-17 10:30:00 | 520.70 | 520.61 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-10-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:30:00 | 526.50 | 524.40 | 0.00 | ORB-long ORB[521.30,525.10] vol=3.3x ATR=1.35 |
| Stop hit — per-position SL triggered | 2025-10-29 10:45:00 | 525.15 | 524.67 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-10-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 09:35:00 | 532.30 | 537.52 | 0.00 | ORB-short ORB[535.00,543.00] vol=2.1x ATR=2.28 |
| Stop hit — per-position SL triggered | 2025-10-30 09:40:00 | 534.58 | 536.91 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 10:15:00 | 529.20 | 531.08 | 0.00 | ORB-short ORB[532.00,535.85] vol=1.8x ATR=1.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 10:35:00 | 527.10 | 530.70 | 0.00 | T1 1.5R @ 527.10 |
| Stop hit — per-position SL triggered | 2025-11-06 11:05:00 | 529.20 | 530.22 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-11-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 09:35:00 | 541.55 | 537.02 | 0.00 | ORB-long ORB[531.05,538.90] vol=2.1x ATR=2.09 |
| Stop hit — per-position SL triggered | 2025-11-14 15:20:00 | 540.35 | 540.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — BUY (started 2025-11-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-18 11:05:00 | 549.40 | 535.91 | 0.00 | ORB-long ORB[530.35,538.20] vol=2.6x ATR=2.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-18 11:10:00 | 553.41 | 536.47 | 0.00 | T1 1.5R @ 553.41 |
| Stop hit — per-position SL triggered | 2025-11-18 11:15:00 | 549.40 | 536.61 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-11-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 09:55:00 | 538.10 | 539.16 | 0.00 | ORB-short ORB[538.50,542.85] vol=2.8x ATR=0.93 |
| Stop hit — per-position SL triggered | 2025-11-21 10:10:00 | 539.03 | 538.47 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-11-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 11:10:00 | 539.55 | 541.27 | 0.00 | ORB-short ORB[540.45,546.95] vol=3.5x ATR=1.24 |
| Stop hit — per-position SL triggered | 2025-11-27 11:30:00 | 540.79 | 541.26 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-12-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 09:35:00 | 532.35 | 534.23 | 0.00 | ORB-short ORB[533.55,539.40] vol=2.4x ATR=1.43 |
| Stop hit — per-position SL triggered | 2025-12-01 09:50:00 | 533.78 | 533.07 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-12-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 11:05:00 | 538.70 | 536.52 | 0.00 | ORB-long ORB[534.15,538.25] vol=1.7x ATR=1.22 |
| Stop hit — per-position SL triggered | 2025-12-05 12:00:00 | 537.48 | 536.96 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-12-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 10:25:00 | 523.75 | 523.51 | 0.00 | ORB-long ORB[518.00,523.65] vol=1.9x ATR=1.30 |
| Stop hit — per-position SL triggered | 2025-12-19 10:55:00 | 522.45 | 523.52 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-12-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 10:55:00 | 542.75 | 540.21 | 0.00 | ORB-long ORB[536.40,541.10] vol=1.8x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 11:10:00 | 544.77 | 541.16 | 0.00 | T1 1.5R @ 544.77 |
| Target hit | 2025-12-24 15:20:00 | 544.95 | 544.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — BUY (started 2025-12-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 09:50:00 | 545.10 | 542.59 | 0.00 | ORB-long ORB[540.05,545.05] vol=2.1x ATR=1.59 |
| Target hit | 2025-12-26 15:20:00 | 545.35 | 544.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2025-12-30 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 11:00:00 | 529.75 | 535.74 | 0.00 | ORB-short ORB[537.00,544.70] vol=2.0x ATR=1.33 |
| Stop hit — per-position SL triggered | 2025-12-30 11:20:00 | 531.08 | 533.67 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-12-31 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:10:00 | 538.00 | 536.92 | 0.00 | ORB-long ORB[531.15,536.85] vol=2.1x ATR=1.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 10:15:00 | 540.63 | 537.89 | 0.00 | T1 1.5R @ 540.63 |
| Stop hit — per-position SL triggered | 2025-12-31 10:50:00 | 538.00 | 538.08 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2026-01-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 11:00:00 | 537.75 | 534.47 | 0.00 | ORB-long ORB[532.00,537.70] vol=3.0x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 11:05:00 | 539.60 | 535.52 | 0.00 | T1 1.5R @ 539.60 |
| Stop hit — per-position SL triggered | 2026-01-02 11:55:00 | 537.75 | 536.06 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2026-01-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-06 11:05:00 | 542.50 | 540.04 | 0.00 | ORB-long ORB[536.40,541.85] vol=1.7x ATR=1.53 |
| Stop hit — per-position SL triggered | 2026-01-06 11:15:00 | 540.97 | 540.07 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2026-01-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-07 10:45:00 | 529.35 | 531.64 | 0.00 | ORB-short ORB[532.50,537.05] vol=1.7x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 11:00:00 | 526.98 | 531.22 | 0.00 | T1 1.5R @ 526.98 |
| Stop hit — per-position SL triggered | 2026-01-07 11:30:00 | 529.35 | 530.80 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:15:00 | 529.80 | 532.31 | 0.00 | ORB-short ORB[530.05,535.75] vol=2.4x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:20:00 | 527.43 | 531.26 | 0.00 | T1 1.5R @ 527.43 |
| Stop hit — per-position SL triggered | 2026-01-08 13:20:00 | 529.80 | 529.71 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2026-01-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-16 10:30:00 | 503.55 | 504.76 | 0.00 | ORB-short ORB[504.00,508.80] vol=2.7x ATR=1.48 |
| Stop hit — per-position SL triggered | 2026-01-16 11:00:00 | 505.03 | 504.42 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2026-01-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 10:50:00 | 473.60 | 476.50 | 0.00 | ORB-short ORB[475.35,479.45] vol=1.9x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 11:00:00 | 471.67 | 476.24 | 0.00 | T1 1.5R @ 471.67 |
| Target hit | 2026-01-23 15:20:00 | 461.05 | 467.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — SELL (started 2026-01-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 11:00:00 | 471.90 | 479.32 | 0.00 | ORB-short ORB[479.80,484.90] vol=1.6x ATR=1.61 |
| Stop hit — per-position SL triggered | 2026-01-29 11:20:00 | 473.51 | 478.47 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2026-02-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-02 09:35:00 | 496.75 | 492.39 | 0.00 | ORB-long ORB[489.30,495.00] vol=1.5x ATR=2.39 |
| Stop hit — per-position SL triggered | 2026-02-02 09:45:00 | 494.36 | 494.51 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2026-02-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:00:00 | 510.15 | 508.42 | 0.00 | ORB-long ORB[505.00,509.25] vol=2.3x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:40:00 | 512.50 | 509.03 | 0.00 | T1 1.5R @ 512.50 |
| Stop hit — per-position SL triggered | 2026-02-09 11:45:00 | 510.15 | 509.04 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2026-02-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 09:45:00 | 490.85 | 489.92 | 0.00 | ORB-long ORB[485.50,489.95] vol=5.4x ATR=1.37 |
| Stop hit — per-position SL triggered | 2026-02-25 10:00:00 | 489.48 | 490.14 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 10:15:00 | 469.65 | 466.82 | 0.00 | ORB-long ORB[462.75,468.35] vol=1.9x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 11:05:00 | 472.35 | 467.99 | 0.00 | T1 1.5R @ 472.35 |
| Stop hit — per-position SL triggered | 2026-03-18 11:10:00 | 469.65 | 468.01 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2026-03-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:55:00 | 435.00 | 440.93 | 0.00 | ORB-short ORB[438.80,445.20] vol=1.8x ATR=1.92 |
| Stop hit — per-position SL triggered | 2026-03-24 11:25:00 | 436.92 | 440.29 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-04-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:55:00 | 465.50 | 462.87 | 0.00 | ORB-long ORB[457.05,462.00] vol=1.9x ATR=1.79 |
| Stop hit — per-position SL triggered | 2026-04-10 10:55:00 | 463.71 | 463.96 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-04-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 09:45:00 | 583.65 | 577.93 | 0.00 | ORB-long ORB[574.35,582.90] vol=2.3x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:15:00 | 587.45 | 582.68 | 0.00 | T1 1.5R @ 587.45 |
| Stop hit — per-position SL triggered | 2026-04-28 11:05:00 | 583.65 | 584.38 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2026-04-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:30:00 | 568.20 | 572.35 | 0.00 | ORB-short ORB[570.55,578.00] vol=1.5x ATR=2.58 |
| Stop hit — per-position SL triggered | 2026-04-30 10:40:00 | 570.78 | 572.24 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2026-05-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:50:00 | 570.25 | 572.56 | 0.00 | ORB-short ORB[570.55,578.65] vol=2.6x ATR=3.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 10:45:00 | 565.68 | 571.31 | 0.00 | T1 1.5R @ 565.68 |
| Stop hit — per-position SL triggered | 2026-05-08 11:10:00 | 570.25 | 570.32 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-15 10:10:00 | 604.95 | 2025-05-15 10:20:00 | 599.98 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest1 | 2025-05-21 09:40:00 | 575.80 | 2025-05-21 10:00:00 | 571.93 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2025-05-21 09:40:00 | 575.80 | 2025-05-21 10:25:00 | 575.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-26 10:20:00 | 592.60 | 2025-05-26 10:30:00 | 590.65 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-05-27 09:45:00 | 601.05 | 2025-05-27 09:55:00 | 598.62 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-06-05 09:35:00 | 586.40 | 2025-06-05 09:45:00 | 588.88 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-06-17 09:35:00 | 607.40 | 2025-06-17 09:40:00 | 604.71 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-06-24 10:30:00 | 610.35 | 2025-06-24 10:40:00 | 613.62 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-06-24 10:30:00 | 610.35 | 2025-06-24 12:45:00 | 611.95 | TARGET_HIT | 0.50 | 0.26% |
| SELL | retest1 | 2025-06-26 11:10:00 | 610.55 | 2025-06-26 12:10:00 | 608.03 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-06-26 11:10:00 | 610.55 | 2025-06-26 12:25:00 | 610.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-01 10:30:00 | 608.20 | 2025-07-01 10:35:00 | 609.78 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-07-02 09:40:00 | 604.00 | 2025-07-02 09:45:00 | 605.63 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-07-03 10:00:00 | 627.40 | 2025-07-03 10:05:00 | 631.27 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-07-03 10:00:00 | 627.40 | 2025-07-03 10:15:00 | 627.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-09 09:30:00 | 648.70 | 2025-07-09 10:30:00 | 652.42 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-07-09 09:30:00 | 648.70 | 2025-07-09 11:35:00 | 648.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-10 11:00:00 | 644.10 | 2025-07-10 11:20:00 | 645.74 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-07-11 10:00:00 | 639.90 | 2025-07-11 11:35:00 | 642.06 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-07-18 09:30:00 | 656.95 | 2025-07-18 09:35:00 | 660.33 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-07-18 09:30:00 | 656.95 | 2025-07-18 09:40:00 | 656.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-22 11:05:00 | 660.65 | 2025-07-22 11:15:00 | 662.82 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-07-23 09:30:00 | 651.10 | 2025-07-23 09:55:00 | 646.57 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2025-07-23 09:30:00 | 651.10 | 2025-07-23 15:20:00 | 629.15 | TARGET_HIT | 0.50 | 3.37% |
| SELL | retest1 | 2025-08-12 09:30:00 | 517.05 | 2025-08-12 09:55:00 | 514.48 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-08-12 09:30:00 | 517.05 | 2025-08-12 10:00:00 | 517.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-21 09:35:00 | 526.70 | 2025-08-21 09:55:00 | 527.95 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-08-22 10:35:00 | 534.50 | 2025-08-22 10:55:00 | 532.85 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-08-28 09:30:00 | 523.00 | 2025-08-28 09:40:00 | 524.72 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-09-05 10:55:00 | 508.30 | 2025-09-05 11:15:00 | 509.66 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-09-09 09:55:00 | 512.60 | 2025-09-09 10:05:00 | 511.02 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-09-11 11:05:00 | 514.50 | 2025-09-11 11:50:00 | 512.74 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-09-11 11:05:00 | 514.50 | 2025-09-11 15:20:00 | 513.20 | TARGET_HIT | 0.50 | 0.25% |
| BUY | retest1 | 2025-09-12 10:35:00 | 520.70 | 2025-09-12 10:45:00 | 519.46 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-09-18 11:05:00 | 535.80 | 2025-09-18 11:45:00 | 533.99 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-09-18 11:05:00 | 535.80 | 2025-09-18 14:55:00 | 535.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-22 11:00:00 | 545.50 | 2025-09-22 11:10:00 | 543.64 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-09-23 10:40:00 | 536.00 | 2025-09-23 10:45:00 | 533.95 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-09-23 10:40:00 | 536.00 | 2025-09-23 10:50:00 | 536.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-24 09:45:00 | 523.50 | 2025-09-24 10:05:00 | 525.43 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-09-25 11:00:00 | 516.35 | 2025-09-25 11:10:00 | 517.99 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-10-01 11:00:00 | 523.75 | 2025-10-01 11:55:00 | 522.44 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-10-03 11:05:00 | 520.55 | 2025-10-03 11:45:00 | 521.61 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-10-07 10:55:00 | 521.15 | 2025-10-07 11:00:00 | 522.38 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-10-14 09:40:00 | 523.60 | 2025-10-14 09:45:00 | 524.70 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-10-16 09:55:00 | 520.35 | 2025-10-16 10:30:00 | 521.60 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-10-17 09:30:00 | 520.70 | 2025-10-17 09:45:00 | 518.96 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-10-17 09:30:00 | 520.70 | 2025-10-17 10:30:00 | 520.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-29 10:30:00 | 526.50 | 2025-10-29 10:45:00 | 525.15 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-10-30 09:35:00 | 532.30 | 2025-10-30 09:40:00 | 534.58 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-11-06 10:15:00 | 529.20 | 2025-11-06 10:35:00 | 527.10 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-11-06 10:15:00 | 529.20 | 2025-11-06 11:05:00 | 529.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-14 09:35:00 | 541.55 | 2025-11-14 15:20:00 | 540.35 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-11-18 11:05:00 | 549.40 | 2025-11-18 11:10:00 | 553.41 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2025-11-18 11:05:00 | 549.40 | 2025-11-18 11:15:00 | 549.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-21 09:55:00 | 538.10 | 2025-11-21 10:10:00 | 539.03 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-11-27 11:10:00 | 539.55 | 2025-11-27 11:30:00 | 540.79 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-12-01 09:35:00 | 532.35 | 2025-12-01 09:50:00 | 533.78 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-12-05 11:05:00 | 538.70 | 2025-12-05 12:00:00 | 537.48 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-12-19 10:25:00 | 523.75 | 2025-12-19 10:55:00 | 522.45 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-12-24 10:55:00 | 542.75 | 2025-12-24 11:10:00 | 544.77 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-12-24 10:55:00 | 542.75 | 2025-12-24 15:20:00 | 544.95 | TARGET_HIT | 0.50 | 0.41% |
| BUY | retest1 | 2025-12-26 09:50:00 | 545.10 | 2025-12-26 15:20:00 | 545.35 | TARGET_HIT | 1.00 | 0.05% |
| SELL | retest1 | 2025-12-30 11:00:00 | 529.75 | 2025-12-30 11:20:00 | 531.08 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-12-31 10:10:00 | 538.00 | 2025-12-31 10:15:00 | 540.63 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-12-31 10:10:00 | 538.00 | 2025-12-31 10:50:00 | 538.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-02 11:00:00 | 537.75 | 2026-01-02 11:05:00 | 539.60 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2026-01-02 11:00:00 | 537.75 | 2026-01-02 11:55:00 | 537.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-06 11:05:00 | 542.50 | 2026-01-06 11:15:00 | 540.97 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-01-07 10:45:00 | 529.35 | 2026-01-07 11:00:00 | 526.98 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-01-07 10:45:00 | 529.35 | 2026-01-07 11:30:00 | 529.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-08 10:15:00 | 529.80 | 2026-01-08 11:20:00 | 527.43 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-01-08 10:15:00 | 529.80 | 2026-01-08 13:20:00 | 529.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-16 10:30:00 | 503.55 | 2026-01-16 11:00:00 | 505.03 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-01-23 10:50:00 | 473.60 | 2026-01-23 11:00:00 | 471.67 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2026-01-23 10:50:00 | 473.60 | 2026-01-23 15:20:00 | 461.05 | TARGET_HIT | 0.50 | 2.65% |
| SELL | retest1 | 2026-01-29 11:00:00 | 471.90 | 2026-01-29 11:20:00 | 473.51 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-02-02 09:35:00 | 496.75 | 2026-02-02 09:45:00 | 494.36 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-02-09 11:00:00 | 510.15 | 2026-02-09 11:40:00 | 512.50 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-02-09 11:00:00 | 510.15 | 2026-02-09 11:45:00 | 510.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-25 09:45:00 | 490.85 | 2026-02-25 10:00:00 | 489.48 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-03-18 10:15:00 | 469.65 | 2026-03-18 11:05:00 | 472.35 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-03-18 10:15:00 | 469.65 | 2026-03-18 11:10:00 | 469.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-24 10:55:00 | 435.00 | 2026-03-24 11:25:00 | 436.92 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-04-10 09:55:00 | 465.50 | 2026-04-10 10:55:00 | 463.71 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-28 09:45:00 | 583.65 | 2026-04-28 10:15:00 | 587.45 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-04-28 09:45:00 | 583.65 | 2026-04-28 11:05:00 | 583.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-30 10:30:00 | 568.20 | 2026-04-30 10:40:00 | 570.78 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-05-08 09:50:00 | 570.25 | 2026-05-08 10:45:00 | 565.68 | PARTIAL | 0.50 | 0.80% |
| SELL | retest1 | 2026-05-08 09:50:00 | 570.25 | 2026-05-08 11:10:00 | 570.25 | STOP_HIT | 0.50 | 0.00% |
