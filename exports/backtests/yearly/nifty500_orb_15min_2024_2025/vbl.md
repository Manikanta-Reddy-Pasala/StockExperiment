# Varun Beverages Ltd. (VBL)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-05-05 15:25:00 (18108 bars)
- **Last close:** 540.00
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
| ENTRY1 | 58 |
| ENTRY2 | 0 |
| PARTIAL | 31 |
| TARGET_HIT | 13 |
| STOP_HIT | 45 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 89 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 44 / 45
- **Target hits / Stop hits / Partials:** 13 / 45 / 31
- **Avg / median % per leg:** 0.19% / 0.00%
- **Sum % (uncompounded):** 17.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 12 | 38.7% | 2 | 19 | 10 | 0.10% | 3.1% |
| BUY @ 2nd Alert (retest1) | 31 | 12 | 38.7% | 2 | 19 | 10 | 0.10% | 3.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 58 | 32 | 55.2% | 11 | 26 | 21 | 0.24% | 13.9% |
| SELL @ 2nd Alert (retest1) | 58 | 32 | 55.2% | 11 | 26 | 21 | 0.24% | 13.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 89 | 44 | 49.4% | 13 | 45 | 31 | 0.19% | 17.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 09:55:00 | 604.24 | 598.72 | 0.00 | ORB-long ORB[594.40,602.36] vol=1.7x ATR=3.02 |
| Stop hit — per-position SL triggered | 2024-05-15 10:10:00 | 601.22 | 600.25 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 09:30:00 | 592.56 | 595.14 | 0.00 | ORB-short ORB[593.76,599.72] vol=1.8x ATR=1.47 |
| Stop hit — per-position SL triggered | 2024-05-16 09:35:00 | 594.03 | 594.92 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-23 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-23 10:30:00 | 601.30 | 604.60 | 0.00 | ORB-short ORB[604.18,608.00] vol=2.9x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 10:40:00 | 599.59 | 603.99 | 0.00 | T1 1.5R @ 599.59 |
| Stop hit — per-position SL triggered | 2024-05-23 11:20:00 | 601.30 | 603.34 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 09:50:00 | 594.18 | 596.03 | 0.00 | ORB-short ORB[594.80,600.36] vol=1.9x ATR=2.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-27 10:55:00 | 590.87 | 595.10 | 0.00 | T1 1.5R @ 590.87 |
| Target hit | 2024-05-27 13:10:00 | 593.80 | 593.80 | 0.00 | Trail-exit close>VWAP |

### Cycle 5 — SELL (started 2024-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 11:15:00 | 592.20 | 595.58 | 0.00 | ORB-short ORB[593.20,597.38] vol=3.5x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 12:20:00 | 590.33 | 593.08 | 0.00 | T1 1.5R @ 590.33 |
| Target hit | 2024-05-28 15:20:00 | 586.56 | 591.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2024-05-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 09:30:00 | 566.16 | 569.27 | 0.00 | ORB-short ORB[567.60,575.52] vol=1.6x ATR=2.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 09:50:00 | 562.44 | 567.56 | 0.00 | T1 1.5R @ 562.44 |
| Stop hit — per-position SL triggered | 2024-05-31 10:20:00 | 566.16 | 566.99 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 09:30:00 | 608.42 | 603.49 | 0.00 | ORB-long ORB[599.76,607.58] vol=2.4x ATR=1.60 |
| Stop hit — per-position SL triggered | 2024-06-07 09:35:00 | 606.82 | 604.60 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 10:40:00 | 645.36 | 638.43 | 0.00 | ORB-long ORB[629.82,637.92] vol=6.9x ATR=2.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 10:45:00 | 649.29 | 639.48 | 0.00 | T1 1.5R @ 649.29 |
| Stop hit — per-position SL triggered | 2024-06-14 10:50:00 | 645.36 | 639.83 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 10:45:00 | 640.02 | 644.31 | 0.00 | ORB-short ORB[644.12,650.46] vol=1.5x ATR=1.49 |
| Stop hit — per-position SL triggered | 2024-06-21 10:50:00 | 641.51 | 644.17 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-06-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 09:30:00 | 636.60 | 634.03 | 0.00 | ORB-long ORB[628.64,635.98] vol=1.8x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 09:35:00 | 639.42 | 635.23 | 0.00 | T1 1.5R @ 639.42 |
| Stop hit — per-position SL triggered | 2024-06-25 09:40:00 | 636.60 | 635.42 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-26 10:55:00 | 628.32 | 631.43 | 0.00 | ORB-short ORB[629.30,635.20] vol=2.7x ATR=1.72 |
| Stop hit — per-position SL triggered | 2024-06-26 11:05:00 | 630.04 | 631.31 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-01 11:15:00 | 645.46 | 648.99 | 0.00 | ORB-short ORB[645.60,652.00] vol=1.5x ATR=1.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 11:40:00 | 642.78 | 648.01 | 0.00 | T1 1.5R @ 642.78 |
| Target hit | 2024-07-01 15:20:00 | 644.18 | 643.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2024-07-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-08 09:40:00 | 653.74 | 649.73 | 0.00 | ORB-long ORB[644.62,652.48] vol=2.7x ATR=2.31 |
| Stop hit — per-position SL triggered | 2024-07-08 10:05:00 | 651.43 | 651.06 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 11:15:00 | 640.50 | 645.58 | 0.00 | ORB-short ORB[646.68,651.50] vol=2.1x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 14:30:00 | 637.25 | 641.54 | 0.00 | T1 1.5R @ 637.25 |
| Target hit | 2024-07-10 15:20:00 | 639.98 | 641.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — SELL (started 2024-07-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 10:30:00 | 633.30 | 636.91 | 0.00 | ORB-short ORB[634.78,639.52] vol=3.8x ATR=2.08 |
| Stop hit — per-position SL triggered | 2024-07-12 10:50:00 | 635.38 | 636.67 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 11:15:00 | 643.04 | 637.12 | 0.00 | ORB-long ORB[629.74,637.38] vol=2.6x ATR=1.75 |
| Stop hit — per-position SL triggered | 2024-07-15 11:20:00 | 641.29 | 637.24 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-16 10:40:00 | 646.18 | 650.42 | 0.00 | ORB-short ORB[648.80,657.88] vol=2.1x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 11:50:00 | 643.25 | 649.38 | 0.00 | T1 1.5R @ 643.25 |
| Target hit | 2024-07-16 15:20:00 | 639.40 | 644.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2024-07-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 10:40:00 | 618.90 | 621.84 | 0.00 | ORB-short ORB[622.42,629.34] vol=1.7x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 616.42 | 619.83 | 0.00 | T1 1.5R @ 616.42 |
| Stop hit — per-position SL triggered | 2024-07-23 13:10:00 | 618.90 | 615.69 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 11:00:00 | 670.80 | 664.91 | 0.00 | ORB-long ORB[661.60,669.60] vol=3.7x ATR=1.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 12:55:00 | 673.42 | 666.90 | 0.00 | T1 1.5R @ 673.42 |
| Stop hit — per-position SL triggered | 2024-07-26 13:50:00 | 670.80 | 668.70 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-08-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 10:50:00 | 629.18 | 633.53 | 0.00 | ORB-short ORB[633.28,637.20] vol=3.2x ATR=1.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 11:05:00 | 626.78 | 632.59 | 0.00 | T1 1.5R @ 626.78 |
| Stop hit — per-position SL triggered | 2024-08-01 11:25:00 | 629.18 | 632.05 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 11:00:00 | 619.70 | 611.26 | 0.00 | ORB-long ORB[603.90,612.74] vol=4.6x ATR=2.70 |
| Stop hit — per-position SL triggered | 2024-08-21 11:10:00 | 617.00 | 612.09 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 09:30:00 | 634.88 | 627.96 | 0.00 | ORB-long ORB[621.34,627.20] vol=3.8x ATR=2.71 |
| Stop hit — per-position SL triggered | 2024-08-22 09:35:00 | 632.17 | 629.12 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 10:55:00 | 626.40 | 630.79 | 0.00 | ORB-short ORB[630.60,635.94] vol=2.0x ATR=1.57 |
| Stop hit — per-position SL triggered | 2024-08-27 11:55:00 | 627.97 | 630.01 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 620.18 | 624.38 | 0.00 | ORB-short ORB[622.12,631.18] vol=2.7x ATR=2.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-28 10:35:00 | 616.79 | 621.46 | 0.00 | T1 1.5R @ 616.79 |
| Target hit | 2024-08-28 15:20:00 | 610.04 | 614.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — SELL (started 2024-09-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 09:50:00 | 604.18 | 608.66 | 0.00 | ORB-short ORB[604.22,613.16] vol=1.8x ATR=2.30 |
| Stop hit — per-position SL triggered | 2024-09-05 10:25:00 | 606.48 | 607.41 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-09-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:10:00 | 599.40 | 600.50 | 0.00 | ORB-short ORB[600.22,606.80] vol=2.2x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 13:00:00 | 597.05 | 599.76 | 0.00 | T1 1.5R @ 597.05 |
| Stop hit — per-position SL triggered | 2024-09-06 15:05:00 | 599.40 | 598.46 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-09-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:55:00 | 617.60 | 609.93 | 0.00 | ORB-long ORB[602.56,611.12] vol=1.6x ATR=4.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 10:10:00 | 624.34 | 614.63 | 0.00 | T1 1.5R @ 624.34 |
| Target hit | 2024-09-11 12:30:00 | 626.96 | 626.99 | 0.00 | Trail-exit close<VWAP |

### Cycle 28 — BUY (started 2024-09-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 09:50:00 | 636.35 | 626.93 | 0.00 | ORB-long ORB[616.70,626.00] vol=2.7x ATR=3.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 09:55:00 | 640.92 | 629.97 | 0.00 | T1 1.5R @ 640.92 |
| Stop hit — per-position SL triggered | 2024-09-17 10:00:00 | 636.35 | 631.29 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-09-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-23 11:00:00 | 651.00 | 653.42 | 0.00 | ORB-short ORB[652.30,657.00] vol=2.6x ATR=1.90 |
| Stop hit — per-position SL triggered | 2024-09-23 11:05:00 | 652.90 | 653.38 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-09-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 10:55:00 | 644.10 | 653.88 | 0.00 | ORB-short ORB[654.00,663.00] vol=1.7x ATR=1.98 |
| Stop hit — per-position SL triggered | 2024-09-24 11:00:00 | 646.08 | 653.34 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-10-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 11:05:00 | 588.65 | 596.28 | 0.00 | ORB-short ORB[594.60,600.55] vol=1.6x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:45:00 | 585.67 | 594.99 | 0.00 | T1 1.5R @ 585.67 |
| Stop hit — per-position SL triggered | 2024-10-17 12:35:00 | 588.65 | 593.49 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-08 11:15:00 | 591.60 | 594.84 | 0.00 | ORB-short ORB[591.75,598.90] vol=2.9x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-08 12:40:00 | 589.59 | 593.89 | 0.00 | T1 1.5R @ 589.59 |
| Stop hit — per-position SL triggered | 2024-11-08 13:25:00 | 591.60 | 593.51 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-11-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 09:45:00 | 590.50 | 593.88 | 0.00 | ORB-short ORB[591.55,599.00] vol=1.5x ATR=2.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 13:35:00 | 586.73 | 591.60 | 0.00 | T1 1.5R @ 586.73 |
| Target hit | 2024-11-12 15:20:00 | 583.90 | 589.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — SELL (started 2024-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 11:15:00 | 618.80 | 624.06 | 0.00 | ORB-short ORB[621.10,629.20] vol=2.7x ATR=1.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 11:30:00 | 616.09 | 623.31 | 0.00 | T1 1.5R @ 616.09 |
| Stop hit — per-position SL triggered | 2024-11-29 11:45:00 | 618.80 | 622.79 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-12-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 09:45:00 | 651.80 | 648.69 | 0.00 | ORB-long ORB[642.75,650.85] vol=1.6x ATR=2.17 |
| Stop hit — per-position SL triggered | 2024-12-10 10:00:00 | 649.63 | 649.25 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-12-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 09:40:00 | 638.75 | 642.08 | 0.00 | ORB-short ORB[639.00,646.40] vol=1.7x ATR=1.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 10:05:00 | 636.04 | 640.87 | 0.00 | T1 1.5R @ 636.04 |
| Stop hit — per-position SL triggered | 2024-12-12 10:15:00 | 638.75 | 640.68 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-12-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-13 10:55:00 | 639.25 | 642.80 | 0.00 | ORB-short ORB[642.10,649.80] vol=1.8x ATR=1.86 |
| Stop hit — per-position SL triggered | 2024-12-13 11:30:00 | 641.11 | 642.33 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-12-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 10:25:00 | 650.70 | 647.80 | 0.00 | ORB-long ORB[644.00,649.50] vol=2.6x ATR=1.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 10:35:00 | 653.21 | 648.73 | 0.00 | T1 1.5R @ 653.21 |
| Stop hit — per-position SL triggered | 2024-12-16 10:45:00 | 650.70 | 648.90 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-12-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 11:10:00 | 626.50 | 624.48 | 0.00 | ORB-long ORB[620.00,626.45] vol=1.7x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 11:40:00 | 628.66 | 624.93 | 0.00 | T1 1.5R @ 628.66 |
| Stop hit — per-position SL triggered | 2024-12-24 12:05:00 | 626.50 | 625.23 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-12-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 11:10:00 | 625.80 | 624.73 | 0.00 | ORB-long ORB[621.00,624.55] vol=1.7x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 11:25:00 | 627.68 | 625.06 | 0.00 | T1 1.5R @ 627.68 |
| Stop hit — per-position SL triggered | 2024-12-30 11:35:00 | 625.80 | 625.38 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-01-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:45:00 | 658.40 | 652.11 | 0.00 | ORB-long ORB[648.00,655.00] vol=3.1x ATR=1.85 |
| Stop hit — per-position SL triggered | 2025-01-02 10:50:00 | 656.55 | 652.84 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-01-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:45:00 | 635.00 | 639.16 | 0.00 | ORB-short ORB[640.50,645.95] vol=4.4x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:15:00 | 632.03 | 637.80 | 0.00 | T1 1.5R @ 632.03 |
| Target hit | 2025-01-06 15:20:00 | 629.80 | 632.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 43 — SELL (started 2025-01-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 10:45:00 | 584.50 | 587.63 | 0.00 | ORB-short ORB[585.00,591.35] vol=1.6x ATR=2.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 10:55:00 | 581.49 | 586.96 | 0.00 | T1 1.5R @ 581.49 |
| Target hit | 2025-01-13 15:20:00 | 574.90 | 579.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — SELL (started 2025-01-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-22 11:05:00 | 535.15 | 540.84 | 0.00 | ORB-short ORB[538.80,546.80] vol=2.7x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 12:00:00 | 532.74 | 539.06 | 0.00 | T1 1.5R @ 532.74 |
| Target hit | 2025-01-22 14:55:00 | 534.80 | 533.05 | 0.00 | Trail-exit close>VWAP |

### Cycle 45 — SELL (started 2025-01-31 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-31 10:45:00 | 539.30 | 539.85 | 0.00 | ORB-short ORB[539.35,544.00] vol=2.5x ATR=1.44 |
| Stop hit — per-position SL triggered | 2025-01-31 11:00:00 | 540.74 | 540.15 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-02-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 10:45:00 | 568.75 | 576.09 | 0.00 | ORB-short ORB[577.40,585.85] vol=2.1x ATR=1.84 |
| Stop hit — per-position SL triggered | 2025-02-06 11:40:00 | 570.59 | 574.43 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-03-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-05 10:50:00 | 475.00 | 480.18 | 0.00 | ORB-short ORB[479.00,485.70] vol=1.9x ATR=2.87 |
| Stop hit — per-position SL triggered | 2025-03-05 11:30:00 | 477.87 | 478.94 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-06 11:15:00 | 479.85 | 481.66 | 0.00 | ORB-short ORB[481.35,487.40] vol=2.6x ATR=1.70 |
| Stop hit — per-position SL triggered | 2025-03-06 11:40:00 | 481.55 | 481.45 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-03-12 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-12 10:35:00 | 488.80 | 485.74 | 0.00 | ORB-long ORB[482.35,488.35] vol=1.7x ATR=1.81 |
| Stop hit — per-position SL triggered | 2025-03-12 11:25:00 | 486.99 | 487.01 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-03-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-17 09:50:00 | 500.00 | 496.21 | 0.00 | ORB-long ORB[491.55,498.30] vol=1.7x ATR=2.01 |
| Stop hit — per-position SL triggered | 2025-03-17 10:10:00 | 497.99 | 496.69 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-03-19 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 10:05:00 | 538.30 | 535.12 | 0.00 | ORB-long ORB[530.05,536.60] vol=2.1x ATR=2.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 10:35:00 | 541.30 | 536.25 | 0.00 | T1 1.5R @ 541.30 |
| Target hit | 2025-03-19 13:45:00 | 539.25 | 540.48 | 0.00 | Trail-exit close<VWAP |

### Cycle 52 — SELL (started 2025-03-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-25 10:45:00 | 525.10 | 528.75 | 0.00 | ORB-short ORB[528.95,535.00] vol=2.3x ATR=2.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-25 11:05:00 | 521.81 | 528.29 | 0.00 | T1 1.5R @ 521.81 |
| Target hit | 2025-03-25 15:20:00 | 516.75 | 520.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — BUY (started 2025-04-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 10:45:00 | 545.50 | 540.81 | 0.00 | ORB-long ORB[535.45,543.55] vol=2.4x ATR=1.88 |
| Stop hit — per-position SL triggered | 2025-04-02 12:10:00 | 543.62 | 542.67 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-04-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-16 11:00:00 | 544.85 | 549.74 | 0.00 | ORB-short ORB[548.45,553.00] vol=1.8x ATR=1.40 |
| Stop hit — per-position SL triggered | 2025-04-16 11:10:00 | 546.25 | 549.59 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-04-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:05:00 | 548.90 | 553.72 | 0.00 | ORB-short ORB[551.95,557.80] vol=1.5x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 10:55:00 | 545.85 | 552.20 | 0.00 | T1 1.5R @ 545.85 |
| Stop hit — per-position SL triggered | 2025-04-23 11:05:00 | 548.90 | 552.09 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-04-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-24 09:45:00 | 541.85 | 549.17 | 0.00 | ORB-short ORB[548.50,553.20] vol=1.8x ATR=1.80 |
| Stop hit — per-position SL triggered | 2025-04-24 09:50:00 | 543.65 | 548.67 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-04-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 09:40:00 | 529.35 | 533.44 | 0.00 | ORB-short ORB[531.50,539.00] vol=2.6x ATR=2.11 |
| Stop hit — per-position SL triggered | 2025-04-29 09:55:00 | 531.46 | 532.59 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-04-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-30 10:25:00 | 533.60 | 531.48 | 0.00 | ORB-long ORB[526.10,531.90] vol=5.2x ATR=1.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-30 11:05:00 | 536.27 | 531.89 | 0.00 | T1 1.5R @ 536.27 |
| Stop hit — per-position SL triggered | 2025-04-30 11:20:00 | 533.60 | 532.32 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-15 09:55:00 | 604.24 | 2024-05-15 10:10:00 | 601.22 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-05-16 09:30:00 | 592.56 | 2024-05-16 09:35:00 | 594.03 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-05-23 10:30:00 | 601.30 | 2024-05-23 10:40:00 | 599.59 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2024-05-23 10:30:00 | 601.30 | 2024-05-23 11:20:00 | 601.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-27 09:50:00 | 594.18 | 2024-05-27 10:55:00 | 590.87 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-05-27 09:50:00 | 594.18 | 2024-05-27 13:10:00 | 593.80 | TARGET_HIT | 0.50 | 0.06% |
| SELL | retest1 | 2024-05-28 11:15:00 | 592.20 | 2024-05-28 12:20:00 | 590.33 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-05-28 11:15:00 | 592.20 | 2024-05-28 15:20:00 | 586.56 | TARGET_HIT | 0.50 | 0.95% |
| SELL | retest1 | 2024-05-31 09:30:00 | 566.16 | 2024-05-31 09:50:00 | 562.44 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-05-31 09:30:00 | 566.16 | 2024-05-31 10:20:00 | 566.16 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-07 09:30:00 | 608.42 | 2024-06-07 09:35:00 | 606.82 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-06-14 10:40:00 | 645.36 | 2024-06-14 10:45:00 | 649.29 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-06-14 10:40:00 | 645.36 | 2024-06-14 10:50:00 | 645.36 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-21 10:45:00 | 640.02 | 2024-06-21 10:50:00 | 641.51 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-06-25 09:30:00 | 636.60 | 2024-06-25 09:35:00 | 639.42 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-06-25 09:30:00 | 636.60 | 2024-06-25 09:40:00 | 636.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-26 10:55:00 | 628.32 | 2024-06-26 11:05:00 | 630.04 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-07-01 11:15:00 | 645.46 | 2024-07-01 11:40:00 | 642.78 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-07-01 11:15:00 | 645.46 | 2024-07-01 15:20:00 | 644.18 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2024-07-08 09:40:00 | 653.74 | 2024-07-08 10:05:00 | 651.43 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-07-10 11:15:00 | 640.50 | 2024-07-10 14:30:00 | 637.25 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-07-10 11:15:00 | 640.50 | 2024-07-10 15:20:00 | 639.98 | TARGET_HIT | 0.50 | 0.08% |
| SELL | retest1 | 2024-07-12 10:30:00 | 633.30 | 2024-07-12 10:50:00 | 635.38 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-07-15 11:15:00 | 643.04 | 2024-07-15 11:20:00 | 641.29 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-07-16 10:40:00 | 646.18 | 2024-07-16 11:50:00 | 643.25 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-07-16 10:40:00 | 646.18 | 2024-07-16 15:20:00 | 639.40 | TARGET_HIT | 0.50 | 1.05% |
| SELL | retest1 | 2024-07-23 10:40:00 | 618.90 | 2024-07-23 12:15:00 | 616.42 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-07-23 10:40:00 | 618.90 | 2024-07-23 13:10:00 | 618.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-26 11:00:00 | 670.80 | 2024-07-26 12:55:00 | 673.42 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-07-26 11:00:00 | 670.80 | 2024-07-26 13:50:00 | 670.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-01 10:50:00 | 629.18 | 2024-08-01 11:05:00 | 626.78 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-08-01 10:50:00 | 629.18 | 2024-08-01 11:25:00 | 629.18 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-21 11:00:00 | 619.70 | 2024-08-21 11:10:00 | 617.00 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-08-22 09:30:00 | 634.88 | 2024-08-22 09:35:00 | 632.17 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-08-27 10:55:00 | 626.40 | 2024-08-27 11:55:00 | 627.97 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-08-28 09:30:00 | 620.18 | 2024-08-28 10:35:00 | 616.79 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-08-28 09:30:00 | 620.18 | 2024-08-28 15:20:00 | 610.04 | TARGET_HIT | 0.50 | 1.64% |
| SELL | retest1 | 2024-09-05 09:50:00 | 604.18 | 2024-09-05 10:25:00 | 606.48 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-09-06 10:10:00 | 599.40 | 2024-09-06 13:00:00 | 597.05 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-09-06 10:10:00 | 599.40 | 2024-09-06 15:05:00 | 599.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-11 09:55:00 | 617.60 | 2024-09-11 10:10:00 | 624.34 | PARTIAL | 0.50 | 1.09% |
| BUY | retest1 | 2024-09-11 09:55:00 | 617.60 | 2024-09-11 12:30:00 | 626.96 | TARGET_HIT | 0.50 | 1.52% |
| BUY | retest1 | 2024-09-17 09:50:00 | 636.35 | 2024-09-17 09:55:00 | 640.92 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2024-09-17 09:50:00 | 636.35 | 2024-09-17 10:00:00 | 636.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-23 11:00:00 | 651.00 | 2024-09-23 11:05:00 | 652.90 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-09-24 10:55:00 | 644.10 | 2024-09-24 11:00:00 | 646.08 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-10-17 11:05:00 | 588.65 | 2024-10-17 11:45:00 | 585.67 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-10-17 11:05:00 | 588.65 | 2024-10-17 12:35:00 | 588.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-08 11:15:00 | 591.60 | 2024-11-08 12:40:00 | 589.59 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-11-08 11:15:00 | 591.60 | 2024-11-08 13:25:00 | 591.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-12 09:45:00 | 590.50 | 2024-11-12 13:35:00 | 586.73 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2024-11-12 09:45:00 | 590.50 | 2024-11-12 15:20:00 | 583.90 | TARGET_HIT | 0.50 | 1.12% |
| SELL | retest1 | 2024-11-29 11:15:00 | 618.80 | 2024-11-29 11:30:00 | 616.09 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-11-29 11:15:00 | 618.80 | 2024-11-29 11:45:00 | 618.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-10 09:45:00 | 651.80 | 2024-12-10 10:00:00 | 649.63 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-12-12 09:40:00 | 638.75 | 2024-12-12 10:05:00 | 636.04 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-12-12 09:40:00 | 638.75 | 2024-12-12 10:15:00 | 638.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-13 10:55:00 | 639.25 | 2024-12-13 11:30:00 | 641.11 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-12-16 10:25:00 | 650.70 | 2024-12-16 10:35:00 | 653.21 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2024-12-16 10:25:00 | 650.70 | 2024-12-16 10:45:00 | 650.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-24 11:10:00 | 626.50 | 2024-12-24 11:40:00 | 628.66 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2024-12-24 11:10:00 | 626.50 | 2024-12-24 12:05:00 | 626.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-30 11:10:00 | 625.80 | 2024-12-30 11:25:00 | 627.68 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2024-12-30 11:10:00 | 625.80 | 2024-12-30 11:35:00 | 625.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-02 10:45:00 | 658.40 | 2025-01-02 10:50:00 | 656.55 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-01-06 10:45:00 | 635.00 | 2025-01-06 11:15:00 | 632.03 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-01-06 10:45:00 | 635.00 | 2025-01-06 15:20:00 | 629.80 | TARGET_HIT | 0.50 | 0.82% |
| SELL | retest1 | 2025-01-13 10:45:00 | 584.50 | 2025-01-13 10:55:00 | 581.49 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2025-01-13 10:45:00 | 584.50 | 2025-01-13 15:20:00 | 574.90 | TARGET_HIT | 0.50 | 1.64% |
| SELL | retest1 | 2025-01-22 11:05:00 | 535.15 | 2025-01-22 12:00:00 | 532.74 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-01-22 11:05:00 | 535.15 | 2025-01-22 14:55:00 | 534.80 | TARGET_HIT | 0.50 | 0.07% |
| SELL | retest1 | 2025-01-31 10:45:00 | 539.30 | 2025-01-31 11:00:00 | 540.74 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-02-06 10:45:00 | 568.75 | 2025-02-06 11:40:00 | 570.59 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-03-05 10:50:00 | 475.00 | 2025-03-05 11:30:00 | 477.87 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest1 | 2025-03-06 11:15:00 | 479.85 | 2025-03-06 11:40:00 | 481.55 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-03-12 10:35:00 | 488.80 | 2025-03-12 11:25:00 | 486.99 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-03-17 09:50:00 | 500.00 | 2025-03-17 10:10:00 | 497.99 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-03-19 10:05:00 | 538.30 | 2025-03-19 10:35:00 | 541.30 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-03-19 10:05:00 | 538.30 | 2025-03-19 13:45:00 | 539.25 | TARGET_HIT | 0.50 | 0.18% |
| SELL | retest1 | 2025-03-25 10:45:00 | 525.10 | 2025-03-25 11:05:00 | 521.81 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2025-03-25 10:45:00 | 525.10 | 2025-03-25 15:20:00 | 516.75 | TARGET_HIT | 0.50 | 1.59% |
| BUY | retest1 | 2025-04-02 10:45:00 | 545.50 | 2025-04-02 12:10:00 | 543.62 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-04-16 11:00:00 | 544.85 | 2025-04-16 11:10:00 | 546.25 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-04-23 10:05:00 | 548.90 | 2025-04-23 10:55:00 | 545.85 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-04-23 10:05:00 | 548.90 | 2025-04-23 11:05:00 | 548.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-24 09:45:00 | 541.85 | 2025-04-24 09:50:00 | 543.65 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-04-29 09:40:00 | 529.35 | 2025-04-29 09:55:00 | 531.46 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-04-30 10:25:00 | 533.60 | 2025-04-30 11:05:00 | 536.27 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-04-30 10:25:00 | 533.60 | 2025-04-30 11:20:00 | 533.60 | STOP_HIT | 0.50 | 0.00% |
