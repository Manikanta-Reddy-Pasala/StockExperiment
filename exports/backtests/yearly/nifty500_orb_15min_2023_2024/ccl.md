# CCL Products (I) Ltd. (CCL)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (55354 bars)
- **Last close:** 1122.00
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
| ENTRY1 | 113 |
| ENTRY2 | 0 |
| PARTIAL | 47 |
| TARGET_HIT | 31 |
| STOP_HIT | 82 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 160 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 78 / 82
- **Target hits / Stop hits / Partials:** 31 / 82 / 47
- **Avg / median % per leg:** 0.21% / 0.00%
- **Sum % (uncompounded):** 33.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 76 | 39 | 51.3% | 15 | 37 | 24 | 0.22% | 16.9% |
| BUY @ 2nd Alert (retest1) | 76 | 39 | 51.3% | 15 | 37 | 24 | 0.22% | 16.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 84 | 39 | 46.4% | 16 | 45 | 23 | 0.20% | 16.6% |
| SELL @ 2nd Alert (retest1) | 84 | 39 | 46.4% | 16 | 45 | 23 | 0.20% | 16.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 160 | 78 | 48.8% | 31 | 82 | 47 | 0.21% | 33.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-18 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-18 10:30:00 | 621.05 | 617.02 | 0.00 | ORB-long ORB[609.95,617.85] vol=1.5x ATR=2.34 |
| Stop hit — per-position SL triggered | 2023-05-18 10:35:00 | 618.71 | 617.20 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 09:45:00 | 604.05 | 606.96 | 0.00 | ORB-short ORB[606.00,614.70] vol=1.8x ATR=2.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-19 10:10:00 | 601.02 | 606.25 | 0.00 | T1 1.5R @ 601.02 |
| Stop hit — per-position SL triggered | 2023-05-19 10:35:00 | 604.05 | 604.80 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-05-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-24 09:35:00 | 610.80 | 607.75 | 0.00 | ORB-long ORB[600.00,609.00] vol=3.9x ATR=2.42 |
| Stop hit — per-position SL triggered | 2023-05-24 11:55:00 | 608.38 | 609.47 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-06-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-02 09:35:00 | 644.00 | 640.81 | 0.00 | ORB-long ORB[636.45,641.60] vol=2.5x ATR=1.84 |
| Stop hit — per-position SL triggered | 2023-06-02 09:40:00 | 642.16 | 640.98 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2023-06-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-08 09:30:00 | 636.40 | 639.32 | 0.00 | ORB-short ORB[637.75,643.35] vol=1.7x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-08 09:50:00 | 633.94 | 635.04 | 0.00 | T1 1.5R @ 633.94 |
| Target hit | 2023-06-08 10:55:00 | 634.00 | 633.93 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — BUY (started 2023-06-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-12 11:10:00 | 641.70 | 637.94 | 0.00 | ORB-long ORB[635.05,640.35] vol=8.4x ATR=1.85 |
| Stop hit — per-position SL triggered | 2023-06-12 11:35:00 | 639.85 | 638.55 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-06-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-13 09:55:00 | 635.40 | 638.06 | 0.00 | ORB-short ORB[636.80,645.00] vol=1.8x ATR=2.09 |
| Stop hit — per-position SL triggered | 2023-06-13 11:00:00 | 637.49 | 637.46 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-06-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-15 10:55:00 | 637.90 | 640.33 | 0.00 | ORB-short ORB[639.25,644.35] vol=2.1x ATR=1.34 |
| Stop hit — per-position SL triggered | 2023-06-15 11:10:00 | 639.24 | 640.27 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-06-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-16 10:00:00 | 645.25 | 642.79 | 0.00 | ORB-long ORB[639.90,645.05] vol=2.2x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-16 10:40:00 | 647.90 | 643.91 | 0.00 | T1 1.5R @ 647.90 |
| Target hit | 2023-06-16 11:15:00 | 646.55 | 647.21 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2023-06-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-22 09:50:00 | 662.30 | 656.18 | 0.00 | ORB-long ORB[651.95,657.70] vol=3.7x ATR=2.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-22 10:15:00 | 666.23 | 660.24 | 0.00 | T1 1.5R @ 666.23 |
| Stop hit — per-position SL triggered | 2023-06-22 10:25:00 | 662.30 | 661.22 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2023-06-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-23 09:40:00 | 650.55 | 653.99 | 0.00 | ORB-short ORB[651.75,659.00] vol=2.0x ATR=2.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-23 09:45:00 | 647.41 | 651.66 | 0.00 | T1 1.5R @ 647.41 |
| Stop hit — per-position SL triggered | 2023-06-23 09:50:00 | 650.55 | 651.73 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-06-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-28 10:45:00 | 650.65 | 652.23 | 0.00 | ORB-short ORB[652.60,657.45] vol=2.4x ATR=1.81 |
| Stop hit — per-position SL triggered | 2023-06-28 10:50:00 | 652.46 | 652.26 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-06-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-30 10:50:00 | 659.00 | 660.62 | 0.00 | ORB-short ORB[660.00,669.00] vol=1.6x ATR=1.73 |
| Stop hit — per-position SL triggered | 2023-06-30 10:55:00 | 660.73 | 660.62 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2023-07-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-04 09:30:00 | 663.10 | 663.60 | 0.00 | ORB-short ORB[663.45,670.00] vol=10.0x ATR=1.88 |
| Stop hit — per-position SL triggered | 2023-07-04 10:35:00 | 664.98 | 663.45 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2023-07-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 10:55:00 | 683.10 | 677.80 | 0.00 | ORB-long ORB[674.30,681.70] vol=3.1x ATR=2.24 |
| Stop hit — per-position SL triggered | 2023-07-05 11:00:00 | 680.86 | 677.87 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-07-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-10 10:45:00 | 713.80 | 709.34 | 0.00 | ORB-long ORB[705.65,712.85] vol=3.2x ATR=2.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-10 11:00:00 | 717.02 | 711.43 | 0.00 | T1 1.5R @ 717.02 |
| Target hit | 2023-07-10 15:20:00 | 723.00 | 717.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2023-07-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-13 10:05:00 | 731.25 | 726.47 | 0.00 | ORB-long ORB[722.45,727.00] vol=2.2x ATR=2.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-13 10:10:00 | 735.23 | 730.22 | 0.00 | T1 1.5R @ 735.23 |
| Stop hit — per-position SL triggered | 2023-07-13 10:15:00 | 731.25 | 730.75 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-07-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-21 10:05:00 | 619.05 | 616.02 | 0.00 | ORB-long ORB[610.20,617.50] vol=1.7x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-21 10:25:00 | 622.68 | 617.04 | 0.00 | T1 1.5R @ 622.68 |
| Target hit | 2023-07-21 15:20:00 | 627.00 | 622.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — BUY (started 2023-07-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 09:30:00 | 629.50 | 625.85 | 0.00 | ORB-long ORB[619.60,627.85] vol=2.1x ATR=2.54 |
| Stop hit — per-position SL triggered | 2023-07-25 09:45:00 | 626.96 | 626.28 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-07-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-26 10:10:00 | 613.20 | 617.19 | 0.00 | ORB-short ORB[613.70,621.70] vol=2.3x ATR=2.45 |
| Stop hit — per-position SL triggered | 2023-07-26 14:00:00 | 615.65 | 615.58 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-08-02 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-02 11:00:00 | 620.25 | 618.26 | 0.00 | ORB-long ORB[614.10,618.80] vol=3.5x ATR=1.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-02 11:55:00 | 622.28 | 618.88 | 0.00 | T1 1.5R @ 622.28 |
| Stop hit — per-position SL triggered | 2023-08-02 12:20:00 | 620.25 | 619.22 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-08-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-11 09:30:00 | 612.50 | 611.09 | 0.00 | ORB-long ORB[608.05,611.90] vol=2.3x ATR=1.74 |
| Stop hit — per-position SL triggered | 2023-08-11 10:20:00 | 610.76 | 611.69 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-08-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-16 09:35:00 | 610.25 | 606.23 | 0.00 | ORB-long ORB[600.95,607.55] vol=1.8x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-16 09:50:00 | 612.74 | 608.65 | 0.00 | T1 1.5R @ 612.74 |
| Target hit | 2023-08-16 11:25:00 | 611.10 | 611.52 | 0.00 | Trail-exit close<VWAP |

### Cycle 24 — BUY (started 2023-08-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-17 09:30:00 | 614.00 | 611.54 | 0.00 | ORB-long ORB[604.00,611.95] vol=4.8x ATR=2.27 |
| Stop hit — per-position SL triggered | 2023-08-17 11:15:00 | 611.73 | 613.59 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-08-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-18 09:30:00 | 613.25 | 615.00 | 0.00 | ORB-short ORB[613.45,619.50] vol=2.7x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-18 09:55:00 | 610.94 | 614.16 | 0.00 | T1 1.5R @ 610.94 |
| Stop hit — per-position SL triggered | 2023-08-18 10:30:00 | 613.25 | 613.39 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-08-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-23 11:00:00 | 608.00 | 609.58 | 0.00 | ORB-short ORB[609.10,614.00] vol=6.5x ATR=1.16 |
| Stop hit — per-position SL triggered | 2023-08-23 11:10:00 | 609.16 | 609.54 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-08-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-24 09:55:00 | 610.65 | 613.69 | 0.00 | ORB-short ORB[613.00,617.05] vol=1.9x ATR=1.46 |
| Stop hit — per-position SL triggered | 2023-08-24 11:00:00 | 612.11 | 612.56 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2023-08-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-30 10:00:00 | 606.00 | 609.08 | 0.00 | ORB-short ORB[608.00,617.00] vol=1.9x ATR=2.33 |
| Stop hit — per-position SL triggered | 2023-08-30 11:00:00 | 608.33 | 607.63 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2023-09-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 10:00:00 | 622.00 | 618.25 | 0.00 | ORB-long ORB[614.10,618.40] vol=2.9x ATR=1.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-05 13:55:00 | 624.67 | 622.58 | 0.00 | T1 1.5R @ 624.67 |
| Target hit | 2023-09-05 15:20:00 | 628.70 | 623.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 30 — BUY (started 2023-09-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 10:05:00 | 635.60 | 631.88 | 0.00 | ORB-long ORB[628.40,635.00] vol=1.7x ATR=2.04 |
| Stop hit — per-position SL triggered | 2023-09-06 10:40:00 | 633.56 | 632.88 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2023-09-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-08 10:30:00 | 670.40 | 665.57 | 0.00 | ORB-long ORB[657.30,666.35] vol=4.7x ATR=2.75 |
| Stop hit — per-position SL triggered | 2023-09-08 12:05:00 | 667.65 | 667.26 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-09-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-13 10:00:00 | 651.35 | 658.89 | 0.00 | ORB-short ORB[655.30,663.75] vol=1.6x ATR=4.78 |
| Stop hit — per-position SL triggered | 2023-09-13 10:35:00 | 656.13 | 657.75 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-09-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-15 09:50:00 | 673.00 | 669.72 | 0.00 | ORB-long ORB[664.05,672.00] vol=2.2x ATR=3.02 |
| Stop hit — per-position SL triggered | 2023-09-15 10:20:00 | 669.98 | 670.42 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-09-25 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-25 10:20:00 | 626.20 | 620.97 | 0.00 | ORB-long ORB[616.05,622.45] vol=1.7x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-25 10:30:00 | 629.10 | 623.65 | 0.00 | T1 1.5R @ 629.10 |
| Target hit | 2023-09-25 13:45:00 | 629.60 | 630.49 | 0.00 | Trail-exit close<VWAP |

### Cycle 35 — BUY (started 2023-09-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-26 10:10:00 | 644.10 | 642.00 | 0.00 | ORB-long ORB[637.85,644.00] vol=1.7x ATR=2.02 |
| Stop hit — per-position SL triggered | 2023-09-26 10:20:00 | 642.08 | 642.16 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-09-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 11:00:00 | 637.60 | 634.44 | 0.00 | ORB-long ORB[630.45,637.00] vol=8.9x ATR=2.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-29 11:10:00 | 640.89 | 635.15 | 0.00 | T1 1.5R @ 640.89 |
| Stop hit — per-position SL triggered | 2023-09-29 11:20:00 | 637.60 | 635.28 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-10-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-06 10:20:00 | 652.45 | 656.53 | 0.00 | ORB-short ORB[653.50,659.30] vol=1.7x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-06 10:45:00 | 649.63 | 655.33 | 0.00 | T1 1.5R @ 649.63 |
| Target hit | 2023-10-06 13:35:00 | 648.05 | 647.51 | 0.00 | Trail-exit close>VWAP |

### Cycle 38 — BUY (started 2023-10-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-09 10:05:00 | 651.20 | 646.73 | 0.00 | ORB-long ORB[637.10,646.75] vol=1.8x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-09 11:05:00 | 655.90 | 648.92 | 0.00 | T1 1.5R @ 655.90 |
| Stop hit — per-position SL triggered | 2023-10-09 12:45:00 | 651.20 | 651.40 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2023-10-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-16 11:00:00 | 646.55 | 653.48 | 0.00 | ORB-short ORB[659.10,663.00] vol=1.6x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-16 11:15:00 | 643.46 | 652.21 | 0.00 | T1 1.5R @ 643.46 |
| Stop hit — per-position SL triggered | 2023-10-16 11:20:00 | 646.55 | 652.14 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2023-10-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-25 09:30:00 | 621.30 | 623.27 | 0.00 | ORB-short ORB[622.00,626.65] vol=1.5x ATR=2.16 |
| Stop hit — per-position SL triggered | 2023-10-25 09:40:00 | 623.46 | 623.50 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-10-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-27 10:40:00 | 627.20 | 621.78 | 0.00 | ORB-long ORB[615.70,622.25] vol=1.9x ATR=1.94 |
| Stop hit — per-position SL triggered | 2023-10-27 10:45:00 | 625.26 | 622.31 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2023-10-30 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-30 10:25:00 | 620.85 | 617.50 | 0.00 | ORB-long ORB[613.25,620.75] vol=2.0x ATR=1.69 |
| Stop hit — per-position SL triggered | 2023-10-30 10:30:00 | 619.16 | 617.60 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-10-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-31 09:45:00 | 612.00 | 614.60 | 0.00 | ORB-short ORB[613.30,620.60] vol=2.8x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-31 10:30:00 | 608.21 | 611.80 | 0.00 | T1 1.5R @ 608.21 |
| Target hit | 2023-10-31 15:20:00 | 588.00 | 601.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — BUY (started 2023-11-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-01 10:55:00 | 611.05 | 605.41 | 0.00 | ORB-long ORB[593.00,601.60] vol=4.1x ATR=2.11 |
| Stop hit — per-position SL triggered | 2023-11-01 12:30:00 | 608.94 | 607.55 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-11-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 10:35:00 | 620.20 | 618.13 | 0.00 | ORB-long ORB[613.70,619.30] vol=3.1x ATR=2.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-06 11:00:00 | 624.19 | 618.68 | 0.00 | T1 1.5R @ 624.19 |
| Target hit | 2023-11-06 15:20:00 | 635.00 | 627.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — SELL (started 2023-11-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-09 10:30:00 | 625.70 | 633.02 | 0.00 | ORB-short ORB[634.55,641.45] vol=3.3x ATR=2.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-09 10:35:00 | 622.50 | 629.33 | 0.00 | T1 1.5R @ 622.50 |
| Target hit | 2023-11-09 13:00:00 | 621.65 | 621.59 | 0.00 | Trail-exit close>VWAP |

### Cycle 47 — BUY (started 2023-11-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-10 10:00:00 | 627.25 | 624.74 | 0.00 | ORB-long ORB[621.60,627.00] vol=1.9x ATR=1.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-10 11:45:00 | 629.91 | 626.91 | 0.00 | T1 1.5R @ 629.91 |
| Target hit | 2023-11-10 13:25:00 | 630.05 | 632.91 | 0.00 | Trail-exit close<VWAP |

### Cycle 48 — SELL (started 2023-11-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-13 10:35:00 | 625.50 | 629.03 | 0.00 | ORB-short ORB[627.35,631.75] vol=5.1x ATR=2.02 |
| Stop hit — per-position SL triggered | 2023-11-13 10:40:00 | 627.52 | 628.84 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2023-11-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-20 10:00:00 | 626.90 | 629.60 | 0.00 | ORB-short ORB[627.90,633.90] vol=3.4x ATR=2.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-20 10:40:00 | 623.69 | 628.09 | 0.00 | T1 1.5R @ 623.69 |
| Target hit | 2023-11-20 15:20:00 | 619.95 | 621.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — SELL (started 2023-11-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-21 09:35:00 | 621.90 | 622.75 | 0.00 | ORB-short ORB[622.30,625.45] vol=1.8x ATR=1.38 |
| Stop hit — per-position SL triggered | 2023-11-21 09:50:00 | 623.28 | 622.51 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2023-11-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 09:30:00 | 636.30 | 637.91 | 0.00 | ORB-short ORB[636.50,641.35] vol=1.7x ATR=1.31 |
| Stop hit — per-position SL triggered | 2023-11-24 10:45:00 | 637.61 | 636.92 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-11-29 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 10:00:00 | 638.90 | 635.77 | 0.00 | ORB-long ORB[630.00,635.10] vol=2.2x ATR=2.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-29 10:05:00 | 641.91 | 637.76 | 0.00 | T1 1.5R @ 641.91 |
| Target hit | 2023-11-29 15:20:00 | 652.55 | 649.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — SELL (started 2023-11-30 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-30 10:40:00 | 645.35 | 649.78 | 0.00 | ORB-short ORB[650.20,654.90] vol=2.4x ATR=1.77 |
| Stop hit — per-position SL triggered | 2023-11-30 11:30:00 | 647.12 | 648.94 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-12-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-06 10:05:00 | 665.95 | 660.75 | 0.00 | ORB-long ORB[654.90,659.95] vol=2.9x ATR=2.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-06 10:55:00 | 670.01 | 664.33 | 0.00 | T1 1.5R @ 670.01 |
| Target hit | 2023-12-06 13:55:00 | 667.85 | 668.38 | 0.00 | Trail-exit close<VWAP |

### Cycle 55 — BUY (started 2023-12-08 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-08 09:35:00 | 658.70 | 657.66 | 0.00 | ORB-long ORB[653.35,658.55] vol=3.3x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-08 09:40:00 | 661.96 | 659.51 | 0.00 | T1 1.5R @ 661.96 |
| Stop hit — per-position SL triggered | 2023-12-08 10:25:00 | 658.70 | 660.83 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2023-12-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-12 11:05:00 | 643.15 | 645.69 | 0.00 | ORB-short ORB[645.30,650.35] vol=3.6x ATR=1.25 |
| Stop hit — per-position SL triggered | 2023-12-12 12:05:00 | 644.40 | 645.43 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2023-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-13 10:15:00 | 641.90 | 643.57 | 0.00 | ORB-short ORB[642.95,648.30] vol=1.6x ATR=2.36 |
| Stop hit — per-position SL triggered | 2023-12-13 12:25:00 | 644.26 | 642.73 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2023-12-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-14 10:40:00 | 655.30 | 652.29 | 0.00 | ORB-long ORB[650.00,653.05] vol=2.2x ATR=1.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-14 10:50:00 | 657.91 | 653.10 | 0.00 | T1 1.5R @ 657.91 |
| Stop hit — per-position SL triggered | 2023-12-14 11:05:00 | 655.30 | 653.81 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2023-12-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-15 10:45:00 | 638.20 | 642.74 | 0.00 | ORB-short ORB[640.00,645.95] vol=2.3x ATR=1.77 |
| Stop hit — per-position SL triggered | 2023-12-15 11:05:00 | 639.97 | 641.03 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2023-12-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-18 10:50:00 | 637.25 | 639.18 | 0.00 | ORB-short ORB[637.80,642.80] vol=1.9x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-18 11:50:00 | 634.77 | 637.60 | 0.00 | T1 1.5R @ 634.77 |
| Target hit | 2023-12-18 15:20:00 | 628.50 | 633.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — SELL (started 2023-12-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 10:25:00 | 627.30 | 631.55 | 0.00 | ORB-short ORB[628.90,633.85] vol=2.3x ATR=2.00 |
| Stop hit — per-position SL triggered | 2023-12-19 10:35:00 | 629.30 | 630.79 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2023-12-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-21 10:45:00 | 631.35 | 627.08 | 0.00 | ORB-long ORB[621.85,631.00] vol=7.1x ATR=2.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-21 10:50:00 | 634.78 | 627.26 | 0.00 | T1 1.5R @ 634.78 |
| Target hit | 2023-12-21 15:20:00 | 634.00 | 632.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 63 — BUY (started 2023-12-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 11:05:00 | 640.35 | 636.74 | 0.00 | ORB-long ORB[630.75,639.05] vol=2.5x ATR=1.88 |
| Stop hit — per-position SL triggered | 2023-12-22 11:15:00 | 638.47 | 637.02 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2023-12-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-26 11:10:00 | 635.30 | 630.73 | 0.00 | ORB-long ORB[625.50,632.35] vol=1.8x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-26 11:20:00 | 637.64 | 633.57 | 0.00 | T1 1.5R @ 637.64 |
| Target hit | 2023-12-26 15:20:00 | 640.75 | 636.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 65 — SELL (started 2023-12-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-27 11:05:00 | 636.25 | 636.51 | 0.00 | ORB-short ORB[637.80,643.30] vol=12.7x ATR=1.34 |
| Stop hit — per-position SL triggered | 2023-12-27 11:10:00 | 637.59 | 636.52 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2023-12-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-29 10:40:00 | 641.05 | 636.11 | 0.00 | ORB-long ORB[624.15,633.50] vol=8.1x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-29 10:55:00 | 644.76 | 638.34 | 0.00 | T1 1.5R @ 644.76 |
| Target hit | 2023-12-29 15:10:00 | 643.00 | 643.17 | 0.00 | Trail-exit close<VWAP |

### Cycle 67 — SELL (started 2024-01-02 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 10:25:00 | 629.80 | 631.71 | 0.00 | ORB-short ORB[632.20,635.95] vol=3.8x ATR=1.36 |
| Stop hit — per-position SL triggered | 2024-01-02 11:15:00 | 631.16 | 631.05 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2024-01-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-03 11:00:00 | 626.70 | 628.23 | 0.00 | ORB-short ORB[628.95,632.95] vol=1.7x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-03 11:45:00 | 624.72 | 627.18 | 0.00 | T1 1.5R @ 624.72 |
| Target hit | 2024-01-03 15:20:00 | 624.20 | 624.90 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — BUY (started 2024-01-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 09:55:00 | 635.60 | 634.12 | 0.00 | ORB-long ORB[630.60,634.65] vol=2.0x ATR=1.54 |
| Stop hit — per-position SL triggered | 2024-01-05 10:10:00 | 634.06 | 634.23 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2024-01-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-08 09:30:00 | 628.15 | 631.10 | 0.00 | ORB-short ORB[629.30,635.20] vol=1.5x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-08 10:55:00 | 625.40 | 628.36 | 0.00 | T1 1.5R @ 625.40 |
| Stop hit — per-position SL triggered | 2024-01-08 12:55:00 | 628.15 | 625.59 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2024-01-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 11:10:00 | 633.55 | 630.95 | 0.00 | ORB-long ORB[630.30,632.35] vol=6.0x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-09 12:20:00 | 635.86 | 632.46 | 0.00 | T1 1.5R @ 635.86 |
| Stop hit — per-position SL triggered | 2024-01-09 12:50:00 | 633.55 | 633.28 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2024-01-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 10:55:00 | 642.25 | 640.31 | 0.00 | ORB-long ORB[636.60,641.10] vol=1.5x ATR=1.52 |
| Stop hit — per-position SL triggered | 2024-01-11 13:30:00 | 640.73 | 641.36 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2024-01-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-12 11:00:00 | 636.75 | 638.14 | 0.00 | ORB-short ORB[638.00,641.15] vol=1.9x ATR=1.09 |
| Stop hit — per-position SL triggered | 2024-01-12 11:20:00 | 637.84 | 637.83 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2024-01-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-15 10:00:00 | 641.35 | 643.67 | 0.00 | ORB-short ORB[642.85,648.00] vol=1.6x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-15 10:50:00 | 638.97 | 642.17 | 0.00 | T1 1.5R @ 638.97 |
| Stop hit — per-position SL triggered | 2024-01-15 14:35:00 | 641.35 | 638.80 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2024-01-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-18 09:45:00 | 623.40 | 626.18 | 0.00 | ORB-short ORB[625.60,630.00] vol=2.8x ATR=1.49 |
| Stop hit — per-position SL triggered | 2024-01-18 09:50:00 | 624.89 | 626.14 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2024-01-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 09:30:00 | 634.00 | 631.72 | 0.00 | ORB-long ORB[629.35,632.85] vol=2.3x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-19 10:35:00 | 636.82 | 634.24 | 0.00 | T1 1.5R @ 636.82 |
| Stop hit — per-position SL triggered | 2024-01-19 12:15:00 | 634.00 | 635.25 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2024-01-20 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-20 10:20:00 | 629.85 | 631.27 | 0.00 | ORB-short ORB[631.25,636.60] vol=3.0x ATR=1.45 |
| Target hit | 2024-01-20 15:20:00 | 629.30 | 629.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 78 — BUY (started 2024-01-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-23 09:55:00 | 634.15 | 631.65 | 0.00 | ORB-long ORB[628.75,634.00] vol=2.4x ATR=1.61 |
| Stop hit — per-position SL triggered | 2024-01-23 10:00:00 | 632.54 | 631.95 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2024-01-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 10:35:00 | 656.05 | 651.35 | 0.00 | ORB-long ORB[644.10,652.95] vol=6.0x ATR=2.37 |
| Stop hit — per-position SL triggered | 2024-01-29 10:40:00 | 653.68 | 651.47 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2024-01-31 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-31 10:25:00 | 670.30 | 671.92 | 0.00 | ORB-short ORB[671.00,678.90] vol=4.7x ATR=2.18 |
| Stop hit — per-position SL triggered | 2024-01-31 11:35:00 | 672.48 | 671.26 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2024-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-02 10:15:00 | 670.00 | 672.58 | 0.00 | ORB-short ORB[670.20,677.30] vol=2.6x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 10:20:00 | 667.70 | 672.11 | 0.00 | T1 1.5R @ 667.70 |
| Target hit | 2024-02-02 15:20:00 | 644.55 | 656.37 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 82 — BUY (started 2024-02-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-07 11:05:00 | 654.80 | 651.42 | 0.00 | ORB-long ORB[647.65,652.80] vol=2.5x ATR=2.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-07 12:00:00 | 658.04 | 652.28 | 0.00 | T1 1.5R @ 658.04 |
| Target hit | 2024-02-07 15:20:00 | 670.00 | 659.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 83 — SELL (started 2024-02-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-09 09:40:00 | 653.60 | 656.22 | 0.00 | ORB-short ORB[657.00,665.25] vol=3.7x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-09 09:45:00 | 649.90 | 654.96 | 0.00 | T1 1.5R @ 649.90 |
| Target hit | 2024-02-09 12:55:00 | 650.05 | 648.99 | 0.00 | Trail-exit close>VWAP |

### Cycle 84 — SELL (started 2024-02-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-13 10:40:00 | 640.90 | 642.28 | 0.00 | ORB-short ORB[642.00,648.80] vol=4.0x ATR=2.11 |
| Stop hit — per-position SL triggered | 2024-02-13 10:55:00 | 643.01 | 642.33 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2024-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-16 10:15:00 | 654.00 | 656.14 | 0.00 | ORB-short ORB[654.20,660.95] vol=1.6x ATR=2.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-16 10:45:00 | 650.76 | 655.92 | 0.00 | T1 1.5R @ 650.76 |
| Target hit | 2024-02-16 15:20:00 | 646.10 | 648.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 86 — BUY (started 2024-02-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 09:55:00 | 658.00 | 653.15 | 0.00 | ORB-long ORB[646.05,653.15] vol=2.9x ATR=2.68 |
| Stop hit — per-position SL triggered | 2024-02-21 10:10:00 | 655.32 | 654.48 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2024-02-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-22 09:30:00 | 638.00 | 641.92 | 0.00 | ORB-short ORB[640.50,648.80] vol=2.9x ATR=2.78 |
| Stop hit — per-position SL triggered | 2024-02-22 15:20:00 | 638.25 | 639.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 88 — SELL (started 2024-02-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-23 09:30:00 | 635.00 | 637.10 | 0.00 | ORB-short ORB[636.55,641.20] vol=5.5x ATR=1.68 |
| Stop hit — per-position SL triggered | 2024-02-23 11:25:00 | 636.68 | 635.74 | 0.00 | SL hit |

### Cycle 89 — BUY (started 2024-02-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-26 11:00:00 | 642.40 | 639.77 | 0.00 | ORB-long ORB[638.00,642.35] vol=4.7x ATR=1.35 |
| Stop hit — per-position SL triggered | 2024-02-26 11:05:00 | 641.05 | 639.79 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2024-02-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-28 09:50:00 | 658.65 | 653.38 | 0.00 | ORB-long ORB[647.70,655.45] vol=2.8x ATR=2.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-28 09:55:00 | 662.38 | 658.56 | 0.00 | T1 1.5R @ 662.38 |
| Target hit | 2024-02-28 10:25:00 | 662.35 | 662.96 | 0.00 | Trail-exit close<VWAP |

### Cycle 91 — SELL (started 2024-02-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-29 10:10:00 | 629.50 | 634.45 | 0.00 | ORB-short ORB[633.50,642.20] vol=1.7x ATR=2.75 |
| Target hit | 2024-02-29 15:20:00 | 629.10 | 630.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 92 — SELL (started 2024-03-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-04 09:45:00 | 627.10 | 629.25 | 0.00 | ORB-short ORB[627.75,632.50] vol=1.6x ATR=1.84 |
| Target hit | 2024-03-04 15:20:00 | 626.05 | 626.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 93 — SELL (started 2024-03-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-11 09:35:00 | 608.00 | 610.54 | 0.00 | ORB-short ORB[613.30,616.20] vol=2.7x ATR=2.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 11:10:00 | 604.19 | 607.05 | 0.00 | T1 1.5R @ 604.19 |
| Stop hit — per-position SL triggered | 2024-03-11 13:50:00 | 608.00 | 605.62 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2024-03-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-12 09:30:00 | 600.80 | 602.38 | 0.00 | ORB-short ORB[601.15,607.55] vol=2.5x ATR=2.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 09:50:00 | 597.64 | 600.58 | 0.00 | T1 1.5R @ 597.64 |
| Target hit | 2024-03-12 09:50:00 | 600.75 | 600.58 | 0.00 | Trail-exit close>VWAP |

### Cycle 95 — SELL (started 2024-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-13 11:15:00 | 600.95 | 602.60 | 0.00 | ORB-short ORB[603.95,609.50] vol=1.6x ATR=1.69 |
| Stop hit — per-position SL triggered | 2024-03-13 12:00:00 | 602.64 | 602.30 | 0.00 | SL hit |

### Cycle 96 — SELL (started 2024-03-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-15 09:30:00 | 577.25 | 581.79 | 0.00 | ORB-short ORB[581.65,589.45] vol=4.9x ATR=2.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-15 11:00:00 | 573.90 | 578.35 | 0.00 | T1 1.5R @ 573.90 |
| Stop hit — per-position SL triggered | 2024-03-15 13:00:00 | 577.25 | 576.87 | 0.00 | SL hit |

### Cycle 97 — BUY (started 2024-03-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-18 10:10:00 | 586.55 | 585.69 | 0.00 | ORB-long ORB[578.30,584.85] vol=4.0x ATR=3.23 |
| Stop hit — per-position SL triggered | 2024-03-18 10:25:00 | 583.32 | 585.90 | 0.00 | SL hit |

### Cycle 98 — SELL (started 2024-03-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 11:00:00 | 579.00 | 579.06 | 0.00 | ORB-short ORB[580.10,585.00] vol=7.9x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-19 11:30:00 | 576.75 | 578.58 | 0.00 | T1 1.5R @ 576.75 |
| Target hit | 2024-03-19 15:20:00 | 570.00 | 569.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 99 — BUY (started 2024-03-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-20 09:55:00 | 568.80 | 566.92 | 0.00 | ORB-long ORB[564.40,568.70] vol=2.2x ATR=1.72 |
| Stop hit — per-position SL triggered | 2024-03-20 10:05:00 | 567.08 | 567.20 | 0.00 | SL hit |

### Cycle 100 — SELL (started 2024-03-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-21 09:50:00 | 573.00 | 575.45 | 0.00 | ORB-short ORB[574.95,579.00] vol=2.7x ATR=2.25 |
| Stop hit — per-position SL triggered | 2024-03-21 10:00:00 | 575.25 | 575.41 | 0.00 | SL hit |

### Cycle 101 — SELL (started 2024-03-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-26 10:05:00 | 591.55 | 596.20 | 0.00 | ORB-short ORB[595.35,601.00] vol=1.6x ATR=2.55 |
| Stop hit — per-position SL triggered | 2024-03-26 10:10:00 | 594.10 | 596.07 | 0.00 | SL hit |

### Cycle 102 — SELL (started 2024-03-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-28 09:30:00 | 586.85 | 588.25 | 0.00 | ORB-short ORB[588.00,591.30] vol=1.9x ATR=1.67 |
| Stop hit — per-position SL triggered | 2024-03-28 09:45:00 | 588.52 | 587.88 | 0.00 | SL hit |

### Cycle 103 — BUY (started 2024-04-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-05 09:30:00 | 596.20 | 594.55 | 0.00 | ORB-long ORB[593.00,595.00] vol=2.1x ATR=1.52 |
| Stop hit — per-position SL triggered | 2024-04-05 09:35:00 | 594.68 | 594.59 | 0.00 | SL hit |

### Cycle 104 — SELL (started 2024-04-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-12 11:00:00 | 584.85 | 588.08 | 0.00 | ORB-short ORB[586.80,592.00] vol=4.4x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-12 11:05:00 | 582.49 | 587.33 | 0.00 | T1 1.5R @ 582.49 |
| Stop hit — per-position SL triggered | 2024-04-12 11:10:00 | 584.85 | 587.27 | 0.00 | SL hit |

### Cycle 105 — BUY (started 2024-04-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 10:25:00 | 579.05 | 577.18 | 0.00 | ORB-long ORB[574.00,578.60] vol=2.0x ATR=1.78 |
| Stop hit — per-position SL triggered | 2024-04-16 10:30:00 | 577.27 | 577.20 | 0.00 | SL hit |

### Cycle 106 — BUY (started 2024-04-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-18 09:55:00 | 578.80 | 575.83 | 0.00 | ORB-long ORB[573.05,578.40] vol=1.9x ATR=1.79 |
| Stop hit — per-position SL triggered | 2024-04-18 10:25:00 | 577.01 | 576.17 | 0.00 | SL hit |

### Cycle 107 — BUY (started 2024-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-22 09:35:00 | 577.50 | 574.75 | 0.00 | ORB-long ORB[570.75,575.75] vol=2.5x ATR=1.90 |
| Stop hit — per-position SL triggered | 2024-04-22 09:40:00 | 575.60 | 574.78 | 0.00 | SL hit |

### Cycle 108 — SELL (started 2024-04-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-25 10:35:00 | 578.10 | 582.09 | 0.00 | ORB-short ORB[581.35,584.20] vol=2.3x ATR=1.51 |
| Stop hit — per-position SL triggered | 2024-04-25 11:25:00 | 579.61 | 580.33 | 0.00 | SL hit |

### Cycle 109 — SELL (started 2024-04-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-26 10:00:00 | 580.40 | 583.00 | 0.00 | ORB-short ORB[582.90,585.75] vol=2.0x ATR=1.88 |
| Stop hit — per-position SL triggered | 2024-04-26 10:10:00 | 582.28 | 581.97 | 0.00 | SL hit |

### Cycle 110 — SELL (started 2024-05-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-02 10:45:00 | 583.65 | 588.11 | 0.00 | ORB-short ORB[587.05,594.50] vol=1.9x ATR=1.67 |
| Stop hit — per-position SL triggered | 2024-05-02 11:00:00 | 585.32 | 587.70 | 0.00 | SL hit |

### Cycle 111 — SELL (started 2024-05-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-03 10:45:00 | 578.80 | 582.13 | 0.00 | ORB-short ORB[580.30,586.75] vol=3.8x ATR=1.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-03 11:20:00 | 576.06 | 581.12 | 0.00 | T1 1.5R @ 576.06 |
| Target hit | 2024-05-03 14:30:00 | 577.35 | 577.27 | 0.00 | Trail-exit close>VWAP |

### Cycle 112 — SELL (started 2024-05-06 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-06 11:10:00 | 578.55 | 579.73 | 0.00 | ORB-short ORB[582.00,586.95] vol=1.6x ATR=1.79 |
| Stop hit — per-position SL triggered | 2024-05-06 11:15:00 | 580.34 | 579.73 | 0.00 | SL hit |

### Cycle 113 — SELL (started 2024-05-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 10:05:00 | 576.90 | 578.46 | 0.00 | ORB-short ORB[577.00,582.50] vol=1.6x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 13:00:00 | 574.67 | 577.08 | 0.00 | T1 1.5R @ 574.67 |
| Stop hit — per-position SL triggered | 2024-05-09 15:10:00 | 576.90 | 575.71 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-18 10:30:00 | 621.05 | 2023-05-18 10:35:00 | 618.71 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2023-05-19 09:45:00 | 604.05 | 2023-05-19 10:10:00 | 601.02 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2023-05-19 09:45:00 | 604.05 | 2023-05-19 10:35:00 | 604.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-24 09:35:00 | 610.80 | 2023-05-24 11:55:00 | 608.38 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2023-06-02 09:35:00 | 644.00 | 2023-06-02 09:40:00 | 642.16 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-06-08 09:30:00 | 636.40 | 2023-06-08 09:50:00 | 633.94 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-06-08 09:30:00 | 636.40 | 2023-06-08 10:55:00 | 634.00 | TARGET_HIT | 0.50 | 0.38% |
| BUY | retest1 | 2023-06-12 11:10:00 | 641.70 | 2023-06-12 11:35:00 | 639.85 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-06-13 09:55:00 | 635.40 | 2023-06-13 11:00:00 | 637.49 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-06-15 10:55:00 | 637.90 | 2023-06-15 11:10:00 | 639.24 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-06-16 10:00:00 | 645.25 | 2023-06-16 10:40:00 | 647.90 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-06-16 10:00:00 | 645.25 | 2023-06-16 11:15:00 | 646.55 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2023-06-22 09:50:00 | 662.30 | 2023-06-22 10:15:00 | 666.23 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2023-06-22 09:50:00 | 662.30 | 2023-06-22 10:25:00 | 662.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-23 09:40:00 | 650.55 | 2023-06-23 09:45:00 | 647.41 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2023-06-23 09:40:00 | 650.55 | 2023-06-23 09:50:00 | 650.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-28 10:45:00 | 650.65 | 2023-06-28 10:50:00 | 652.46 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-06-30 10:50:00 | 659.00 | 2023-06-30 10:55:00 | 660.73 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-07-04 09:30:00 | 663.10 | 2023-07-04 10:35:00 | 664.98 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-07-05 10:55:00 | 683.10 | 2023-07-05 11:00:00 | 680.86 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-07-10 10:45:00 | 713.80 | 2023-07-10 11:00:00 | 717.02 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2023-07-10 10:45:00 | 713.80 | 2023-07-10 15:20:00 | 723.00 | TARGET_HIT | 0.50 | 1.29% |
| BUY | retest1 | 2023-07-13 10:05:00 | 731.25 | 2023-07-13 10:10:00 | 735.23 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2023-07-13 10:05:00 | 731.25 | 2023-07-13 10:15:00 | 731.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-21 10:05:00 | 619.05 | 2023-07-21 10:25:00 | 622.68 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2023-07-21 10:05:00 | 619.05 | 2023-07-21 15:20:00 | 627.00 | TARGET_HIT | 0.50 | 1.28% |
| BUY | retest1 | 2023-07-25 09:30:00 | 629.50 | 2023-07-25 09:45:00 | 626.96 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2023-07-26 10:10:00 | 613.20 | 2023-07-26 14:00:00 | 615.65 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2023-08-02 11:00:00 | 620.25 | 2023-08-02 11:55:00 | 622.28 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2023-08-02 11:00:00 | 620.25 | 2023-08-02 12:20:00 | 620.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-11 09:30:00 | 612.50 | 2023-08-11 10:20:00 | 610.76 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-08-16 09:35:00 | 610.25 | 2023-08-16 09:50:00 | 612.74 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2023-08-16 09:35:00 | 610.25 | 2023-08-16 11:25:00 | 611.10 | TARGET_HIT | 0.50 | 0.14% |
| BUY | retest1 | 2023-08-17 09:30:00 | 614.00 | 2023-08-17 11:15:00 | 611.73 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2023-08-18 09:30:00 | 613.25 | 2023-08-18 09:55:00 | 610.94 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2023-08-18 09:30:00 | 613.25 | 2023-08-18 10:30:00 | 613.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-23 11:00:00 | 608.00 | 2023-08-23 11:10:00 | 609.16 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-08-24 09:55:00 | 610.65 | 2023-08-24 11:00:00 | 612.11 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-08-30 10:00:00 | 606.00 | 2023-08-30 11:00:00 | 608.33 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2023-09-05 10:00:00 | 622.00 | 2023-09-05 13:55:00 | 624.67 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2023-09-05 10:00:00 | 622.00 | 2023-09-05 15:20:00 | 628.70 | TARGET_HIT | 0.50 | 1.08% |
| BUY | retest1 | 2023-09-06 10:05:00 | 635.60 | 2023-09-06 10:40:00 | 633.56 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-09-08 10:30:00 | 670.40 | 2023-09-08 12:05:00 | 667.65 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2023-09-13 10:00:00 | 651.35 | 2023-09-13 10:35:00 | 656.13 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest1 | 2023-09-15 09:50:00 | 673.00 | 2023-09-15 10:20:00 | 669.98 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2023-09-25 10:20:00 | 626.20 | 2023-09-25 10:30:00 | 629.10 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2023-09-25 10:20:00 | 626.20 | 2023-09-25 13:45:00 | 629.60 | TARGET_HIT | 0.50 | 0.54% |
| BUY | retest1 | 2023-09-26 10:10:00 | 644.10 | 2023-09-26 10:20:00 | 642.08 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-09-29 11:00:00 | 637.60 | 2023-09-29 11:10:00 | 640.89 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-09-29 11:00:00 | 637.60 | 2023-09-29 11:20:00 | 637.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-06 10:20:00 | 652.45 | 2023-10-06 10:45:00 | 649.63 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2023-10-06 10:20:00 | 652.45 | 2023-10-06 13:35:00 | 648.05 | TARGET_HIT | 0.50 | 0.67% |
| BUY | retest1 | 2023-10-09 10:05:00 | 651.20 | 2023-10-09 11:05:00 | 655.90 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2023-10-09 10:05:00 | 651.20 | 2023-10-09 12:45:00 | 651.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-16 11:00:00 | 646.55 | 2023-10-16 11:15:00 | 643.46 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2023-10-16 11:00:00 | 646.55 | 2023-10-16 11:20:00 | 646.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-25 09:30:00 | 621.30 | 2023-10-25 09:40:00 | 623.46 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-10-27 10:40:00 | 627.20 | 2023-10-27 10:45:00 | 625.26 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-10-30 10:25:00 | 620.85 | 2023-10-30 10:30:00 | 619.16 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-10-31 09:45:00 | 612.00 | 2023-10-31 10:30:00 | 608.21 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2023-10-31 09:45:00 | 612.00 | 2023-10-31 15:20:00 | 588.00 | TARGET_HIT | 0.50 | 3.92% |
| BUY | retest1 | 2023-11-01 10:55:00 | 611.05 | 2023-11-01 12:30:00 | 608.94 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-11-06 10:35:00 | 620.20 | 2023-11-06 11:00:00 | 624.19 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2023-11-06 10:35:00 | 620.20 | 2023-11-06 15:20:00 | 635.00 | TARGET_HIT | 0.50 | 2.39% |
| SELL | retest1 | 2023-11-09 10:30:00 | 625.70 | 2023-11-09 10:35:00 | 622.50 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2023-11-09 10:30:00 | 625.70 | 2023-11-09 13:00:00 | 621.65 | TARGET_HIT | 0.50 | 0.65% |
| BUY | retest1 | 2023-11-10 10:00:00 | 627.25 | 2023-11-10 11:45:00 | 629.91 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2023-11-10 10:00:00 | 627.25 | 2023-11-10 13:25:00 | 630.05 | TARGET_HIT | 0.50 | 0.45% |
| SELL | retest1 | 2023-11-13 10:35:00 | 625.50 | 2023-11-13 10:40:00 | 627.52 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-11-20 10:00:00 | 626.90 | 2023-11-20 10:40:00 | 623.69 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2023-11-20 10:00:00 | 626.90 | 2023-11-20 15:20:00 | 619.95 | TARGET_HIT | 0.50 | 1.11% |
| SELL | retest1 | 2023-11-21 09:35:00 | 621.90 | 2023-11-21 09:50:00 | 623.28 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-11-24 09:30:00 | 636.30 | 2023-11-24 10:45:00 | 637.61 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-11-29 10:00:00 | 638.90 | 2023-11-29 10:05:00 | 641.91 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2023-11-29 10:00:00 | 638.90 | 2023-11-29 15:20:00 | 652.55 | TARGET_HIT | 0.50 | 2.14% |
| SELL | retest1 | 2023-11-30 10:40:00 | 645.35 | 2023-11-30 11:30:00 | 647.12 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2023-12-06 10:05:00 | 665.95 | 2023-12-06 10:55:00 | 670.01 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2023-12-06 10:05:00 | 665.95 | 2023-12-06 13:55:00 | 667.85 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2023-12-08 09:35:00 | 658.70 | 2023-12-08 09:40:00 | 661.96 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2023-12-08 09:35:00 | 658.70 | 2023-12-08 10:25:00 | 658.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-12 11:05:00 | 643.15 | 2023-12-12 12:05:00 | 644.40 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-12-13 10:15:00 | 641.90 | 2023-12-13 12:25:00 | 644.26 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-12-14 10:40:00 | 655.30 | 2023-12-14 10:50:00 | 657.91 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-12-14 10:40:00 | 655.30 | 2023-12-14 11:05:00 | 655.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-15 10:45:00 | 638.20 | 2023-12-15 11:05:00 | 639.97 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2023-12-18 10:50:00 | 637.25 | 2023-12-18 11:50:00 | 634.77 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-12-18 10:50:00 | 637.25 | 2023-12-18 15:20:00 | 628.50 | TARGET_HIT | 0.50 | 1.37% |
| SELL | retest1 | 2023-12-19 10:25:00 | 627.30 | 2023-12-19 10:35:00 | 629.30 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-12-21 10:45:00 | 631.35 | 2023-12-21 10:50:00 | 634.78 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2023-12-21 10:45:00 | 631.35 | 2023-12-21 15:20:00 | 634.00 | TARGET_HIT | 0.50 | 0.42% |
| BUY | retest1 | 2023-12-22 11:05:00 | 640.35 | 2023-12-22 11:15:00 | 638.47 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-12-26 11:10:00 | 635.30 | 2023-12-26 11:20:00 | 637.64 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2023-12-26 11:10:00 | 635.30 | 2023-12-26 15:20:00 | 640.75 | TARGET_HIT | 0.50 | 0.86% |
| SELL | retest1 | 2023-12-27 11:05:00 | 636.25 | 2023-12-27 11:10:00 | 637.59 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-12-29 10:40:00 | 641.05 | 2023-12-29 10:55:00 | 644.76 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2023-12-29 10:40:00 | 641.05 | 2023-12-29 15:10:00 | 643.00 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2024-01-02 10:25:00 | 629.80 | 2024-01-02 11:15:00 | 631.16 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-01-03 11:00:00 | 626.70 | 2024-01-03 11:45:00 | 624.72 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-01-03 11:00:00 | 626.70 | 2024-01-03 15:20:00 | 624.20 | TARGET_HIT | 0.50 | 0.40% |
| BUY | retest1 | 2024-01-05 09:55:00 | 635.60 | 2024-01-05 10:10:00 | 634.06 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-01-08 09:30:00 | 628.15 | 2024-01-08 10:55:00 | 625.40 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-01-08 09:30:00 | 628.15 | 2024-01-08 12:55:00 | 628.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-09 11:10:00 | 633.55 | 2024-01-09 12:20:00 | 635.86 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-01-09 11:10:00 | 633.55 | 2024-01-09 12:50:00 | 633.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-11 10:55:00 | 642.25 | 2024-01-11 13:30:00 | 640.73 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-01-12 11:00:00 | 636.75 | 2024-01-12 11:20:00 | 637.84 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-01-15 10:00:00 | 641.35 | 2024-01-15 10:50:00 | 638.97 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-01-15 10:00:00 | 641.35 | 2024-01-15 14:35:00 | 641.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-18 09:45:00 | 623.40 | 2024-01-18 09:50:00 | 624.89 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-01-19 09:30:00 | 634.00 | 2024-01-19 10:35:00 | 636.82 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-01-19 09:30:00 | 634.00 | 2024-01-19 12:15:00 | 634.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-20 10:20:00 | 629.85 | 2024-01-20 15:20:00 | 629.30 | TARGET_HIT | 1.00 | 0.09% |
| BUY | retest1 | 2024-01-23 09:55:00 | 634.15 | 2024-01-23 10:00:00 | 632.54 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-01-29 10:35:00 | 656.05 | 2024-01-29 10:40:00 | 653.68 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-01-31 10:25:00 | 670.30 | 2024-01-31 11:35:00 | 672.48 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-02-02 10:15:00 | 670.00 | 2024-02-02 10:20:00 | 667.70 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-02-02 10:15:00 | 670.00 | 2024-02-02 15:20:00 | 644.55 | TARGET_HIT | 0.50 | 3.80% |
| BUY | retest1 | 2024-02-07 11:05:00 | 654.80 | 2024-02-07 12:00:00 | 658.04 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-02-07 11:05:00 | 654.80 | 2024-02-07 15:20:00 | 670.00 | TARGET_HIT | 0.50 | 2.32% |
| SELL | retest1 | 2024-02-09 09:40:00 | 653.60 | 2024-02-09 09:45:00 | 649.90 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-02-09 09:40:00 | 653.60 | 2024-02-09 12:55:00 | 650.05 | TARGET_HIT | 0.50 | 0.54% |
| SELL | retest1 | 2024-02-13 10:40:00 | 640.90 | 2024-02-13 10:55:00 | 643.01 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-02-16 10:15:00 | 654.00 | 2024-02-16 10:45:00 | 650.76 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-02-16 10:15:00 | 654.00 | 2024-02-16 15:20:00 | 646.10 | TARGET_HIT | 0.50 | 1.21% |
| BUY | retest1 | 2024-02-21 09:55:00 | 658.00 | 2024-02-21 10:10:00 | 655.32 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-02-22 09:30:00 | 638.00 | 2024-02-22 15:20:00 | 638.25 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest1 | 2024-02-23 09:30:00 | 635.00 | 2024-02-23 11:25:00 | 636.68 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-02-26 11:00:00 | 642.40 | 2024-02-26 11:05:00 | 641.05 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-02-28 09:50:00 | 658.65 | 2024-02-28 09:55:00 | 662.38 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-02-28 09:50:00 | 658.65 | 2024-02-28 10:25:00 | 662.35 | TARGET_HIT | 0.50 | 0.56% |
| SELL | retest1 | 2024-02-29 10:10:00 | 629.50 | 2024-02-29 15:20:00 | 629.10 | TARGET_HIT | 1.00 | 0.06% |
| SELL | retest1 | 2024-03-04 09:45:00 | 627.10 | 2024-03-04 15:20:00 | 626.05 | TARGET_HIT | 1.00 | 0.17% |
| SELL | retest1 | 2024-03-11 09:35:00 | 608.00 | 2024-03-11 11:10:00 | 604.19 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-03-11 09:35:00 | 608.00 | 2024-03-11 13:50:00 | 608.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-12 09:30:00 | 600.80 | 2024-03-12 09:50:00 | 597.64 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-03-12 09:30:00 | 600.80 | 2024-03-12 09:50:00 | 600.75 | TARGET_HIT | 0.50 | 0.01% |
| SELL | retest1 | 2024-03-13 11:15:00 | 600.95 | 2024-03-13 12:00:00 | 602.64 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-03-15 09:30:00 | 577.25 | 2024-03-15 11:00:00 | 573.90 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-03-15 09:30:00 | 577.25 | 2024-03-15 13:00:00 | 577.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-18 10:10:00 | 586.55 | 2024-03-18 10:25:00 | 583.32 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2024-03-19 11:00:00 | 579.00 | 2024-03-19 11:30:00 | 576.75 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-03-19 11:00:00 | 579.00 | 2024-03-19 15:20:00 | 570.00 | TARGET_HIT | 0.50 | 1.55% |
| BUY | retest1 | 2024-03-20 09:55:00 | 568.80 | 2024-03-20 10:05:00 | 567.08 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-03-21 09:50:00 | 573.00 | 2024-03-21 10:00:00 | 575.25 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-03-26 10:05:00 | 591.55 | 2024-03-26 10:10:00 | 594.10 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-03-28 09:30:00 | 586.85 | 2024-03-28 09:45:00 | 588.52 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-04-05 09:30:00 | 596.20 | 2024-04-05 09:35:00 | 594.68 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-04-12 11:00:00 | 584.85 | 2024-04-12 11:05:00 | 582.49 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-04-12 11:00:00 | 584.85 | 2024-04-12 11:10:00 | 584.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-16 10:25:00 | 579.05 | 2024-04-16 10:30:00 | 577.27 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-04-18 09:55:00 | 578.80 | 2024-04-18 10:25:00 | 577.01 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-04-22 09:35:00 | 577.50 | 2024-04-22 09:40:00 | 575.60 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-04-25 10:35:00 | 578.10 | 2024-04-25 11:25:00 | 579.61 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-04-26 10:00:00 | 580.40 | 2024-04-26 10:10:00 | 582.28 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-05-02 10:45:00 | 583.65 | 2024-05-02 11:00:00 | 585.32 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-05-03 10:45:00 | 578.80 | 2024-05-03 11:20:00 | 576.06 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-05-03 10:45:00 | 578.80 | 2024-05-03 14:30:00 | 577.35 | TARGET_HIT | 0.50 | 0.25% |
| SELL | retest1 | 2024-05-06 11:10:00 | 578.55 | 2024-05-06 11:15:00 | 580.34 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-05-09 10:05:00 | 576.90 | 2024-05-09 13:00:00 | 574.67 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-05-09 10:05:00 | 576.90 | 2024-05-09 15:10:00 | 576.90 | STOP_HIT | 0.50 | 0.00% |
