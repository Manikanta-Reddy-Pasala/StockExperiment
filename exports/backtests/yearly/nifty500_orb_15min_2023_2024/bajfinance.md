# Bajaj Finance Ltd. (BAJFINANCE)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (52280 bars)
- **Last close:** 954.50
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
| ENTRY1 | 95 |
| ENTRY2 | 0 |
| PARTIAL | 37 |
| TARGET_HIT | 10 |
| STOP_HIT | 85 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 132 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 47 / 85
- **Target hits / Stop hits / Partials:** 10 / 85 / 37
- **Avg / median % per leg:** 0.06% / 0.00%
- **Sum % (uncompounded):** 7.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 59 | 17 | 28.8% | 4 | 42 | 13 | -0.04% | -2.6% |
| BUY @ 2nd Alert (retest1) | 59 | 17 | 28.8% | 4 | 42 | 13 | -0.04% | -2.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 73 | 30 | 41.1% | 6 | 43 | 24 | 0.14% | 9.9% |
| SELL @ 2nd Alert (retest1) | 73 | 30 | 41.1% | 6 | 43 | 24 | 0.14% | 9.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 132 | 47 | 35.6% | 10 | 85 | 37 | 0.06% | 7.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-12 10:25:00 | 670.57 | 665.50 | 0.00 | ORB-long ORB[663.31,665.88] vol=3.8x ATR=2.03 |
| Stop hit — per-position SL triggered | 2023-05-12 10:55:00 | 668.54 | 666.68 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-16 09:30:00 | 678.98 | 675.76 | 0.00 | ORB-long ORB[670.00,677.60] vol=4.5x ATR=1.74 |
| Stop hit — per-position SL triggered | 2023-05-16 09:35:00 | 677.24 | 676.36 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-17 11:00:00 | 672.50 | 676.37 | 0.00 | ORB-short ORB[675.11,681.05] vol=2.2x ATR=1.40 |
| Stop hit — per-position SL triggered | 2023-05-17 11:10:00 | 673.90 | 675.93 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-05-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-18 09:30:00 | 682.00 | 678.22 | 0.00 | ORB-long ORB[674.52,680.30] vol=1.7x ATR=1.61 |
| Stop hit — per-position SL triggered | 2023-05-18 09:35:00 | 680.39 | 678.60 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-24 11:00:00 | 683.50 | 680.39 | 0.00 | ORB-long ORB[672.98,679.24] vol=1.9x ATR=1.36 |
| Stop hit — per-position SL triggered | 2023-05-24 11:40:00 | 682.14 | 680.99 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2023-05-31 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-31 11:10:00 | 697.81 | 700.27 | 0.00 | ORB-short ORB[698.67,702.29] vol=2.3x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-31 11:25:00 | 696.35 | 699.98 | 0.00 | T1 1.5R @ 696.35 |
| Stop hit — per-position SL triggered | 2023-05-31 14:25:00 | 697.81 | 697.53 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-06-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-01 11:10:00 | 702.60 | 699.18 | 0.00 | ORB-long ORB[695.16,702.46] vol=2.4x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-01 11:15:00 | 704.36 | 699.56 | 0.00 | T1 1.5R @ 704.36 |
| Stop hit — per-position SL triggered | 2023-06-01 11:25:00 | 702.60 | 699.70 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-06-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-05 09:50:00 | 703.62 | 705.56 | 0.00 | ORB-short ORB[705.12,709.16] vol=1.9x ATR=1.82 |
| Stop hit — per-position SL triggered | 2023-06-05 09:55:00 | 705.44 | 705.53 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-06-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-06 10:35:00 | 708.50 | 705.92 | 0.00 | ORB-long ORB[702.50,707.50] vol=2.5x ATR=1.23 |
| Stop hit — per-position SL triggered | 2023-06-06 10:55:00 | 707.27 | 706.30 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-06-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-12 09:30:00 | 710.08 | 707.77 | 0.00 | ORB-long ORB[704.16,708.98] vol=2.7x ATR=1.57 |
| Stop hit — per-position SL triggered | 2023-06-12 10:40:00 | 708.51 | 708.93 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-13 10:15:00 | 714.60 | 713.39 | 0.00 | ORB-long ORB[710.00,714.02] vol=1.6x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-13 10:30:00 | 716.04 | 713.91 | 0.00 | T1 1.5R @ 716.04 |
| Target hit | 2023-06-13 13:50:00 | 715.88 | 716.19 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — SELL (started 2023-06-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-14 11:00:00 | 711.59 | 712.27 | 0.00 | ORB-short ORB[712.50,718.20] vol=1.5x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-14 11:30:00 | 709.91 | 712.04 | 0.00 | T1 1.5R @ 709.91 |
| Stop hit — per-position SL triggered | 2023-06-14 14:00:00 | 711.59 | 711.33 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2023-06-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-16 10:35:00 | 718.18 | 716.73 | 0.00 | ORB-long ORB[713.10,717.00] vol=1.9x ATR=1.19 |
| Stop hit — per-position SL triggered | 2023-06-16 10:50:00 | 716.99 | 717.09 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2023-06-21 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-21 11:05:00 | 718.82 | 723.58 | 0.00 | ORB-short ORB[723.30,728.59] vol=1.6x ATR=1.34 |
| Stop hit — per-position SL triggered | 2023-06-21 11:25:00 | 720.16 | 723.04 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-06-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-22 09:40:00 | 714.80 | 716.91 | 0.00 | ORB-short ORB[715.60,720.42] vol=1.7x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-22 09:45:00 | 713.07 | 716.13 | 0.00 | T1 1.5R @ 713.07 |
| Target hit | 2023-06-22 15:20:00 | 702.71 | 708.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2023-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-26 10:15:00 | 694.50 | 699.46 | 0.00 | ORB-short ORB[695.00,702.70] vol=1.6x ATR=1.67 |
| Stop hit — per-position SL triggered | 2023-06-26 10:20:00 | 696.17 | 699.18 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-06-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-27 10:30:00 | 696.49 | 699.19 | 0.00 | ORB-short ORB[698.80,703.00] vol=2.1x ATR=1.30 |
| Stop hit — per-position SL triggered | 2023-06-27 12:45:00 | 697.79 | 698.32 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-06-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 09:30:00 | 710.12 | 707.26 | 0.00 | ORB-long ORB[704.00,707.50] vol=2.6x ATR=1.44 |
| Stop hit — per-position SL triggered | 2023-06-28 09:40:00 | 708.68 | 708.11 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2023-06-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-30 09:30:00 | 718.38 | 715.43 | 0.00 | ORB-long ORB[709.12,718.10] vol=1.7x ATR=1.64 |
| Stop hit — per-position SL triggered | 2023-06-30 09:45:00 | 716.74 | 715.99 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-07-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-03 09:35:00 | 722.62 | 720.87 | 0.00 | ORB-long ORB[716.00,722.00] vol=2.7x ATR=1.65 |
| Stop hit — per-position SL triggered | 2023-07-03 09:40:00 | 720.97 | 720.79 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-07-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-07 10:00:00 | 768.59 | 770.25 | 0.00 | ORB-short ORB[769.22,774.89] vol=3.8x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-07 10:45:00 | 765.82 | 769.70 | 0.00 | T1 1.5R @ 765.82 |
| Target hit | 2023-07-07 15:20:00 | 761.95 | 766.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — SELL (started 2023-07-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-17 11:10:00 | 745.41 | 749.24 | 0.00 | ORB-short ORB[747.13,754.20] vol=1.5x ATR=1.64 |
| Stop hit — per-position SL triggered | 2023-07-17 11:30:00 | 747.05 | 748.85 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-07-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-19 10:40:00 | 750.76 | 748.17 | 0.00 | ORB-long ORB[743.00,748.50] vol=2.2x ATR=1.89 |
| Stop hit — per-position SL triggered | 2023-07-19 11:15:00 | 748.87 | 748.48 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-07-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-24 10:35:00 | 764.21 | 761.27 | 0.00 | ORB-long ORB[756.04,763.50] vol=1.8x ATR=1.75 |
| Stop hit — per-position SL triggered | 2023-07-24 10:45:00 | 762.46 | 761.42 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-07-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-25 10:40:00 | 756.12 | 759.82 | 0.00 | ORB-short ORB[760.20,764.94] vol=1.5x ATR=1.73 |
| Stop hit — per-position SL triggered | 2023-07-25 10:55:00 | 757.85 | 759.44 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-07-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 10:30:00 | 766.83 | 763.39 | 0.00 | ORB-long ORB[758.00,763.50] vol=3.1x ATR=1.68 |
| Stop hit — per-position SL triggered | 2023-07-26 11:30:00 | 765.15 | 764.51 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-07-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-27 11:15:00 | 743.45 | 748.95 | 0.00 | ORB-short ORB[747.13,757.79] vol=2.1x ATR=2.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-27 13:50:00 | 740.03 | 746.10 | 0.00 | T1 1.5R @ 740.03 |
| Target hit | 2023-07-27 15:20:00 | 728.30 | 740.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 28 — SELL (started 2023-08-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-01 10:50:00 | 727.50 | 730.11 | 0.00 | ORB-short ORB[729.03,733.00] vol=1.8x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-01 12:20:00 | 725.57 | 729.07 | 0.00 | T1 1.5R @ 725.57 |
| Stop hit — per-position SL triggered | 2023-08-01 12:50:00 | 727.50 | 728.41 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-09-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-11 10:55:00 | 739.03 | 741.80 | 0.00 | ORB-short ORB[741.00,744.40] vol=1.8x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-11 13:25:00 | 736.62 | 740.26 | 0.00 | T1 1.5R @ 736.62 |
| Stop hit — per-position SL triggered | 2023-09-11 14:45:00 | 739.03 | 739.80 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-09-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 09:55:00 | 733.98 | 737.68 | 0.00 | ORB-short ORB[738.03,742.50] vol=2.1x ATR=1.96 |
| Stop hit — per-position SL triggered | 2023-09-12 10:55:00 | 735.94 | 736.69 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2023-09-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-14 09:45:00 | 745.57 | 747.57 | 0.00 | ORB-short ORB[746.82,751.17] vol=1.6x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-14 10:00:00 | 743.34 | 746.78 | 0.00 | T1 1.5R @ 743.34 |
| Stop hit — per-position SL triggered | 2023-09-14 10:25:00 | 745.57 | 746.14 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-09-15 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-15 10:40:00 | 752.40 | 748.86 | 0.00 | ORB-long ORB[747.10,751.60] vol=2.0x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-15 11:00:00 | 754.45 | 749.71 | 0.00 | T1 1.5R @ 754.45 |
| Stop hit — per-position SL triggered | 2023-09-15 11:10:00 | 752.40 | 749.82 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-09-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-20 09:40:00 | 760.29 | 756.83 | 0.00 | ORB-long ORB[748.30,758.80] vol=2.1x ATR=1.87 |
| Stop hit — per-position SL triggered | 2023-09-20 09:45:00 | 758.42 | 757.04 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-26 11:15:00 | 785.65 | 781.62 | 0.00 | ORB-long ORB[777.50,784.60] vol=3.0x ATR=1.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-26 11:30:00 | 788.17 | 782.42 | 0.00 | T1 1.5R @ 788.17 |
| Stop hit — per-position SL triggered | 2023-09-26 12:35:00 | 785.65 | 783.82 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2023-09-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-27 09:30:00 | 778.16 | 781.16 | 0.00 | ORB-short ORB[780.33,785.50] vol=1.6x ATR=1.86 |
| Stop hit — per-position SL triggered | 2023-09-27 09:35:00 | 780.02 | 780.96 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2023-09-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-29 10:10:00 | 783.60 | 778.04 | 0.00 | ORB-long ORB[772.50,778.65] vol=2.1x ATR=2.03 |
| Stop hit — per-position SL triggered | 2023-09-29 10:25:00 | 781.57 | 779.08 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-10-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-03 09:30:00 | 784.37 | 781.48 | 0.00 | ORB-long ORB[775.50,783.74] vol=2.2x ATR=2.05 |
| Stop hit — per-position SL triggered | 2023-10-03 09:35:00 | 782.32 | 781.59 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2023-10-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 09:30:00 | 797.08 | 792.80 | 0.00 | ORB-long ORB[784.50,796.22] vol=2.5x ATR=2.09 |
| Stop hit — per-position SL triggered | 2023-10-06 09:45:00 | 794.99 | 794.11 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2023-10-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-12 11:10:00 | 805.00 | 805.90 | 0.00 | ORB-short ORB[805.15,812.70] vol=1.8x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-12 11:45:00 | 803.31 | 805.51 | 0.00 | T1 1.5R @ 803.31 |
| Stop hit — per-position SL triggered | 2023-10-12 12:10:00 | 805.00 | 805.27 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2023-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-16 11:15:00 | 800.00 | 803.26 | 0.00 | ORB-short ORB[801.02,804.50] vol=1.6x ATR=1.62 |
| Stop hit — per-position SL triggered | 2023-10-16 11:30:00 | 801.62 | 803.09 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2023-10-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-26 10:55:00 | 748.51 | 752.67 | 0.00 | ORB-short ORB[755.65,765.60] vol=3.4x ATR=2.32 |
| Stop hit — per-position SL triggered | 2023-10-26 11:35:00 | 750.83 | 752.19 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2023-10-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-30 09:30:00 | 740.75 | 743.28 | 0.00 | ORB-short ORB[741.33,750.63] vol=1.9x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-30 09:40:00 | 737.57 | 741.46 | 0.00 | T1 1.5R @ 737.57 |
| Stop hit — per-position SL triggered | 2023-10-30 10:05:00 | 740.75 | 740.48 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2023-11-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-01 11:10:00 | 753.43 | 751.73 | 0.00 | ORB-long ORB[747.61,752.00] vol=1.6x ATR=1.49 |
| Stop hit — per-position SL triggered | 2023-11-01 12:10:00 | 751.94 | 751.99 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2023-11-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-03 10:45:00 | 744.60 | 748.34 | 0.00 | ORB-short ORB[749.00,753.80] vol=1.9x ATR=1.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-03 12:00:00 | 741.73 | 746.73 | 0.00 | T1 1.5R @ 741.73 |
| Stop hit — per-position SL triggered | 2023-11-03 13:40:00 | 744.60 | 743.31 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2023-11-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-07 10:55:00 | 752.00 | 758.14 | 0.00 | ORB-short ORB[756.10,764.00] vol=3.2x ATR=2.03 |
| Stop hit — per-position SL triggered | 2023-11-07 11:05:00 | 754.03 | 757.78 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2023-11-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-09 10:35:00 | 740.45 | 743.29 | 0.00 | ORB-short ORB[741.22,748.00] vol=2.7x ATR=1.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-09 12:45:00 | 738.25 | 741.86 | 0.00 | T1 1.5R @ 738.25 |
| Target hit | 2023-11-09 15:20:00 | 739.19 | 740.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 47 — SELL (started 2023-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-13 10:15:00 | 733.52 | 738.00 | 0.00 | ORB-short ORB[737.38,745.00] vol=4.2x ATR=1.85 |
| Stop hit — per-position SL triggered | 2023-11-13 10:40:00 | 735.37 | 737.12 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-11-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-20 11:05:00 | 709.85 | 716.98 | 0.00 | ORB-short ORB[715.10,723.90] vol=3.1x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-20 11:20:00 | 707.38 | 715.74 | 0.00 | T1 1.5R @ 707.38 |
| Stop hit — per-position SL triggered | 2023-11-20 11:45:00 | 709.85 | 714.98 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-11-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 11:10:00 | 712.17 | 707.28 | 0.00 | ORB-long ORB[703.11,711.86] vol=2.4x ATR=1.58 |
| Stop hit — per-position SL triggered | 2023-11-21 11:45:00 | 710.59 | 707.72 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2023-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-22 11:15:00 | 716.20 | 711.66 | 0.00 | ORB-long ORB[706.55,711.90] vol=9.5x ATR=1.40 |
| Stop hit — per-position SL triggered | 2023-11-22 11:20:00 | 714.80 | 711.87 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2023-11-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-23 09:50:00 | 712.28 | 714.18 | 0.00 | ORB-short ORB[713.50,716.00] vol=2.0x ATR=1.28 |
| Stop hit — per-position SL triggered | 2023-11-23 09:55:00 | 713.56 | 714.06 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2023-11-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-24 10:00:00 | 706.83 | 708.06 | 0.00 | ORB-short ORB[707.00,711.68] vol=1.9x ATR=1.31 |
| Stop hit — per-position SL triggered | 2023-11-24 10:05:00 | 708.14 | 708.05 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-12-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-06 09:30:00 | 740.50 | 738.77 | 0.00 | ORB-long ORB[736.51,739.10] vol=1.7x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-06 09:40:00 | 742.74 | 740.38 | 0.00 | T1 1.5R @ 742.74 |
| Target hit | 2023-12-06 10:35:00 | 741.47 | 741.89 | 0.00 | Trail-exit close<VWAP |

### Cycle 54 — BUY (started 2023-12-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-12 09:30:00 | 733.24 | 731.44 | 0.00 | ORB-long ORB[728.00,732.90] vol=1.7x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-12 10:05:00 | 735.07 | 732.33 | 0.00 | T1 1.5R @ 735.07 |
| Stop hit — per-position SL triggered | 2023-12-12 10:10:00 | 733.24 | 732.39 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2023-12-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-15 09:30:00 | 746.10 | 748.92 | 0.00 | ORB-short ORB[747.83,752.89] vol=2.0x ATR=1.61 |
| Stop hit — per-position SL triggered | 2023-12-15 09:55:00 | 747.71 | 748.03 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2023-12-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-20 10:40:00 | 768.50 | 765.27 | 0.00 | ORB-long ORB[763.90,767.00] vol=2.5x ATR=1.43 |
| Stop hit — per-position SL triggered | 2023-12-20 10:50:00 | 767.07 | 765.51 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2023-12-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 09:30:00 | 731.50 | 727.33 | 0.00 | ORB-long ORB[720.13,730.50] vol=1.7x ATR=2.14 |
| Stop hit — per-position SL triggered | 2023-12-27 09:35:00 | 729.36 | 727.66 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2023-12-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-29 10:55:00 | 727.60 | 724.36 | 0.00 | ORB-long ORB[720.52,724.30] vol=2.2x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-29 11:50:00 | 729.88 | 725.70 | 0.00 | T1 1.5R @ 729.88 |
| Target hit | 2023-12-29 15:20:00 | 733.20 | 728.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 59 — SELL (started 2024-01-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-01 10:55:00 | 728.75 | 730.33 | 0.00 | ORB-short ORB[729.00,733.70] vol=2.5x ATR=1.12 |
| Stop hit — per-position SL triggered | 2024-01-01 12:00:00 | 729.87 | 729.82 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2024-01-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-02 09:40:00 | 733.56 | 730.91 | 0.00 | ORB-long ORB[728.01,732.50] vol=1.7x ATR=1.45 |
| Stop hit — per-position SL triggered | 2024-01-02 09:55:00 | 732.11 | 731.66 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2024-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-08 10:15:00 | 774.62 | 767.10 | 0.00 | ORB-long ORB[763.13,768.00] vol=1.8x ATR=2.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-08 10:30:00 | 778.16 | 770.24 | 0.00 | T1 1.5R @ 778.16 |
| Stop hit — per-position SL triggered | 2024-01-08 11:45:00 | 774.62 | 772.96 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-01-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-12 09:45:00 | 764.32 | 766.61 | 0.00 | ORB-short ORB[766.41,771.90] vol=6.5x ATR=1.85 |
| Stop hit — per-position SL triggered | 2024-01-12 10:10:00 | 766.17 | 766.34 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2024-01-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-15 11:10:00 | 757.58 | 759.29 | 0.00 | ORB-short ORB[760.16,769.80] vol=1.6x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-15 12:10:00 | 755.37 | 758.31 | 0.00 | T1 1.5R @ 755.37 |
| Target hit | 2024-01-15 15:20:00 | 747.27 | 752.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — SELL (started 2024-01-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-19 10:55:00 | 730.58 | 734.51 | 0.00 | ORB-short ORB[731.65,736.20] vol=1.5x ATR=1.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-19 12:10:00 | 728.09 | 732.99 | 0.00 | T1 1.5R @ 728.09 |
| Stop hit — per-position SL triggered | 2024-01-19 14:25:00 | 730.58 | 731.38 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2024-01-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-31 11:05:00 | 690.50 | 687.25 | 0.00 | ORB-long ORB[681.20,687.20] vol=4.2x ATR=1.42 |
| Stop hit — per-position SL triggered | 2024-01-31 11:15:00 | 689.08 | 687.38 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-02-01 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-01 10:05:00 | 678.30 | 681.93 | 0.00 | ORB-short ORB[681.80,688.00] vol=1.7x ATR=1.55 |
| Stop hit — per-position SL triggered | 2024-02-01 10:40:00 | 679.85 | 681.11 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2024-02-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 11:10:00 | 685.04 | 681.39 | 0.00 | ORB-long ORB[675.30,680.94] vol=1.6x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 11:15:00 | 687.41 | 682.11 | 0.00 | T1 1.5R @ 687.41 |
| Stop hit — per-position SL triggered | 2024-02-02 12:25:00 | 685.04 | 684.59 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2024-02-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-06 10:40:00 | 654.45 | 657.90 | 0.00 | ORB-short ORB[656.32,664.80] vol=1.9x ATR=1.95 |
| Stop hit — per-position SL triggered | 2024-02-06 10:50:00 | 656.40 | 657.64 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2024-02-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-07 09:35:00 | 668.46 | 666.16 | 0.00 | ORB-long ORB[662.53,666.61] vol=2.3x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-07 09:40:00 | 671.29 | 667.09 | 0.00 | T1 1.5R @ 671.29 |
| Stop hit — per-position SL triggered | 2024-02-07 09:50:00 | 668.46 | 667.44 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2024-02-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 10:10:00 | 666.97 | 668.23 | 0.00 | ORB-short ORB[667.31,673.75] vol=1.6x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-08 10:15:00 | 664.20 | 667.68 | 0.00 | T1 1.5R @ 664.20 |
| Stop hit — per-position SL triggered | 2024-02-08 10:30:00 | 666.97 | 667.25 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2024-02-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-12 10:50:00 | 658.73 | 659.00 | 0.00 | ORB-short ORB[659.16,667.19] vol=2.1x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-12 11:30:00 | 656.48 | 658.89 | 0.00 | T1 1.5R @ 656.48 |
| Stop hit — per-position SL triggered | 2024-02-12 12:15:00 | 658.73 | 658.69 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2024-02-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-15 10:25:00 | 658.80 | 662.68 | 0.00 | ORB-short ORB[662.56,667.74] vol=1.7x ATR=1.65 |
| Stop hit — per-position SL triggered | 2024-02-15 11:10:00 | 660.45 | 661.67 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2024-02-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-16 10:55:00 | 662.65 | 663.41 | 0.00 | ORB-short ORB[663.11,666.00] vol=1.6x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-16 11:00:00 | 661.24 | 663.18 | 0.00 | T1 1.5R @ 661.24 |
| Stop hit — per-position SL triggered | 2024-02-16 11:05:00 | 662.65 | 663.17 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2024-02-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-19 09:30:00 | 671.69 | 668.69 | 0.00 | ORB-long ORB[662.53,670.50] vol=2.3x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-19 10:00:00 | 674.03 | 670.57 | 0.00 | T1 1.5R @ 674.03 |
| Stop hit — per-position SL triggered | 2024-02-19 10:05:00 | 671.69 | 670.66 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2024-02-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-20 10:10:00 | 674.50 | 672.57 | 0.00 | ORB-long ORB[668.13,673.43] vol=1.8x ATR=1.88 |
| Stop hit — per-position SL triggered | 2024-02-20 10:20:00 | 672.62 | 672.63 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2024-02-21 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-21 10:25:00 | 673.67 | 674.89 | 0.00 | ORB-short ORB[675.00,679.00] vol=2.2x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-21 11:20:00 | 671.53 | 674.42 | 0.00 | T1 1.5R @ 671.53 |
| Stop hit — per-position SL triggered | 2024-02-21 13:15:00 | 673.67 | 673.93 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2024-02-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-22 10:55:00 | 657.63 | 661.79 | 0.00 | ORB-short ORB[661.25,670.50] vol=1.9x ATR=1.63 |
| Stop hit — per-position SL triggered | 2024-02-22 11:10:00 | 659.26 | 661.46 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2024-03-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-12 09:30:00 | 644.38 | 641.04 | 0.00 | ORB-long ORB[637.68,643.52] vol=1.6x ATR=2.09 |
| Stop hit — per-position SL triggered | 2024-03-12 10:40:00 | 642.29 | 643.25 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2024-03-13 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-13 10:10:00 | 650.80 | 646.16 | 0.00 | ORB-long ORB[640.74,645.78] vol=2.2x ATR=2.07 |
| Stop hit — per-position SL triggered | 2024-03-13 10:25:00 | 648.73 | 646.99 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2024-03-15 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-15 10:05:00 | 654.68 | 645.81 | 0.00 | ORB-long ORB[636.00,642.20] vol=2.8x ATR=2.96 |
| Stop hit — per-position SL triggered | 2024-03-15 10:35:00 | 651.72 | 648.90 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2024-03-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-18 10:50:00 | 653.94 | 649.34 | 0.00 | ORB-long ORB[646.61,652.38] vol=2.4x ATR=2.13 |
| Stop hit — per-position SL triggered | 2024-03-18 11:15:00 | 651.81 | 649.90 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2024-03-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-19 10:10:00 | 657.20 | 653.92 | 0.00 | ORB-long ORB[650.50,655.20] vol=2.0x ATR=1.57 |
| Stop hit — per-position SL triggered | 2024-03-19 10:15:00 | 655.63 | 654.07 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2024-03-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-20 10:30:00 | 666.30 | 664.90 | 0.00 | ORB-long ORB[660.60,666.06] vol=1.6x ATR=2.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-20 12:15:00 | 669.53 | 666.03 | 0.00 | T1 1.5R @ 669.53 |
| Stop hit — per-position SL triggered | 2024-03-20 13:55:00 | 666.30 | 667.79 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2024-03-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-22 11:00:00 | 676.10 | 674.84 | 0.00 | ORB-long ORB[668.63,675.49] vol=2.2x ATR=1.45 |
| Stop hit — per-position SL triggered | 2024-03-22 11:45:00 | 674.65 | 674.92 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2024-04-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 10:45:00 | 732.30 | 728.67 | 0.00 | ORB-long ORB[722.01,732.27] vol=2.1x ATR=2.28 |
| Stop hit — per-position SL triggered | 2024-04-01 12:10:00 | 730.02 | 729.54 | 0.00 | SL hit |

### Cycle 86 — SELL (started 2024-04-02 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-02 10:10:00 | 715.26 | 718.98 | 0.00 | ORB-short ORB[718.40,725.50] vol=1.9x ATR=2.01 |
| Stop hit — per-position SL triggered | 2024-04-02 10:35:00 | 717.27 | 717.93 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2024-04-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 10:50:00 | 726.11 | 730.67 | 0.00 | ORB-short ORB[730.50,735.00] vol=1.6x ATR=2.16 |
| Stop hit — per-position SL triggered | 2024-04-04 11:00:00 | 728.27 | 730.51 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2024-04-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-05 10:35:00 | 717.50 | 721.62 | 0.00 | ORB-short ORB[717.65,725.90] vol=2.0x ATR=2.03 |
| Stop hit — per-position SL triggered | 2024-04-05 10:55:00 | 719.53 | 721.41 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2024-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-08 11:15:00 | 716.50 | 718.49 | 0.00 | ORB-short ORB[718.18,722.70] vol=1.6x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-08 11:30:00 | 715.03 | 718.23 | 0.00 | T1 1.5R @ 715.03 |
| Stop hit — per-position SL triggered | 2024-04-08 12:10:00 | 716.50 | 717.77 | 0.00 | SL hit |

### Cycle 90 — BUY (started 2024-04-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-15 10:55:00 | 718.84 | 713.62 | 0.00 | ORB-long ORB[710.22,718.18] vol=1.9x ATR=1.88 |
| Stop hit — per-position SL triggered | 2024-04-15 11:40:00 | 716.96 | 714.23 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2024-04-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-16 11:05:00 | 695.50 | 697.08 | 0.00 | ORB-short ORB[696.28,703.43] vol=4.0x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-16 12:30:00 | 693.23 | 696.72 | 0.00 | T1 1.5R @ 693.23 |
| Stop hit — per-position SL triggered | 2024-04-16 13:30:00 | 695.50 | 696.58 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2024-04-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 10:35:00 | 731.89 | 729.98 | 0.00 | ORB-long ORB[725.46,729.90] vol=2.1x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-24 11:10:00 | 734.20 | 731.93 | 0.00 | T1 1.5R @ 734.20 |
| Target hit | 2024-04-24 11:50:00 | 733.10 | 734.12 | 0.00 | Trail-exit close<VWAP |

### Cycle 93 — SELL (started 2024-04-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-25 10:05:00 | 720.41 | 728.56 | 0.00 | ORB-short ORB[726.28,734.27] vol=2.4x ATR=2.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-25 10:10:00 | 716.69 | 725.25 | 0.00 | T1 1.5R @ 716.69 |
| Stop hit — per-position SL triggered | 2024-04-25 10:15:00 | 720.41 | 724.84 | 0.00 | SL hit |

### Cycle 94 — SELL (started 2024-05-07 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 09:50:00 | 681.93 | 686.36 | 0.00 | ORB-short ORB[685.20,690.30] vol=1.6x ATR=1.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 10:15:00 | 679.07 | 684.90 | 0.00 | T1 1.5R @ 679.07 |
| Stop hit — per-position SL triggered | 2024-05-07 11:00:00 | 681.93 | 683.94 | 0.00 | SL hit |

### Cycle 95 — SELL (started 2024-05-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-09 11:00:00 | 671.90 | 677.53 | 0.00 | ORB-short ORB[677.70,681.74] vol=1.7x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 11:30:00 | 669.65 | 676.50 | 0.00 | T1 1.5R @ 669.65 |
| Target hit | 2024-05-09 15:20:00 | 658.91 | 669.11 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-12 10:25:00 | 670.57 | 2023-05-12 10:55:00 | 668.54 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-05-16 09:30:00 | 678.98 | 2023-05-16 09:35:00 | 677.24 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-05-17 11:00:00 | 672.50 | 2023-05-17 11:10:00 | 673.90 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-05-18 09:30:00 | 682.00 | 2023-05-18 09:35:00 | 680.39 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-05-24 11:00:00 | 683.50 | 2023-05-24 11:40:00 | 682.14 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-05-31 11:10:00 | 697.81 | 2023-05-31 11:25:00 | 696.35 | PARTIAL | 0.50 | 0.21% |
| SELL | retest1 | 2023-05-31 11:10:00 | 697.81 | 2023-05-31 14:25:00 | 697.81 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-01 11:10:00 | 702.60 | 2023-06-01 11:15:00 | 704.36 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2023-06-01 11:10:00 | 702.60 | 2023-06-01 11:25:00 | 702.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-05 09:50:00 | 703.62 | 2023-06-05 09:55:00 | 705.44 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-06-06 10:35:00 | 708.50 | 2023-06-06 10:55:00 | 707.27 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2023-06-12 09:30:00 | 710.08 | 2023-06-12 10:40:00 | 708.51 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-06-13 10:15:00 | 714.60 | 2023-06-13 10:30:00 | 716.04 | PARTIAL | 0.50 | 0.20% |
| BUY | retest1 | 2023-06-13 10:15:00 | 714.60 | 2023-06-13 13:50:00 | 715.88 | TARGET_HIT | 0.50 | 0.18% |
| SELL | retest1 | 2023-06-14 11:00:00 | 711.59 | 2023-06-14 11:30:00 | 709.91 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2023-06-14 11:00:00 | 711.59 | 2023-06-14 14:00:00 | 711.59 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-16 10:35:00 | 718.18 | 2023-06-16 10:50:00 | 716.99 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-06-21 11:05:00 | 718.82 | 2023-06-21 11:25:00 | 720.16 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-06-22 09:40:00 | 714.80 | 2023-06-22 09:45:00 | 713.07 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2023-06-22 09:40:00 | 714.80 | 2023-06-22 15:20:00 | 702.71 | TARGET_HIT | 0.50 | 1.69% |
| SELL | retest1 | 2023-06-26 10:15:00 | 694.50 | 2023-06-26 10:20:00 | 696.17 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-06-27 10:30:00 | 696.49 | 2023-06-27 12:45:00 | 697.79 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-06-28 09:30:00 | 710.12 | 2023-06-28 09:40:00 | 708.68 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-06-30 09:30:00 | 718.38 | 2023-06-30 09:45:00 | 716.74 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-07-03 09:35:00 | 722.62 | 2023-07-03 09:40:00 | 720.97 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-07-07 10:00:00 | 768.59 | 2023-07-07 10:45:00 | 765.82 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2023-07-07 10:00:00 | 768.59 | 2023-07-07 15:20:00 | 761.95 | TARGET_HIT | 0.50 | 0.86% |
| SELL | retest1 | 2023-07-17 11:10:00 | 745.41 | 2023-07-17 11:30:00 | 747.05 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-07-19 10:40:00 | 750.76 | 2023-07-19 11:15:00 | 748.87 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-07-24 10:35:00 | 764.21 | 2023-07-24 10:45:00 | 762.46 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-07-25 10:40:00 | 756.12 | 2023-07-25 10:55:00 | 757.85 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-07-26 10:30:00 | 766.83 | 2023-07-26 11:30:00 | 765.15 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-07-27 11:15:00 | 743.45 | 2023-07-27 13:50:00 | 740.03 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2023-07-27 11:15:00 | 743.45 | 2023-07-27 15:20:00 | 728.30 | TARGET_HIT | 0.50 | 2.04% |
| SELL | retest1 | 2023-08-01 10:50:00 | 727.50 | 2023-08-01 12:20:00 | 725.57 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-08-01 10:50:00 | 727.50 | 2023-08-01 12:50:00 | 727.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-11 10:55:00 | 739.03 | 2023-09-11 13:25:00 | 736.62 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2023-09-11 10:55:00 | 739.03 | 2023-09-11 14:45:00 | 739.03 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-12 09:55:00 | 733.98 | 2023-09-12 10:55:00 | 735.94 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-09-14 09:45:00 | 745.57 | 2023-09-14 10:00:00 | 743.34 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2023-09-14 09:45:00 | 745.57 | 2023-09-14 10:25:00 | 745.57 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-15 10:40:00 | 752.40 | 2023-09-15 11:00:00 | 754.45 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2023-09-15 10:40:00 | 752.40 | 2023-09-15 11:10:00 | 752.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-20 09:40:00 | 760.29 | 2023-09-20 09:45:00 | 758.42 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-09-26 11:15:00 | 785.65 | 2023-09-26 11:30:00 | 788.17 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2023-09-26 11:15:00 | 785.65 | 2023-09-26 12:35:00 | 785.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-27 09:30:00 | 778.16 | 2023-09-27 09:35:00 | 780.02 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-09-29 10:10:00 | 783.60 | 2023-09-29 10:25:00 | 781.57 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-10-03 09:30:00 | 784.37 | 2023-10-03 09:35:00 | 782.32 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-10-06 09:30:00 | 797.08 | 2023-10-06 09:45:00 | 794.99 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-10-12 11:10:00 | 805.00 | 2023-10-12 11:45:00 | 803.31 | PARTIAL | 0.50 | 0.21% |
| SELL | retest1 | 2023-10-12 11:10:00 | 805.00 | 2023-10-12 12:10:00 | 805.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-16 11:15:00 | 800.00 | 2023-10-16 11:30:00 | 801.62 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-10-26 10:55:00 | 748.51 | 2023-10-26 11:35:00 | 750.83 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-10-30 09:30:00 | 740.75 | 2023-10-30 09:40:00 | 737.57 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2023-10-30 09:30:00 | 740.75 | 2023-10-30 10:05:00 | 740.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-01 11:10:00 | 753.43 | 2023-11-01 12:10:00 | 751.94 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-11-03 10:45:00 | 744.60 | 2023-11-03 12:00:00 | 741.73 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-11-03 10:45:00 | 744.60 | 2023-11-03 13:40:00 | 744.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-07 10:55:00 | 752.00 | 2023-11-07 11:05:00 | 754.03 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-11-09 10:35:00 | 740.45 | 2023-11-09 12:45:00 | 738.25 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2023-11-09 10:35:00 | 740.45 | 2023-11-09 15:20:00 | 739.19 | TARGET_HIT | 0.50 | 0.17% |
| SELL | retest1 | 2023-11-13 10:15:00 | 733.52 | 2023-11-13 10:40:00 | 735.37 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-11-20 11:05:00 | 709.85 | 2023-11-20 11:20:00 | 707.38 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2023-11-20 11:05:00 | 709.85 | 2023-11-20 11:45:00 | 709.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-21 11:10:00 | 712.17 | 2023-11-21 11:45:00 | 710.59 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-11-22 11:15:00 | 716.20 | 2023-11-22 11:20:00 | 714.80 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-11-23 09:50:00 | 712.28 | 2023-11-23 09:55:00 | 713.56 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-11-24 10:00:00 | 706.83 | 2023-11-24 10:05:00 | 708.14 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-12-06 09:30:00 | 740.50 | 2023-12-06 09:40:00 | 742.74 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2023-12-06 09:30:00 | 740.50 | 2023-12-06 10:35:00 | 741.47 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2023-12-12 09:30:00 | 733.24 | 2023-12-12 10:05:00 | 735.07 | PARTIAL | 0.50 | 0.25% |
| BUY | retest1 | 2023-12-12 09:30:00 | 733.24 | 2023-12-12 10:10:00 | 733.24 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-15 09:30:00 | 746.10 | 2023-12-15 09:55:00 | 747.71 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-12-20 10:40:00 | 768.50 | 2023-12-20 10:50:00 | 767.07 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-12-27 09:30:00 | 731.50 | 2023-12-27 09:35:00 | 729.36 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-12-29 10:55:00 | 727.60 | 2023-12-29 11:50:00 | 729.88 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-12-29 10:55:00 | 727.60 | 2023-12-29 15:20:00 | 733.20 | TARGET_HIT | 0.50 | 0.77% |
| SELL | retest1 | 2024-01-01 10:55:00 | 728.75 | 2024-01-01 12:00:00 | 729.87 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2024-01-02 09:40:00 | 733.56 | 2024-01-02 09:55:00 | 732.11 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-01-08 10:15:00 | 774.62 | 2024-01-08 10:30:00 | 778.16 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-01-08 10:15:00 | 774.62 | 2024-01-08 11:45:00 | 774.62 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-12 09:45:00 | 764.32 | 2024-01-12 10:10:00 | 766.17 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-01-15 11:10:00 | 757.58 | 2024-01-15 12:10:00 | 755.37 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-01-15 11:10:00 | 757.58 | 2024-01-15 15:20:00 | 747.27 | TARGET_HIT | 0.50 | 1.36% |
| SELL | retest1 | 2024-01-19 10:55:00 | 730.58 | 2024-01-19 12:10:00 | 728.09 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-01-19 10:55:00 | 730.58 | 2024-01-19 14:25:00 | 730.58 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-31 11:05:00 | 690.50 | 2024-01-31 11:15:00 | 689.08 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-02-01 10:05:00 | 678.30 | 2024-02-01 10:40:00 | 679.85 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-02-02 11:10:00 | 685.04 | 2024-02-02 11:15:00 | 687.41 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-02-02 11:10:00 | 685.04 | 2024-02-02 12:25:00 | 685.04 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-06 10:40:00 | 654.45 | 2024-02-06 10:50:00 | 656.40 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-02-07 09:35:00 | 668.46 | 2024-02-07 09:40:00 | 671.29 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-02-07 09:35:00 | 668.46 | 2024-02-07 09:50:00 | 668.46 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-08 10:10:00 | 666.97 | 2024-02-08 10:15:00 | 664.20 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-02-08 10:10:00 | 666.97 | 2024-02-08 10:30:00 | 666.97 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-12 10:50:00 | 658.73 | 2024-02-12 11:30:00 | 656.48 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-02-12 10:50:00 | 658.73 | 2024-02-12 12:15:00 | 658.73 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-15 10:25:00 | 658.80 | 2024-02-15 11:10:00 | 660.45 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-02-16 10:55:00 | 662.65 | 2024-02-16 11:00:00 | 661.24 | PARTIAL | 0.50 | 0.21% |
| SELL | retest1 | 2024-02-16 10:55:00 | 662.65 | 2024-02-16 11:05:00 | 662.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-19 09:30:00 | 671.69 | 2024-02-19 10:00:00 | 674.03 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-02-19 09:30:00 | 671.69 | 2024-02-19 10:05:00 | 671.69 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-20 10:10:00 | 674.50 | 2024-02-20 10:20:00 | 672.62 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-02-21 10:25:00 | 673.67 | 2024-02-21 11:20:00 | 671.53 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-02-21 10:25:00 | 673.67 | 2024-02-21 13:15:00 | 673.67 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-22 10:55:00 | 657.63 | 2024-02-22 11:10:00 | 659.26 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-03-12 09:30:00 | 644.38 | 2024-03-12 10:40:00 | 642.29 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-03-13 10:10:00 | 650.80 | 2024-03-13 10:25:00 | 648.73 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-03-15 10:05:00 | 654.68 | 2024-03-15 10:35:00 | 651.72 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-03-18 10:50:00 | 653.94 | 2024-03-18 11:15:00 | 651.81 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-03-19 10:10:00 | 657.20 | 2024-03-19 10:15:00 | 655.63 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-03-20 10:30:00 | 666.30 | 2024-03-20 12:15:00 | 669.53 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-03-20 10:30:00 | 666.30 | 2024-03-20 13:55:00 | 666.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-22 11:00:00 | 676.10 | 2024-03-22 11:45:00 | 674.65 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-04-01 10:45:00 | 732.30 | 2024-04-01 12:10:00 | 730.02 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-04-02 10:10:00 | 715.26 | 2024-04-02 10:35:00 | 717.27 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-04-04 10:50:00 | 726.11 | 2024-04-04 11:00:00 | 728.27 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-04-05 10:35:00 | 717.50 | 2024-04-05 10:55:00 | 719.53 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-04-08 11:15:00 | 716.50 | 2024-04-08 11:30:00 | 715.03 | PARTIAL | 0.50 | 0.20% |
| SELL | retest1 | 2024-04-08 11:15:00 | 716.50 | 2024-04-08 12:10:00 | 716.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-15 10:55:00 | 718.84 | 2024-04-15 11:40:00 | 716.96 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-04-16 11:05:00 | 695.50 | 2024-04-16 12:30:00 | 693.23 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-04-16 11:05:00 | 695.50 | 2024-04-16 13:30:00 | 695.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-24 10:35:00 | 731.89 | 2024-04-24 11:10:00 | 734.20 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2024-04-24 10:35:00 | 731.89 | 2024-04-24 11:50:00 | 733.10 | TARGET_HIT | 0.50 | 0.17% |
| SELL | retest1 | 2024-04-25 10:05:00 | 720.41 | 2024-04-25 10:10:00 | 716.69 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-04-25 10:05:00 | 720.41 | 2024-04-25 10:15:00 | 720.41 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-07 09:50:00 | 681.93 | 2024-05-07 10:15:00 | 679.07 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-05-07 09:50:00 | 681.93 | 2024-05-07 11:00:00 | 681.93 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-09 11:00:00 | 671.90 | 2024-05-09 11:30:00 | 669.65 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-05-09 11:00:00 | 671.90 | 2024-05-09 15:20:00 | 658.91 | TARGET_HIT | 0.50 | 1.93% |
