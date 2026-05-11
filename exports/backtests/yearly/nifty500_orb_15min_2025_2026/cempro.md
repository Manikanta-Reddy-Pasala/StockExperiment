# Cemindia Projects Ltd. (CEMPRO)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (15388 bars)
- **Last close:** 955.20
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
| ENTRY1 | 53 |
| ENTRY2 | 0 |
| PARTIAL | 19 |
| TARGET_HIT | 6 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 72 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 47
- **Target hits / Stop hits / Partials:** 6 / 47 / 19
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 12.42%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 33 | 10 | 30.3% | 3 | 23 | 7 | 0.15% | 4.9% |
| BUY @ 2nd Alert (retest1) | 33 | 10 | 30.3% | 3 | 23 | 7 | 0.15% | 4.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 39 | 15 | 38.5% | 3 | 24 | 12 | 0.19% | 7.5% |
| SELL @ 2nd Alert (retest1) | 39 | 15 | 38.5% | 3 | 24 | 12 | 0.19% | 7.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 72 | 25 | 34.7% | 6 | 47 | 19 | 0.17% | 12.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-22 09:50:00 | 662.20 | 656.41 | 0.00 | ORB-long ORB[650.80,657.00] vol=1.6x ATR=2.90 |
| Stop hit — per-position SL triggered | 2025-05-22 09:55:00 | 659.30 | 657.19 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-28 09:50:00 | 717.60 | 722.86 | 0.00 | ORB-short ORB[721.50,732.15] vol=1.5x ATR=3.52 |
| Stop hit — per-position SL triggered | 2025-05-28 11:35:00 | 721.12 | 720.66 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-29 11:15:00 | 722.10 | 726.58 | 0.00 | ORB-short ORB[729.00,737.40] vol=1.9x ATR=2.52 |
| Stop hit — per-position SL triggered | 2025-05-29 13:55:00 | 724.62 | 725.36 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 11:15:00 | 723.40 | 721.04 | 0.00 | ORB-long ORB[714.50,722.00] vol=1.7x ATR=2.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-02 11:50:00 | 726.87 | 721.31 | 0.00 | T1 1.5R @ 726.87 |
| Target hit | 2025-06-02 15:20:00 | 738.85 | 727.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2025-06-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 10:05:00 | 745.70 | 742.68 | 0.00 | ORB-long ORB[735.25,741.85] vol=3.6x ATR=3.23 |
| Stop hit — per-position SL triggered | 2025-06-04 10:25:00 | 742.47 | 743.07 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 09:35:00 | 766.00 | 760.13 | 0.00 | ORB-long ORB[752.00,762.00] vol=1.8x ATR=3.57 |
| Stop hit — per-position SL triggered | 2025-06-05 10:05:00 | 762.43 | 763.45 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 09:35:00 | 785.65 | 794.39 | 0.00 | ORB-short ORB[791.00,802.60] vol=1.8x ATR=5.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 11:40:00 | 777.82 | 786.71 | 0.00 | T1 1.5R @ 777.82 |
| Target hit | 2025-06-12 15:20:00 | 762.95 | 772.91 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2025-06-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-13 10:20:00 | 771.40 | 757.38 | 0.00 | ORB-long ORB[748.80,759.00] vol=1.5x ATR=4.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 11:00:00 | 778.44 | 761.00 | 0.00 | T1 1.5R @ 778.44 |
| Stop hit — per-position SL triggered | 2025-06-13 11:35:00 | 771.40 | 762.07 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 10:10:00 | 817.45 | 811.38 | 0.00 | ORB-long ORB[805.50,815.00] vol=2.9x ATR=3.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-24 10:25:00 | 823.27 | 813.61 | 0.00 | T1 1.5R @ 823.27 |
| Stop hit — per-position SL triggered | 2025-06-24 10:30:00 | 817.45 | 813.68 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-07-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-03 09:35:00 | 919.75 | 923.78 | 0.00 | ORB-short ORB[920.75,932.00] vol=1.9x ATR=4.16 |
| Stop hit — per-position SL triggered | 2025-07-03 09:50:00 | 923.91 | 923.38 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-07-07 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-07 10:10:00 | 875.75 | 886.74 | 0.00 | ORB-short ORB[887.40,899.50] vol=2.5x ATR=4.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 10:30:00 | 869.00 | 884.51 | 0.00 | T1 1.5R @ 869.00 |
| Stop hit — per-position SL triggered | 2025-07-07 12:30:00 | 875.75 | 878.02 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 10:15:00 | 810.65 | 819.99 | 0.00 | ORB-short ORB[816.90,828.10] vol=1.6x ATR=3.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:35:00 | 805.63 | 816.18 | 0.00 | T1 1.5R @ 805.63 |
| Stop hit — per-position SL triggered | 2025-07-18 11:25:00 | 810.65 | 814.85 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 09:35:00 | 805.85 | 811.89 | 0.00 | ORB-short ORB[810.00,816.00] vol=1.8x ATR=2.70 |
| Stop hit — per-position SL triggered | 2025-07-23 09:50:00 | 808.55 | 811.15 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2025-07-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 09:30:00 | 802.50 | 809.49 | 0.00 | ORB-short ORB[805.00,814.50] vol=1.9x ATR=2.96 |
| Stop hit — per-position SL triggered | 2025-07-24 09:45:00 | 805.46 | 807.28 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-08-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 09:30:00 | 742.15 | 748.07 | 0.00 | ORB-short ORB[744.20,754.80] vol=1.8x ATR=3.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 09:35:00 | 737.39 | 746.55 | 0.00 | T1 1.5R @ 737.39 |
| Stop hit — per-position SL triggered | 2025-08-06 10:15:00 | 742.15 | 743.75 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-08-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-08 10:50:00 | 729.40 | 730.74 | 0.00 | ORB-short ORB[730.15,740.30] vol=1.6x ATR=3.12 |
| Stop hit — per-position SL triggered | 2025-08-08 11:00:00 | 732.52 | 730.70 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-08-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 09:45:00 | 781.45 | 771.47 | 0.00 | ORB-long ORB[762.00,773.30] vol=2.1x ATR=4.79 |
| Stop hit — per-position SL triggered | 2025-08-13 10:00:00 | 776.66 | 772.73 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-08-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 10:45:00 | 771.15 | 781.02 | 0.00 | ORB-short ORB[785.85,796.45] vol=1.8x ATR=2.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-14 11:50:00 | 767.09 | 778.45 | 0.00 | T1 1.5R @ 767.09 |
| Stop hit — per-position SL triggered | 2025-08-14 12:05:00 | 771.15 | 778.15 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-08-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:35:00 | 772.60 | 764.88 | 0.00 | ORB-long ORB[758.85,766.60] vol=3.1x ATR=3.04 |
| Stop hit — per-position SL triggered | 2025-08-21 09:40:00 | 769.56 | 765.24 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-08-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 09:35:00 | 750.60 | 754.57 | 0.00 | ORB-short ORB[752.80,761.30] vol=3.1x ATR=2.41 |
| Stop hit — per-position SL triggered | 2025-08-22 09:45:00 | 753.01 | 754.26 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-08-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 10:00:00 | 776.60 | 768.32 | 0.00 | ORB-long ORB[761.20,768.70] vol=2.8x ATR=2.56 |
| Stop hit — per-position SL triggered | 2025-08-25 10:05:00 | 774.04 | 769.08 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-10-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 09:40:00 | 812.40 | 818.71 | 0.00 | ORB-short ORB[816.60,825.45] vol=1.6x ATR=7.50 |
| Stop hit — per-position SL triggered | 2025-10-08 10:05:00 | 819.90 | 817.31 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-10-09 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-09 10:55:00 | 802.75 | 806.43 | 0.00 | ORB-short ORB[810.15,815.00] vol=3.9x ATR=2.44 |
| Stop hit — per-position SL triggered | 2025-10-09 11:30:00 | 805.19 | 805.34 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-10-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 10:50:00 | 805.00 | 811.03 | 0.00 | ORB-short ORB[808.80,817.45] vol=2.5x ATR=2.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 11:20:00 | 801.64 | 809.40 | 0.00 | T1 1.5R @ 801.64 |
| Stop hit — per-position SL triggered | 2025-10-13 11:50:00 | 805.00 | 808.71 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-10-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 09:40:00 | 795.55 | 799.99 | 0.00 | ORB-short ORB[799.45,808.40] vol=8.2x ATR=3.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 09:50:00 | 790.84 | 798.68 | 0.00 | T1 1.5R @ 790.84 |
| Target hit | 2025-10-14 15:20:00 | 765.95 | 775.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — SELL (started 2025-10-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-16 10:25:00 | 778.30 | 785.40 | 0.00 | ORB-short ORB[784.00,792.90] vol=1.8x ATR=3.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 11:05:00 | 773.60 | 782.70 | 0.00 | T1 1.5R @ 773.60 |
| Stop hit — per-position SL triggered | 2025-10-16 13:10:00 | 778.30 | 778.20 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2025-10-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 09:35:00 | 777.40 | 772.93 | 0.00 | ORB-long ORB[764.10,773.55] vol=2.5x ATR=3.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 09:50:00 | 782.43 | 775.35 | 0.00 | T1 1.5R @ 782.43 |
| Stop hit — per-position SL triggered | 2025-10-17 10:15:00 | 777.40 | 776.10 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-10-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 10:45:00 | 776.75 | 768.85 | 0.00 | ORB-long ORB[768.00,773.55] vol=1.6x ATR=2.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 11:35:00 | 780.73 | 770.67 | 0.00 | T1 1.5R @ 780.73 |
| Target hit | 2025-10-20 15:20:00 | 796.00 | 786.39 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — BUY (started 2025-10-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-23 09:35:00 | 801.90 | 797.15 | 0.00 | ORB-long ORB[791.00,799.20] vol=1.9x ATR=3.44 |
| Stop hit — per-position SL triggered | 2025-10-23 09:50:00 | 798.46 | 798.11 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-10-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-24 10:45:00 | 809.75 | 802.50 | 0.00 | ORB-long ORB[794.60,804.25] vol=2.2x ATR=3.05 |
| Stop hit — per-position SL triggered | 2025-10-24 10:55:00 | 806.70 | 802.85 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-10-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 10:55:00 | 801.90 | 795.60 | 0.00 | ORB-long ORB[791.45,801.25] vol=4.4x ATR=2.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 11:00:00 | 806.18 | 797.16 | 0.00 | T1 1.5R @ 806.18 |
| Target hit | 2025-10-27 15:20:00 | 830.80 | 819.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-29 11:15:00 | 831.45 | 836.44 | 0.00 | ORB-short ORB[837.00,845.95] vol=2.0x ATR=2.19 |
| Stop hit — per-position SL triggered | 2025-10-29 11:20:00 | 833.64 | 836.40 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-11-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-06 09:30:00 | 873.30 | 867.32 | 0.00 | ORB-long ORB[861.00,869.50] vol=2.2x ATR=3.08 |
| Stop hit — per-position SL triggered | 2025-11-06 09:40:00 | 870.22 | 868.37 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-11-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 09:30:00 | 829.50 | 836.18 | 0.00 | ORB-short ORB[835.25,844.70] vol=2.8x ATR=4.91 |
| Stop hit — per-position SL triggered | 2025-11-07 09:45:00 | 834.41 | 834.27 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 11:15:00 | 828.40 | 822.03 | 0.00 | ORB-long ORB[814.80,826.40] vol=2.4x ATR=2.71 |
| Stop hit — per-position SL triggered | 2025-11-12 11:55:00 | 825.69 | 822.53 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-11-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 10:10:00 | 803.00 | 812.09 | 0.00 | ORB-short ORB[811.20,820.90] vol=1.6x ATR=2.75 |
| Stop hit — per-position SL triggered | 2025-11-20 10:40:00 | 805.75 | 809.33 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-11-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:20:00 | 797.00 | 805.90 | 0.00 | ORB-short ORB[803.80,812.80] vol=3.0x ATR=2.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 10:30:00 | 793.51 | 804.07 | 0.00 | T1 1.5R @ 793.51 |
| Stop hit — per-position SL triggered | 2025-11-21 10:45:00 | 797.00 | 802.59 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-11-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 10:50:00 | 773.20 | 776.84 | 0.00 | ORB-short ORB[779.00,787.35] vol=2.4x ATR=2.34 |
| Stop hit — per-position SL triggered | 2025-11-24 11:10:00 | 775.54 | 776.73 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-11-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 09:35:00 | 787.40 | 782.88 | 0.00 | ORB-long ORB[772.90,784.00] vol=3.7x ATR=2.82 |
| Stop hit — per-position SL triggered | 2025-11-26 09:55:00 | 784.58 | 783.31 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-11-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 10:45:00 | 802.60 | 795.64 | 0.00 | ORB-long ORB[787.30,798.25] vol=1.6x ATR=1.93 |
| Stop hit — per-position SL triggered | 2025-11-27 10:55:00 | 800.67 | 796.26 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-12-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:10:00 | 804.85 | 813.68 | 0.00 | ORB-short ORB[816.95,823.90] vol=2.6x ATR=2.44 |
| Stop hit — per-position SL triggered | 2025-12-08 10:20:00 | 807.29 | 813.04 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-12-09 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 10:20:00 | 806.65 | 792.96 | 0.00 | ORB-long ORB[785.10,792.00] vol=1.7x ATR=3.87 |
| Stop hit — per-position SL triggered | 2025-12-09 10:35:00 | 802.78 | 794.63 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-12-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 10:00:00 | 832.15 | 829.81 | 0.00 | ORB-long ORB[822.10,831.60] vol=2.8x ATR=2.79 |
| Stop hit — per-position SL triggered | 2025-12-15 10:05:00 | 829.36 | 829.87 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-12-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 09:40:00 | 812.10 | 814.08 | 0.00 | ORB-short ORB[813.55,818.70] vol=1.6x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 09:50:00 | 809.37 | 812.77 | 0.00 | T1 1.5R @ 809.37 |
| Stop hit — per-position SL triggered | 2025-12-17 10:00:00 | 812.10 | 812.59 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 10:15:00 | 796.50 | 793.89 | 0.00 | ORB-long ORB[786.00,796.35] vol=1.8x ATR=2.80 |
| Stop hit — per-position SL triggered | 2025-12-19 11:05:00 | 793.70 | 794.20 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-12-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 11:10:00 | 792.35 | 797.04 | 0.00 | ORB-short ORB[794.75,799.30] vol=1.5x ATR=1.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 11:15:00 | 789.81 | 795.49 | 0.00 | T1 1.5R @ 789.81 |
| Target hit | 2025-12-29 13:55:00 | 790.00 | 789.69 | 0.00 | Trail-exit close>VWAP |

### Cycle 47 — SELL (started 2026-01-02 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-02 10:10:00 | 765.20 | 768.50 | 0.00 | ORB-short ORB[768.50,773.95] vol=2.8x ATR=1.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 10:15:00 | 762.41 | 767.24 | 0.00 | T1 1.5R @ 762.41 |
| Stop hit — per-position SL triggered | 2026-01-02 10:40:00 | 765.20 | 765.18 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2026-01-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 09:35:00 | 757.10 | 753.12 | 0.00 | ORB-long ORB[745.55,756.30] vol=1.5x ATR=2.93 |
| Stop hit — per-position SL triggered | 2026-01-07 09:40:00 | 754.17 | 753.27 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2026-01-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-23 11:00:00 | 640.00 | 644.95 | 0.00 | ORB-short ORB[642.00,649.40] vol=3.6x ATR=1.69 |
| Stop hit — per-position SL triggered | 2026-01-23 11:05:00 | 641.69 | 644.72 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2026-02-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 09:50:00 | 617.90 | 612.94 | 0.00 | ORB-long ORB[608.55,614.20] vol=1.5x ATR=3.23 |
| Stop hit — per-position SL triggered | 2026-02-13 09:55:00 | 614.67 | 613.34 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2026-02-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:50:00 | 603.80 | 596.99 | 0.00 | ORB-long ORB[589.15,596.55] vol=2.7x ATR=2.87 |
| Stop hit — per-position SL triggered | 2026-02-17 10:00:00 | 600.93 | 598.01 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2026-03-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 09:55:00 | 567.40 | 563.45 | 0.00 | ORB-long ORB[558.30,563.85] vol=4.3x ATR=2.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:10:00 | 571.29 | 564.25 | 0.00 | T1 1.5R @ 571.29 |
| Stop hit — per-position SL triggered | 2026-03-06 10:45:00 | 567.40 | 566.39 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2026-03-17 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:25:00 | 538.40 | 535.45 | 0.00 | ORB-long ORB[530.05,537.45] vol=2.9x ATR=2.55 |
| Stop hit — per-position SL triggered | 2026-03-17 11:05:00 | 535.85 | 535.80 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-22 09:50:00 | 662.20 | 2025-05-22 09:55:00 | 659.30 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-05-28 09:50:00 | 717.60 | 2025-05-28 11:35:00 | 721.12 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2025-05-29 11:15:00 | 722.10 | 2025-05-29 13:55:00 | 724.62 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-06-02 11:15:00 | 723.40 | 2025-06-02 11:50:00 | 726.87 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-06-02 11:15:00 | 723.40 | 2025-06-02 15:20:00 | 738.85 | TARGET_HIT | 0.50 | 2.14% |
| BUY | retest1 | 2025-06-04 10:05:00 | 745.70 | 2025-06-04 10:25:00 | 742.47 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-06-05 09:35:00 | 766.00 | 2025-06-05 10:05:00 | 762.43 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2025-06-12 09:35:00 | 785.65 | 2025-06-12 11:40:00 | 777.82 | PARTIAL | 0.50 | 1.00% |
| SELL | retest1 | 2025-06-12 09:35:00 | 785.65 | 2025-06-12 15:20:00 | 762.95 | TARGET_HIT | 0.50 | 2.89% |
| BUY | retest1 | 2025-06-13 10:20:00 | 771.40 | 2025-06-13 11:00:00 | 778.44 | PARTIAL | 0.50 | 0.91% |
| BUY | retest1 | 2025-06-13 10:20:00 | 771.40 | 2025-06-13 11:35:00 | 771.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-24 10:10:00 | 817.45 | 2025-06-24 10:25:00 | 823.27 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2025-06-24 10:10:00 | 817.45 | 2025-06-24 10:30:00 | 817.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-03 09:35:00 | 919.75 | 2025-07-03 09:50:00 | 923.91 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-07-07 10:10:00 | 875.75 | 2025-07-07 10:30:00 | 869.00 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2025-07-07 10:10:00 | 875.75 | 2025-07-07 12:30:00 | 875.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-18 10:15:00 | 810.65 | 2025-07-18 10:35:00 | 805.63 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-07-18 10:15:00 | 810.65 | 2025-07-18 11:25:00 | 810.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-23 09:35:00 | 805.85 | 2025-07-23 09:50:00 | 808.55 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-07-24 09:30:00 | 802.50 | 2025-07-24 09:45:00 | 805.46 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-08-06 09:30:00 | 742.15 | 2025-08-06 09:35:00 | 737.39 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2025-08-06 09:30:00 | 742.15 | 2025-08-06 10:15:00 | 742.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-08 10:50:00 | 729.40 | 2025-08-08 11:00:00 | 732.52 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-08-13 09:45:00 | 781.45 | 2025-08-13 10:00:00 | 776.66 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest1 | 2025-08-14 10:45:00 | 771.15 | 2025-08-14 11:50:00 | 767.09 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-08-14 10:45:00 | 771.15 | 2025-08-14 12:05:00 | 771.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-21 09:35:00 | 772.60 | 2025-08-21 09:40:00 | 769.56 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-08-22 09:35:00 | 750.60 | 2025-08-22 09:45:00 | 753.01 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-08-25 10:00:00 | 776.60 | 2025-08-25 10:05:00 | 774.04 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-10-08 09:40:00 | 812.40 | 2025-10-08 10:05:00 | 819.90 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest1 | 2025-10-09 10:55:00 | 802.75 | 2025-10-09 11:30:00 | 805.19 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-10-13 10:50:00 | 805.00 | 2025-10-13 11:20:00 | 801.64 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-10-13 10:50:00 | 805.00 | 2025-10-13 11:50:00 | 805.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-14 09:40:00 | 795.55 | 2025-10-14 09:50:00 | 790.84 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2025-10-14 09:40:00 | 795.55 | 2025-10-14 15:20:00 | 765.95 | TARGET_HIT | 0.50 | 3.72% |
| SELL | retest1 | 2025-10-16 10:25:00 | 778.30 | 2025-10-16 11:05:00 | 773.60 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2025-10-16 10:25:00 | 778.30 | 2025-10-16 13:10:00 | 778.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-17 09:35:00 | 777.40 | 2025-10-17 09:50:00 | 782.43 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2025-10-17 09:35:00 | 777.40 | 2025-10-17 10:15:00 | 777.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-20 10:45:00 | 776.75 | 2025-10-20 11:35:00 | 780.73 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-10-20 10:45:00 | 776.75 | 2025-10-20 15:20:00 | 796.00 | TARGET_HIT | 0.50 | 2.48% |
| BUY | retest1 | 2025-10-23 09:35:00 | 801.90 | 2025-10-23 09:50:00 | 798.46 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-10-24 10:45:00 | 809.75 | 2025-10-24 10:55:00 | 806.70 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-10-27 10:55:00 | 801.90 | 2025-10-27 11:00:00 | 806.18 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-10-27 10:55:00 | 801.90 | 2025-10-27 15:20:00 | 830.80 | TARGET_HIT | 0.50 | 3.60% |
| SELL | retest1 | 2025-10-29 11:15:00 | 831.45 | 2025-10-29 11:20:00 | 833.64 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-11-06 09:30:00 | 873.30 | 2025-11-06 09:40:00 | 870.22 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-11-07 09:30:00 | 829.50 | 2025-11-07 09:45:00 | 834.41 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest1 | 2025-11-12 11:15:00 | 828.40 | 2025-11-12 11:55:00 | 825.69 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-11-20 10:10:00 | 803.00 | 2025-11-20 10:40:00 | 805.75 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-11-21 10:20:00 | 797.00 | 2025-11-21 10:30:00 | 793.51 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2025-11-21 10:20:00 | 797.00 | 2025-11-21 10:45:00 | 797.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-24 10:50:00 | 773.20 | 2025-11-24 11:10:00 | 775.54 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-11-26 09:35:00 | 787.40 | 2025-11-26 09:55:00 | 784.58 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-11-27 10:45:00 | 802.60 | 2025-11-27 10:55:00 | 800.67 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-12-08 10:10:00 | 804.85 | 2025-12-08 10:20:00 | 807.29 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-12-09 10:20:00 | 806.65 | 2025-12-09 10:35:00 | 802.78 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-12-15 10:00:00 | 832.15 | 2025-12-15 10:05:00 | 829.36 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-12-17 09:40:00 | 812.10 | 2025-12-17 09:50:00 | 809.37 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-12-17 09:40:00 | 812.10 | 2025-12-17 10:00:00 | 812.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-19 10:15:00 | 796.50 | 2025-12-19 11:05:00 | 793.70 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-12-29 11:10:00 | 792.35 | 2025-12-29 11:15:00 | 789.81 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-12-29 11:10:00 | 792.35 | 2025-12-29 13:55:00 | 790.00 | TARGET_HIT | 0.50 | 0.30% |
| SELL | retest1 | 2026-01-02 10:10:00 | 765.20 | 2026-01-02 10:15:00 | 762.41 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-01-02 10:10:00 | 765.20 | 2026-01-02 10:40:00 | 765.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-07 09:35:00 | 757.10 | 2026-01-07 09:40:00 | 754.17 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-01-23 11:00:00 | 640.00 | 2026-01-23 11:05:00 | 641.69 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-02-13 09:50:00 | 617.90 | 2026-02-13 09:55:00 | 614.67 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2026-02-17 09:50:00 | 603.80 | 2026-02-17 10:00:00 | 600.93 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-03-06 09:55:00 | 567.40 | 2026-03-06 10:10:00 | 571.29 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2026-03-06 09:55:00 | 567.40 | 2026-03-06 10:45:00 | 567.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 10:25:00 | 538.40 | 2026-03-17 11:05:00 | 535.85 | STOP_HIT | 1.00 | -0.47% |
