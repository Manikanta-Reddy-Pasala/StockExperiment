# Aurobindo Pharma Ltd. (AUROPHARMA)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2026-05-08 15:25:00 (55355 bars)
- **Last close:** 1487.70
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
| ENTRY1 | 78 |
| ENTRY2 | 0 |
| PARTIAL | 35 |
| TARGET_HIT | 13 |
| STOP_HIT | 65 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 113 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 49 / 64
- **Target hits / Stop hits / Partials:** 13 / 65 / 35
- **Avg / median % per leg:** 0.21% / 0.00%
- **Sum % (uncompounded):** 23.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 75 | 36 | 48.0% | 10 | 40 | 25 | 0.29% | 21.7% |
| BUY @ 2nd Alert (retest1) | 75 | 36 | 48.0% | 10 | 40 | 25 | 0.29% | 21.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 38 | 13 | 34.2% | 3 | 25 | 10 | 0.04% | 1.5% |
| SELL @ 2nd Alert (retest1) | 38 | 13 | 34.2% | 3 | 25 | 10 | 0.04% | 1.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 113 | 49 | 43.4% | 13 | 65 | 35 | 0.21% | 23.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-12 11:15:00 | 611.00 | 605.43 | 0.00 | ORB-long ORB[603.45,609.50] vol=1.7x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-12 12:25:00 | 614.25 | 607.17 | 0.00 | T1 1.5R @ 614.25 |
| Stop hit — per-position SL triggered | 2023-05-12 13:55:00 | 611.00 | 607.90 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-15 10:25:00 | 610.60 | 606.55 | 0.00 | ORB-long ORB[602.60,607.80] vol=2.3x ATR=1.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-15 11:40:00 | 613.31 | 608.53 | 0.00 | T1 1.5R @ 613.31 |
| Stop hit — per-position SL triggered | 2023-05-15 13:05:00 | 610.60 | 610.19 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-18 09:35:00 | 613.05 | 615.48 | 0.00 | ORB-short ORB[613.85,620.55] vol=1.9x ATR=1.85 |
| Stop hit — per-position SL triggered | 2023-05-18 09:45:00 | 614.90 | 615.19 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-06-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-01 11:15:00 | 649.60 | 652.74 | 0.00 | ORB-short ORB[654.00,658.65] vol=1.8x ATR=1.42 |
| Stop hit — per-position SL triggered | 2023-06-01 13:05:00 | 651.02 | 652.30 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2023-06-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-06 09:50:00 | 654.45 | 657.32 | 0.00 | ORB-short ORB[655.45,662.75] vol=1.5x ATR=1.53 |
| Stop hit — per-position SL triggered | 2023-06-06 10:00:00 | 655.98 | 657.09 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2023-06-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-14 11:15:00 | 671.60 | 676.43 | 0.00 | ORB-short ORB[677.05,683.40] vol=1.6x ATR=1.32 |
| Stop hit — per-position SL triggered | 2023-06-14 11:35:00 | 672.92 | 675.90 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-06-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-19 09:35:00 | 689.00 | 686.11 | 0.00 | ORB-long ORB[680.00,686.95] vol=1.5x ATR=2.08 |
| Stop hit — per-position SL triggered | 2023-06-19 10:00:00 | 686.92 | 687.71 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-21 11:15:00 | 674.80 | 678.78 | 0.00 | ORB-short ORB[675.60,682.50] vol=5.9x ATR=1.22 |
| Stop hit — per-position SL triggered | 2023-06-21 11:20:00 | 676.02 | 678.55 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-06-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-22 09:55:00 | 685.05 | 683.99 | 0.00 | ORB-long ORB[678.30,685.00] vol=2.7x ATR=1.58 |
| Stop hit — per-position SL triggered | 2023-06-22 10:00:00 | 683.47 | 683.98 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-06-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-23 09:45:00 | 673.00 | 670.68 | 0.00 | ORB-long ORB[665.75,671.30] vol=8.6x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-23 10:15:00 | 675.98 | 671.35 | 0.00 | T1 1.5R @ 675.98 |
| Target hit | 2023-06-23 15:20:00 | 700.00 | 688.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — BUY (started 2023-07-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-05 09:50:00 | 723.35 | 718.61 | 0.00 | ORB-long ORB[714.75,720.45] vol=2.3x ATR=2.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 09:55:00 | 726.58 | 719.87 | 0.00 | T1 1.5R @ 726.58 |
| Stop hit — per-position SL triggered | 2023-07-05 10:00:00 | 723.35 | 720.10 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-07-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-06 09:45:00 | 735.95 | 731.86 | 0.00 | ORB-long ORB[725.00,733.95] vol=1.7x ATR=2.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-06 09:55:00 | 739.31 | 733.03 | 0.00 | T1 1.5R @ 739.31 |
| Target hit | 2023-07-06 15:20:00 | 762.45 | 751.68 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2023-07-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 09:45:00 | 741.15 | 734.77 | 0.00 | ORB-long ORB[730.65,736.85] vol=1.6x ATR=2.32 |
| Stop hit — per-position SL triggered | 2023-07-12 09:50:00 | 738.83 | 735.10 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-07-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 09:35:00 | 732.00 | 728.31 | 0.00 | ORB-long ORB[723.40,730.60] vol=1.6x ATR=2.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-14 09:55:00 | 735.33 | 731.72 | 0.00 | T1 1.5R @ 735.33 |
| Target hit | 2023-07-14 11:45:00 | 736.80 | 736.80 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — BUY (started 2023-07-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-19 09:40:00 | 764.45 | 761.52 | 0.00 | ORB-long ORB[756.00,763.00] vol=2.6x ATR=2.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-19 09:45:00 | 768.42 | 763.35 | 0.00 | T1 1.5R @ 768.42 |
| Stop hit — per-position SL triggered | 2023-07-19 10:25:00 | 764.45 | 765.72 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2023-07-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-20 10:55:00 | 760.00 | 754.92 | 0.00 | ORB-long ORB[748.65,754.00] vol=2.2x ATR=2.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-20 11:25:00 | 763.40 | 756.77 | 0.00 | T1 1.5R @ 763.40 |
| Target hit | 2023-07-20 15:20:00 | 773.00 | 765.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2023-07-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-21 09:40:00 | 778.75 | 775.75 | 0.00 | ORB-long ORB[771.00,778.00] vol=1.5x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-21 09:50:00 | 781.71 | 777.93 | 0.00 | T1 1.5R @ 781.71 |
| Target hit | 2023-07-21 11:10:00 | 781.55 | 782.96 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — BUY (started 2023-07-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-24 09:40:00 | 781.35 | 776.04 | 0.00 | ORB-long ORB[771.00,777.75] vol=3.3x ATR=2.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-24 10:00:00 | 785.13 | 778.53 | 0.00 | T1 1.5R @ 785.13 |
| Target hit | 2023-07-24 15:00:00 | 785.90 | 786.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 19 — BUY (started 2023-07-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-25 09:50:00 | 789.45 | 786.87 | 0.00 | ORB-long ORB[780.05,787.60] vol=1.7x ATR=2.06 |
| Stop hit — per-position SL triggered | 2023-07-25 09:55:00 | 787.39 | 786.89 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-07-26 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-26 10:05:00 | 791.50 | 789.02 | 0.00 | ORB-long ORB[784.55,791.00] vol=2.1x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-26 10:10:00 | 794.57 | 790.28 | 0.00 | T1 1.5R @ 794.57 |
| Stop hit — per-position SL triggered | 2023-07-26 10:20:00 | 791.50 | 790.61 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-08-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-18 09:30:00 | 877.70 | 873.07 | 0.00 | ORB-long ORB[865.00,874.90] vol=1.9x ATR=2.51 |
| Stop hit — per-position SL triggered | 2023-08-18 09:35:00 | 875.19 | 873.97 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-08-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-24 11:10:00 | 846.75 | 842.63 | 0.00 | ORB-long ORB[838.90,846.20] vol=2.2x ATR=1.92 |
| Stop hit — per-position SL triggered | 2023-08-24 11:15:00 | 844.83 | 842.79 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 10:15:00 | 832.35 | 834.77 | 0.00 | ORB-short ORB[833.80,841.00] vol=2.0x ATR=2.36 |
| Stop hit — per-position SL triggered | 2023-08-25 10:20:00 | 834.71 | 834.74 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-09-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-01 10:50:00 | 835.05 | 832.28 | 0.00 | ORB-long ORB[829.10,834.90] vol=1.6x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-01 11:05:00 | 838.04 | 833.25 | 0.00 | T1 1.5R @ 838.04 |
| Stop hit — per-position SL triggered | 2023-09-01 11:15:00 | 835.05 | 833.56 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-09-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 09:45:00 | 834.35 | 829.08 | 0.00 | ORB-long ORB[820.10,830.90] vol=1.6x ATR=2.46 |
| Stop hit — per-position SL triggered | 2023-09-05 09:55:00 | 831.89 | 829.72 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-09-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 09:40:00 | 863.30 | 858.37 | 0.00 | ORB-long ORB[852.00,859.60] vol=2.0x ATR=3.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-06 10:20:00 | 868.28 | 861.48 | 0.00 | T1 1.5R @ 868.28 |
| Stop hit — per-position SL triggered | 2023-09-06 10:50:00 | 863.30 | 862.54 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-09-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-08 11:00:00 | 852.90 | 855.08 | 0.00 | ORB-short ORB[857.30,863.80] vol=1.8x ATR=1.83 |
| Stop hit — per-position SL triggered | 2023-09-08 11:20:00 | 854.73 | 854.83 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-09-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-11 09:35:00 | 867.80 | 864.17 | 0.00 | ORB-long ORB[855.10,866.05] vol=2.1x ATR=2.19 |
| Stop hit — per-position SL triggered | 2023-09-11 09:55:00 | 865.61 | 866.08 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2023-09-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-13 09:30:00 | 887.90 | 882.47 | 0.00 | ORB-long ORB[871.00,883.55] vol=4.8x ATR=3.34 |
| Stop hit — per-position SL triggered | 2023-09-13 09:35:00 | 884.56 | 882.53 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-09-14 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-14 11:05:00 | 891.80 | 893.55 | 0.00 | ORB-short ORB[892.05,904.50] vol=2.7x ATR=2.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-14 11:25:00 | 887.68 | 893.24 | 0.00 | T1 1.5R @ 887.68 |
| Stop hit — per-position SL triggered | 2023-09-14 12:55:00 | 891.80 | 892.10 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2023-09-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-20 09:50:00 | 895.15 | 893.32 | 0.00 | ORB-long ORB[887.00,894.70] vol=2.1x ATR=2.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-20 10:05:00 | 899.01 | 894.08 | 0.00 | T1 1.5R @ 899.01 |
| Stop hit — per-position SL triggered | 2023-09-20 10:10:00 | 895.15 | 894.30 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2023-09-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-27 09:40:00 | 865.70 | 861.36 | 0.00 | ORB-long ORB[852.90,864.15] vol=1.6x ATR=2.66 |
| Stop hit — per-position SL triggered | 2023-09-27 09:45:00 | 863.04 | 861.54 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2023-10-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-06 09:45:00 | 889.00 | 884.79 | 0.00 | ORB-long ORB[877.10,886.20] vol=1.7x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-06 10:20:00 | 892.10 | 887.21 | 0.00 | T1 1.5R @ 892.10 |
| Stop hit — per-position SL triggered | 2023-10-06 10:45:00 | 889.00 | 888.25 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-10-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-09 09:30:00 | 907.20 | 897.91 | 0.00 | ORB-long ORB[887.85,899.35] vol=2.6x ATR=3.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-09 09:35:00 | 912.46 | 902.12 | 0.00 | T1 1.5R @ 912.46 |
| Stop hit — per-position SL triggered | 2023-10-09 09:40:00 | 907.20 | 903.10 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2023-10-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-11 09:35:00 | 899.20 | 901.86 | 0.00 | ORB-short ORB[901.20,904.95] vol=1.6x ATR=1.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-11 09:40:00 | 896.63 | 901.13 | 0.00 | T1 1.5R @ 896.63 |
| Stop hit — per-position SL triggered | 2023-10-11 09:55:00 | 899.20 | 899.62 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-10-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-12 09:30:00 | 911.80 | 915.46 | 0.00 | ORB-short ORB[912.05,922.00] vol=2.3x ATR=2.82 |
| Stop hit — per-position SL triggered | 2023-10-12 09:40:00 | 914.62 | 915.26 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-10-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-13 10:55:00 | 914.80 | 913.89 | 0.00 | ORB-long ORB[906.50,914.75] vol=4.2x ATR=2.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-13 11:55:00 | 918.19 | 914.54 | 0.00 | T1 1.5R @ 918.19 |
| Target hit | 2023-10-13 15:20:00 | 920.25 | 916.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — BUY (started 2023-10-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-17 10:50:00 | 918.85 | 914.23 | 0.00 | ORB-long ORB[908.65,914.00] vol=1.6x ATR=1.58 |
| Stop hit — per-position SL triggered | 2023-10-17 10:55:00 | 917.27 | 914.77 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2023-10-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-18 09:45:00 | 902.75 | 906.98 | 0.00 | ORB-short ORB[905.40,911.80] vol=1.8x ATR=2.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-18 09:50:00 | 899.40 | 905.39 | 0.00 | T1 1.5R @ 899.40 |
| Stop hit — per-position SL triggered | 2023-10-18 10:00:00 | 902.75 | 905.02 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2023-10-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-19 11:05:00 | 891.75 | 894.56 | 0.00 | ORB-short ORB[892.50,899.70] vol=1.6x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-19 11:20:00 | 888.61 | 894.05 | 0.00 | T1 1.5R @ 888.61 |
| Stop hit — per-position SL triggered | 2023-10-19 12:20:00 | 891.75 | 892.85 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-10-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-27 11:00:00 | 863.90 | 860.94 | 0.00 | ORB-long ORB[851.55,860.75] vol=3.8x ATR=2.06 |
| Stop hit — per-position SL triggered | 2023-10-27 11:05:00 | 861.84 | 860.96 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2023-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-30 10:15:00 | 863.75 | 859.37 | 0.00 | ORB-long ORB[851.30,862.90] vol=2.3x ATR=2.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-30 10:25:00 | 868.08 | 860.62 | 0.00 | T1 1.5R @ 868.08 |
| Stop hit — per-position SL triggered | 2023-10-30 11:10:00 | 863.75 | 861.72 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-11-01 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-01 10:10:00 | 836.05 | 841.47 | 0.00 | ORB-short ORB[839.35,851.85] vol=2.1x ATR=2.99 |
| Stop hit — per-position SL triggered | 2023-11-01 10:20:00 | 839.04 | 841.23 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2023-11-02 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 10:10:00 | 870.55 | 867.62 | 0.00 | ORB-long ORB[860.15,869.35] vol=2.0x ATR=2.87 |
| Stop hit — per-position SL triggered | 2023-11-02 10:30:00 | 867.68 | 868.05 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-11-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 09:55:00 | 924.90 | 915.40 | 0.00 | ORB-long ORB[903.60,916.45] vol=3.3x ATR=3.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-08 10:05:00 | 930.71 | 919.03 | 0.00 | T1 1.5R @ 930.71 |
| Target hit | 2023-11-08 13:50:00 | 931.75 | 932.84 | 0.00 | Trail-exit close<VWAP |

### Cycle 46 — BUY (started 2023-11-09 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-09 11:05:00 | 951.00 | 940.41 | 0.00 | ORB-long ORB[929.30,940.25] vol=3.0x ATR=3.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-09 11:20:00 | 955.61 | 942.88 | 0.00 | T1 1.5R @ 955.61 |
| Stop hit — per-position SL triggered | 2023-11-09 12:05:00 | 951.00 | 945.83 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2023-11-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-22 10:35:00 | 1038.45 | 1030.12 | 0.00 | ORB-long ORB[1022.60,1035.00] vol=2.9x ATR=3.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-22 11:55:00 | 1043.92 | 1033.49 | 0.00 | T1 1.5R @ 1043.92 |
| Stop hit — per-position SL triggered | 2023-11-22 12:25:00 | 1038.45 | 1034.91 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-11-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-23 09:40:00 | 1037.55 | 1041.78 | 0.00 | ORB-short ORB[1041.10,1055.00] vol=2.5x ATR=3.67 |
| Stop hit — per-position SL triggered | 2023-11-23 10:10:00 | 1041.22 | 1040.75 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2023-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 11:15:00 | 1023.30 | 1019.87 | 0.00 | ORB-long ORB[1010.85,1019.65] vol=1.9x ATR=3.07 |
| Stop hit — per-position SL triggered | 2023-11-29 11:30:00 | 1020.23 | 1020.01 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2023-12-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-04 09:50:00 | 1026.50 | 1031.47 | 0.00 | ORB-short ORB[1030.95,1046.00] vol=1.8x ATR=3.41 |
| Stop hit — per-position SL triggered | 2023-12-04 10:35:00 | 1029.91 | 1029.37 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2023-12-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-05 10:40:00 | 1030.00 | 1033.80 | 0.00 | ORB-short ORB[1034.60,1047.80] vol=1.7x ATR=2.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-05 11:00:00 | 1025.55 | 1032.79 | 0.00 | T1 1.5R @ 1025.55 |
| Target hit | 2023-12-05 15:00:00 | 1026.20 | 1025.82 | 0.00 | Trail-exit close>VWAP |

### Cycle 52 — BUY (started 2023-12-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-07 09:55:00 | 1020.45 | 1015.38 | 0.00 | ORB-long ORB[1009.65,1017.05] vol=3.2x ATR=3.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-07 10:15:00 | 1025.11 | 1017.59 | 0.00 | T1 1.5R @ 1025.11 |
| Stop hit — per-position SL triggered | 2023-12-07 10:20:00 | 1020.45 | 1017.82 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2023-12-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-11 10:55:00 | 998.25 | 1004.06 | 0.00 | ORB-short ORB[1002.00,1016.00] vol=4.1x ATR=3.66 |
| Stop hit — per-position SL triggered | 2023-12-11 11:15:00 | 1001.91 | 1003.76 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-12-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-13 09:30:00 | 1027.00 | 1020.99 | 0.00 | ORB-long ORB[1010.30,1021.95] vol=4.3x ATR=3.40 |
| Stop hit — per-position SL triggered | 2023-12-13 09:45:00 | 1023.60 | 1023.74 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2023-12-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-15 09:55:00 | 1039.00 | 1040.92 | 0.00 | ORB-short ORB[1040.25,1050.35] vol=6.0x ATR=3.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-15 10:45:00 | 1034.38 | 1040.08 | 0.00 | T1 1.5R @ 1034.38 |
| Target hit | 2023-12-15 15:20:00 | 1024.55 | 1033.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — BUY (started 2023-12-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-19 11:05:00 | 1032.30 | 1022.76 | 0.00 | ORB-long ORB[1021.15,1026.00] vol=1.6x ATR=3.11 |
| Stop hit — per-position SL triggered | 2023-12-19 11:50:00 | 1029.19 | 1024.84 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2023-12-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 09:45:00 | 1077.75 | 1072.17 | 0.00 | ORB-long ORB[1061.70,1073.60] vol=2.2x ATR=3.61 |
| Stop hit — per-position SL triggered | 2023-12-27 09:55:00 | 1074.14 | 1072.90 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2023-12-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-28 11:10:00 | 1057.30 | 1058.86 | 0.00 | ORB-short ORB[1063.15,1075.00] vol=4.3x ATR=3.40 |
| Stop hit — per-position SL triggered | 2023-12-28 11:20:00 | 1060.70 | 1059.01 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-01-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-03 09:35:00 | 1080.95 | 1087.32 | 0.00 | ORB-short ORB[1086.95,1096.80] vol=2.1x ATR=3.75 |
| Stop hit — per-position SL triggered | 2024-01-03 09:40:00 | 1084.70 | 1086.87 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2024-01-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 10:25:00 | 1103.75 | 1096.23 | 0.00 | ORB-long ORB[1091.90,1100.00] vol=1.5x ATR=3.41 |
| Stop hit — per-position SL triggered | 2024-01-04 10:50:00 | 1100.34 | 1098.67 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2024-01-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-09 09:55:00 | 1119.95 | 1121.65 | 0.00 | ORB-short ORB[1120.35,1130.00] vol=6.2x ATR=3.71 |
| Stop hit — per-position SL triggered | 2024-01-09 10:00:00 | 1123.66 | 1121.79 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-01-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-12 11:05:00 | 1084.35 | 1097.05 | 0.00 | ORB-short ORB[1100.00,1113.00] vol=4.8x ATR=3.10 |
| Stop hit — per-position SL triggered | 2024-01-12 11:20:00 | 1087.45 | 1095.86 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2024-01-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-15 09:40:00 | 1117.95 | 1112.12 | 0.00 | ORB-long ORB[1097.35,1112.25] vol=2.8x ATR=3.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-15 09:45:00 | 1123.73 | 1116.48 | 0.00 | T1 1.5R @ 1123.73 |
| Target hit | 2024-01-15 15:20:00 | 1150.25 | 1143.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — BUY (started 2024-01-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-20 10:40:00 | 1157.25 | 1152.94 | 0.00 | ORB-long ORB[1140.35,1156.70] vol=5.9x ATR=5.35 |
| Stop hit — per-position SL triggered | 2024-01-20 10:45:00 | 1151.90 | 1152.64 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-02-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-07 09:40:00 | 1002.05 | 1013.53 | 0.00 | ORB-short ORB[1009.55,1022.75] vol=1.9x ATR=5.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-07 09:50:00 | 993.94 | 1004.40 | 0.00 | T1 1.5R @ 993.94 |
| Stop hit — per-position SL triggered | 2024-02-07 09:55:00 | 1002.05 | 1002.93 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2024-02-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-20 10:55:00 | 1045.20 | 1037.09 | 0.00 | ORB-long ORB[1032.45,1044.90] vol=3.1x ATR=3.26 |
| Stop hit — per-position SL triggered | 2024-02-20 11:00:00 | 1041.94 | 1037.45 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2024-02-26 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-26 09:50:00 | 1027.35 | 1033.10 | 0.00 | ORB-short ORB[1036.05,1047.00] vol=5.0x ATR=4.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-26 09:55:00 | 1020.63 | 1030.67 | 0.00 | T1 1.5R @ 1020.63 |
| Stop hit — per-position SL triggered | 2024-02-26 10:05:00 | 1027.35 | 1029.06 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2024-02-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-28 10:55:00 | 1035.30 | 1044.77 | 0.00 | ORB-short ORB[1038.55,1050.55] vol=2.3x ATR=3.26 |
| Stop hit — per-position SL triggered | 2024-02-28 11:15:00 | 1038.56 | 1043.27 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2024-03-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-02 09:45:00 | 1056.90 | 1046.71 | 0.00 | ORB-long ORB[1037.15,1045.95] vol=2.0x ATR=4.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-02 09:50:00 | 1062.97 | 1049.54 | 0.00 | T1 1.5R @ 1062.97 |
| Stop hit — per-position SL triggered | 2024-03-04 09:15:00 | 1080.00 | 0.00 | 0.00 | EOD overnight gap close |

### Cycle 70 — SELL (started 2024-03-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 09:55:00 | 1062.00 | 1070.79 | 0.00 | ORB-short ORB[1066.10,1081.80] vol=2.0x ATR=4.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 10:00:00 | 1055.74 | 1068.15 | 0.00 | T1 1.5R @ 1055.74 |
| Target hit | 2024-03-06 13:40:00 | 1058.35 | 1056.42 | 0.00 | Trail-exit close>VWAP |

### Cycle 71 — SELL (started 2024-03-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-15 11:00:00 | 991.55 | 1001.85 | 0.00 | ORB-short ORB[1002.30,1015.80] vol=3.3x ATR=4.47 |
| Stop hit — per-position SL triggered | 2024-03-15 11:10:00 | 996.02 | 1001.34 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2024-04-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-10 09:40:00 | 1123.10 | 1127.53 | 0.00 | ORB-short ORB[1125.00,1134.90] vol=1.5x ATR=3.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-10 10:35:00 | 1118.33 | 1125.42 | 0.00 | T1 1.5R @ 1118.33 |
| Stop hit — per-position SL triggered | 2024-04-10 14:15:00 | 1123.10 | 1122.44 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2024-04-25 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 10:35:00 | 1095.80 | 1086.80 | 0.00 | ORB-long ORB[1081.05,1095.45] vol=1.8x ATR=3.93 |
| Stop hit — per-position SL triggered | 2024-04-25 10:40:00 | 1091.87 | 1087.20 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2024-04-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-29 09:35:00 | 1146.50 | 1138.67 | 0.00 | ORB-long ORB[1127.25,1139.05] vol=2.4x ATR=3.98 |
| Stop hit — per-position SL triggered | 2024-04-29 09:50:00 | 1142.52 | 1143.06 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2024-05-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-06 10:55:00 | 1160.00 | 1150.96 | 0.00 | ORB-long ORB[1141.60,1155.55] vol=1.9x ATR=3.92 |
| Stop hit — per-position SL triggered | 2024-05-06 11:15:00 | 1156.08 | 1152.30 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2024-05-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-08 10:25:00 | 1123.90 | 1121.36 | 0.00 | ORB-long ORB[1111.25,1123.80] vol=1.7x ATR=4.22 |
| Stop hit — per-position SL triggered | 2024-05-08 10:55:00 | 1119.68 | 1122.21 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2024-05-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-09 09:35:00 | 1140.70 | 1134.65 | 0.00 | ORB-long ORB[1126.60,1139.85] vol=2.4x ATR=4.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 10:10:00 | 1146.99 | 1140.89 | 0.00 | T1 1.5R @ 1146.99 |
| Target hit | 2024-05-09 10:35:00 | 1145.10 | 1145.73 | 0.00 | Trail-exit close<VWAP |

### Cycle 78 — BUY (started 2024-05-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-10 10:00:00 | 1134.15 | 1119.47 | 0.00 | ORB-long ORB[1106.05,1122.55] vol=2.2x ATR=5.94 |
| Stop hit — per-position SL triggered | 2024-05-10 14:20:00 | 1128.21 | 1130.46 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-12 11:15:00 | 611.00 | 2023-05-12 12:25:00 | 614.25 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2023-05-12 11:15:00 | 611.00 | 2023-05-12 13:55:00 | 611.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-15 10:25:00 | 610.60 | 2023-05-15 11:40:00 | 613.31 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-05-15 10:25:00 | 610.60 | 2023-05-15 13:05:00 | 610.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-05-18 09:35:00 | 613.05 | 2023-05-18 09:45:00 | 614.90 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-06-01 11:15:00 | 649.60 | 2023-06-01 13:05:00 | 651.02 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-06-06 09:50:00 | 654.45 | 2023-06-06 10:00:00 | 655.98 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-06-14 11:15:00 | 671.60 | 2023-06-14 11:35:00 | 672.92 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-06-19 09:35:00 | 689.00 | 2023-06-19 10:00:00 | 686.92 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-06-21 11:15:00 | 674.80 | 2023-06-21 11:20:00 | 676.02 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2023-06-22 09:55:00 | 685.05 | 2023-06-22 10:00:00 | 683.47 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-06-23 09:45:00 | 673.00 | 2023-06-23 10:15:00 | 675.98 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-06-23 09:45:00 | 673.00 | 2023-06-23 15:20:00 | 700.00 | TARGET_HIT | 0.50 | 4.01% |
| BUY | retest1 | 2023-07-05 09:50:00 | 723.35 | 2023-07-05 09:55:00 | 726.58 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2023-07-05 09:50:00 | 723.35 | 2023-07-05 10:00:00 | 723.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-06 09:45:00 | 735.95 | 2023-07-06 09:55:00 | 739.31 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2023-07-06 09:45:00 | 735.95 | 2023-07-06 15:20:00 | 762.45 | TARGET_HIT | 0.50 | 3.60% |
| BUY | retest1 | 2023-07-12 09:45:00 | 741.15 | 2023-07-12 09:50:00 | 738.83 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-07-14 09:35:00 | 732.00 | 2023-07-14 09:55:00 | 735.33 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2023-07-14 09:35:00 | 732.00 | 2023-07-14 11:45:00 | 736.80 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2023-07-19 09:40:00 | 764.45 | 2023-07-19 09:45:00 | 768.42 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2023-07-19 09:40:00 | 764.45 | 2023-07-19 10:25:00 | 764.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-20 10:55:00 | 760.00 | 2023-07-20 11:25:00 | 763.40 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2023-07-20 10:55:00 | 760.00 | 2023-07-20 15:20:00 | 773.00 | TARGET_HIT | 0.50 | 1.71% |
| BUY | retest1 | 2023-07-21 09:40:00 | 778.75 | 2023-07-21 09:50:00 | 781.71 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2023-07-21 09:40:00 | 778.75 | 2023-07-21 11:10:00 | 781.55 | TARGET_HIT | 0.50 | 0.36% |
| BUY | retest1 | 2023-07-24 09:40:00 | 781.35 | 2023-07-24 10:00:00 | 785.13 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2023-07-24 09:40:00 | 781.35 | 2023-07-24 15:00:00 | 785.90 | TARGET_HIT | 0.50 | 0.58% |
| BUY | retest1 | 2023-07-25 09:50:00 | 789.45 | 2023-07-25 09:55:00 | 787.39 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-07-26 10:05:00 | 791.50 | 2023-07-26 10:10:00 | 794.57 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2023-07-26 10:05:00 | 791.50 | 2023-07-26 10:20:00 | 791.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-18 09:30:00 | 877.70 | 2023-08-18 09:35:00 | 875.19 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-08-24 11:10:00 | 846.75 | 2023-08-24 11:15:00 | 844.83 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-08-25 10:15:00 | 832.35 | 2023-08-25 10:20:00 | 834.71 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-09-01 10:50:00 | 835.05 | 2023-09-01 11:05:00 | 838.04 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-09-01 10:50:00 | 835.05 | 2023-09-01 11:15:00 | 835.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-05 09:45:00 | 834.35 | 2023-09-05 09:55:00 | 831.89 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-09-06 09:40:00 | 863.30 | 2023-09-06 10:20:00 | 868.28 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2023-09-06 09:40:00 | 863.30 | 2023-09-06 10:50:00 | 863.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-08 11:00:00 | 852.90 | 2023-09-08 11:20:00 | 854.73 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-09-11 09:35:00 | 867.80 | 2023-09-11 09:55:00 | 865.61 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-09-13 09:30:00 | 887.90 | 2023-09-13 09:35:00 | 884.56 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2023-09-14 11:05:00 | 891.80 | 2023-09-14 11:25:00 | 887.68 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2023-09-14 11:05:00 | 891.80 | 2023-09-14 12:55:00 | 891.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-20 09:50:00 | 895.15 | 2023-09-20 10:05:00 | 899.01 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2023-09-20 09:50:00 | 895.15 | 2023-09-20 10:10:00 | 895.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-27 09:40:00 | 865.70 | 2023-09-27 09:45:00 | 863.04 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-10-06 09:45:00 | 889.00 | 2023-10-06 10:20:00 | 892.10 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-10-06 09:45:00 | 889.00 | 2023-10-06 10:45:00 | 889.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-09 09:30:00 | 907.20 | 2023-10-09 09:35:00 | 912.46 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2023-10-09 09:30:00 | 907.20 | 2023-10-09 09:40:00 | 907.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-11 09:35:00 | 899.20 | 2023-10-11 09:40:00 | 896.63 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2023-10-11 09:35:00 | 899.20 | 2023-10-11 09:55:00 | 899.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-12 09:30:00 | 911.80 | 2023-10-12 09:40:00 | 914.62 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-10-13 10:55:00 | 914.80 | 2023-10-13 11:55:00 | 918.19 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2023-10-13 10:55:00 | 914.80 | 2023-10-13 15:20:00 | 920.25 | TARGET_HIT | 0.50 | 0.60% |
| BUY | retest1 | 2023-10-17 10:50:00 | 918.85 | 2023-10-17 10:55:00 | 917.27 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2023-10-18 09:45:00 | 902.75 | 2023-10-18 09:50:00 | 899.40 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2023-10-18 09:45:00 | 902.75 | 2023-10-18 10:00:00 | 902.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-10-19 11:05:00 | 891.75 | 2023-10-19 11:20:00 | 888.61 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2023-10-19 11:05:00 | 891.75 | 2023-10-19 12:20:00 | 891.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-10-27 11:00:00 | 863.90 | 2023-10-27 11:05:00 | 861.84 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-10-30 10:15:00 | 863.75 | 2023-10-30 10:25:00 | 868.08 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2023-10-30 10:15:00 | 863.75 | 2023-10-30 11:10:00 | 863.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-01 10:10:00 | 836.05 | 2023-11-01 10:20:00 | 839.04 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-11-02 10:10:00 | 870.55 | 2023-11-02 10:30:00 | 867.68 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-11-08 09:55:00 | 924.90 | 2023-11-08 10:05:00 | 930.71 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2023-11-08 09:55:00 | 924.90 | 2023-11-08 13:50:00 | 931.75 | TARGET_HIT | 0.50 | 0.74% |
| BUY | retest1 | 2023-11-09 11:05:00 | 951.00 | 2023-11-09 11:20:00 | 955.61 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2023-11-09 11:05:00 | 951.00 | 2023-11-09 12:05:00 | 951.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-22 10:35:00 | 1038.45 | 2023-11-22 11:55:00 | 1043.92 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2023-11-22 10:35:00 | 1038.45 | 2023-11-22 12:25:00 | 1038.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-23 09:40:00 | 1037.55 | 2023-11-23 10:10:00 | 1041.22 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2023-11-29 11:15:00 | 1023.30 | 2023-11-29 11:30:00 | 1020.23 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-12-04 09:50:00 | 1026.50 | 2023-12-04 10:35:00 | 1029.91 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-12-05 10:40:00 | 1030.00 | 2023-12-05 11:00:00 | 1025.55 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2023-12-05 10:40:00 | 1030.00 | 2023-12-05 15:00:00 | 1026.20 | TARGET_HIT | 0.50 | 0.37% |
| BUY | retest1 | 2023-12-07 09:55:00 | 1020.45 | 2023-12-07 10:15:00 | 1025.11 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2023-12-07 09:55:00 | 1020.45 | 2023-12-07 10:20:00 | 1020.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-11 10:55:00 | 998.25 | 2023-12-11 11:15:00 | 1001.91 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-12-13 09:30:00 | 1027.00 | 2023-12-13 09:45:00 | 1023.60 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-12-15 09:55:00 | 1039.00 | 2023-12-15 10:45:00 | 1034.38 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2023-12-15 09:55:00 | 1039.00 | 2023-12-15 15:20:00 | 1024.55 | TARGET_HIT | 0.50 | 1.39% |
| BUY | retest1 | 2023-12-19 11:05:00 | 1032.30 | 2023-12-19 11:50:00 | 1029.19 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-12-27 09:45:00 | 1077.75 | 2023-12-27 09:55:00 | 1074.14 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2023-12-28 11:10:00 | 1057.30 | 2023-12-28 11:20:00 | 1060.70 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-01-03 09:35:00 | 1080.95 | 2024-01-03 09:40:00 | 1084.70 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-01-04 10:25:00 | 1103.75 | 2024-01-04 10:50:00 | 1100.34 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-01-09 09:55:00 | 1119.95 | 2024-01-09 10:00:00 | 1123.66 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-01-12 11:05:00 | 1084.35 | 2024-01-12 11:20:00 | 1087.45 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-01-15 09:40:00 | 1117.95 | 2024-01-15 09:45:00 | 1123.73 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-01-15 09:40:00 | 1117.95 | 2024-01-15 15:20:00 | 1150.25 | TARGET_HIT | 0.50 | 2.89% |
| BUY | retest1 | 2024-01-20 10:40:00 | 1157.25 | 2024-01-20 10:45:00 | 1151.90 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-02-07 09:40:00 | 1002.05 | 2024-02-07 09:50:00 | 993.94 | PARTIAL | 0.50 | 0.81% |
| SELL | retest1 | 2024-02-07 09:40:00 | 1002.05 | 2024-02-07 09:55:00 | 1002.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-20 10:55:00 | 1045.20 | 2024-02-20 11:00:00 | 1041.94 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-02-26 09:50:00 | 1027.35 | 2024-02-26 09:55:00 | 1020.63 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2024-02-26 09:50:00 | 1027.35 | 2024-02-26 10:05:00 | 1027.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-28 10:55:00 | 1035.30 | 2024-02-28 11:15:00 | 1038.56 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-03-02 09:45:00 | 1056.90 | 2024-03-02 09:50:00 | 1062.97 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-03-02 09:45:00 | 1056.90 | 2024-03-04 09:15:00 | 1080.00 | STOP_HIT | 0.50 | 2.19% |
| SELL | retest1 | 2024-03-06 09:55:00 | 1062.00 | 2024-03-06 10:00:00 | 1055.74 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-03-06 09:55:00 | 1062.00 | 2024-03-06 13:40:00 | 1058.35 | TARGET_HIT | 0.50 | 0.34% |
| SELL | retest1 | 2024-03-15 11:00:00 | 991.55 | 2024-03-15 11:10:00 | 996.02 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-04-10 09:40:00 | 1123.10 | 2024-04-10 10:35:00 | 1118.33 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-04-10 09:40:00 | 1123.10 | 2024-04-10 14:15:00 | 1123.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-25 10:35:00 | 1095.80 | 2024-04-25 10:40:00 | 1091.87 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-04-29 09:35:00 | 1146.50 | 2024-04-29 09:50:00 | 1142.52 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-05-06 10:55:00 | 1160.00 | 2024-05-06 11:15:00 | 1156.08 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-05-08 10:25:00 | 1123.90 | 2024-05-08 10:55:00 | 1119.68 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-05-09 09:35:00 | 1140.70 | 2024-05-09 10:10:00 | 1146.99 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-05-09 09:35:00 | 1140.70 | 2024-05-09 10:35:00 | 1145.10 | TARGET_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2024-05-10 10:00:00 | 1134.15 | 2024-05-10 14:20:00 | 1128.21 | STOP_HIT | 1.00 | -0.52% |
