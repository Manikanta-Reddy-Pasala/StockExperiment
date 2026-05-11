# JSWSTEEL (JSWSTEEL)

## Backtest Summary

- **Window:** 2023-11-08 09:15:00 → 2026-05-08 15:25:00 (44458 bars)
- **Last close:** 1272.00
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
| ENTRY1 | 36 |
| ENTRY2 | 0 |
| PARTIAL | 18 |
| TARGET_HIT | 3 |
| STOP_HIT | 33 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 54 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 33
- **Target hits / Stop hits / Partials:** 3 / 33 / 18
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 5.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 35 | 13 | 37.1% | 1 | 22 | 12 | 0.12% | 4.1% |
| BUY @ 2nd Alert (retest1) | 35 | 13 | 37.1% | 1 | 22 | 12 | 0.12% | 4.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 19 | 8 | 42.1% | 2 | 11 | 6 | 0.06% | 1.1% |
| SELL @ 2nd Alert (retest1) | 19 | 8 | 42.1% | 2 | 11 | 6 | 0.06% | 1.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 54 | 21 | 38.9% | 3 | 33 | 18 | 0.10% | 5.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-08 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-08 10:35:00 | 752.90 | 750.26 | 0.00 | ORB-long ORB[745.05,752.75] vol=4.3x ATR=3.53 |
| Stop hit — per-position SL triggered | 2023-11-08 11:10:00 | 749.37 | 750.16 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-11-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-09 10:25:00 | 751.85 | 753.62 | 0.00 | ORB-short ORB[756.10,759.90] vol=2.3x ATR=1.61 |
| Stop hit — per-position SL triggered | 2023-11-09 10:50:00 | 753.46 | 753.07 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2023-11-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-13 09:55:00 | 759.30 | 755.79 | 0.00 | ORB-long ORB[753.00,757.55] vol=2.6x ATR=1.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-13 10:00:00 | 762.00 | 757.87 | 0.00 | T1 1.5R @ 762.00 |
| Stop hit — per-position SL triggered | 2023-11-13 10:10:00 | 759.30 | 758.19 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2023-11-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-15 09:35:00 | 777.90 | 774.87 | 0.00 | ORB-long ORB[769.00,777.50] vol=1.6x ATR=2.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-15 09:45:00 | 782.09 | 776.55 | 0.00 | T1 1.5R @ 782.09 |
| Stop hit — per-position SL triggered | 2023-11-15 10:05:00 | 777.90 | 777.04 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2023-11-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-20 11:10:00 | 766.75 | 769.12 | 0.00 | ORB-short ORB[767.65,772.00] vol=1.6x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-20 11:40:00 | 764.85 | 768.58 | 0.00 | T1 1.5R @ 764.85 |
| Stop hit — per-position SL triggered | 2023-11-20 12:05:00 | 766.75 | 767.85 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-11-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 09:30:00 | 776.00 | 772.19 | 0.00 | ORB-long ORB[765.95,774.30] vol=1.8x ATR=1.97 |
| Stop hit — per-position SL triggered | 2023-11-21 09:35:00 | 774.03 | 772.64 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-11-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-24 10:10:00 | 783.20 | 779.13 | 0.00 | ORB-long ORB[776.40,779.90] vol=3.0x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-24 10:15:00 | 785.84 | 780.69 | 0.00 | T1 1.5R @ 785.84 |
| Stop hit — per-position SL triggered | 2023-11-24 10:25:00 | 783.20 | 781.28 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-11-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-29 09:50:00 | 786.25 | 785.12 | 0.00 | ORB-long ORB[783.20,786.00] vol=1.5x ATR=1.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-29 10:20:00 | 788.11 | 785.84 | 0.00 | T1 1.5R @ 788.11 |
| Stop hit — per-position SL triggered | 2023-11-29 10:50:00 | 786.25 | 786.77 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-11-30 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-30 10:20:00 | 797.35 | 793.12 | 0.00 | ORB-long ORB[789.60,794.65] vol=2.1x ATR=1.81 |
| Stop hit — per-position SL triggered | 2023-11-30 12:30:00 | 795.54 | 796.06 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-12-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-05 10:50:00 | 823.25 | 820.20 | 0.00 | ORB-long ORB[814.20,819.80] vol=1.7x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-05 11:00:00 | 825.62 | 820.79 | 0.00 | T1 1.5R @ 825.62 |
| Stop hit — per-position SL triggered | 2023-12-05 11:10:00 | 823.25 | 820.98 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-12-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-12 11:05:00 | 859.20 | 851.96 | 0.00 | ORB-long ORB[846.60,855.00] vol=2.6x ATR=2.67 |
| Stop hit — per-position SL triggered | 2023-12-12 11:15:00 | 856.53 | 853.30 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-12-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-13 09:55:00 | 852.05 | 854.88 | 0.00 | ORB-short ORB[854.00,859.10] vol=1.5x ATR=2.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-13 10:20:00 | 848.27 | 853.49 | 0.00 | T1 1.5R @ 848.27 |
| Target hit | 2023-12-13 14:30:00 | 847.70 | 847.12 | 0.00 | Trail-exit close>VWAP |

### Cycle 13 — BUY (started 2023-12-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 09:40:00 | 866.45 | 862.07 | 0.00 | ORB-long ORB[855.65,861.30] vol=2.8x ATR=2.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-27 09:50:00 | 869.92 | 866.27 | 0.00 | T1 1.5R @ 869.92 |
| Stop hit — per-position SL triggered | 2023-12-27 10:25:00 | 866.45 | 867.18 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2023-12-29 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-29 10:50:00 | 876.35 | 877.24 | 0.00 | ORB-short ORB[876.55,884.55] vol=2.0x ATR=2.59 |
| Stop hit — per-position SL triggered | 2023-12-29 12:45:00 | 878.94 | 877.05 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-01-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 09:55:00 | 874.45 | 878.92 | 0.00 | ORB-short ORB[876.25,883.85] vol=2.1x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-02 10:20:00 | 870.15 | 877.41 | 0.00 | T1 1.5R @ 870.15 |
| Target hit | 2024-01-02 14:35:00 | 870.75 | 869.17 | 0.00 | Trail-exit close>VWAP |

### Cycle 16 — BUY (started 2024-01-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 09:55:00 | 840.55 | 837.52 | 0.00 | ORB-long ORB[835.60,840.15] vol=1.8x ATR=2.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-05 10:15:00 | 843.57 | 838.75 | 0.00 | T1 1.5R @ 843.57 |
| Stop hit — per-position SL triggered | 2024-01-05 10:30:00 | 840.55 | 839.14 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-01-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-11 11:15:00 | 825.60 | 829.65 | 0.00 | ORB-short ORB[830.35,836.75] vol=3.7x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-11 12:00:00 | 822.95 | 828.62 | 0.00 | T1 1.5R @ 822.95 |
| Stop hit — per-position SL triggered | 2024-01-11 12:35:00 | 825.60 | 827.85 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-01-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-18 09:35:00 | 799.00 | 806.80 | 0.00 | ORB-short ORB[803.85,814.50] vol=2.1x ATR=2.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-18 09:55:00 | 795.17 | 803.54 | 0.00 | T1 1.5R @ 795.17 |
| Stop hit — per-position SL triggered | 2024-01-18 10:10:00 | 799.00 | 802.20 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-02-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 11:10:00 | 814.95 | 810.50 | 0.00 | ORB-long ORB[805.10,814.05] vol=1.7x ATR=2.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 11:20:00 | 818.08 | 811.13 | 0.00 | T1 1.5R @ 818.08 |
| Stop hit — per-position SL triggered | 2024-02-02 12:30:00 | 814.95 | 814.30 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-03-13 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-13 10:35:00 | 819.85 | 812.66 | 0.00 | ORB-long ORB[809.35,819.25] vol=2.3x ATR=3.12 |
| Stop hit — per-position SL triggered | 2024-03-13 10:55:00 | 816.73 | 813.32 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 11:15:00 | 797.15 | 803.25 | 0.00 | ORB-short ORB[799.00,809.95] vol=2.0x ATR=2.55 |
| Stop hit — per-position SL triggered | 2024-03-19 11:25:00 | 799.70 | 803.06 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 11:15:00 | 837.65 | 826.49 | 0.00 | ORB-long ORB[820.05,827.85] vol=2.0x ATR=2.42 |
| Stop hit — per-position SL triggered | 2024-03-28 11:25:00 | 835.23 | 827.03 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-04-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-03 10:45:00 | 869.40 | 873.26 | 0.00 | ORB-short ORB[870.70,879.00] vol=1.8x ATR=2.85 |
| Stop hit — per-position SL triggered | 2024-04-03 11:05:00 | 872.25 | 872.96 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-04-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 09:50:00 | 866.05 | 871.21 | 0.00 | ORB-short ORB[870.05,878.65] vol=1.7x ATR=2.96 |
| Stop hit — per-position SL triggered | 2024-04-04 09:55:00 | 869.01 | 870.74 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-04-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-08 11:05:00 | 867.00 | 864.04 | 0.00 | ORB-long ORB[860.25,866.55] vol=1.7x ATR=1.88 |
| Stop hit — per-position SL triggered | 2024-04-08 11:30:00 | 865.12 | 864.44 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-04-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-09 09:55:00 | 881.85 | 875.68 | 0.00 | ORB-long ORB[871.50,881.80] vol=2.0x ATR=2.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-09 10:00:00 | 886.07 | 877.96 | 0.00 | T1 1.5R @ 886.07 |
| Stop hit — per-position SL triggered | 2024-04-09 10:05:00 | 881.85 | 878.32 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-04-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-12 09:40:00 | 865.15 | 872.45 | 0.00 | ORB-short ORB[871.15,883.90] vol=2.8x ATR=3.20 |
| Stop hit — per-position SL triggered | 2024-04-12 09:50:00 | 868.35 | 871.09 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-04-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-15 10:35:00 | 879.00 | 864.99 | 0.00 | ORB-long ORB[850.25,862.90] vol=1.8x ATR=3.64 |
| Stop hit — per-position SL triggered | 2024-04-15 10:50:00 | 875.36 | 868.91 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-16 09:30:00 | 863.00 | 858.61 | 0.00 | ORB-long ORB[853.20,860.00] vol=2.4x ATR=2.58 |
| Stop hit — per-position SL triggered | 2024-04-16 09:40:00 | 860.42 | 858.93 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-04-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-18 11:00:00 | 852.25 | 848.30 | 0.00 | ORB-long ORB[845.25,851.95] vol=1.5x ATR=2.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-18 11:40:00 | 856.20 | 849.03 | 0.00 | T1 1.5R @ 856.20 |
| Stop hit — per-position SL triggered | 2024-04-18 13:25:00 | 852.25 | 850.64 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-04-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 10:25:00 | 860.10 | 857.06 | 0.00 | ORB-long ORB[852.10,859.90] vol=1.6x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-23 10:55:00 | 863.90 | 857.73 | 0.00 | T1 1.5R @ 863.90 |
| Stop hit — per-position SL triggered | 2024-04-23 11:30:00 | 860.10 | 858.23 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 09:30:00 | 859.60 | 857.37 | 0.00 | ORB-long ORB[850.00,859.05] vol=1.8x ATR=2.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-24 10:00:00 | 863.23 | 860.17 | 0.00 | T1 1.5R @ 863.23 |
| Target hit | 2024-04-24 15:20:00 | 885.10 | 876.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — SELL (started 2024-04-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-30 11:05:00 | 891.00 | 894.10 | 0.00 | ORB-short ORB[894.40,902.75] vol=2.1x ATR=2.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-30 11:45:00 | 887.60 | 893.47 | 0.00 | T1 1.5R @ 887.60 |
| Stop hit — per-position SL triggered | 2024-04-30 12:00:00 | 891.00 | 893.23 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-05-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-03 09:35:00 | 900.05 | 899.01 | 0.00 | ORB-long ORB[892.05,899.00] vol=2.3x ATR=2.75 |
| Stop hit — per-position SL triggered | 2024-05-03 10:05:00 | 897.30 | 899.36 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2024-05-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-07 10:50:00 | 860.10 | 870.96 | 0.00 | ORB-short ORB[871.55,881.05] vol=2.0x ATR=2.67 |
| Stop hit — per-position SL triggered | 2024-05-07 10:55:00 | 862.77 | 870.57 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-05-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-08 09:45:00 | 863.95 | 859.45 | 0.00 | ORB-long ORB[850.40,861.25] vol=1.9x ATR=3.20 |
| Stop hit — per-position SL triggered | 2024-05-08 10:30:00 | 860.75 | 861.25 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-11-08 10:35:00 | 752.90 | 2023-11-08 11:10:00 | 749.37 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2023-11-09 10:25:00 | 751.85 | 2023-11-09 10:50:00 | 753.46 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2023-11-13 09:55:00 | 759.30 | 2023-11-13 10:00:00 | 762.00 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-11-13 09:55:00 | 759.30 | 2023-11-13 10:10:00 | 759.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-15 09:35:00 | 777.90 | 2023-11-15 09:45:00 | 782.09 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2023-11-15 09:35:00 | 777.90 | 2023-11-15 10:05:00 | 777.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-20 11:10:00 | 766.75 | 2023-11-20 11:40:00 | 764.85 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2023-11-20 11:10:00 | 766.75 | 2023-11-20 12:05:00 | 766.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-21 09:30:00 | 776.00 | 2023-11-21 09:35:00 | 774.03 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-11-24 10:10:00 | 783.20 | 2023-11-24 10:15:00 | 785.84 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2023-11-24 10:10:00 | 783.20 | 2023-11-24 10:25:00 | 783.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-29 09:50:00 | 786.25 | 2023-11-29 10:20:00 | 788.11 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2023-11-29 09:50:00 | 786.25 | 2023-11-29 10:50:00 | 786.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-30 10:20:00 | 797.35 | 2023-11-30 12:30:00 | 795.54 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2023-12-05 10:50:00 | 823.25 | 2023-12-05 11:00:00 | 825.62 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2023-12-05 10:50:00 | 823.25 | 2023-12-05 11:10:00 | 823.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-12 11:05:00 | 859.20 | 2023-12-12 11:15:00 | 856.53 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-12-13 09:55:00 | 852.05 | 2023-12-13 10:20:00 | 848.27 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2023-12-13 09:55:00 | 852.05 | 2023-12-13 14:30:00 | 847.70 | TARGET_HIT | 0.50 | 0.51% |
| BUY | retest1 | 2023-12-27 09:40:00 | 866.45 | 2023-12-27 09:50:00 | 869.92 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2023-12-27 09:40:00 | 866.45 | 2023-12-27 10:25:00 | 866.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-29 10:50:00 | 876.35 | 2023-12-29 12:45:00 | 878.94 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-01-02 09:55:00 | 874.45 | 2024-01-02 10:20:00 | 870.15 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-01-02 09:55:00 | 874.45 | 2024-01-02 14:35:00 | 870.75 | TARGET_HIT | 0.50 | 0.42% |
| BUY | retest1 | 2024-01-05 09:55:00 | 840.55 | 2024-01-05 10:15:00 | 843.57 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-01-05 09:55:00 | 840.55 | 2024-01-05 10:30:00 | 840.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-11 11:15:00 | 825.60 | 2024-01-11 12:00:00 | 822.95 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-01-11 11:15:00 | 825.60 | 2024-01-11 12:35:00 | 825.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-18 09:35:00 | 799.00 | 2024-01-18 09:55:00 | 795.17 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-01-18 09:35:00 | 799.00 | 2024-01-18 10:10:00 | 799.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-02 11:10:00 | 814.95 | 2024-02-02 11:20:00 | 818.08 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-02-02 11:10:00 | 814.95 | 2024-02-02 12:30:00 | 814.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-13 10:35:00 | 819.85 | 2024-03-13 10:55:00 | 816.73 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-03-19 11:15:00 | 797.15 | 2024-03-19 11:25:00 | 799.70 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-03-28 11:15:00 | 837.65 | 2024-03-28 11:25:00 | 835.23 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-04-03 10:45:00 | 869.40 | 2024-04-03 11:05:00 | 872.25 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-04-04 09:50:00 | 866.05 | 2024-04-04 09:55:00 | 869.01 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-04-08 11:05:00 | 867.00 | 2024-04-08 11:30:00 | 865.12 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-04-09 09:55:00 | 881.85 | 2024-04-09 10:00:00 | 886.07 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-04-09 09:55:00 | 881.85 | 2024-04-09 10:05:00 | 881.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-04-12 09:40:00 | 865.15 | 2024-04-12 09:50:00 | 868.35 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-04-15 10:35:00 | 879.00 | 2024-04-15 10:50:00 | 875.36 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-04-16 09:30:00 | 863.00 | 2024-04-16 09:40:00 | 860.42 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-04-18 11:00:00 | 852.25 | 2024-04-18 11:40:00 | 856.20 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-04-18 11:00:00 | 852.25 | 2024-04-18 13:25:00 | 852.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-23 10:25:00 | 860.10 | 2024-04-23 10:55:00 | 863.90 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2024-04-23 10:25:00 | 860.10 | 2024-04-23 11:30:00 | 860.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-24 09:30:00 | 859.60 | 2024-04-24 10:00:00 | 863.23 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-04-24 09:30:00 | 859.60 | 2024-04-24 15:20:00 | 885.10 | TARGET_HIT | 0.50 | 2.97% |
| SELL | retest1 | 2024-04-30 11:05:00 | 891.00 | 2024-04-30 11:45:00 | 887.60 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-04-30 11:05:00 | 891.00 | 2024-04-30 12:00:00 | 891.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-03 09:35:00 | 900.05 | 2024-05-03 10:05:00 | 897.30 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-05-07 10:50:00 | 860.10 | 2024-05-07 10:55:00 | 862.77 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-05-08 09:45:00 | 863.95 | 2024-05-08 10:30:00 | 860.75 | STOP_HIT | 1.00 | -0.37% |
