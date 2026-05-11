# Chalet Hotels Ltd. (CHALET)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 787.00
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
| ENTRY1 | 47 |
| ENTRY2 | 0 |
| PARTIAL | 22 |
| TARGET_HIT | 8 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 69 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 39
- **Target hits / Stop hits / Partials:** 8 / 39 / 22
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 12.23%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 10 | 38.5% | 2 | 16 | 8 | 0.26% | 6.9% |
| BUY @ 2nd Alert (retest1) | 26 | 10 | 38.5% | 2 | 16 | 8 | 0.26% | 6.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 43 | 20 | 46.5% | 6 | 23 | 14 | 0.12% | 5.4% |
| SELL @ 2nd Alert (retest1) | 43 | 20 | 46.5% | 6 | 23 | 14 | 0.12% | 5.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 69 | 30 | 43.5% | 8 | 39 | 22 | 0.18% | 12.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 11:05:00 | 766.70 | 771.79 | 0.00 | ORB-short ORB[770.00,779.65] vol=1.8x ATR=2.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-22 11:25:00 | 762.44 | 771.25 | 0.00 | T1 1.5R @ 762.44 |
| Stop hit — per-position SL triggered | 2024-05-22 14:05:00 | 766.70 | 768.19 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 10:45:00 | 779.05 | 774.86 | 0.00 | ORB-long ORB[767.45,778.75] vol=2.4x ATR=2.67 |
| Stop hit — per-position SL triggered | 2024-05-23 11:05:00 | 776.38 | 775.38 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-29 10:45:00 | 757.95 | 764.01 | 0.00 | ORB-short ORB[762.80,767.25] vol=2.9x ATR=2.11 |
| Stop hit — per-position SL triggered | 2024-05-29 10:50:00 | 760.06 | 763.64 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-06-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 09:40:00 | 779.00 | 774.57 | 0.00 | ORB-long ORB[767.55,777.95] vol=2.2x ATR=3.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-06 09:45:00 | 783.93 | 776.56 | 0.00 | T1 1.5R @ 783.93 |
| Stop hit — per-position SL triggered | 2024-06-06 11:45:00 | 779.00 | 781.13 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-06-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-13 10:50:00 | 842.95 | 848.94 | 0.00 | ORB-short ORB[843.30,853.05] vol=1.6x ATR=2.54 |
| Stop hit — per-position SL triggered | 2024-06-13 12:20:00 | 845.49 | 847.97 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-06-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 09:55:00 | 836.15 | 840.48 | 0.00 | ORB-short ORB[839.60,848.90] vol=1.7x ATR=3.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 12:50:00 | 830.28 | 837.61 | 0.00 | T1 1.5R @ 830.28 |
| Stop hit — per-position SL triggered | 2024-06-25 15:00:00 | 836.15 | 835.85 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-07-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 09:50:00 | 842.50 | 852.20 | 0.00 | ORB-short ORB[852.05,858.00] vol=1.6x ATR=3.55 |
| Stop hit — per-position SL triggered | 2024-07-04 10:10:00 | 846.05 | 851.03 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-07-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 09:35:00 | 851.50 | 854.85 | 0.00 | ORB-short ORB[852.05,858.00] vol=2.5x ATR=2.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 09:50:00 | 847.40 | 852.58 | 0.00 | T1 1.5R @ 847.40 |
| Stop hit — per-position SL triggered | 2024-07-05 10:20:00 | 851.50 | 851.67 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-07-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 11:05:00 | 840.90 | 847.60 | 0.00 | ORB-short ORB[843.00,853.65] vol=2.0x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 13:35:00 | 837.20 | 844.95 | 0.00 | T1 1.5R @ 837.20 |
| Stop hit — per-position SL triggered | 2024-07-11 15:10:00 | 840.90 | 842.26 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-07-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 10:50:00 | 832.45 | 836.24 | 0.00 | ORB-short ORB[833.00,840.40] vol=2.1x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 11:20:00 | 828.75 | 835.18 | 0.00 | T1 1.5R @ 828.75 |
| Target hit | 2024-07-12 15:20:00 | 820.65 | 827.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 11 — SELL (started 2024-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-25 11:15:00 | 825.00 | 833.27 | 0.00 | ORB-short ORB[833.20,845.15] vol=1.6x ATR=4.05 |
| Stop hit — per-position SL triggered | 2024-07-25 11:25:00 | 829.05 | 833.20 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-08-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 10:50:00 | 838.20 | 841.82 | 0.00 | ORB-short ORB[840.00,849.35] vol=2.3x ATR=2.22 |
| Stop hit — per-position SL triggered | 2024-08-01 10:55:00 | 840.42 | 841.72 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-08-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 09:30:00 | 802.90 | 798.06 | 0.00 | ORB-long ORB[790.40,801.05] vol=2.0x ATR=3.65 |
| Stop hit — per-position SL triggered | 2024-08-08 09:50:00 | 799.25 | 799.74 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-08-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 10:50:00 | 779.15 | 788.57 | 0.00 | ORB-short ORB[786.50,797.45] vol=1.5x ATR=3.40 |
| Stop hit — per-position SL triggered | 2024-08-13 13:30:00 | 782.55 | 783.08 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-08-22 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-22 10:05:00 | 786.10 | 784.37 | 0.00 | ORB-long ORB[774.45,784.70] vol=13.2x ATR=2.27 |
| Stop hit — per-position SL triggered | 2024-08-22 10:10:00 | 783.83 | 785.24 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-08-26 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 10:25:00 | 808.00 | 800.05 | 0.00 | ORB-long ORB[794.30,803.90] vol=2.2x ATR=3.42 |
| Stop hit — per-position SL triggered | 2024-08-26 10:35:00 | 804.58 | 800.53 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 10:15:00 | 896.50 | 890.08 | 0.00 | ORB-long ORB[880.40,892.45] vol=2.7x ATR=3.94 |
| Stop hit — per-position SL triggered | 2024-09-10 10:20:00 | 892.56 | 890.71 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-09-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-23 11:00:00 | 881.35 | 885.46 | 0.00 | ORB-short ORB[882.05,894.45] vol=1.6x ATR=3.24 |
| Stop hit — per-position SL triggered | 2024-09-23 11:05:00 | 884.59 | 885.40 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-10-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:40:00 | 850.90 | 858.63 | 0.00 | ORB-short ORB[855.85,866.75] vol=2.1x ATR=3.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:45:00 | 845.56 | 857.34 | 0.00 | T1 1.5R @ 845.56 |
| Stop hit — per-position SL triggered | 2024-10-07 11:15:00 | 850.90 | 854.30 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-10-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-11 09:35:00 | 874.00 | 877.28 | 0.00 | ORB-short ORB[876.05,885.90] vol=4.4x ATR=2.83 |
| Stop hit — per-position SL triggered | 2024-10-11 09:50:00 | 876.83 | 877.00 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-10-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 11:05:00 | 890.00 | 891.47 | 0.00 | ORB-short ORB[891.85,899.80] vol=6.0x ATR=2.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 11:10:00 | 886.70 | 890.53 | 0.00 | T1 1.5R @ 886.70 |
| Target hit | 2024-10-16 13:25:00 | 889.35 | 889.08 | 0.00 | Trail-exit close>VWAP |

### Cycle 22 — SELL (started 2024-10-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:40:00 | 880.10 | 885.10 | 0.00 | ORB-short ORB[883.25,894.90] vol=2.0x ATR=3.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 10:05:00 | 875.03 | 881.16 | 0.00 | T1 1.5R @ 875.03 |
| Stop hit — per-position SL triggered | 2024-10-17 10:15:00 | 880.10 | 880.67 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-10-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-31 09:40:00 | 852.80 | 848.07 | 0.00 | ORB-long ORB[836.60,844.45] vol=1.7x ATR=3.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-31 09:45:00 | 858.20 | 849.68 | 0.00 | T1 1.5R @ 858.20 |
| Target hit | 2024-10-31 11:25:00 | 889.95 | 891.30 | 0.00 | Trail-exit close<VWAP |

### Cycle 24 — BUY (started 2024-11-06 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 10:40:00 | 873.00 | 868.20 | 0.00 | ORB-long ORB[860.00,868.50] vol=2.4x ATR=3.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 10:45:00 | 877.87 | 869.81 | 0.00 | T1 1.5R @ 877.87 |
| Stop hit — per-position SL triggered | 2024-11-06 11:15:00 | 873.00 | 873.42 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-11-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 09:40:00 | 891.85 | 885.10 | 0.00 | ORB-long ORB[876.50,888.95] vol=2.3x ATR=3.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-08 09:50:00 | 897.65 | 888.28 | 0.00 | T1 1.5R @ 897.65 |
| Stop hit — per-position SL triggered | 2024-11-08 10:15:00 | 891.85 | 890.02 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-11-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 09:30:00 | 912.65 | 907.18 | 0.00 | ORB-long ORB[884.85,896.00] vol=7.4x ATR=6.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:35:00 | 922.43 | 913.66 | 0.00 | T1 1.5R @ 922.43 |
| Target hit | 2024-11-11 10:15:00 | 926.70 | 926.84 | 0.00 | Trail-exit close<VWAP |

### Cycle 27 — SELL (started 2024-11-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:30:00 | 840.40 | 847.51 | 0.00 | ORB-short ORB[846.90,858.00] vol=3.7x ATR=3.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:40:00 | 835.28 | 836.72 | 0.00 | T1 1.5R @ 835.28 |
| Target hit | 2024-11-13 09:50:00 | 837.60 | 835.93 | 0.00 | Trail-exit close>VWAP |

### Cycle 28 — SELL (started 2024-11-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 09:35:00 | 835.30 | 839.19 | 0.00 | ORB-short ORB[836.05,844.35] vol=1.9x ATR=4.99 |
| Stop hit — per-position SL triggered | 2024-11-18 09:50:00 | 840.29 | 838.13 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-11-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 09:40:00 | 852.30 | 845.45 | 0.00 | ORB-long ORB[837.00,844.00] vol=2.4x ATR=5.04 |
| Stop hit — per-position SL triggered | 2024-11-19 14:35:00 | 847.26 | 851.06 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-12-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-02 10:40:00 | 884.30 | 889.38 | 0.00 | ORB-short ORB[886.25,898.15] vol=2.1x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-02 11:15:00 | 879.85 | 884.54 | 0.00 | T1 1.5R @ 879.85 |
| Target hit | 2024-12-02 15:20:00 | 877.55 | 881.85 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — BUY (started 2024-12-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 09:45:00 | 904.00 | 901.18 | 0.00 | ORB-long ORB[892.15,903.60] vol=2.7x ATR=3.54 |
| Stop hit — per-position SL triggered | 2024-12-06 09:55:00 | 900.46 | 901.56 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-12-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 10:25:00 | 918.00 | 901.13 | 0.00 | ORB-long ORB[890.65,903.00] vol=3.8x ATR=3.85 |
| Stop hit — per-position SL triggered | 2024-12-09 10:30:00 | 914.15 | 913.92 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-12-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-24 09:35:00 | 952.70 | 957.34 | 0.00 | ORB-short ORB[955.00,967.00] vol=2.4x ATR=5.03 |
| Stop hit — per-position SL triggered | 2024-12-24 09:40:00 | 957.73 | 957.18 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-12-31 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-31 09:50:00 | 973.75 | 975.57 | 0.00 | ORB-short ORB[973.95,986.40] vol=1.9x ATR=4.71 |
| Stop hit — per-position SL triggered | 2024-12-31 09:55:00 | 978.46 | 978.32 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-01-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:50:00 | 988.10 | 984.00 | 0.00 | ORB-long ORB[973.55,982.70] vol=8.0x ATR=3.68 |
| Stop hit — per-position SL triggered | 2025-01-01 11:00:00 | 984.42 | 985.26 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-01-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:35:00 | 931.55 | 938.73 | 0.00 | ORB-short ORB[939.00,949.05] vol=2.3x ATR=3.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 10:45:00 | 927.00 | 934.51 | 0.00 | T1 1.5R @ 927.00 |
| Target hit | 2025-01-09 15:20:00 | 918.00 | 924.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — BUY (started 2025-01-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:45:00 | 787.85 | 780.00 | 0.00 | ORB-long ORB[771.00,779.65] vol=3.3x ATR=3.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 11:25:00 | 792.75 | 784.22 | 0.00 | T1 1.5R @ 792.75 |
| Stop hit — per-position SL triggered | 2025-01-23 11:30:00 | 787.85 | 784.34 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-01-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 09:45:00 | 766.05 | 771.83 | 0.00 | ORB-short ORB[773.00,784.10] vol=3.0x ATR=3.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 09:50:00 | 760.08 | 766.06 | 0.00 | T1 1.5R @ 760.08 |
| Stop hit — per-position SL triggered | 2025-01-24 12:15:00 | 766.05 | 762.83 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-01-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 10:05:00 | 751.25 | 742.53 | 0.00 | ORB-long ORB[731.20,740.25] vol=1.9x ATR=4.21 |
| Stop hit — per-position SL triggered | 2025-01-29 10:10:00 | 747.04 | 743.29 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-02-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 10:45:00 | 766.80 | 770.70 | 0.00 | ORB-short ORB[769.55,774.65] vol=2.2x ATR=2.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 10:50:00 | 763.59 | 770.23 | 0.00 | T1 1.5R @ 763.59 |
| Target hit | 2025-02-01 12:30:00 | 763.35 | 763.21 | 0.00 | Trail-exit close>VWAP |

### Cycle 41 — SELL (started 2025-02-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 09:45:00 | 650.15 | 655.45 | 0.00 | ORB-short ORB[670.95,678.85] vol=3.3x ATR=4.87 |
| Stop hit — per-position SL triggered | 2025-02-18 10:05:00 | 655.02 | 654.47 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 10:15:00 | 702.40 | 712.17 | 0.00 | ORB-short ORB[710.60,721.00] vol=1.6x ATR=3.73 |
| Stop hit — per-position SL triggered | 2025-02-21 10:30:00 | 706.13 | 711.66 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-03-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:40:00 | 816.85 | 809.11 | 0.00 | ORB-long ORB[797.10,808.70] vol=5.8x ATR=3.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 10:00:00 | 822.04 | 815.52 | 0.00 | T1 1.5R @ 822.04 |
| Stop hit — per-position SL triggered | 2025-03-21 10:10:00 | 816.85 | 816.07 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-04-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-16 09:40:00 | 823.00 | 827.99 | 0.00 | ORB-short ORB[825.05,835.00] vol=1.6x ATR=2.88 |
| Stop hit — per-position SL triggered | 2025-04-16 10:05:00 | 825.88 | 827.00 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-04-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 10:00:00 | 823.40 | 829.48 | 0.00 | ORB-short ORB[828.00,833.95] vol=1.5x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-23 10:10:00 | 819.69 | 827.37 | 0.00 | T1 1.5R @ 819.69 |
| Stop hit — per-position SL triggered | 2025-04-23 11:40:00 | 823.40 | 824.12 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-04-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:55:00 | 812.00 | 818.74 | 0.00 | ORB-short ORB[816.65,827.40] vol=2.6x ATR=3.72 |
| Stop hit — per-position SL triggered | 2025-04-25 10:10:00 | 815.72 | 817.71 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-05-08 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 10:55:00 | 820.25 | 817.48 | 0.00 | ORB-long ORB[809.75,819.80] vol=1.9x ATR=2.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 11:05:00 | 824.03 | 818.53 | 0.00 | T1 1.5R @ 824.03 |
| Stop hit — per-position SL triggered | 2025-05-08 13:05:00 | 820.25 | 821.00 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-22 11:05:00 | 766.70 | 2024-05-22 11:25:00 | 762.44 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-05-22 11:05:00 | 766.70 | 2024-05-22 14:05:00 | 766.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-23 10:45:00 | 779.05 | 2024-05-23 11:05:00 | 776.38 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-05-29 10:45:00 | 757.95 | 2024-05-29 10:50:00 | 760.06 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-06-06 09:40:00 | 779.00 | 2024-06-06 09:45:00 | 783.93 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-06-06 09:40:00 | 779.00 | 2024-06-06 11:45:00 | 779.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-13 10:50:00 | 842.95 | 2024-06-13 12:20:00 | 845.49 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-06-25 09:55:00 | 836.15 | 2024-06-25 12:50:00 | 830.28 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2024-06-25 09:55:00 | 836.15 | 2024-06-25 15:00:00 | 836.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-04 09:50:00 | 842.50 | 2024-07-04 10:10:00 | 846.05 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-07-05 09:35:00 | 851.50 | 2024-07-05 09:50:00 | 847.40 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-07-05 09:35:00 | 851.50 | 2024-07-05 10:20:00 | 851.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-11 11:05:00 | 840.90 | 2024-07-11 13:35:00 | 837.20 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-07-11 11:05:00 | 840.90 | 2024-07-11 15:10:00 | 840.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-12 10:50:00 | 832.45 | 2024-07-12 11:20:00 | 828.75 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-07-12 10:50:00 | 832.45 | 2024-07-12 15:20:00 | 820.65 | TARGET_HIT | 0.50 | 1.42% |
| SELL | retest1 | 2024-07-25 11:15:00 | 825.00 | 2024-07-25 11:25:00 | 829.05 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-08-01 10:50:00 | 838.20 | 2024-08-01 10:55:00 | 840.42 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-08-08 09:30:00 | 802.90 | 2024-08-08 09:50:00 | 799.25 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-08-13 10:50:00 | 779.15 | 2024-08-13 13:30:00 | 782.55 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-08-22 10:05:00 | 786.10 | 2024-08-22 10:10:00 | 783.83 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-08-26 10:25:00 | 808.00 | 2024-08-26 10:35:00 | 804.58 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-09-10 10:15:00 | 896.50 | 2024-09-10 10:20:00 | 892.56 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-09-23 11:00:00 | 881.35 | 2024-09-23 11:05:00 | 884.59 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-10-07 10:40:00 | 850.90 | 2024-10-07 10:45:00 | 845.56 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-10-07 10:40:00 | 850.90 | 2024-10-07 11:15:00 | 850.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-11 09:35:00 | 874.00 | 2024-10-11 09:50:00 | 876.83 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-10-16 11:05:00 | 890.00 | 2024-10-16 11:10:00 | 886.70 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2024-10-16 11:05:00 | 890.00 | 2024-10-16 13:25:00 | 889.35 | TARGET_HIT | 0.50 | 0.07% |
| SELL | retest1 | 2024-10-17 09:40:00 | 880.10 | 2024-10-17 10:05:00 | 875.03 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-10-17 09:40:00 | 880.10 | 2024-10-17 10:15:00 | 880.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-31 09:40:00 | 852.80 | 2024-10-31 09:45:00 | 858.20 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-10-31 09:40:00 | 852.80 | 2024-10-31 11:25:00 | 889.95 | TARGET_HIT | 0.50 | 4.36% |
| BUY | retest1 | 2024-11-06 10:40:00 | 873.00 | 2024-11-06 10:45:00 | 877.87 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-11-06 10:40:00 | 873.00 | 2024-11-06 11:15:00 | 873.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-08 09:40:00 | 891.85 | 2024-11-08 09:50:00 | 897.65 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-11-08 09:40:00 | 891.85 | 2024-11-08 10:15:00 | 891.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-11 09:30:00 | 912.65 | 2024-11-11 09:35:00 | 922.43 | PARTIAL | 0.50 | 1.07% |
| BUY | retest1 | 2024-11-11 09:30:00 | 912.65 | 2024-11-11 10:15:00 | 926.70 | TARGET_HIT | 0.50 | 1.54% |
| SELL | retest1 | 2024-11-13 09:30:00 | 840.40 | 2024-11-13 09:40:00 | 835.28 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-11-13 09:30:00 | 840.40 | 2024-11-13 09:50:00 | 837.60 | TARGET_HIT | 0.50 | 0.33% |
| SELL | retest1 | 2024-11-18 09:35:00 | 835.30 | 2024-11-18 09:50:00 | 840.29 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest1 | 2024-11-19 09:40:00 | 852.30 | 2024-11-19 14:35:00 | 847.26 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest1 | 2024-12-02 10:40:00 | 884.30 | 2024-12-02 11:15:00 | 879.85 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-12-02 10:40:00 | 884.30 | 2024-12-02 15:20:00 | 877.55 | TARGET_HIT | 0.50 | 0.76% |
| BUY | retest1 | 2024-12-06 09:45:00 | 904.00 | 2024-12-06 09:55:00 | 900.46 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-12-09 10:25:00 | 918.00 | 2024-12-09 10:30:00 | 914.15 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-12-24 09:35:00 | 952.70 | 2024-12-24 09:40:00 | 957.73 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2024-12-31 09:50:00 | 973.75 | 2024-12-31 09:55:00 | 978.46 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-01-01 10:50:00 | 988.10 | 2025-01-01 11:00:00 | 984.42 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-01-09 10:35:00 | 931.55 | 2025-01-09 10:45:00 | 927.00 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-01-09 10:35:00 | 931.55 | 2025-01-09 15:20:00 | 918.00 | TARGET_HIT | 0.50 | 1.45% |
| BUY | retest1 | 2025-01-23 10:45:00 | 787.85 | 2025-01-23 11:25:00 | 792.75 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-01-23 10:45:00 | 787.85 | 2025-01-23 11:30:00 | 787.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-24 09:45:00 | 766.05 | 2025-01-24 09:50:00 | 760.08 | PARTIAL | 0.50 | 0.78% |
| SELL | retest1 | 2025-01-24 09:45:00 | 766.05 | 2025-01-24 12:15:00 | 766.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-29 10:05:00 | 751.25 | 2025-01-29 10:10:00 | 747.04 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2025-02-01 10:45:00 | 766.80 | 2025-02-01 10:50:00 | 763.59 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-02-01 10:45:00 | 766.80 | 2025-02-01 12:30:00 | 763.35 | TARGET_HIT | 0.50 | 0.45% |
| SELL | retest1 | 2025-02-18 09:45:00 | 650.15 | 2025-02-18 10:05:00 | 655.02 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest1 | 2025-02-21 10:15:00 | 702.40 | 2025-02-21 10:30:00 | 706.13 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2025-03-21 09:40:00 | 816.85 | 2025-03-21 10:00:00 | 822.04 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-03-21 09:40:00 | 816.85 | 2025-03-21 10:10:00 | 816.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-16 09:40:00 | 823.00 | 2025-04-16 10:05:00 | 825.88 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-04-23 10:00:00 | 823.40 | 2025-04-23 10:10:00 | 819.69 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-04-23 10:00:00 | 823.40 | 2025-04-23 11:40:00 | 823.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-25 09:55:00 | 812.00 | 2025-04-25 10:10:00 | 815.72 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-05-08 10:55:00 | 820.25 | 2025-05-08 11:05:00 | 824.03 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-05-08 10:55:00 | 820.25 | 2025-05-08 13:05:00 | 820.25 | STOP_HIT | 0.50 | 0.00% |
