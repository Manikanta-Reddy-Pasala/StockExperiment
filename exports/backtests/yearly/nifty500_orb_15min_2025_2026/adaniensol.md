# Adani Energy Solutions Ltd. (ADANIENSOL)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 1351.60
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
| ENTRY1 | 96 |
| ENTRY2 | 0 |
| PARTIAL | 36 |
| TARGET_HIT | 21 |
| STOP_HIT | 75 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 132 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 57 / 75
- **Target hits / Stop hits / Partials:** 21 / 75 / 36
- **Avg / median % per leg:** 0.13% / 0.00%
- **Sum % (uncompounded):** 16.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 64 | 35 | 54.7% | 15 | 29 | 20 | 0.26% | 16.5% |
| BUY @ 2nd Alert (retest1) | 64 | 35 | 54.7% | 15 | 29 | 20 | 0.26% | 16.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 68 | 22 | 32.4% | 6 | 46 | 16 | 0.00% | 0.2% |
| SELL @ 2nd Alert (retest1) | 68 | 22 | 32.4% | 6 | 46 | 16 | 0.00% | 0.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 132 | 57 | 43.2% | 21 | 75 | 36 | 0.13% | 16.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-14 09:30:00 | 890.80 | 895.32 | 0.00 | ORB-short ORB[892.00,903.95] vol=2.4x ATR=4.28 |
| Stop hit — per-position SL triggered | 2025-05-14 15:20:00 | 890.95 | 891.33 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2025-05-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 09:45:00 | 875.10 | 869.73 | 0.00 | ORB-long ORB[863.00,872.70] vol=1.5x ATR=3.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-21 10:50:00 | 880.50 | 873.05 | 0.00 | T1 1.5R @ 880.50 |
| Stop hit — per-position SL triggered | 2025-05-21 11:20:00 | 875.10 | 873.67 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-28 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 10:05:00 | 888.70 | 884.63 | 0.00 | ORB-long ORB[881.25,886.55] vol=1.9x ATR=2.79 |
| Stop hit — per-position SL triggered | 2025-05-28 10:10:00 | 885.91 | 884.70 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-30 10:35:00 | 864.85 | 875.01 | 0.00 | ORB-short ORB[879.55,889.45] vol=2.6x ATR=3.50 |
| Stop hit — per-position SL triggered | 2025-05-30 11:25:00 | 868.35 | 873.15 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 10:00:00 | 885.55 | 878.99 | 0.00 | ORB-long ORB[872.05,885.00] vol=2.3x ATR=3.84 |
| Stop hit — per-position SL triggered | 2025-06-02 10:10:00 | 881.71 | 879.29 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-04 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 10:40:00 | 863.70 | 857.61 | 0.00 | ORB-long ORB[856.60,862.00] vol=2.7x ATR=2.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 10:55:00 | 867.98 | 860.69 | 0.00 | T1 1.5R @ 867.98 |
| Stop hit — per-position SL triggered | 2025-06-04 11:05:00 | 863.70 | 861.96 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 10:55:00 | 874.60 | 868.11 | 0.00 | ORB-long ORB[864.05,869.95] vol=3.5x ATR=2.79 |
| Stop hit — per-position SL triggered | 2025-06-05 11:05:00 | 871.81 | 869.27 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 11:00:00 | 886.95 | 882.36 | 0.00 | ORB-long ORB[872.45,884.35] vol=2.7x ATR=3.10 |
| Stop hit — per-position SL triggered | 2025-06-06 11:15:00 | 883.85 | 882.87 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-09 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 10:40:00 | 890.10 | 883.79 | 0.00 | ORB-long ORB[880.55,888.55] vol=2.4x ATR=2.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-09 10:45:00 | 894.22 | 886.34 | 0.00 | T1 1.5R @ 894.22 |
| Target hit | 2025-06-09 15:20:00 | 902.20 | 899.62 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — SELL (started 2025-06-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-10 10:00:00 | 898.80 | 901.60 | 0.00 | ORB-short ORB[900.35,908.80] vol=1.6x ATR=3.18 |
| Stop hit — per-position SL triggered | 2025-06-10 10:15:00 | 901.98 | 900.45 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-11 10:20:00 | 909.95 | 916.05 | 0.00 | ORB-short ORB[914.50,924.35] vol=1.8x ATR=3.14 |
| Stop hit — per-position SL triggered | 2025-06-11 10:40:00 | 913.09 | 915.76 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-06-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 11:10:00 | 900.10 | 902.69 | 0.00 | ORB-short ORB[900.25,909.00] vol=2.8x ATR=3.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-12 13:15:00 | 895.51 | 900.54 | 0.00 | T1 1.5R @ 895.51 |
| Target hit | 2025-06-12 15:20:00 | 875.40 | 894.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2025-06-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-17 10:45:00 | 857.20 | 861.54 | 0.00 | ORB-short ORB[860.70,867.35] vol=1.7x ATR=2.15 |
| Stop hit — per-position SL triggered | 2025-06-17 11:30:00 | 859.35 | 860.53 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 10:45:00 | 835.65 | 830.99 | 0.00 | ORB-long ORB[825.50,834.50] vol=1.6x ATR=3.50 |
| Stop hit — per-position SL triggered | 2025-06-20 12:20:00 | 832.15 | 832.40 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-06-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 09:30:00 | 860.10 | 854.16 | 0.00 | ORB-long ORB[848.00,856.60] vol=3.3x ATR=3.91 |
| Stop hit — per-position SL triggered | 2025-06-24 09:35:00 | 856.19 | 855.09 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-06-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-25 11:00:00 | 849.50 | 855.49 | 0.00 | ORB-short ORB[851.35,863.70] vol=1.7x ATR=2.42 |
| Stop hit — per-position SL triggered | 2025-06-25 14:20:00 | 851.92 | 851.45 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-06-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 09:45:00 | 886.00 | 876.72 | 0.00 | ORB-long ORB[863.00,873.00] vol=2.0x ATR=3.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 10:00:00 | 891.29 | 881.35 | 0.00 | T1 1.5R @ 891.29 |
| Target hit | 2025-06-27 11:50:00 | 888.55 | 890.06 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — SELL (started 2025-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-30 11:15:00 | 879.30 | 880.67 | 0.00 | ORB-short ORB[880.00,887.45] vol=2.6x ATR=2.10 |
| Stop hit — per-position SL triggered | 2025-06-30 11:25:00 | 881.40 | 880.68 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-01 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 10:55:00 | 878.00 | 882.45 | 0.00 | ORB-short ORB[880.85,889.85] vol=3.5x ATR=2.43 |
| Stop hit — per-position SL triggered | 2025-07-01 11:05:00 | 880.43 | 882.26 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-03 11:15:00 | 872.75 | 875.70 | 0.00 | ORB-short ORB[873.00,882.15] vol=2.0x ATR=2.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-03 11:40:00 | 869.31 | 874.59 | 0.00 | T1 1.5R @ 869.31 |
| Target hit | 2025-07-03 13:00:00 | 872.50 | 872.48 | 0.00 | Trail-exit close>VWAP |

### Cycle 21 — BUY (started 2025-07-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 10:50:00 | 880.05 | 879.05 | 0.00 | ORB-long ORB[872.75,876.95] vol=3.0x ATR=2.01 |
| Stop hit — per-position SL triggered | 2025-07-04 11:00:00 | 878.04 | 879.16 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-07-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 10:40:00 | 883.95 | 879.30 | 0.00 | ORB-long ORB[871.05,878.70] vol=2.3x ATR=2.49 |
| Stop hit — per-position SL triggered | 2025-07-07 11:05:00 | 881.46 | 880.89 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 11:15:00 | 882.65 | 886.43 | 0.00 | ORB-short ORB[883.05,892.50] vol=2.1x ATR=2.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 12:10:00 | 879.41 | 884.16 | 0.00 | T1 1.5R @ 879.41 |
| Target hit | 2025-07-08 14:55:00 | 882.10 | 881.81 | 0.00 | Trail-exit close>VWAP |

### Cycle 24 — BUY (started 2025-07-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 10:25:00 | 895.20 | 890.02 | 0.00 | ORB-long ORB[885.85,893.50] vol=1.8x ATR=2.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-09 10:50:00 | 899.26 | 892.78 | 0.00 | T1 1.5R @ 899.26 |
| Stop hit — per-position SL triggered | 2025-07-09 11:35:00 | 895.20 | 895.85 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-07-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 10:30:00 | 881.50 | 888.59 | 0.00 | ORB-short ORB[890.00,895.55] vol=1.7x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 11:00:00 | 878.37 | 886.07 | 0.00 | T1 1.5R @ 878.37 |
| Stop hit — per-position SL triggered | 2025-07-11 11:40:00 | 881.50 | 884.93 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-07-14 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 10:25:00 | 885.90 | 875.59 | 0.00 | ORB-long ORB[867.30,877.85] vol=1.6x ATR=3.27 |
| Stop hit — per-position SL triggered | 2025-07-14 10:30:00 | 882.63 | 876.12 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-07-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 09:30:00 | 877.00 | 878.81 | 0.00 | ORB-short ORB[877.50,880.95] vol=1.6x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 10:15:00 | 874.61 | 877.32 | 0.00 | T1 1.5R @ 874.61 |
| Stop hit — per-position SL triggered | 2025-07-17 11:15:00 | 877.00 | 876.97 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-07-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 09:45:00 | 871.35 | 874.18 | 0.00 | ORB-short ORB[873.35,876.90] vol=1.7x ATR=1.61 |
| Stop hit — per-position SL triggered | 2025-07-18 09:50:00 | 872.96 | 874.38 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 10:15:00 | 878.20 | 870.23 | 0.00 | ORB-long ORB[862.10,872.05] vol=2.1x ATR=2.68 |
| Stop hit — per-position SL triggered | 2025-07-21 13:10:00 | 875.52 | 874.31 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-07-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 09:35:00 | 872.60 | 875.66 | 0.00 | ORB-short ORB[873.60,879.80] vol=1.6x ATR=1.86 |
| Stop hit — per-position SL triggered | 2025-07-22 09:45:00 | 874.46 | 874.82 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2025-07-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 09:30:00 | 858.70 | 863.15 | 0.00 | ORB-short ORB[859.10,868.00] vol=2.1x ATR=2.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 09:40:00 | 855.35 | 861.15 | 0.00 | T1 1.5R @ 855.35 |
| Target hit | 2025-07-24 14:50:00 | 855.65 | 851.95 | 0.00 | Trail-exit close>VWAP |

### Cycle 32 — SELL (started 2025-07-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-29 09:55:00 | 807.85 | 813.06 | 0.00 | ORB-short ORB[811.20,822.40] vol=2.4x ATR=3.35 |
| Stop hit — per-position SL triggered | 2025-07-29 10:00:00 | 811.20 | 812.95 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-08-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-05 09:45:00 | 785.70 | 793.57 | 0.00 | ORB-short ORB[795.70,804.70] vol=1.8x ATR=2.78 |
| Stop hit — per-position SL triggered | 2025-08-05 10:05:00 | 788.48 | 790.23 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-08-06 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 10:25:00 | 794.15 | 795.72 | 0.00 | ORB-short ORB[795.05,802.15] vol=4.4x ATR=2.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 10:30:00 | 790.33 | 794.80 | 0.00 | T1 1.5R @ 790.33 |
| Target hit | 2025-08-06 11:55:00 | 790.10 | 789.94 | 0.00 | Trail-exit close>VWAP |

### Cycle 35 — SELL (started 2025-08-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 11:05:00 | 791.70 | 794.67 | 0.00 | ORB-short ORB[792.00,800.15] vol=7.1x ATR=1.94 |
| Stop hit — per-position SL triggered | 2025-08-12 11:20:00 | 793.64 | 794.60 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-08-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 10:05:00 | 801.25 | 794.34 | 0.00 | ORB-long ORB[783.00,793.40] vol=1.8x ATR=2.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 10:10:00 | 805.43 | 796.51 | 0.00 | T1 1.5R @ 805.43 |
| Target hit | 2025-08-18 15:20:00 | 818.00 | 809.32 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — BUY (started 2025-08-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 11:00:00 | 824.10 | 818.54 | 0.00 | ORB-long ORB[813.05,821.35] vol=1.9x ATR=2.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 11:05:00 | 827.37 | 819.69 | 0.00 | T1 1.5R @ 827.37 |
| Stop hit — per-position SL triggered | 2025-08-19 11:40:00 | 824.10 | 820.57 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-08-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 10:45:00 | 830.70 | 826.56 | 0.00 | ORB-long ORB[823.25,830.30] vol=2.3x ATR=1.85 |
| Stop hit — per-position SL triggered | 2025-08-20 10:55:00 | 828.85 | 826.84 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-21 11:15:00 | 819.90 | 823.87 | 0.00 | ORB-short ORB[823.10,829.20] vol=3.2x ATR=1.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 12:00:00 | 817.18 | 822.93 | 0.00 | T1 1.5R @ 817.18 |
| Stop hit — per-position SL triggered | 2025-08-21 12:05:00 | 819.90 | 822.78 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 11:15:00 | 805.50 | 809.83 | 0.00 | ORB-short ORB[808.15,816.00] vol=1.6x ATR=1.59 |
| Stop hit — per-position SL triggered | 2025-08-22 12:00:00 | 807.09 | 809.25 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-08-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:40:00 | 791.45 | 794.74 | 0.00 | ORB-short ORB[792.65,803.45] vol=1.5x ATR=2.16 |
| Stop hit — per-position SL triggered | 2025-08-26 09:50:00 | 793.61 | 794.54 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-08-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-28 11:05:00 | 790.20 | 782.13 | 0.00 | ORB-long ORB[775.50,786.00] vol=1.7x ATR=2.20 |
| Stop hit — per-position SL triggered | 2025-08-28 11:35:00 | 788.00 | 782.96 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-08-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 09:35:00 | 764.30 | 767.36 | 0.00 | ORB-short ORB[765.60,776.40] vol=1.5x ATR=3.08 |
| Stop hit — per-position SL triggered | 2025-08-29 10:15:00 | 767.38 | 765.14 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-01 10:15:00 | 765.90 | 770.68 | 0.00 | ORB-short ORB[769.35,776.00] vol=4.6x ATR=2.91 |
| Stop hit — per-position SL triggered | 2025-09-01 15:20:00 | 766.60 | 767.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 45 — SELL (started 2025-09-03 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-03 10:35:00 | 760.55 | 764.25 | 0.00 | ORB-short ORB[763.20,768.90] vol=1.8x ATR=1.65 |
| Stop hit — per-position SL triggered | 2025-09-03 11:00:00 | 762.20 | 763.81 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:15:00 | 750.00 | 755.07 | 0.00 | ORB-short ORB[754.00,761.45] vol=2.1x ATR=2.01 |
| Stop hit — per-position SL triggered | 2025-09-05 10:20:00 | 752.01 | 754.55 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-09-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 10:00:00 | 767.85 | 764.22 | 0.00 | ORB-long ORB[757.65,766.70] vol=1.5x ATR=2.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 10:05:00 | 771.78 | 765.39 | 0.00 | T1 1.5R @ 771.78 |
| Target hit | 2025-09-08 15:20:00 | 776.30 | 771.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 10:15:00 | 824.50 | 820.04 | 0.00 | ORB-long ORB[813.75,819.85] vol=1.5x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-11 10:25:00 | 828.80 | 821.01 | 0.00 | T1 1.5R @ 828.80 |
| Target hit | 2025-09-11 10:55:00 | 825.30 | 825.76 | 0.00 | Trail-exit close<VWAP |

### Cycle 49 — SELL (started 2025-09-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 11:10:00 | 813.05 | 815.34 | 0.00 | ORB-short ORB[813.75,817.75] vol=1.7x ATR=1.44 |
| Stop hit — per-position SL triggered | 2025-09-12 11:45:00 | 814.49 | 815.10 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-09-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 09:30:00 | 832.80 | 829.19 | 0.00 | ORB-long ORB[824.30,830.90] vol=1.7x ATR=2.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-18 10:40:00 | 835.82 | 832.73 | 0.00 | T1 1.5R @ 835.82 |
| Target hit | 2025-09-18 12:05:00 | 835.20 | 837.86 | 0.00 | Trail-exit close<VWAP |

### Cycle 51 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 11:15:00 | 881.60 | 892.05 | 0.00 | ORB-short ORB[889.50,898.70] vol=6.4x ATR=3.14 |
| Stop hit — per-position SL triggered | 2025-09-26 11:20:00 | 884.74 | 891.57 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-09-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-29 10:10:00 | 874.60 | 880.92 | 0.00 | ORB-short ORB[877.90,887.30] vol=1.8x ATR=3.29 |
| Stop hit — per-position SL triggered | 2025-09-29 10:15:00 | 877.89 | 880.75 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-30 11:15:00 | 865.95 | 870.97 | 0.00 | ORB-short ORB[871.00,876.50] vol=1.8x ATR=2.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 11:20:00 | 862.82 | 870.36 | 0.00 | T1 1.5R @ 862.82 |
| Stop hit — per-position SL triggered | 2025-09-30 11:25:00 | 865.95 | 870.23 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-03 11:15:00 | 904.35 | 900.25 | 0.00 | ORB-long ORB[892.30,902.00] vol=1.7x ATR=2.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 11:20:00 | 908.21 | 901.01 | 0.00 | T1 1.5R @ 908.21 |
| Target hit | 2025-10-03 15:20:00 | 915.70 | 909.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — SELL (started 2025-10-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 11:05:00 | 912.70 | 919.73 | 0.00 | ORB-short ORB[913.90,923.45] vol=2.7x ATR=3.30 |
| Stop hit — per-position SL triggered | 2025-10-08 12:50:00 | 916.00 | 917.16 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2025-10-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-10 09:40:00 | 923.95 | 926.19 | 0.00 | ORB-short ORB[924.05,932.95] vol=1.6x ATR=2.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-10 10:45:00 | 919.95 | 924.24 | 0.00 | T1 1.5R @ 919.95 |
| Stop hit — per-position SL triggered | 2025-10-10 13:35:00 | 923.95 | 922.47 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-10-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 09:35:00 | 932.25 | 927.62 | 0.00 | ORB-long ORB[922.65,931.85] vol=2.0x ATR=3.48 |
| Stop hit — per-position SL triggered | 2025-10-13 09:50:00 | 928.77 | 928.25 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-10-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-15 09:40:00 | 925.60 | 931.02 | 0.00 | ORB-short ORB[930.35,937.70] vol=2.4x ATR=2.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 10:40:00 | 921.18 | 927.47 | 0.00 | T1 1.5R @ 921.18 |
| Stop hit — per-position SL triggered | 2025-10-15 11:45:00 | 925.60 | 926.65 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-10-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-17 09:50:00 | 937.25 | 941.65 | 0.00 | ORB-short ORB[939.00,950.00] vol=2.1x ATR=2.89 |
| Stop hit — per-position SL triggered | 2025-10-17 10:15:00 | 940.14 | 938.76 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2025-10-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 10:50:00 | 938.40 | 932.84 | 0.00 | ORB-long ORB[928.50,932.70] vol=2.3x ATR=2.43 |
| Stop hit — per-position SL triggered | 2025-10-20 12:10:00 | 935.97 | 934.59 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-10-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 09:40:00 | 931.05 | 925.05 | 0.00 | ORB-long ORB[917.95,928.05] vol=1.8x ATR=3.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 10:00:00 | 935.81 | 928.13 | 0.00 | T1 1.5R @ 935.81 |
| Target hit | 2025-10-29 11:35:00 | 965.55 | 965.56 | 0.00 | Trail-exit close<VWAP |

### Cycle 62 — BUY (started 2025-10-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:45:00 | 976.00 | 970.11 | 0.00 | ORB-long ORB[960.10,971.05] vol=2.4x ATR=3.86 |
| Stop hit — per-position SL triggered | 2025-10-31 10:15:00 | 972.14 | 971.80 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-11-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 10:30:00 | 1009.95 | 1002.65 | 0.00 | ORB-long ORB[992.00,1004.55] vol=2.3x ATR=3.46 |
| Stop hit — per-position SL triggered | 2025-11-04 10:35:00 | 1006.49 | 1003.09 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-11-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 10:20:00 | 951.55 | 953.03 | 0.00 | ORB-short ORB[952.40,963.70] vol=2.1x ATR=2.96 |
| Stop hit — per-position SL triggered | 2025-11-11 10:40:00 | 954.51 | 953.13 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-11-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-12 09:40:00 | 1002.20 | 994.50 | 0.00 | ORB-long ORB[986.00,993.90] vol=1.8x ATR=4.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 09:50:00 | 1008.49 | 999.40 | 0.00 | T1 1.5R @ 1008.49 |
| Target hit | 2025-11-12 10:15:00 | 1006.50 | 1008.65 | 0.00 | Trail-exit close<VWAP |

### Cycle 66 — BUY (started 2025-11-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-14 09:40:00 | 1040.95 | 1034.16 | 0.00 | ORB-long ORB[1024.10,1039.05] vol=2.0x ATR=3.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-14 09:45:00 | 1046.83 | 1037.54 | 0.00 | T1 1.5R @ 1046.83 |
| Stop hit — per-position SL triggered | 2025-11-14 09:50:00 | 1040.95 | 1038.44 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2025-11-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-17 10:30:00 | 1017.30 | 1021.25 | 0.00 | ORB-short ORB[1020.05,1029.00] vol=1.8x ATR=3.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-17 10:50:00 | 1012.77 | 1020.22 | 0.00 | T1 1.5R @ 1012.77 |
| Stop hit — per-position SL triggered | 2025-11-17 11:50:00 | 1017.30 | 1019.22 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-11-21 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:50:00 | 979.50 | 988.04 | 0.00 | ORB-short ORB[990.20,996.85] vol=3.4x ATR=2.57 |
| Stop hit — per-position SL triggered | 2025-11-21 10:55:00 | 982.07 | 987.92 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-11-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 09:30:00 | 988.90 | 985.30 | 0.00 | ORB-long ORB[974.70,988.00] vol=1.6x ATR=3.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-26 09:50:00 | 993.83 | 990.14 | 0.00 | T1 1.5R @ 993.83 |
| Target hit | 2025-11-26 10:45:00 | 990.50 | 992.24 | 0.00 | Trail-exit close<VWAP |

### Cycle 70 — SELL (started 2025-11-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 09:30:00 | 981.25 | 985.43 | 0.00 | ORB-short ORB[982.05,994.15] vol=1.7x ATR=2.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 10:00:00 | 977.30 | 983.41 | 0.00 | T1 1.5R @ 977.30 |
| Stop hit — per-position SL triggered | 2025-11-27 10:20:00 | 981.25 | 982.02 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-11-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-28 09:40:00 | 991.45 | 985.73 | 0.00 | ORB-long ORB[977.90,988.70] vol=1.6x ATR=3.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 09:45:00 | 996.78 | 987.91 | 0.00 | T1 1.5R @ 996.78 |
| Target hit | 2025-11-28 10:55:00 | 993.35 | 993.58 | 0.00 | Trail-exit close<VWAP |

### Cycle 72 — BUY (started 2025-12-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 11:05:00 | 973.65 | 967.94 | 0.00 | ORB-long ORB[965.10,973.10] vol=1.7x ATR=2.83 |
| Stop hit — per-position SL triggered | 2025-12-05 12:00:00 | 970.82 | 968.59 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2025-12-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-08 10:30:00 | 971.10 | 974.64 | 0.00 | ORB-short ORB[971.45,980.90] vol=1.8x ATR=2.85 |
| Stop hit — per-position SL triggered | 2025-12-08 10:45:00 | 973.95 | 974.43 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2025-12-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 11:10:00 | 1005.60 | 1006.88 | 0.00 | ORB-short ORB[1006.70,1014.70] vol=1.7x ATR=1.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-16 11:30:00 | 1003.07 | 1006.55 | 0.00 | T1 1.5R @ 1003.07 |
| Target hit | 2025-12-16 15:20:00 | 991.95 | 1000.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — SELL (started 2025-12-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 10:50:00 | 983.25 | 988.97 | 0.00 | ORB-short ORB[988.15,994.00] vol=1.7x ATR=2.30 |
| Stop hit — per-position SL triggered | 2025-12-17 11:35:00 | 985.55 | 988.20 | 0.00 | SL hit |

### Cycle 76 — SELL (started 2025-12-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-19 11:05:00 | 974.85 | 976.98 | 0.00 | ORB-short ORB[976.35,980.95] vol=1.8x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-19 11:30:00 | 971.96 | 976.53 | 0.00 | T1 1.5R @ 971.96 |
| Stop hit — per-position SL triggered | 2025-12-19 11:55:00 | 974.85 | 976.34 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2025-12-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 09:45:00 | 1005.10 | 1000.77 | 0.00 | ORB-long ORB[991.10,1004.60] vol=2.3x ATR=2.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 10:10:00 | 1009.58 | 1004.88 | 0.00 | T1 1.5R @ 1009.58 |
| Target hit | 2025-12-26 11:10:00 | 1011.95 | 1012.18 | 0.00 | Trail-exit close<VWAP |

### Cycle 78 — BUY (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 10:15:00 | 1007.65 | 1003.22 | 0.00 | ORB-long ORB[996.30,1005.50] vol=1.7x ATR=2.08 |
| Stop hit — per-position SL triggered | 2025-12-30 10:30:00 | 1005.57 | 1003.71 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2026-01-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 09:35:00 | 1038.00 | 1033.20 | 0.00 | ORB-long ORB[1026.30,1032.90] vol=2.4x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 09:40:00 | 1042.70 | 1037.74 | 0.00 | T1 1.5R @ 1042.70 |
| Target hit | 2026-01-01 10:15:00 | 1044.40 | 1048.92 | 0.00 | Trail-exit close<VWAP |

### Cycle 80 — SELL (started 2026-01-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-05 10:00:00 | 1042.50 | 1048.63 | 0.00 | ORB-short ORB[1048.00,1057.70] vol=2.3x ATR=3.94 |
| Stop hit — per-position SL triggered | 2026-01-05 10:20:00 | 1046.44 | 1047.88 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2026-01-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-07 09:45:00 | 1029.30 | 1035.87 | 0.00 | ORB-short ORB[1035.10,1044.60] vol=1.9x ATR=2.75 |
| Stop hit — per-position SL triggered | 2026-01-07 09:50:00 | 1032.05 | 1035.36 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:15:00 | 1008.10 | 1019.26 | 0.00 | ORB-short ORB[1021.00,1032.00] vol=1.6x ATR=2.86 |
| Stop hit — per-position SL triggered | 2026-01-08 11:35:00 | 1010.96 | 1016.43 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2026-02-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-01 11:00:00 | 904.80 | 890.89 | 0.00 | ORB-long ORB[878.10,888.20] vol=1.6x ATR=3.03 |
| Stop hit — per-position SL triggered | 2026-02-01 11:15:00 | 901.77 | 892.40 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2026-02-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:35:00 | 1014.25 | 1023.70 | 0.00 | ORB-short ORB[1019.80,1033.90] vol=2.2x ATR=3.57 |
| Stop hit — per-position SL triggered | 2026-02-10 10:40:00 | 1017.82 | 1021.51 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2026-02-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 09:35:00 | 1021.65 | 1017.81 | 0.00 | ORB-long ORB[1009.85,1019.00] vol=2.0x ATR=3.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 09:45:00 | 1026.97 | 1019.28 | 0.00 | T1 1.5R @ 1026.97 |
| Target hit | 2026-02-11 15:20:00 | 1033.80 | 1028.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 86 — BUY (started 2026-02-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:40:00 | 1014.05 | 1010.08 | 0.00 | ORB-long ORB[996.00,1005.00] vol=8.9x ATR=3.42 |
| Stop hit — per-position SL triggered | 2026-02-17 10:45:00 | 1010.63 | 1010.76 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2026-02-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:40:00 | 1015.95 | 1026.23 | 0.00 | ORB-short ORB[1028.40,1035.70] vol=2.0x ATR=3.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:00:00 | 1011.42 | 1023.89 | 0.00 | T1 1.5R @ 1011.42 |
| Stop hit — per-position SL triggered | 2026-02-19 11:55:00 | 1015.95 | 1020.47 | 0.00 | SL hit |

### Cycle 88 — BUY (started 2026-02-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 10:35:00 | 1031.00 | 1027.81 | 0.00 | ORB-long ORB[1015.05,1030.45] vol=1.6x ATR=3.04 |
| Stop hit — per-position SL triggered | 2026-02-27 11:25:00 | 1027.96 | 1028.19 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2026-03-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:55:00 | 952.50 | 957.40 | 0.00 | ORB-short ORB[953.40,963.80] vol=1.9x ATR=4.22 |
| Stop hit — per-position SL triggered | 2026-03-04 10:00:00 | 956.72 | 957.22 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2026-03-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 09:35:00 | 982.60 | 991.34 | 0.00 | ORB-short ORB[986.90,1001.30] vol=1.8x ATR=5.56 |
| Stop hit — per-position SL triggered | 2026-03-10 10:10:00 | 988.16 | 988.59 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2026-03-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:05:00 | 992.00 | 984.76 | 0.00 | ORB-long ORB[976.00,988.40] vol=2.0x ATR=4.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 10:35:00 | 998.64 | 988.07 | 0.00 | T1 1.5R @ 998.64 |
| Target hit | 2026-03-12 15:20:00 | 1003.30 | 1001.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 92 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 1019.20 | 1010.41 | 0.00 | ORB-long ORB[1000.00,1013.90] vol=1.7x ATR=4.50 |
| Stop hit — per-position SL triggered | 2026-03-18 10:05:00 | 1014.70 | 1015.79 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2026-03-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 11:00:00 | 944.40 | 963.03 | 0.00 | ORB-short ORB[987.60,1000.90] vol=2.0x ATR=5.34 |
| Stop hit — per-position SL triggered | 2026-03-23 11:05:00 | 949.74 | 962.08 | 0.00 | SL hit |

### Cycle 94 — BUY (started 2026-04-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 10:35:00 | 969.75 | 953.12 | 0.00 | ORB-long ORB[936.10,949.60] vol=2.3x ATR=5.80 |
| Stop hit — per-position SL triggered | 2026-04-06 11:30:00 | 963.95 | 959.69 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2026-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:45:00 | 1191.80 | 1180.74 | 0.00 | ORB-long ORB[1174.30,1186.40] vol=2.4x ATR=5.72 |
| Stop hit — per-position SL triggered | 2026-04-16 09:55:00 | 1186.08 | 1181.15 | 0.00 | SL hit |

### Cycle 96 — SELL (started 2026-05-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 10:30:00 | 1389.40 | 1389.61 | 0.00 | ORB-short ORB[1394.70,1410.70] vol=3.3x ATR=6.59 |
| Stop hit — per-position SL triggered | 2026-05-07 10:35:00 | 1395.99 | 1389.99 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-14 09:30:00 | 890.80 | 2025-05-14 15:20:00 | 890.95 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest1 | 2025-05-21 09:45:00 | 875.10 | 2025-05-21 10:50:00 | 880.50 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-05-21 09:45:00 | 875.10 | 2025-05-21 11:20:00 | 875.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-28 10:05:00 | 888.70 | 2025-05-28 10:10:00 | 885.91 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-05-30 10:35:00 | 864.85 | 2025-05-30 11:25:00 | 868.35 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-06-02 10:00:00 | 885.55 | 2025-06-02 10:10:00 | 881.71 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-06-04 10:40:00 | 863.70 | 2025-06-04 10:55:00 | 867.98 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-06-04 10:40:00 | 863.70 | 2025-06-04 11:05:00 | 863.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-05 10:55:00 | 874.60 | 2025-06-05 11:05:00 | 871.81 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-06-06 11:00:00 | 886.95 | 2025-06-06 11:15:00 | 883.85 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-06-09 10:40:00 | 890.10 | 2025-06-09 10:45:00 | 894.22 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-06-09 10:40:00 | 890.10 | 2025-06-09 15:20:00 | 902.20 | TARGET_HIT | 0.50 | 1.36% |
| SELL | retest1 | 2025-06-10 10:00:00 | 898.80 | 2025-06-10 10:15:00 | 901.98 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-06-11 10:20:00 | 909.95 | 2025-06-11 10:40:00 | 913.09 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-06-12 11:10:00 | 900.10 | 2025-06-12 13:15:00 | 895.51 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2025-06-12 11:10:00 | 900.10 | 2025-06-12 15:20:00 | 875.40 | TARGET_HIT | 0.50 | 2.74% |
| SELL | retest1 | 2025-06-17 10:45:00 | 857.20 | 2025-06-17 11:30:00 | 859.35 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-06-20 10:45:00 | 835.65 | 2025-06-20 12:20:00 | 832.15 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-06-24 09:30:00 | 860.10 | 2025-06-24 09:35:00 | 856.19 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-06-25 11:00:00 | 849.50 | 2025-06-25 14:20:00 | 851.92 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-06-27 09:45:00 | 886.00 | 2025-06-27 10:00:00 | 891.29 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-06-27 09:45:00 | 886.00 | 2025-06-27 11:50:00 | 888.55 | TARGET_HIT | 0.50 | 0.29% |
| SELL | retest1 | 2025-06-30 11:15:00 | 879.30 | 2025-06-30 11:25:00 | 881.40 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-07-01 10:55:00 | 878.00 | 2025-07-01 11:05:00 | 880.43 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-07-03 11:15:00 | 872.75 | 2025-07-03 11:40:00 | 869.31 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-07-03 11:15:00 | 872.75 | 2025-07-03 13:00:00 | 872.50 | TARGET_HIT | 0.50 | 0.03% |
| BUY | retest1 | 2025-07-04 10:50:00 | 880.05 | 2025-07-04 11:00:00 | 878.04 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-07-07 10:40:00 | 883.95 | 2025-07-07 11:05:00 | 881.46 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-07-08 11:15:00 | 882.65 | 2025-07-08 12:10:00 | 879.41 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-07-08 11:15:00 | 882.65 | 2025-07-08 14:55:00 | 882.10 | TARGET_HIT | 0.50 | 0.06% |
| BUY | retest1 | 2025-07-09 10:25:00 | 895.20 | 2025-07-09 10:50:00 | 899.26 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-07-09 10:25:00 | 895.20 | 2025-07-09 11:35:00 | 895.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-11 10:30:00 | 881.50 | 2025-07-11 11:00:00 | 878.37 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2025-07-11 10:30:00 | 881.50 | 2025-07-11 11:40:00 | 881.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-14 10:25:00 | 885.90 | 2025-07-14 10:30:00 | 882.63 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-07-17 09:30:00 | 877.00 | 2025-07-17 10:15:00 | 874.61 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-07-17 09:30:00 | 877.00 | 2025-07-17 11:15:00 | 877.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-18 09:45:00 | 871.35 | 2025-07-18 09:50:00 | 872.96 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-07-21 10:15:00 | 878.20 | 2025-07-21 13:10:00 | 875.52 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-07-22 09:35:00 | 872.60 | 2025-07-22 09:45:00 | 874.46 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-07-24 09:30:00 | 858.70 | 2025-07-24 09:40:00 | 855.35 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-07-24 09:30:00 | 858.70 | 2025-07-24 14:50:00 | 855.65 | TARGET_HIT | 0.50 | 0.36% |
| SELL | retest1 | 2025-07-29 09:55:00 | 807.85 | 2025-07-29 10:00:00 | 811.20 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-08-05 09:45:00 | 785.70 | 2025-08-05 10:05:00 | 788.48 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-08-06 10:25:00 | 794.15 | 2025-08-06 10:30:00 | 790.33 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-08-06 10:25:00 | 794.15 | 2025-08-06 11:55:00 | 790.10 | TARGET_HIT | 0.50 | 0.51% |
| SELL | retest1 | 2025-08-12 11:05:00 | 791.70 | 2025-08-12 11:20:00 | 793.64 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-08-18 10:05:00 | 801.25 | 2025-08-18 10:10:00 | 805.43 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-08-18 10:05:00 | 801.25 | 2025-08-18 15:20:00 | 818.00 | TARGET_HIT | 0.50 | 2.09% |
| BUY | retest1 | 2025-08-19 11:00:00 | 824.10 | 2025-08-19 11:05:00 | 827.37 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-08-19 11:00:00 | 824.10 | 2025-08-19 11:40:00 | 824.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-20 10:45:00 | 830.70 | 2025-08-20 10:55:00 | 828.85 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-08-21 11:15:00 | 819.90 | 2025-08-21 12:00:00 | 817.18 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-08-21 11:15:00 | 819.90 | 2025-08-21 12:05:00 | 819.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-22 11:15:00 | 805.50 | 2025-08-22 12:00:00 | 807.09 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-08-26 09:40:00 | 791.45 | 2025-08-26 09:50:00 | 793.61 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-08-28 11:05:00 | 790.20 | 2025-08-28 11:35:00 | 788.00 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-08-29 09:35:00 | 764.30 | 2025-08-29 10:15:00 | 767.38 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-09-01 10:15:00 | 765.90 | 2025-09-01 15:20:00 | 766.60 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest1 | 2025-09-03 10:35:00 | 760.55 | 2025-09-03 11:00:00 | 762.20 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-09-05 10:15:00 | 750.00 | 2025-09-05 10:20:00 | 752.01 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-09-08 10:00:00 | 767.85 | 2025-09-08 10:05:00 | 771.78 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-09-08 10:00:00 | 767.85 | 2025-09-08 15:20:00 | 776.30 | TARGET_HIT | 0.50 | 1.10% |
| BUY | retest1 | 2025-09-11 10:15:00 | 824.50 | 2025-09-11 10:25:00 | 828.80 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-09-11 10:15:00 | 824.50 | 2025-09-11 10:55:00 | 825.30 | TARGET_HIT | 0.50 | 0.10% |
| SELL | retest1 | 2025-09-12 11:10:00 | 813.05 | 2025-09-12 11:45:00 | 814.49 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-09-18 09:30:00 | 832.80 | 2025-09-18 10:40:00 | 835.82 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-09-18 09:30:00 | 832.80 | 2025-09-18 12:05:00 | 835.20 | TARGET_HIT | 0.50 | 0.29% |
| SELL | retest1 | 2025-09-26 11:15:00 | 881.60 | 2025-09-26 11:20:00 | 884.74 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-09-29 10:10:00 | 874.60 | 2025-09-29 10:15:00 | 877.89 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-09-30 11:15:00 | 865.95 | 2025-09-30 11:20:00 | 862.82 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-09-30 11:15:00 | 865.95 | 2025-09-30 11:25:00 | 865.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-03 11:15:00 | 904.35 | 2025-10-03 11:20:00 | 908.21 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-10-03 11:15:00 | 904.35 | 2025-10-03 15:20:00 | 915.70 | TARGET_HIT | 0.50 | 1.26% |
| SELL | retest1 | 2025-10-08 11:05:00 | 912.70 | 2025-10-08 12:50:00 | 916.00 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-10-10 09:40:00 | 923.95 | 2025-10-10 10:45:00 | 919.95 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-10-10 09:40:00 | 923.95 | 2025-10-10 13:35:00 | 923.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-10-13 09:35:00 | 932.25 | 2025-10-13 09:50:00 | 928.77 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-10-15 09:40:00 | 925.60 | 2025-10-15 10:40:00 | 921.18 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-10-15 09:40:00 | 925.60 | 2025-10-15 11:45:00 | 925.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-17 09:50:00 | 937.25 | 2025-10-17 10:15:00 | 940.14 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-10-20 10:50:00 | 938.40 | 2025-10-20 12:10:00 | 935.97 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-10-29 09:40:00 | 931.05 | 2025-10-29 10:00:00 | 935.81 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-10-29 09:40:00 | 931.05 | 2025-10-29 11:35:00 | 965.55 | TARGET_HIT | 0.50 | 3.71% |
| BUY | retest1 | 2025-10-31 09:45:00 | 976.00 | 2025-10-31 10:15:00 | 972.14 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-11-04 10:30:00 | 1009.95 | 2025-11-04 10:35:00 | 1006.49 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-11-11 10:20:00 | 951.55 | 2025-11-11 10:40:00 | 954.51 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-11-12 09:40:00 | 1002.20 | 2025-11-12 09:50:00 | 1008.49 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2025-11-12 09:40:00 | 1002.20 | 2025-11-12 10:15:00 | 1006.50 | TARGET_HIT | 0.50 | 0.43% |
| BUY | retest1 | 2025-11-14 09:40:00 | 1040.95 | 2025-11-14 09:45:00 | 1046.83 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-11-14 09:40:00 | 1040.95 | 2025-11-14 09:50:00 | 1040.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-17 10:30:00 | 1017.30 | 2025-11-17 10:50:00 | 1012.77 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-11-17 10:30:00 | 1017.30 | 2025-11-17 11:50:00 | 1017.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-21 10:50:00 | 979.50 | 2025-11-21 10:55:00 | 982.07 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-11-26 09:30:00 | 988.90 | 2025-11-26 09:50:00 | 993.83 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-11-26 09:30:00 | 988.90 | 2025-11-26 10:45:00 | 990.50 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2025-11-27 09:30:00 | 981.25 | 2025-11-27 10:00:00 | 977.30 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-11-27 09:30:00 | 981.25 | 2025-11-27 10:20:00 | 981.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-28 09:40:00 | 991.45 | 2025-11-28 09:45:00 | 996.78 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-11-28 09:40:00 | 991.45 | 2025-11-28 10:55:00 | 993.35 | TARGET_HIT | 0.50 | 0.19% |
| BUY | retest1 | 2025-12-05 11:05:00 | 973.65 | 2025-12-05 12:00:00 | 970.82 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-12-08 10:30:00 | 971.10 | 2025-12-08 10:45:00 | 973.95 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-12-16 11:10:00 | 1005.60 | 2025-12-16 11:30:00 | 1003.07 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-12-16 11:10:00 | 1005.60 | 2025-12-16 15:20:00 | 991.95 | TARGET_HIT | 0.50 | 1.36% |
| SELL | retest1 | 2025-12-17 10:50:00 | 983.25 | 2025-12-17 11:35:00 | 985.55 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-12-19 11:05:00 | 974.85 | 2025-12-19 11:30:00 | 971.96 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-12-19 11:05:00 | 974.85 | 2025-12-19 11:55:00 | 974.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-26 09:45:00 | 1005.10 | 2025-12-26 10:10:00 | 1009.58 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-12-26 09:45:00 | 1005.10 | 2025-12-26 11:10:00 | 1011.95 | TARGET_HIT | 0.50 | 0.68% |
| BUY | retest1 | 2025-12-30 10:15:00 | 1007.65 | 2025-12-30 10:30:00 | 1005.57 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-01-01 09:35:00 | 1038.00 | 2026-01-01 09:40:00 | 1042.70 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2026-01-01 09:35:00 | 1038.00 | 2026-01-01 10:15:00 | 1044.40 | TARGET_HIT | 0.50 | 0.62% |
| SELL | retest1 | 2026-01-05 10:00:00 | 1042.50 | 2026-01-05 10:20:00 | 1046.44 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-01-07 09:45:00 | 1029.30 | 2026-01-07 09:50:00 | 1032.05 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-01-08 11:15:00 | 1008.10 | 2026-01-08 11:35:00 | 1010.96 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-01 11:00:00 | 904.80 | 2026-02-01 11:15:00 | 901.77 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-02-10 10:35:00 | 1014.25 | 2026-02-10 10:40:00 | 1017.82 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-11 09:35:00 | 1021.65 | 2026-02-11 09:45:00 | 1026.97 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-02-11 09:35:00 | 1021.65 | 2026-02-11 15:20:00 | 1033.80 | TARGET_HIT | 0.50 | 1.19% |
| BUY | retest1 | 2026-02-17 10:40:00 | 1014.05 | 2026-02-17 10:45:00 | 1010.63 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-19 10:40:00 | 1015.95 | 2026-02-19 11:00:00 | 1011.42 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-02-19 10:40:00 | 1015.95 | 2026-02-19 11:55:00 | 1015.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-27 10:35:00 | 1031.00 | 2026-02-27 11:25:00 | 1027.96 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-04 09:55:00 | 952.50 | 2026-03-04 10:00:00 | 956.72 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-03-10 09:35:00 | 982.60 | 2026-03-10 10:10:00 | 988.16 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2026-03-12 10:05:00 | 992.00 | 2026-03-12 10:35:00 | 998.64 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2026-03-12 10:05:00 | 992.00 | 2026-03-12 15:20:00 | 1003.30 | TARGET_HIT | 0.50 | 1.14% |
| BUY | retest1 | 2026-03-18 09:30:00 | 1019.20 | 2026-03-18 10:05:00 | 1014.70 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-03-23 11:00:00 | 944.40 | 2026-03-23 11:05:00 | 949.74 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2026-04-06 10:35:00 | 969.75 | 2026-04-06 11:30:00 | 963.95 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest1 | 2026-04-16 09:45:00 | 1191.80 | 2026-04-16 09:55:00 | 1186.08 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-05-07 10:30:00 | 1389.40 | 2026-05-07 10:35:00 | 1395.99 | STOP_HIT | 1.00 | -0.47% |
