# PNB Housing Finance Ltd. (PNBHOUSING)

## Backtest Summary

- **Window:** 2024-11-07 09:15:00 → 2026-05-08 15:25:00 (27688 bars)
- **Last close:** 1088.90
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
| ENTRY1 | 22 |
| ENTRY2 | 0 |
| PARTIAL | 12 |
| TARGET_HIT | 7 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 34 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 15
- **Target hits / Stop hits / Partials:** 7 / 15 / 12
- **Avg / median % per leg:** 0.25% / 0.47%
- **Sum % (uncompounded):** 8.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 12 | 60.0% | 4 | 8 | 8 | 0.25% | 5.0% |
| BUY @ 2nd Alert (retest1) | 20 | 12 | 60.0% | 4 | 8 | 8 | 0.25% | 5.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 7 | 50.0% | 3 | 7 | 4 | 0.24% | 3.3% |
| SELL @ 2nd Alert (retest1) | 14 | 7 | 50.0% | 3 | 7 | 4 | 0.24% | 3.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 34 | 19 | 55.9% | 7 | 15 | 12 | 0.25% | 8.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-12-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-02 10:45:00 | 871.65 | 878.70 | 0.00 | ORB-short ORB[880.00,891.70] vol=1.7x ATR=3.65 |
| Stop hit — per-position SL triggered | 2024-12-02 11:30:00 | 875.30 | 877.80 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-12-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 10:40:00 | 888.95 | 886.42 | 0.00 | ORB-long ORB[875.55,888.85] vol=1.6x ATR=3.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 11:20:00 | 894.03 | 887.57 | 0.00 | T1 1.5R @ 894.03 |
| Stop hit — per-position SL triggered | 2024-12-03 12:05:00 | 888.95 | 888.43 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-12-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 10:10:00 | 948.85 | 955.42 | 0.00 | ORB-short ORB[953.05,964.25] vol=1.7x ATR=3.59 |
| Stop hit — per-position SL triggered | 2024-12-10 11:00:00 | 952.44 | 953.00 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-12-12 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 10:50:00 | 934.95 | 939.84 | 0.00 | ORB-short ORB[938.75,948.30] vol=2.0x ATR=3.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 11:45:00 | 930.05 | 937.13 | 0.00 | T1 1.5R @ 930.05 |
| Target hit | 2024-12-12 15:05:00 | 930.60 | 928.64 | 0.00 | Trail-exit close>VWAP |

### Cycle 5 — SELL (started 2024-12-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 10:30:00 | 929.05 | 933.44 | 0.00 | ORB-short ORB[930.00,939.20] vol=2.2x ATR=3.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 13:40:00 | 924.11 | 931.05 | 0.00 | T1 1.5R @ 924.11 |
| Target hit | 2024-12-16 15:20:00 | 919.80 | 928.65 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — BUY (started 2024-12-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 09:30:00 | 867.00 | 858.47 | 0.00 | ORB-long ORB[851.85,862.50] vol=2.0x ATR=4.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:45:00 | 873.09 | 860.99 | 0.00 | T1 1.5R @ 873.09 |
| Stop hit — per-position SL triggered | 2024-12-19 10:00:00 | 867.00 | 862.29 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-12-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 10:10:00 | 867.90 | 861.30 | 0.00 | ORB-long ORB[847.00,858.40] vol=1.7x ATR=4.27 |
| Stop hit — per-position SL triggered | 2024-12-27 10:25:00 | 863.63 | 862.02 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-01-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:50:00 | 893.55 | 883.72 | 0.00 | ORB-long ORB[873.30,883.25] vol=1.5x ATR=4.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 10:55:00 | 899.63 | 885.55 | 0.00 | T1 1.5R @ 899.63 |
| Target hit | 2025-01-01 12:25:00 | 899.45 | 899.92 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2025-01-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 10:00:00 | 918.15 | 909.56 | 0.00 | ORB-long ORB[896.90,910.30] vol=1.5x ATR=3.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 10:10:00 | 923.82 | 913.03 | 0.00 | T1 1.5R @ 923.82 |
| Target hit | 2025-01-02 14:50:00 | 923.75 | 923.80 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — SELL (started 2025-01-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:45:00 | 870.25 | 875.95 | 0.00 | ORB-short ORB[871.50,879.60] vol=1.6x ATR=3.07 |
| Stop hit — per-position SL triggered | 2025-01-09 10:50:00 | 873.32 | 875.80 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-02-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 10:45:00 | 895.60 | 889.39 | 0.00 | ORB-long ORB[886.05,893.65] vol=4.4x ATR=2.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 10:50:00 | 900.02 | 891.61 | 0.00 | T1 1.5R @ 900.02 |
| Stop hit — per-position SL triggered | 2025-02-01 11:00:00 | 895.60 | 894.28 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-02-04 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 11:05:00 | 870.95 | 880.09 | 0.00 | ORB-short ORB[875.55,885.15] vol=3.0x ATR=2.04 |
| Stop hit — per-position SL triggered | 2025-02-04 11:10:00 | 872.99 | 879.82 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-05 11:15:00 | 894.55 | 885.84 | 0.00 | ORB-long ORB[870.40,880.75] vol=1.5x ATR=4.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-05 12:15:00 | 900.60 | 888.96 | 0.00 | T1 1.5R @ 900.60 |
| Target hit | 2025-02-05 15:20:00 | 899.05 | 895.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — SELL (started 2025-02-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 09:30:00 | 908.05 | 913.32 | 0.00 | ORB-short ORB[910.10,916.90] vol=2.4x ATR=4.12 |
| Stop hit — per-position SL triggered | 2025-02-07 09:35:00 | 912.17 | 913.06 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-02-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 10:55:00 | 865.05 | 874.89 | 0.00 | ORB-short ORB[882.55,892.50] vol=3.2x ATR=4.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 11:10:00 | 858.43 | 872.37 | 0.00 | T1 1.5R @ 858.43 |
| Stop hit — per-position SL triggered | 2025-02-10 11:15:00 | 865.05 | 871.58 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-02-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-11 09:30:00 | 845.85 | 853.45 | 0.00 | ORB-short ORB[852.35,863.95] vol=2.4x ATR=3.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:40:00 | 840.11 | 850.24 | 0.00 | T1 1.5R @ 840.11 |
| Target hit | 2025-02-11 15:20:00 | 830.85 | 838.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2025-03-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-05 09:45:00 | 793.50 | 786.77 | 0.00 | ORB-long ORB[778.10,789.90] vol=1.9x ATR=5.14 |
| Stop hit — per-position SL triggered | 2025-03-05 10:15:00 | 788.36 | 789.12 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-03-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 11:10:00 | 819.50 | 813.81 | 0.00 | ORB-long ORB[805.10,816.30] vol=2.3x ATR=2.69 |
| Stop hit — per-position SL triggered | 2025-03-07 11:50:00 | 816.81 | 815.20 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-03-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 09:55:00 | 813.60 | 803.81 | 0.00 | ORB-long ORB[794.80,805.75] vol=2.1x ATR=4.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-13 10:05:00 | 819.60 | 808.50 | 0.00 | T1 1.5R @ 819.60 |
| Stop hit — per-position SL triggered | 2025-03-13 10:50:00 | 813.60 | 811.87 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2025-04-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 09:40:00 | 888.75 | 880.59 | 0.00 | ORB-long ORB[868.25,879.95] vol=1.7x ATR=4.53 |
| Stop hit — per-position SL triggered | 2025-04-02 09:45:00 | 884.22 | 881.11 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-04-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-03 09:30:00 | 907.20 | 913.43 | 0.00 | ORB-short ORB[908.10,920.00] vol=1.6x ATR=5.16 |
| Stop hit — per-position SL triggered | 2025-04-03 10:25:00 | 912.36 | 910.75 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 11:15:00 | 989.95 | 983.40 | 0.00 | ORB-long ORB[972.05,981.65] vol=3.3x ATR=3.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-16 12:05:00 | 994.51 | 986.69 | 0.00 | T1 1.5R @ 994.51 |
| Target hit | 2025-04-16 13:30:00 | 992.70 | 992.76 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-12-02 10:45:00 | 871.65 | 2024-12-02 11:30:00 | 875.30 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-12-03 10:40:00 | 888.95 | 2024-12-03 11:20:00 | 894.03 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-12-03 10:40:00 | 888.95 | 2024-12-03 12:05:00 | 888.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-10 10:10:00 | 948.85 | 2024-12-10 11:00:00 | 952.44 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-12-12 10:50:00 | 934.95 | 2024-12-12 11:45:00 | 930.05 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-12-12 10:50:00 | 934.95 | 2024-12-12 15:05:00 | 930.60 | TARGET_HIT | 0.50 | 0.47% |
| SELL | retest1 | 2024-12-16 10:30:00 | 929.05 | 2024-12-16 13:40:00 | 924.11 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-12-16 10:30:00 | 929.05 | 2024-12-16 15:20:00 | 919.80 | TARGET_HIT | 0.50 | 1.00% |
| BUY | retest1 | 2024-12-19 09:30:00 | 867.00 | 2024-12-19 09:45:00 | 873.09 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2024-12-19 09:30:00 | 867.00 | 2024-12-19 10:00:00 | 867.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-27 10:10:00 | 867.90 | 2024-12-27 10:25:00 | 863.63 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-01-01 10:50:00 | 893.55 | 2025-01-01 10:55:00 | 899.63 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2025-01-01 10:50:00 | 893.55 | 2025-01-01 12:25:00 | 899.45 | TARGET_HIT | 0.50 | 0.66% |
| BUY | retest1 | 2025-01-02 10:00:00 | 918.15 | 2025-01-02 10:10:00 | 923.82 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-01-02 10:00:00 | 918.15 | 2025-01-02 14:50:00 | 923.75 | TARGET_HIT | 0.50 | 0.61% |
| SELL | retest1 | 2025-01-09 10:45:00 | 870.25 | 2025-01-09 10:50:00 | 873.32 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-02-01 10:45:00 | 895.60 | 2025-02-01 10:50:00 | 900.02 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-02-01 10:45:00 | 895.60 | 2025-02-01 11:00:00 | 895.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-04 11:05:00 | 870.95 | 2025-02-04 11:10:00 | 872.99 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-02-05 11:15:00 | 894.55 | 2025-02-05 12:15:00 | 900.60 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2025-02-05 11:15:00 | 894.55 | 2025-02-05 15:20:00 | 899.05 | TARGET_HIT | 0.50 | 0.50% |
| SELL | retest1 | 2025-02-07 09:30:00 | 908.05 | 2025-02-07 09:35:00 | 912.17 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-02-10 10:55:00 | 865.05 | 2025-02-10 11:10:00 | 858.43 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2025-02-10 10:55:00 | 865.05 | 2025-02-10 11:15:00 | 865.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-11 09:30:00 | 845.85 | 2025-02-11 09:40:00 | 840.11 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2025-02-11 09:30:00 | 845.85 | 2025-02-11 15:20:00 | 830.85 | TARGET_HIT | 0.50 | 1.77% |
| BUY | retest1 | 2025-03-05 09:45:00 | 793.50 | 2025-03-05 10:15:00 | 788.36 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest1 | 2025-03-07 11:10:00 | 819.50 | 2025-03-07 11:50:00 | 816.81 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-03-13 09:55:00 | 813.60 | 2025-03-13 10:05:00 | 819.60 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2025-03-13 09:55:00 | 813.60 | 2025-03-13 10:50:00 | 813.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-02 09:40:00 | 888.75 | 2025-04-02 09:45:00 | 884.22 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2025-04-03 09:30:00 | 907.20 | 2025-04-03 10:25:00 | 912.36 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2025-04-16 11:15:00 | 989.95 | 2025-04-16 12:05:00 | 994.51 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-04-16 11:15:00 | 989.95 | 2025-04-16 13:30:00 | 992.70 | TARGET_HIT | 0.50 | 0.28% |
