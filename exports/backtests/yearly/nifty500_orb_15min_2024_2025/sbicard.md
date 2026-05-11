# SBI Cards and Payment Services Ltd. (SBICARD)

## Backtest Summary

- **Window:** 2025-02-05 09:15:00 → 2026-05-08 15:25:00 (21538 bars)
- **Last close:** 645.00
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
| ENTRY1 | 26 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 5 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 21
- **Target hits / Stop hits / Partials:** 5 / 21 / 10
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 5.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 8 | 47.1% | 3 | 9 | 5 | 0.29% | 5.0% |
| BUY @ 2nd Alert (retest1) | 17 | 8 | 47.1% | 3 | 9 | 5 | 0.29% | 5.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 19 | 7 | 36.8% | 2 | 12 | 5 | 0.01% | 0.3% |
| SELL @ 2nd Alert (retest1) | 19 | 7 | 36.8% | 2 | 12 | 5 | 0.01% | 0.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 36 | 15 | 41.7% | 5 | 21 | 10 | 0.15% | 5.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-05 11:15:00 | 824.75 | 831.66 | 0.00 | ORB-short ORB[833.75,839.80] vol=1.9x ATR=2.41 |
| Stop hit — per-position SL triggered | 2025-02-05 11:20:00 | 827.16 | 831.51 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-02-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 10:15:00 | 800.95 | 807.97 | 0.00 | ORB-short ORB[807.40,812.70] vol=3.0x ATR=3.14 |
| Stop hit — per-position SL triggered | 2025-02-07 10:20:00 | 804.09 | 807.77 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-02-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-12 09:30:00 | 775.80 | 778.57 | 0.00 | ORB-short ORB[776.65,784.30] vol=1.5x ATR=2.30 |
| Stop hit — per-position SL triggered | 2025-02-12 09:35:00 | 778.10 | 778.35 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-02-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-14 09:45:00 | 870.00 | 864.15 | 0.00 | ORB-long ORB[858.15,866.95] vol=2.7x ATR=2.79 |
| Stop hit — per-position SL triggered | 2025-02-14 09:50:00 | 867.21 | 865.00 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-19 10:15:00 | 859.20 | 861.35 | 0.00 | ORB-short ORB[859.85,867.95] vol=1.8x ATR=2.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-19 10:40:00 | 855.47 | 860.77 | 0.00 | T1 1.5R @ 855.47 |
| Stop hit — per-position SL triggered | 2025-02-19 11:30:00 | 859.20 | 858.52 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2025-02-20 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-20 10:00:00 | 851.80 | 854.27 | 0.00 | ORB-short ORB[852.65,859.45] vol=1.9x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-20 11:25:00 | 848.61 | 852.30 | 0.00 | T1 1.5R @ 848.61 |
| Stop hit — per-position SL triggered | 2025-02-20 11:35:00 | 851.80 | 852.18 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-02-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 09:40:00 | 842.50 | 845.80 | 0.00 | ORB-short ORB[843.55,850.75] vol=1.9x ATR=1.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-21 09:45:00 | 839.82 | 844.46 | 0.00 | T1 1.5R @ 839.82 |
| Target hit | 2025-02-21 14:45:00 | 833.15 | 832.74 | 0.00 | Trail-exit close>VWAP |

### Cycle 8 — BUY (started 2025-02-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-27 09:40:00 | 858.80 | 853.27 | 0.00 | ORB-long ORB[844.00,856.20] vol=1.7x ATR=2.73 |
| Stop hit — per-position SL triggered | 2025-02-27 09:55:00 | 856.07 | 854.30 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-03-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-06 09:35:00 | 839.45 | 842.48 | 0.00 | ORB-short ORB[840.00,850.00] vol=1.7x ATR=2.35 |
| Stop hit — per-position SL triggered | 2025-03-06 10:05:00 | 841.80 | 840.63 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-03-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 10:40:00 | 845.35 | 840.62 | 0.00 | ORB-long ORB[834.00,841.00] vol=2.4x ATR=2.04 |
| Stop hit — per-position SL triggered | 2025-03-07 10:45:00 | 843.31 | 840.90 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-03-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-13 10:50:00 | 831.85 | 835.71 | 0.00 | ORB-short ORB[834.00,842.00] vol=1.7x ATR=1.88 |
| Stop hit — per-position SL triggered | 2025-03-13 11:15:00 | 833.73 | 835.19 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-03-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-18 10:05:00 | 842.75 | 844.98 | 0.00 | ORB-short ORB[844.05,851.65] vol=1.6x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 10:50:00 | 840.11 | 843.80 | 0.00 | T1 1.5R @ 840.11 |
| Target hit | 2025-03-18 14:20:00 | 841.45 | 841.29 | 0.00 | Trail-exit close>VWAP |

### Cycle 13 — BUY (started 2025-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 10:15:00 | 849.05 | 845.63 | 0.00 | ORB-long ORB[839.70,849.00] vol=3.9x ATR=2.38 |
| Stop hit — per-position SL triggered | 2025-03-19 10:25:00 | 846.67 | 845.76 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-03-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 11:05:00 | 850.10 | 845.77 | 0.00 | ORB-long ORB[844.50,850.00] vol=2.5x ATR=1.93 |
| Stop hit — per-position SL triggered | 2025-03-20 11:30:00 | 848.17 | 846.45 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-03-21 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-21 11:10:00 | 852.65 | 859.02 | 0.00 | ORB-short ORB[858.35,866.00] vol=1.6x ATR=2.11 |
| Stop hit — per-position SL triggered | 2025-03-21 11:20:00 | 854.76 | 858.81 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2025-03-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 09:35:00 | 867.50 | 861.02 | 0.00 | ORB-long ORB[852.05,862.00] vol=4.3x ATR=2.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-25 09:40:00 | 871.89 | 863.98 | 0.00 | T1 1.5R @ 871.89 |
| Stop hit — per-position SL triggered | 2025-03-25 10:30:00 | 867.50 | 869.92 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-03-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 11:10:00 | 874.40 | 869.18 | 0.00 | ORB-long ORB[863.10,869.35] vol=1.7x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-26 11:15:00 | 877.65 | 870.27 | 0.00 | T1 1.5R @ 877.65 |
| Stop hit — per-position SL triggered | 2025-03-26 11:55:00 | 874.40 | 871.64 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-04-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-01 11:00:00 | 850.50 | 869.47 | 0.00 | ORB-short ORB[872.00,881.00] vol=1.8x ATR=3.19 |
| Stop hit — per-position SL triggered | 2025-04-01 11:25:00 | 853.69 | 866.81 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-04-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 09:30:00 | 863.15 | 859.93 | 0.00 | ORB-long ORB[856.30,862.10] vol=1.7x ATR=2.40 |
| Stop hit — per-position SL triggered | 2025-04-02 09:35:00 | 860.75 | 860.33 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-04-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-09 10:30:00 | 824.25 | 841.14 | 0.00 | ORB-short ORB[840.20,851.80] vol=1.8x ATR=3.43 |
| Stop hit — per-position SL triggered | 2025-04-09 10:35:00 | 827.68 | 840.66 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-04-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 10:25:00 | 868.45 | 865.52 | 0.00 | ORB-long ORB[857.80,866.00] vol=2.2x ATR=2.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-15 10:45:00 | 871.83 | 866.91 | 0.00 | T1 1.5R @ 871.83 |
| Target hit | 2025-04-15 15:20:00 | 883.75 | 875.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2025-04-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 10:20:00 | 892.40 | 888.20 | 0.00 | ORB-long ORB[882.25,890.45] vol=1.6x ATR=2.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-17 10:35:00 | 895.93 | 889.72 | 0.00 | T1 1.5R @ 895.93 |
| Target hit | 2025-04-17 15:20:00 | 906.25 | 900.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 23 — SELL (started 2025-04-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-22 10:55:00 | 904.05 | 908.12 | 0.00 | ORB-short ORB[905.10,915.00] vol=2.1x ATR=2.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-22 11:10:00 | 900.74 | 906.75 | 0.00 | T1 1.5R @ 900.74 |
| Stop hit — per-position SL triggered | 2025-04-22 11:35:00 | 904.05 | 904.58 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-05-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-02 10:55:00 | 888.60 | 886.47 | 0.00 | ORB-long ORB[877.00,884.80] vol=5.9x ATR=2.72 |
| Stop hit — per-position SL triggered | 2025-05-02 11:30:00 | 885.88 | 886.70 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-05-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 10:25:00 | 891.60 | 886.34 | 0.00 | ORB-long ORB[877.00,885.80] vol=2.4x ATR=2.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 10:35:00 | 894.90 | 887.97 | 0.00 | T1 1.5R @ 894.90 |
| Target hit | 2025-05-05 15:20:00 | 905.95 | 899.01 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-06 11:15:00 | 892.30 | 897.15 | 0.00 | ORB-short ORB[901.45,910.75] vol=5.7x ATR=2.42 |
| Stop hit — per-position SL triggered | 2025-05-06 11:45:00 | 894.72 | 896.10 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-02-05 11:15:00 | 824.75 | 2025-02-05 11:20:00 | 827.16 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-02-07 10:15:00 | 800.95 | 2025-02-07 10:20:00 | 804.09 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-02-12 09:30:00 | 775.80 | 2025-02-12 09:35:00 | 778.10 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-02-14 09:45:00 | 870.00 | 2025-02-14 09:50:00 | 867.21 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-02-19 10:15:00 | 859.20 | 2025-02-19 10:40:00 | 855.47 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-02-19 10:15:00 | 859.20 | 2025-02-19 11:30:00 | 859.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-20 10:00:00 | 851.80 | 2025-02-20 11:25:00 | 848.61 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-02-20 10:00:00 | 851.80 | 2025-02-20 11:35:00 | 851.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-21 09:40:00 | 842.50 | 2025-02-21 09:45:00 | 839.82 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-02-21 09:40:00 | 842.50 | 2025-02-21 14:45:00 | 833.15 | TARGET_HIT | 0.50 | 1.11% |
| BUY | retest1 | 2025-02-27 09:40:00 | 858.80 | 2025-02-27 09:55:00 | 856.07 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-03-06 09:35:00 | 839.45 | 2025-03-06 10:05:00 | 841.80 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-03-07 10:40:00 | 845.35 | 2025-03-07 10:45:00 | 843.31 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-03-13 10:50:00 | 831.85 | 2025-03-13 11:15:00 | 833.73 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-03-18 10:05:00 | 842.75 | 2025-03-18 10:50:00 | 840.11 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-03-18 10:05:00 | 842.75 | 2025-03-18 14:20:00 | 841.45 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2025-03-19 10:15:00 | 849.05 | 2025-03-19 10:25:00 | 846.67 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-03-20 11:05:00 | 850.10 | 2025-03-20 11:30:00 | 848.17 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-03-21 11:10:00 | 852.65 | 2025-03-21 11:20:00 | 854.76 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-03-25 09:35:00 | 867.50 | 2025-03-25 09:40:00 | 871.89 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2025-03-25 09:35:00 | 867.50 | 2025-03-25 10:30:00 | 867.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-26 11:10:00 | 874.40 | 2025-03-26 11:15:00 | 877.65 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-03-26 11:10:00 | 874.40 | 2025-03-26 11:55:00 | 874.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-01 11:00:00 | 850.50 | 2025-04-01 11:25:00 | 853.69 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-04-02 09:30:00 | 863.15 | 2025-04-02 09:35:00 | 860.75 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-04-09 10:30:00 | 824.25 | 2025-04-09 10:35:00 | 827.68 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-04-15 10:25:00 | 868.45 | 2025-04-15 10:45:00 | 871.83 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-04-15 10:25:00 | 868.45 | 2025-04-15 15:20:00 | 883.75 | TARGET_HIT | 0.50 | 1.76% |
| BUY | retest1 | 2025-04-17 10:20:00 | 892.40 | 2025-04-17 10:35:00 | 895.93 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2025-04-17 10:20:00 | 892.40 | 2025-04-17 15:20:00 | 906.25 | TARGET_HIT | 0.50 | 1.55% |
| SELL | retest1 | 2025-04-22 10:55:00 | 904.05 | 2025-04-22 11:10:00 | 900.74 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2025-04-22 10:55:00 | 904.05 | 2025-04-22 11:35:00 | 904.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-02 10:55:00 | 888.60 | 2025-05-02 11:30:00 | 885.88 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-05-05 10:25:00 | 891.60 | 2025-05-05 10:35:00 | 894.90 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-05-05 10:25:00 | 891.60 | 2025-05-05 15:20:00 | 905.95 | TARGET_HIT | 0.50 | 1.61% |
| SELL | retest1 | 2025-05-06 11:15:00 | 892.30 | 2025-05-06 11:45:00 | 894.72 | STOP_HIT | 1.00 | -0.27% |
