# Can Fin Homes Ltd. (CANFINHOME)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 878.10
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
| ENTRY1 | 18 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 4 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 14
- **Target hits / Stop hits / Partials:** 4 / 14 / 7
- **Avg / median % per leg:** 0.16% / 0.00%
- **Sum % (uncompounded):** 4.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.11% | 1.1% |
| BUY @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.11% | 1.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 15 | 7 | 46.7% | 3 | 8 | 4 | 0.20% | 2.9% |
| SELL @ 2nd Alert (retest1) | 15 | 7 | 46.7% | 3 | 8 | 4 | 0.20% | 2.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 25 | 11 | 44.0% | 4 | 14 | 7 | 0.16% | 4.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:05:00 | 937.70 | 943.23 | 0.00 | ORB-short ORB[941.10,952.95] vol=1.7x ATR=2.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:10:00 | 933.21 | 942.23 | 0.00 | T1 1.5R @ 933.21 |
| Target hit | 2026-02-10 15:20:00 | 912.50 | 924.67 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 10:55:00 | 902.45 | 914.35 | 0.00 | ORB-short ORB[916.20,925.20] vol=3.2x ATR=3.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 11:20:00 | 896.99 | 910.97 | 0.00 | T1 1.5R @ 896.99 |
| Target hit | 2026-02-11 15:15:00 | 899.40 | 899.36 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — SELL (started 2026-02-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 10:45:00 | 866.65 | 873.11 | 0.00 | ORB-short ORB[875.00,884.80] vol=2.5x ATR=2.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 11:00:00 | 862.55 | 870.14 | 0.00 | T1 1.5R @ 862.55 |
| Stop hit — per-position SL triggered | 2026-02-13 11:10:00 | 866.65 | 868.27 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:20:00 | 894.75 | 887.23 | 0.00 | ORB-long ORB[876.00,886.10] vol=4.0x ATR=3.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 11:25:00 | 899.79 | 890.39 | 0.00 | T1 1.5R @ 899.79 |
| Target hit | 2026-02-16 15:20:00 | 903.30 | 895.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-02-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 10:10:00 | 893.85 | 897.42 | 0.00 | ORB-short ORB[896.60,905.75] vol=2.2x ATR=2.73 |
| Stop hit — per-position SL triggered | 2026-02-17 10:30:00 | 896.58 | 896.95 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:40:00 | 911.95 | 908.79 | 0.00 | ORB-long ORB[901.20,910.35] vol=1.6x ATR=2.63 |
| Stop hit — per-position SL triggered | 2026-02-18 10:30:00 | 909.32 | 910.12 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:20:00 | 899.40 | 903.21 | 0.00 | ORB-short ORB[902.25,909.75] vol=2.3x ATR=2.50 |
| Stop hit — per-position SL triggered | 2026-02-19 10:25:00 | 901.90 | 903.06 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-02-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:40:00 | 895.00 | 893.12 | 0.00 | ORB-long ORB[889.55,893.85] vol=3.3x ATR=1.78 |
| Stop hit — per-position SL triggered | 2026-02-26 09:50:00 | 893.22 | 893.21 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 10:10:00 | 868.20 | 865.17 | 0.00 | ORB-long ORB[858.40,867.65] vol=1.9x ATR=2.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 10:30:00 | 871.77 | 866.29 | 0.00 | T1 1.5R @ 871.77 |
| Stop hit — per-position SL triggered | 2026-03-11 10:35:00 | 868.20 | 866.37 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-17 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:50:00 | 852.55 | 847.04 | 0.00 | ORB-long ORB[840.00,850.00] vol=1.7x ATR=3.09 |
| Stop hit — per-position SL triggered | 2026-03-17 11:15:00 | 849.46 | 847.34 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 11:00:00 | 850.00 | 841.36 | 0.00 | ORB-long ORB[835.65,845.90] vol=3.9x ATR=2.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 11:05:00 | 854.14 | 843.83 | 0.00 | T1 1.5R @ 854.14 |
| Stop hit — per-position SL triggered | 2026-03-25 11:20:00 | 850.00 | 848.93 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-06 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-06 10:20:00 | 801.80 | 806.85 | 0.00 | ORB-short ORB[803.05,814.40] vol=2.5x ATR=3.80 |
| Stop hit — per-position SL triggered | 2026-04-06 10:45:00 | 805.60 | 806.34 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:30:00 | 855.80 | 859.89 | 0.00 | ORB-short ORB[858.00,867.35] vol=2.5x ATR=3.26 |
| Stop hit — per-position SL triggered | 2026-04-16 10:05:00 | 859.06 | 857.89 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:30:00 | 908.00 | 898.22 | 0.00 | ORB-long ORB[890.00,901.95] vol=1.7x ATR=3.78 |
| Stop hit — per-position SL triggered | 2026-04-22 09:50:00 | 904.22 | 903.47 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:15:00 | 879.80 | 890.98 | 0.00 | ORB-short ORB[894.00,905.40] vol=6.1x ATR=2.76 |
| Stop hit — per-position SL triggered | 2026-04-24 11:20:00 | 882.56 | 890.75 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2026-04-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:55:00 | 877.50 | 886.19 | 0.00 | ORB-short ORB[886.50,898.55] vol=1.8x ATR=3.27 |
| Stop hit — per-position SL triggered | 2026-04-29 13:50:00 | 880.77 | 882.52 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-04-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:10:00 | 852.35 | 861.71 | 0.00 | ORB-short ORB[861.35,874.00] vol=1.7x ATR=2.62 |
| Stop hit — per-position SL triggered | 2026-04-30 11:25:00 | 854.97 | 858.79 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-05-07 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 11:10:00 | 889.85 | 893.30 | 0.00 | ORB-short ORB[897.00,909.55] vol=3.2x ATR=2.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 11:50:00 | 886.77 | 892.03 | 0.00 | T1 1.5R @ 886.77 |
| Target hit | 2026-05-07 15:20:00 | 885.95 | 888.72 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 10:05:00 | 937.70 | 2026-02-10 10:10:00 | 933.21 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-02-10 10:05:00 | 937.70 | 2026-02-10 15:20:00 | 912.50 | TARGET_HIT | 0.50 | 2.69% |
| SELL | retest1 | 2026-02-11 10:55:00 | 902.45 | 2026-02-11 11:20:00 | 896.99 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-02-11 10:55:00 | 902.45 | 2026-02-11 15:15:00 | 899.40 | TARGET_HIT | 0.50 | 0.34% |
| SELL | retest1 | 2026-02-13 10:45:00 | 866.65 | 2026-02-13 11:00:00 | 862.55 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-02-13 10:45:00 | 866.65 | 2026-02-13 11:10:00 | 866.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-16 10:20:00 | 894.75 | 2026-02-16 11:25:00 | 899.79 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-02-16 10:20:00 | 894.75 | 2026-02-16 15:20:00 | 903.30 | TARGET_HIT | 0.50 | 0.96% |
| SELL | retest1 | 2026-02-17 10:10:00 | 893.85 | 2026-02-17 10:30:00 | 896.58 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-18 09:40:00 | 911.95 | 2026-02-18 10:30:00 | 909.32 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-19 10:20:00 | 899.40 | 2026-02-19 10:25:00 | 901.90 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-02-26 09:40:00 | 895.00 | 2026-02-26 09:50:00 | 893.22 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2026-03-11 10:10:00 | 868.20 | 2026-03-11 10:30:00 | 871.77 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-03-11 10:10:00 | 868.20 | 2026-03-11 10:35:00 | 868.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 10:50:00 | 852.55 | 2026-03-17 11:15:00 | 849.46 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-03-25 11:00:00 | 850.00 | 2026-03-25 11:05:00 | 854.14 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-03-25 11:00:00 | 850.00 | 2026-03-25 11:20:00 | 850.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-06 10:20:00 | 801.80 | 2026-04-06 10:45:00 | 805.60 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2026-04-16 09:30:00 | 855.80 | 2026-04-16 10:05:00 | 859.06 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2026-04-22 09:30:00 | 908.00 | 2026-04-22 09:50:00 | 904.22 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-04-24 11:15:00 | 879.80 | 2026-04-24 11:20:00 | 882.56 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-04-29 10:55:00 | 877.50 | 2026-04-29 13:50:00 | 880.77 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-04-30 10:10:00 | 852.35 | 2026-04-30 11:25:00 | 854.97 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-05-07 11:10:00 | 889.85 | 2026-05-07 11:50:00 | 886.77 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2026-05-07 11:10:00 | 889.85 | 2026-05-07 15:20:00 | 885.95 | TARGET_HIT | 0.50 | 0.44% |
