# Godrej Industries Ltd. (GODREJIND)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1202.00
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
| ENTRY1 | 15 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 12
- **Target hits / Stop hits / Partials:** 3 / 12 / 7
- **Avg / median % per leg:** 0.28% / 0.00%
- **Sum % (uncompounded):** 6.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.10% | 0.7% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.10% | 0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 15 | 7 | 46.7% | 2 | 8 | 5 | 0.37% | 5.6% |
| SELL @ 2nd Alert (retest1) | 15 | 7 | 46.7% | 2 | 8 | 5 | 0.37% | 5.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 10 | 45.5% | 3 | 12 | 7 | 0.28% | 6.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:55:00 | 979.45 | 970.88 | 0.00 | ORB-long ORB[965.90,974.00] vol=2.0x ATR=2.64 |
| Stop hit — per-position SL triggered | 2026-02-17 11:00:00 | 976.81 | 973.77 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 11:15:00 | 976.70 | 979.30 | 0.00 | ORB-short ORB[979.65,986.95] vol=2.5x ATR=1.47 |
| Stop hit — per-position SL triggered | 2026-02-18 11:40:00 | 978.17 | 979.10 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 11:05:00 | 1033.65 | 1038.87 | 0.00 | ORB-short ORB[1038.70,1052.00] vol=5.1x ATR=2.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 12:40:00 | 1029.80 | 1037.09 | 0.00 | T1 1.5R @ 1029.80 |
| Stop hit — per-position SL triggered | 2026-02-25 13:00:00 | 1033.65 | 1034.75 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:50:00 | 933.30 | 937.94 | 0.00 | ORB-short ORB[934.45,943.10] vol=2.5x ATR=2.71 |
| Stop hit — per-position SL triggered | 2026-03-05 11:05:00 | 936.01 | 937.61 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 932.65 | 938.02 | 0.00 | ORB-short ORB[934.80,942.80] vol=1.8x ATR=2.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 10:50:00 | 929.12 | 937.70 | 0.00 | T1 1.5R @ 929.12 |
| Stop hit — per-position SL triggered | 2026-03-06 11:20:00 | 932.65 | 937.27 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 10:30:00 | 915.00 | 923.54 | 0.00 | ORB-short ORB[918.95,930.85] vol=3.5x ATR=3.63 |
| Target hit | 2026-03-10 15:20:00 | 912.45 | 915.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — SELL (started 2026-03-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:25:00 | 909.55 | 916.17 | 0.00 | ORB-short ORB[912.10,921.45] vol=2.3x ATR=3.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:20:00 | 904.75 | 912.38 | 0.00 | T1 1.5R @ 904.75 |
| Stop hit — per-position SL triggered | 2026-03-11 12:20:00 | 909.55 | 911.20 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:50:00 | 878.00 | 880.98 | 0.00 | ORB-short ORB[882.55,895.00] vol=3.3x ATR=3.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:15:00 | 873.36 | 879.05 | 0.00 | T1 1.5R @ 873.36 |
| Target hit | 2026-03-13 15:20:00 | 840.65 | 852.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — SELL (started 2026-03-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 09:45:00 | 779.45 | 788.60 | 0.00 | ORB-short ORB[790.80,802.20] vol=5.0x ATR=4.77 |
| Stop hit — per-position SL triggered | 2026-03-24 11:45:00 | 784.22 | 783.16 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:45:00 | 927.00 | 922.06 | 0.00 | ORB-long ORB[917.40,926.10] vol=2.2x ATR=2.93 |
| Stop hit — per-position SL triggered | 2026-04-21 09:50:00 | 924.07 | 922.33 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-22 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:45:00 | 933.05 | 931.13 | 0.00 | ORB-long ORB[925.45,932.45] vol=1.9x ATR=3.28 |
| Stop hit — per-position SL triggered | 2026-04-22 09:55:00 | 929.77 | 931.29 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:30:00 | 993.40 | 1001.71 | 0.00 | ORB-short ORB[997.05,1009.00] vol=3.0x ATR=2.96 |
| Stop hit — per-position SL triggered | 2026-04-28 09:50:00 | 996.36 | 999.22 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-04-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 09:50:00 | 976.55 | 985.17 | 0.00 | ORB-short ORB[983.80,998.00] vol=1.5x ATR=3.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:00:00 | 970.73 | 975.89 | 0.00 | T1 1.5R @ 970.73 |
| Stop hit — per-position SL triggered | 2026-04-29 13:20:00 | 976.55 | 966.60 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-05-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:30:00 | 976.25 | 969.13 | 0.00 | ORB-long ORB[960.45,972.80] vol=2.0x ATR=4.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 09:35:00 | 982.69 | 972.87 | 0.00 | T1 1.5R @ 982.69 |
| Stop hit — per-position SL triggered | 2026-05-05 10:10:00 | 976.25 | 975.53 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-05-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:00:00 | 999.05 | 993.22 | 0.00 | ORB-long ORB[985.00,995.00] vol=4.7x ATR=4.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 10:15:00 | 1005.19 | 995.31 | 0.00 | T1 1.5R @ 1005.19 |
| Target hit | 2026-05-06 12:40:00 | 1002.40 | 1002.80 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-17 09:55:00 | 979.45 | 2026-02-17 11:00:00 | 976.81 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-02-18 11:15:00 | 976.70 | 2026-02-18 11:40:00 | 978.17 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2026-02-25 11:05:00 | 1033.65 | 2026-02-25 12:40:00 | 1029.80 | PARTIAL | 0.50 | 0.37% |
| SELL | retest1 | 2026-02-25 11:05:00 | 1033.65 | 2026-02-25 13:00:00 | 1033.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-05 10:50:00 | 933.30 | 2026-03-05 11:05:00 | 936.01 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-03-06 10:45:00 | 932.65 | 2026-03-06 10:50:00 | 929.12 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-03-06 10:45:00 | 932.65 | 2026-03-06 11:20:00 | 932.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-10 10:30:00 | 915.00 | 2026-03-10 15:20:00 | 912.45 | TARGET_HIT | 1.00 | 0.28% |
| SELL | retest1 | 2026-03-11 10:25:00 | 909.55 | 2026-03-11 11:20:00 | 904.75 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-03-11 10:25:00 | 909.55 | 2026-03-11 12:20:00 | 909.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-13 09:50:00 | 878.00 | 2026-03-13 10:15:00 | 873.36 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-03-13 09:50:00 | 878.00 | 2026-03-13 15:20:00 | 840.65 | TARGET_HIT | 0.50 | 4.25% |
| SELL | retest1 | 2026-03-24 09:45:00 | 779.45 | 2026-03-24 11:45:00 | 784.22 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest1 | 2026-04-21 09:45:00 | 927.00 | 2026-04-21 09:50:00 | 924.07 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-04-22 09:45:00 | 933.05 | 2026-04-22 09:55:00 | 929.77 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-28 09:30:00 | 993.40 | 2026-04-28 09:50:00 | 996.36 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-04-29 09:50:00 | 976.55 | 2026-04-29 10:00:00 | 970.73 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-04-29 09:50:00 | 976.55 | 2026-04-29 13:20:00 | 976.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-05 09:30:00 | 976.25 | 2026-05-05 09:35:00 | 982.69 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-05-05 09:30:00 | 976.25 | 2026-05-05 10:10:00 | 976.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-06 10:00:00 | 999.05 | 2026-05-06 10:15:00 | 1005.19 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2026-05-06 10:00:00 | 999.05 | 2026-05-06 12:40:00 | 1002.40 | TARGET_HIT | 0.50 | 0.34% |
