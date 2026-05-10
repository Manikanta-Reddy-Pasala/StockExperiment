# Zydus Lifesciences Ltd. (ZYDUSLIFE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 939.00
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
| ENTRY1 | 21 |
| ENTRY2 | 0 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 18
- **Target hits / Stop hits / Partials:** 3 / 18 / 7
- **Avg / median % per leg:** 0.05% / 0.00%
- **Sum % (uncompounded):** 1.43%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 8 | 38.1% | 3 | 13 | 5 | 0.07% | 1.6% |
| BUY @ 2nd Alert (retest1) | 21 | 8 | 38.1% | 3 | 13 | 5 | 0.07% | 1.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 2 | 28.6% | 0 | 5 | 2 | -0.02% | -0.1% |
| SELL @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 0 | 5 | 2 | -0.02% | -0.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 28 | 10 | 35.7% | 3 | 18 | 7 | 0.05% | 1.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 09:30:00 | 908.90 | 903.22 | 0.00 | ORB-long ORB[896.05,906.95] vol=2.0x ATR=2.77 |
| Stop hit — per-position SL triggered | 2026-02-12 09:35:00 | 906.13 | 904.58 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:00:00 | 903.55 | 907.16 | 0.00 | ORB-short ORB[904.15,910.85] vol=1.5x ATR=1.62 |
| Stop hit — per-position SL triggered | 2026-02-18 10:10:00 | 905.17 | 906.89 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:50:00 | 904.60 | 907.29 | 0.00 | ORB-short ORB[904.65,910.05] vol=2.2x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:35:00 | 902.14 | 906.50 | 0.00 | T1 1.5R @ 902.14 |
| Stop hit — per-position SL triggered | 2026-02-23 13:40:00 | 904.60 | 905.04 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:45:00 | 905.75 | 903.82 | 0.00 | ORB-long ORB[900.25,905.70] vol=1.8x ATR=1.88 |
| Stop hit — per-position SL triggered | 2026-02-24 10:25:00 | 903.87 | 904.13 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:35:00 | 930.35 | 927.43 | 0.00 | ORB-long ORB[922.25,928.95] vol=2.7x ATR=2.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 09:45:00 | 933.61 | 929.24 | 0.00 | T1 1.5R @ 933.61 |
| Target hit | 2026-02-26 10:40:00 | 935.15 | 936.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2026-03-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 11:05:00 | 935.85 | 932.41 | 0.00 | ORB-long ORB[923.10,935.60] vol=1.6x ATR=2.12 |
| Stop hit — per-position SL triggered | 2026-03-11 11:10:00 | 933.73 | 932.50 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-12 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:25:00 | 920.00 | 914.53 | 0.00 | ORB-long ORB[909.50,918.50] vol=1.5x ATR=2.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 11:00:00 | 924.23 | 915.97 | 0.00 | T1 1.5R @ 924.23 |
| Stop hit — per-position SL triggered | 2026-03-12 12:10:00 | 920.00 | 917.31 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:40:00 | 908.00 | 909.80 | 0.00 | ORB-short ORB[912.95,924.60] vol=2.1x ATR=2.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:45:00 | 903.68 | 909.35 | 0.00 | T1 1.5R @ 903.68 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 908.00 | 909.37 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 10:55:00 | 885.35 | 895.62 | 0.00 | ORB-short ORB[901.45,911.90] vol=2.3x ATR=2.79 |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 888.14 | 894.75 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 11:05:00 | 904.15 | 901.08 | 0.00 | ORB-long ORB[889.00,901.50] vol=1.5x ATR=1.84 |
| Stop hit — per-position SL triggered | 2026-03-18 11:35:00 | 902.31 | 901.49 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-03-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:00:00 | 867.90 | 870.03 | 0.00 | ORB-short ORB[868.40,879.90] vol=1.6x ATR=3.25 |
| Stop hit — per-position SL triggered | 2026-03-24 11:30:00 | 871.15 | 869.30 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-03-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:55:00 | 905.05 | 897.66 | 0.00 | ORB-long ORB[886.50,894.80] vol=1.7x ATR=1.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 12:05:00 | 908.01 | 900.31 | 0.00 | T1 1.5R @ 908.01 |
| Stop hit — per-position SL triggered | 2026-03-25 14:20:00 | 905.05 | 903.99 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-03-27 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 11:05:00 | 909.30 | 904.35 | 0.00 | ORB-long ORB[894.00,906.35] vol=1.7x ATR=2.33 |
| Stop hit — per-position SL triggered | 2026-03-27 11:25:00 | 906.97 | 904.69 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 910.50 | 905.16 | 0.00 | ORB-long ORB[900.00,907.05] vol=1.5x ATR=2.57 |
| Stop hit — per-position SL triggered | 2026-04-10 10:05:00 | 907.93 | 907.01 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:40:00 | 921.25 | 908.05 | 0.00 | ORB-long ORB[893.40,906.95] vol=1.9x ATR=3.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 10:55:00 | 926.21 | 910.46 | 0.00 | T1 1.5R @ 926.21 |
| Target hit | 2026-04-13 15:20:00 | 923.45 | 918.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2026-04-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 10:30:00 | 946.45 | 941.63 | 0.00 | ORB-long ORB[935.00,942.00] vol=2.1x ATR=2.22 |
| Stop hit — per-position SL triggered | 2026-04-17 12:30:00 | 944.23 | 943.66 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 11:15:00 | 932.90 | 927.14 | 0.00 | ORB-long ORB[922.05,928.50] vol=2.7x ATR=1.74 |
| Stop hit — per-position SL triggered | 2026-04-22 11:25:00 | 931.16 | 927.21 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-04-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:35:00 | 935.45 | 930.87 | 0.00 | ORB-long ORB[921.70,934.45] vol=1.5x ATR=2.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 09:40:00 | 939.41 | 933.33 | 0.00 | T1 1.5R @ 939.41 |
| Target hit | 2026-04-23 10:40:00 | 949.10 | 949.26 | 0.00 | Trail-exit close<VWAP |

### Cycle 19 — BUY (started 2026-04-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:00:00 | 949.90 | 939.52 | 0.00 | ORB-long ORB[927.45,940.00] vol=3.1x ATR=3.51 |
| Stop hit — per-position SL triggered | 2026-04-27 11:05:00 | 946.39 | 944.99 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2026-05-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 11:00:00 | 910.15 | 903.94 | 0.00 | ORB-long ORB[897.30,903.95] vol=1.8x ATR=2.05 |
| Stop hit — per-position SL triggered | 2026-05-05 11:20:00 | 908.10 | 904.90 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2026-05-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 10:10:00 | 924.75 | 920.39 | 0.00 | ORB-long ORB[915.00,920.50] vol=2.1x ATR=2.42 |
| Stop hit — per-position SL triggered | 2026-05-06 10:15:00 | 922.33 | 920.66 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-12 09:30:00 | 908.90 | 2026-02-12 09:35:00 | 906.13 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-02-18 10:00:00 | 903.55 | 2026-02-18 10:10:00 | 905.17 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-02-23 10:50:00 | 904.60 | 2026-02-23 11:35:00 | 902.14 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2026-02-23 10:50:00 | 904.60 | 2026-02-23 13:40:00 | 904.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-24 09:45:00 | 905.75 | 2026-02-24 10:25:00 | 903.87 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-02-26 09:35:00 | 930.35 | 2026-02-26 09:45:00 | 933.61 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-02-26 09:35:00 | 930.35 | 2026-02-26 10:40:00 | 935.15 | TARGET_HIT | 0.50 | 0.52% |
| BUY | retest1 | 2026-03-11 11:05:00 | 935.85 | 2026-03-11 11:10:00 | 933.73 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-03-12 10:25:00 | 920.00 | 2026-03-12 11:00:00 | 924.23 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-03-12 10:25:00 | 920.00 | 2026-03-12 12:10:00 | 920.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-13 10:40:00 | 908.00 | 2026-03-13 10:45:00 | 903.68 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2026-03-13 10:40:00 | 908.00 | 2026-03-13 10:50:00 | 908.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-16 10:55:00 | 885.35 | 2026-03-16 11:15:00 | 888.14 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-03-18 11:05:00 | 904.15 | 2026-03-18 11:35:00 | 902.31 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2026-03-24 10:00:00 | 867.90 | 2026-03-24 11:30:00 | 871.15 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-03-25 10:55:00 | 905.05 | 2026-03-25 12:05:00 | 908.01 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-03-25 10:55:00 | 905.05 | 2026-03-25 14:20:00 | 905.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-27 11:05:00 | 909.30 | 2026-03-27 11:25:00 | 906.97 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-04-10 09:30:00 | 910.50 | 2026-04-10 10:05:00 | 907.93 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2026-04-13 10:40:00 | 921.25 | 2026-04-13 10:55:00 | 926.21 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-04-13 10:40:00 | 921.25 | 2026-04-13 15:20:00 | 923.45 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2026-04-17 10:30:00 | 946.45 | 2026-04-17 12:30:00 | 944.23 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-04-22 11:15:00 | 932.90 | 2026-04-22 11:25:00 | 931.16 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-04-23 09:35:00 | 935.45 | 2026-04-23 09:40:00 | 939.41 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-04-23 09:35:00 | 935.45 | 2026-04-23 10:40:00 | 949.10 | TARGET_HIT | 0.50 | 1.46% |
| BUY | retest1 | 2026-04-27 10:00:00 | 949.90 | 2026-04-27 11:05:00 | 946.39 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-05-05 11:00:00 | 910.15 | 2026-05-05 11:20:00 | 908.10 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-05-06 10:10:00 | 924.75 | 2026-05-06 10:15:00 | 922.33 | STOP_HIT | 1.00 | -0.26% |
