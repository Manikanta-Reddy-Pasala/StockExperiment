# Jubilant Pharmova Ltd. (JUBLPHARMA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1009.00
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
| ENTRY1 | 14 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 4 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 10
- **Target hits / Stop hits / Partials:** 4 / 10 / 5
- **Avg / median % per leg:** 0.05% / 0.00%
- **Sum % (uncompounded):** 0.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 4 | 40.0% | 2 | 6 | 2 | -0.05% | -0.5% |
| BUY @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 2 | 6 | 2 | -0.05% | -0.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 5 | 55.6% | 2 | 4 | 3 | 0.16% | 1.4% |
| SELL @ 2nd Alert (retest1) | 9 | 5 | 55.6% | 2 | 4 | 3 | 0.16% | 1.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 9 | 47.4% | 4 | 10 | 5 | 0.05% | 0.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 11:05:00 | 937.85 | 941.27 | 0.00 | ORB-short ORB[940.30,949.90] vol=1.5x ATR=3.38 |
| Stop hit — per-position SL triggered | 2026-02-10 12:30:00 | 941.23 | 940.57 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2026-02-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 11:00:00 | 923.10 | 927.40 | 0.00 | ORB-short ORB[926.00,938.95] vol=1.5x ATR=2.00 |
| Stop hit — per-position SL triggered | 2026-02-12 11:15:00 | 925.10 | 926.45 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-18 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:20:00 | 898.00 | 902.97 | 0.00 | ORB-short ORB[898.60,908.45] vol=3.7x ATR=3.11 |
| Stop hit — per-position SL triggered | 2026-02-18 11:50:00 | 901.11 | 901.65 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-27 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:30:00 | 875.60 | 878.57 | 0.00 | ORB-short ORB[876.30,884.90] vol=1.8x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:00:00 | 872.42 | 877.19 | 0.00 | T1 1.5R @ 872.42 |
| Target hit | 2026-02-27 13:10:00 | 870.60 | 870.40 | 0.00 | Trail-exit close>VWAP |

### Cycle 5 — BUY (started 2026-03-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 10:20:00 | 820.70 | 816.31 | 0.00 | ORB-long ORB[811.80,819.20] vol=2.2x ATR=3.00 |
| Stop hit — per-position SL triggered | 2026-03-11 10:35:00 | 817.70 | 816.40 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-19 11:10:00 | 851.50 | 846.07 | 0.00 | ORB-long ORB[844.55,850.05] vol=1.6x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 11:40:00 | 855.14 | 847.30 | 0.00 | T1 1.5R @ 855.14 |
| Target hit | 2026-03-19 13:05:00 | 852.35 | 852.51 | 0.00 | Trail-exit close<VWAP |

### Cycle 7 — BUY (started 2026-03-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:05:00 | 883.60 | 873.78 | 0.00 | ORB-long ORB[861.00,873.85] vol=3.2x ATR=4.66 |
| Stop hit — per-position SL triggered | 2026-03-25 11:15:00 | 878.94 | 876.84 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-04-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 11:00:00 | 892.55 | 887.28 | 0.00 | ORB-long ORB[875.05,881.10] vol=11.8x ATR=3.24 |
| Stop hit — per-position SL triggered | 2026-04-10 11:30:00 | 889.31 | 888.32 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 10:45:00 | 912.10 | 908.25 | 0.00 | ORB-long ORB[903.05,909.55] vol=6.6x ATR=2.09 |
| Stop hit — per-position SL triggered | 2026-04-21 10:55:00 | 910.01 | 908.57 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 09:35:00 | 911.40 | 909.80 | 0.00 | ORB-long ORB[900.15,910.30] vol=4.8x ATR=2.70 |
| Stop hit — per-position SL triggered | 2026-04-22 09:50:00 | 908.70 | 909.80 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-23 09:30:00 | 935.50 | 931.92 | 0.00 | ORB-long ORB[921.20,932.45] vol=4.3x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 09:40:00 | 939.81 | 933.96 | 0.00 | T1 1.5R @ 939.81 |
| Target hit | 2026-04-23 10:25:00 | 940.05 | 941.91 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — SELL (started 2026-04-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 09:45:00 | 945.70 | 950.42 | 0.00 | ORB-short ORB[948.35,958.00] vol=2.0x ATR=2.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 09:55:00 | 942.10 | 947.99 | 0.00 | T1 1.5R @ 942.10 |
| Target hit | 2026-04-28 12:55:00 | 939.60 | 936.57 | 0.00 | Trail-exit close>VWAP |

### Cycle 13 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:15:00 | 941.00 | 936.07 | 0.00 | ORB-long ORB[927.25,936.10] vol=2.1x ATR=2.17 |
| Stop hit — per-position SL triggered | 2026-05-04 11:35:00 | 938.83 | 937.55 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-05-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 11:00:00 | 955.40 | 958.28 | 0.00 | ORB-short ORB[957.00,964.00] vol=1.8x ATR=2.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 11:10:00 | 951.72 | 957.96 | 0.00 | T1 1.5R @ 951.72 |
| Stop hit — per-position SL triggered | 2026-05-07 11:30:00 | 955.40 | 957.64 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 11:05:00 | 937.85 | 2026-02-10 12:30:00 | 941.23 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-12 11:00:00 | 923.10 | 2026-02-12 11:15:00 | 925.10 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-02-18 10:20:00 | 898.00 | 2026-02-18 11:50:00 | 901.11 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-02-27 10:30:00 | 875.60 | 2026-02-27 11:00:00 | 872.42 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-02-27 10:30:00 | 875.60 | 2026-02-27 13:10:00 | 870.60 | TARGET_HIT | 0.50 | 0.57% |
| BUY | retest1 | 2026-03-11 10:20:00 | 820.70 | 2026-03-11 10:35:00 | 817.70 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-03-19 11:10:00 | 851.50 | 2026-03-19 11:40:00 | 855.14 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-03-19 11:10:00 | 851.50 | 2026-03-19 13:05:00 | 852.35 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2026-03-25 10:05:00 | 883.60 | 2026-03-25 11:15:00 | 878.94 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2026-04-10 11:00:00 | 892.55 | 2026-04-10 11:30:00 | 889.31 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-04-21 10:45:00 | 912.10 | 2026-04-21 10:55:00 | 910.01 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-04-22 09:35:00 | 911.40 | 2026-04-22 09:50:00 | 908.70 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-04-23 09:30:00 | 935.50 | 2026-04-23 09:40:00 | 939.81 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-04-23 09:30:00 | 935.50 | 2026-04-23 10:25:00 | 940.05 | TARGET_HIT | 0.50 | 0.49% |
| SELL | retest1 | 2026-04-28 09:45:00 | 945.70 | 2026-04-28 09:55:00 | 942.10 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-04-28 09:45:00 | 945.70 | 2026-04-28 12:55:00 | 939.60 | TARGET_HIT | 0.50 | 0.65% |
| BUY | retest1 | 2026-05-04 11:15:00 | 941.00 | 2026-05-04 11:35:00 | 938.83 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-05-07 11:00:00 | 955.40 | 2026-05-07 11:10:00 | 951.72 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2026-05-07 11:00:00 | 955.40 | 2026-05-07 11:30:00 | 955.40 | STOP_HIT | 0.50 | 0.00% |
