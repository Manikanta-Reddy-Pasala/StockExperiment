# C.E. Info Systems Ltd. (MAPMYINDIA)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 957.00
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
| ENTRY1 | 12 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 10
- **Target hits / Stop hits / Partials:** 2 / 10 / 4
- **Avg / median % per leg:** 0.11% / 0.00%
- **Sum % (uncompounded):** 1.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.02% | 0.3% |
| BUY @ 2nd Alert (retest1) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.02% | 0.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.30% | 1.5% |
| SELL @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.30% | 1.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 16 | 6 | 37.5% | 2 | 10 | 4 | 0.11% | 1.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 10:05:00 | 1295.40 | 1289.70 | 0.00 | ORB-long ORB[1278.30,1295.00] vol=2.7x ATR=4.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 10:10:00 | 1302.78 | 1294.38 | 0.00 | T1 1.5R @ 1302.78 |
| Target hit | 2026-02-10 11:30:00 | 1308.90 | 1311.98 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — SELL (started 2026-02-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:55:00 | 1103.30 | 1109.16 | 0.00 | ORB-short ORB[1108.00,1119.30] vol=2.3x ATR=3.57 |
| Stop hit — per-position SL triggered | 2026-02-23 11:00:00 | 1106.87 | 1109.02 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-26 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:45:00 | 1075.80 | 1062.32 | 0.00 | ORB-long ORB[1049.00,1061.90] vol=2.7x ATR=5.36 |
| Stop hit — per-position SL triggered | 2026-02-26 09:55:00 | 1070.44 | 1063.09 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-04 10:45:00 | 1009.90 | 1003.77 | 0.00 | ORB-long ORB[991.20,1003.00] vol=3.7x ATR=4.47 |
| Stop hit — per-position SL triggered | 2026-03-04 11:30:00 | 1005.43 | 1008.43 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:55:00 | 998.80 | 1003.39 | 0.00 | ORB-short ORB[1000.00,1009.00] vol=5.0x ATR=2.42 |
| Stop hit — per-position SL triggered | 2026-03-06 13:55:00 | 1001.22 | 1002.03 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:20:00 | 951.70 | 957.47 | 0.00 | ORB-short ORB[953.00,965.30] vol=2.1x ATR=4.53 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 956.23 | 956.90 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 11:15:00 | 885.30 | 876.21 | 0.00 | ORB-long ORB[868.90,878.70] vol=3.5x ATR=3.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 11:45:00 | 890.42 | 877.90 | 0.00 | T1 1.5R @ 890.42 |
| Stop hit — per-position SL triggered | 2026-03-20 11:50:00 | 885.30 | 878.77 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:00:00 | 893.80 | 887.93 | 0.00 | ORB-long ORB[879.70,890.50] vol=1.6x ATR=5.21 |
| Stop hit — per-position SL triggered | 2026-03-25 10:20:00 | 888.59 | 888.52 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-04-15 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 10:40:00 | 956.55 | 945.80 | 0.00 | ORB-long ORB[935.00,948.65] vol=2.9x ATR=4.65 |
| Stop hit — per-position SL triggered | 2026-04-15 10:50:00 | 951.90 | 946.48 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-04-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 09:30:00 | 939.60 | 946.48 | 0.00 | ORB-short ORB[943.40,955.90] vol=2.4x ATR=4.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:40:00 | 933.00 | 941.61 | 0.00 | T1 1.5R @ 933.00 |
| Target hit | 2026-04-16 14:45:00 | 922.55 | 922.03 | 0.00 | Trail-exit close>VWAP |

### Cycle 11 — BUY (started 2026-05-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:00:00 | 940.10 | 932.48 | 0.00 | ORB-long ORB[922.80,934.65] vol=1.9x ATR=4.24 |
| Stop hit — per-position SL triggered | 2026-05-04 10:10:00 | 935.86 | 933.07 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-06 09:30:00 | 957.85 | 951.99 | 0.00 | ORB-long ORB[943.60,950.05] vol=6.2x ATR=3.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:35:00 | 962.99 | 955.94 | 0.00 | T1 1.5R @ 962.99 |
| Stop hit — per-position SL triggered | 2026-05-06 10:05:00 | 957.85 | 960.32 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-10 10:05:00 | 1295.40 | 2026-02-10 10:10:00 | 1302.78 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2026-02-10 10:05:00 | 1295.40 | 2026-02-10 11:30:00 | 1308.90 | TARGET_HIT | 0.50 | 1.04% |
| SELL | retest1 | 2026-02-23 10:55:00 | 1103.30 | 2026-02-23 11:00:00 | 1106.87 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2026-02-26 09:45:00 | 1075.80 | 2026-02-26 09:55:00 | 1070.44 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2026-03-04 10:45:00 | 1009.90 | 2026-03-04 11:30:00 | 1005.43 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-03-06 10:55:00 | 998.80 | 2026-03-06 13:55:00 | 1001.22 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-03-13 10:20:00 | 951.70 | 2026-03-13 10:50:00 | 956.23 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2026-03-20 11:15:00 | 885.30 | 2026-03-20 11:45:00 | 890.42 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-03-20 11:15:00 | 885.30 | 2026-03-20 11:50:00 | 885.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-25 10:00:00 | 893.80 | 2026-03-25 10:20:00 | 888.59 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2026-04-15 10:40:00 | 956.55 | 2026-04-15 10:50:00 | 951.90 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2026-04-16 09:30:00 | 939.60 | 2026-04-16 09:40:00 | 933.00 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2026-04-16 09:30:00 | 939.60 | 2026-04-16 14:45:00 | 922.55 | TARGET_HIT | 0.50 | 1.81% |
| BUY | retest1 | 2026-05-04 10:00:00 | 940.10 | 2026-05-04 10:10:00 | 935.86 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-05-06 09:30:00 | 957.85 | 2026-05-06 09:35:00 | 962.99 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-05-06 09:30:00 | 957.85 | 2026-05-06 10:05:00 | 957.85 | STOP_HIT | 0.50 | 0.00% |
