# Sai Life Sciences Ltd. (SAILIFE)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1117.90
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
| ENTRY1 | 8 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 7
- **Target hits / Stop hits / Partials:** 1 / 7 / 3
- **Avg / median % per leg:** 0.14% / 0.00%
- **Sum % (uncompounded):** 1.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.29% | 2.0% |
| BUY @ 2nd Alert (retest1) | 7 | 3 | 42.9% | 1 | 4 | 2 | 0.29% | 2.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.11% | -0.4% |
| SELL @ 2nd Alert (retest1) | 4 | 1 | 25.0% | 0 | 3 | 1 | -0.11% | -0.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 11 | 4 | 36.4% | 1 | 7 | 3 | 0.14% | 1.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 09:40:00 | 941.70 | 935.96 | 0.00 | ORB-long ORB[930.00,940.30] vol=1.5x ATR=2.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 10:00:00 | 945.32 | 938.63 | 0.00 | T1 1.5R @ 945.32 |
| Stop hit — per-position SL triggered | 2026-02-20 10:05:00 | 941.70 | 938.77 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:10:00 | 943.90 | 933.43 | 0.00 | ORB-long ORB[920.15,933.40] vol=1.8x ATR=3.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:40:00 | 949.55 | 939.65 | 0.00 | T1 1.5R @ 949.55 |
| Target hit | 2026-02-26 15:20:00 | 966.00 | 953.52 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-03-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:05:00 | 976.30 | 989.66 | 0.00 | ORB-short ORB[991.55,1005.40] vol=2.2x ATR=4.43 |
| Stop hit — per-position SL triggered | 2026-03-13 10:10:00 | 980.73 | 989.26 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-03-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:50:00 | 992.10 | 983.76 | 0.00 | ORB-long ORB[969.00,982.05] vol=2.7x ATR=4.98 |
| Stop hit — per-position SL triggered | 2026-03-17 09:55:00 | 987.12 | 984.36 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-04-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-07 09:40:00 | 929.45 | 934.64 | 0.00 | ORB-short ORB[934.05,947.20] vol=1.9x ATR=3.87 |
| Stop hit — per-position SL triggered | 2026-04-07 09:55:00 | 933.32 | 933.84 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-04-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-21 10:20:00 | 950.55 | 955.51 | 0.00 | ORB-short ORB[953.30,961.45] vol=2.1x ATR=2.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:35:00 | 946.55 | 953.31 | 0.00 | T1 1.5R @ 946.55 |
| Stop hit — per-position SL triggered | 2026-04-21 11:00:00 | 950.55 | 952.01 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-30 09:40:00 | 1080.55 | 1070.05 | 0.00 | ORB-long ORB[1055.30,1070.85] vol=4.1x ATR=5.72 |
| Stop hit — per-position SL triggered | 2026-04-30 09:50:00 | 1074.83 | 1072.50 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-05-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:45:00 | 1090.70 | 1084.30 | 0.00 | ORB-long ORB[1080.00,1089.00] vol=1.5x ATR=3.15 |
| Stop hit — per-position SL triggered | 2026-05-07 10:55:00 | 1087.55 | 1084.87 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-20 09:40:00 | 941.70 | 2026-02-20 10:00:00 | 945.32 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2026-02-20 09:40:00 | 941.70 | 2026-02-20 10:05:00 | 941.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-26 10:10:00 | 943.90 | 2026-02-26 10:40:00 | 949.55 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-02-26 10:10:00 | 943.90 | 2026-02-26 15:20:00 | 966.00 | TARGET_HIT | 0.50 | 2.34% |
| SELL | retest1 | 2026-03-13 10:05:00 | 976.30 | 2026-03-13 10:10:00 | 980.73 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2026-03-17 09:50:00 | 992.10 | 2026-03-17 09:55:00 | 987.12 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2026-04-07 09:40:00 | 929.45 | 2026-04-07 09:55:00 | 933.32 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-04-21 10:20:00 | 950.55 | 2026-04-21 10:35:00 | 946.55 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-04-21 10:20:00 | 950.55 | 2026-04-21 11:00:00 | 950.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-30 09:40:00 | 1080.55 | 2026-04-30 09:50:00 | 1074.83 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2026-05-07 10:45:00 | 1090.70 | 2026-05-07 10:55:00 | 1087.55 | STOP_HIT | 1.00 | -0.29% |
