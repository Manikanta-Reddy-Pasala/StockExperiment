# SHRIRAMFIN (SHRIRAMFIN)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1003.05
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
| ENTRY1 | 9 |
| ENTRY2 | 0 |
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 6
- **Target hits / Stop hits / Partials:** 3 / 6 / 4
- **Avg / median % per leg:** 0.49% / 0.36%
- **Sum % (uncompounded):** 6.42%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 2 | 1 | 2 | 1.04% | 5.2% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 2 | 1 | 2 | 1.04% | 5.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.15% | 1.2% |
| SELL @ 2nd Alert (retest1) | 8 | 3 | 37.5% | 1 | 5 | 2 | 0.15% | 1.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 13 | 7 | 53.8% | 3 | 6 | 4 | 0.49% | 6.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 11:10:00 | 1025.00 | 1017.29 | 0.00 | ORB-long ORB[1003.40,1011.70] vol=2.6x ATR=2.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:30:00 | 1029.31 | 1019.19 | 0.00 | T1 1.5R @ 1029.31 |
| Target hit | 2026-02-09 15:20:00 | 1064.90 | 1040.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 10:45:00 | 1061.80 | 1066.36 | 0.00 | ORB-short ORB[1067.20,1081.90] vol=2.1x ATR=3.29 |
| Stop hit — per-position SL triggered | 2026-02-17 11:00:00 | 1065.09 | 1066.01 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2026-02-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:30:00 | 1077.70 | 1070.45 | 0.00 | ORB-long ORB[1055.80,1065.00] vol=1.8x ATR=3.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:15:00 | 1082.22 | 1072.75 | 0.00 | T1 1.5R @ 1082.22 |
| Target hit | 2026-02-25 15:20:00 | 1086.30 | 1080.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2026-03-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 09:35:00 | 945.40 | 957.38 | 0.00 | ORB-short ORB[954.50,966.90] vol=2.0x ATR=5.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 09:55:00 | 937.27 | 952.26 | 0.00 | T1 1.5R @ 937.27 |
| Stop hit — per-position SL triggered | 2026-03-20 10:45:00 | 945.40 | 943.63 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 09:45:00 | 902.40 | 908.35 | 0.00 | ORB-short ORB[903.00,916.00] vol=2.0x ATR=6.03 |
| Stop hit — per-position SL triggered | 2026-03-24 09:55:00 | 908.43 | 907.56 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-04-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 10:25:00 | 1028.70 | 1034.20 | 0.00 | ORB-short ORB[1030.00,1043.50] vol=2.1x ATR=3.98 |
| Stop hit — per-position SL triggered | 2026-04-15 11:40:00 | 1032.68 | 1032.91 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:50:00 | 1031.05 | 1024.85 | 0.00 | ORB-long ORB[1013.15,1025.60] vol=1.5x ATR=3.56 |
| Stop hit — per-position SL triggered | 2026-04-17 10:00:00 | 1027.49 | 1025.37 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-04-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:35:00 | 1030.25 | 1037.60 | 0.00 | ORB-short ORB[1036.35,1051.30] vol=2.3x ATR=2.34 |
| Stop hit — per-position SL triggered | 2026-04-22 10:40:00 | 1032.59 | 1037.13 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:05:00 | 967.70 | 973.36 | 0.00 | ORB-short ORB[968.00,976.25] vol=3.8x ATR=2.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:15:00 | 964.25 | 972.79 | 0.00 | T1 1.5R @ 964.25 |
| Target hit | 2026-04-28 15:20:00 | 952.10 | 959.62 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 11:10:00 | 1025.00 | 2026-02-09 11:30:00 | 1029.31 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-02-09 11:10:00 | 1025.00 | 2026-02-09 15:20:00 | 1064.90 | TARGET_HIT | 0.50 | 3.89% |
| SELL | retest1 | 2026-02-17 10:45:00 | 1061.80 | 2026-02-17 11:00:00 | 1065.09 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-25 10:30:00 | 1077.70 | 2026-02-25 11:15:00 | 1082.22 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-02-25 10:30:00 | 1077.70 | 2026-02-25 15:20:00 | 1086.30 | TARGET_HIT | 0.50 | 0.80% |
| SELL | retest1 | 2026-03-20 09:35:00 | 945.40 | 2026-03-20 09:55:00 | 937.27 | PARTIAL | 0.50 | 0.86% |
| SELL | retest1 | 2026-03-20 09:35:00 | 945.40 | 2026-03-20 10:45:00 | 945.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-24 09:45:00 | 902.40 | 2026-03-24 09:55:00 | 908.43 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest1 | 2026-04-15 10:25:00 | 1028.70 | 2026-04-15 11:40:00 | 1032.68 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-17 09:50:00 | 1031.05 | 2026-04-17 10:00:00 | 1027.49 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-04-22 10:35:00 | 1030.25 | 2026-04-22 10:40:00 | 1032.59 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-04-28 11:05:00 | 967.70 | 2026-04-28 11:15:00 | 964.25 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-04-28 11:05:00 | 967.70 | 2026-04-28 15:20:00 | 952.10 | TARGET_HIT | 0.50 | 1.61% |
