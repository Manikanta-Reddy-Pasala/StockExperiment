# Laurus Labs Ltd. (LAURUSLABS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1225.20
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
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 9
- **Target hits / Stop hits / Partials:** 3 / 9 / 7
- **Avg / median % per leg:** 0.45% / 0.18%
- **Sum % (uncompounded):** 8.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 9 | 56.2% | 3 | 7 | 6 | 0.53% | 8.4% |
| BUY @ 2nd Alert (retest1) | 16 | 9 | 56.2% | 3 | 7 | 6 | 0.53% | 8.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.04% | 0.1% |
| SELL @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 2 | 1 | 0.04% | 0.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 10 | 52.6% | 3 | 9 | 7 | 0.45% | 8.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:30:00 | 972.65 | 963.81 | 0.00 | ORB-long ORB[949.15,959.50] vol=1.6x ATR=5.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:00:00 | 980.82 | 967.66 | 0.00 | T1 1.5R @ 980.82 |
| Target hit | 2026-02-09 15:20:00 | 984.65 | 974.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-02-11 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 09:55:00 | 971.30 | 965.90 | 0.00 | ORB-long ORB[962.85,970.10] vol=1.6x ATR=2.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 10:10:00 | 975.16 | 968.24 | 0.00 | T1 1.5R @ 975.16 |
| Target hit | 2026-02-11 15:20:00 | 1015.20 | 995.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-13 10:55:00 | 1015.05 | 1009.00 | 0.00 | ORB-long ORB[1005.00,1014.90] vol=2.3x ATR=3.32 |
| Stop hit — per-position SL triggered | 2026-02-13 11:00:00 | 1011.73 | 1009.10 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:40:00 | 1016.25 | 1013.55 | 0.00 | ORB-long ORB[1006.05,1012.45] vol=1.5x ATR=2.29 |
| Stop hit — per-position SL triggered | 2026-02-17 10:50:00 | 1013.96 | 1013.69 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:05:00 | 1022.50 | 1018.22 | 0.00 | ORB-long ORB[1011.45,1020.00] vol=1.9x ATR=2.61 |
| Stop hit — per-position SL triggered | 2026-02-18 10:30:00 | 1019.89 | 1020.05 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:40:00 | 1014.30 | 1023.58 | 0.00 | ORB-short ORB[1022.85,1032.95] vol=2.6x ATR=2.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 11:45:00 | 1010.08 | 1020.69 | 0.00 | T1 1.5R @ 1010.08 |
| Stop hit — per-position SL triggered | 2026-02-23 12:10:00 | 1014.30 | 1020.23 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:35:00 | 1089.00 | 1082.11 | 0.00 | ORB-long ORB[1073.00,1084.60] vol=2.0x ATR=3.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 09:50:00 | 1094.82 | 1087.32 | 0.00 | T1 1.5R @ 1094.82 |
| Target hit | 2026-02-26 10:30:00 | 1090.95 | 1090.96 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — BUY (started 2026-03-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 10:00:00 | 1055.20 | 1044.15 | 0.00 | ORB-long ORB[1031.00,1046.10] vol=1.6x ATR=4.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 10:15:00 | 1061.48 | 1048.30 | 0.00 | T1 1.5R @ 1061.48 |
| Stop hit — per-position SL triggered | 2026-03-11 13:40:00 | 1055.20 | 1056.83 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 978.50 | 974.14 | 0.00 | ORB-long ORB[968.50,976.90] vol=2.2x ATR=3.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 09:45:00 | 984.21 | 976.01 | 0.00 | T1 1.5R @ 984.21 |
| Stop hit — per-position SL triggered | 2026-03-18 10:00:00 | 978.50 | 978.25 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-24 09:45:00 | 1137.00 | 1131.80 | 0.00 | ORB-long ORB[1122.20,1135.00] vol=1.5x ATR=3.65 |
| Stop hit — per-position SL triggered | 2026-04-24 09:50:00 | 1133.35 | 1131.88 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:15:00 | 1160.60 | 1165.11 | 0.00 | ORB-short ORB[1162.20,1175.50] vol=3.1x ATR=3.49 |
| Stop hit — per-position SL triggered | 2026-05-05 11:45:00 | 1164.09 | 1164.79 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:15:00 | 1204.30 | 1196.27 | 0.00 | ORB-long ORB[1183.70,1200.00] vol=1.9x ATR=5.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 11:00:00 | 1212.12 | 1201.69 | 0.00 | T1 1.5R @ 1212.12 |
| Stop hit — per-position SL triggered | 2026-05-07 13:15:00 | 1204.30 | 1206.45 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:30:00 | 972.65 | 2026-02-09 11:00:00 | 980.82 | PARTIAL | 0.50 | 0.84% |
| BUY | retest1 | 2026-02-09 10:30:00 | 972.65 | 2026-02-09 15:20:00 | 984.65 | TARGET_HIT | 0.50 | 1.23% |
| BUY | retest1 | 2026-02-11 09:55:00 | 971.30 | 2026-02-11 10:10:00 | 975.16 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-02-11 09:55:00 | 971.30 | 2026-02-11 15:20:00 | 1015.20 | TARGET_HIT | 0.50 | 4.52% |
| BUY | retest1 | 2026-02-13 10:55:00 | 1015.05 | 2026-02-13 11:00:00 | 1011.73 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-17 10:40:00 | 1016.25 | 2026-02-17 10:50:00 | 1013.96 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-02-18 10:05:00 | 1022.50 | 2026-02-18 10:30:00 | 1019.89 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-23 10:40:00 | 1014.30 | 2026-02-23 11:45:00 | 1010.08 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-02-23 10:40:00 | 1014.30 | 2026-02-23 12:10:00 | 1014.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-26 09:35:00 | 1089.00 | 2026-02-26 09:50:00 | 1094.82 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-02-26 09:35:00 | 1089.00 | 2026-02-26 10:30:00 | 1090.95 | TARGET_HIT | 0.50 | 0.18% |
| BUY | retest1 | 2026-03-11 10:00:00 | 1055.20 | 2026-03-11 10:15:00 | 1061.48 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2026-03-11 10:00:00 | 1055.20 | 2026-03-11 13:40:00 | 1055.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-18 09:30:00 | 978.50 | 2026-03-18 09:45:00 | 984.21 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-03-18 09:30:00 | 978.50 | 2026-03-18 10:00:00 | 978.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-24 09:45:00 | 1137.00 | 2026-04-24 09:50:00 | 1133.35 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-05-05 11:15:00 | 1160.60 | 2026-05-05 11:45:00 | 1164.09 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-07 10:15:00 | 1204.30 | 2026-05-07 11:00:00 | 1212.12 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-05-07 10:15:00 | 1204.30 | 2026-05-07 13:15:00 | 1204.30 | STOP_HIT | 0.50 | 0.00% |
