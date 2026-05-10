# Max Healthcare Institute Ltd. (MAXHEALTH)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1013.10
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
| ENTRY1 | 17 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 16
- **Target hits / Stop hits / Partials:** 1 / 16 / 5
- **Avg / median % per leg:** -0.03% / -0.20%
- **Sum % (uncompounded):** -0.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 3 | 27.3% | 1 | 8 | 2 | -0.07% | -0.7% |
| BUY @ 2nd Alert (retest1) | 11 | 3 | 27.3% | 1 | 8 | 2 | -0.07% | -0.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 11 | 3 | 27.3% | 0 | 8 | 3 | 0.01% | 0.1% |
| SELL @ 2nd Alert (retest1) | 11 | 3 | 27.3% | 0 | 8 | 3 | 0.01% | 0.1% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 6 | 27.3% | 1 | 16 | 5 | -0.03% | -0.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 11:05:00 | 1080.50 | 1076.86 | 0.00 | ORB-long ORB[1073.25,1080.00] vol=5.7x ATR=2.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 12:30:00 | 1084.09 | 1078.06 | 0.00 | T1 1.5R @ 1084.09 |
| Target hit | 2026-02-18 15:20:00 | 1085.75 | 1082.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:55:00 | 1090.20 | 1093.77 | 0.00 | ORB-short ORB[1094.50,1109.00] vol=2.5x ATR=3.25 |
| Stop hit — per-position SL triggered | 2026-02-27 11:45:00 | 1093.45 | 1092.47 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 11:15:00 | 1038.20 | 1040.05 | 0.00 | ORB-short ORB[1040.10,1049.50] vol=1.6x ATR=3.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 11:25:00 | 1033.67 | 1039.75 | 0.00 | T1 1.5R @ 1033.67 |
| Stop hit — per-position SL triggered | 2026-03-06 11:35:00 | 1038.20 | 1039.39 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-11 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:55:00 | 1031.30 | 1032.18 | 0.00 | ORB-short ORB[1032.00,1047.40] vol=1.7x ATR=2.68 |
| Stop hit — per-position SL triggered | 2026-03-11 12:00:00 | 1033.98 | 1031.38 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:55:00 | 996.70 | 1006.73 | 0.00 | ORB-short ORB[1007.50,1020.80] vol=1.6x ATR=3.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:00:00 | 992.09 | 1005.53 | 0.00 | T1 1.5R @ 992.09 |
| Stop hit — per-position SL triggered | 2026-03-13 11:20:00 | 996.70 | 1003.82 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-03-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:20:00 | 980.50 | 977.16 | 0.00 | ORB-long ORB[969.60,978.00] vol=3.0x ATR=2.91 |
| Stop hit — per-position SL triggered | 2026-03-17 10:25:00 | 977.59 | 977.17 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 10:50:00 | 973.60 | 980.87 | 0.00 | ORB-short ORB[976.40,987.90] vol=1.7x ATR=2.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 11:20:00 | 969.16 | 977.75 | 0.00 | T1 1.5R @ 969.16 |
| Stop hit — per-position SL triggered | 2026-03-18 11:50:00 | 973.60 | 976.06 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-20 11:05:00 | 963.50 | 960.05 | 0.00 | ORB-long ORB[951.50,961.50] vol=2.8x ATR=2.38 |
| Stop hit — per-position SL triggered | 2026-03-20 11:20:00 | 961.12 | 961.69 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-27 11:10:00 | 974.80 | 981.84 | 0.00 | ORB-short ORB[978.00,990.80] vol=2.0x ATR=2.81 |
| Stop hit — per-position SL triggered | 2026-03-27 11:40:00 | 977.61 | 980.79 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 09:40:00 | 956.30 | 950.91 | 0.00 | ORB-long ORB[941.50,953.00] vol=2.1x ATR=3.02 |
| Stop hit — per-position SL triggered | 2026-04-09 09:50:00 | 953.28 | 951.54 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 11:15:00 | 947.50 | 948.54 | 0.00 | ORB-short ORB[949.60,959.45] vol=3.9x ATR=2.35 |
| Stop hit — per-position SL triggered | 2026-04-10 11:20:00 | 949.85 | 948.55 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-13 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:25:00 | 948.50 | 944.34 | 0.00 | ORB-long ORB[933.05,945.75] vol=2.6x ATR=3.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 10:35:00 | 953.79 | 944.86 | 0.00 | T1 1.5R @ 953.79 |
| Stop hit — per-position SL triggered | 2026-04-13 10:40:00 | 948.50 | 944.92 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 11:00:00 | 1000.25 | 995.15 | 0.00 | ORB-long ORB[983.00,993.70] vol=1.5x ATR=2.52 |
| Stop hit — per-position SL triggered | 2026-04-17 11:10:00 | 997.73 | 995.41 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 11:15:00 | 1012.25 | 1009.02 | 0.00 | ORB-long ORB[1000.00,1010.85] vol=1.7x ATR=2.46 |
| Stop hit — per-position SL triggered | 2026-04-27 11:30:00 | 1009.79 | 1009.10 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 09:30:00 | 1023.60 | 1016.83 | 0.00 | ORB-long ORB[1006.40,1021.45] vol=2.3x ATR=3.49 |
| Stop hit — per-position SL triggered | 2026-04-29 09:40:00 | 1020.11 | 1018.36 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-05-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:05:00 | 1013.50 | 1003.52 | 0.00 | ORB-long ORB[992.10,1002.00] vol=4.6x ATR=4.01 |
| Stop hit — per-position SL triggered | 2026-05-04 10:10:00 | 1009.49 | 1003.76 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-05-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:00:00 | 1006.35 | 1009.60 | 0.00 | ORB-short ORB[1007.75,1015.65] vol=1.8x ATR=2.02 |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 1008.37 | 1009.47 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-18 11:05:00 | 1080.50 | 2026-02-18 12:30:00 | 1084.09 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-02-18 11:05:00 | 1080.50 | 2026-02-18 15:20:00 | 1085.75 | TARGET_HIT | 0.50 | 0.49% |
| SELL | retest1 | 2026-02-27 10:55:00 | 1090.20 | 2026-02-27 11:45:00 | 1093.45 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-06 11:15:00 | 1038.20 | 2026-03-06 11:25:00 | 1033.67 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-03-06 11:15:00 | 1038.20 | 2026-03-06 11:35:00 | 1038.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-11 10:55:00 | 1031.30 | 2026-03-11 12:00:00 | 1033.98 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-03-13 10:55:00 | 996.70 | 2026-03-13 11:00:00 | 992.09 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-03-13 10:55:00 | 996.70 | 2026-03-13 11:20:00 | 996.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 10:20:00 | 980.50 | 2026-03-17 10:25:00 | 977.59 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-18 10:50:00 | 973.60 | 2026-03-18 11:20:00 | 969.16 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2026-03-18 10:50:00 | 973.60 | 2026-03-18 11:50:00 | 973.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-20 11:05:00 | 963.50 | 2026-03-20 11:20:00 | 961.12 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-03-27 11:10:00 | 974.80 | 2026-03-27 11:40:00 | 977.61 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2026-04-09 09:40:00 | 956.30 | 2026-04-09 09:50:00 | 953.28 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2026-04-10 11:15:00 | 947.50 | 2026-04-10 11:20:00 | 949.85 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-04-13 10:25:00 | 948.50 | 2026-04-13 10:35:00 | 953.79 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-04-13 10:25:00 | 948.50 | 2026-04-13 10:40:00 | 948.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-17 11:00:00 | 1000.25 | 2026-04-17 11:10:00 | 997.73 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-04-27 11:15:00 | 1012.25 | 2026-04-27 11:30:00 | 1009.79 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-04-29 09:30:00 | 1023.60 | 2026-04-29 09:40:00 | 1020.11 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-05-04 10:05:00 | 1013.50 | 2026-05-04 10:10:00 | 1009.49 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-05-06 11:00:00 | 1006.35 | 2026-05-06 11:15:00 | 1008.37 | STOP_HIT | 1.00 | -0.20% |
