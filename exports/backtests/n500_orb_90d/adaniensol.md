# Adani Energy Solutions Ltd. (ADANIENSOL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1351.60
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
| ENTRY1 | 13 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 11
- **Target hits / Stop hits / Partials:** 2 / 11 / 3
- **Avg / median % per leg:** -0.04% / -0.34%
- **Sum % (uncompounded):** -0.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 4 | 44.4% | 2 | 5 | 2 | 0.15% | 1.4% |
| BUY @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 2 | 5 | 2 | 0.15% | 1.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.28% | -2.0% |
| SELL @ 2nd Alert (retest1) | 7 | 1 | 14.3% | 0 | 6 | 1 | -0.28% | -2.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 16 | 5 | 31.2% | 2 | 11 | 3 | -0.04% | -0.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 10:35:00 | 1014.25 | 1023.70 | 0.00 | ORB-short ORB[1019.80,1033.90] vol=2.2x ATR=3.58 |
| Stop hit — per-position SL triggered | 2026-02-10 10:40:00 | 1017.83 | 1021.51 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 09:35:00 | 1021.65 | 1017.81 | 0.00 | ORB-long ORB[1009.85,1019.00] vol=2.0x ATR=3.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 09:45:00 | 1026.97 | 1019.28 | 0.00 | T1 1.5R @ 1026.97 |
| Target hit | 2026-02-11 15:20:00 | 1033.80 | 1028.61 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-17 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 10:40:00 | 1014.05 | 1010.08 | 0.00 | ORB-long ORB[996.00,1005.00] vol=8.9x ATR=3.42 |
| Stop hit — per-position SL triggered | 2026-02-17 10:45:00 | 1010.63 | 1010.76 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:40:00 | 1015.95 | 1026.23 | 0.00 | ORB-short ORB[1028.40,1035.70] vol=2.0x ATR=3.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:00:00 | 1011.42 | 1023.89 | 0.00 | T1 1.5R @ 1011.42 |
| Stop hit — per-position SL triggered | 2026-02-19 11:55:00 | 1015.95 | 1020.47 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2026-02-27 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 10:35:00 | 1031.00 | 1027.81 | 0.00 | ORB-long ORB[1015.05,1030.45] vol=1.6x ATR=3.04 |
| Stop hit — per-position SL triggered | 2026-02-27 11:25:00 | 1027.96 | 1028.19 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:55:00 | 952.50 | 957.40 | 0.00 | ORB-short ORB[953.40,963.80] vol=1.9x ATR=4.22 |
| Stop hit — per-position SL triggered | 2026-03-04 10:00:00 | 956.72 | 957.22 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-03-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 09:35:00 | 982.60 | 991.34 | 0.00 | ORB-short ORB[986.90,1001.30] vol=1.8x ATR=5.56 |
| Stop hit — per-position SL triggered | 2026-03-10 10:10:00 | 988.16 | 988.59 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:05:00 | 992.00 | 984.76 | 0.00 | ORB-long ORB[976.00,988.40] vol=2.0x ATR=4.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 10:35:00 | 998.64 | 988.07 | 0.00 | T1 1.5R @ 998.64 |
| Target hit | 2026-03-12 15:20:00 | 1003.30 | 1001.50 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 9 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 1019.20 | 1010.41 | 0.00 | ORB-long ORB[1000.00,1013.90] vol=1.7x ATR=4.50 |
| Stop hit — per-position SL triggered | 2026-03-18 10:05:00 | 1014.70 | 1015.79 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 11:00:00 | 944.40 | 963.03 | 0.00 | ORB-short ORB[987.60,1000.90] vol=2.0x ATR=5.34 |
| Stop hit — per-position SL triggered | 2026-03-23 11:05:00 | 949.74 | 962.08 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-04-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-06 10:35:00 | 969.75 | 953.12 | 0.00 | ORB-long ORB[936.10,949.60] vol=2.3x ATR=5.80 |
| Stop hit — per-position SL triggered | 2026-04-06 11:30:00 | 963.95 | 959.69 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-16 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-16 09:45:00 | 1191.80 | 1180.74 | 0.00 | ORB-long ORB[1174.30,1186.40] vol=2.4x ATR=5.72 |
| Stop hit — per-position SL triggered | 2026-04-16 09:55:00 | 1186.08 | 1181.15 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-05-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 10:30:00 | 1389.40 | 1389.61 | 0.00 | ORB-short ORB[1394.70,1410.70] vol=3.3x ATR=6.59 |
| Stop hit — per-position SL triggered | 2026-05-07 10:35:00 | 1395.99 | 1389.99 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-10 10:35:00 | 1014.25 | 2026-02-10 10:40:00 | 1017.83 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-02-11 09:35:00 | 1021.65 | 2026-02-11 09:45:00 | 1026.97 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-02-11 09:35:00 | 1021.65 | 2026-02-11 15:20:00 | 1033.80 | TARGET_HIT | 0.50 | 1.19% |
| BUY | retest1 | 2026-02-17 10:40:00 | 1014.05 | 2026-02-17 10:45:00 | 1010.63 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-02-19 10:40:00 | 1015.95 | 2026-02-19 11:00:00 | 1011.42 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-02-19 10:40:00 | 1015.95 | 2026-02-19 11:55:00 | 1015.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-27 10:35:00 | 1031.00 | 2026-02-27 11:25:00 | 1027.96 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-03-04 09:55:00 | 952.50 | 2026-03-04 10:00:00 | 956.72 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-03-10 09:35:00 | 982.60 | 2026-03-10 10:10:00 | 988.16 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2026-03-12 10:05:00 | 992.00 | 2026-03-12 10:35:00 | 998.64 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2026-03-12 10:05:00 | 992.00 | 2026-03-12 15:20:00 | 1003.30 | TARGET_HIT | 0.50 | 1.14% |
| BUY | retest1 | 2026-03-18 09:30:00 | 1019.20 | 2026-03-18 10:05:00 | 1014.70 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-03-23 11:00:00 | 944.40 | 2026-03-23 11:05:00 | 949.74 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2026-04-06 10:35:00 | 969.75 | 2026-04-06 11:30:00 | 963.95 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest1 | 2026-04-16 09:45:00 | 1191.80 | 2026-04-16 09:55:00 | 1186.08 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-05-07 10:30:00 | 1389.40 | 2026-05-07 10:35:00 | 1395.99 | STOP_HIT | 1.00 | -0.47% |
