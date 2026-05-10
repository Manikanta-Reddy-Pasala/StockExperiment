# One 97 Communications Ltd. (PAYTM)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 1188.00
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
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 10
- **Target hits / Stop hits / Partials:** 2 / 10 / 3
- **Avg / median % per leg:** 0.37% / -0.30%
- **Sum % (uncompounded):** 5.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 2 | 3 | 2 | 1.03% | 7.2% |
| BUY @ 2nd Alert (retest1) | 7 | 4 | 57.1% | 2 | 3 | 2 | 1.03% | 7.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 1 | 12.5% | 0 | 7 | 1 | -0.21% | -1.7% |
| SELL @ 2nd Alert (retest1) | 8 | 1 | 12.5% | 0 | 7 | 1 | -0.21% | -1.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 5 | 33.3% | 2 | 10 | 3 | 0.37% | 5.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-11 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:05:00 | 1152.20 | 1156.41 | 0.00 | ORB-short ORB[1152.80,1166.00] vol=1.8x ATR=3.76 |
| Stop hit — per-position SL triggered | 2026-02-11 11:10:00 | 1155.96 | 1156.33 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:35:00 | 1134.90 | 1121.80 | 0.00 | ORB-long ORB[1110.60,1122.00] vol=2.2x ATR=4.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 09:40:00 | 1142.21 | 1127.00 | 0.00 | T1 1.5R @ 1142.21 |
| Target hit | 2026-02-17 15:20:00 | 1172.00 | 1149.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-02-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 09:35:00 | 1191.00 | 1181.76 | 0.00 | ORB-long ORB[1168.60,1184.50] vol=2.0x ATR=5.68 |
| Stop hit — per-position SL triggered | 2026-02-18 11:50:00 | 1185.32 | 1187.72 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-03-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:00:00 | 1042.90 | 1052.99 | 0.00 | ORB-short ORB[1047.10,1061.90] vol=1.6x ATR=4.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 11:40:00 | 1036.49 | 1050.97 | 0.00 | T1 1.5R @ 1036.49 |
| Stop hit — per-position SL triggered | 2026-03-05 12:20:00 | 1042.90 | 1049.53 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 10:55:00 | 985.60 | 993.09 | 0.00 | ORB-short ORB[993.70,1005.50] vol=2.3x ATR=3.83 |
| Stop hit — per-position SL triggered | 2026-03-13 11:25:00 | 989.43 | 991.93 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 09:50:00 | 1048.60 | 1053.91 | 0.00 | ORB-short ORB[1053.30,1064.20] vol=1.9x ATR=4.98 |
| Stop hit — per-position SL triggered | 2026-03-20 10:05:00 | 1053.58 | 1053.38 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-04-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:40:00 | 1074.00 | 1066.06 | 0.00 | ORB-long ORB[1056.30,1071.40] vol=1.7x ATR=6.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-08 10:25:00 | 1083.50 | 1071.83 | 0.00 | T1 1.5R @ 1083.50 |
| Target hit | 2026-04-08 15:20:00 | 1113.00 | 1094.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2026-04-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 10:05:00 | 1099.65 | 1106.25 | 0.00 | ORB-short ORB[1104.55,1115.80] vol=2.0x ATR=4.65 |
| Stop hit — per-position SL triggered | 2026-04-10 10:15:00 | 1104.30 | 1106.01 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-04-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 10:10:00 | 1118.30 | 1130.18 | 0.00 | ORB-short ORB[1130.50,1144.80] vol=1.6x ATR=4.38 |
| Stop hit — per-position SL triggered | 2026-04-29 10:40:00 | 1122.68 | 1128.02 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:15:00 | 1121.00 | 1113.95 | 0.00 | ORB-long ORB[1097.80,1114.00] vol=2.3x ATR=3.42 |
| Stop hit — per-position SL triggered | 2026-05-04 11:20:00 | 1117.58 | 1114.08 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-05-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:05:00 | 1105.50 | 1114.17 | 0.00 | ORB-short ORB[1106.20,1120.00] vol=2.2x ATR=3.30 |
| Stop hit — per-position SL triggered | 2026-05-05 11:10:00 | 1108.80 | 1114.04 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-05-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 11:05:00 | 1217.30 | 1202.29 | 0.00 | ORB-long ORB[1191.10,1206.80] vol=2.2x ATR=5.04 |
| Stop hit — per-position SL triggered | 2026-05-08 11:35:00 | 1212.26 | 1205.68 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-11 11:05:00 | 1152.20 | 2026-02-11 11:10:00 | 1155.96 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-02-17 09:35:00 | 1134.90 | 2026-02-17 09:40:00 | 1142.21 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2026-02-17 09:35:00 | 1134.90 | 2026-02-17 15:20:00 | 1172.00 | TARGET_HIT | 0.50 | 3.27% |
| BUY | retest1 | 2026-02-18 09:35:00 | 1191.00 | 2026-02-18 11:50:00 | 1185.32 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-03-05 11:00:00 | 1042.90 | 2026-03-05 11:40:00 | 1036.49 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2026-03-05 11:00:00 | 1042.90 | 2026-03-05 12:20:00 | 1042.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-13 10:55:00 | 985.60 | 2026-03-13 11:25:00 | 989.43 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-03-20 09:50:00 | 1048.60 | 2026-03-20 10:05:00 | 1053.58 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-04-08 09:40:00 | 1074.00 | 2026-04-08 10:25:00 | 1083.50 | PARTIAL | 0.50 | 0.88% |
| BUY | retest1 | 2026-04-08 09:40:00 | 1074.00 | 2026-04-08 15:20:00 | 1113.00 | TARGET_HIT | 0.50 | 3.63% |
| SELL | retest1 | 2026-04-10 10:05:00 | 1099.65 | 2026-04-10 10:15:00 | 1104.30 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-04-29 10:10:00 | 1118.30 | 2026-04-29 10:40:00 | 1122.68 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-05-04 11:15:00 | 1121.00 | 2026-05-04 11:20:00 | 1117.58 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2026-05-05 11:05:00 | 1105.50 | 2026-05-05 11:10:00 | 1108.80 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-05-08 11:05:00 | 1217.30 | 2026-05-08 11:35:00 | 1212.26 | STOP_HIT | 1.00 | -0.41% |
