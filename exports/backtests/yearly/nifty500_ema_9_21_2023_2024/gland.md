# Gland Pharma Ltd. (GLAND)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1906.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 215 |
| ALERT1 | 147 |
| ALERT2 | 144 |
| ALERT2_SKIP | 77 |
| ALERT3 | 401 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 220 |
| PARTIAL | 34 |
| TARGET_HIT | 12 |
| STOP_HIT | 214 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 260 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 98 / 162
- **Target hits / Stop hits / Partials:** 12 / 214 / 34
- **Avg / median % per leg:** 0.62% / -0.68%
- **Sum % (uncompounded):** 162.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 107 | 20 | 18.7% | 4 | 102 | 1 | -0.74% | -78.9% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 0 | 6 | 1 | -0.46% | -3.2% |
| BUY @ 3rd Alert (retest2) | 100 | 18 | 18.0% | 4 | 96 | 0 | -0.76% | -75.6% |
| SELL (all) | 153 | 78 | 51.0% | 8 | 112 | 33 | 1.57% | 240.9% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.39% | -1.4% |
| SELL @ 3rd Alert (retest2) | 152 | 78 | 51.3% | 8 | 111 | 33 | 1.59% | 242.3% |
| retest1 (combined) | 8 | 2 | 25.0% | 0 | 7 | 1 | -0.58% | -4.6% |
| retest2 (combined) | 252 | 96 | 38.1% | 12 | 207 | 33 | 0.66% | 166.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 09:15:00 | 936.00 | 921.79 | 919.93 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-02 15:15:00 | 922.10 | 924.56 | 924.78 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-05 09:15:00 | 934.20 | 926.49 | 925.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-06 09:15:00 | 950.00 | 936.16 | 931.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-07 11:15:00 | 943.65 | 943.97 | 939.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-07 11:30:00 | 943.85 | 943.97 | 939.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 11:15:00 | 955.05 | 959.27 | 955.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-12 12:00:00 | 955.05 | 959.27 | 955.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 12:15:00 | 954.10 | 958.24 | 955.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-13 09:15:00 | 984.80 | 956.80 | 955.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-21 11:15:00 | 1005.00 | 1016.45 | 1017.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2023-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 11:15:00 | 1005.00 | 1016.45 | 1017.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 09:15:00 | 1002.20 | 1009.14 | 1012.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 11:15:00 | 1006.00 | 988.09 | 993.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 11:15:00 | 1006.00 | 988.09 | 993.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 11:15:00 | 1006.00 | 988.09 | 993.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 11:45:00 | 1004.00 | 988.09 | 993.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 12:15:00 | 1012.00 | 992.87 | 995.34 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2023-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 14:15:00 | 1020.00 | 1001.31 | 998.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 09:15:00 | 1041.05 | 1019.61 | 1011.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 09:15:00 | 1060.00 | 1060.86 | 1048.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-03 09:15:00 | 1060.00 | 1060.86 | 1048.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 1060.00 | 1060.86 | 1048.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 09:45:00 | 1046.40 | 1060.86 | 1048.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 1061.15 | 1063.81 | 1056.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 09:45:00 | 1064.00 | 1063.81 | 1056.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 12:15:00 | 1053.00 | 1061.21 | 1057.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 13:00:00 | 1053.00 | 1061.21 | 1057.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 13:15:00 | 1056.20 | 1060.21 | 1057.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 14:15:00 | 1049.70 | 1060.21 | 1057.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 14:15:00 | 1052.00 | 1058.57 | 1056.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 15:15:00 | 1054.00 | 1058.57 | 1056.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 15:15:00 | 1054.00 | 1057.65 | 1056.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-05 09:15:00 | 1060.45 | 1057.65 | 1056.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-07 11:15:00 | 1056.70 | 1059.95 | 1060.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2023-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 11:15:00 | 1056.70 | 1059.95 | 1060.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 12:15:00 | 1054.25 | 1058.81 | 1059.65 | Break + close below crossover candle low |

### Cycle 7 — BUY (started 2023-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 09:15:00 | 1110.00 | 1063.94 | 1061.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-14 12:15:00 | 1137.00 | 1120.60 | 1107.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 11:15:00 | 1160.00 | 1163.25 | 1148.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-18 12:00:00 | 1160.00 | 1163.25 | 1148.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 12:15:00 | 1161.00 | 1162.80 | 1149.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 12:45:00 | 1160.80 | 1162.80 | 1149.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 1244.95 | 1238.18 | 1223.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-25 09:15:00 | 1256.15 | 1240.01 | 1231.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-26 10:15:00 | 1226.80 | 1231.44 | 1231.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2023-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 10:15:00 | 1226.80 | 1231.44 | 1231.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-26 13:15:00 | 1209.00 | 1225.21 | 1228.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-27 09:15:00 | 1223.75 | 1216.10 | 1222.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 09:15:00 | 1223.75 | 1216.10 | 1222.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 1223.75 | 1216.10 | 1222.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-27 10:00:00 | 1223.75 | 1216.10 | 1222.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 10:15:00 | 1224.10 | 1217.70 | 1222.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-27 10:45:00 | 1224.35 | 1217.70 | 1222.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 11:15:00 | 1229.20 | 1220.00 | 1223.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-27 12:00:00 | 1229.20 | 1220.00 | 1223.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2023-07-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 12:15:00 | 1267.05 | 1229.41 | 1227.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-28 11:15:00 | 1297.55 | 1269.69 | 1251.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-31 14:15:00 | 1319.75 | 1320.87 | 1295.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-31 15:00:00 | 1319.75 | 1320.87 | 1295.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 15:15:00 | 1298.00 | 1316.29 | 1295.70 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 12:15:00 | 1290.05 | 1296.25 | 1296.81 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-08-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 09:15:00 | 1330.05 | 1299.03 | 1297.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 09:15:00 | 1353.55 | 1323.80 | 1312.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-07 09:15:00 | 1320.65 | 1332.33 | 1323.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-07 09:15:00 | 1320.65 | 1332.33 | 1323.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 09:15:00 | 1320.65 | 1332.33 | 1323.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 09:30:00 | 1315.00 | 1332.33 | 1323.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 10:15:00 | 1314.45 | 1328.75 | 1322.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-07 11:00:00 | 1314.45 | 1328.75 | 1322.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 11:15:00 | 1319.05 | 1326.81 | 1322.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-07 12:45:00 | 1340.90 | 1336.23 | 1327.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-08-08 09:15:00 | 1474.99 | 1396.50 | 1358.26 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2023-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 12:15:00 | 1594.50 | 1613.49 | 1616.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-16 09:15:00 | 1576.20 | 1600.58 | 1608.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-17 09:15:00 | 1638.00 | 1580.90 | 1590.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 09:15:00 | 1638.00 | 1580.90 | 1590.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 09:15:00 | 1638.00 | 1580.90 | 1590.77 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2023-08-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 11:15:00 | 1629.00 | 1599.98 | 1598.31 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 13:15:00 | 1573.00 | 1603.14 | 1604.04 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 11:15:00 | 1603.45 | 1594.75 | 1593.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 15:15:00 | 1609.00 | 1600.39 | 1596.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 09:15:00 | 1595.10 | 1599.33 | 1596.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 09:15:00 | 1595.10 | 1599.33 | 1596.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 09:15:00 | 1595.10 | 1599.33 | 1596.82 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2023-08-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 13:15:00 | 1584.35 | 1594.46 | 1595.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-23 14:15:00 | 1580.70 | 1591.71 | 1593.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-24 09:15:00 | 1597.90 | 1592.19 | 1593.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-24 09:15:00 | 1597.90 | 1592.19 | 1593.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 09:15:00 | 1597.90 | 1592.19 | 1593.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-24 09:30:00 | 1604.55 | 1592.19 | 1593.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2023-08-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-24 10:15:00 | 1604.95 | 1594.74 | 1594.70 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-08-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 11:15:00 | 1587.95 | 1593.38 | 1594.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-24 14:15:00 | 1578.25 | 1590.55 | 1592.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 09:15:00 | 1575.70 | 1562.33 | 1572.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 09:15:00 | 1575.70 | 1562.33 | 1572.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 1575.70 | 1562.33 | 1572.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 10:00:00 | 1575.70 | 1562.33 | 1572.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 10:15:00 | 1583.05 | 1566.47 | 1573.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-28 15:00:00 | 1570.95 | 1573.11 | 1575.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-29 09:15:00 | 1604.60 | 1578.90 | 1577.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2023-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 09:15:00 | 1604.60 | 1578.90 | 1577.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 11:15:00 | 1610.15 | 1596.07 | 1588.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 12:15:00 | 1594.55 | 1595.76 | 1589.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-30 12:30:00 | 1598.00 | 1595.76 | 1589.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 14:15:00 | 1597.90 | 1595.79 | 1590.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-30 14:45:00 | 1586.10 | 1595.79 | 1590.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 13:15:00 | 1714.50 | 1732.45 | 1711.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 14:00:00 | 1714.50 | 1732.45 | 1711.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 14:15:00 | 1708.45 | 1727.65 | 1710.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 15:00:00 | 1708.45 | 1727.65 | 1710.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 15:15:00 | 1714.00 | 1724.92 | 1711.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-05 09:15:00 | 1748.00 | 1724.92 | 1711.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-05 12:15:00 | 1718.95 | 1725.72 | 1715.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-05 12:15:00 | 1701.10 | 1720.79 | 1714.07 | SL hit (close<static) qty=1.00 sl=1705.00 alert=retest2 |

### Cycle 20 — SELL (started 2023-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-05 15:15:00 | 1692.00 | 1709.21 | 1710.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-11 09:15:00 | 1675.20 | 1692.49 | 1696.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-11 15:15:00 | 1684.00 | 1683.84 | 1689.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-11 15:15:00 | 1684.00 | 1683.84 | 1689.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 15:15:00 | 1684.00 | 1683.84 | 1689.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:15:00 | 1698.00 | 1683.84 | 1689.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 1656.00 | 1678.27 | 1686.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:30:00 | 1666.05 | 1678.27 | 1686.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 09:15:00 | 1633.50 | 1654.78 | 1668.53 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2023-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 09:15:00 | 1660.00 | 1657.64 | 1657.58 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2023-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-15 12:15:00 | 1647.85 | 1656.27 | 1657.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-15 13:15:00 | 1644.90 | 1653.99 | 1655.92 | Break + close below crossover candle low |

### Cycle 23 — BUY (started 2023-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 14:15:00 | 1700.45 | 1663.28 | 1659.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 15:15:00 | 1708.30 | 1672.29 | 1664.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 15:15:00 | 1687.00 | 1691.81 | 1680.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-20 09:15:00 | 1677.05 | 1691.81 | 1680.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 1686.25 | 1690.70 | 1681.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-21 13:15:00 | 1695.90 | 1684.17 | 1682.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-21 13:45:00 | 1695.05 | 1685.75 | 1682.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-21 14:30:00 | 1700.00 | 1687.40 | 1683.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-22 09:15:00 | 1695.75 | 1687.88 | 1684.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 1687.30 | 1687.77 | 1684.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-22 10:00:00 | 1687.30 | 1687.77 | 1684.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 10:15:00 | 1701.50 | 1690.51 | 1686.29 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-09-25 14:15:00 | 1657.25 | 1686.10 | 1687.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2023-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-25 14:15:00 | 1657.25 | 1686.10 | 1687.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-25 15:15:00 | 1652.00 | 1679.28 | 1684.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-26 10:15:00 | 1685.55 | 1677.70 | 1682.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 10:15:00 | 1685.55 | 1677.70 | 1682.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 10:15:00 | 1685.55 | 1677.70 | 1682.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 11:00:00 | 1685.55 | 1677.70 | 1682.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 11:15:00 | 1686.00 | 1679.36 | 1683.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-26 12:30:00 | 1658.50 | 1672.49 | 1679.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-29 10:15:00 | 1674.10 | 1661.88 | 1661.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2023-09-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 10:15:00 | 1674.10 | 1661.88 | 1661.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-29 11:15:00 | 1688.50 | 1667.21 | 1664.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-29 14:15:00 | 1674.95 | 1676.25 | 1669.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-29 15:00:00 | 1674.95 | 1676.25 | 1669.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 15:15:00 | 1665.15 | 1674.03 | 1669.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-03 09:15:00 | 1698.50 | 1674.03 | 1669.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-04 12:15:00 | 1676.95 | 1681.51 | 1678.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-04 14:00:00 | 1678.30 | 1679.03 | 1677.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 09:15:00 | 1678.65 | 1677.51 | 1677.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-05 09:15:00 | 1674.20 | 1676.85 | 1677.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2023-10-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 09:15:00 | 1674.20 | 1676.85 | 1677.11 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2023-10-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 10:15:00 | 1680.00 | 1677.48 | 1677.37 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-10-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 12:15:00 | 1672.15 | 1676.81 | 1677.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-06 15:15:00 | 1665.60 | 1673.52 | 1675.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 10:15:00 | 1634.10 | 1631.38 | 1646.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-10 11:00:00 | 1634.10 | 1631.38 | 1646.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 09:15:00 | 1642.00 | 1632.14 | 1640.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 09:45:00 | 1643.15 | 1632.14 | 1640.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 10:15:00 | 1653.80 | 1636.47 | 1641.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 10:30:00 | 1664.55 | 1636.47 | 1641.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 11:15:00 | 1639.45 | 1637.07 | 1641.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-11 12:45:00 | 1636.00 | 1636.85 | 1640.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-12 10:45:00 | 1637.25 | 1637.04 | 1639.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-12 13:45:00 | 1635.10 | 1636.10 | 1638.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-18 14:15:00 | 1609.00 | 1601.69 | 1600.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2023-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-18 14:15:00 | 1609.00 | 1601.69 | 1600.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-18 15:15:00 | 1611.00 | 1603.55 | 1601.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-19 10:15:00 | 1603.65 | 1604.01 | 1602.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-19 11:00:00 | 1603.65 | 1604.01 | 1602.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 1607.90 | 1607.20 | 1604.89 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2023-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 11:15:00 | 1581.05 | 1601.62 | 1602.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 13:15:00 | 1563.20 | 1590.32 | 1597.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 09:15:00 | 1535.75 | 1530.60 | 1552.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-25 09:45:00 | 1536.80 | 1530.60 | 1552.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 09:15:00 | 1510.00 | 1513.65 | 1531.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-26 10:00:00 | 1510.00 | 1513.65 | 1531.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 14:15:00 | 1544.20 | 1513.06 | 1523.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-26 15:00:00 | 1544.20 | 1513.06 | 1523.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 15:15:00 | 1535.05 | 1517.46 | 1524.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 09:15:00 | 1553.25 | 1517.46 | 1524.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 1546.95 | 1523.36 | 1526.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-27 10:15:00 | 1541.45 | 1523.36 | 1526.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-27 11:15:00 | 1554.95 | 1533.37 | 1530.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2023-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 11:15:00 | 1554.95 | 1533.37 | 1530.51 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-11-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 10:15:00 | 1521.40 | 1538.96 | 1540.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-02 10:15:00 | 1519.45 | 1530.77 | 1535.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-03 12:15:00 | 1518.00 | 1506.05 | 1515.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-03 12:15:00 | 1518.00 | 1506.05 | 1515.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 12:15:00 | 1518.00 | 1506.05 | 1515.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-03 13:00:00 | 1518.00 | 1506.05 | 1515.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 13:15:00 | 1536.00 | 1512.04 | 1517.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-03 14:00:00 | 1536.00 | 1512.04 | 1517.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 09:15:00 | 1522.10 | 1513.80 | 1516.71 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-06 10:15:00 | 1542.00 | 1519.44 | 1519.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 11:15:00 | 1576.65 | 1530.88 | 1524.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-08 10:15:00 | 1634.05 | 1634.29 | 1604.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-08 11:00:00 | 1634.05 | 1634.29 | 1604.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 1610.60 | 1626.19 | 1613.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 10:00:00 | 1610.60 | 1626.19 | 1613.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 10:15:00 | 1609.50 | 1622.85 | 1612.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 10:45:00 | 1602.25 | 1622.85 | 1612.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 12:15:00 | 1611.30 | 1618.81 | 1612.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 12:30:00 | 1611.45 | 1618.81 | 1612.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 13:15:00 | 1611.00 | 1617.25 | 1612.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 13:30:00 | 1610.85 | 1617.25 | 1612.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 14:15:00 | 1611.00 | 1616.00 | 1612.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 14:30:00 | 1614.00 | 1616.00 | 1612.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 15:15:00 | 1615.95 | 1615.99 | 1612.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-10 09:15:00 | 1600.80 | 1615.99 | 1612.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 1616.00 | 1615.99 | 1612.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-10 09:30:00 | 1602.60 | 1615.99 | 1612.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 10:15:00 | 1617.40 | 1616.27 | 1613.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-10 15:00:00 | 1630.00 | 1619.87 | 1616.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-13 10:00:00 | 1626.95 | 1624.75 | 1619.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-13 10:30:00 | 1626.00 | 1624.98 | 1620.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-13 11:00:00 | 1625.90 | 1624.98 | 1620.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 1627.30 | 1625.94 | 1622.62 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-11-15 11:15:00 | 1609.45 | 1621.32 | 1621.01 | SL hit (close<static) qty=1.00 sl=1612.25 alert=retest2 |

### Cycle 34 — SELL (started 2023-11-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-15 12:15:00 | 1617.10 | 1620.48 | 1620.66 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-11-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-16 09:15:00 | 1637.15 | 1622.73 | 1621.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-20 09:15:00 | 1658.00 | 1636.19 | 1630.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-20 13:15:00 | 1657.55 | 1658.66 | 1644.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-20 14:00:00 | 1657.55 | 1658.66 | 1644.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 14:15:00 | 1630.00 | 1652.93 | 1643.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 14:30:00 | 1629.90 | 1652.93 | 1643.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 15:15:00 | 1630.00 | 1648.34 | 1642.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 09:15:00 | 1667.45 | 1648.34 | 1642.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-12-01 09:15:00 | 1834.20 | 1786.66 | 1778.57 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2023-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 11:15:00 | 1770.50 | 1796.90 | 1798.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-06 12:15:00 | 1755.00 | 1788.52 | 1794.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-08 09:15:00 | 1786.95 | 1769.68 | 1776.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-08 09:15:00 | 1786.95 | 1769.68 | 1776.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 09:15:00 | 1786.95 | 1769.68 | 1776.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-08 09:30:00 | 1784.50 | 1769.68 | 1776.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 10:15:00 | 1785.55 | 1772.85 | 1776.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-08 10:30:00 | 1785.05 | 1772.85 | 1776.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 14:15:00 | 1776.20 | 1775.58 | 1777.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-08 15:00:00 | 1776.20 | 1775.58 | 1777.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 15:15:00 | 1770.00 | 1774.47 | 1776.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-11 09:15:00 | 1762.05 | 1774.47 | 1776.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-11 11:15:00 | 1784.95 | 1778.22 | 1777.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2023-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 11:15:00 | 1784.95 | 1778.22 | 1777.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 12:15:00 | 1800.00 | 1782.58 | 1779.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-13 09:15:00 | 1802.10 | 1810.03 | 1801.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 09:15:00 | 1802.10 | 1810.03 | 1801.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 09:15:00 | 1802.10 | 1810.03 | 1801.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-14 13:00:00 | 1845.00 | 1818.82 | 1809.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-14 14:45:00 | 1830.00 | 1822.65 | 1813.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-14 15:15:00 | 1831.30 | 1822.65 | 1813.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 10:30:00 | 1837.60 | 1820.34 | 1815.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 1829.50 | 1833.57 | 1826.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 10:00:00 | 1829.50 | 1833.57 | 1826.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 10:15:00 | 1860.70 | 1839.00 | 1829.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 10:45:00 | 1838.75 | 1839.00 | 1829.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 09:15:00 | 1866.25 | 1854.22 | 1842.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-20 11:00:00 | 1882.40 | 1859.86 | 1846.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 13:15:00 | 1837.90 | 1861.98 | 1851.08 | SL hit (close<static) qty=1.00 sl=1842.10 alert=retest2 |

### Cycle 38 — SELL (started 2023-12-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-21 09:15:00 | 1817.80 | 1839.84 | 1842.44 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 09:15:00 | 1933.70 | 1842.12 | 1838.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 13:15:00 | 1988.10 | 1921.11 | 1895.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 15:15:00 | 1924.95 | 1935.10 | 1920.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-28 15:15:00 | 1924.95 | 1935.10 | 1920.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 15:15:00 | 1924.95 | 1935.10 | 1920.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-29 09:15:00 | 1918.50 | 1935.10 | 1920.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 1926.95 | 1933.47 | 1921.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 09:15:00 | 1946.40 | 1926.98 | 1922.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-01 14:15:00 | 1902.00 | 1918.90 | 1920.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-01 14:15:00 | 1902.00 | 1918.90 | 1920.62 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-02 09:15:00 | 1947.15 | 1921.93 | 1921.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-03 09:15:00 | 2012.05 | 1968.64 | 1949.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-04 13:15:00 | 2012.75 | 2025.12 | 2002.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-04 14:00:00 | 2012.75 | 2025.12 | 2002.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 14:15:00 | 1974.15 | 2014.93 | 2000.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-04 14:45:00 | 1974.00 | 2014.93 | 2000.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 15:15:00 | 1980.00 | 2007.94 | 1998.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-05 09:15:00 | 2018.25 | 2007.94 | 1998.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-05 15:15:00 | 1990.00 | 1995.41 | 1995.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-05 15:15:00 | 1990.00 | 1995.41 | 1995.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 09:15:00 | 1978.65 | 1992.06 | 1994.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 12:15:00 | 1944.75 | 1939.61 | 1957.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-09 13:00:00 | 1944.75 | 1939.61 | 1957.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 1922.85 | 1937.59 | 1950.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-10 10:45:00 | 1904.15 | 1932.06 | 1947.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 11:15:00 | 1915.30 | 1917.78 | 1929.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-11 14:15:00 | 1966.00 | 1932.00 | 1932.80 | SL hit (close>static) qty=1.00 sl=1963.05 alert=retest2 |

### Cycle 43 — BUY (started 2024-01-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-11 15:15:00 | 1945.00 | 1934.60 | 1933.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-12 10:15:00 | 1971.00 | 1945.99 | 1939.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-15 09:15:00 | 1957.75 | 1959.67 | 1950.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-15 10:00:00 | 1957.75 | 1959.67 | 1950.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 11:15:00 | 1952.90 | 1957.89 | 1951.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 11:30:00 | 1955.95 | 1957.89 | 1951.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 12:15:00 | 1961.00 | 1958.51 | 1952.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-16 09:30:00 | 1963.95 | 1956.86 | 1953.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-16 11:15:00 | 1932.65 | 1951.24 | 1951.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 11:15:00 | 1932.65 | 1951.24 | 1951.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 12:15:00 | 1914.80 | 1943.95 | 1947.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-17 09:15:00 | 1953.05 | 1942.53 | 1945.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 09:15:00 | 1953.05 | 1942.53 | 1945.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 1953.05 | 1942.53 | 1945.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-17 09:45:00 | 1954.80 | 1942.53 | 1945.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 10:15:00 | 1932.70 | 1940.57 | 1944.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-17 12:00:00 | 1918.10 | 1936.07 | 1942.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-17 15:15:00 | 1960.75 | 1941.30 | 1942.22 | SL hit (close>static) qty=1.00 sl=1953.05 alert=retest2 |

### Cycle 45 — BUY (started 2024-01-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-18 13:15:00 | 1951.05 | 1940.00 | 1939.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-18 14:15:00 | 1953.75 | 1942.75 | 1941.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-19 14:15:00 | 1962.75 | 1975.39 | 1962.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 14:15:00 | 1962.75 | 1975.39 | 1962.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 14:15:00 | 1962.75 | 1975.39 | 1962.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-19 15:00:00 | 1962.75 | 1975.39 | 1962.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 15:15:00 | 1969.95 | 1974.30 | 1963.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-20 09:15:00 | 1977.15 | 1974.30 | 1963.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-20 12:15:00 | 1949.45 | 1963.52 | 1961.18 | SL hit (close<static) qty=1.00 sl=1958.30 alert=retest2 |

### Cycle 46 — SELL (started 2024-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 13:15:00 | 1941.80 | 1959.18 | 1959.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 09:15:00 | 1924.65 | 1946.92 | 1953.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 09:15:00 | 1950.00 | 1916.54 | 1930.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 09:15:00 | 1950.00 | 1916.54 | 1930.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 09:15:00 | 1950.00 | 1916.54 | 1930.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 09:45:00 | 1952.10 | 1916.54 | 1930.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 10:15:00 | 1965.35 | 1926.30 | 1933.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 10:30:00 | 1968.80 | 1926.30 | 1933.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2024-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 12:15:00 | 1969.25 | 1940.26 | 1938.87 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-01-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 09:15:00 | 1915.95 | 1937.93 | 1938.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 11:15:00 | 1911.00 | 1929.66 | 1934.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-30 09:15:00 | 1893.05 | 1887.51 | 1900.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 09:15:00 | 1893.05 | 1887.51 | 1900.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 09:15:00 | 1893.05 | 1887.51 | 1900.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-30 09:45:00 | 1903.00 | 1887.51 | 1900.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 14:15:00 | 1875.00 | 1883.25 | 1893.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-30 14:30:00 | 1900.00 | 1883.25 | 1893.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 15:15:00 | 1897.00 | 1886.00 | 1893.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-31 09:15:00 | 1912.20 | 1886.00 | 1893.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 09:15:00 | 1920.80 | 1892.96 | 1896.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-31 10:00:00 | 1920.80 | 1892.96 | 1896.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2024-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 10:15:00 | 1936.40 | 1901.65 | 1899.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 12:15:00 | 1941.50 | 1914.78 | 1906.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-02 14:15:00 | 1992.55 | 1996.16 | 1973.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-02 15:00:00 | 1992.55 | 1996.16 | 1973.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 10:15:00 | 2068.45 | 2098.24 | 2065.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-07 11:00:00 | 2068.45 | 2098.24 | 2065.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 11:15:00 | 2051.80 | 2088.95 | 2064.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-07 11:45:00 | 2052.95 | 2088.95 | 2064.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 12:15:00 | 2056.80 | 2082.52 | 2063.66 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2024-02-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 09:15:00 | 2013.65 | 2054.35 | 2054.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 10:15:00 | 1994.10 | 2042.30 | 2049.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-09 14:15:00 | 2003.45 | 2002.49 | 2016.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-09 15:00:00 | 2003.45 | 2002.49 | 2016.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 2003.30 | 2004.17 | 2014.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-12 09:45:00 | 2036.75 | 2004.17 | 2014.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 11:15:00 | 1998.80 | 2002.86 | 2012.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-12 12:45:00 | 1994.15 | 2001.62 | 2010.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-12 14:45:00 | 1993.00 | 2000.61 | 2008.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-13 09:15:00 | 1962.25 | 2000.48 | 2007.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-13 11:45:00 | 1993.65 | 1996.00 | 2003.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 12:15:00 | 2008.65 | 1998.53 | 2004.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 13:00:00 | 2008.65 | 1998.53 | 2004.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 13:15:00 | 1996.00 | 1998.03 | 2003.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-13 14:15:00 | 1992.85 | 1998.03 | 2003.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-15 11:15:00 | 1894.44 | 1935.65 | 1958.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-15 11:15:00 | 1893.35 | 1935.65 | 1958.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-15 11:15:00 | 1893.97 | 1935.65 | 1958.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-15 11:15:00 | 1893.21 | 1935.65 | 1958.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-15 12:15:00 | 1970.50 | 1942.62 | 1959.43 | SL hit (close>ema200) qty=0.50 sl=1942.62 alert=retest2 |

### Cycle 51 — BUY (started 2024-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 11:15:00 | 2037.90 | 1968.86 | 1962.66 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 11:15:00 | 1952.05 | 1980.11 | 1981.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 09:15:00 | 1939.45 | 1956.50 | 1965.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 13:15:00 | 1826.85 | 1826.56 | 1849.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-27 14:00:00 | 1826.85 | 1826.56 | 1849.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 11:15:00 | 1788.15 | 1771.85 | 1789.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 11:45:00 | 1787.00 | 1771.85 | 1789.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 12:15:00 | 1798.60 | 1777.20 | 1790.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 12:45:00 | 1796.65 | 1777.20 | 1790.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 13:15:00 | 1783.00 | 1778.36 | 1789.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 13:30:00 | 1797.05 | 1778.36 | 1789.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 14:15:00 | 1792.30 | 1781.15 | 1789.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 15:00:00 | 1792.30 | 1781.15 | 1789.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 15:15:00 | 1782.00 | 1781.32 | 1788.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 09:15:00 | 1799.50 | 1781.32 | 1788.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 09:15:00 | 1799.00 | 1784.85 | 1789.87 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2024-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-04 11:15:00 | 1798.05 | 1792.83 | 1792.44 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 13:15:00 | 1760.95 | 1787.61 | 1790.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-04 14:15:00 | 1691.05 | 1768.30 | 1781.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-05 12:15:00 | 1766.50 | 1753.53 | 1767.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-05 12:15:00 | 1766.50 | 1753.53 | 1767.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 12:15:00 | 1766.50 | 1753.53 | 1767.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-05 12:45:00 | 1770.50 | 1753.53 | 1767.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 13:15:00 | 1766.10 | 1756.04 | 1767.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-05 13:30:00 | 1772.50 | 1756.04 | 1767.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 14:15:00 | 1752.70 | 1755.37 | 1765.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-06 09:15:00 | 1741.00 | 1756.68 | 1765.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-07 11:15:00 | 1768.95 | 1756.24 | 1757.25 | SL hit (close>static) qty=1.00 sl=1766.10 alert=retest2 |

### Cycle 55 — BUY (started 2024-03-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 13:15:00 | 1744.95 | 1729.78 | 1727.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-15 11:15:00 | 1775.45 | 1747.70 | 1738.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-18 10:15:00 | 1770.95 | 1772.51 | 1757.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-18 10:15:00 | 1770.95 | 1772.51 | 1757.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 10:15:00 | 1770.95 | 1772.51 | 1757.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-18 11:00:00 | 1770.95 | 1772.51 | 1757.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 09:15:00 | 1772.30 | 1779.11 | 1767.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 10:15:00 | 1763.80 | 1779.11 | 1767.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 10:15:00 | 1764.00 | 1776.09 | 1767.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 11:00:00 | 1764.00 | 1776.09 | 1767.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 11:15:00 | 1765.40 | 1773.95 | 1767.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 11:30:00 | 1760.85 | 1773.95 | 1767.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 12:15:00 | 1761.35 | 1771.43 | 1766.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 13:00:00 | 1761.35 | 1771.43 | 1766.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 13:15:00 | 1763.15 | 1769.77 | 1766.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 14:00:00 | 1763.15 | 1769.77 | 1766.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 15:15:00 | 1781.25 | 1772.31 | 1768.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 09:15:00 | 1754.10 | 1772.31 | 1768.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2024-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 09:15:00 | 1737.85 | 1765.42 | 1765.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 10:15:00 | 1727.20 | 1757.78 | 1762.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 09:15:00 | 1746.95 | 1735.70 | 1747.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 09:15:00 | 1746.95 | 1735.70 | 1747.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 1746.95 | 1735.70 | 1747.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 10:00:00 | 1746.95 | 1735.70 | 1747.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 10:15:00 | 1755.05 | 1739.57 | 1748.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 11:00:00 | 1755.05 | 1739.57 | 1748.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 11:15:00 | 1746.20 | 1740.90 | 1747.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 11:45:00 | 1747.55 | 1740.90 | 1747.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 12:15:00 | 1747.25 | 1742.17 | 1747.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 12:45:00 | 1750.10 | 1742.17 | 1747.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 13:15:00 | 1746.80 | 1743.09 | 1747.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 13:45:00 | 1745.70 | 1743.09 | 1747.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 14:15:00 | 1752.80 | 1745.04 | 1748.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 14:45:00 | 1757.40 | 1745.04 | 1748.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 15:15:00 | 1759.90 | 1748.01 | 1749.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-22 09:15:00 | 1768.00 | 1748.01 | 1749.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2024-03-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 09:15:00 | 1759.40 | 1750.29 | 1750.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 11:15:00 | 1781.00 | 1757.58 | 1753.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 09:15:00 | 1773.25 | 1778.62 | 1767.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 1773.25 | 1778.62 | 1767.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 1773.25 | 1778.62 | 1767.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 09:30:00 | 1772.90 | 1778.62 | 1767.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 10:15:00 | 1778.35 | 1778.57 | 1768.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-26 12:15:00 | 1792.70 | 1778.11 | 1769.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 09:15:00 | 1804.95 | 1784.31 | 1775.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-02 12:15:00 | 1793.30 | 1815.27 | 1815.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2024-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-02 12:15:00 | 1793.30 | 1815.27 | 1815.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-04 13:15:00 | 1778.05 | 1789.16 | 1795.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-08 09:15:00 | 1847.95 | 1781.23 | 1783.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-08 09:15:00 | 1847.95 | 1781.23 | 1783.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 09:15:00 | 1847.95 | 1781.23 | 1783.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-08 10:00:00 | 1847.95 | 1781.23 | 1783.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2024-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 10:15:00 | 1863.00 | 1797.58 | 1790.39 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 12:15:00 | 1776.50 | 1804.97 | 1805.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 09:15:00 | 1752.45 | 1781.81 | 1790.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-15 12:15:00 | 1750.00 | 1749.65 | 1763.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-15 13:00:00 | 1750.00 | 1749.65 | 1763.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 1763.25 | 1752.99 | 1760.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 10:00:00 | 1763.25 | 1752.99 | 1760.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 1762.05 | 1754.80 | 1761.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 10:30:00 | 1763.05 | 1754.80 | 1761.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 11:15:00 | 1745.75 | 1752.99 | 1759.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-16 12:30:00 | 1741.25 | 1752.15 | 1758.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-18 11:15:00 | 1770.00 | 1757.43 | 1758.30 | SL hit (close>static) qty=1.00 sl=1763.00 alert=retest2 |

### Cycle 61 — BUY (started 2024-04-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 12:15:00 | 1775.00 | 1760.94 | 1759.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-19 13:15:00 | 1776.80 | 1767.02 | 1763.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 14:15:00 | 1763.00 | 1766.22 | 1763.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 14:15:00 | 1763.00 | 1766.22 | 1763.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 14:15:00 | 1763.00 | 1766.22 | 1763.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-19 15:00:00 | 1763.00 | 1766.22 | 1763.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 15:15:00 | 1760.00 | 1764.97 | 1763.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 10:30:00 | 1772.25 | 1765.87 | 1764.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 12:00:00 | 1772.60 | 1767.22 | 1764.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 15:00:00 | 1775.80 | 1768.57 | 1766.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-24 15:15:00 | 1756.00 | 1771.87 | 1773.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2024-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 15:15:00 | 1756.00 | 1771.87 | 1773.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-25 13:15:00 | 1740.00 | 1757.88 | 1765.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-29 14:15:00 | 1725.80 | 1723.91 | 1735.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-29 15:00:00 | 1725.80 | 1723.91 | 1735.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 15:15:00 | 1725.15 | 1724.16 | 1734.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 09:15:00 | 1732.80 | 1724.16 | 1734.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 09:15:00 | 1725.00 | 1724.33 | 1733.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-30 12:00:00 | 1722.45 | 1724.68 | 1732.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 12:15:00 | 1714.95 | 1715.98 | 1722.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 09:15:00 | 1717.25 | 1717.00 | 1720.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-07 12:15:00 | 1715.05 | 1703.90 | 1703.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2024-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-07 12:15:00 | 1715.05 | 1703.90 | 1703.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-07 13:15:00 | 1721.15 | 1707.35 | 1704.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 15:15:00 | 1700.00 | 1706.99 | 1705.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-07 15:15:00 | 1700.00 | 1706.99 | 1705.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 15:15:00 | 1700.00 | 1706.99 | 1705.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-08 09:15:00 | 1701.10 | 1706.99 | 1705.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 1712.90 | 1708.17 | 1705.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-08 15:15:00 | 1720.00 | 1707.55 | 1706.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-09 11:15:00 | 1700.00 | 1706.20 | 1706.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2024-05-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 11:15:00 | 1700.00 | 1706.20 | 1706.33 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2024-05-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-09 12:15:00 | 1709.25 | 1706.81 | 1706.59 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-05-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 14:15:00 | 1693.80 | 1704.55 | 1705.62 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-05-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 13:15:00 | 1708.10 | 1705.61 | 1705.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-10 14:15:00 | 1711.20 | 1706.73 | 1705.93 | Break + close above crossover candle high |

### Cycle 68 — SELL (started 2024-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 09:15:00 | 1695.65 | 1705.23 | 1705.43 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 10:15:00 | 1739.85 | 1712.16 | 1708.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 13:15:00 | 1750.00 | 1729.78 | 1718.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 14:15:00 | 1778.95 | 1781.77 | 1770.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 14:45:00 | 1781.75 | 1781.77 | 1770.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 1781.00 | 1781.62 | 1771.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:45:00 | 1765.00 | 1779.59 | 1771.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 10:15:00 | 1770.00 | 1777.67 | 1771.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 12:30:00 | 1775.80 | 1776.65 | 1772.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-23 09:15:00 | 1953.38 | 1807.72 | 1795.84 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 1834.95 | 1850.53 | 1852.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 13:15:00 | 1823.65 | 1845.16 | 1849.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 1836.95 | 1834.57 | 1843.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 09:15:00 | 1836.95 | 1834.57 | 1843.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 1836.95 | 1834.57 | 1843.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:45:00 | 1849.10 | 1834.57 | 1843.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 1840.90 | 1835.84 | 1842.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:30:00 | 1840.60 | 1835.84 | 1842.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 11:15:00 | 1861.85 | 1841.04 | 1844.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 11:45:00 | 1860.60 | 1841.04 | 1844.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 12:15:00 | 1861.75 | 1845.18 | 1846.18 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 13:15:00 | 1861.85 | 1848.52 | 1847.61 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 12:15:00 | 1844.00 | 1849.96 | 1850.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 13:15:00 | 1836.90 | 1847.35 | 1849.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 12:15:00 | 1851.50 | 1843.17 | 1845.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 12:15:00 | 1851.50 | 1843.17 | 1845.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 12:15:00 | 1851.50 | 1843.17 | 1845.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 12:30:00 | 1850.60 | 1843.17 | 1845.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 13:15:00 | 1851.45 | 1844.83 | 1846.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 14:15:00 | 1851.75 | 1844.83 | 1846.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 15:15:00 | 1850.00 | 1846.95 | 1846.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:15:00 | 1820.00 | 1846.95 | 1846.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 1789.55 | 1834.87 | 1841.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 12:00:00 | 1754.70 | 1818.84 | 1833.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 1666.96 | 1814.01 | 1830.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-05 09:15:00 | 1836.35 | 1810.26 | 1822.31 | SL hit (close>ema200) qty=0.50 sl=1810.26 alert=retest2 |

### Cycle 73 — BUY (started 2024-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 13:15:00 | 1845.00 | 1828.57 | 1828.24 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-05 15:15:00 | 1789.00 | 1823.20 | 1826.02 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 1836.25 | 1828.09 | 1827.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 1856.90 | 1839.51 | 1834.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 12:15:00 | 1842.20 | 1845.00 | 1838.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-07 12:30:00 | 1843.40 | 1845.00 | 1838.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 13:15:00 | 1838.15 | 1843.63 | 1838.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-07 14:00:00 | 1838.15 | 1843.63 | 1838.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 14:15:00 | 1860.05 | 1846.91 | 1840.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 09:15:00 | 1877.00 | 1848.13 | 1841.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 10:00:00 | 1875.25 | 1853.55 | 1844.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-13 11:15:00 | 1854.15 | 1872.27 | 1874.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2024-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 11:15:00 | 1854.15 | 1872.27 | 1874.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 15:15:00 | 1844.00 | 1861.30 | 1866.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 11:15:00 | 1865.00 | 1857.67 | 1862.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 11:15:00 | 1865.00 | 1857.67 | 1862.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 11:15:00 | 1865.00 | 1857.67 | 1862.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 11:30:00 | 1861.20 | 1857.67 | 1862.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 12:15:00 | 1855.00 | 1857.14 | 1862.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-18 13:45:00 | 1853.00 | 1856.10 | 1861.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 14:15:00 | 1824.00 | 1820.39 | 1820.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2024-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 14:15:00 | 1824.00 | 1820.39 | 1820.21 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2024-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 15:15:00 | 1810.00 | 1818.31 | 1819.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 12:15:00 | 1806.95 | 1813.38 | 1816.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 09:15:00 | 1804.95 | 1802.15 | 1809.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 09:15:00 | 1804.95 | 1802.15 | 1809.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 1804.95 | 1802.15 | 1809.37 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2024-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 11:15:00 | 1820.00 | 1810.82 | 1809.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 13:15:00 | 1825.00 | 1815.12 | 1812.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 09:15:00 | 1818.25 | 1818.64 | 1814.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-01 10:00:00 | 1818.25 | 1818.64 | 1814.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 1825.30 | 1819.97 | 1815.70 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 1814.00 | 1815.76 | 1815.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 09:15:00 | 1809.80 | 1811.33 | 1812.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 11:15:00 | 1819.65 | 1812.36 | 1812.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 11:15:00 | 1819.65 | 1812.36 | 1812.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 11:15:00 | 1819.65 | 1812.36 | 1812.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-05 12:00:00 | 1819.65 | 1812.36 | 1812.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-07-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 12:15:00 | 1841.65 | 1818.21 | 1815.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 10:15:00 | 1875.05 | 1840.47 | 1828.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 14:15:00 | 1988.25 | 1991.79 | 1953.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-10 14:45:00 | 1993.95 | 1991.79 | 1953.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 1987.00 | 1990.84 | 1956.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 09:15:00 | 2025.40 | 1990.84 | 1956.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 09:15:00 | 1988.85 | 2006.32 | 2008.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 09:15:00 | 1988.85 | 2006.32 | 2008.19 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 12:15:00 | 2046.85 | 2013.18 | 2010.68 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 13:15:00 | 2009.90 | 2015.32 | 2015.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 1983.85 | 2008.07 | 2012.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 14:15:00 | 1999.90 | 1994.99 | 2002.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-19 15:00:00 | 1999.90 | 1994.99 | 2002.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 1996.05 | 1994.88 | 2000.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:30:00 | 2007.70 | 1994.88 | 2000.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 1990.00 | 1993.90 | 1999.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:15:00 | 1995.55 | 1993.90 | 1999.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 2002.05 | 1995.53 | 1999.94 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 11:15:00 | 2008.50 | 2000.87 | 2000.77 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 12:15:00 | 1988.50 | 1998.40 | 1999.65 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 12:15:00 | 2016.00 | 2000.48 | 1998.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 13:15:00 | 2020.60 | 2004.50 | 2000.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 1995.85 | 2007.20 | 2003.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 09:15:00 | 1995.85 | 2007.20 | 2003.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 1995.85 | 2007.20 | 2003.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 10:30:00 | 2015.45 | 2009.17 | 2004.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 11:00:00 | 2017.05 | 2009.17 | 2004.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 09:45:00 | 2029.50 | 2015.82 | 2010.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 11:15:00 | 2075.25 | 2110.90 | 2111.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 11:15:00 | 2075.25 | 2110.90 | 2111.06 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-05 12:15:00 | 2113.95 | 2111.51 | 2111.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-05 13:15:00 | 2129.85 | 2115.18 | 2113.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 14:15:00 | 2111.20 | 2114.38 | 2112.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 14:15:00 | 2111.20 | 2114.38 | 2112.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 14:15:00 | 2111.20 | 2114.38 | 2112.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 14:45:00 | 2114.60 | 2114.38 | 2112.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 15:15:00 | 2117.50 | 2115.01 | 2113.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-06 09:15:00 | 2174.55 | 2115.01 | 2113.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 09:15:00 | 2014.10 | 2109.38 | 2118.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2024-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 09:15:00 | 2014.10 | 2109.38 | 2118.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 10:15:00 | 1900.00 | 1927.41 | 1940.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 12:15:00 | 1870.00 | 1869.57 | 1887.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 12:15:00 | 1870.00 | 1869.57 | 1887.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 1870.00 | 1869.57 | 1887.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 13:00:00 | 1870.00 | 1869.57 | 1887.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 1873.25 | 1864.82 | 1879.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:45:00 | 1887.45 | 1864.82 | 1879.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 1908.95 | 1873.66 | 1880.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 12:00:00 | 1908.95 | 1873.66 | 1880.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 1876.10 | 1874.15 | 1880.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 13:15:00 | 1874.10 | 1874.15 | 1880.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 09:15:00 | 1873.00 | 1872.73 | 1877.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 10:15:00 | 1912.05 | 1858.04 | 1852.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 10:15:00 | 1912.05 | 1858.04 | 1852.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 11:15:00 | 1930.70 | 1872.57 | 1859.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 10:15:00 | 1877.40 | 1888.31 | 1875.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 10:15:00 | 1877.40 | 1888.31 | 1875.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 10:15:00 | 1877.40 | 1888.31 | 1875.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 10:45:00 | 1875.40 | 1888.31 | 1875.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 11:15:00 | 1887.00 | 1888.05 | 1876.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 11:30:00 | 1877.60 | 1888.05 | 1876.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 1890.00 | 1895.91 | 1888.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 15:00:00 | 1890.00 | 1895.91 | 1888.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 15:15:00 | 1895.00 | 1895.73 | 1889.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 09:15:00 | 1906.65 | 1895.73 | 1889.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 1896.50 | 1895.88 | 1889.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 09:45:00 | 1917.40 | 1906.07 | 1898.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-10 09:15:00 | 1928.00 | 1911.45 | 1906.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 13:15:00 | 1890.00 | 1916.29 | 1918.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2024-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 13:15:00 | 1890.00 | 1916.29 | 1918.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 14:15:00 | 1888.85 | 1910.80 | 1915.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 10:15:00 | 1904.00 | 1901.47 | 1909.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-12 11:00:00 | 1904.00 | 1901.47 | 1909.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 1871.00 | 1876.38 | 1886.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 14:15:00 | 1865.00 | 1873.07 | 1881.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 15:00:00 | 1865.05 | 1871.47 | 1879.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 12:15:00 | 1845.70 | 1829.69 | 1827.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 12:15:00 | 1845.70 | 1829.69 | 1827.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 09:15:00 | 1857.85 | 1841.73 | 1834.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 12:15:00 | 1856.85 | 1861.10 | 1851.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 13:00:00 | 1856.85 | 1861.10 | 1851.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 1859.70 | 1860.82 | 1852.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 13:45:00 | 1859.95 | 1860.82 | 1852.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 1856.90 | 1859.65 | 1853.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:15:00 | 1847.50 | 1859.65 | 1853.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 1857.20 | 1859.16 | 1853.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:30:00 | 1852.35 | 1859.16 | 1853.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 1848.90 | 1857.11 | 1853.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 11:00:00 | 1848.90 | 1857.11 | 1853.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 1835.25 | 1852.73 | 1851.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 12:00:00 | 1835.25 | 1852.73 | 1851.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2024-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 12:15:00 | 1828.35 | 1847.86 | 1849.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 09:15:00 | 1796.25 | 1836.84 | 1843.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 14:15:00 | 1796.00 | 1791.42 | 1805.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-30 15:00:00 | 1796.00 | 1791.42 | 1805.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 1789.90 | 1791.37 | 1802.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 11:15:00 | 1785.60 | 1791.17 | 1801.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 12:30:00 | 1785.00 | 1789.71 | 1799.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:45:00 | 1782.00 | 1792.06 | 1797.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 09:15:00 | 1754.00 | 1781.18 | 1788.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 1775.60 | 1780.06 | 1787.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 14:45:00 | 1747.45 | 1766.39 | 1777.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 15:15:00 | 1744.00 | 1766.39 | 1777.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 09:30:00 | 1736.10 | 1752.94 | 1769.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 1696.32 | 1741.75 | 1762.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 1695.75 | 1741.75 | 1762.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 1692.90 | 1741.75 | 1762.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 10:15:00 | 1715.55 | 1712.70 | 1734.00 | SL hit (close>ema200) qty=0.50 sl=1712.70 alert=retest2 |

### Cycle 95 — BUY (started 2024-10-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 10:15:00 | 1680.70 | 1651.34 | 1649.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-21 11:15:00 | 1683.10 | 1657.69 | 1652.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 14:15:00 | 1666.30 | 1667.83 | 1658.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-21 14:30:00 | 1666.60 | 1667.83 | 1658.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 1665.85 | 1669.38 | 1661.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:30:00 | 1657.65 | 1669.38 | 1661.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 1663.00 | 1668.10 | 1661.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:30:00 | 1660.95 | 1668.10 | 1661.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 11:15:00 | 1659.00 | 1666.28 | 1661.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 11:45:00 | 1657.65 | 1666.28 | 1661.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 12:15:00 | 1646.00 | 1662.23 | 1659.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 13:00:00 | 1646.00 | 1662.23 | 1659.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 13:15:00 | 1649.00 | 1659.58 | 1658.84 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2024-10-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 14:15:00 | 1646.25 | 1656.92 | 1657.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 15:15:00 | 1640.00 | 1653.53 | 1656.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 12:15:00 | 1628.50 | 1620.00 | 1629.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 12:15:00 | 1628.50 | 1620.00 | 1629.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 12:15:00 | 1628.50 | 1620.00 | 1629.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 12:30:00 | 1634.00 | 1620.00 | 1629.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 13:15:00 | 1631.20 | 1622.24 | 1629.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 14:45:00 | 1626.80 | 1623.84 | 1629.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-24 15:15:00 | 1650.80 | 1629.23 | 1631.84 | SL hit (close>static) qty=1.00 sl=1640.20 alert=retest2 |

### Cycle 97 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 1625.95 | 1621.53 | 1621.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 10:15:00 | 1646.05 | 1628.50 | 1624.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 15:15:00 | 1626.50 | 1634.34 | 1629.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 15:15:00 | 1626.50 | 1634.34 | 1629.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 1626.50 | 1634.34 | 1629.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:15:00 | 1637.00 | 1634.34 | 1629.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 1640.55 | 1635.58 | 1630.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 12:45:00 | 1653.00 | 1637.25 | 1632.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 15:00:00 | 1652.05 | 1640.74 | 1635.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:15:00 | 1653.05 | 1643.96 | 1637.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 1620.30 | 1639.03 | 1636.55 | SL hit (close<static) qty=1.00 sl=1623.80 alert=retest2 |

### Cycle 98 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 1619.75 | 1634.37 | 1634.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 14:15:00 | 1615.50 | 1628.40 | 1631.99 | Break + close below crossover candle low |

### Cycle 99 — BUY (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 09:15:00 | 1785.50 | 1656.24 | 1643.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 10:15:00 | 1805.80 | 1686.15 | 1658.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 1819.10 | 1821.89 | 1777.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 09:45:00 | 1823.80 | 1821.89 | 1777.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 1783.60 | 1805.52 | 1790.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:30:00 | 1781.40 | 1805.52 | 1790.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 1782.05 | 1800.83 | 1789.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 10:45:00 | 1779.80 | 1800.83 | 1789.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 1787.60 | 1795.11 | 1788.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 09:15:00 | 1794.15 | 1787.55 | 1786.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 09:45:00 | 1794.10 | 1791.74 | 1788.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-12 13:00:00 | 1792.30 | 1804.19 | 1801.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-12 13:15:00 | 1781.95 | 1799.74 | 1799.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2024-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 13:15:00 | 1781.95 | 1799.74 | 1799.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 14:15:00 | 1771.60 | 1794.11 | 1797.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 1733.85 | 1730.05 | 1754.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 1733.85 | 1730.05 | 1754.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 1758.45 | 1737.57 | 1753.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 11:45:00 | 1753.50 | 1737.57 | 1753.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 1770.00 | 1744.06 | 1755.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:00:00 | 1770.00 | 1744.06 | 1755.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 1765.95 | 1748.44 | 1756.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:30:00 | 1730.85 | 1754.69 | 1757.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 12:15:00 | 1764.75 | 1759.80 | 1759.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2024-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 12:15:00 | 1764.75 | 1759.80 | 1759.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-18 13:15:00 | 1771.10 | 1762.06 | 1760.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 15:15:00 | 1757.00 | 1771.75 | 1768.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 15:15:00 | 1757.00 | 1771.75 | 1768.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 1757.00 | 1771.75 | 1768.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:15:00 | 1745.00 | 1771.75 | 1768.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 1755.55 | 1768.51 | 1767.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:30:00 | 1743.50 | 1768.51 | 1767.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 1762.95 | 1777.26 | 1774.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 11:00:00 | 1762.95 | 1777.26 | 1774.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 1800.00 | 1781.81 | 1776.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 11:30:00 | 1761.55 | 1781.81 | 1776.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 1765.20 | 1778.49 | 1775.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 12:30:00 | 1763.25 | 1778.49 | 1775.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 13:15:00 | 1763.15 | 1775.42 | 1774.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 13:30:00 | 1760.50 | 1775.42 | 1774.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2024-11-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 14:15:00 | 1756.40 | 1771.62 | 1772.69 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 15:15:00 | 1780.00 | 1772.24 | 1771.92 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 09:15:00 | 1757.60 | 1769.31 | 1770.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 14:15:00 | 1739.85 | 1758.58 | 1764.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 1742.10 | 1739.34 | 1749.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 09:15:00 | 1742.10 | 1739.34 | 1749.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 1742.10 | 1739.34 | 1749.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:00:00 | 1742.10 | 1739.34 | 1749.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 1733.00 | 1738.07 | 1747.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 14:30:00 | 1721.50 | 1732.29 | 1741.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 10:00:00 | 1718.90 | 1726.52 | 1737.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 11:30:00 | 1720.10 | 1727.71 | 1736.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 09:15:00 | 1749.50 | 1736.05 | 1737.31 | SL hit (close>static) qty=1.00 sl=1748.90 alert=retest2 |

### Cycle 105 — BUY (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 10:15:00 | 1763.35 | 1741.51 | 1739.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 11:15:00 | 1780.85 | 1749.38 | 1743.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 10:15:00 | 1803.70 | 1804.54 | 1786.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 10:30:00 | 1800.00 | 1804.54 | 1786.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 1791.00 | 1800.95 | 1791.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:15:00 | 1792.05 | 1800.95 | 1791.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 1804.30 | 1801.62 | 1792.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 10:45:00 | 1809.50 | 1804.70 | 1794.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 11:30:00 | 1814.95 | 1807.65 | 1797.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 09:15:00 | 1814.25 | 1813.96 | 1810.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 11:15:00 | 1800.00 | 1806.96 | 1807.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2024-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 11:15:00 | 1800.00 | 1806.96 | 1807.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 13:15:00 | 1793.85 | 1803.06 | 1805.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 10:15:00 | 1780.00 | 1777.24 | 1786.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 10:15:00 | 1780.00 | 1777.24 | 1786.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 1780.00 | 1777.24 | 1786.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 15:15:00 | 1770.55 | 1778.28 | 1784.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 09:15:00 | 1806.50 | 1782.68 | 1785.10 | SL hit (close>static) qty=1.00 sl=1788.30 alert=retest2 |

### Cycle 107 — BUY (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 10:15:00 | 1809.00 | 1787.95 | 1787.27 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 15:15:00 | 1772.00 | 1787.33 | 1787.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 1748.30 | 1779.52 | 1784.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-17 10:15:00 | 1769.40 | 1745.71 | 1753.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 10:15:00 | 1769.40 | 1745.71 | 1753.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 1769.40 | 1745.71 | 1753.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 11:00:00 | 1769.40 | 1745.71 | 1753.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 11:15:00 | 1744.00 | 1745.37 | 1752.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 12:15:00 | 1735.00 | 1745.37 | 1752.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 14:00:00 | 1743.95 | 1743.42 | 1750.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 09:15:00 | 1770.00 | 1750.57 | 1752.05 | SL hit (close>static) qty=1.00 sl=1769.40 alert=retest2 |

### Cycle 109 — BUY (started 2024-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 10:15:00 | 1775.95 | 1755.65 | 1754.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-20 09:15:00 | 1786.80 | 1774.72 | 1767.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 13:15:00 | 1777.00 | 1778.52 | 1772.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-20 14:00:00 | 1777.00 | 1778.52 | 1772.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 1775.35 | 1777.88 | 1772.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 1775.35 | 1777.88 | 1772.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 1770.00 | 1776.31 | 1772.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:15:00 | 1754.25 | 1776.31 | 1772.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 1764.00 | 1773.85 | 1771.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:30:00 | 1748.25 | 1773.85 | 1771.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 1763.90 | 1771.86 | 1770.74 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2024-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 11:15:00 | 1758.90 | 1769.26 | 1769.67 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2024-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 13:15:00 | 1772.90 | 1769.27 | 1769.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-24 14:15:00 | 1779.55 | 1771.33 | 1770.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 12:15:00 | 1800.00 | 1803.73 | 1793.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-27 13:00:00 | 1800.00 | 1803.73 | 1793.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 1799.40 | 1802.86 | 1794.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 13:30:00 | 1794.80 | 1802.86 | 1794.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 1794.05 | 1803.95 | 1797.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:30:00 | 1788.45 | 1803.95 | 1797.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 1792.80 | 1801.72 | 1796.81 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 14:15:00 | 1779.00 | 1792.35 | 1793.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 10:15:00 | 1770.00 | 1784.92 | 1789.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 14:15:00 | 1782.60 | 1778.51 | 1784.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-31 15:00:00 | 1782.60 | 1778.51 | 1784.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 1789.90 | 1780.78 | 1784.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 09:15:00 | 1768.15 | 1780.78 | 1784.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 11:15:00 | 1820.50 | 1789.13 | 1787.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2025-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 11:15:00 | 1820.50 | 1789.13 | 1787.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 10:15:00 | 1839.70 | 1810.40 | 1799.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 13:15:00 | 1882.50 | 1896.32 | 1874.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 13:45:00 | 1884.95 | 1896.32 | 1874.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 1890.35 | 1893.61 | 1878.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:45:00 | 1881.30 | 1893.61 | 1878.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 1863.20 | 1886.07 | 1877.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:45:00 | 1863.95 | 1886.07 | 1877.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 1858.70 | 1880.60 | 1875.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 12:45:00 | 1859.05 | 1880.60 | 1875.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2025-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 14:15:00 | 1850.00 | 1869.82 | 1871.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 12:15:00 | 1838.75 | 1857.28 | 1864.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 09:15:00 | 1687.35 | 1682.12 | 1712.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 10:00:00 | 1687.35 | 1682.12 | 1712.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 10:15:00 | 1681.35 | 1678.23 | 1693.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 10:30:00 | 1692.05 | 1678.23 | 1693.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 11:15:00 | 1681.05 | 1678.80 | 1692.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 11:30:00 | 1685.25 | 1678.80 | 1692.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 12:15:00 | 1682.75 | 1679.59 | 1691.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 15:15:00 | 1670.00 | 1683.91 | 1687.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 12:15:00 | 1669.70 | 1678.39 | 1683.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 10:15:00 | 1586.50 | 1612.91 | 1632.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-24 10:15:00 | 1586.21 | 1612.91 | 1632.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-27 09:15:00 | 1503.00 | 1560.91 | 1595.63 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 115 — BUY (started 2025-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 12:15:00 | 1520.00 | 1508.09 | 1507.00 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 15:15:00 | 1502.00 | 1506.22 | 1506.45 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 10:15:00 | 1525.35 | 1510.19 | 1508.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 15:15:00 | 1536.00 | 1524.84 | 1519.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 09:15:00 | 1513.40 | 1522.56 | 1518.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 09:15:00 | 1513.40 | 1522.56 | 1518.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 1513.40 | 1522.56 | 1518.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 09:45:00 | 1510.00 | 1522.56 | 1518.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 1533.60 | 1524.76 | 1519.86 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2025-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 09:15:00 | 1439.05 | 1505.03 | 1513.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 11:15:00 | 1437.70 | 1481.21 | 1500.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 15:15:00 | 1480.85 | 1468.57 | 1486.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-05 09:15:00 | 1485.60 | 1468.57 | 1486.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 1491.75 | 1473.20 | 1487.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:00:00 | 1491.75 | 1473.20 | 1487.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 1493.70 | 1477.30 | 1487.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:30:00 | 1494.85 | 1477.30 | 1487.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 11:15:00 | 1493.10 | 1480.46 | 1488.20 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2025-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 09:15:00 | 1512.95 | 1494.27 | 1492.56 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 11:15:00 | 1484.90 | 1501.48 | 1502.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 13:15:00 | 1480.10 | 1494.38 | 1499.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 1452.50 | 1452.14 | 1467.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:00:00 | 1452.50 | 1452.14 | 1467.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 1480.10 | 1457.73 | 1468.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 1480.10 | 1457.73 | 1468.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 1495.25 | 1465.24 | 1470.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 15:15:00 | 1464.90 | 1470.03 | 1472.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-13 09:15:00 | 1502.00 | 1475.60 | 1474.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 09:15:00 | 1502.00 | 1475.60 | 1474.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 10:15:00 | 1509.15 | 1482.31 | 1477.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-13 13:15:00 | 1486.90 | 1489.55 | 1482.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-13 14:00:00 | 1486.90 | 1489.55 | 1482.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 14:15:00 | 1472.90 | 1486.22 | 1481.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-13 15:00:00 | 1472.90 | 1486.22 | 1481.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 15:15:00 | 1506.00 | 1490.18 | 1483.98 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2025-02-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 12:15:00 | 1459.65 | 1477.77 | 1479.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 13:15:00 | 1450.55 | 1472.32 | 1476.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 10:15:00 | 1470.00 | 1467.26 | 1472.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 10:15:00 | 1470.00 | 1467.26 | 1472.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 1470.00 | 1467.26 | 1472.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:30:00 | 1481.10 | 1467.26 | 1472.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 11:15:00 | 1472.95 | 1468.40 | 1472.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 11:45:00 | 1470.70 | 1468.40 | 1472.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 12:15:00 | 1475.10 | 1469.74 | 1472.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:15:00 | 1475.75 | 1469.74 | 1472.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 1469.10 | 1469.61 | 1472.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:45:00 | 1474.00 | 1469.61 | 1472.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 1473.55 | 1470.40 | 1472.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 1473.55 | 1470.40 | 1472.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 1480.00 | 1472.32 | 1473.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 1466.40 | 1472.32 | 1473.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 1473.00 | 1472.46 | 1473.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 09:15:00 | 1459.50 | 1468.57 | 1470.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 11:15:00 | 1477.85 | 1471.68 | 1471.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 1477.85 | 1471.68 | 1471.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 13:15:00 | 1497.30 | 1477.26 | 1474.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 15:15:00 | 1510.00 | 1514.00 | 1500.13 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-21 09:15:00 | 1539.65 | 1514.00 | 1500.13 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-21 14:30:00 | 1529.25 | 1526.39 | 1513.68 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-21 15:00:00 | 1528.40 | 1526.39 | 1513.68 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 1492.35 | 1519.68 | 1512.84 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-24 09:15:00 | 1492.35 | 1519.68 | 1512.84 | SL hit (close<ema400) qty=1.00 sl=1512.84 alert=retest1 |

### Cycle 124 — SELL (started 2025-02-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 13:15:00 | 1533.00 | 1572.57 | 1575.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-04 11:15:00 | 1516.10 | 1544.04 | 1554.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 09:15:00 | 1544.75 | 1534.98 | 1544.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-05 09:15:00 | 1544.75 | 1534.98 | 1544.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 1544.75 | 1534.98 | 1544.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:30:00 | 1554.80 | 1534.98 | 1544.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 1543.20 | 1536.62 | 1544.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 10:30:00 | 1550.90 | 1536.62 | 1544.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 11:15:00 | 1542.20 | 1537.74 | 1544.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:45:00 | 1544.15 | 1537.74 | 1544.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 12:15:00 | 1552.45 | 1540.68 | 1545.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 13:00:00 | 1552.45 | 1540.68 | 1545.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 13:15:00 | 1560.95 | 1544.73 | 1546.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 14:00:00 | 1560.95 | 1544.73 | 1546.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2025-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 14:15:00 | 1578.00 | 1551.39 | 1549.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 15:15:00 | 1580.00 | 1557.11 | 1552.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 14:15:00 | 1642.90 | 1645.55 | 1618.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 15:00:00 | 1642.90 | 1645.55 | 1618.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 1654.15 | 1650.29 | 1634.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 09:30:00 | 1667.20 | 1645.36 | 1634.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 10:15:00 | 1628.30 | 1641.95 | 1634.13 | SL hit (close<static) qty=1.00 sl=1633.50 alert=retest2 |

### Cycle 126 — SELL (started 2025-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 13:15:00 | 1610.15 | 1627.55 | 1628.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 15:15:00 | 1604.10 | 1619.91 | 1624.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 1571.40 | 1564.73 | 1579.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 1571.40 | 1564.73 | 1579.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 1571.40 | 1564.73 | 1579.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 15:15:00 | 1544.40 | 1564.30 | 1574.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 13:15:00 | 1580.70 | 1573.46 | 1573.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 13:15:00 | 1580.70 | 1573.46 | 1573.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 15:15:00 | 1586.35 | 1577.41 | 1575.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 12:15:00 | 1570.15 | 1578.42 | 1576.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 12:15:00 | 1570.15 | 1578.42 | 1576.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 12:15:00 | 1570.15 | 1578.42 | 1576.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 13:00:00 | 1570.15 | 1578.42 | 1576.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 13:15:00 | 1578.45 | 1578.43 | 1576.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 14:30:00 | 1582.80 | 1580.65 | 1578.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 12:15:00 | 1611.35 | 1622.21 | 1622.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 12:15:00 | 1611.35 | 1622.21 | 1622.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 13:15:00 | 1600.45 | 1617.86 | 1620.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 09:15:00 | 1612.25 | 1612.02 | 1617.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 09:15:00 | 1612.25 | 1612.02 | 1617.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 1612.25 | 1612.02 | 1617.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 12:15:00 | 1602.00 | 1611.10 | 1615.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 09:15:00 | 1597.80 | 1605.82 | 1611.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 09:45:00 | 1601.65 | 1604.22 | 1609.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 10:15:00 | 1588.00 | 1555.67 | 1566.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 14:15:00 | 1563.20 | 1569.36 | 1570.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 14:45:00 | 1572.30 | 1569.36 | 1570.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1489.70 | 1550.97 | 1561.94 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 09:15:00 | 1521.90 | 1550.97 | 1561.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 09:15:00 | 1517.91 | 1550.97 | 1561.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 09:15:00 | 1521.57 | 1550.97 | 1561.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 09:15:00 | 1508.60 | 1550.97 | 1561.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 10:15:00 | 1470.00 | 1550.97 | 1561.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-07 09:15:00 | 1441.80 | 1470.50 | 1510.13 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 129 — BUY (started 2025-04-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 15:15:00 | 1427.70 | 1416.46 | 1415.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 10:15:00 | 1434.30 | 1421.80 | 1417.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 1431.50 | 1433.47 | 1426.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 1431.50 | 1433.47 | 1426.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1431.50 | 1433.47 | 1426.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 10:15:00 | 1459.40 | 1445.05 | 1437.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 13:00:00 | 1455.00 | 1455.46 | 1450.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 1458.80 | 1453.36 | 1450.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 10:15:00 | 1468.30 | 1453.57 | 1450.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 1461.40 | 1473.26 | 1467.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 15:00:00 | 1461.40 | 1473.26 | 1467.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 15:15:00 | 1468.00 | 1472.21 | 1467.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 09:30:00 | 1451.30 | 1467.18 | 1465.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 1438.40 | 1461.43 | 1462.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 1438.40 | 1461.43 | 1462.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 10:15:00 | 1423.40 | 1442.74 | 1451.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-30 09:15:00 | 1419.40 | 1416.57 | 1425.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 09:15:00 | 1419.40 | 1416.57 | 1425.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 1419.40 | 1416.57 | 1425.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 11:45:00 | 1413.50 | 1416.75 | 1424.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 13:15:00 | 1411.30 | 1416.56 | 1423.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 14:15:00 | 1412.50 | 1416.85 | 1422.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:00:00 | 1413.20 | 1409.36 | 1416.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 11:15:00 | 1406.30 | 1408.75 | 1415.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 12:15:00 | 1404.90 | 1408.75 | 1415.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 14:15:00 | 1402.30 | 1408.40 | 1414.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 11:15:00 | 1423.60 | 1413.12 | 1414.28 | SL hit (close>static) qty=1.00 sl=1419.80 alert=retest2 |

### Cycle 131 — BUY (started 2025-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 13:15:00 | 1419.90 | 1415.63 | 1415.29 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 12:15:00 | 1409.90 | 1414.87 | 1415.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 09:15:00 | 1400.80 | 1412.14 | 1413.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 1414.20 | 1412.55 | 1413.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 10:15:00 | 1414.20 | 1412.55 | 1413.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 1414.20 | 1412.55 | 1413.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:00:00 | 1414.20 | 1412.55 | 1413.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 1412.10 | 1412.46 | 1413.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:30:00 | 1418.60 | 1412.46 | 1413.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 1401.60 | 1410.29 | 1412.65 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 1428.10 | 1413.78 | 1413.11 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 1394.50 | 1410.66 | 1412.40 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2025-05-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 14:15:00 | 1428.00 | 1413.62 | 1412.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 1430.10 | 1419.06 | 1415.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 1434.70 | 1435.60 | 1428.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 1445.60 | 1436.61 | 1430.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 1445.60 | 1436.61 | 1430.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 14:30:00 | 1456.70 | 1445.86 | 1437.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 10:15:00 | 1524.60 | 1534.19 | 1535.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 10:15:00 | 1524.60 | 1534.19 | 1535.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 09:15:00 | 1518.60 | 1526.40 | 1530.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 10:15:00 | 1550.30 | 1531.18 | 1532.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 10:15:00 | 1550.30 | 1531.18 | 1532.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1550.30 | 1531.18 | 1532.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:00:00 | 1550.30 | 1531.18 | 1532.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 11:15:00 | 1542.00 | 1533.34 | 1532.92 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 14:15:00 | 1529.00 | 1532.98 | 1532.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 09:15:00 | 1521.40 | 1529.71 | 1531.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 14:15:00 | 1526.00 | 1525.76 | 1528.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 14:15:00 | 1526.00 | 1525.76 | 1528.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 1526.00 | 1525.76 | 1528.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 15:00:00 | 1526.00 | 1525.76 | 1528.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 1517.90 | 1524.19 | 1527.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:15:00 | 1522.60 | 1524.19 | 1527.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 1523.60 | 1524.07 | 1527.11 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 11:15:00 | 1551.00 | 1532.88 | 1530.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 12:15:00 | 1591.30 | 1544.57 | 1536.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 1607.90 | 1613.59 | 1597.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 13:00:00 | 1607.90 | 1613.59 | 1597.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 1597.30 | 1610.33 | 1597.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 14:15:00 | 1596.90 | 1610.33 | 1597.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 1589.90 | 1606.25 | 1597.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 15:00:00 | 1589.90 | 1606.25 | 1597.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 1584.00 | 1601.80 | 1595.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:30:00 | 1614.30 | 1604.40 | 1597.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-16 14:15:00 | 1775.73 | 1735.58 | 1707.62 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2025-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 10:15:00 | 1719.30 | 1736.81 | 1738.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 11:15:00 | 1707.20 | 1730.89 | 1735.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 1733.40 | 1718.92 | 1726.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 1733.40 | 1718.92 | 1726.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1733.40 | 1718.92 | 1726.29 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 1741.30 | 1728.36 | 1727.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 1779.50 | 1743.41 | 1735.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 11:15:00 | 1783.40 | 1784.78 | 1777.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 11:45:00 | 1783.50 | 1784.78 | 1777.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 1782.70 | 1784.30 | 1778.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 1782.70 | 1784.30 | 1778.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 1785.10 | 1784.46 | 1779.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 1799.70 | 1784.46 | 1779.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 10:15:00 | 1863.80 | 1875.24 | 1876.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 10:15:00 | 1863.80 | 1875.24 | 1876.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 13:15:00 | 1849.20 | 1863.73 | 1870.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 09:15:00 | 1875.50 | 1860.46 | 1866.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 09:15:00 | 1875.50 | 1860.46 | 1866.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1875.50 | 1860.46 | 1866.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:30:00 | 1874.00 | 1860.46 | 1866.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 1863.90 | 1861.14 | 1866.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 11:30:00 | 1854.00 | 1859.54 | 1865.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 12:15:00 | 1880.00 | 1866.43 | 1865.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 12:15:00 | 1880.00 | 1866.43 | 1865.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 13:15:00 | 1909.70 | 1875.09 | 1869.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 15:15:00 | 1891.30 | 1892.70 | 1884.26 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-16 09:15:00 | 1914.10 | 1892.70 | 1884.26 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1967.00 | 1907.56 | 1891.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 11:30:00 | 1983.90 | 1931.08 | 1905.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 13:15:00 | 1972.90 | 1938.88 | 1911.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 15:00:00 | 1971.90 | 1951.91 | 1922.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:30:00 | 1973.00 | 1959.00 | 1931.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 12:15:00 | 2009.81 | 1978.02 | 1947.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-18 14:15:00 | 2000.00 | 2001.56 | 1981.30 | SL hit (close<ema200) qty=0.50 sl=2001.56 alert=retest1 |

### Cycle 144 — SELL (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 12:15:00 | 1973.10 | 1988.03 | 1988.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 14:15:00 | 1961.90 | 1979.93 | 1984.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 1991.50 | 1979.54 | 1983.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 1991.50 | 1979.54 | 1983.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1991.50 | 1979.54 | 1983.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:45:00 | 1989.00 | 1979.54 | 1983.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — BUY (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 10:15:00 | 2029.80 | 1989.59 | 1987.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 14:15:00 | 2039.90 | 2011.94 | 2000.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 14:15:00 | 2019.30 | 2025.08 | 2014.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 14:30:00 | 2025.00 | 2025.08 | 2014.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 2017.00 | 2023.47 | 2014.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 2009.80 | 2023.47 | 2014.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 2024.50 | 2023.67 | 2015.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 10:15:00 | 2047.00 | 2023.67 | 2015.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 13:30:00 | 2027.00 | 2020.29 | 2016.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 14:45:00 | 2026.00 | 2020.64 | 2016.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 09:15:00 | 2033.40 | 2020.07 | 2016.86 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 2059.60 | 2027.97 | 2020.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 14:45:00 | 2062.90 | 2037.69 | 2028.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 09:15:00 | 2064.40 | 2041.13 | 2031.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 2067.20 | 2065.89 | 2061.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 10:45:00 | 2068.10 | 2062.62 | 2060.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 2061.80 | 2062.46 | 2060.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:45:00 | 2062.20 | 2062.46 | 2060.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 2066.40 | 2063.25 | 2061.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 12:45:00 | 2065.60 | 2063.25 | 2061.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 2063.70 | 2064.75 | 2062.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:15:00 | 2042.70 | 2064.75 | 2062.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 2008.80 | 2053.56 | 2057.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 2008.80 | 2053.56 | 2057.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 2004.50 | 2038.06 | 2049.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 14:15:00 | 1983.00 | 1982.89 | 2004.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 14:45:00 | 1977.50 | 1982.89 | 2004.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1964.70 | 1980.04 | 1999.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:15:00 | 1962.20 | 1980.04 | 1999.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:45:00 | 1963.00 | 1972.29 | 1994.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 15:15:00 | 1958.00 | 1962.63 | 1981.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 12:30:00 | 1953.80 | 1972.21 | 1980.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 1961.00 | 1962.61 | 1973.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:30:00 | 1950.00 | 1957.45 | 1969.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 1968.70 | 1955.69 | 1966.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 12:00:00 | 1968.70 | 1955.69 | 1966.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 1961.00 | 1956.75 | 1966.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 12:30:00 | 1965.00 | 1956.75 | 1966.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 1959.30 | 1958.69 | 1965.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 15:15:00 | 1955.00 | 1958.69 | 1965.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:15:00 | 1949.30 | 1958.12 | 1963.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 09:30:00 | 1955.00 | 1938.52 | 1943.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 12:15:00 | 1957.00 | 1941.92 | 1940.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 12:15:00 | 1957.00 | 1941.92 | 1940.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 13:15:00 | 1988.80 | 1964.56 | 1954.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 10:15:00 | 1968.50 | 1971.56 | 1961.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 11:00:00 | 1968.50 | 1971.56 | 1961.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 1961.30 | 1969.51 | 1961.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:00:00 | 1961.30 | 1969.51 | 1961.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 1973.70 | 1970.35 | 1962.57 | EMA400 retest candle locked (from upside) |

### Cycle 148 — SELL (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 11:15:00 | 1949.40 | 1960.86 | 1961.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 13:15:00 | 1933.40 | 1953.45 | 1957.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 14:15:00 | 1940.00 | 1932.13 | 1941.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 14:15:00 | 1940.00 | 1932.13 | 1941.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 1940.00 | 1932.13 | 1941.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:45:00 | 1940.00 | 1932.13 | 1941.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 1945.40 | 1934.79 | 1941.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:15:00 | 1946.00 | 1934.79 | 1941.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1945.10 | 1936.85 | 1942.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 1949.00 | 1936.85 | 1942.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 1957.30 | 1940.94 | 1943.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:45:00 | 1958.60 | 1940.94 | 1943.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 1950.90 | 1943.35 | 1944.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:00:00 | 1950.90 | 1943.35 | 1944.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 14:15:00 | 1950.00 | 1944.68 | 1944.66 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 1919.10 | 1939.69 | 1942.41 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 14:15:00 | 1960.70 | 1943.12 | 1942.30 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 15:15:00 | 1932.50 | 1945.14 | 1945.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 1927.90 | 1938.99 | 1941.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 09:15:00 | 1900.80 | 1896.54 | 1912.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:00:00 | 1900.80 | 1896.54 | 1912.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1877.50 | 1868.61 | 1878.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 15:15:00 | 1862.30 | 1871.89 | 1877.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 1893.80 | 1879.55 | 1878.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 1893.80 | 1879.55 | 1878.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 09:15:00 | 1925.40 | 1893.14 | 1886.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 13:15:00 | 1887.10 | 1901.91 | 1893.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 1887.10 | 1901.91 | 1893.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 1887.10 | 1901.91 | 1893.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 1887.10 | 1901.91 | 1893.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 1891.50 | 1899.83 | 1893.61 | EMA400 retest candle locked (from upside) |

### Cycle 154 — SELL (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 10:15:00 | 1872.80 | 1889.96 | 1890.42 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 10:15:00 | 1914.90 | 1891.45 | 1889.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 14:15:00 | 1920.80 | 1904.76 | 1897.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 10:15:00 | 1905.80 | 1908.20 | 1900.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 11:00:00 | 1905.80 | 1908.20 | 1900.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 1894.40 | 1905.44 | 1900.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 1921.90 | 1902.05 | 1899.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 10:15:00 | 1915.00 | 1903.44 | 1900.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 12:15:00 | 1978.00 | 2001.93 | 2002.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — SELL (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 12:15:00 | 1978.00 | 2001.93 | 2002.88 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 11:15:00 | 2015.10 | 2003.60 | 2002.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 09:15:00 | 2028.40 | 2012.68 | 2007.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 15:15:00 | 2014.00 | 2017.38 | 2012.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 09:15:00 | 2005.80 | 2017.38 | 2012.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 1996.70 | 2013.25 | 2011.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:00:00 | 1996.70 | 2013.25 | 2011.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 1998.30 | 2010.26 | 2009.93 | EMA400 retest candle locked (from upside) |

### Cycle 158 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 11:15:00 | 2004.80 | 2009.17 | 2009.46 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 12:15:00 | 2014.40 | 2008.88 | 2008.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 13:15:00 | 2018.10 | 2010.72 | 2009.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 10:15:00 | 2011.30 | 2014.35 | 2011.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 10:15:00 | 2011.30 | 2014.35 | 2011.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 2011.30 | 2014.35 | 2011.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 2011.30 | 2014.35 | 2011.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 1999.70 | 2011.42 | 2010.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:45:00 | 1997.10 | 2011.42 | 2010.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — SELL (started 2025-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 12:15:00 | 1995.00 | 2008.14 | 2009.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 13:15:00 | 1987.70 | 2004.05 | 2007.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 10:15:00 | 1999.40 | 1995.61 | 2001.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 10:15:00 | 1999.40 | 1995.61 | 2001.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 1999.40 | 1995.61 | 2001.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:45:00 | 2002.10 | 1995.61 | 2001.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 1983.30 | 1993.15 | 1999.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 12:30:00 | 1977.70 | 1990.08 | 1997.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 14:15:00 | 2023.40 | 1960.33 | 1964.22 | SL hit (close>static) qty=1.00 sl=2001.60 alert=retest2 |

### Cycle 161 — BUY (started 2025-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 15:15:00 | 2000.00 | 1968.27 | 1967.47 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 09:15:00 | 1957.50 | 1977.58 | 1978.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-03 10:15:00 | 1946.10 | 1971.29 | 1975.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 12:15:00 | 1972.00 | 1968.97 | 1973.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 12:15:00 | 1972.00 | 1968.97 | 1973.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 1972.00 | 1968.97 | 1973.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 12:45:00 | 1967.00 | 1968.97 | 1973.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 1954.10 | 1965.99 | 1971.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 09:15:00 | 1947.20 | 1960.57 | 1968.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 10:15:00 | 1949.80 | 1958.86 | 1966.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 10:45:00 | 1941.60 | 1953.69 | 1963.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 09:15:00 | 1953.00 | 1945.27 | 1945.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — BUY (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 09:15:00 | 1953.00 | 1945.27 | 1945.01 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 09:15:00 | 1932.40 | 1945.33 | 1945.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 13:15:00 | 1924.70 | 1936.32 | 1941.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 09:15:00 | 1933.80 | 1933.33 | 1938.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-13 09:30:00 | 1937.40 | 1933.33 | 1938.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 1938.10 | 1934.28 | 1938.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:45:00 | 1938.70 | 1934.28 | 1938.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 1933.80 | 1934.19 | 1937.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:45:00 | 1933.10 | 1934.19 | 1937.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 1915.20 | 1929.41 | 1934.16 | EMA400 retest candle locked (from downside) |

### Cycle 165 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 1933.90 | 1927.26 | 1926.99 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 09:15:00 | 1919.50 | 1925.71 | 1926.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-16 11:15:00 | 1916.80 | 1922.88 | 1924.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 09:15:00 | 1923.60 | 1921.03 | 1922.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 1923.60 | 1921.03 | 1922.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 1923.60 | 1921.03 | 1922.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:30:00 | 1925.10 | 1921.03 | 1922.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 1921.70 | 1921.17 | 1922.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 11:15:00 | 1917.80 | 1921.17 | 1922.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 13:15:00 | 1943.50 | 1925.71 | 1924.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — BUY (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 13:15:00 | 1943.50 | 1925.71 | 1924.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 10:15:00 | 1946.10 | 1933.07 | 1929.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 1927.70 | 1938.65 | 1933.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 14:15:00 | 1927.70 | 1938.65 | 1933.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1927.70 | 1938.65 | 1933.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 1927.70 | 1938.65 | 1933.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 1925.00 | 1935.92 | 1933.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 1934.90 | 1935.92 | 1933.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 10:30:00 | 1936.80 | 1935.22 | 1933.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 13:30:00 | 1933.00 | 1935.34 | 1933.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 14:00:00 | 1933.70 | 1935.34 | 1933.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 1939.50 | 1936.17 | 1934.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 15:00:00 | 1939.50 | 1936.17 | 1934.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 1930.60 | 1935.06 | 1934.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 1951.40 | 1935.06 | 1934.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 1950.70 | 1938.19 | 1935.56 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-27 10:15:00 | 1920.50 | 1934.65 | 1934.19 | SL hit (close<static) qty=1.00 sl=1922.00 alert=retest2 |

### Cycle 168 — SELL (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 11:15:00 | 1915.30 | 1930.78 | 1932.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 12:15:00 | 1893.60 | 1912.42 | 1920.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 1915.90 | 1908.73 | 1915.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 1915.90 | 1908.73 | 1915.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1915.90 | 1908.73 | 1915.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:15:00 | 1912.40 | 1908.73 | 1915.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1912.00 | 1909.38 | 1915.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 12:15:00 | 1904.30 | 1909.65 | 1914.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 13:15:00 | 1909.60 | 1910.00 | 1914.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 12:30:00 | 1910.10 | 1904.41 | 1908.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 15:15:00 | 1910.00 | 1905.79 | 1908.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 1910.00 | 1906.63 | 1908.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:15:00 | 1912.20 | 1906.63 | 1908.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1911.60 | 1908.50 | 1909.41 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-31 13:15:00 | 1938.00 | 1915.36 | 1912.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — BUY (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 13:15:00 | 1938.00 | 1915.36 | 1912.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 09:15:00 | 2000.20 | 1936.39 | 1923.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 09:15:00 | 1890.90 | 1951.65 | 1942.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 09:15:00 | 1890.90 | 1951.65 | 1942.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 1890.90 | 1951.65 | 1942.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:00:00 | 1890.90 | 1951.65 | 1942.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 1901.30 | 1941.58 | 1938.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 11:15:00 | 1916.90 | 1941.58 | 1938.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 12:15:00 | 1920.10 | 1933.91 | 1935.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — SELL (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 12:15:00 | 1920.10 | 1933.91 | 1935.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 14:15:00 | 1910.80 | 1927.14 | 1931.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 09:15:00 | 1879.50 | 1873.80 | 1889.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 09:45:00 | 1876.60 | 1873.80 | 1889.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1871.40 | 1878.10 | 1884.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 11:15:00 | 1863.60 | 1876.42 | 1883.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 12:15:00 | 1866.20 | 1874.68 | 1882.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 13:15:00 | 1865.00 | 1873.28 | 1880.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 10:15:00 | 1847.00 | 1865.27 | 1873.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 1848.30 | 1861.87 | 1871.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 14:45:00 | 1837.00 | 1852.30 | 1863.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 14:30:00 | 1838.50 | 1848.14 | 1855.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 10:15:00 | 1843.80 | 1843.42 | 1852.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 11:45:00 | 1842.30 | 1841.41 | 1849.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 1819.70 | 1828.74 | 1837.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:45:00 | 1818.00 | 1828.74 | 1837.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1796.00 | 1816.33 | 1827.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 15:00:00 | 1786.80 | 1802.27 | 1815.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 09:15:00 | 1770.42 | 1791.03 | 1808.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 09:15:00 | 1772.89 | 1791.03 | 1808.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 09:15:00 | 1771.75 | 1791.03 | 1808.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 09:15:00 | 1754.65 | 1791.03 | 1808.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 10:15:00 | 1745.15 | 1781.92 | 1802.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 10:15:00 | 1746.57 | 1781.92 | 1802.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 10:15:00 | 1751.61 | 1781.92 | 1802.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 10:15:00 | 1750.18 | 1781.92 | 1802.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 14:15:00 | 1770.00 | 1764.72 | 1786.27 | SL hit (close>ema200) qty=0.50 sl=1764.72 alert=retest2 |

### Cycle 171 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 1818.60 | 1795.11 | 1794.04 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 1777.30 | 1792.47 | 1794.08 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2025-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 14:15:00 | 1808.50 | 1795.02 | 1794.63 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 12:15:00 | 1786.00 | 1795.16 | 1795.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 10:15:00 | 1781.30 | 1790.82 | 1792.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 12:15:00 | 1789.60 | 1788.47 | 1791.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 12:15:00 | 1789.60 | 1788.47 | 1791.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 1789.60 | 1788.47 | 1791.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:30:00 | 1788.10 | 1788.47 | 1791.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 1790.90 | 1788.96 | 1791.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 13:30:00 | 1790.50 | 1788.96 | 1791.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 1783.60 | 1787.89 | 1790.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:30:00 | 1791.20 | 1787.89 | 1790.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1784.70 | 1786.13 | 1789.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:30:00 | 1781.90 | 1786.13 | 1789.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1804.50 | 1789.81 | 1790.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 1804.50 | 1789.81 | 1790.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 1804.70 | 1792.78 | 1791.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 12:15:00 | 1808.40 | 1795.91 | 1793.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 14:15:00 | 1794.00 | 1796.74 | 1794.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 14:15:00 | 1794.00 | 1796.74 | 1794.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 1794.00 | 1796.74 | 1794.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 15:00:00 | 1794.00 | 1796.74 | 1794.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 1793.20 | 1796.03 | 1794.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:15:00 | 1787.80 | 1796.03 | 1794.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 1782.30 | 1793.29 | 1793.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 10:00:00 | 1782.30 | 1793.29 | 1793.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 1781.00 | 1790.83 | 1792.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 11:15:00 | 1774.40 | 1787.54 | 1790.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 14:15:00 | 1760.20 | 1754.54 | 1767.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-28 15:00:00 | 1760.20 | 1754.54 | 1767.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1746.00 | 1753.23 | 1764.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 10:30:00 | 1740.70 | 1752.78 | 1763.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 12:00:00 | 1741.60 | 1750.54 | 1761.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 10:15:00 | 1775.00 | 1750.67 | 1755.39 | SL hit (close>static) qty=1.00 sl=1768.20 alert=retest2 |

### Cycle 177 — BUY (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 15:15:00 | 1770.10 | 1758.38 | 1757.70 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 10:15:00 | 1743.00 | 1755.58 | 1757.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 14:15:00 | 1739.20 | 1747.03 | 1752.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 10:15:00 | 1747.90 | 1741.47 | 1747.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 10:15:00 | 1747.90 | 1741.47 | 1747.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 1747.90 | 1741.47 | 1747.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:00:00 | 1747.90 | 1741.47 | 1747.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 1749.70 | 1743.11 | 1748.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:00:00 | 1749.70 | 1743.11 | 1748.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 1749.90 | 1744.47 | 1748.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 1736.90 | 1747.04 | 1748.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:15:00 | 1650.06 | 1657.83 | 1665.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-18 11:15:00 | 1663.00 | 1657.77 | 1664.44 | SL hit (close>ema200) qty=0.50 sl=1657.77 alert=retest2 |

### Cycle 179 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 1684.20 | 1666.89 | 1666.53 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 11:15:00 | 1668.40 | 1677.78 | 1677.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 12:15:00 | 1663.20 | 1674.86 | 1676.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 10:15:00 | 1662.30 | 1661.48 | 1668.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 11:00:00 | 1662.30 | 1661.48 | 1668.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 1662.90 | 1661.73 | 1666.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 13:30:00 | 1663.20 | 1661.73 | 1666.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 1658.00 | 1660.98 | 1665.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 1653.30 | 1660.85 | 1665.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 14:15:00 | 1669.20 | 1651.07 | 1653.15 | SL hit (close>static) qty=1.00 sl=1666.00 alert=retest2 |

### Cycle 181 — BUY (started 2025-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 15:15:00 | 1672.00 | 1655.26 | 1654.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 10:15:00 | 1689.90 | 1666.75 | 1660.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 1696.60 | 1701.40 | 1684.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 10:00:00 | 1696.60 | 1701.40 | 1684.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 1687.20 | 1702.28 | 1693.32 | EMA400 retest candle locked (from upside) |

### Cycle 182 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 1688.50 | 1696.73 | 1697.05 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2026-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 14:15:00 | 1701.90 | 1697.77 | 1697.49 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 1693.50 | 1696.65 | 1697.04 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 12:15:00 | 1706.40 | 1698.08 | 1697.59 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2026-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 14:15:00 | 1692.70 | 1697.15 | 1697.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 15:15:00 | 1689.00 | 1695.52 | 1696.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 1699.00 | 1696.22 | 1696.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 1699.00 | 1696.22 | 1696.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 1699.00 | 1696.22 | 1696.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 1699.00 | 1696.22 | 1696.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 1691.90 | 1695.35 | 1696.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 11:15:00 | 1689.40 | 1695.35 | 1696.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 12:15:00 | 1688.50 | 1694.32 | 1695.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 14:15:00 | 1707.90 | 1697.80 | 1697.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — BUY (started 2026-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 14:15:00 | 1707.90 | 1697.80 | 1697.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 15:15:00 | 1710.00 | 1700.24 | 1698.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 11:15:00 | 1707.00 | 1708.07 | 1702.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 11:15:00 | 1707.00 | 1708.07 | 1702.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 1707.00 | 1708.07 | 1702.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 1707.00 | 1708.07 | 1702.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 1708.00 | 1708.06 | 1703.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:30:00 | 1706.60 | 1708.06 | 1703.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 1702.50 | 1706.95 | 1703.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 14:00:00 | 1702.50 | 1706.95 | 1703.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 1701.20 | 1705.80 | 1703.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 15:00:00 | 1701.20 | 1705.80 | 1703.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 1690.00 | 1702.64 | 1701.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:15:00 | 1674.90 | 1702.64 | 1701.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 1679.40 | 1697.99 | 1699.91 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2026-01-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 14:15:00 | 1703.10 | 1693.00 | 1692.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 15:15:00 | 1710.10 | 1696.42 | 1694.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 10:15:00 | 1731.90 | 1734.63 | 1723.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 10:30:00 | 1734.20 | 1734.63 | 1723.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 1722.50 | 1731.11 | 1723.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:00:00 | 1722.50 | 1731.11 | 1723.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 1715.40 | 1727.97 | 1722.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:00:00 | 1715.40 | 1727.97 | 1722.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 1707.00 | 1723.77 | 1721.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 15:00:00 | 1707.00 | 1723.77 | 1721.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 1701.00 | 1719.22 | 1719.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 13:15:00 | 1683.80 | 1704.52 | 1711.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 11:15:00 | 1685.40 | 1676.25 | 1685.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 11:15:00 | 1685.40 | 1676.25 | 1685.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 11:15:00 | 1685.40 | 1676.25 | 1685.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 12:00:00 | 1685.40 | 1676.25 | 1685.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 1688.90 | 1678.78 | 1686.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 1688.90 | 1678.78 | 1686.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 1682.10 | 1679.44 | 1685.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 14:15:00 | 1680.10 | 1679.44 | 1685.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 1707.00 | 1687.50 | 1688.15 | SL hit (close>static) qty=1.00 sl=1692.40 alert=retest2 |

### Cycle 191 — BUY (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 10:15:00 | 1698.80 | 1689.76 | 1689.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 1723.70 | 1701.45 | 1695.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 1693.80 | 1701.91 | 1696.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 1693.80 | 1701.91 | 1696.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 1693.80 | 1701.91 | 1696.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:45:00 | 1680.60 | 1701.91 | 1696.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 1691.20 | 1699.77 | 1696.14 | EMA400 retest candle locked (from upside) |

### Cycle 192 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 1683.40 | 1693.96 | 1693.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 12:15:00 | 1669.60 | 1683.55 | 1688.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 1682.00 | 1680.37 | 1685.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 09:15:00 | 1669.10 | 1680.37 | 1685.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1685.40 | 1681.37 | 1685.31 | EMA400 retest candle locked (from downside) |

### Cycle 193 — BUY (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 09:15:00 | 1778.00 | 1703.41 | 1693.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 1863.10 | 1795.70 | 1754.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 15:15:00 | 1830.00 | 1830.59 | 1793.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 09:15:00 | 1808.90 | 1830.59 | 1793.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1804.30 | 1829.65 | 1805.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1818.70 | 1829.65 | 1805.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 1850.00 | 1833.72 | 1809.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 1827.70 | 1833.72 | 1809.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 1860.00 | 1838.03 | 1817.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 13:00:00 | 1865.90 | 1845.69 | 1826.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 12:30:00 | 1868.50 | 1878.70 | 1870.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 13:15:00 | 1849.80 | 1869.50 | 1870.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — SELL (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 13:15:00 | 1849.80 | 1869.50 | 1870.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 1834.50 | 1856.59 | 1864.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 1848.80 | 1846.65 | 1855.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 14:15:00 | 1848.80 | 1846.65 | 1855.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 1848.80 | 1846.65 | 1855.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 1848.80 | 1846.65 | 1855.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1847.70 | 1846.32 | 1853.68 | EMA400 retest candle locked (from downside) |

### Cycle 195 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 1879.70 | 1861.13 | 1859.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 14:15:00 | 1886.30 | 1869.02 | 1863.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 10:15:00 | 1865.40 | 1873.05 | 1867.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 10:15:00 | 1865.40 | 1873.05 | 1867.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 1865.40 | 1873.05 | 1867.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:00:00 | 1865.40 | 1873.05 | 1867.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 1871.60 | 1872.76 | 1867.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 14:15:00 | 1880.90 | 1873.64 | 1868.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 15:00:00 | 1887.20 | 1876.35 | 1870.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 09:15:00 | 1858.20 | 1874.38 | 1870.74 | SL hit (close<static) qty=1.00 sl=1863.50 alert=retest2 |

### Cycle 196 — SELL (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 14:15:00 | 1867.70 | 1869.23 | 1869.34 | EMA200 below EMA400 |

### Cycle 197 — BUY (started 2026-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 15:15:00 | 1872.00 | 1869.79 | 1869.58 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 1849.70 | 1865.77 | 1867.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 13:15:00 | 1815.00 | 1841.19 | 1854.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 11:15:00 | 1802.00 | 1794.70 | 1811.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 12:00:00 | 1802.00 | 1794.70 | 1811.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 1808.20 | 1797.91 | 1808.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 1808.20 | 1797.91 | 1808.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 1815.00 | 1801.33 | 1809.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 1811.90 | 1801.33 | 1809.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 1812.00 | 1803.47 | 1809.65 | EMA400 retest candle locked (from downside) |

### Cycle 199 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 1826.90 | 1813.84 | 1813.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 13:15:00 | 1837.10 | 1818.49 | 1815.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 1842.80 | 1850.10 | 1842.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 14:15:00 | 1842.80 | 1850.10 | 1842.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 1842.80 | 1850.10 | 1842.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:30:00 | 1849.80 | 1850.10 | 1842.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 1825.10 | 1845.10 | 1840.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 1816.60 | 1845.10 | 1840.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 1823.60 | 1840.80 | 1839.41 | EMA400 retest candle locked (from upside) |

### Cycle 200 — SELL (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 10:15:00 | 1822.90 | 1837.22 | 1837.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 11:15:00 | 1820.20 | 1833.82 | 1836.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 15:15:00 | 1810.00 | 1805.18 | 1811.44 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-25 09:15:00 | 1795.30 | 1805.18 | 1811.44 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 1809.00 | 1805.95 | 1811.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:45:00 | 1808.70 | 1805.95 | 1811.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 1807.00 | 1806.16 | 1810.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:45:00 | 1807.30 | 1806.16 | 1810.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 1820.20 | 1808.97 | 1811.68 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-25 11:15:00 | 1820.20 | 1808.97 | 1811.68 | SL hit (close>ema400) qty=1.00 sl=1811.68 alert=retest1 |

### Cycle 201 — BUY (started 2026-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 14:15:00 | 1846.50 | 1818.90 | 1815.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 10:15:00 | 1863.90 | 1836.46 | 1825.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 15:15:00 | 1846.10 | 1847.90 | 1835.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 09:15:00 | 1828.50 | 1847.90 | 1835.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1836.00 | 1845.52 | 1835.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:30:00 | 1838.80 | 1845.52 | 1835.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 1836.20 | 1843.66 | 1835.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:30:00 | 1835.10 | 1843.66 | 1835.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 1835.00 | 1841.93 | 1835.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:30:00 | 1835.10 | 1841.93 | 1835.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 1847.50 | 1843.04 | 1836.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:45:00 | 1842.30 | 1843.04 | 1836.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 1834.90 | 1841.41 | 1836.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:30:00 | 1834.90 | 1841.41 | 1836.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 1815.70 | 1836.27 | 1834.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 1815.70 | 1836.27 | 1834.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — SELL (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 15:15:00 | 1822.00 | 1833.42 | 1833.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 1783.60 | 1823.45 | 1829.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 1795.00 | 1792.51 | 1808.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 15:00:00 | 1795.00 | 1792.51 | 1808.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 1692.90 | 1680.01 | 1690.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 10:45:00 | 1681.00 | 1680.46 | 1689.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 13:00:00 | 1681.80 | 1681.09 | 1688.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 15:15:00 | 1673.70 | 1684.01 | 1688.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 11:15:00 | 1679.80 | 1686.72 | 1688.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 1677.50 | 1684.88 | 1687.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 14:45:00 | 1668.70 | 1678.35 | 1683.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 15:15:00 | 1662.50 | 1678.35 | 1683.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 1638.20 | 1662.23 | 1669.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:15:00 | 1596.95 | 1614.28 | 1635.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:15:00 | 1597.71 | 1614.28 | 1635.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:15:00 | 1590.01 | 1614.28 | 1635.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:15:00 | 1595.81 | 1614.28 | 1635.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:15:00 | 1585.26 | 1614.28 | 1635.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 10:15:00 | 1579.38 | 1614.28 | 1635.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 14:15:00 | 1607.80 | 1604.80 | 1623.13 | SL hit (close>ema200) qty=0.50 sl=1604.80 alert=retest2 |

### Cycle 203 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 1647.90 | 1628.50 | 1626.52 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 1620.70 | 1633.15 | 1633.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 1607.00 | 1627.92 | 1630.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 12:15:00 | 1607.00 | 1606.72 | 1616.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 13:00:00 | 1607.00 | 1606.72 | 1616.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 1615.00 | 1608.38 | 1616.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 14:00:00 | 1615.00 | 1608.38 | 1616.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 1627.20 | 1612.14 | 1617.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 1627.20 | 1612.14 | 1617.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 1627.60 | 1615.23 | 1618.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 1619.10 | 1615.23 | 1618.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 10:15:00 | 1644.80 | 1621.42 | 1618.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — BUY (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 10:15:00 | 1644.80 | 1621.42 | 1618.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 11:15:00 | 1653.80 | 1627.89 | 1621.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1681.60 | 1694.57 | 1673.62 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 11:30:00 | 1707.00 | 1697.49 | 1678.67 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 15:00:00 | 1723.00 | 1704.05 | 1686.46 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 1673.30 | 1698.37 | 1686.96 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 1673.30 | 1698.37 | 1686.96 | SL hit (close<ema400) qty=1.00 sl=1686.96 alert=retest1 |

### Cycle 206 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 1668.40 | 1698.19 | 1698.86 | EMA200 below EMA400 |

### Cycle 207 — BUY (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 10:15:00 | 1723.80 | 1699.43 | 1698.38 | EMA200 above EMA400 |

### Cycle 208 — SELL (started 2026-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 09:15:00 | 1682.10 | 1697.32 | 1698.18 | EMA200 below EMA400 |

### Cycle 209 — BUY (started 2026-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 11:15:00 | 1707.70 | 1698.67 | 1698.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 15:15:00 | 1720.00 | 1707.96 | 1703.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 11:15:00 | 1741.60 | 1745.30 | 1735.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 11:45:00 | 1743.00 | 1745.30 | 1735.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 1744.70 | 1746.47 | 1738.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:30:00 | 1740.00 | 1746.47 | 1738.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 15:15:00 | 1749.80 | 1747.14 | 1739.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:15:00 | 1731.90 | 1747.14 | 1739.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1730.40 | 1743.79 | 1738.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1740.00 | 1743.79 | 1738.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 13:15:00 | 1729.20 | 1736.53 | 1736.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 210 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 1729.20 | 1736.53 | 1736.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 14:15:00 | 1718.20 | 1732.87 | 1735.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 15:15:00 | 1733.90 | 1733.07 | 1734.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 15:15:00 | 1733.90 | 1733.07 | 1734.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 1733.90 | 1733.07 | 1734.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:15:00 | 1757.30 | 1733.07 | 1734.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 1765.60 | 1739.58 | 1737.73 | EMA200 above EMA400 |

### Cycle 212 — SELL (started 2026-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 12:15:00 | 1736.90 | 1742.41 | 1742.78 | EMA200 below EMA400 |

### Cycle 213 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 11:15:00 | 1771.30 | 1747.85 | 1744.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 12:15:00 | 1810.00 | 1760.28 | 1750.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 09:15:00 | 1784.70 | 1785.37 | 1775.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 1784.70 | 1785.37 | 1775.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 1784.70 | 1785.37 | 1775.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 09:30:00 | 1798.60 | 1781.04 | 1776.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 10:15:00 | 1795.00 | 1781.04 | 1776.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 12:15:00 | 1814.70 | 1784.40 | 1778.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 09:30:00 | 1816.00 | 1799.12 | 1788.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 1797.90 | 1798.00 | 1792.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:30:00 | 1790.70 | 1798.00 | 1792.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 1795.00 | 1797.40 | 1792.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:15:00 | 1771.30 | 1797.40 | 1792.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 1771.30 | 1792.18 | 1790.58 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-24 10:15:00 | 1752.60 | 1784.26 | 1787.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 214 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 1752.60 | 1784.26 | 1787.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 11:15:00 | 1734.80 | 1774.37 | 1782.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1753.00 | 1740.26 | 1759.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1753.00 | 1740.26 | 1759.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1753.00 | 1740.26 | 1759.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1761.00 | 1740.26 | 1759.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1751.60 | 1742.53 | 1758.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:30:00 | 1752.30 | 1742.53 | 1758.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 1758.00 | 1748.91 | 1757.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:00:00 | 1758.00 | 1748.91 | 1757.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 1776.10 | 1754.35 | 1759.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 15:00:00 | 1776.10 | 1754.35 | 1759.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 1775.30 | 1758.54 | 1760.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 09:15:00 | 1765.10 | 1758.54 | 1760.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 10:00:00 | 1764.10 | 1759.65 | 1761.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 10:30:00 | 1763.90 | 1760.38 | 1761.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 09:45:00 | 1763.70 | 1757.87 | 1759.00 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 1755.20 | 1758.11 | 1758.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:00:00 | 1748.10 | 1756.11 | 1757.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:45:00 | 1746.30 | 1752.73 | 1756.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 1742.00 | 1753.20 | 1756.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 14:15:00 | 1748.30 | 1747.42 | 1751.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 1748.40 | 1747.62 | 1750.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:45:00 | 1755.10 | 1747.62 | 1750.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 1742.80 | 1746.65 | 1750.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 1772.70 | 1746.65 | 1750.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1793.60 | 1756.04 | 1754.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 215 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 1793.60 | 1756.04 | 1754.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 10:15:00 | 1813.20 | 1767.48 | 1759.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 10:15:00 | 1795.10 | 1801.45 | 1784.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 10:30:00 | 1790.90 | 1801.45 | 1784.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 1787.40 | 1798.64 | 1785.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:45:00 | 1785.00 | 1798.64 | 1785.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 1769.70 | 1792.85 | 1783.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:00:00 | 1769.70 | 1792.85 | 1783.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 1771.10 | 1788.50 | 1782.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 1785.00 | 1783.53 | 1781.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-16 12:00:00 | 1361.20 | 2023-05-19 09:15:00 | 1225.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-13 09:15:00 | 984.80 | 2023-06-21 11:15:00 | 1005.00 | STOP_HIT | 1.00 | 2.05% |
| BUY | retest2 | 2023-07-05 09:15:00 | 1060.45 | 2023-07-07 11:15:00 | 1056.70 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2023-07-25 09:15:00 | 1256.15 | 2023-07-26 10:15:00 | 1226.80 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2023-08-07 12:45:00 | 1340.90 | 2023-08-08 09:15:00 | 1474.99 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-08-28 15:00:00 | 1570.95 | 2023-08-29 09:15:00 | 1604.60 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2023-09-05 09:15:00 | 1748.00 | 2023-09-05 12:15:00 | 1701.10 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2023-09-05 12:15:00 | 1718.95 | 2023-09-05 12:15:00 | 1701.10 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2023-09-21 13:15:00 | 1695.90 | 2023-09-25 14:15:00 | 1657.25 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2023-09-21 13:45:00 | 1695.05 | 2023-09-25 14:15:00 | 1657.25 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2023-09-21 14:30:00 | 1700.00 | 2023-09-25 14:15:00 | 1657.25 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2023-09-22 09:15:00 | 1695.75 | 2023-09-25 14:15:00 | 1657.25 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2023-09-26 12:30:00 | 1658.50 | 2023-09-29 10:15:00 | 1674.10 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2023-10-03 09:15:00 | 1698.50 | 2023-10-05 09:15:00 | 1674.20 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2023-10-04 12:15:00 | 1676.95 | 2023-10-05 09:15:00 | 1674.20 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2023-10-04 14:00:00 | 1678.30 | 2023-10-05 09:15:00 | 1674.20 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2023-10-05 09:15:00 | 1678.65 | 2023-10-05 09:15:00 | 1674.20 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2023-10-11 12:45:00 | 1636.00 | 2023-10-18 14:15:00 | 1609.00 | STOP_HIT | 1.00 | 1.65% |
| SELL | retest2 | 2023-10-12 10:45:00 | 1637.25 | 2023-10-18 14:15:00 | 1609.00 | STOP_HIT | 1.00 | 1.73% |
| SELL | retest2 | 2023-10-12 13:45:00 | 1635.10 | 2023-10-18 14:15:00 | 1609.00 | STOP_HIT | 1.00 | 1.60% |
| SELL | retest2 | 2023-10-27 10:15:00 | 1541.45 | 2023-10-27 11:15:00 | 1554.95 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2023-11-10 15:00:00 | 1630.00 | 2023-11-15 11:15:00 | 1609.45 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2023-11-13 10:00:00 | 1626.95 | 2023-11-15 11:15:00 | 1609.45 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2023-11-13 10:30:00 | 1626.00 | 2023-11-15 11:15:00 | 1609.45 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2023-11-13 11:00:00 | 1625.90 | 2023-11-15 11:15:00 | 1609.45 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2023-11-21 09:15:00 | 1667.45 | 2023-12-01 09:15:00 | 1834.20 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-12-11 09:15:00 | 1762.05 | 2023-12-11 11:15:00 | 1784.95 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2023-12-14 13:00:00 | 1845.00 | 2023-12-20 13:15:00 | 1837.90 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2023-12-14 14:45:00 | 1830.00 | 2023-12-21 09:15:00 | 1817.80 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2023-12-14 15:15:00 | 1831.30 | 2023-12-21 09:15:00 | 1817.80 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2023-12-18 10:30:00 | 1837.60 | 2023-12-21 09:15:00 | 1817.80 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2023-12-20 11:00:00 | 1882.40 | 2023-12-21 09:15:00 | 1817.80 | STOP_HIT | 1.00 | -3.43% |
| BUY | retest2 | 2024-01-01 09:15:00 | 1946.40 | 2024-01-01 14:15:00 | 1902.00 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2024-01-05 09:15:00 | 2018.25 | 2024-01-05 15:15:00 | 1990.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-01-10 10:45:00 | 1904.15 | 2024-01-11 14:15:00 | 1966.00 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2024-01-11 11:15:00 | 1915.30 | 2024-01-11 14:15:00 | 1966.00 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2024-01-16 09:30:00 | 1963.95 | 2024-01-16 11:15:00 | 1932.65 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-01-17 12:00:00 | 1918.10 | 2024-01-17 15:15:00 | 1960.75 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2024-01-18 09:15:00 | 1917.40 | 2024-01-18 13:15:00 | 1951.05 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-01-20 09:15:00 | 1977.15 | 2024-01-20 12:15:00 | 1949.45 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-02-12 12:45:00 | 1994.15 | 2024-02-15 11:15:00 | 1894.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-12 14:45:00 | 1993.00 | 2024-02-15 11:15:00 | 1893.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-02-13 09:15:00 | 1962.25 | 2024-02-15 11:15:00 | 1893.97 | PARTIAL | 0.50 | 3.48% |
| SELL | retest2 | 2024-02-13 11:45:00 | 1993.65 | 2024-02-15 11:15:00 | 1893.21 | PARTIAL | 0.50 | 5.04% |
| SELL | retest2 | 2024-02-12 12:45:00 | 1994.15 | 2024-02-15 12:15:00 | 1970.50 | STOP_HIT | 0.50 | 1.19% |
| SELL | retest2 | 2024-02-12 14:45:00 | 1993.00 | 2024-02-15 12:15:00 | 1970.50 | STOP_HIT | 0.50 | 1.13% |
| SELL | retest2 | 2024-02-13 09:15:00 | 1962.25 | 2024-02-15 12:15:00 | 1970.50 | STOP_HIT | 0.50 | -0.42% |
| SELL | retest2 | 2024-02-13 11:45:00 | 1993.65 | 2024-02-15 12:15:00 | 1970.50 | STOP_HIT | 0.50 | 1.16% |
| SELL | retest2 | 2024-02-13 14:15:00 | 1992.85 | 2024-02-16 11:15:00 | 2037.90 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2024-03-06 09:15:00 | 1741.00 | 2024-03-07 11:15:00 | 1768.95 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2024-03-07 12:45:00 | 1740.65 | 2024-03-07 15:15:00 | 1767.50 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-03-11 09:15:00 | 1737.55 | 2024-03-14 13:15:00 | 1744.95 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2024-03-11 09:45:00 | 1751.80 | 2024-03-14 13:15:00 | 1744.95 | STOP_HIT | 1.00 | 0.39% |
| SELL | retest2 | 2024-03-12 09:15:00 | 1712.15 | 2024-03-14 13:15:00 | 1744.95 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-03-14 11:45:00 | 1741.90 | 2024-03-14 13:15:00 | 1744.95 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2024-03-14 12:30:00 | 1743.15 | 2024-03-14 13:15:00 | 1744.95 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2024-03-26 12:15:00 | 1792.70 | 2024-04-02 12:15:00 | 1793.30 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2024-03-27 09:15:00 | 1804.95 | 2024-04-02 12:15:00 | 1793.30 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-04-16 12:30:00 | 1741.25 | 2024-04-18 11:15:00 | 1770.00 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-04-22 10:30:00 | 1772.25 | 2024-04-24 15:15:00 | 1756.00 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-04-22 12:00:00 | 1772.60 | 2024-04-24 15:15:00 | 1756.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-04-22 15:00:00 | 1775.80 | 2024-04-24 15:15:00 | 1756.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-04-30 12:00:00 | 1722.45 | 2024-05-07 12:15:00 | 1715.05 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2024-05-02 12:15:00 | 1714.95 | 2024-05-07 12:15:00 | 1715.05 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2024-05-03 09:15:00 | 1717.25 | 2024-05-07 12:15:00 | 1715.05 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2024-05-08 15:15:00 | 1720.00 | 2024-05-09 11:15:00 | 1700.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-05-17 12:30:00 | 1775.80 | 2024-05-23 09:15:00 | 1953.38 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-04 12:00:00 | 1754.70 | 2024-06-04 12:15:00 | 1666.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 12:00:00 | 1754.70 | 2024-06-05 09:15:00 | 1836.35 | STOP_HIT | 0.50 | -4.65% |
| BUY | retest2 | 2024-06-10 09:15:00 | 1877.00 | 2024-06-13 11:15:00 | 1854.15 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-06-10 10:00:00 | 1875.25 | 2024-06-13 11:15:00 | 1854.15 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-06-18 13:45:00 | 1853.00 | 2024-06-25 14:15:00 | 1824.00 | STOP_HIT | 1.00 | 1.57% |
| BUY | retest2 | 2024-07-11 09:15:00 | 2025.40 | 2024-07-16 09:15:00 | 1988.85 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2024-07-25 10:30:00 | 2015.45 | 2024-08-05 11:15:00 | 2075.25 | STOP_HIT | 1.00 | 2.97% |
| BUY | retest2 | 2024-07-25 11:00:00 | 2017.05 | 2024-08-05 11:15:00 | 2075.25 | STOP_HIT | 1.00 | 2.89% |
| BUY | retest2 | 2024-07-26 09:45:00 | 2029.50 | 2024-08-05 11:15:00 | 2075.25 | STOP_HIT | 1.00 | 2.25% |
| BUY | retest2 | 2024-08-06 09:15:00 | 2174.55 | 2024-08-07 09:15:00 | 2014.10 | STOP_HIT | 1.00 | -7.38% |
| SELL | retest2 | 2024-08-27 13:15:00 | 1874.10 | 2024-09-03 10:15:00 | 1912.05 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2024-08-28 09:15:00 | 1873.00 | 2024-09-03 10:15:00 | 1912.05 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2024-09-09 09:45:00 | 1917.40 | 2024-09-11 13:15:00 | 1890.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-09-10 09:15:00 | 1928.00 | 2024-09-11 13:15:00 | 1890.00 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2024-09-16 14:15:00 | 1865.00 | 2024-09-23 12:15:00 | 1845.70 | STOP_HIT | 1.00 | 1.03% |
| SELL | retest2 | 2024-09-16 15:00:00 | 1865.05 | 2024-09-23 12:15:00 | 1845.70 | STOP_HIT | 1.00 | 1.04% |
| SELL | retest2 | 2024-10-01 11:15:00 | 1785.60 | 2024-10-07 10:15:00 | 1696.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 12:30:00 | 1785.00 | 2024-10-07 10:15:00 | 1695.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 09:45:00 | 1782.00 | 2024-10-07 10:15:00 | 1692.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 11:15:00 | 1785.60 | 2024-10-08 10:15:00 | 1715.55 | STOP_HIT | 0.50 | 3.92% |
| SELL | retest2 | 2024-10-01 12:30:00 | 1785.00 | 2024-10-08 10:15:00 | 1715.55 | STOP_HIT | 0.50 | 3.89% |
| SELL | retest2 | 2024-10-03 09:45:00 | 1782.00 | 2024-10-08 10:15:00 | 1715.55 | STOP_HIT | 0.50 | 3.73% |
| SELL | retest2 | 2024-10-04 09:15:00 | 1754.00 | 2024-10-15 10:15:00 | 1666.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-04 14:45:00 | 1747.45 | 2024-10-15 11:15:00 | 1660.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-04 15:15:00 | 1744.00 | 2024-10-15 11:15:00 | 1656.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-07 09:30:00 | 1736.10 | 2024-10-15 12:15:00 | 1649.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-04 09:15:00 | 1754.00 | 2024-10-16 11:15:00 | 1664.95 | STOP_HIT | 0.50 | 5.08% |
| SELL | retest2 | 2024-10-04 14:45:00 | 1747.45 | 2024-10-16 11:15:00 | 1664.95 | STOP_HIT | 0.50 | 4.72% |
| SELL | retest2 | 2024-10-04 15:15:00 | 1744.00 | 2024-10-16 11:15:00 | 1664.95 | STOP_HIT | 0.50 | 4.53% |
| SELL | retest2 | 2024-10-07 09:30:00 | 1736.10 | 2024-10-16 11:15:00 | 1664.95 | STOP_HIT | 0.50 | 4.10% |
| SELL | retest2 | 2024-10-24 14:45:00 | 1626.80 | 2024-10-24 15:15:00 | 1650.80 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-10-25 09:30:00 | 1616.20 | 2024-10-29 14:15:00 | 1625.95 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-10-28 13:15:00 | 1625.35 | 2024-10-29 14:15:00 | 1625.95 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2024-10-28 15:00:00 | 1624.20 | 2024-10-29 14:15:00 | 1625.95 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2024-10-31 12:45:00 | 1653.00 | 2024-11-04 09:15:00 | 1620.30 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-10-31 15:00:00 | 1652.05 | 2024-11-04 09:15:00 | 1620.30 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2024-11-01 18:15:00 | 1653.05 | 2024-11-04 09:15:00 | 1620.30 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-11-11 09:15:00 | 1794.15 | 2024-11-12 13:15:00 | 1781.95 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-11-11 09:45:00 | 1794.10 | 2024-11-12 13:15:00 | 1781.95 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-11-12 13:00:00 | 1792.30 | 2024-11-12 13:15:00 | 1781.95 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-11-18 09:30:00 | 1730.85 | 2024-11-18 12:15:00 | 1764.75 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-11-28 14:30:00 | 1721.50 | 2024-12-02 09:15:00 | 1749.50 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-11-29 10:00:00 | 1718.90 | 2024-12-02 09:15:00 | 1749.50 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-11-29 11:30:00 | 1720.10 | 2024-12-02 09:15:00 | 1749.50 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2024-12-05 10:45:00 | 1809.50 | 2024-12-09 11:15:00 | 1800.00 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-12-05 11:30:00 | 1814.95 | 2024-12-09 11:15:00 | 1800.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-12-09 09:15:00 | 1814.25 | 2024-12-09 11:15:00 | 1800.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-12-11 15:15:00 | 1770.55 | 2024-12-12 09:15:00 | 1806.50 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-12-17 12:15:00 | 1735.00 | 2024-12-18 09:15:00 | 1770.00 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2024-12-17 14:00:00 | 1743.95 | 2024-12-18 09:15:00 | 1770.00 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-01-01 09:15:00 | 1768.15 | 2025-01-01 11:15:00 | 1820.50 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-01-20 15:15:00 | 1670.00 | 2025-01-24 10:15:00 | 1586.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 12:15:00 | 1669.70 | 2025-01-24 10:15:00 | 1586.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-20 15:15:00 | 1670.00 | 2025-01-27 09:15:00 | 1503.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-21 12:15:00 | 1669.70 | 2025-01-27 09:15:00 | 1502.73 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-12 15:15:00 | 1464.90 | 2025-02-13 09:15:00 | 1502.00 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2025-02-19 09:15:00 | 1459.50 | 2025-02-19 11:15:00 | 1477.85 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest1 | 2025-02-21 09:15:00 | 1539.65 | 2025-02-24 09:15:00 | 1492.35 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest1 | 2025-02-21 14:30:00 | 1529.25 | 2025-02-24 09:15:00 | 1492.35 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest1 | 2025-02-21 15:00:00 | 1528.40 | 2025-02-24 09:15:00 | 1492.35 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-02-25 09:15:00 | 1611.00 | 2025-02-28 13:15:00 | 1533.00 | STOP_HIT | 1.00 | -4.84% |
| BUY | retest2 | 2025-02-25 10:15:00 | 1598.25 | 2025-02-28 13:15:00 | 1533.00 | STOP_HIT | 1.00 | -4.08% |
| BUY | retest2 | 2025-02-25 12:00:00 | 1603.60 | 2025-02-28 13:15:00 | 1533.00 | STOP_HIT | 1.00 | -4.40% |
| BUY | retest2 | 2025-02-25 13:45:00 | 1597.70 | 2025-02-28 13:15:00 | 1533.00 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest2 | 2025-03-11 09:30:00 | 1667.20 | 2025-03-11 10:15:00 | 1628.30 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-03-17 15:15:00 | 1544.40 | 2025-03-19 13:15:00 | 1580.70 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-03-20 14:30:00 | 1582.80 | 2025-03-27 12:15:00 | 1611.35 | STOP_HIT | 1.00 | 1.80% |
| SELL | retest2 | 2025-03-28 12:15:00 | 1602.00 | 2025-04-04 09:15:00 | 1521.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 09:15:00 | 1597.80 | 2025-04-04 09:15:00 | 1517.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 09:45:00 | 1601.65 | 2025-04-04 09:15:00 | 1521.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 10:15:00 | 1588.00 | 2025-04-04 09:15:00 | 1508.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 12:15:00 | 1602.00 | 2025-04-07 09:15:00 | 1441.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-01 09:15:00 | 1597.80 | 2025-04-07 09:15:00 | 1438.02 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-01 09:45:00 | 1601.65 | 2025-04-07 09:15:00 | 1441.49 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-03 10:15:00 | 1588.00 | 2025-04-07 09:15:00 | 1429.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-04 10:15:00 | 1470.00 | 2025-04-07 09:15:00 | 1323.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-21 10:15:00 | 1459.40 | 2025-04-25 10:15:00 | 1438.40 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-04-22 13:00:00 | 1455.00 | 2025-04-25 10:15:00 | 1438.40 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-04-23 09:15:00 | 1458.80 | 2025-04-25 10:15:00 | 1438.40 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-04-23 10:15:00 | 1468.30 | 2025-04-25 10:15:00 | 1438.40 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-04-30 11:45:00 | 1413.50 | 2025-05-05 11:15:00 | 1423.60 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-04-30 13:15:00 | 1411.30 | 2025-05-05 11:15:00 | 1423.60 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-04-30 14:15:00 | 1412.50 | 2025-05-05 13:15:00 | 1419.90 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-05-02 11:00:00 | 1413.20 | 2025-05-05 13:15:00 | 1419.90 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-05-02 12:15:00 | 1404.90 | 2025-05-05 13:15:00 | 1419.90 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-05-02 14:15:00 | 1402.30 | 2025-05-05 13:15:00 | 1419.90 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-05-14 14:30:00 | 1456.70 | 2025-05-26 10:15:00 | 1524.60 | STOP_HIT | 1.00 | 4.66% |
| BUY | retest2 | 2025-06-04 09:30:00 | 1614.30 | 2025-06-16 14:15:00 | 1775.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-30 09:15:00 | 1799.70 | 2025-07-10 10:15:00 | 1863.80 | STOP_HIT | 1.00 | 3.56% |
| SELL | retest2 | 2025-07-11 11:30:00 | 1854.00 | 2025-07-14 12:15:00 | 1880.00 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest1 | 2025-07-16 09:15:00 | 1914.10 | 2025-07-17 12:15:00 | 2009.81 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-07-16 09:15:00 | 1914.10 | 2025-07-18 14:15:00 | 2000.00 | STOP_HIT | 0.50 | 4.49% |
| BUY | retest2 | 2025-07-16 11:30:00 | 1983.90 | 2025-07-22 12:15:00 | 1973.10 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-07-16 13:15:00 | 1972.90 | 2025-07-22 12:15:00 | 1973.10 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-07-16 15:00:00 | 1971.90 | 2025-07-22 12:15:00 | 1973.10 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-07-17 09:30:00 | 1973.00 | 2025-07-22 12:15:00 | 1973.10 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-07-25 10:15:00 | 2047.00 | 2025-08-01 09:15:00 | 2008.80 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-07-25 13:30:00 | 2027.00 | 2025-08-01 09:15:00 | 2008.80 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-07-25 14:45:00 | 2026.00 | 2025-08-01 09:15:00 | 2008.80 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-07-28 09:15:00 | 2033.40 | 2025-08-01 09:15:00 | 2008.80 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-07-28 14:45:00 | 2062.90 | 2025-08-01 09:15:00 | 2008.80 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-07-29 09:15:00 | 2064.40 | 2025-08-01 09:15:00 | 2008.80 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2025-07-31 09:15:00 | 2067.20 | 2025-08-01 09:15:00 | 2008.80 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2025-07-31 10:45:00 | 2068.10 | 2025-08-01 09:15:00 | 2008.80 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2025-08-05 10:15:00 | 1962.20 | 2025-08-13 12:15:00 | 1957.00 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2025-08-05 10:45:00 | 1963.00 | 2025-08-13 12:15:00 | 1957.00 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2025-08-05 15:15:00 | 1958.00 | 2025-08-13 12:15:00 | 1957.00 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2025-08-06 12:30:00 | 1953.80 | 2025-08-13 12:15:00 | 1957.00 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-08-07 15:15:00 | 1955.00 | 2025-08-13 12:15:00 | 1957.00 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-08-08 10:15:00 | 1949.30 | 2025-08-13 12:15:00 | 1957.00 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-08-12 09:30:00 | 1955.00 | 2025-08-13 12:15:00 | 1957.00 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-09-02 15:15:00 | 1862.30 | 2025-09-04 09:15:00 | 1893.80 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-09-11 09:15:00 | 1921.90 | 2025-09-17 12:15:00 | 1978.00 | STOP_HIT | 1.00 | 2.92% |
| BUY | retest2 | 2025-09-11 10:15:00 | 1915.00 | 2025-09-17 12:15:00 | 1978.00 | STOP_HIT | 1.00 | 3.29% |
| SELL | retest2 | 2025-09-25 12:30:00 | 1977.70 | 2025-09-29 14:15:00 | 2023.40 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-10-06 09:15:00 | 1947.20 | 2025-10-09 09:15:00 | 1953.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-10-06 10:15:00 | 1949.80 | 2025-10-09 09:15:00 | 1953.00 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-10-06 10:45:00 | 1941.60 | 2025-10-09 09:15:00 | 1953.00 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-10-17 11:15:00 | 1917.80 | 2025-10-17 13:15:00 | 1943.50 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-10-24 09:15:00 | 1934.90 | 2025-10-27 10:15:00 | 1920.50 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-10-24 10:30:00 | 1936.80 | 2025-10-27 10:15:00 | 1920.50 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-10-24 13:30:00 | 1933.00 | 2025-10-27 10:15:00 | 1920.50 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-10-24 14:00:00 | 1933.70 | 2025-10-27 10:15:00 | 1920.50 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-10-29 12:15:00 | 1904.30 | 2025-10-31 13:15:00 | 1938.00 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-10-29 13:15:00 | 1909.60 | 2025-10-31 13:15:00 | 1938.00 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-10-30 12:30:00 | 1910.10 | 2025-10-31 13:15:00 | 1938.00 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-10-30 15:15:00 | 1910.00 | 2025-10-31 13:15:00 | 1938.00 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-11-04 11:15:00 | 1916.90 | 2025-11-04 12:15:00 | 1920.10 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2025-11-11 11:15:00 | 1863.60 | 2025-11-19 09:15:00 | 1770.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-11 12:15:00 | 1866.20 | 2025-11-19 09:15:00 | 1772.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-11 13:15:00 | 1865.00 | 2025-11-19 09:15:00 | 1771.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 10:15:00 | 1847.00 | 2025-11-19 09:15:00 | 1754.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 14:45:00 | 1837.00 | 2025-11-19 10:15:00 | 1745.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-13 14:30:00 | 1838.50 | 2025-11-19 10:15:00 | 1746.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-14 10:15:00 | 1843.80 | 2025-11-19 10:15:00 | 1751.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-14 11:45:00 | 1842.30 | 2025-11-19 10:15:00 | 1750.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-11 11:15:00 | 1863.60 | 2025-11-19 14:15:00 | 1770.00 | STOP_HIT | 0.50 | 5.02% |
| SELL | retest2 | 2025-11-11 12:15:00 | 1866.20 | 2025-11-19 14:15:00 | 1770.00 | STOP_HIT | 0.50 | 5.15% |
| SELL | retest2 | 2025-11-11 13:15:00 | 1865.00 | 2025-11-19 14:15:00 | 1770.00 | STOP_HIT | 0.50 | 5.09% |
| SELL | retest2 | 2025-11-12 10:15:00 | 1847.00 | 2025-11-19 14:15:00 | 1770.00 | STOP_HIT | 0.50 | 4.17% |
| SELL | retest2 | 2025-11-12 14:45:00 | 1837.00 | 2025-11-19 14:15:00 | 1770.00 | STOP_HIT | 0.50 | 3.65% |
| SELL | retest2 | 2025-11-13 14:30:00 | 1838.50 | 2025-11-19 14:15:00 | 1770.00 | STOP_HIT | 0.50 | 3.73% |
| SELL | retest2 | 2025-11-14 10:15:00 | 1843.80 | 2025-11-19 14:15:00 | 1770.00 | STOP_HIT | 0.50 | 4.00% |
| SELL | retest2 | 2025-11-14 11:45:00 | 1842.30 | 2025-11-19 14:15:00 | 1770.00 | STOP_HIT | 0.50 | 3.92% |
| SELL | retest2 | 2025-11-18 15:00:00 | 1786.80 | 2025-11-20 13:15:00 | 1818.60 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-12-01 10:30:00 | 1740.70 | 2025-12-02 10:15:00 | 1775.00 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-12-01 12:00:00 | 1741.60 | 2025-12-02 10:15:00 | 1775.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-12-08 09:15:00 | 1736.90 | 2025-12-18 09:15:00 | 1650.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-08 09:15:00 | 1736.90 | 2025-12-18 11:15:00 | 1663.00 | STOP_HIT | 0.50 | 4.25% |
| SELL | retest2 | 2025-12-29 09:15:00 | 1653.30 | 2025-12-30 14:15:00 | 1669.20 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2026-01-07 11:15:00 | 1689.40 | 2026-01-07 14:15:00 | 1707.90 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-01-07 12:15:00 | 1688.50 | 2026-01-07 14:15:00 | 1707.90 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2026-01-21 14:15:00 | 1680.10 | 2026-01-22 09:15:00 | 1707.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2026-02-02 13:00:00 | 1865.90 | 2026-02-05 13:15:00 | 1849.80 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-02-04 12:30:00 | 1868.50 | 2026-02-05 13:15:00 | 1849.80 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2026-02-10 14:15:00 | 1880.90 | 2026-02-11 09:15:00 | 1858.20 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2026-02-10 15:00:00 | 1887.20 | 2026-02-11 09:15:00 | 1858.20 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest1 | 2026-02-25 09:15:00 | 1795.30 | 2026-02-25 11:15:00 | 1820.20 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-03-10 10:45:00 | 1681.00 | 2026-03-16 10:15:00 | 1596.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-10 13:00:00 | 1681.80 | 2026-03-16 10:15:00 | 1597.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-10 15:15:00 | 1673.70 | 2026-03-16 10:15:00 | 1590.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 11:15:00 | 1679.80 | 2026-03-16 10:15:00 | 1595.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 14:45:00 | 1668.70 | 2026-03-16 10:15:00 | 1585.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 15:15:00 | 1662.50 | 2026-03-16 10:15:00 | 1579.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-10 10:45:00 | 1681.00 | 2026-03-16 14:15:00 | 1607.80 | STOP_HIT | 0.50 | 4.35% |
| SELL | retest2 | 2026-03-10 13:00:00 | 1681.80 | 2026-03-16 14:15:00 | 1607.80 | STOP_HIT | 0.50 | 4.40% |
| SELL | retest2 | 2026-03-10 15:15:00 | 1673.70 | 2026-03-16 14:15:00 | 1607.80 | STOP_HIT | 0.50 | 3.94% |
| SELL | retest2 | 2026-03-11 11:15:00 | 1679.80 | 2026-03-16 14:15:00 | 1607.80 | STOP_HIT | 0.50 | 4.29% |
| SELL | retest2 | 2026-03-11 14:45:00 | 1668.70 | 2026-03-16 14:15:00 | 1607.80 | STOP_HIT | 0.50 | 3.65% |
| SELL | retest2 | 2026-03-11 15:15:00 | 1662.50 | 2026-03-16 14:15:00 | 1607.80 | STOP_HIT | 0.50 | 3.29% |
| SELL | retest2 | 2026-03-13 09:15:00 | 1638.20 | 2026-03-18 09:15:00 | 1647.90 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2026-03-23 09:15:00 | 1619.10 | 2026-03-24 10:15:00 | 1644.80 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest1 | 2026-03-27 11:30:00 | 1707.00 | 2026-03-30 09:15:00 | 1673.30 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest1 | 2026-03-27 15:00:00 | 1723.00 | 2026-03-30 09:15:00 | 1673.30 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2026-03-30 11:15:00 | 1710.60 | 2026-04-02 09:15:00 | 1679.30 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2026-03-30 12:30:00 | 1704.60 | 2026-04-02 09:15:00 | 1679.30 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2026-03-30 13:45:00 | 1704.40 | 2026-04-02 10:15:00 | 1668.40 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2026-04-01 09:15:00 | 1713.30 | 2026-04-02 10:15:00 | 1668.40 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2026-04-01 10:30:00 | 1731.00 | 2026-04-02 10:15:00 | 1668.40 | STOP_HIT | 1.00 | -3.62% |
| BUY | retest2 | 2026-04-01 12:15:00 | 1733.90 | 2026-04-02 10:15:00 | 1668.40 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest2 | 2026-04-13 10:15:00 | 1740.00 | 2026-04-13 13:15:00 | 1729.20 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-04-22 09:30:00 | 1798.60 | 2026-04-24 10:15:00 | 1752.60 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2026-04-22 10:15:00 | 1795.00 | 2026-04-24 10:15:00 | 1752.60 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2026-04-22 12:15:00 | 1814.70 | 2026-04-24 10:15:00 | 1752.60 | STOP_HIT | 1.00 | -3.42% |
| BUY | retest2 | 2026-04-23 09:30:00 | 1816.00 | 2026-04-24 10:15:00 | 1752.60 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2026-04-28 09:15:00 | 1765.10 | 2026-05-04 09:15:00 | 1793.60 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-04-28 10:00:00 | 1764.10 | 2026-05-04 09:15:00 | 1793.60 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2026-04-28 10:30:00 | 1763.90 | 2026-05-04 09:15:00 | 1793.60 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-04-29 09:45:00 | 1763.70 | 2026-05-04 09:15:00 | 1793.60 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-04-29 14:00:00 | 1748.10 | 2026-05-04 09:15:00 | 1793.60 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2026-04-29 14:45:00 | 1746.30 | 2026-05-04 09:15:00 | 1793.60 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2026-04-30 09:15:00 | 1742.00 | 2026-05-04 09:15:00 | 1793.60 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2026-04-30 14:15:00 | 1748.30 | 2026-05-04 09:15:00 | 1793.60 | STOP_HIT | 1.00 | -2.59% |
