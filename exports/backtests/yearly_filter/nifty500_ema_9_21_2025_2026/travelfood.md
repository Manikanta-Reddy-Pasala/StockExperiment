# Travel Food Services Ltd. (TRAVELFOOD)

## Backtest Summary

- **Window:** 2025-07-14 09:15:00 → 2026-05-11 15:15:00 (1416 bars)
- **Last close:** 1199.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 47 |
| ALERT1 | 33 |
| ALERT2 | 32 |
| ALERT2_SKIP | 21 |
| ALERT3 | 119 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 62 |
| PARTIAL | 14 |
| TARGET_HIT | 3 |
| STOP_HIT | 61 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 78 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 38 / 40
- **Target hits / Stop hits / Partials:** 3 / 61 / 14
- **Avg / median % per leg:** 0.86% / -0.16%
- **Sum % (uncompounded):** 66.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 27 | 9 | 33.3% | 3 | 23 | 1 | -0.06% | -1.5% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 5.04% | 10.1% |
| BUY @ 3rd Alert (retest2) | 25 | 7 | 28.0% | 3 | 22 | 0 | -0.46% | -11.6% |
| SELL (all) | 51 | 29 | 56.9% | 0 | 38 | 13 | 1.34% | 68.2% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.89% | -1.9% |
| SELL @ 3rd Alert (retest2) | 50 | 29 | 58.0% | 0 | 37 | 13 | 1.40% | 70.1% |
| retest1 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | 2.73% | 8.2% |
| retest2 (combined) | 75 | 36 | 48.0% | 3 | 59 | 13 | 0.78% | 58.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 09:15:00 | 1138.90 | 1154.27 | 1155.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 12:15:00 | 1114.00 | 1139.76 | 1148.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 13:15:00 | 1107.00 | 1103.39 | 1114.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-30 14:00:00 | 1107.00 | 1103.39 | 1114.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 1047.20 | 1026.07 | 1033.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 14:45:00 | 1051.00 | 1026.07 | 1033.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 1044.50 | 1029.76 | 1034.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:15:00 | 1036.20 | 1029.76 | 1034.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 1033.60 | 1027.42 | 1032.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:00:00 | 1033.60 | 1027.42 | 1032.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 1039.60 | 1029.86 | 1032.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 13:00:00 | 1039.60 | 1029.86 | 1032.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 1046.90 | 1033.27 | 1034.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:00:00 | 1046.90 | 1033.27 | 1034.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 14:15:00 | 1051.20 | 1036.85 | 1035.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 09:15:00 | 1078.50 | 1047.61 | 1040.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-11 09:15:00 | 1073.50 | 1104.46 | 1089.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 09:15:00 | 1073.50 | 1104.46 | 1089.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 1073.50 | 1104.46 | 1089.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:00:00 | 1073.50 | 1104.46 | 1089.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 1069.30 | 1097.43 | 1087.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:00:00 | 1069.30 | 1097.43 | 1087.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 1085.50 | 1085.46 | 1084.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:15:00 | 1128.80 | 1085.46 | 1084.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 1148.00 | 1097.97 | 1089.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 12:30:00 | 1164.10 | 1124.31 | 1105.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 14:30:00 | 1157.40 | 1139.32 | 1115.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 09:45:00 | 1167.60 | 1141.87 | 1120.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 11:30:00 | 1164.90 | 1148.44 | 1127.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 1124.00 | 1146.12 | 1136.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:00:00 | 1124.00 | 1146.12 | 1136.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 1124.50 | 1141.80 | 1135.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:30:00 | 1115.30 | 1141.80 | 1135.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-14 14:15:00 | 1118.00 | 1131.47 | 1131.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 14:15:00 | 1118.00 | 1131.47 | 1131.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 11:15:00 | 1107.30 | 1122.72 | 1127.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 15:15:00 | 1116.80 | 1106.09 | 1111.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 15:15:00 | 1116.80 | 1106.09 | 1111.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 1116.80 | 1106.09 | 1111.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:15:00 | 1136.80 | 1106.09 | 1111.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 1142.00 | 1113.27 | 1114.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:45:00 | 1141.70 | 1113.27 | 1114.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 10:15:00 | 1167.50 | 1124.12 | 1119.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 11:15:00 | 1184.30 | 1136.15 | 1125.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 09:15:00 | 1218.50 | 1219.32 | 1192.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 09:45:00 | 1217.50 | 1219.32 | 1192.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1212.10 | 1225.95 | 1217.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 1205.60 | 1225.95 | 1217.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 1206.10 | 1221.98 | 1216.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:45:00 | 1208.70 | 1221.98 | 1216.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 1204.00 | 1218.38 | 1215.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 12:00:00 | 1204.00 | 1218.38 | 1215.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 1214.40 | 1217.37 | 1215.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 13:30:00 | 1210.70 | 1217.37 | 1215.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 1203.80 | 1214.65 | 1214.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 1203.80 | 1214.65 | 1214.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 15:15:00 | 1210.00 | 1213.72 | 1213.88 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 10:15:00 | 1228.30 | 1213.72 | 1212.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-29 11:15:00 | 1247.40 | 1220.46 | 1215.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-01 14:15:00 | 1252.90 | 1255.81 | 1242.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-01 14:45:00 | 1253.00 | 1255.81 | 1242.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 1250.60 | 1255.27 | 1248.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:30:00 | 1256.80 | 1255.27 | 1248.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 1245.00 | 1253.22 | 1248.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 1254.60 | 1253.22 | 1248.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 1270.60 | 1256.69 | 1250.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 10:30:00 | 1285.30 | 1263.69 | 1254.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 14:45:00 | 1284.50 | 1276.10 | 1264.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 15:15:00 | 1285.00 | 1276.10 | 1264.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:30:00 | 1289.10 | 1285.66 | 1270.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1284.00 | 1307.35 | 1292.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 1284.00 | 1307.35 | 1292.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 1304.40 | 1306.76 | 1293.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 11:45:00 | 1312.20 | 1307.01 | 1294.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 12:30:00 | 1310.00 | 1307.51 | 1296.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 14:15:00 | 1287.00 | 1296.18 | 1296.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-09-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 14:15:00 | 1287.00 | 1296.18 | 1296.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 09:15:00 | 1282.00 | 1291.87 | 1294.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 13:15:00 | 1303.90 | 1291.97 | 1293.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 13:15:00 | 1303.90 | 1291.97 | 1293.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 1303.90 | 1291.97 | 1293.35 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 15:15:00 | 1300.00 | 1295.04 | 1294.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 1322.00 | 1300.43 | 1297.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 10:15:00 | 1296.40 | 1299.62 | 1297.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 10:15:00 | 1296.40 | 1299.62 | 1297.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 1296.40 | 1299.62 | 1297.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:00:00 | 1296.40 | 1299.62 | 1297.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 1283.60 | 1296.42 | 1295.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 1283.60 | 1296.42 | 1295.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 12:15:00 | 1280.80 | 1293.30 | 1294.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 10:15:00 | 1263.10 | 1282.16 | 1288.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 09:15:00 | 1282.50 | 1269.13 | 1276.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 1282.50 | 1269.13 | 1276.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1282.50 | 1269.13 | 1276.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 10:45:00 | 1280.90 | 1271.58 | 1277.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 11:45:00 | 1280.20 | 1273.51 | 1277.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 12:45:00 | 1280.00 | 1274.85 | 1277.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 11:15:00 | 1273.80 | 1258.50 | 1256.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 11:15:00 | 1273.80 | 1258.50 | 1256.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 12:15:00 | 1283.60 | 1263.52 | 1258.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 09:15:00 | 1329.20 | 1337.61 | 1313.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-23 10:00:00 | 1329.20 | 1337.61 | 1313.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 1322.80 | 1332.15 | 1314.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 1317.50 | 1332.15 | 1314.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1340.00 | 1333.27 | 1321.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 11:30:00 | 1346.90 | 1337.27 | 1325.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 14:00:00 | 1346.60 | 1340.02 | 1329.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 09:30:00 | 1352.40 | 1341.15 | 1332.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 10:45:00 | 1347.70 | 1342.50 | 1333.86 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 1346.00 | 1344.02 | 1337.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:30:00 | 1347.00 | 1344.02 | 1337.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 1335.00 | 1343.00 | 1338.07 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-29 10:15:00 | 1323.20 | 1334.88 | 1336.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2025-09-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 10:15:00 | 1323.20 | 1334.88 | 1336.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 11:15:00 | 1322.70 | 1332.44 | 1334.87 | Break + close below crossover candle low |

### Cycle 12 — BUY (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 09:15:00 | 1370.00 | 1337.33 | 1335.75 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 15:15:00 | 1322.00 | 1335.61 | 1336.69 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 11:15:00 | 1347.30 | 1337.74 | 1337.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 1372.00 | 1347.51 | 1342.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 1373.10 | 1373.84 | 1360.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 09:30:00 | 1383.90 | 1373.84 | 1360.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 1379.50 | 1373.46 | 1366.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 1409.10 | 1376.21 | 1373.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 14:15:00 | 1347.20 | 1380.52 | 1379.55 | SL hit (close<static) qty=1.00 sl=1365.00 alert=retest2 |

### Cycle 15 — SELL (started 2025-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 15:15:00 | 1353.00 | 1375.01 | 1377.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 09:15:00 | 1344.40 | 1368.89 | 1374.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 10:15:00 | 1354.90 | 1349.08 | 1357.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 10:15:00 | 1354.90 | 1349.08 | 1357.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 1354.90 | 1349.08 | 1357.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 1354.90 | 1349.08 | 1357.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 1355.50 | 1350.37 | 1357.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:00:00 | 1355.50 | 1350.37 | 1357.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 1357.80 | 1351.85 | 1357.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:30:00 | 1358.00 | 1351.85 | 1357.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 1365.00 | 1354.48 | 1357.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:45:00 | 1368.10 | 1354.48 | 1357.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 1360.10 | 1355.61 | 1358.18 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 09:15:00 | 1372.80 | 1359.88 | 1359.72 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 12:15:00 | 1343.40 | 1358.42 | 1359.36 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 1362.50 | 1359.46 | 1359.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 14:15:00 | 1378.00 | 1363.48 | 1361.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 1371.80 | 1380.68 | 1374.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 1371.80 | 1380.68 | 1374.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 1371.80 | 1380.68 | 1374.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:15:00 | 1369.10 | 1380.68 | 1374.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 1370.10 | 1378.56 | 1373.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:30:00 | 1369.00 | 1378.56 | 1373.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 1371.40 | 1377.63 | 1374.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 1371.40 | 1377.63 | 1374.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 1364.10 | 1374.93 | 1373.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:45:00 | 1361.10 | 1374.93 | 1373.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 15:15:00 | 1360.00 | 1371.38 | 1371.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 09:15:00 | 1351.90 | 1367.48 | 1370.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 12:15:00 | 1340.30 | 1334.76 | 1344.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 12:15:00 | 1340.30 | 1334.76 | 1344.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 1340.30 | 1334.76 | 1344.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:45:00 | 1345.90 | 1334.76 | 1344.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 1340.50 | 1335.91 | 1344.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 13:30:00 | 1342.00 | 1335.91 | 1344.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1336.20 | 1335.96 | 1343.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:30:00 | 1344.90 | 1335.96 | 1343.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 1347.00 | 1338.17 | 1343.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 1334.90 | 1338.17 | 1343.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 10:15:00 | 1333.30 | 1338.28 | 1343.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 14:15:00 | 1335.00 | 1331.75 | 1338.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 09:45:00 | 1335.00 | 1329.63 | 1335.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 1330.00 | 1329.70 | 1334.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:00:00 | 1330.00 | 1329.70 | 1334.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 1330.10 | 1329.78 | 1334.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:00:00 | 1330.10 | 1329.78 | 1334.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 1320.10 | 1327.84 | 1333.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:45:00 | 1321.10 | 1327.84 | 1333.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 1331.00 | 1328.48 | 1333.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:30:00 | 1330.30 | 1328.48 | 1333.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 1320.30 | 1326.84 | 1331.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:30:00 | 1333.10 | 1326.84 | 1331.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1322.50 | 1325.68 | 1330.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 15:15:00 | 1317.00 | 1326.30 | 1328.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 10:15:00 | 1268.15 | 1306.87 | 1318.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 10:15:00 | 1266.63 | 1306.87 | 1318.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 10:15:00 | 1268.25 | 1306.87 | 1318.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-29 10:15:00 | 1268.25 | 1306.87 | 1318.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 12:15:00 | 1302.50 | 1301.41 | 1314.05 | SL hit (close>ema200) qty=0.50 sl=1301.41 alert=retest2 |

### Cycle 20 — BUY (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 12:15:00 | 1328.60 | 1313.85 | 1312.15 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 1306.50 | 1312.69 | 1313.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 1302.00 | 1310.43 | 1311.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 14:15:00 | 1290.90 | 1284.83 | 1292.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 14:15:00 | 1290.90 | 1284.83 | 1292.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 1290.90 | 1284.83 | 1292.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 1290.90 | 1284.83 | 1292.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1290.00 | 1285.87 | 1292.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 1282.00 | 1285.87 | 1292.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 10:00:00 | 1284.60 | 1285.61 | 1291.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 1300.00 | 1288.49 | 1292.49 | SL hit (close>static) qty=1.00 sl=1295.50 alert=retest2 |

### Cycle 22 — BUY (started 2025-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 12:15:00 | 1297.00 | 1289.96 | 1289.67 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 13:15:00 | 1276.90 | 1287.34 | 1288.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 09:15:00 | 1271.90 | 1281.92 | 1285.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 12:15:00 | 1279.20 | 1278.99 | 1283.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 12:15:00 | 1279.20 | 1278.99 | 1283.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 1279.20 | 1278.99 | 1283.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 1265.20 | 1277.53 | 1281.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 14:00:00 | 1270.70 | 1273.53 | 1277.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 1265.00 | 1273.83 | 1276.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 09:45:00 | 1271.20 | 1274.54 | 1275.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 1270.40 | 1273.71 | 1274.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:45:00 | 1270.00 | 1273.71 | 1274.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 1276.20 | 1273.18 | 1274.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:00:00 | 1276.20 | 1273.18 | 1274.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 1275.00 | 1273.55 | 1274.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 13:45:00 | 1278.20 | 1273.55 | 1274.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-17 14:15:00 | 1285.30 | 1275.90 | 1275.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2025-11-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 14:15:00 | 1285.30 | 1275.90 | 1275.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 13:15:00 | 1294.00 | 1285.46 | 1280.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 10:15:00 | 1290.60 | 1291.12 | 1285.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 10:15:00 | 1290.60 | 1291.12 | 1285.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 1290.60 | 1291.12 | 1285.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:00:00 | 1290.60 | 1291.12 | 1285.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 1348.70 | 1347.27 | 1334.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 14:30:00 | 1355.50 | 1349.40 | 1340.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 09:15:00 | 1370.50 | 1350.52 | 1341.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 09:15:00 | 1402.80 | 1364.06 | 1354.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 1363.10 | 1372.77 | 1368.07 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 1355.30 | 1365.97 | 1365.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:30:00 | 1351.10 | 1365.97 | 1365.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-28 11:15:00 | 1355.70 | 1363.92 | 1364.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 1355.70 | 1363.92 | 1364.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 1345.00 | 1357.73 | 1361.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 1370.00 | 1358.21 | 1360.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 1370.00 | 1358.21 | 1360.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1370.00 | 1358.21 | 1360.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:00:00 | 1370.00 | 1358.21 | 1360.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1379.70 | 1362.51 | 1362.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:00:00 | 1379.70 | 1362.51 | 1362.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 11:15:00 | 1379.80 | 1365.97 | 1364.18 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 1344.10 | 1364.79 | 1365.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 1335.50 | 1358.93 | 1362.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 11:15:00 | 1322.20 | 1317.41 | 1329.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 12:00:00 | 1322.20 | 1317.41 | 1329.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 1335.00 | 1321.34 | 1329.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:00:00 | 1335.00 | 1321.34 | 1329.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 1330.40 | 1323.15 | 1329.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:15:00 | 1335.00 | 1323.15 | 1329.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1334.90 | 1327.40 | 1330.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:45:00 | 1337.30 | 1327.40 | 1330.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 1326.80 | 1327.28 | 1329.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:45:00 | 1332.70 | 1327.28 | 1329.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 1326.70 | 1327.16 | 1329.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 13:15:00 | 1322.00 | 1326.93 | 1329.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 09:15:00 | 1255.90 | 1308.79 | 1319.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 1277.50 | 1268.96 | 1285.65 | SL hit (close>ema200) qty=0.50 sl=1268.96 alert=retest2 |

### Cycle 28 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 1284.70 | 1279.84 | 1279.61 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 1272.00 | 1278.66 | 1279.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 13:15:00 | 1268.10 | 1274.12 | 1276.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 15:15:00 | 1257.90 | 1255.13 | 1262.67 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 09:30:00 | 1242.60 | 1252.18 | 1260.64 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 1251.10 | 1248.80 | 1255.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 13:45:00 | 1250.00 | 1248.80 | 1255.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 1266.10 | 1252.26 | 1256.81 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-17 14:15:00 | 1266.10 | 1252.26 | 1256.81 | SL hit (close>ema400) qty=1.00 sl=1256.81 alert=retest1 |

### Cycle 30 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 1178.00 | 1168.26 | 1167.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 13:15:00 | 1183.40 | 1172.30 | 1169.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 1188.10 | 1194.76 | 1186.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 10:15:00 | 1188.10 | 1194.76 | 1186.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 1188.10 | 1194.76 | 1186.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:45:00 | 1187.60 | 1194.76 | 1186.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 1192.90 | 1194.39 | 1187.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:30:00 | 1189.00 | 1194.39 | 1187.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 1183.40 | 1192.19 | 1187.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 13:00:00 | 1183.40 | 1192.19 | 1187.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 1185.90 | 1190.93 | 1186.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 15:15:00 | 1194.00 | 1189.81 | 1186.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 1165.30 | 1185.58 | 1185.46 | SL hit (close<static) qty=1.00 sl=1182.00 alert=retest2 |

### Cycle 31 — SELL (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 10:15:00 | 1155.90 | 1179.64 | 1182.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 12:15:00 | 1152.80 | 1170.28 | 1177.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 14:15:00 | 1139.20 | 1137.65 | 1153.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-08 14:45:00 | 1139.20 | 1137.65 | 1153.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 1148.00 | 1139.97 | 1151.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:45:00 | 1151.00 | 1139.97 | 1151.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 1156.80 | 1143.33 | 1151.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:00:00 | 1156.80 | 1143.33 | 1151.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 1150.00 | 1144.67 | 1151.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 15:00:00 | 1132.00 | 1143.67 | 1149.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 14:30:00 | 1143.20 | 1145.93 | 1147.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 1160.00 | 1148.75 | 1148.90 | SL hit (close>static) qty=1.00 sl=1157.40 alert=retest2 |

### Cycle 32 — BUY (started 2026-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 09:15:00 | 1164.00 | 1151.80 | 1150.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 10:15:00 | 1172.40 | 1155.92 | 1152.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 14:15:00 | 1150.50 | 1158.60 | 1155.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 14:15:00 | 1150.50 | 1158.60 | 1155.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 1150.50 | 1158.60 | 1155.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:45:00 | 1150.00 | 1158.60 | 1155.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 1155.00 | 1157.88 | 1155.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 1142.50 | 1157.88 | 1155.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 1140.40 | 1154.38 | 1153.80 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 10:15:00 | 1139.70 | 1151.45 | 1152.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 09:15:00 | 1127.30 | 1141.37 | 1146.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 13:15:00 | 1142.80 | 1135.18 | 1141.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 13:15:00 | 1142.80 | 1135.18 | 1141.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 1142.80 | 1135.18 | 1141.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:00:00 | 1142.80 | 1135.18 | 1141.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 1146.00 | 1137.35 | 1141.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:45:00 | 1146.90 | 1137.35 | 1141.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 1144.30 | 1138.74 | 1141.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 1125.00 | 1138.74 | 1141.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 10:00:00 | 1127.30 | 1136.45 | 1140.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 1068.75 | 1094.36 | 1107.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 1070.93 | 1094.36 | 1107.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 1093.50 | 1087.34 | 1096.87 | SL hit (close>ema200) qty=0.50 sl=1087.34 alert=retest2 |

### Cycle 34 — BUY (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 14:15:00 | 1070.00 | 1063.01 | 1062.40 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 1047.10 | 1059.89 | 1061.09 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 1074.90 | 1062.36 | 1061.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 10:15:00 | 1093.50 | 1071.95 | 1066.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 1090.20 | 1091.36 | 1081.46 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-04 12:15:00 | 1108.10 | 1091.57 | 1082.45 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1104.20 | 1108.72 | 1101.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:30:00 | 1097.20 | 1108.72 | 1101.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1105.40 | 1108.06 | 1102.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 1103.90 | 1108.06 | 1102.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1124.70 | 1113.95 | 1107.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 10:15:00 | 1133.20 | 1113.95 | 1107.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 13:00:00 | 1134.90 | 1122.44 | 1113.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 15:15:00 | 1163.50 | 1136.05 | 1122.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-11 09:15:00 | 1164.50 | 1172.13 | 1153.36 | SL hit (close<ema200) qty=0.50 sl=1172.13 alert=retest1 |

### Cycle 37 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 1208.50 | 1218.36 | 1219.10 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2026-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 13:15:00 | 1231.70 | 1221.33 | 1220.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 1240.90 | 1230.90 | 1226.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 15:15:00 | 1235.70 | 1237.40 | 1232.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 15:15:00 | 1235.70 | 1237.40 | 1232.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 1235.70 | 1237.40 | 1232.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 1221.00 | 1237.40 | 1232.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1224.90 | 1234.90 | 1231.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:30:00 | 1221.00 | 1234.90 | 1231.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 1225.90 | 1233.10 | 1230.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:45:00 | 1222.30 | 1233.10 | 1230.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 1228.00 | 1231.66 | 1230.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:45:00 | 1230.00 | 1231.66 | 1230.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 1231.50 | 1231.63 | 1230.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:30:00 | 1228.70 | 1231.63 | 1230.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 1230.30 | 1231.36 | 1230.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 1230.30 | 1231.36 | 1230.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 1240.00 | 1233.09 | 1231.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 1197.80 | 1233.09 | 1231.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1181.60 | 1222.79 | 1226.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 1149.90 | 1180.50 | 1199.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 1180.00 | 1166.20 | 1180.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 1180.00 | 1166.20 | 1180.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 1180.00 | 1166.20 | 1180.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:30:00 | 1180.00 | 1166.20 | 1180.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 1173.30 | 1167.62 | 1179.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:30:00 | 1180.00 | 1167.62 | 1179.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 1178.90 | 1171.20 | 1179.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:30:00 | 1179.10 | 1171.20 | 1179.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 1178.10 | 1172.58 | 1179.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:30:00 | 1180.00 | 1172.58 | 1179.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 1184.90 | 1175.05 | 1179.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 1184.90 | 1175.05 | 1179.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 1180.00 | 1176.04 | 1179.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 1182.40 | 1176.04 | 1179.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1181.90 | 1177.21 | 1180.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 1131.90 | 1180.60 | 1180.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-10 12:15:00 | 1191.80 | 1174.70 | 1172.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 1191.80 | 1174.70 | 1172.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 1194.00 | 1183.34 | 1177.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 1185.90 | 1187.73 | 1182.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 15:00:00 | 1185.90 | 1187.73 | 1182.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1180.20 | 1186.22 | 1182.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 1160.50 | 1186.22 | 1182.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1186.20 | 1186.22 | 1182.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 1174.00 | 1186.22 | 1182.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 1180.30 | 1185.04 | 1182.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 1180.30 | 1185.04 | 1182.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 1188.90 | 1185.81 | 1182.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 13:45:00 | 1195.00 | 1188.25 | 1184.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 1163.40 | 1184.42 | 1183.95 | SL hit (close<static) qty=1.00 sl=1176.00 alert=retest2 |

### Cycle 41 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 1157.40 | 1179.01 | 1181.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 1147.30 | 1170.51 | 1177.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 14:15:00 | 1180.60 | 1172.12 | 1176.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 14:15:00 | 1180.60 | 1172.12 | 1176.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 1180.60 | 1172.12 | 1176.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 15:00:00 | 1180.60 | 1172.12 | 1176.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 1166.00 | 1170.90 | 1175.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 15:00:00 | 1155.30 | 1169.51 | 1173.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 09:45:00 | 1157.60 | 1165.13 | 1170.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:00:00 | 1158.50 | 1163.80 | 1169.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 15:15:00 | 1097.53 | 1119.94 | 1132.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 15:15:00 | 1099.72 | 1119.94 | 1132.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 15:15:00 | 1100.58 | 1119.94 | 1132.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-20 12:15:00 | 1116.70 | 1114.98 | 1125.88 | SL hit (close>ema200) qty=0.50 sl=1114.98 alert=retest2 |

### Cycle 42 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 1139.90 | 1116.91 | 1115.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 1155.90 | 1131.46 | 1123.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 09:15:00 | 1149.40 | 1158.94 | 1148.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 1149.40 | 1158.94 | 1148.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 1149.40 | 1158.94 | 1148.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:15:00 | 1151.60 | 1158.94 | 1148.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 1140.80 | 1155.31 | 1148.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 11:00:00 | 1140.80 | 1155.31 | 1148.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 11:15:00 | 1152.00 | 1154.65 | 1148.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 12:15:00 | 1172.00 | 1154.65 | 1148.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-30 14:15:00 | 1289.20 | 1187.12 | 1165.57 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2026-04-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 13:15:00 | 1300.40 | 1309.15 | 1309.60 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2026-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 15:15:00 | 1330.00 | 1313.28 | 1311.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 10:15:00 | 1346.00 | 1320.29 | 1314.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 14:15:00 | 1303.70 | 1321.07 | 1317.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-17 14:15:00 | 1303.70 | 1321.07 | 1317.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 14:15:00 | 1303.70 | 1321.07 | 1317.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 15:00:00 | 1303.70 | 1321.07 | 1317.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 15:15:00 | 1310.00 | 1318.86 | 1316.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:45:00 | 1310.00 | 1317.89 | 1316.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 1310.40 | 1316.39 | 1316.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 12:00:00 | 1315.80 | 1316.27 | 1316.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 12:15:00 | 1306.60 | 1314.34 | 1315.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2026-04-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 12:15:00 | 1306.60 | 1314.34 | 1315.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 14:15:00 | 1283.50 | 1307.56 | 1311.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 10:15:00 | 1308.00 | 1303.12 | 1308.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 10:15:00 | 1308.00 | 1303.12 | 1308.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 1308.00 | 1303.12 | 1308.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 11:00:00 | 1308.00 | 1303.12 | 1308.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 1308.00 | 1304.09 | 1308.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:00:00 | 1308.00 | 1304.09 | 1308.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 1310.90 | 1305.45 | 1308.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:30:00 | 1305.90 | 1305.45 | 1308.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 1310.10 | 1306.38 | 1308.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:00:00 | 1310.10 | 1306.38 | 1308.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 1298.60 | 1304.83 | 1307.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 15:15:00 | 1294.00 | 1304.83 | 1307.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 13:15:00 | 1312.10 | 1305.84 | 1306.73 | SL hit (close>static) qty=1.00 sl=1311.80 alert=retest2 |

### Cycle 46 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 1304.20 | 1282.86 | 1280.90 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 11:15:00 | 1261.00 | 1282.50 | 1284.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 10:15:00 | 1248.50 | 1271.27 | 1277.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-07 15:15:00 | 1253.90 | 1251.75 | 1258.74 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:30:00 | 1242.00 | 1251.26 | 1257.88 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 10:15:00 | 1244.60 | 1251.26 | 1257.88 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 14:15:00 | 1244.20 | 1249.28 | 1254.75 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-11 09:15:00 | 1225.10 | 1249.35 | 1253.83 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-08-12 12:30:00 | 1164.10 | 2025-08-14 14:15:00 | 1118.00 | STOP_HIT | 1.00 | -3.96% |
| BUY | retest2 | 2025-08-12 14:30:00 | 1157.40 | 2025-08-14 14:15:00 | 1118.00 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2025-08-13 09:45:00 | 1167.60 | 2025-08-14 14:15:00 | 1118.00 | STOP_HIT | 1.00 | -4.25% |
| BUY | retest2 | 2025-08-13 11:30:00 | 1164.90 | 2025-08-14 14:15:00 | 1118.00 | STOP_HIT | 1.00 | -4.03% |
| BUY | retest2 | 2025-09-03 10:30:00 | 1285.30 | 2025-09-08 14:15:00 | 1287.00 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2025-09-03 14:45:00 | 1284.50 | 2025-09-08 14:15:00 | 1287.00 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-09-03 15:15:00 | 1285.00 | 2025-09-08 14:15:00 | 1287.00 | STOP_HIT | 1.00 | 0.16% |
| BUY | retest2 | 2025-09-04 09:30:00 | 1289.10 | 2025-09-08 14:15:00 | 1287.00 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-09-05 11:45:00 | 1312.20 | 2025-09-08 14:15:00 | 1287.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-09-05 12:30:00 | 1310.00 | 2025-09-08 14:15:00 | 1287.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-09-12 10:45:00 | 1280.90 | 2025-09-18 11:15:00 | 1273.80 | STOP_HIT | 1.00 | 0.55% |
| SELL | retest2 | 2025-09-12 11:45:00 | 1280.20 | 2025-09-18 11:15:00 | 1273.80 | STOP_HIT | 1.00 | 0.50% |
| SELL | retest2 | 2025-09-12 12:45:00 | 1280.00 | 2025-09-18 11:15:00 | 1273.80 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2025-09-24 11:30:00 | 1346.90 | 2025-09-29 10:15:00 | 1323.20 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-09-24 14:00:00 | 1346.60 | 2025-09-29 10:15:00 | 1323.20 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-09-25 09:30:00 | 1352.40 | 2025-09-29 10:15:00 | 1323.20 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-09-25 10:45:00 | 1347.70 | 2025-09-29 10:15:00 | 1323.20 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-10-09 09:15:00 | 1409.10 | 2025-10-09 14:15:00 | 1347.20 | STOP_HIT | 1.00 | -4.39% |
| SELL | retest2 | 2025-10-24 09:15:00 | 1334.90 | 2025-10-29 10:15:00 | 1268.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-24 10:15:00 | 1333.30 | 2025-10-29 10:15:00 | 1266.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-24 14:15:00 | 1335.00 | 2025-10-29 10:15:00 | 1268.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-27 09:45:00 | 1335.00 | 2025-10-29 10:15:00 | 1268.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-24 09:15:00 | 1334.90 | 2025-10-29 12:15:00 | 1302.50 | STOP_HIT | 0.50 | 2.43% |
| SELL | retest2 | 2025-10-24 10:15:00 | 1333.30 | 2025-10-29 12:15:00 | 1302.50 | STOP_HIT | 0.50 | 2.31% |
| SELL | retest2 | 2025-10-24 14:15:00 | 1335.00 | 2025-10-29 12:15:00 | 1302.50 | STOP_HIT | 0.50 | 2.43% |
| SELL | retest2 | 2025-10-27 09:45:00 | 1335.00 | 2025-10-29 12:15:00 | 1302.50 | STOP_HIT | 0.50 | 2.43% |
| SELL | retest2 | 2025-10-28 15:15:00 | 1317.00 | 2025-10-31 12:15:00 | 1328.60 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-11-10 09:15:00 | 1282.00 | 2025-11-10 10:15:00 | 1300.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-11-10 10:00:00 | 1284.60 | 2025-11-10 10:15:00 | 1300.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-11-10 12:30:00 | 1282.10 | 2025-11-11 11:15:00 | 1305.10 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-11-13 09:15:00 | 1265.20 | 2025-11-17 14:15:00 | 1285.30 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-11-13 14:00:00 | 1270.70 | 2025-11-17 14:15:00 | 1285.30 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-11-14 09:15:00 | 1265.00 | 2025-11-17 14:15:00 | 1285.30 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-11-17 09:45:00 | 1271.20 | 2025-11-17 14:15:00 | 1285.30 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-11-25 14:30:00 | 1355.50 | 2025-11-28 11:15:00 | 1355.70 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-11-26 09:15:00 | 1370.50 | 2025-11-28 11:15:00 | 1355.70 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-11-27 09:15:00 | 1402.80 | 2025-11-28 11:15:00 | 1355.70 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest2 | 2025-11-28 09:15:00 | 1363.10 | 2025-11-28 11:15:00 | 1355.70 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-12-05 13:15:00 | 1322.00 | 2025-12-08 09:15:00 | 1255.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 13:15:00 | 1322.00 | 2025-12-09 12:15:00 | 1277.50 | STOP_HIT | 0.50 | 3.37% |
| SELL | retest1 | 2025-12-17 09:30:00 | 1242.60 | 2025-12-17 14:15:00 | 1266.10 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-12-18 09:30:00 | 1232.00 | 2025-12-26 14:15:00 | 1170.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-18 10:30:00 | 1231.90 | 2025-12-26 14:15:00 | 1170.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-22 11:00:00 | 1231.00 | 2025-12-26 14:15:00 | 1169.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-18 09:30:00 | 1232.00 | 2025-12-31 09:15:00 | 1155.60 | STOP_HIT | 0.50 | 6.20% |
| SELL | retest2 | 2025-12-18 10:30:00 | 1231.90 | 2025-12-31 09:15:00 | 1155.60 | STOP_HIT | 0.50 | 6.19% |
| SELL | retest2 | 2025-12-22 11:00:00 | 1231.00 | 2025-12-31 09:15:00 | 1155.60 | STOP_HIT | 0.50 | 6.13% |
| BUY | retest2 | 2026-01-06 15:15:00 | 1194.00 | 2026-01-07 09:15:00 | 1165.30 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2026-01-09 15:00:00 | 1132.00 | 2026-01-12 15:15:00 | 1160.00 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2026-01-12 14:30:00 | 1143.20 | 2026-01-12 15:15:00 | 1160.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-01-19 09:15:00 | 1125.00 | 2026-01-21 10:15:00 | 1068.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 10:00:00 | 1127.30 | 2026-01-21 10:15:00 | 1070.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 09:15:00 | 1125.00 | 2026-01-22 09:15:00 | 1093.50 | STOP_HIT | 0.50 | 2.80% |
| SELL | retest2 | 2026-01-19 10:00:00 | 1127.30 | 2026-01-22 09:15:00 | 1093.50 | STOP_HIT | 0.50 | 3.00% |
| BUY | retest1 | 2026-02-04 12:15:00 | 1108.10 | 2026-02-09 15:15:00 | 1163.50 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-02-04 12:15:00 | 1108.10 | 2026-02-11 09:15:00 | 1164.50 | STOP_HIT | 0.50 | 5.09% |
| BUY | retest2 | 2026-02-09 10:15:00 | 1133.20 | 2026-02-13 09:15:00 | 1246.52 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-09 13:00:00 | 1134.90 | 2026-02-13 09:15:00 | 1248.39 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1131.90 | 2026-03-10 12:15:00 | 1191.80 | STOP_HIT | 1.00 | -5.29% |
| BUY | retest2 | 2026-03-12 13:45:00 | 1195.00 | 2026-03-13 09:15:00 | 1163.40 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2026-03-16 15:00:00 | 1155.30 | 2026-03-19 15:15:00 | 1097.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-17 09:45:00 | 1157.60 | 2026-03-19 15:15:00 | 1099.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-17 11:00:00 | 1158.50 | 2026-03-19 15:15:00 | 1100.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-16 15:00:00 | 1155.30 | 2026-03-20 12:15:00 | 1116.70 | STOP_HIT | 0.50 | 3.34% |
| SELL | retest2 | 2026-03-17 09:45:00 | 1157.60 | 2026-03-20 12:15:00 | 1116.70 | STOP_HIT | 0.50 | 3.53% |
| SELL | retest2 | 2026-03-17 11:00:00 | 1158.50 | 2026-03-20 12:15:00 | 1116.70 | STOP_HIT | 0.50 | 3.61% |
| BUY | retest2 | 2026-03-30 12:15:00 | 1172.00 | 2026-03-30 14:15:00 | 1289.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-20 12:00:00 | 1315.80 | 2026-04-20 12:15:00 | 1306.60 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2026-04-21 15:15:00 | 1294.00 | 2026-04-22 13:15:00 | 1312.10 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-04-23 10:45:00 | 1293.80 | 2026-05-04 09:15:00 | 1314.90 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2026-04-27 09:30:00 | 1296.10 | 2026-05-04 09:15:00 | 1314.90 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-04-27 13:45:00 | 1295.60 | 2026-05-04 09:15:00 | 1314.90 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2026-04-29 11:00:00 | 1276.30 | 2026-05-04 09:15:00 | 1314.90 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2026-04-29 11:45:00 | 1277.40 | 2026-05-04 09:15:00 | 1314.90 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2026-04-29 12:15:00 | 1277.40 | 2026-05-04 09:15:00 | 1314.90 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2026-04-29 12:45:00 | 1275.10 | 2026-05-04 09:15:00 | 1314.90 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2026-04-30 09:15:00 | 1276.30 | 2026-05-04 09:15:00 | 1314.90 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2026-04-30 10:15:00 | 1273.60 | 2026-05-04 09:15:00 | 1314.90 | STOP_HIT | 1.00 | -3.24% |
