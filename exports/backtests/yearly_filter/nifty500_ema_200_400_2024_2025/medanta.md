# Global Health Ltd. (MEDANTA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1202.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT2_SKIP | 2 |
| ALERT3 | 39 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 27 |
| PARTIAL | 8 |
| TARGET_HIT | 7 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 35 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 16
- **Target hits / Stop hits / Partials:** 7 / 20 / 8
- **Avg / median % per leg:** 2.22% / 0.91%
- **Sum % (uncompounded):** 77.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 3 | 20.0% | 3 | 12 | 0 | 0.36% | 5.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 3 | 20.0% | 3 | 12 | 0 | 0.36% | 5.4% |
| SELL (all) | 20 | 16 | 80.0% | 4 | 8 | 8 | 3.62% | 72.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 20 | 16 | 80.0% | 4 | 8 | 8 | 3.62% | 72.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 35 | 19 | 54.3% | 7 | 20 | 8 | 2.22% | 77.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 15:15:00 | 1185.00 | 1306.38 | 1306.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-03 09:15:00 | 1172.65 | 1305.05 | 1305.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-13 11:15:00 | 1258.50 | 1253.05 | 1274.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-13 12:00:00 | 1258.50 | 1253.05 | 1274.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 14:15:00 | 1274.15 | 1253.50 | 1274.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-13 15:00:00 | 1274.15 | 1253.50 | 1274.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 15:15:00 | 1280.00 | 1253.76 | 1274.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:15:00 | 1304.90 | 1253.76 | 1274.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 1320.05 | 1254.42 | 1275.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 10:00:00 | 1320.05 | 1254.42 | 1275.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 15:15:00 | 1290.00 | 1282.09 | 1286.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:15:00 | 1289.20 | 1282.09 | 1286.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 1294.95 | 1282.22 | 1286.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:00:00 | 1294.95 | 1282.22 | 1286.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 1297.10 | 1282.37 | 1286.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:45:00 | 1296.30 | 1282.37 | 1286.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 1289.95 | 1283.73 | 1287.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:45:00 | 1294.95 | 1283.73 | 1287.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 1302.85 | 1283.92 | 1287.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 11:45:00 | 1304.15 | 1283.92 | 1287.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 1284.30 | 1284.09 | 1287.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:30:00 | 1295.60 | 1284.09 | 1287.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 1293.40 | 1284.18 | 1287.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 11:00:00 | 1293.40 | 1284.18 | 1287.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 1292.15 | 1284.26 | 1287.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 12:15:00 | 1293.65 | 1284.26 | 1287.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 12:15:00 | 1293.15 | 1284.35 | 1287.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 12:30:00 | 1290.60 | 1284.35 | 1287.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 1286.90 | 1283.07 | 1286.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:45:00 | 1286.35 | 1283.07 | 1286.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 14:15:00 | 1275.85 | 1283.00 | 1286.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 13:00:00 | 1263.30 | 1280.59 | 1284.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 14:00:00 | 1262.55 | 1280.41 | 1284.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 12:45:00 | 1264.25 | 1275.89 | 1281.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 10:15:00 | 1200.13 | 1270.04 | 1278.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 10:15:00 | 1199.42 | 1270.04 | 1278.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 10:15:00 | 1201.04 | 1270.04 | 1278.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-30 09:15:00 | 1251.00 | 1236.74 | 1256.45 | SL hit (close>ema200) qty=0.50 sl=1236.74 alert=retest2 |

### Cycle 2 — BUY (started 2024-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 11:15:00 | 1178.80 | 1093.10 | 1092.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 12:15:00 | 1180.55 | 1093.97 | 1093.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 15:15:00 | 1106.00 | 1107.54 | 1100.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-13 09:15:00 | 1103.60 | 1107.54 | 1100.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 1097.30 | 1107.44 | 1100.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:00:00 | 1097.30 | 1107.44 | 1100.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 1098.70 | 1107.35 | 1100.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:30:00 | 1097.30 | 1107.35 | 1100.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 1104.05 | 1107.31 | 1100.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:30:00 | 1103.20 | 1107.31 | 1100.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 1111.90 | 1107.98 | 1101.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:30:00 | 1113.55 | 1107.98 | 1101.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 15:15:00 | 1107.25 | 1108.00 | 1101.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:30:00 | 1123.00 | 1108.07 | 1101.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 1123.50 | 1110.22 | 1103.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 10:15:00 | 1130.85 | 1110.22 | 1103.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 11:15:00 | 1132.50 | 1110.40 | 1103.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 13:00:00 | 1131.65 | 1110.83 | 1103.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 09:30:00 | 1130.75 | 1111.49 | 1104.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 1094.25 | 1111.80 | 1104.44 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-20 14:15:00 | 1094.25 | 1111.80 | 1104.44 | SL hit (close<static) qty=1.00 sl=1100.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 14:15:00 | 1062.90 | 1100.04 | 1100.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 1056.45 | 1094.68 | 1097.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 11:15:00 | 1071.10 | 1067.82 | 1081.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 12:00:00 | 1071.10 | 1067.82 | 1081.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 1081.20 | 1067.96 | 1081.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 13:45:00 | 1080.90 | 1067.96 | 1081.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 1069.00 | 1067.97 | 1081.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 15:15:00 | 1066.00 | 1067.97 | 1081.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 11:15:00 | 1012.70 | 1065.25 | 1078.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-31 12:15:00 | 1060.65 | 1058.50 | 1073.41 | SL hit (close>ema200) qty=0.50 sl=1058.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 11:15:00 | 1130.50 | 1082.84 | 1082.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 12:15:00 | 1134.15 | 1083.35 | 1083.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-17 10:15:00 | 1085.60 | 1086.93 | 1084.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 10:15:00 | 1085.60 | 1086.93 | 1084.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 1085.60 | 1086.93 | 1084.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-17 11:00:00 | 1085.60 | 1086.93 | 1084.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 11:15:00 | 1092.30 | 1086.98 | 1084.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-17 12:45:00 | 1094.75 | 1087.04 | 1084.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-17 13:30:00 | 1095.75 | 1087.14 | 1085.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-17 14:00:00 | 1096.50 | 1087.14 | 1085.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-02-20 14:15:00 | 1204.23 | 1099.47 | 1091.65 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 11:15:00 | 1139.60 | 1198.63 | 1198.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 12:15:00 | 1130.90 | 1197.96 | 1198.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 1185.50 | 1172.30 | 1183.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 1185.50 | 1172.30 | 1183.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1185.50 | 1172.30 | 1183.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 1185.50 | 1172.30 | 1183.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 1190.00 | 1172.48 | 1183.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 09:15:00 | 1184.10 | 1173.29 | 1183.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 13:15:00 | 1204.70 | 1174.35 | 1183.61 | SL hit (close>static) qty=1.00 sl=1192.30 alert=retest2 |

### Cycle 6 — BUY (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 15:15:00 | 1300.00 | 1192.16 | 1191.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 13:15:00 | 1309.20 | 1197.59 | 1194.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 14:15:00 | 1364.80 | 1370.62 | 1329.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 15:00:00 | 1364.80 | 1370.62 | 1329.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 13:15:00 | 1332.30 | 1368.55 | 1330.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 13:45:00 | 1330.80 | 1368.55 | 1330.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 1322.60 | 1368.09 | 1330.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 15:00:00 | 1322.60 | 1368.09 | 1330.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 1321.00 | 1367.62 | 1330.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 1320.70 | 1367.62 | 1330.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 1327.90 | 1366.29 | 1330.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 13:15:00 | 1329.40 | 1365.91 | 1330.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 15:15:00 | 1332.00 | 1365.14 | 1330.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 11:15:00 | 1319.20 | 1362.21 | 1335.34 | SL hit (close<static) qty=1.00 sl=1321.10 alert=retest2 |

### Cycle 7 — SELL (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 13:15:00 | 1247.50 | 1336.19 | 1336.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 09:15:00 | 1189.30 | 1333.17 | 1334.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 11:15:00 | 1260.80 | 1260.61 | 1289.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 11:30:00 | 1263.80 | 1260.61 | 1289.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1227.50 | 1188.25 | 1220.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:45:00 | 1228.00 | 1188.25 | 1220.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1227.90 | 1188.65 | 1220.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:45:00 | 1228.70 | 1188.65 | 1220.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 1225.10 | 1189.37 | 1220.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 15:00:00 | 1210.70 | 1192.37 | 1221.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 1234.80 | 1192.96 | 1221.06 | SL hit (close>static) qty=1.00 sl=1227.30 alert=retest2 |

### Cycle 8 — BUY (started 2026-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 12:15:00 | 1186.30 | 1101.69 | 1101.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 1216.10 | 1105.27 | 1103.41 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-07-08 13:00:00 | 1263.30 | 2024-07-15 10:15:00 | 1200.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-08 14:00:00 | 1262.55 | 2024-07-15 10:15:00 | 1199.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-11 12:45:00 | 1264.25 | 2024-07-15 10:15:00 | 1201.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-08 13:00:00 | 1263.30 | 2024-07-30 09:15:00 | 1251.00 | STOP_HIT | 0.50 | 0.97% |
| SELL | retest2 | 2024-07-08 14:00:00 | 1262.55 | 2024-07-30 09:15:00 | 1251.00 | STOP_HIT | 0.50 | 0.91% |
| SELL | retest2 | 2024-07-11 12:45:00 | 1264.25 | 2024-07-30 09:15:00 | 1251.00 | STOP_HIT | 0.50 | 1.05% |
| SELL | retest2 | 2024-07-31 10:15:00 | 1261.95 | 2024-08-05 09:15:00 | 1198.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-31 12:15:00 | 1253.10 | 2024-08-05 09:15:00 | 1190.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-31 14:15:00 | 1251.65 | 2024-08-05 09:15:00 | 1189.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-31 10:15:00 | 1261.95 | 2024-08-09 12:15:00 | 1135.76 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-31 12:15:00 | 1253.10 | 2024-08-12 09:15:00 | 1127.79 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-31 14:15:00 | 1251.65 | 2024-08-12 09:15:00 | 1126.49 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-12-19 10:15:00 | 1130.85 | 2024-12-20 14:15:00 | 1094.25 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2024-12-19 11:15:00 | 1132.50 | 2024-12-20 14:15:00 | 1094.25 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2024-12-19 13:00:00 | 1131.65 | 2024-12-20 14:15:00 | 1094.25 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest2 | 2024-12-20 09:30:00 | 1130.75 | 2024-12-20 14:15:00 | 1094.25 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2024-12-27 09:15:00 | 1109.65 | 2024-12-27 14:15:00 | 1091.55 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-01-02 11:45:00 | 1114.00 | 2025-01-03 12:15:00 | 1092.30 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-01-23 15:15:00 | 1066.00 | 2025-01-27 11:15:00 | 1012.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 15:15:00 | 1066.00 | 2025-01-31 12:15:00 | 1060.65 | STOP_HIT | 0.50 | 0.50% |
| SELL | retest2 | 2025-02-01 14:30:00 | 1065.00 | 2025-02-05 09:15:00 | 1105.35 | STOP_HIT | 1.00 | -3.79% |
| SELL | retest2 | 2025-02-03 09:30:00 | 1066.95 | 2025-02-05 09:15:00 | 1105.35 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2025-02-17 12:45:00 | 1094.75 | 2025-02-20 14:15:00 | 1204.23 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-17 13:30:00 | 1095.75 | 2025-02-20 14:15:00 | 1205.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-17 14:00:00 | 1096.50 | 2025-02-20 14:15:00 | 1206.15 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-04 09:15:00 | 1184.10 | 2025-07-07 13:15:00 | 1204.70 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-09-15 13:15:00 | 1329.40 | 2025-09-23 11:15:00 | 1319.20 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-09-15 15:15:00 | 1332.00 | 2025-09-23 11:15:00 | 1319.20 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-09-25 14:45:00 | 1329.90 | 2025-09-26 09:15:00 | 1302.80 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-09-30 09:30:00 | 1332.70 | 2025-09-30 13:15:00 | 1308.80 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-10-06 09:15:00 | 1339.00 | 2025-10-31 12:15:00 | 1321.10 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-10-31 13:45:00 | 1332.80 | 2025-10-31 14:15:00 | 1319.30 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2026-01-06 15:00:00 | 1210.70 | 2026-01-07 09:15:00 | 1234.80 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2026-01-08 13:00:00 | 1214.20 | 2026-01-16 09:15:00 | 1153.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 13:00:00 | 1214.20 | 2026-01-20 09:15:00 | 1092.78 | TARGET_HIT | 0.50 | 10.00% |
