# Aurobindo Pharma Ltd. (AUROPHARMA)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 1487.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 4 |
| ALERT3 | 58 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 46 |
| PARTIAL | 4 |
| TARGET_HIT | 9 |
| STOP_HIT | 37 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 50 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 37
- **Target hits / Stop hits / Partials:** 9 / 37 / 4
- **Avg / median % per leg:** 1.11% / -1.19%
- **Sum % (uncompounded):** 55.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 33 | 5 | 15.2% | 5 | 28 | 0 | 0.39% | 13.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 33 | 5 | 15.2% | 5 | 28 | 0 | 0.39% | 13.0% |
| SELL (all) | 17 | 8 | 47.1% | 4 | 9 | 4 | 2.49% | 42.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 17 | 8 | 47.1% | 4 | 9 | 4 | 2.49% | 42.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 50 | 13 | 26.0% | 9 | 37 | 4 | 1.11% | 55.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 14:15:00 | 1018.20 | 1044.31 | 1044.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 09:15:00 | 1001.70 | 1043.67 | 1044.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-26 13:15:00 | 1047.20 | 1036.16 | 1039.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 13:15:00 | 1047.20 | 1036.16 | 1039.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 13:15:00 | 1047.20 | 1036.16 | 1039.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-26 14:00:00 | 1047.20 | 1036.16 | 1039.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 14:15:00 | 1050.45 | 1036.31 | 1039.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-26 14:30:00 | 1050.00 | 1036.31 | 1039.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 11:15:00 | 1109.10 | 1043.12 | 1043.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-03 09:15:00 | 1125.25 | 1050.45 | 1046.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-22 12:15:00 | 1080.65 | 1082.06 | 1067.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-22 12:45:00 | 1079.70 | 1082.06 | 1067.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 1193.35 | 1173.05 | 1137.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 13:30:00 | 1206.00 | 1173.37 | 1138.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 14:30:00 | 1204.90 | 1173.65 | 1138.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 15:00:00 | 1201.75 | 1173.65 | 1138.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 09:15:00 | 1218.40 | 1173.86 | 1138.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 1189.25 | 1214.84 | 1180.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:30:00 | 1184.65 | 1214.84 | 1180.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Target hit | 2024-07-08 09:15:00 | 1326.60 | 1223.62 | 1192.11 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-11-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 12:15:00 | 1369.50 | 1452.03 | 1452.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 11:15:00 | 1355.65 | 1444.17 | 1448.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 09:15:00 | 1269.35 | 1268.62 | 1318.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-20 09:45:00 | 1268.00 | 1268.62 | 1318.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 1313.55 | 1267.80 | 1308.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:30:00 | 1326.35 | 1267.80 | 1308.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 1331.45 | 1268.43 | 1308.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 11:00:00 | 1331.45 | 1268.43 | 1308.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 11:15:00 | 1327.00 | 1269.01 | 1308.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 12:15:00 | 1329.60 | 1269.01 | 1308.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 1308.05 | 1284.40 | 1312.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 11:45:00 | 1300.65 | 1284.68 | 1312.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-07 10:15:00 | 1322.60 | 1286.37 | 1312.17 | SL hit (close>static) qty=1.00 sl=1315.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 14:15:00 | 1208.00 | 1167.04 | 1166.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 09:15:00 | 1228.30 | 1168.09 | 1167.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 09:15:00 | 1181.50 | 1182.26 | 1175.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 09:45:00 | 1180.20 | 1182.26 | 1175.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 12:15:00 | 1172.50 | 1182.20 | 1175.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:00:00 | 1172.50 | 1182.20 | 1175.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 1162.80 | 1182.01 | 1175.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 1162.80 | 1182.01 | 1175.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 1184.80 | 1193.15 | 1183.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:00:00 | 1184.80 | 1193.15 | 1183.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 1179.00 | 1193.01 | 1183.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 13:00:00 | 1179.00 | 1193.01 | 1183.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 1183.00 | 1192.91 | 1183.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:15:00 | 1178.50 | 1192.91 | 1183.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 1189.40 | 1192.87 | 1183.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:30:00 | 1186.30 | 1192.87 | 1183.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 1193.40 | 1192.82 | 1183.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 14:45:00 | 1197.30 | 1192.71 | 1183.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 1201.40 | 1192.70 | 1183.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 10:45:00 | 1198.50 | 1192.73 | 1184.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 09:30:00 | 1203.40 | 1192.32 | 1184.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1184.90 | 1192.24 | 1184.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:00:00 | 1184.90 | 1192.24 | 1184.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 1177.60 | 1192.10 | 1184.03 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-27 11:15:00 | 1177.60 | 1192.10 | 1184.03 | SL hit (close<static) qty=1.00 sl=1179.20 alert=retest2 |

### Cycle 5 — SELL (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 14:15:00 | 1137.80 | 1177.62 | 1177.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 09:15:00 | 1134.50 | 1168.35 | 1172.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 09:15:00 | 1143.40 | 1141.52 | 1155.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 09:15:00 | 1143.40 | 1141.52 | 1155.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1143.40 | 1141.52 | 1155.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 12:30:00 | 1132.50 | 1141.33 | 1155.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 1128.40 | 1141.12 | 1155.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 12:15:00 | 1162.40 | 1140.58 | 1154.01 | SL hit (close>static) qty=1.00 sl=1159.90 alert=retest2 |

### Cycle 6 — BUY (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 13:15:00 | 1103.20 | 1097.75 | 1097.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 1107.90 | 1097.90 | 1097.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 1097.50 | 1098.81 | 1098.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 14:15:00 | 1097.50 | 1098.81 | 1098.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1097.50 | 1098.81 | 1098.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 1097.50 | 1098.81 | 1098.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 1100.00 | 1098.82 | 1098.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 1098.60 | 1098.82 | 1098.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1103.50 | 1098.86 | 1098.32 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 1089.00 | 1097.74 | 1097.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 11:15:00 | 1084.60 | 1097.61 | 1097.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 14:15:00 | 1097.80 | 1097.40 | 1097.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 14:15:00 | 1097.80 | 1097.40 | 1097.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 1097.80 | 1097.40 | 1097.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 1097.80 | 1097.40 | 1097.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 1103.50 | 1097.46 | 1097.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 1107.30 | 1097.46 | 1097.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1109.90 | 1097.58 | 1097.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 1109.90 | 1097.58 | 1097.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 1113.70 | 1097.93 | 1097.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 09:15:00 | 1123.70 | 1099.09 | 1098.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 1187.00 | 1189.47 | 1160.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 13:00:00 | 1187.00 | 1189.47 | 1160.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 1160.50 | 1187.69 | 1160.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 1177.70 | 1187.69 | 1160.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-13 10:45:00 | 1167.70 | 1200.15 | 1183.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 1160.40 | 1199.76 | 1183.86 | SL hit (close<static) qty=1.00 sl=1160.50 alert=retest2 |

### Cycle 9 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 1129.10 | 1173.45 | 1173.54 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 12:15:00 | 1191.20 | 1173.66 | 1173.61 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 1157.40 | 1173.41 | 1173.48 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1228.60 | 1173.71 | 1173.63 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 1131.20 | 1174.73 | 1174.75 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 1222.50 | 1174.56 | 1174.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-09 13:15:00 | 1246.90 | 1190.02 | 1183.04 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-10-25 14:30:00 | 864.90 | 2023-10-26 09:15:00 | 847.55 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2023-10-27 10:45:00 | 864.45 | 2023-10-27 14:15:00 | 853.45 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2023-10-30 10:15:00 | 867.85 | 2023-10-30 11:15:00 | 853.80 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2023-10-31 09:15:00 | 867.85 | 2023-10-31 11:15:00 | 851.90 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2023-11-01 12:45:00 | 857.65 | 2023-11-03 10:15:00 | 850.85 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2023-11-01 13:15:00 | 857.40 | 2023-11-03 10:15:00 | 850.85 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2023-11-02 14:00:00 | 857.00 | 2023-11-03 10:15:00 | 850.85 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2023-11-02 14:45:00 | 857.45 | 2023-11-03 10:15:00 | 850.85 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2023-11-03 09:15:00 | 862.25 | 2023-11-03 10:15:00 | 850.85 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2023-11-06 09:15:00 | 860.85 | 2023-11-08 12:15:00 | 946.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 13:30:00 | 1206.00 | 2024-07-08 09:15:00 | 1326.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 14:30:00 | 1204.90 | 2024-07-08 09:15:00 | 1325.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 15:00:00 | 1201.75 | 2024-07-08 09:15:00 | 1321.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-05 09:15:00 | 1218.40 | 2024-07-10 14:15:00 | 1340.24 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-07 09:15:00 | 1470.25 | 2024-10-07 10:15:00 | 1455.95 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-10-08 10:00:00 | 1472.40 | 2024-10-21 14:15:00 | 1457.45 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-10-11 09:15:00 | 1487.00 | 2024-10-21 14:15:00 | 1457.45 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-10-15 14:15:00 | 1471.30 | 2024-10-21 14:15:00 | 1457.45 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-10-18 12:00:00 | 1475.05 | 2024-10-21 14:15:00 | 1457.45 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-10-18 13:15:00 | 1475.25 | 2024-10-21 14:15:00 | 1457.45 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-10-21 09:15:00 | 1475.25 | 2024-10-21 14:15:00 | 1457.45 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-10-21 09:45:00 | 1485.40 | 2024-10-21 14:15:00 | 1457.45 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2024-10-22 09:15:00 | 1476.40 | 2024-10-22 12:15:00 | 1453.90 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-10-23 11:45:00 | 1464.20 | 2024-10-23 12:15:00 | 1452.40 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-01-06 11:45:00 | 1300.65 | 2025-01-07 10:15:00 | 1322.60 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-01-07 14:45:00 | 1301.15 | 2025-01-10 09:15:00 | 1236.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-07 14:45:00 | 1301.15 | 2025-01-13 09:15:00 | 1171.04 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-05-23 14:45:00 | 1197.30 | 2025-05-27 11:15:00 | 1177.60 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-05-26 09:15:00 | 1201.40 | 2025-05-27 11:15:00 | 1177.60 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-05-26 10:45:00 | 1198.50 | 2025-05-27 11:15:00 | 1177.60 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-05-27 09:30:00 | 1203.40 | 2025-05-27 11:15:00 | 1177.60 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-05-27 15:00:00 | 1191.90 | 2025-05-28 09:15:00 | 1167.10 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-06-30 12:30:00 | 1132.50 | 2025-07-02 12:15:00 | 1162.40 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2025-07-01 09:15:00 | 1128.40 | 2025-07-02 12:15:00 | 1162.40 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2025-07-10 11:00:00 | 1131.60 | 2025-07-17 10:15:00 | 1157.90 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-07-10 12:45:00 | 1131.30 | 2025-07-17 11:15:00 | 1161.20 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2025-07-16 11:15:00 | 1146.50 | 2025-07-17 11:15:00 | 1161.20 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-07-18 10:15:00 | 1143.40 | 2025-07-29 14:15:00 | 1159.60 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-07-18 13:00:00 | 1143.70 | 2025-07-29 14:15:00 | 1159.60 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-07-21 12:45:00 | 1145.30 | 2025-07-29 14:15:00 | 1159.60 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-07-31 11:15:00 | 1147.00 | 2025-08-01 09:15:00 | 1089.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 13:45:00 | 1150.60 | 2025-08-01 09:15:00 | 1093.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 14:15:00 | 1148.10 | 2025-08-01 09:15:00 | 1090.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 11:15:00 | 1147.00 | 2025-08-11 09:15:00 | 1032.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-31 13:45:00 | 1150.60 | 2025-08-11 09:15:00 | 1035.54 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-31 14:15:00 | 1148.10 | 2025-08-11 09:15:00 | 1033.29 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-12-10 09:15:00 | 1177.70 | 2026-01-13 11:15:00 | 1160.40 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2026-01-13 10:45:00 | 1167.70 | 2026-01-13 11:15:00 | 1160.40 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2026-01-13 13:00:00 | 1166.10 | 2026-01-20 09:15:00 | 1158.90 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-01-13 14:45:00 | 1169.10 | 2026-01-20 09:15:00 | 1158.90 | STOP_HIT | 1.00 | -0.87% |
