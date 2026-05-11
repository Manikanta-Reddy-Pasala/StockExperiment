# Tejas Networks Ltd. (TEJASNET)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 515.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 138 |
| ALERT1 | 98 |
| ALERT2 | 98 |
| ALERT2_SKIP | 54 |
| ALERT3 | 252 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 123 |
| PARTIAL | 28 |
| TARGET_HIT | 19 |
| STOP_HIT | 107 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 154 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 69 / 85
- **Target hits / Stop hits / Partials:** 19 / 107 / 28
- **Avg / median % per leg:** 1.21% / -0.23%
- **Sum % (uncompounded):** 186.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 48 | 10 | 20.8% | 5 | 42 | 1 | -0.72% | -34.8% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.80% | 9.6% |
| BUY @ 3rd Alert (retest2) | 46 | 8 | 17.4% | 5 | 41 | 0 | -0.97% | -44.4% |
| SELL (all) | 106 | 59 | 55.7% | 14 | 65 | 27 | 2.08% | 220.8% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.99% | -6.0% |
| SELL @ 3rd Alert (retest2) | 104 | 59 | 56.7% | 14 | 63 | 27 | 2.18% | 226.8% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 3 | 1 | 0.91% | 3.6% |
| retest2 (combined) | 150 | 67 | 44.7% | 19 | 104 | 27 | 1.22% | 182.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 10:15:00 | 1183.00 | 1197.36 | 1198.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 12:15:00 | 1177.45 | 1190.44 | 1194.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 15:15:00 | 1165.00 | 1159.94 | 1171.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-23 09:15:00 | 1170.50 | 1159.94 | 1171.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 1169.05 | 1161.76 | 1171.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:45:00 | 1175.00 | 1161.76 | 1171.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 1180.00 | 1165.41 | 1171.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 11:00:00 | 1180.00 | 1165.41 | 1171.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 1192.00 | 1170.73 | 1173.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 11:45:00 | 1196.30 | 1170.73 | 1173.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 14:15:00 | 1167.45 | 1173.95 | 1174.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 14:30:00 | 1177.20 | 1173.95 | 1174.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 15:15:00 | 1173.00 | 1173.76 | 1174.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:15:00 | 1185.00 | 1173.76 | 1174.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 1179.40 | 1174.89 | 1175.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:30:00 | 1178.00 | 1174.89 | 1175.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 10:15:00 | 1184.70 | 1176.85 | 1175.99 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 11:15:00 | 1160.00 | 1173.48 | 1174.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 14:15:00 | 1154.85 | 1168.58 | 1172.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 09:15:00 | 1175.00 | 1167.83 | 1170.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 09:15:00 | 1175.00 | 1167.83 | 1170.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 1175.00 | 1167.83 | 1170.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:45:00 | 1163.00 | 1167.83 | 1170.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 1176.70 | 1169.60 | 1171.47 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2024-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 12:15:00 | 1177.90 | 1172.97 | 1172.78 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 14:15:00 | 1160.85 | 1170.43 | 1171.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 09:15:00 | 1141.45 | 1164.24 | 1168.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 13:15:00 | 1163.00 | 1162.15 | 1166.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 13:15:00 | 1163.00 | 1162.15 | 1166.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 1163.00 | 1162.15 | 1166.02 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 10:15:00 | 1184.00 | 1169.40 | 1168.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 13:15:00 | 1194.80 | 1179.52 | 1173.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 14:15:00 | 1176.60 | 1178.93 | 1174.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 14:15:00 | 1176.60 | 1178.93 | 1174.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 1176.60 | 1178.93 | 1174.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 14:45:00 | 1176.15 | 1178.93 | 1174.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 15:15:00 | 1173.00 | 1177.75 | 1173.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:15:00 | 1165.45 | 1177.75 | 1173.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 1152.25 | 1172.65 | 1172.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 10:00:00 | 1152.25 | 1172.65 | 1172.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2024-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 10:15:00 | 1151.20 | 1168.36 | 1170.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 11:15:00 | 1144.10 | 1163.51 | 1167.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 12:15:00 | 1149.15 | 1145.57 | 1153.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 12:15:00 | 1149.15 | 1145.57 | 1153.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 12:15:00 | 1149.15 | 1145.57 | 1153.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 13:00:00 | 1149.15 | 1145.57 | 1153.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 1106.80 | 1138.46 | 1148.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:45:00 | 1153.75 | 1138.46 | 1148.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1166.50 | 1139.80 | 1147.50 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 12:15:00 | 1168.65 | 1154.15 | 1152.94 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 1052.35 | 1136.65 | 1146.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 1040.50 | 1073.92 | 1105.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 1084.85 | 1076.11 | 1103.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 10:15:00 | 1084.85 | 1076.11 | 1103.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 1084.85 | 1076.11 | 1103.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 1084.85 | 1076.11 | 1103.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 1124.75 | 1085.83 | 1105.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:30:00 | 1125.00 | 1085.83 | 1105.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 1123.80 | 1093.43 | 1107.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:15:00 | 1128.25 | 1093.43 | 1107.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 1171.60 | 1123.26 | 1118.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 10:15:00 | 1192.00 | 1137.01 | 1124.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 09:15:00 | 1351.45 | 1355.84 | 1311.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 09:30:00 | 1348.95 | 1355.84 | 1311.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 11:15:00 | 1366.65 | 1369.98 | 1356.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 11:45:00 | 1363.60 | 1369.98 | 1356.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 12:15:00 | 1365.80 | 1369.14 | 1357.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 12:30:00 | 1357.45 | 1369.14 | 1357.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 15:15:00 | 1359.80 | 1366.46 | 1359.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 09:15:00 | 1369.65 | 1366.46 | 1359.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 1372.10 | 1367.59 | 1360.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 13:00:00 | 1414.30 | 1376.18 | 1369.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 10:15:00 | 1398.00 | 1409.95 | 1411.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 10:15:00 | 1398.00 | 1409.95 | 1411.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 11:15:00 | 1396.50 | 1407.26 | 1409.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 09:15:00 | 1421.00 | 1403.94 | 1406.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 09:15:00 | 1421.00 | 1403.94 | 1406.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 1421.00 | 1403.94 | 1406.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:45:00 | 1427.35 | 1403.94 | 1406.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2024-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 10:15:00 | 1444.65 | 1412.08 | 1409.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 11:15:00 | 1459.00 | 1421.47 | 1414.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 13:15:00 | 1416.05 | 1424.64 | 1417.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 13:15:00 | 1416.05 | 1424.64 | 1417.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 1416.05 | 1424.64 | 1417.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:45:00 | 1419.25 | 1424.64 | 1417.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 1437.60 | 1427.23 | 1419.10 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 09:15:00 | 1413.25 | 1420.49 | 1420.61 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 10:15:00 | 1437.75 | 1423.94 | 1422.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 09:15:00 | 1451.00 | 1436.96 | 1431.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 09:15:00 | 1437.50 | 1445.80 | 1440.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 09:15:00 | 1437.50 | 1445.80 | 1440.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 1437.50 | 1445.80 | 1440.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:00:00 | 1437.50 | 1445.80 | 1440.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 1437.50 | 1444.14 | 1440.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:30:00 | 1437.00 | 1444.14 | 1440.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 1441.10 | 1443.53 | 1440.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 11:30:00 | 1438.20 | 1443.53 | 1440.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 12:15:00 | 1434.10 | 1441.64 | 1439.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 12:45:00 | 1433.50 | 1441.64 | 1439.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 13:15:00 | 1432.90 | 1439.90 | 1438.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 14:00:00 | 1432.90 | 1439.90 | 1438.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2024-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 14:15:00 | 1431.30 | 1438.18 | 1438.28 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 10:15:00 | 1445.00 | 1439.11 | 1438.63 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 11:15:00 | 1423.90 | 1438.50 | 1439.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 15:15:00 | 1419.00 | 1431.53 | 1435.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-12 09:15:00 | 1391.95 | 1369.60 | 1379.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 09:15:00 | 1391.95 | 1369.60 | 1379.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 1391.95 | 1369.60 | 1379.02 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 11:15:00 | 1425.20 | 1389.59 | 1387.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-18 11:15:00 | 1439.70 | 1429.91 | 1422.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 15:15:00 | 1434.00 | 1434.74 | 1427.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-19 09:15:00 | 1436.00 | 1434.74 | 1427.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 1435.00 | 1434.79 | 1428.44 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2024-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 11:15:00 | 1386.75 | 1417.34 | 1421.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-22 13:15:00 | 1329.45 | 1371.50 | 1392.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 1314.90 | 1300.28 | 1331.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 1314.90 | 1300.28 | 1331.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 1314.90 | 1300.28 | 1331.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 14:45:00 | 1287.75 | 1308.73 | 1318.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 15:15:00 | 1290.00 | 1308.73 | 1318.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 12:15:00 | 1290.05 | 1301.08 | 1306.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 12:45:00 | 1289.70 | 1299.46 | 1305.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 1289.90 | 1293.36 | 1300.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-30 15:00:00 | 1289.45 | 1293.83 | 1298.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-31 09:30:00 | 1285.70 | 1291.96 | 1296.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 12:15:00 | 1225.50 | 1247.83 | 1266.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 12:15:00 | 1225.55 | 1247.83 | 1266.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 12:15:00 | 1225.21 | 1247.83 | 1266.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 12:15:00 | 1224.98 | 1247.83 | 1266.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-01 14:15:00 | 1223.36 | 1241.73 | 1259.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-02 09:15:00 | 1221.41 | 1233.92 | 1253.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-08-05 09:15:00 | 1158.98 | 1200.97 | 1225.31 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 20 — BUY (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 11:15:00 | 1256.80 | 1191.16 | 1188.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 12:15:00 | 1283.25 | 1209.58 | 1197.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 13:15:00 | 1244.90 | 1253.31 | 1233.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 13:30:00 | 1250.30 | 1253.31 | 1233.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 1234.35 | 1249.52 | 1233.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 15:00:00 | 1234.35 | 1249.52 | 1233.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 1240.85 | 1247.78 | 1234.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 1254.05 | 1247.78 | 1234.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 09:15:00 | 1225.00 | 1243.23 | 1233.37 | SL hit (close<static) qty=1.00 sl=1230.65 alert=retest2 |

### Cycle 21 — SELL (started 2024-08-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 14:15:00 | 1204.45 | 1228.29 | 1229.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 09:15:00 | 1203.20 | 1219.86 | 1225.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 10:15:00 | 1253.85 | 1226.66 | 1227.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 10:15:00 | 1253.85 | 1226.66 | 1227.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 1253.85 | 1226.66 | 1227.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 11:00:00 | 1253.85 | 1226.66 | 1227.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 11:15:00 | 1234.60 | 1228.25 | 1228.44 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2024-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 12:15:00 | 1243.10 | 1231.22 | 1229.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 15:15:00 | 1257.00 | 1239.53 | 1234.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 09:15:00 | 1232.00 | 1238.02 | 1234.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 09:15:00 | 1232.00 | 1238.02 | 1234.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 1232.00 | 1238.02 | 1234.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:00:00 | 1232.00 | 1238.02 | 1234.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 1239.05 | 1238.23 | 1234.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 11:15:00 | 1247.95 | 1238.23 | 1234.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 13:15:00 | 1218.00 | 1232.41 | 1232.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2024-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 13:15:00 | 1218.00 | 1232.41 | 1232.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 14:15:00 | 1212.95 | 1228.52 | 1230.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 1212.50 | 1208.91 | 1216.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 1212.50 | 1208.91 | 1216.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 1212.50 | 1208.91 | 1216.00 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2024-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 12:15:00 | 1223.95 | 1216.66 | 1216.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 13:15:00 | 1228.90 | 1221.25 | 1219.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 1294.00 | 1298.78 | 1280.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 09:30:00 | 1288.40 | 1298.78 | 1280.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 1287.55 | 1297.24 | 1286.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:45:00 | 1288.00 | 1297.24 | 1286.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 1287.00 | 1295.19 | 1286.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 1281.80 | 1295.19 | 1286.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 1285.50 | 1293.25 | 1286.53 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2024-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 14:15:00 | 1262.00 | 1282.19 | 1283.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 09:15:00 | 1251.10 | 1273.22 | 1278.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 12:15:00 | 1274.00 | 1268.00 | 1274.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 12:15:00 | 1274.00 | 1268.00 | 1274.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 1274.00 | 1268.00 | 1274.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 13:00:00 | 1274.00 | 1268.00 | 1274.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 13:15:00 | 1310.05 | 1276.41 | 1277.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 13:30:00 | 1317.90 | 1276.41 | 1277.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2024-08-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 14:15:00 | 1328.90 | 1286.91 | 1282.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 15:15:00 | 1344.00 | 1298.33 | 1287.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 14:15:00 | 1315.00 | 1315.14 | 1302.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 14:15:00 | 1315.00 | 1315.14 | 1302.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 1315.00 | 1315.14 | 1302.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 14:45:00 | 1302.95 | 1315.14 | 1302.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 1297.00 | 1310.85 | 1303.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 10:15:00 | 1340.15 | 1310.98 | 1305.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 11:15:00 | 1342.05 | 1315.72 | 1308.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 12:15:00 | 1338.25 | 1319.51 | 1310.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 10:30:00 | 1339.60 | 1327.40 | 1318.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 1324.20 | 1326.56 | 1319.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:30:00 | 1325.50 | 1326.56 | 1319.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 13:15:00 | 1336.00 | 1328.45 | 1321.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 14:15:00 | 1337.75 | 1328.45 | 1321.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 09:45:00 | 1337.00 | 1333.30 | 1325.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 13:15:00 | 1339.50 | 1333.62 | 1327.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 10:15:00 | 1307.50 | 1328.06 | 1327.61 | SL hit (close<static) qty=1.00 sl=1318.00 alert=retest2 |

### Cycle 27 — SELL (started 2024-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 11:15:00 | 1313.75 | 1325.20 | 1326.35 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2024-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 13:15:00 | 1330.00 | 1324.20 | 1323.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 14:15:00 | 1339.05 | 1327.17 | 1324.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 1326.30 | 1328.95 | 1326.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 09:15:00 | 1326.30 | 1328.95 | 1326.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 1326.30 | 1328.95 | 1326.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 1326.30 | 1328.95 | 1326.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 1340.95 | 1331.35 | 1327.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 11:30:00 | 1345.75 | 1333.77 | 1329.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 12:15:00 | 1325.10 | 1332.04 | 1328.67 | SL hit (close<static) qty=1.00 sl=1325.40 alert=retest2 |

### Cycle 29 — SELL (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 09:15:00 | 1285.85 | 1318.86 | 1323.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 12:15:00 | 1276.85 | 1300.00 | 1312.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 10:15:00 | 1298.95 | 1295.06 | 1304.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 10:45:00 | 1299.70 | 1295.06 | 1304.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 1279.00 | 1274.04 | 1283.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 14:30:00 | 1259.00 | 1270.18 | 1277.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 10:30:00 | 1258.20 | 1266.68 | 1269.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 11:00:00 | 1257.05 | 1266.68 | 1269.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 11:30:00 | 1258.65 | 1263.86 | 1268.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 13:15:00 | 1267.00 | 1262.33 | 1266.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 14:00:00 | 1267.00 | 1262.33 | 1266.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 1261.95 | 1262.25 | 1266.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 15:15:00 | 1254.60 | 1262.25 | 1266.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 12:15:00 | 1196.05 | 1223.39 | 1238.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 13:15:00 | 1195.29 | 1220.55 | 1235.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 13:15:00 | 1194.20 | 1220.55 | 1235.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 13:15:00 | 1195.72 | 1220.55 | 1235.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-19 14:15:00 | 1232.55 | 1222.95 | 1235.13 | SL hit (close>ema200) qty=0.50 sl=1222.95 alert=retest2 |

### Cycle 30 — BUY (started 2024-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 11:15:00 | 1247.70 | 1236.68 | 1235.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 14:15:00 | 1263.55 | 1243.27 | 1239.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 12:15:00 | 1231.00 | 1247.06 | 1243.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 12:15:00 | 1231.00 | 1247.06 | 1243.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 1231.00 | 1247.06 | 1243.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 13:00:00 | 1231.00 | 1247.06 | 1243.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 1234.75 | 1244.60 | 1243.06 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2024-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 14:15:00 | 1229.90 | 1241.66 | 1241.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 09:15:00 | 1224.00 | 1236.26 | 1239.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 10:15:00 | 1223.95 | 1223.00 | 1229.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-27 11:00:00 | 1223.95 | 1223.00 | 1229.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 1190.00 | 1163.15 | 1176.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:00:00 | 1190.00 | 1163.15 | 1176.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 1197.00 | 1169.92 | 1178.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 13:15:00 | 1195.40 | 1169.92 | 1178.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 1160.05 | 1175.15 | 1178.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:30:00 | 1137.00 | 1165.01 | 1173.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 10:45:00 | 1146.25 | 1139.77 | 1153.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 14:15:00 | 1192.35 | 1158.36 | 1158.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 14:15:00 | 1192.35 | 1158.36 | 1158.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 1200.20 | 1171.15 | 1164.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 14:15:00 | 1174.55 | 1180.60 | 1172.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 14:15:00 | 1174.55 | 1180.60 | 1172.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 14:15:00 | 1174.55 | 1180.60 | 1172.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 15:00:00 | 1174.55 | 1180.60 | 1172.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 15:15:00 | 1188.00 | 1182.08 | 1174.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 09:15:00 | 1193.05 | 1182.08 | 1174.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 15:00:00 | 1194.00 | 1193.48 | 1184.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:15:00 | 1191.00 | 1192.86 | 1189.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 12:00:00 | 1189.40 | 1194.00 | 1192.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 1196.00 | 1194.19 | 1193.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 13:30:00 | 1198.45 | 1194.19 | 1193.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 1189.90 | 1193.33 | 1192.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 14:30:00 | 1189.05 | 1193.33 | 1192.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 15:15:00 | 1189.95 | 1192.66 | 1192.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 09:15:00 | 1194.95 | 1192.66 | 1192.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 09:15:00 | 1174.35 | 1188.99 | 1190.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2024-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 09:15:00 | 1174.35 | 1188.99 | 1190.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 11:15:00 | 1168.60 | 1182.98 | 1187.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 09:15:00 | 1198.40 | 1148.48 | 1158.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 09:15:00 | 1198.40 | 1148.48 | 1158.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 1198.40 | 1148.48 | 1158.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 10:00:00 | 1198.40 | 1148.48 | 1158.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 1191.95 | 1157.18 | 1161.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 10:45:00 | 1196.00 | 1157.18 | 1161.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2024-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 11:15:00 | 1193.45 | 1164.43 | 1164.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-21 09:15:00 | 1348.10 | 1213.12 | 1188.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 12:15:00 | 1272.05 | 1288.48 | 1257.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-22 13:00:00 | 1272.05 | 1288.48 | 1257.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 13:15:00 | 1282.00 | 1287.19 | 1259.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-23 09:15:00 | 1295.50 | 1281.46 | 1261.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-23 09:45:00 | 1297.00 | 1285.77 | 1265.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-25 09:15:00 | 1256.25 | 1294.54 | 1293.60 | SL hit (close<static) qty=1.00 sl=1258.70 alert=retest2 |

### Cycle 35 — SELL (started 2024-10-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 10:15:00 | 1246.10 | 1284.85 | 1289.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 10:15:00 | 1224.35 | 1245.86 | 1258.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 14:15:00 | 1248.95 | 1241.83 | 1252.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-29 15:00:00 | 1248.95 | 1241.83 | 1252.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 1270.00 | 1247.80 | 1253.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:45:00 | 1267.60 | 1247.80 | 1253.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 10:15:00 | 1270.25 | 1252.29 | 1254.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 10:45:00 | 1273.35 | 1252.29 | 1254.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2024-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 12:15:00 | 1264.90 | 1256.55 | 1256.36 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 14:15:00 | 1252.75 | 1255.59 | 1255.94 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2024-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 09:15:00 | 1260.00 | 1256.22 | 1256.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 10:15:00 | 1267.40 | 1258.45 | 1257.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 10:15:00 | 1308.70 | 1321.78 | 1298.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 11:00:00 | 1308.70 | 1321.78 | 1298.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 15:15:00 | 1312.00 | 1319.08 | 1306.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:15:00 | 1317.95 | 1319.08 | 1306.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 1306.00 | 1316.46 | 1306.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:15:00 | 1303.80 | 1316.46 | 1306.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 1306.20 | 1314.41 | 1306.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:45:00 | 1303.25 | 1314.41 | 1306.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 11:15:00 | 1295.20 | 1310.57 | 1305.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 11:30:00 | 1291.50 | 1310.57 | 1305.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 12:15:00 | 1302.90 | 1309.04 | 1304.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 13:15:00 | 1312.65 | 1309.04 | 1304.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-08 09:15:00 | 1443.92 | 1385.64 | 1364.79 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2024-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 14:15:00 | 1348.30 | 1366.14 | 1368.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 12:15:00 | 1327.25 | 1347.46 | 1357.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 12:15:00 | 1269.90 | 1265.14 | 1289.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 13:00:00 | 1269.90 | 1265.14 | 1289.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 12:15:00 | 1284.80 | 1268.72 | 1279.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:30:00 | 1285.55 | 1268.72 | 1279.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 1288.00 | 1272.58 | 1279.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 13:45:00 | 1289.10 | 1272.58 | 1279.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 14:15:00 | 1278.55 | 1273.77 | 1279.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 15:15:00 | 1280.00 | 1273.77 | 1279.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 15:15:00 | 1280.00 | 1275.02 | 1279.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:15:00 | 1309.30 | 1275.02 | 1279.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 1309.20 | 1281.85 | 1282.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 1309.20 | 1281.85 | 1282.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 1328.00 | 1291.08 | 1286.59 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2024-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 10:15:00 | 1277.60 | 1288.34 | 1289.35 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 1304.25 | 1285.70 | 1284.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 13:15:00 | 1336.20 | 1315.69 | 1304.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 15:15:00 | 1342.00 | 1343.15 | 1329.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 09:15:00 | 1342.75 | 1343.15 | 1329.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 1354.10 | 1345.34 | 1331.58 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2024-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 11:15:00 | 1319.20 | 1329.22 | 1330.23 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 10:15:00 | 1347.15 | 1329.39 | 1329.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 09:15:00 | 1357.85 | 1346.95 | 1339.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 12:15:00 | 1343.20 | 1346.91 | 1341.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 12:15:00 | 1343.20 | 1346.91 | 1341.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 12:15:00 | 1343.20 | 1346.91 | 1341.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 09:15:00 | 1351.00 | 1343.12 | 1340.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-04 11:15:00 | 1340.65 | 1346.06 | 1343.00 | SL hit (close<static) qty=1.00 sl=1340.75 alert=retest2 |

### Cycle 45 — SELL (started 2024-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 15:15:00 | 1339.00 | 1342.74 | 1342.91 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 09:15:00 | 1352.90 | 1344.77 | 1343.82 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2024-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 13:15:00 | 1338.00 | 1345.64 | 1346.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 09:15:00 | 1329.95 | 1340.74 | 1343.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 10:15:00 | 1329.40 | 1327.76 | 1334.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-11 11:00:00 | 1329.40 | 1327.76 | 1334.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 1292.00 | 1316.84 | 1325.72 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2024-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 14:15:00 | 1325.40 | 1318.70 | 1317.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 09:15:00 | 1332.00 | 1322.53 | 1319.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-16 10:15:00 | 1312.10 | 1320.44 | 1319.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 10:15:00 | 1312.10 | 1320.44 | 1319.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 1312.10 | 1320.44 | 1319.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:00:00 | 1312.10 | 1320.44 | 1319.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 1312.30 | 1318.81 | 1318.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:45:00 | 1311.60 | 1318.81 | 1318.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2024-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 12:15:00 | 1313.70 | 1317.79 | 1318.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 14:15:00 | 1308.00 | 1315.29 | 1316.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 10:15:00 | 1258.80 | 1257.79 | 1269.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-20 11:00:00 | 1258.80 | 1257.79 | 1269.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 1224.30 | 1206.96 | 1225.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 09:15:00 | 1191.85 | 1202.36 | 1208.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 10:45:00 | 1192.30 | 1198.59 | 1205.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 11:30:00 | 1192.15 | 1196.62 | 1204.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 14:15:00 | 1181.95 | 1178.71 | 1187.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 1185.85 | 1180.14 | 1187.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 09:30:00 | 1172.00 | 1179.63 | 1185.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 10:30:00 | 1175.70 | 1179.75 | 1185.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 11:15:00 | 1178.00 | 1179.75 | 1185.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 13:00:00 | 1176.20 | 1179.31 | 1184.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 13:15:00 | 1179.85 | 1179.42 | 1183.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 13:30:00 | 1186.60 | 1179.42 | 1183.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 14:15:00 | 1179.60 | 1179.46 | 1183.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 14:30:00 | 1183.40 | 1179.46 | 1183.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 1199.20 | 1183.97 | 1184.82 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-02 09:15:00 | 1199.20 | 1183.97 | 1184.82 | SL hit (close>static) qty=1.00 sl=1194.00 alert=retest2 |

### Cycle 50 — BUY (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 10:15:00 | 1202.00 | 1187.58 | 1186.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 11:15:00 | 1210.65 | 1192.19 | 1188.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 12:15:00 | 1206.75 | 1207.37 | 1200.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 13:00:00 | 1206.75 | 1207.37 | 1200.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 1193.15 | 1204.00 | 1199.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 1193.15 | 1204.00 | 1199.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 1196.00 | 1202.40 | 1199.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 1171.35 | 1202.40 | 1199.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 1170.80 | 1196.08 | 1196.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 1154.00 | 1187.66 | 1193.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 1160.60 | 1158.29 | 1173.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 1160.60 | 1158.29 | 1173.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 1160.60 | 1158.29 | 1173.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 1171.10 | 1158.29 | 1173.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 1171.00 | 1162.33 | 1168.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:15:00 | 1158.00 | 1162.33 | 1168.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 1159.90 | 1161.85 | 1167.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 10:15:00 | 1149.00 | 1161.85 | 1167.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 11:00:00 | 1149.85 | 1159.45 | 1166.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 11:30:00 | 1148.00 | 1156.41 | 1164.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 1091.55 | 1106.56 | 1127.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 1092.36 | 1106.56 | 1127.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 1090.60 | 1106.56 | 1127.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 09:15:00 | 1034.10 | 1071.17 | 1097.40 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 52 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 1068.40 | 1052.75 | 1052.39 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2025-01-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 09:15:00 | 1045.50 | 1053.96 | 1054.10 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2025-01-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 15:15:00 | 1060.00 | 1052.75 | 1052.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 09:15:00 | 1075.50 | 1057.30 | 1054.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 1110.05 | 1113.29 | 1091.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 1110.05 | 1113.29 | 1091.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 1091.60 | 1108.95 | 1091.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 1091.60 | 1108.95 | 1091.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 1098.85 | 1106.93 | 1092.03 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 1075.30 | 1085.10 | 1085.36 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 09:15:00 | 1103.45 | 1086.46 | 1084.45 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2025-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 09:15:00 | 1001.55 | 1075.99 | 1082.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 15:15:00 | 988.70 | 1019.67 | 1046.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 886.00 | 875.44 | 914.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 09:45:00 | 887.05 | 875.44 | 914.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 914.25 | 887.20 | 901.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:45:00 | 918.45 | 887.20 | 901.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 908.05 | 891.37 | 902.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 10:30:00 | 913.35 | 891.37 | 902.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 907.00 | 888.67 | 894.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 10:30:00 | 913.10 | 888.67 | 894.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 11:15:00 | 910.00 | 892.94 | 896.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 12:00:00 | 910.00 | 892.94 | 896.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2025-01-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 13:15:00 | 913.85 | 899.55 | 898.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 14:15:00 | 917.05 | 903.05 | 900.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 911.00 | 911.09 | 905.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 911.00 | 911.09 | 905.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 911.00 | 911.09 | 905.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 906.40 | 911.09 | 905.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 906.00 | 910.08 | 905.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 907.25 | 910.08 | 905.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 905.60 | 909.18 | 905.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 14:15:00 | 894.90 | 909.18 | 905.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 14:15:00 | 891.20 | 905.58 | 904.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 14:45:00 | 885.00 | 905.58 | 904.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2025-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 15:15:00 | 884.10 | 901.29 | 902.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 870.15 | 895.06 | 899.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 877.00 | 867.09 | 879.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 877.00 | 867.09 | 879.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 877.00 | 867.09 | 879.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:45:00 | 885.50 | 867.09 | 879.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 874.20 | 868.52 | 879.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 12:30:00 | 870.95 | 870.10 | 878.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-05 09:15:00 | 903.85 | 880.15 | 880.49 | SL hit (close>static) qty=1.00 sl=884.90 alert=retest2 |

### Cycle 60 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 10:15:00 | 901.80 | 884.48 | 882.43 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2025-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 12:15:00 | 877.00 | 883.25 | 883.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 13:15:00 | 871.95 | 880.99 | 882.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 10:15:00 | 880.05 | 877.88 | 880.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 10:15:00 | 880.05 | 877.88 | 880.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 880.05 | 877.88 | 880.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:00:00 | 880.05 | 877.88 | 880.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 875.05 | 877.32 | 879.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 12:15:00 | 874.90 | 877.32 | 879.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 12:45:00 | 874.80 | 876.74 | 879.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 13:45:00 | 871.80 | 874.58 | 878.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:30:00 | 863.30 | 870.07 | 875.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 831.15 | 851.37 | 862.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 831.06 | 851.37 | 862.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 828.21 | 851.37 | 862.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 820.13 | 851.37 | 862.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-12 09:15:00 | 787.41 | 822.46 | 840.21 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 62 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 808.65 | 788.56 | 786.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 13:15:00 | 822.20 | 810.71 | 801.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 10:15:00 | 809.75 | 813.59 | 806.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 10:15:00 | 809.75 | 813.59 | 806.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 809.75 | 813.59 | 806.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 809.75 | 813.59 | 806.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 811.60 | 813.19 | 806.60 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 759.10 | 797.44 | 801.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 10:15:00 | 751.25 | 764.13 | 773.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 711.80 | 706.20 | 724.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 711.80 | 706.20 | 724.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 737.85 | 717.08 | 724.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:45:00 | 744.15 | 717.08 | 724.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 729.30 | 719.52 | 724.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:15:00 | 728.05 | 719.52 | 724.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 14:30:00 | 727.80 | 723.22 | 724.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 736.15 | 727.33 | 726.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2025-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 09:15:00 | 736.15 | 727.33 | 726.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 746.95 | 736.21 | 732.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 14:15:00 | 736.50 | 738.62 | 735.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 15:00:00 | 736.50 | 738.62 | 735.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 731.40 | 737.18 | 734.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:15:00 | 743.10 | 737.18 | 734.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 740.15 | 737.77 | 735.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 11:00:00 | 752.80 | 740.78 | 736.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 09:15:00 | 722.85 | 735.78 | 736.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2025-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 09:15:00 | 722.85 | 735.78 | 736.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 12:15:00 | 715.50 | 727.03 | 731.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 665.50 | 658.54 | 667.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-18 10:00:00 | 665.50 | 658.54 | 667.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 662.55 | 659.34 | 667.14 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 681.45 | 669.92 | 669.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 10:15:00 | 696.80 | 675.30 | 672.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 15:15:00 | 707.95 | 709.22 | 699.21 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:15:00 | 774.50 | 709.22 | 699.21 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 809.45 | 729.27 | 709.23 | EMA400 retest candle locked (from upside) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 09:15:00 | 813.23 | 729.27 | 709.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 10:30:00 | 823.80 | 747.81 | 719.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 11:00:00 | 822.00 | 747.81 | 719.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 12:15:00 | 821.15 | 761.93 | 728.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 10:00:00 | 823.25 | 792.44 | 757.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 09:15:00 | 810.20 | 817.56 | 791.22 | SL hit (close<ema200) qty=0.50 sl=817.56 alert=retest1 |

### Cycle 67 — SELL (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 11:15:00 | 767.40 | 787.69 | 788.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 13:15:00 | 755.00 | 777.70 | 783.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 09:15:00 | 773.45 | 768.86 | 777.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 09:15:00 | 773.45 | 768.86 | 777.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 773.45 | 768.86 | 777.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:00:00 | 773.45 | 768.86 | 777.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 778.95 | 770.87 | 777.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:45:00 | 781.35 | 770.87 | 777.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 776.00 | 771.90 | 777.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 12:15:00 | 773.00 | 771.90 | 777.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 09:45:00 | 773.00 | 774.18 | 776.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 09:15:00 | 802.70 | 773.06 | 773.89 | SL hit (close>static) qty=1.00 sl=783.00 alert=retest2 |

### Cycle 68 — BUY (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 10:15:00 | 794.95 | 777.43 | 775.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 855.60 | 808.04 | 795.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 841.65 | 855.12 | 832.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-04 10:00:00 | 841.65 | 855.12 | 832.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 843.90 | 852.88 | 833.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 15:15:00 | 851.00 | 843.81 | 834.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 760.95 | 828.39 | 829.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 760.95 | 828.39 | 829.21 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 836.60 | 814.42 | 813.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 858.55 | 836.69 | 826.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 15:15:00 | 862.25 | 862.77 | 853.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 09:15:00 | 847.85 | 862.77 | 853.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 849.05 | 860.03 | 852.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 09:15:00 | 884.40 | 857.61 | 856.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 849.80 | 874.77 | 878.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 849.80 | 874.77 | 878.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 09:15:00 | 768.70 | 843.94 | 860.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 10:15:00 | 713.40 | 711.22 | 735.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-02 10:45:00 | 715.00 | 711.22 | 735.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 725.60 | 702.51 | 711.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:30:00 | 736.25 | 702.51 | 711.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 722.00 | 706.41 | 712.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:30:00 | 712.40 | 708.39 | 712.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 676.78 | 688.41 | 695.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 15:15:00 | 682.75 | 679.56 | 686.66 | SL hit (close>ema200) qty=0.50 sl=679.56 alert=retest2 |

### Cycle 72 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 700.00 | 691.60 | 690.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 705.00 | 696.98 | 693.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 13:15:00 | 709.50 | 710.96 | 706.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 14:00:00 | 709.50 | 710.96 | 706.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 704.50 | 709.18 | 707.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 12:00:00 | 704.50 | 709.18 | 707.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 705.50 | 708.44 | 707.10 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2025-05-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 15:15:00 | 703.50 | 706.14 | 706.28 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2025-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 09:15:00 | 709.45 | 706.80 | 706.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 10:15:00 | 723.50 | 710.14 | 708.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 15:15:00 | 740.65 | 741.96 | 732.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 09:15:00 | 730.00 | 741.96 | 732.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 728.15 | 739.19 | 731.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:00:00 | 728.15 | 739.19 | 731.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 732.50 | 737.86 | 731.93 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 724.00 | 729.50 | 729.64 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 746.55 | 732.91 | 731.18 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 13:15:00 | 739.15 | 741.31 | 741.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 13:15:00 | 736.40 | 739.23 | 740.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 10:15:00 | 709.10 | 707.22 | 712.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 11:00:00 | 709.10 | 707.22 | 712.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 707.15 | 707.21 | 711.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:30:00 | 712.00 | 707.21 | 711.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 710.50 | 707.87 | 711.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:00:00 | 710.50 | 707.87 | 711.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 706.90 | 707.67 | 711.26 | EMA400 retest candle locked (from downside) |

### Cycle 78 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 718.60 | 713.39 | 712.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 09:15:00 | 726.40 | 717.34 | 715.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 12:15:00 | 725.70 | 726.25 | 722.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 13:00:00 | 725.70 | 726.25 | 722.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 13:15:00 | 722.50 | 725.50 | 722.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 14:00:00 | 722.50 | 725.50 | 722.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 722.25 | 724.85 | 722.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 15:15:00 | 722.80 | 724.85 | 722.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 722.80 | 724.44 | 722.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:30:00 | 722.50 | 723.89 | 722.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 721.50 | 723.41 | 722.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:45:00 | 719.80 | 723.41 | 722.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2025-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 12:15:00 | 716.90 | 721.60 | 721.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 13:15:00 | 715.55 | 720.39 | 721.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 09:15:00 | 720.40 | 719.10 | 720.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 720.40 | 719.10 | 720.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 720.40 | 719.10 | 720.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:00:00 | 720.40 | 719.10 | 720.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 718.15 | 718.91 | 720.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 13:00:00 | 716.50 | 718.44 | 719.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 15:15:00 | 706.90 | 693.15 | 692.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2025-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 15:15:00 | 706.90 | 693.15 | 692.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 09:15:00 | 719.90 | 702.32 | 699.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 13:15:00 | 716.70 | 717.16 | 712.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 13:45:00 | 716.55 | 717.16 | 712.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 714.50 | 716.99 | 713.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:00:00 | 714.50 | 716.99 | 713.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 713.85 | 716.36 | 713.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:30:00 | 713.90 | 716.36 | 713.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 714.55 | 716.00 | 713.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 12:30:00 | 715.75 | 715.90 | 713.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 13:00:00 | 715.50 | 715.90 | 713.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 709.20 | 714.28 | 713.62 | SL hit (close<static) qty=1.00 sl=713.10 alert=retest2 |

### Cycle 81 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 708.30 | 713.09 | 713.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 13:15:00 | 706.60 | 710.22 | 711.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 14:15:00 | 708.90 | 707.89 | 709.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 14:15:00 | 708.90 | 707.89 | 709.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 708.90 | 707.89 | 709.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 15:00:00 | 708.90 | 707.89 | 709.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 709.30 | 708.17 | 709.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:15:00 | 703.65 | 708.17 | 709.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 706.00 | 707.74 | 709.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:45:00 | 702.10 | 707.01 | 708.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 12:30:00 | 703.15 | 706.22 | 707.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 13:00:00 | 703.05 | 706.22 | 707.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 14:00:00 | 702.60 | 705.50 | 707.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 703.20 | 705.05 | 706.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 13:45:00 | 701.60 | 703.46 | 705.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:30:00 | 701.05 | 702.12 | 704.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 11:30:00 | 701.15 | 698.88 | 700.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:45:00 | 701.50 | 698.96 | 699.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 701.00 | 699.47 | 699.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:30:00 | 702.00 | 699.47 | 699.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 700.40 | 699.66 | 699.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:15:00 | 701.50 | 699.66 | 699.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-10 14:15:00 | 702.65 | 700.26 | 700.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2025-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 14:15:00 | 702.65 | 700.26 | 700.18 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 09:15:00 | 696.80 | 699.80 | 700.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 13:15:00 | 695.00 | 697.79 | 698.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 697.30 | 696.38 | 697.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 697.30 | 696.38 | 697.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 697.30 | 696.38 | 697.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:00:00 | 697.30 | 696.38 | 697.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 698.75 | 696.86 | 697.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:00:00 | 698.75 | 696.86 | 697.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 697.85 | 697.06 | 697.94 | EMA400 retest candle locked (from downside) |

### Cycle 84 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 703.00 | 698.32 | 698.24 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 09:15:00 | 653.95 | 689.45 | 694.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 14:15:00 | 626.20 | 629.62 | 635.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 15:15:00 | 616.95 | 616.72 | 624.01 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 09:15:00 | 608.85 | 616.72 | 624.01 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 621.00 | 617.58 | 623.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:00:00 | 621.00 | 617.58 | 623.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 627.35 | 619.53 | 624.06 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 627.35 | 619.53 | 624.06 | SL hit (close>ema400) qty=1.00 sl=624.06 alert=retest1 |

### Cycle 86 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 575.80 | 557.85 | 555.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 577.45 | 561.77 | 557.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 13:15:00 | 581.30 | 581.37 | 576.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 14:00:00 | 581.30 | 581.37 | 576.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 611.90 | 618.01 | 609.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:45:00 | 609.70 | 618.01 | 609.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 610.80 | 614.08 | 610.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 15:00:00 | 610.80 | 614.08 | 610.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 607.45 | 612.75 | 610.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 598.60 | 612.75 | 610.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 611.25 | 612.45 | 610.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 593.20 | 612.45 | 610.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 600.55 | 610.07 | 609.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:00:00 | 600.55 | 610.07 | 609.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 600.60 | 608.18 | 608.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 596.30 | 602.02 | 605.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 588.70 | 587.29 | 592.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 588.70 | 587.29 | 592.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 588.70 | 587.29 | 592.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 590.75 | 587.29 | 592.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 586.80 | 587.50 | 590.01 | EMA400 retest candle locked (from downside) |

### Cycle 88 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 617.70 | 593.07 | 591.33 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 590.40 | 599.35 | 600.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 585.80 | 596.64 | 598.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 595.70 | 592.74 | 595.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 595.70 | 592.74 | 595.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 595.70 | 592.74 | 595.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 589.60 | 592.62 | 594.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 14:45:00 | 590.70 | 591.60 | 591.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 10:15:00 | 593.80 | 592.14 | 592.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 10:15:00 | 593.80 | 592.14 | 592.03 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 13:15:00 | 591.00 | 591.99 | 592.00 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 594.60 | 592.16 | 592.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 10:15:00 | 596.00 | 592.93 | 592.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 14:15:00 | 593.30 | 594.02 | 593.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 14:15:00 | 593.30 | 594.02 | 593.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 593.30 | 594.02 | 593.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:45:00 | 593.30 | 594.02 | 593.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 593.05 | 593.82 | 593.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 602.85 | 593.82 | 593.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 15:15:00 | 597.00 | 595.44 | 594.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 608.90 | 612.53 | 612.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 608.90 | 612.53 | 612.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 11:15:00 | 605.50 | 610.17 | 611.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 09:15:00 | 587.65 | 586.05 | 590.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-26 10:00:00 | 587.65 | 586.05 | 590.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 588.45 | 586.53 | 590.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 11:15:00 | 585.20 | 586.53 | 590.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 13:15:00 | 585.30 | 586.64 | 589.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 09:15:00 | 599.30 | 589.32 | 590.01 | SL hit (close>static) qty=1.00 sl=597.40 alert=retest2 |

### Cycle 94 — BUY (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 11:15:00 | 593.80 | 591.00 | 590.70 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2025-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 11:15:00 | 587.90 | 590.64 | 590.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 12:15:00 | 587.00 | 589.92 | 590.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 09:15:00 | 586.95 | 585.11 | 586.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 09:15:00 | 586.95 | 585.11 | 586.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 586.95 | 585.11 | 586.74 | EMA400 retest candle locked (from downside) |

### Cycle 96 — BUY (started 2025-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 15:15:00 | 592.50 | 588.32 | 587.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 09:15:00 | 607.60 | 592.18 | 589.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 09:15:00 | 595.80 | 597.79 | 594.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 10:00:00 | 595.80 | 597.79 | 594.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 594.25 | 597.09 | 594.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 594.25 | 597.09 | 594.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 594.05 | 596.48 | 594.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:15:00 | 592.40 | 596.48 | 594.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 594.00 | 595.98 | 594.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:30:00 | 593.00 | 595.98 | 594.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 593.30 | 595.45 | 594.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:30:00 | 593.70 | 595.45 | 594.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 591.20 | 593.90 | 593.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 591.20 | 593.90 | 593.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 591.10 | 593.34 | 593.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 588.25 | 591.72 | 592.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 11:15:00 | 594.85 | 590.98 | 591.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 11:15:00 | 594.85 | 590.98 | 591.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 594.85 | 590.98 | 591.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 594.85 | 590.98 | 591.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 598.50 | 592.48 | 592.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 13:00:00 | 598.50 | 592.48 | 592.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 13:15:00 | 595.45 | 593.08 | 592.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 11:15:00 | 604.80 | 596.59 | 594.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 14:15:00 | 598.65 | 599.28 | 596.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-10 15:00:00 | 598.65 | 599.28 | 596.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 596.80 | 598.88 | 596.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 594.20 | 598.88 | 596.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 593.25 | 597.75 | 596.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 593.25 | 597.75 | 596.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 594.65 | 597.13 | 596.38 | EMA400 retest candle locked (from upside) |

### Cycle 99 — SELL (started 2025-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 14:15:00 | 592.70 | 595.33 | 595.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 15:15:00 | 591.90 | 594.65 | 595.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 12:15:00 | 588.15 | 586.85 | 589.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 13:00:00 | 588.15 | 586.85 | 589.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 586.40 | 586.76 | 589.02 | EMA400 retest candle locked (from downside) |

### Cycle 100 — BUY (started 2025-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 12:15:00 | 592.50 | 589.45 | 589.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 597.60 | 591.82 | 590.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 591.10 | 592.60 | 591.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 12:15:00 | 591.10 | 592.60 | 591.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 591.10 | 592.60 | 591.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 591.10 | 592.60 | 591.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 589.95 | 592.07 | 591.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 589.30 | 592.07 | 591.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 589.60 | 591.57 | 591.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 15:00:00 | 589.60 | 591.57 | 591.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 589.00 | 591.06 | 590.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:15:00 | 544.50 | 591.06 | 590.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 09:15:00 | 543.70 | 581.59 | 586.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 15:15:00 | 538.90 | 542.25 | 552.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 544.25 | 540.18 | 545.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 544.25 | 540.18 | 545.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 544.25 | 540.18 | 545.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 11:00:00 | 541.95 | 542.91 | 544.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 12:15:00 | 541.50 | 542.80 | 544.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 13:00:00 | 541.90 | 542.62 | 544.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 541.80 | 542.88 | 543.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 542.10 | 542.09 | 543.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:30:00 | 542.35 | 542.09 | 543.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 543.30 | 542.35 | 543.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:00:00 | 543.30 | 542.35 | 543.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 541.35 | 542.15 | 542.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:30:00 | 543.70 | 542.15 | 542.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 543.00 | 542.32 | 542.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 545.45 | 542.32 | 542.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 541.00 | 542.06 | 542.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 10:30:00 | 539.60 | 541.58 | 542.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 11:30:00 | 540.00 | 536.98 | 537.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 13:45:00 | 539.50 | 537.24 | 537.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 556.70 | 540.61 | 539.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 09:15:00 | 556.70 | 540.61 | 539.15 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 10:15:00 | 532.05 | 539.42 | 539.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 11:15:00 | 531.35 | 537.81 | 539.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 15:15:00 | 507.00 | 506.84 | 512.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-12 09:15:00 | 514.50 | 506.84 | 512.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 517.30 | 508.93 | 513.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:45:00 | 518.10 | 508.93 | 513.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 531.55 | 513.45 | 514.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:00:00 | 531.55 | 513.45 | 514.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 11:15:00 | 544.00 | 519.56 | 517.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 14:15:00 | 548.05 | 532.34 | 524.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 11:15:00 | 527.90 | 534.69 | 528.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 11:15:00 | 527.90 | 534.69 | 528.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 527.90 | 534.69 | 528.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:00:00 | 527.90 | 534.69 | 528.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 527.70 | 533.30 | 528.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:45:00 | 526.70 | 533.30 | 528.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 527.25 | 532.09 | 528.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 527.25 | 532.09 | 528.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 525.50 | 530.77 | 528.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 525.50 | 530.77 | 528.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 522.50 | 526.57 | 526.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 514.75 | 520.64 | 522.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 10:15:00 | 516.55 | 513.42 | 515.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 10:15:00 | 516.55 | 513.42 | 515.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 516.55 | 513.42 | 515.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:00:00 | 516.55 | 513.42 | 515.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 515.25 | 513.79 | 515.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:30:00 | 516.70 | 513.79 | 515.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 12:15:00 | 513.20 | 513.67 | 515.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 13:15:00 | 512.10 | 513.67 | 515.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 14:30:00 | 512.00 | 512.65 | 514.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:30:00 | 509.45 | 511.11 | 513.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 14:15:00 | 486.50 | 494.34 | 501.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 14:15:00 | 486.40 | 494.34 | 501.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 14:15:00 | 483.98 | 494.34 | 501.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 481.40 | 480.84 | 488.50 | SL hit (close>ema200) qty=0.50 sl=480.84 alert=retest2 |

### Cycle 106 — BUY (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 10:15:00 | 510.95 | 491.14 | 489.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 14:15:00 | 523.75 | 510.30 | 505.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 510.80 | 512.25 | 507.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-03 10:00:00 | 510.80 | 512.25 | 507.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 509.05 | 510.93 | 508.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 12:30:00 | 508.35 | 510.93 | 508.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 508.70 | 510.48 | 508.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 14:00:00 | 508.70 | 510.48 | 508.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 511.30 | 510.65 | 508.44 | EMA400 retest candle locked (from upside) |

### Cycle 107 — SELL (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 11:15:00 | 504.25 | 507.51 | 507.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 14:15:00 | 502.20 | 505.19 | 506.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 475.60 | 473.47 | 479.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 09:15:00 | 475.60 | 473.47 | 479.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 475.60 | 473.47 | 479.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:30:00 | 471.95 | 472.34 | 478.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 09:15:00 | 495.15 | 472.20 | 472.21 | SL hit (close>static) qty=1.00 sl=482.00 alert=retest2 |

### Cycle 108 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 486.25 | 475.01 | 473.48 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 11:15:00 | 469.70 | 474.43 | 474.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 10:15:00 | 466.90 | 470.60 | 472.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 14:15:00 | 468.90 | 468.72 | 470.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-16 15:00:00 | 468.90 | 468.72 | 470.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 462.60 | 467.52 | 469.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 10:15:00 | 462.00 | 467.52 | 469.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 13:15:00 | 454.30 | 452.76 | 452.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2025-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 13:15:00 | 454.30 | 452.76 | 452.58 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 12:15:00 | 452.00 | 453.33 | 453.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 13:15:00 | 451.75 | 453.02 | 453.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 09:15:00 | 457.75 | 451.22 | 451.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 457.75 | 451.22 | 451.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 457.75 | 451.22 | 451.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:00:00 | 457.75 | 451.22 | 451.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 450.00 | 450.98 | 451.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:15:00 | 447.90 | 450.98 | 451.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 15:15:00 | 456.60 | 452.36 | 451.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 15:15:00 | 456.60 | 452.36 | 451.86 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 11:15:00 | 450.10 | 451.55 | 451.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 12:15:00 | 449.95 | 451.23 | 451.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 451.35 | 451.25 | 451.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 13:15:00 | 451.35 | 451.25 | 451.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 451.35 | 451.25 | 451.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 451.35 | 451.25 | 451.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 451.25 | 451.25 | 451.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 451.25 | 451.25 | 451.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 452.00 | 451.40 | 451.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 456.60 | 451.40 | 451.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 453.60 | 451.84 | 451.67 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 14:15:00 | 450.00 | 451.50 | 451.62 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 453.60 | 451.81 | 451.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 15:15:00 | 454.50 | 453.23 | 452.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 451.65 | 452.92 | 452.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 451.65 | 452.92 | 452.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 451.65 | 452.92 | 452.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:00:00 | 451.65 | 452.92 | 452.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 450.40 | 452.41 | 452.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:30:00 | 450.10 | 452.41 | 452.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 12:15:00 | 450.00 | 451.70 | 451.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 13:15:00 | 449.30 | 451.22 | 451.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 10:15:00 | 446.80 | 446.23 | 448.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 10:15:00 | 446.80 | 446.23 | 448.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 446.80 | 446.23 | 448.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 446.80 | 446.23 | 448.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 449.80 | 446.97 | 447.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:00:00 | 446.50 | 447.40 | 447.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:45:00 | 444.20 | 446.84 | 447.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 10:15:00 | 424.17 | 437.96 | 442.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 10:15:00 | 421.99 | 437.96 | 442.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-12 09:15:00 | 401.85 | 415.93 | 428.70 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 118 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 333.35 | 313.44 | 313.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 344.75 | 336.58 | 330.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 334.60 | 337.45 | 332.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 12:15:00 | 334.60 | 337.45 | 332.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 334.60 | 337.45 | 332.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 331.95 | 337.45 | 332.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 331.25 | 336.21 | 332.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 331.25 | 336.21 | 332.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 331.30 | 335.23 | 332.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 331.30 | 335.23 | 332.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 331.50 | 334.48 | 332.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 325.00 | 334.48 | 332.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 332.50 | 334.09 | 332.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:30:00 | 329.10 | 334.09 | 332.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 322.75 | 331.82 | 331.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 322.75 | 331.82 | 331.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 321.55 | 329.76 | 330.63 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 340.85 | 331.76 | 330.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 14:15:00 | 347.30 | 341.66 | 337.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 12:15:00 | 344.80 | 345.25 | 341.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 13:15:00 | 343.00 | 345.25 | 341.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 339.85 | 343.65 | 341.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 15:00:00 | 339.85 | 343.65 | 341.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 339.00 | 342.72 | 341.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 332.40 | 342.72 | 341.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 328.70 | 339.92 | 339.98 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 351.00 | 340.38 | 339.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 353.20 | 342.94 | 340.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 14:15:00 | 353.15 | 354.43 | 349.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 14:45:00 | 353.00 | 354.43 | 349.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 350.45 | 353.64 | 350.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:15:00 | 346.00 | 353.64 | 350.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 342.75 | 351.46 | 349.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 342.75 | 351.46 | 349.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 338.30 | 348.83 | 348.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 338.30 | 348.83 | 348.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 11:15:00 | 337.80 | 346.62 | 347.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 334.40 | 340.49 | 343.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 10:15:00 | 332.20 | 330.38 | 333.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 12:15:00 | 331.70 | 330.93 | 333.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 331.70 | 330.93 | 333.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 12:30:00 | 332.00 | 330.93 | 333.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 331.00 | 330.99 | 332.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:30:00 | 332.10 | 330.99 | 332.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 345.05 | 333.81 | 333.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 346.15 | 333.81 | 333.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 10:15:00 | 339.20 | 334.89 | 334.35 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 09:15:00 | 329.85 | 334.71 | 334.81 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 15:15:00 | 336.75 | 334.45 | 334.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 09:15:00 | 337.95 | 335.15 | 334.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 10:15:00 | 333.90 | 334.90 | 334.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 10:15:00 | 333.90 | 334.90 | 334.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 333.90 | 334.90 | 334.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 333.90 | 334.90 | 334.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 332.40 | 334.40 | 334.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 331.30 | 333.46 | 334.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 11:15:00 | 329.30 | 327.42 | 329.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 11:15:00 | 329.30 | 327.42 | 329.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 329.30 | 327.42 | 329.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:00:00 | 329.30 | 327.42 | 329.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 328.10 | 327.55 | 329.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 14:00:00 | 327.30 | 327.50 | 328.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 14:45:00 | 324.35 | 327.48 | 328.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 15:15:00 | 325.70 | 327.48 | 328.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 345.40 | 323.21 | 322.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 345.40 | 323.21 | 322.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 11:15:00 | 357.90 | 333.48 | 327.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 11:15:00 | 478.70 | 481.76 | 448.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-04 12:00:00 | 478.70 | 481.76 | 448.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 481.20 | 492.03 | 479.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 480.35 | 492.03 | 479.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 474.10 | 488.44 | 479.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:00:00 | 474.10 | 488.44 | 479.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 474.55 | 485.66 | 478.89 | EMA400 retest candle locked (from upside) |

### Cycle 129 — SELL (started 2026-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 15:15:00 | 461.90 | 475.20 | 475.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 428.90 | 465.94 | 471.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 459.25 | 440.71 | 452.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 459.25 | 440.71 | 452.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 459.25 | 440.71 | 452.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:30:00 | 461.95 | 440.71 | 452.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 458.75 | 444.32 | 452.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 11:45:00 | 454.50 | 445.95 | 452.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 13:15:00 | 453.70 | 447.74 | 452.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-10 14:15:00 | 462.90 | 451.34 | 453.70 | SL hit (close>static) qty=1.00 sl=462.80 alert=retest2 |

### Cycle 130 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 483.75 | 460.60 | 457.68 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2026-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 11:15:00 | 444.00 | 459.12 | 460.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 428.30 | 445.04 | 452.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 09:15:00 | 440.15 | 433.42 | 441.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 09:15:00 | 440.15 | 433.42 | 441.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 440.15 | 433.42 | 441.34 | EMA400 retest candle locked (from downside) |

### Cycle 132 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 445.05 | 442.34 | 442.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 452.70 | 444.41 | 443.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 15:15:00 | 448.00 | 448.27 | 445.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 09:15:00 | 453.45 | 448.27 | 445.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 446.30 | 447.87 | 445.88 | EMA400 retest candle locked (from upside) |

### Cycle 133 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 436.05 | 443.51 | 444.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 431.50 | 441.10 | 443.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 14:15:00 | 434.95 | 434.84 | 438.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 14:45:00 | 433.30 | 434.84 | 438.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 436.00 | 435.08 | 438.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 416.70 | 435.08 | 438.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 11:15:00 | 426.85 | 423.36 | 423.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 426.85 | 423.36 | 423.26 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 406.50 | 420.24 | 421.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 403.95 | 414.68 | 419.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 15:15:00 | 415.00 | 411.63 | 415.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 15:15:00 | 415.00 | 411.63 | 415.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 415.00 | 411.63 | 415.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 09:15:00 | 403.10 | 411.63 | 415.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 15:15:00 | 382.94 | 395.52 | 404.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 409.00 | 398.21 | 404.93 | SL hit (close>ema200) qty=0.50 sl=398.21 alert=retest2 |

### Cycle 136 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 421.45 | 409.89 | 408.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 11:15:00 | 422.00 | 417.39 | 414.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 420.65 | 422.01 | 418.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 420.65 | 422.01 | 418.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 420.65 | 422.01 | 418.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 432.25 | 419.78 | 418.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 10:00:00 | 429.45 | 444.55 | 443.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 10:15:00 | 431.30 | 441.90 | 442.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 10:15:00 | 431.30 | 441.90 | 442.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-17 10:15:00 | 424.50 | 432.09 | 436.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 09:15:00 | 408.40 | 408.34 | 411.96 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:15:00 | 404.00 | 408.17 | 411.56 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 403.15 | 404.13 | 407.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 10:15:00 | 408.65 | 404.13 | 407.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 415.90 | 406.49 | 408.47 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-24 10:15:00 | 415.90 | 406.49 | 408.47 | SL hit (close>ema400) qty=1.00 sl=408.47 alert=retest1 |

### Cycle 138 — BUY (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 15:15:00 | 412.75 | 409.76 | 409.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 10:15:00 | 413.60 | 411.11 | 410.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 09:15:00 | 411.80 | 412.55 | 411.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 09:15:00 | 411.80 | 412.55 | 411.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 411.80 | 412.55 | 411.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:30:00 | 413.05 | 412.55 | 411.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 414.50 | 412.94 | 411.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 10:45:00 | 423.00 | 415.23 | 413.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 14:00:00 | 417.40 | 417.03 | 414.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 15:15:00 | 419.20 | 416.83 | 414.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 409.40 | 415.72 | 414.72 | SL hit (close<static) qty=1.00 sl=411.35 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-15 12:30:00 | 1188.25 | 2024-05-21 10:15:00 | 1183.00 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2024-05-21 10:15:00 | 1185.20 | 2024-05-21 10:15:00 | 1183.00 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2024-06-19 13:00:00 | 1414.30 | 2024-06-26 10:15:00 | 1398.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-07-25 14:45:00 | 1287.75 | 2024-08-01 12:15:00 | 1225.50 | PARTIAL | 0.50 | 4.83% |
| SELL | retest2 | 2024-07-25 15:15:00 | 1290.00 | 2024-08-01 12:15:00 | 1225.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-29 12:15:00 | 1290.05 | 2024-08-01 12:15:00 | 1225.21 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2024-07-29 12:45:00 | 1289.70 | 2024-08-01 12:15:00 | 1224.98 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2024-07-30 15:00:00 | 1289.45 | 2024-08-01 14:15:00 | 1223.36 | PARTIAL | 0.50 | 5.13% |
| SELL | retest2 | 2024-07-31 09:30:00 | 1285.70 | 2024-08-02 09:15:00 | 1221.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-25 14:45:00 | 1287.75 | 2024-08-05 09:15:00 | 1158.98 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-25 15:15:00 | 1290.00 | 2024-08-05 09:15:00 | 1161.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-29 12:15:00 | 1290.05 | 2024-08-05 09:15:00 | 1161.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-29 12:45:00 | 1289.70 | 2024-08-05 09:15:00 | 1160.73 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-30 15:00:00 | 1289.45 | 2024-08-05 09:15:00 | 1160.51 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-31 09:30:00 | 1285.70 | 2024-08-05 09:15:00 | 1157.13 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-08-09 09:15:00 | 1254.05 | 2024-08-09 09:15:00 | 1225.00 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2024-08-13 11:15:00 | 1247.95 | 2024-08-13 13:15:00 | 1218.00 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2024-08-30 10:15:00 | 1340.15 | 2024-09-04 10:15:00 | 1307.50 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2024-08-30 11:15:00 | 1342.05 | 2024-09-04 10:15:00 | 1307.50 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2024-08-30 12:15:00 | 1338.25 | 2024-09-04 10:15:00 | 1307.50 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2024-09-02 10:30:00 | 1339.60 | 2024-09-04 11:15:00 | 1313.75 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-09-02 14:15:00 | 1337.75 | 2024-09-04 11:15:00 | 1313.75 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-09-03 09:45:00 | 1337.00 | 2024-09-04 11:15:00 | 1313.75 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2024-09-03 13:15:00 | 1339.50 | 2024-09-04 11:15:00 | 1313.75 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2024-09-06 11:30:00 | 1345.75 | 2024-09-06 12:15:00 | 1325.10 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-09-12 14:30:00 | 1259.00 | 2024-09-19 12:15:00 | 1196.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-17 10:30:00 | 1258.20 | 2024-09-19 13:15:00 | 1195.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-17 11:00:00 | 1257.05 | 2024-09-19 13:15:00 | 1194.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-17 11:30:00 | 1258.65 | 2024-09-19 13:15:00 | 1195.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-12 14:30:00 | 1259.00 | 2024-09-19 14:15:00 | 1232.55 | STOP_HIT | 0.50 | 2.10% |
| SELL | retest2 | 2024-09-17 10:30:00 | 1258.20 | 2024-09-19 14:15:00 | 1232.55 | STOP_HIT | 0.50 | 2.04% |
| SELL | retest2 | 2024-09-17 11:00:00 | 1257.05 | 2024-09-19 14:15:00 | 1232.55 | STOP_HIT | 0.50 | 1.95% |
| SELL | retest2 | 2024-09-17 11:30:00 | 1258.65 | 2024-09-19 14:15:00 | 1232.55 | STOP_HIT | 0.50 | 2.07% |
| SELL | retest2 | 2024-09-17 15:15:00 | 1254.60 | 2024-09-23 11:15:00 | 1247.70 | STOP_HIT | 1.00 | 0.55% |
| SELL | retest2 | 2024-09-23 11:00:00 | 1258.10 | 2024-09-23 11:15:00 | 1247.70 | STOP_HIT | 1.00 | 0.83% |
| SELL | retest2 | 2024-10-07 10:30:00 | 1137.00 | 2024-10-08 14:15:00 | 1192.35 | STOP_HIT | 1.00 | -4.87% |
| SELL | retest2 | 2024-10-08 10:45:00 | 1146.25 | 2024-10-08 14:15:00 | 1192.35 | STOP_HIT | 1.00 | -4.02% |
| BUY | retest2 | 2024-10-10 09:15:00 | 1193.05 | 2024-10-16 09:15:00 | 1174.35 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2024-10-10 15:00:00 | 1194.00 | 2024-10-16 09:15:00 | 1174.35 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-10-14 09:15:00 | 1191.00 | 2024-10-16 09:15:00 | 1174.35 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-10-15 12:00:00 | 1189.40 | 2024-10-16 09:15:00 | 1174.35 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-10-16 09:15:00 | 1194.95 | 2024-10-16 09:15:00 | 1174.35 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-10-23 09:15:00 | 1295.50 | 2024-10-25 09:15:00 | 1256.25 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2024-10-23 09:45:00 | 1297.00 | 2024-10-25 09:15:00 | 1256.25 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2024-11-05 13:15:00 | 1312.65 | 2024-11-08 09:15:00 | 1443.92 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-04 09:15:00 | 1351.00 | 2024-12-04 11:15:00 | 1340.65 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-12-05 09:15:00 | 1363.35 | 2024-12-05 12:15:00 | 1336.15 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2024-12-05 10:00:00 | 1351.75 | 2024-12-05 12:15:00 | 1336.15 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-12-30 09:15:00 | 1191.85 | 2025-01-02 09:15:00 | 1199.20 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-12-30 10:45:00 | 1192.30 | 2025-01-02 09:15:00 | 1199.20 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-12-30 11:30:00 | 1192.15 | 2025-01-02 09:15:00 | 1199.20 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2024-12-31 14:15:00 | 1181.95 | 2025-01-02 09:15:00 | 1199.20 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-01-01 09:30:00 | 1172.00 | 2025-01-02 10:15:00 | 1202.00 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-01-01 10:30:00 | 1175.70 | 2025-01-02 10:15:00 | 1202.00 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-01-01 11:15:00 | 1178.00 | 2025-01-02 10:15:00 | 1202.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-01-01 13:00:00 | 1176.20 | 2025-01-02 10:15:00 | 1202.00 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-01-08 10:15:00 | 1149.00 | 2025-01-10 09:15:00 | 1091.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 11:00:00 | 1149.85 | 2025-01-10 09:15:00 | 1092.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 11:30:00 | 1148.00 | 2025-01-10 09:15:00 | 1090.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 10:15:00 | 1149.00 | 2025-01-13 09:15:00 | 1034.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-08 11:00:00 | 1149.85 | 2025-01-13 09:15:00 | 1034.87 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-08 11:30:00 | 1148.00 | 2025-01-13 09:15:00 | 1033.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-04 12:30:00 | 870.95 | 2025-02-05 09:15:00 | 903.85 | STOP_HIT | 1.00 | -3.78% |
| SELL | retest2 | 2025-02-07 12:15:00 | 874.90 | 2025-02-11 09:15:00 | 831.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 12:45:00 | 874.80 | 2025-02-11 09:15:00 | 831.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 13:45:00 | 871.80 | 2025-02-11 09:15:00 | 828.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:30:00 | 863.30 | 2025-02-11 09:15:00 | 820.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 12:15:00 | 874.90 | 2025-02-12 09:15:00 | 787.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-07 12:45:00 | 874.80 | 2025-02-12 09:15:00 | 787.32 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-07 13:45:00 | 871.80 | 2025-02-12 10:15:00 | 784.62 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-10 09:30:00 | 863.30 | 2025-02-12 11:15:00 | 821.40 | STOP_HIT | 0.50 | 4.85% |
| SELL | retest2 | 2025-02-13 15:15:00 | 826.00 | 2025-02-14 13:15:00 | 784.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 15:15:00 | 826.00 | 2025-02-17 13:15:00 | 781.00 | STOP_HIT | 0.50 | 5.45% |
| SELL | retest2 | 2025-03-04 11:15:00 | 728.05 | 2025-03-05 09:15:00 | 736.15 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-03-04 14:30:00 | 727.80 | 2025-03-05 09:15:00 | 736.15 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-03-07 11:00:00 | 752.80 | 2025-03-10 09:15:00 | 722.85 | STOP_HIT | 1.00 | -3.98% |
| BUY | retest1 | 2025-03-21 09:15:00 | 774.50 | 2025-03-21 09:15:00 | 813.23 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-03-21 09:15:00 | 774.50 | 2025-03-25 09:15:00 | 810.20 | STOP_HIT | 0.50 | 4.61% |
| BUY | retest2 | 2025-03-21 10:30:00 | 823.80 | 2025-03-26 11:15:00 | 767.40 | STOP_HIT | 1.00 | -6.85% |
| BUY | retest2 | 2025-03-21 11:00:00 | 822.00 | 2025-03-26 11:15:00 | 767.40 | STOP_HIT | 1.00 | -6.64% |
| BUY | retest2 | 2025-03-21 12:15:00 | 821.15 | 2025-03-26 11:15:00 | 767.40 | STOP_HIT | 1.00 | -6.55% |
| BUY | retest2 | 2025-03-24 10:00:00 | 823.25 | 2025-03-26 11:15:00 | 767.40 | STOP_HIT | 1.00 | -6.78% |
| SELL | retest2 | 2025-03-27 12:15:00 | 773.00 | 2025-04-01 09:15:00 | 802.70 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2025-03-28 09:45:00 | 773.00 | 2025-04-01 09:15:00 | 802.70 | STOP_HIT | 1.00 | -3.84% |
| BUY | retest2 | 2025-04-04 15:15:00 | 851.00 | 2025-04-07 09:15:00 | 760.95 | STOP_HIT | 1.00 | -10.58% |
| BUY | retest2 | 2025-04-22 09:15:00 | 884.40 | 2025-04-25 10:15:00 | 849.80 | STOP_HIT | 1.00 | -3.91% |
| SELL | retest2 | 2025-05-06 11:30:00 | 712.40 | 2025-05-09 09:15:00 | 676.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 11:30:00 | 712.40 | 2025-05-09 15:15:00 | 682.75 | STOP_HIT | 0.50 | 4.16% |
| SELL | retest2 | 2025-06-11 13:00:00 | 716.50 | 2025-06-20 15:15:00 | 706.90 | STOP_HIT | 1.00 | 1.34% |
| BUY | retest2 | 2025-06-30 12:30:00 | 715.75 | 2025-07-01 09:15:00 | 709.20 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-06-30 13:00:00 | 715.50 | 2025-07-01 09:15:00 | 709.20 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-07-04 11:45:00 | 702.10 | 2025-07-10 14:15:00 | 702.65 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-07-04 12:30:00 | 703.15 | 2025-07-10 14:15:00 | 702.65 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2025-07-04 13:00:00 | 703.05 | 2025-07-10 14:15:00 | 702.65 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-07-04 14:00:00 | 702.60 | 2025-07-10 14:15:00 | 702.65 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-07-07 13:45:00 | 701.60 | 2025-07-10 14:15:00 | 702.65 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-07-08 09:30:00 | 701.05 | 2025-07-10 14:15:00 | 702.65 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-07-09 11:30:00 | 701.15 | 2025-07-10 14:15:00 | 702.65 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-07-10 10:45:00 | 701.50 | 2025-07-10 14:15:00 | 702.65 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-07-24 09:15:00 | 608.85 | 2025-07-24 10:15:00 | 627.35 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2025-07-24 12:45:00 | 617.85 | 2025-07-28 09:15:00 | 586.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 09:15:00 | 610.75 | 2025-07-28 13:15:00 | 580.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 12:45:00 | 617.85 | 2025-07-29 09:15:00 | 588.50 | STOP_HIT | 0.50 | 4.75% |
| SELL | retest2 | 2025-07-25 09:15:00 | 610.75 | 2025-07-29 09:15:00 | 588.50 | STOP_HIT | 0.50 | 3.64% |
| SELL | retest2 | 2025-09-09 09:15:00 | 589.60 | 2025-09-11 10:15:00 | 593.80 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-09-10 14:45:00 | 590.70 | 2025-09-11 10:15:00 | 593.80 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-09-15 09:15:00 | 602.85 | 2025-09-22 09:15:00 | 608.90 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest2 | 2025-09-15 15:15:00 | 597.00 | 2025-09-22 09:15:00 | 608.90 | STOP_HIT | 1.00 | 1.99% |
| SELL | retest2 | 2025-09-26 11:15:00 | 585.20 | 2025-09-29 09:15:00 | 599.30 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2025-09-26 13:15:00 | 585.30 | 2025-09-29 09:15:00 | 599.30 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2025-10-28 11:00:00 | 541.95 | 2025-11-04 09:15:00 | 556.70 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2025-10-28 12:15:00 | 541.50 | 2025-11-04 09:15:00 | 556.70 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-10-28 13:00:00 | 541.90 | 2025-11-04 09:15:00 | 556.70 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-10-29 09:15:00 | 541.80 | 2025-11-04 09:15:00 | 556.70 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2025-10-30 10:30:00 | 539.60 | 2025-11-04 09:15:00 | 556.70 | STOP_HIT | 1.00 | -3.17% |
| SELL | retest2 | 2025-11-03 11:30:00 | 540.00 | 2025-11-04 09:15:00 | 556.70 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2025-11-03 13:45:00 | 539.50 | 2025-11-04 09:15:00 | 556.70 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2025-11-20 13:15:00 | 512.10 | 2025-11-24 14:15:00 | 486.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 14:30:00 | 512.00 | 2025-11-24 14:15:00 | 486.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-21 09:30:00 | 509.45 | 2025-11-24 14:15:00 | 483.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 13:15:00 | 512.10 | 2025-11-26 09:15:00 | 481.40 | STOP_HIT | 0.50 | 5.99% |
| SELL | retest2 | 2025-11-20 14:30:00 | 512.00 | 2025-11-26 09:15:00 | 481.40 | STOP_HIT | 0.50 | 5.98% |
| SELL | retest2 | 2025-11-21 09:30:00 | 509.45 | 2025-11-26 09:15:00 | 481.40 | STOP_HIT | 0.50 | 5.51% |
| SELL | retest2 | 2025-12-10 10:30:00 | 471.95 | 2025-12-12 09:15:00 | 495.15 | STOP_HIT | 1.00 | -4.92% |
| SELL | retest2 | 2025-12-17 10:15:00 | 462.00 | 2025-12-22 13:15:00 | 454.30 | STOP_HIT | 1.00 | 1.67% |
| SELL | retest2 | 2025-12-29 11:15:00 | 447.90 | 2025-12-29 15:15:00 | 456.60 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-01-08 11:00:00 | 446.50 | 2026-01-09 10:15:00 | 424.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:45:00 | 444.20 | 2026-01-09 10:15:00 | 421.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:00:00 | 446.50 | 2026-01-12 09:15:00 | 401.85 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-08 11:45:00 | 444.20 | 2026-01-12 09:15:00 | 399.78 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-23 14:00:00 | 327.30 | 2026-02-26 09:15:00 | 345.40 | STOP_HIT | 1.00 | -5.53% |
| SELL | retest2 | 2026-02-23 14:45:00 | 324.35 | 2026-02-26 09:15:00 | 345.40 | STOP_HIT | 1.00 | -6.49% |
| SELL | retest2 | 2026-02-23 15:15:00 | 325.70 | 2026-02-26 09:15:00 | 345.40 | STOP_HIT | 1.00 | -6.05% |
| SELL | retest2 | 2026-03-10 11:45:00 | 454.50 | 2026-03-10 14:15:00 | 462.90 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2026-03-10 13:15:00 | 453.70 | 2026-03-10 14:15:00 | 462.90 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2026-03-23 09:15:00 | 416.70 | 2026-03-25 11:15:00 | 426.85 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2026-03-30 09:15:00 | 403.10 | 2026-03-30 15:15:00 | 382.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-30 09:15:00 | 403.10 | 2026-04-01 09:15:00 | 409.00 | STOP_HIT | 0.50 | -1.46% |
| SELL | retest2 | 2026-04-01 10:15:00 | 407.25 | 2026-04-01 11:15:00 | 417.50 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2026-04-01 10:45:00 | 407.60 | 2026-04-01 11:15:00 | 417.50 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2026-04-08 09:15:00 | 432.25 | 2026-04-16 10:15:00 | 431.30 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2026-04-16 10:00:00 | 429.45 | 2026-04-16 10:15:00 | 431.30 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest1 | 2026-04-23 11:15:00 | 404.00 | 2026-04-24 10:15:00 | 415.90 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2026-04-24 13:00:00 | 410.25 | 2026-04-24 15:15:00 | 412.75 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2026-04-24 14:45:00 | 409.70 | 2026-04-24 15:15:00 | 412.75 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2026-04-29 10:45:00 | 423.00 | 2026-04-30 09:15:00 | 409.40 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest2 | 2026-04-29 14:00:00 | 417.40 | 2026-04-30 09:15:00 | 409.40 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2026-04-29 15:15:00 | 419.20 | 2026-04-30 09:15:00 | 409.40 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2026-04-30 10:30:00 | 419.00 | 2026-04-30 15:15:00 | 413.70 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2026-04-30 13:30:00 | 419.40 | 2026-04-30 15:15:00 | 413.70 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2026-04-30 14:45:00 | 419.05 | 2026-05-06 10:15:00 | 460.90 | TARGET_HIT | 1.00 | 9.99% |
| BUY | retest2 | 2026-05-04 09:30:00 | 419.50 | 2026-05-06 10:15:00 | 461.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-05-04 13:30:00 | 419.05 | 2026-05-06 10:15:00 | 460.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-05-05 14:30:00 | 426.70 | 2026-05-06 10:15:00 | 469.37 | TARGET_HIT | 1.00 | 10.00% |
