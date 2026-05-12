# Max Healthcare Institute Ltd. (MAXHEALTH)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1013.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 79 |
| ALERT1 | 48 |
| ALERT2 | 48 |
| ALERT2_SKIP | 21 |
| ALERT3 | 119 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 58 |
| PARTIAL | 8 |
| TARGET_HIT | 0 |
| STOP_HIT | 58 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 66 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 38 / 28
- **Target hits / Stop hits / Partials:** 0 / 58 / 8
- **Avg / median % per leg:** 0.88% / 0.44%
- **Sum % (uncompounded):** 58.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 37 | 18 | 48.6% | 0 | 37 | 0 | 0.20% | 7.5% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 3 | 0 | -0.28% | -0.8% |
| BUY @ 3rd Alert (retest2) | 34 | 16 | 47.1% | 0 | 34 | 0 | 0.24% | 8.3% |
| SELL (all) | 29 | 20 | 69.0% | 0 | 21 | 8 | 1.75% | 50.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 29 | 20 | 69.0% | 0 | 21 | 8 | 1.75% | 50.8% |
| retest1 (combined) | 3 | 2 | 66.7% | 0 | 3 | 0 | -0.28% | -0.8% |
| retest2 (combined) | 63 | 36 | 57.1% | 0 | 55 | 8 | 0.94% | 59.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1150.90 | 1133.17 | 1131.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 1160.80 | 1145.96 | 1138.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 13:15:00 | 1170.50 | 1170.83 | 1162.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 14:00:00 | 1170.50 | 1170.83 | 1162.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 1175.60 | 1184.51 | 1177.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 09:30:00 | 1180.30 | 1184.51 | 1177.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 1174.20 | 1182.45 | 1177.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 11:00:00 | 1174.20 | 1182.45 | 1177.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 11:15:00 | 1171.00 | 1180.16 | 1176.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 12:30:00 | 1175.40 | 1178.17 | 1176.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 13:15:00 | 1163.60 | 1175.26 | 1174.89 | SL hit (close<static) qty=1.00 sl=1170.50 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 14:15:00 | 1165.60 | 1173.32 | 1174.04 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 1188.50 | 1176.46 | 1175.26 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 10:15:00 | 1157.00 | 1173.47 | 1175.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 1149.00 | 1162.59 | 1167.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 12:15:00 | 1147.10 | 1145.38 | 1153.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 13:00:00 | 1147.10 | 1145.38 | 1153.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 1153.00 | 1147.61 | 1152.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:15:00 | 1157.80 | 1147.61 | 1152.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 1156.20 | 1149.33 | 1152.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:15:00 | 1160.10 | 1149.33 | 1152.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 1166.90 | 1152.84 | 1154.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 1166.90 | 1152.84 | 1154.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 1165.50 | 1155.37 | 1155.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 13:15:00 | 1169.80 | 1160.14 | 1157.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 14:15:00 | 1173.50 | 1174.02 | 1168.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 15:00:00 | 1173.50 | 1174.02 | 1168.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1167.60 | 1172.48 | 1168.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 1167.60 | 1172.48 | 1168.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1164.30 | 1170.84 | 1168.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:30:00 | 1164.40 | 1170.84 | 1168.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 1155.60 | 1166.52 | 1166.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 12:45:00 | 1156.50 | 1166.52 | 1166.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 13:15:00 | 1155.90 | 1164.40 | 1165.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 14:15:00 | 1148.10 | 1158.40 | 1161.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 15:15:00 | 1144.50 | 1138.87 | 1147.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 1136.00 | 1138.29 | 1146.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1136.00 | 1138.29 | 1146.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:30:00 | 1142.50 | 1138.29 | 1146.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 1150.50 | 1136.15 | 1140.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 1150.50 | 1136.15 | 1140.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 1148.40 | 1138.60 | 1141.52 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 12:15:00 | 1156.30 | 1144.49 | 1143.84 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 1138.70 | 1146.27 | 1146.32 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 1150.00 | 1145.73 | 1145.41 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 12:15:00 | 1138.70 | 1144.32 | 1144.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 09:15:00 | 1132.90 | 1139.33 | 1142.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 10:15:00 | 1148.50 | 1141.17 | 1142.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 10:15:00 | 1148.50 | 1141.17 | 1142.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 1148.50 | 1141.17 | 1142.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:00:00 | 1148.50 | 1141.17 | 1142.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 11:15:00 | 1157.30 | 1144.39 | 1143.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 13:15:00 | 1164.30 | 1150.87 | 1147.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 12:15:00 | 1180.00 | 1183.84 | 1174.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 13:00:00 | 1180.00 | 1183.84 | 1174.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1211.80 | 1192.39 | 1185.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 11:00:00 | 1216.20 | 1197.15 | 1188.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 13:00:00 | 1217.10 | 1203.29 | 1192.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:30:00 | 1219.40 | 1205.50 | 1197.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 10:15:00 | 1203.00 | 1225.03 | 1226.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 10:15:00 | 1203.00 | 1225.03 | 1226.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 11:15:00 | 1190.80 | 1218.19 | 1222.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 1188.00 | 1173.55 | 1187.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 1188.00 | 1173.55 | 1187.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1188.00 | 1173.55 | 1187.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:00:00 | 1188.00 | 1173.55 | 1187.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1193.20 | 1177.48 | 1188.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:45:00 | 1193.60 | 1177.48 | 1188.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1201.80 | 1182.35 | 1189.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:45:00 | 1201.10 | 1182.35 | 1189.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 1213.80 | 1195.48 | 1194.19 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-06-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 15:15:00 | 1190.00 | 1194.36 | 1194.80 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 1198.60 | 1194.96 | 1194.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 1209.20 | 1199.47 | 1197.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 14:15:00 | 1276.90 | 1277.17 | 1266.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 15:00:00 | 1276.90 | 1277.17 | 1266.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 1271.50 | 1277.37 | 1273.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:45:00 | 1287.10 | 1279.06 | 1274.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 15:00:00 | 1285.90 | 1280.19 | 1276.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 1290.50 | 1296.38 | 1296.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 09:15:00 | 1290.50 | 1296.38 | 1296.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 10:15:00 | 1266.60 | 1290.42 | 1293.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 1244.90 | 1230.80 | 1243.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 1244.90 | 1230.80 | 1243.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 1244.90 | 1230.80 | 1243.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 1247.20 | 1230.80 | 1243.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1253.60 | 1235.36 | 1244.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:45:00 | 1252.50 | 1235.36 | 1244.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 1258.80 | 1240.05 | 1245.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:00:00 | 1258.80 | 1240.05 | 1245.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 1249.50 | 1244.78 | 1246.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:45:00 | 1252.70 | 1244.78 | 1246.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 1252.50 | 1246.33 | 1247.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:15:00 | 1243.90 | 1246.33 | 1247.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 1257.40 | 1248.92 | 1248.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 1262.70 | 1254.73 | 1251.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 14:15:00 | 1265.30 | 1265.58 | 1259.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 14:30:00 | 1264.70 | 1265.58 | 1259.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1252.60 | 1262.73 | 1259.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:00:00 | 1252.60 | 1262.73 | 1259.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 1256.90 | 1261.56 | 1259.17 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 12:15:00 | 1247.40 | 1257.74 | 1257.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 14:15:00 | 1243.30 | 1254.42 | 1256.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 12:15:00 | 1224.90 | 1222.36 | 1231.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 13:00:00 | 1224.90 | 1222.36 | 1231.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1212.60 | 1218.43 | 1226.41 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 10:15:00 | 1249.80 | 1227.00 | 1226.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 14:15:00 | 1256.20 | 1238.80 | 1232.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 1261.20 | 1264.17 | 1253.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 09:30:00 | 1261.10 | 1264.17 | 1253.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 1275.60 | 1266.46 | 1255.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 15:00:00 | 1282.50 | 1270.50 | 1260.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 1249.90 | 1268.59 | 1262.66 | SL hit (close<static) qty=1.00 sl=1255.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 12:15:00 | 1233.50 | 1258.26 | 1258.79 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 09:15:00 | 1266.90 | 1258.23 | 1258.21 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 1249.80 | 1260.13 | 1260.86 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 10:15:00 | 1273.40 | 1253.64 | 1251.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 11:15:00 | 1279.50 | 1258.81 | 1254.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 10:15:00 | 1260.70 | 1265.02 | 1260.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 10:15:00 | 1260.70 | 1265.02 | 1260.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 1260.70 | 1265.02 | 1260.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 1260.70 | 1265.02 | 1260.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 1266.50 | 1265.32 | 1260.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:30:00 | 1258.20 | 1265.32 | 1260.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 1264.40 | 1265.13 | 1261.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:30:00 | 1261.00 | 1265.13 | 1261.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 1258.70 | 1265.46 | 1262.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:30:00 | 1257.70 | 1265.46 | 1262.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 1252.50 | 1262.87 | 1261.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:45:00 | 1255.20 | 1262.87 | 1261.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 11:15:00 | 1251.50 | 1260.59 | 1260.86 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 12:15:00 | 1263.50 | 1256.23 | 1255.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 1270.20 | 1261.54 | 1258.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 14:15:00 | 1258.40 | 1262.96 | 1260.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 14:15:00 | 1258.40 | 1262.96 | 1260.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 1258.40 | 1262.96 | 1260.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:00:00 | 1258.40 | 1262.96 | 1260.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 1262.50 | 1262.87 | 1260.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 09:15:00 | 1291.50 | 1262.87 | 1260.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 13:00:00 | 1272.90 | 1276.37 | 1269.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 13:45:00 | 1273.30 | 1275.30 | 1269.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 09:15:00 | 1227.40 | 1263.33 | 1265.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 1227.40 | 1263.33 | 1265.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 12:15:00 | 1212.00 | 1241.43 | 1253.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 11:15:00 | 1218.50 | 1217.99 | 1227.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 11:30:00 | 1215.20 | 1217.99 | 1227.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 1228.60 | 1220.94 | 1227.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 14:00:00 | 1228.60 | 1220.94 | 1227.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 1228.40 | 1222.44 | 1227.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 14:30:00 | 1229.60 | 1222.44 | 1227.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 1233.00 | 1224.55 | 1227.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 09:15:00 | 1221.80 | 1224.55 | 1227.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 10:30:00 | 1226.90 | 1226.15 | 1228.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 14:45:00 | 1226.80 | 1224.37 | 1226.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 09:15:00 | 1242.60 | 1228.60 | 1228.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 1242.60 | 1228.60 | 1228.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 10:15:00 | 1249.00 | 1232.68 | 1229.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 10:15:00 | 1241.00 | 1242.56 | 1237.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 11:00:00 | 1241.00 | 1242.56 | 1237.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 1237.10 | 1240.96 | 1237.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 13:00:00 | 1237.10 | 1240.96 | 1237.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 1236.90 | 1240.15 | 1237.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:15:00 | 1236.90 | 1240.15 | 1237.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 1235.60 | 1239.24 | 1237.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:45:00 | 1234.40 | 1239.24 | 1237.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 1229.90 | 1237.37 | 1236.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 1212.70 | 1237.37 | 1236.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 09:15:00 | 1215.80 | 1233.06 | 1234.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 1181.90 | 1215.04 | 1223.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1168.30 | 1165.93 | 1181.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 1168.30 | 1165.93 | 1181.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1178.90 | 1164.52 | 1173.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 1178.90 | 1164.52 | 1173.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1179.80 | 1167.58 | 1174.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:30:00 | 1182.60 | 1167.58 | 1174.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 1184.50 | 1172.63 | 1175.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 1184.50 | 1172.63 | 1175.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 1180.80 | 1177.17 | 1177.07 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 09:15:00 | 1166.50 | 1175.04 | 1176.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 12:15:00 | 1156.30 | 1167.93 | 1172.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 14:15:00 | 1161.40 | 1158.97 | 1163.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-03 15:00:00 | 1161.40 | 1158.97 | 1163.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 1160.00 | 1159.18 | 1163.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 1167.80 | 1159.18 | 1163.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1170.70 | 1161.48 | 1164.14 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2025-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 14:15:00 | 1170.70 | 1165.36 | 1164.69 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 1155.70 | 1163.90 | 1164.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 14:15:00 | 1149.40 | 1161.00 | 1163.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 14:15:00 | 1152.10 | 1150.46 | 1155.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 14:15:00 | 1152.10 | 1150.46 | 1155.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 1152.10 | 1150.46 | 1155.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:00:00 | 1152.10 | 1150.46 | 1155.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 1153.00 | 1150.97 | 1155.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 1168.60 | 1150.97 | 1155.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1171.60 | 1155.10 | 1156.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 1171.60 | 1155.10 | 1156.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 1165.80 | 1157.24 | 1157.52 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 11:15:00 | 1167.00 | 1159.19 | 1158.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 12:15:00 | 1169.30 | 1161.21 | 1159.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 09:15:00 | 1174.50 | 1175.67 | 1170.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 10:00:00 | 1174.50 | 1175.67 | 1170.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1178.00 | 1178.83 | 1174.98 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 09:15:00 | 1158.80 | 1172.47 | 1173.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 11:15:00 | 1150.40 | 1165.80 | 1170.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 15:15:00 | 1162.00 | 1161.59 | 1166.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-17 09:15:00 | 1157.20 | 1161.59 | 1166.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 1152.30 | 1159.73 | 1165.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 10:30:00 | 1147.20 | 1156.27 | 1163.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 13:15:00 | 1170.70 | 1158.07 | 1159.03 | SL hit (close>static) qty=1.00 sl=1166.50 alert=retest2 |

### Cycle 35 — BUY (started 2025-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 14:15:00 | 1172.30 | 1160.92 | 1160.23 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 12:15:00 | 1155.10 | 1159.64 | 1160.10 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 14:15:00 | 1160.90 | 1160.46 | 1160.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 09:15:00 | 1171.10 | 1162.75 | 1161.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 09:15:00 | 1159.40 | 1167.30 | 1165.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 09:15:00 | 1159.40 | 1167.30 | 1165.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 1159.40 | 1167.30 | 1165.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:30:00 | 1155.80 | 1167.30 | 1165.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 1157.20 | 1165.28 | 1164.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:45:00 | 1159.10 | 1165.28 | 1164.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 11:15:00 | 1155.70 | 1163.36 | 1163.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 1139.20 | 1154.66 | 1159.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 1140.80 | 1139.22 | 1147.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 09:30:00 | 1141.00 | 1139.22 | 1147.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 1142.30 | 1139.61 | 1143.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 1113.80 | 1139.61 | 1143.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 09:30:00 | 1131.90 | 1129.32 | 1134.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 10:30:00 | 1132.40 | 1128.92 | 1133.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 10:15:00 | 1075.31 | 1103.42 | 1112.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 10:15:00 | 1075.78 | 1103.42 | 1112.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 1108.90 | 1085.13 | 1096.85 | SL hit (close>ema200) qty=0.50 sl=1085.13 alert=retest2 |

### Cycle 39 — BUY (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 12:15:00 | 1131.10 | 1104.46 | 1103.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 13:15:00 | 1135.10 | 1110.59 | 1106.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 15:15:00 | 1128.00 | 1129.44 | 1121.90 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 09:15:00 | 1135.00 | 1129.44 | 1121.90 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 10:00:00 | 1133.00 | 1130.15 | 1122.91 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1150.30 | 1140.76 | 1132.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 10:15:00 | 1153.80 | 1140.76 | 1132.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 11:00:00 | 1155.00 | 1143.61 | 1134.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 11:00:00 | 1161.90 | 1152.82 | 1144.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 10:15:00 | 1142.30 | 1152.92 | 1149.27 | SL hit (close<ema400) qty=1.00 sl=1149.27 alert=retest1 |

### Cycle 40 — SELL (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 15:15:00 | 1142.60 | 1146.98 | 1147.40 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 11:15:00 | 1152.50 | 1148.39 | 1147.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 13:15:00 | 1157.00 | 1150.45 | 1149.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 14:15:00 | 1155.20 | 1159.07 | 1155.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 14:15:00 | 1155.20 | 1159.07 | 1155.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 1155.20 | 1159.07 | 1155.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 1155.20 | 1159.07 | 1155.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 1158.00 | 1158.86 | 1155.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 1167.40 | 1158.86 | 1155.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 12:15:00 | 1191.80 | 1199.49 | 1200.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 1191.80 | 1199.49 | 1200.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 1185.70 | 1196.73 | 1199.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 14:15:00 | 1187.00 | 1185.70 | 1190.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 15:00:00 | 1187.00 | 1185.70 | 1190.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1183.50 | 1185.47 | 1189.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 1191.70 | 1185.47 | 1189.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 1187.90 | 1182.39 | 1185.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 1186.20 | 1182.39 | 1185.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1188.20 | 1183.55 | 1185.95 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 14:15:00 | 1188.60 | 1187.05 | 1186.92 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 09:15:00 | 1176.50 | 1185.48 | 1186.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 1162.00 | 1176.31 | 1180.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 15:15:00 | 1150.30 | 1146.80 | 1155.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-04 10:00:00 | 1152.60 | 1147.96 | 1155.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 1141.80 | 1139.08 | 1146.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:00:00 | 1141.80 | 1139.08 | 1146.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 1140.90 | 1139.44 | 1145.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:30:00 | 1143.10 | 1139.44 | 1145.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 1122.40 | 1132.14 | 1139.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 10:15:00 | 1121.00 | 1132.14 | 1139.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 12:45:00 | 1120.00 | 1126.09 | 1131.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 12:15:00 | 1114.90 | 1103.38 | 1102.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 12:15:00 | 1114.90 | 1103.38 | 1102.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 09:15:00 | 1127.30 | 1115.17 | 1109.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 14:15:00 | 1118.60 | 1120.19 | 1114.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 14:15:00 | 1118.60 | 1120.19 | 1114.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 1118.60 | 1120.19 | 1114.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 1118.60 | 1120.19 | 1114.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 1139.50 | 1123.26 | 1116.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 10:15:00 | 1147.00 | 1123.26 | 1116.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 15:15:00 | 1152.00 | 1161.05 | 1162.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 15:15:00 | 1152.00 | 1161.05 | 1162.16 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 1170.80 | 1164.32 | 1163.49 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 14:15:00 | 1161.80 | 1163.04 | 1163.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 10:15:00 | 1159.10 | 1161.87 | 1162.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 13:15:00 | 1162.00 | 1161.53 | 1162.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 13:15:00 | 1162.00 | 1161.53 | 1162.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1162.00 | 1161.53 | 1162.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:30:00 | 1162.00 | 1161.53 | 1162.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 1161.20 | 1161.46 | 1162.06 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 11:15:00 | 1165.00 | 1162.38 | 1162.31 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 10:15:00 | 1161.10 | 1162.34 | 1162.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 11:15:00 | 1155.70 | 1161.01 | 1161.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 10:15:00 | 1085.60 | 1084.47 | 1095.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 10:30:00 | 1086.20 | 1084.47 | 1095.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 1091.50 | 1085.99 | 1094.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:00:00 | 1091.50 | 1085.99 | 1094.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 1096.50 | 1088.10 | 1094.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:45:00 | 1096.40 | 1088.10 | 1094.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 1098.40 | 1090.16 | 1094.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 1098.40 | 1090.16 | 1094.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 1099.30 | 1091.98 | 1095.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 1099.10 | 1091.98 | 1095.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 1076.10 | 1075.02 | 1079.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:30:00 | 1071.30 | 1075.82 | 1079.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 11:15:00 | 1086.40 | 1079.33 | 1079.58 | SL hit (close>static) qty=1.00 sl=1082.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 1082.30 | 1079.92 | 1079.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 14:15:00 | 1088.00 | 1081.82 | 1080.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 09:15:00 | 1082.90 | 1083.20 | 1081.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 1082.90 | 1083.20 | 1081.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1082.90 | 1083.20 | 1081.62 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 1066.30 | 1078.57 | 1080.11 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 15:15:00 | 1083.50 | 1079.93 | 1079.86 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 1072.50 | 1079.63 | 1080.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 15:15:00 | 1070.00 | 1077.70 | 1079.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 1045.50 | 1043.93 | 1056.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 11:00:00 | 1045.50 | 1043.93 | 1056.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 1056.00 | 1047.09 | 1055.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 1056.00 | 1047.09 | 1055.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 1048.20 | 1047.31 | 1054.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:45:00 | 1052.10 | 1047.31 | 1054.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1072.40 | 1052.62 | 1055.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 1072.40 | 1052.62 | 1055.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 1073.60 | 1059.85 | 1058.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 1078.50 | 1065.38 | 1061.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 1075.10 | 1075.78 | 1070.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:15:00 | 1069.10 | 1075.78 | 1070.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 1071.60 | 1074.94 | 1070.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 1066.00 | 1074.94 | 1070.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 1074.00 | 1074.76 | 1071.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 10:30:00 | 1069.10 | 1074.76 | 1071.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 1075.00 | 1075.45 | 1072.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 1074.80 | 1075.45 | 1072.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1082.10 | 1076.71 | 1073.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:15:00 | 1082.40 | 1076.71 | 1073.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 11:45:00 | 1082.70 | 1079.32 | 1075.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 13:15:00 | 1072.80 | 1076.27 | 1076.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 13:15:00 | 1072.80 | 1076.27 | 1076.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 1068.00 | 1073.05 | 1074.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 11:15:00 | 1052.20 | 1047.53 | 1054.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 12:00:00 | 1052.20 | 1047.53 | 1054.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 1052.80 | 1046.93 | 1051.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:00:00 | 1052.80 | 1046.93 | 1051.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 1051.70 | 1047.88 | 1051.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:15:00 | 1049.20 | 1047.88 | 1051.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 1055.40 | 1050.28 | 1050.93 | SL hit (close>static) qty=1.00 sl=1054.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 1059.00 | 1052.02 | 1051.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 12:15:00 | 1061.30 | 1053.88 | 1052.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 1057.70 | 1057.77 | 1055.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 1057.70 | 1057.77 | 1055.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1057.70 | 1057.77 | 1055.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 1049.90 | 1057.77 | 1055.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1055.30 | 1057.28 | 1055.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:00:00 | 1055.30 | 1057.28 | 1055.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 1055.70 | 1056.96 | 1055.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 13:45:00 | 1057.10 | 1055.73 | 1054.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 14:15:00 | 1050.80 | 1054.75 | 1054.55 | SL hit (close<static) qty=1.00 sl=1052.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 15:15:00 | 1050.70 | 1053.94 | 1054.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 1047.60 | 1051.83 | 1053.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 12:15:00 | 1055.70 | 1051.99 | 1052.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 12:15:00 | 1055.70 | 1051.99 | 1052.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 1055.70 | 1051.99 | 1052.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 13:00:00 | 1055.70 | 1051.99 | 1052.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 1056.60 | 1052.91 | 1053.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 14:15:00 | 1050.30 | 1052.91 | 1053.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 997.78 | 1016.46 | 1024.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 14:15:00 | 1014.10 | 1013.33 | 1019.70 | SL hit (close>ema200) qty=0.50 sl=1013.33 alert=retest2 |

### Cycle 59 — BUY (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 09:15:00 | 1033.20 | 1021.05 | 1019.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 12:15:00 | 1037.80 | 1027.49 | 1023.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 1028.90 | 1030.44 | 1026.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 1028.90 | 1030.44 | 1026.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1028.90 | 1030.44 | 1026.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 14:30:00 | 1038.40 | 1031.67 | 1028.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 1022.00 | 1030.03 | 1028.18 | SL hit (close<static) qty=1.00 sl=1025.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 1017.90 | 1026.06 | 1026.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 13:15:00 | 1013.10 | 1021.91 | 1024.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 1002.30 | 996.55 | 1004.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 12:15:00 | 1002.30 | 996.55 | 1004.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 1002.30 | 996.55 | 1004.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 12:45:00 | 1001.20 | 996.55 | 1004.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 1000.30 | 997.30 | 1004.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 997.20 | 1000.23 | 1004.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:45:00 | 997.30 | 999.22 | 1003.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 12:45:00 | 997.10 | 999.06 | 1002.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 13:15:00 | 995.40 | 999.06 | 1002.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 997.60 | 997.89 | 1001.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:45:00 | 1001.90 | 997.89 | 1001.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 999.20 | 998.41 | 1000.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:30:00 | 1001.70 | 998.41 | 1000.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 996.90 | 997.79 | 1000.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 11:30:00 | 998.60 | 997.79 | 1000.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 1003.90 | 999.01 | 1000.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:00:00 | 1003.90 | 999.01 | 1000.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 994.40 | 998.09 | 999.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 15:00:00 | 988.30 | 996.13 | 998.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 09:15:00 | 947.34 | 959.96 | 971.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 09:15:00 | 947.43 | 959.96 | 971.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 09:15:00 | 947.25 | 959.96 | 971.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 09:15:00 | 945.63 | 959.96 | 971.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-30 09:15:00 | 962.30 | 955.25 | 962.40 | SL hit (close>ema200) qty=0.50 sl=955.25 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 11:15:00 | 986.70 | 964.62 | 962.63 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 937.00 | 962.08 | 963.75 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 982.05 | 961.95 | 961.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 1012.50 | 996.03 | 982.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 1011.00 | 1026.51 | 1015.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 1011.00 | 1026.51 | 1015.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1011.00 | 1026.51 | 1015.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 1011.00 | 1026.51 | 1015.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1025.65 | 1026.34 | 1016.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:15:00 | 1028.65 | 1026.34 | 1016.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 15:15:00 | 1010.50 | 1019.25 | 1020.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-02-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-09 15:15:00 | 1010.50 | 1019.25 | 1020.23 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 15:15:00 | 1021.70 | 1019.86 | 1019.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 09:15:00 | 1043.00 | 1024.49 | 1021.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 1047.65 | 1055.83 | 1047.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 1047.65 | 1055.83 | 1047.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 1047.65 | 1055.83 | 1047.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:00:00 | 1047.65 | 1055.83 | 1047.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 1047.60 | 1054.18 | 1047.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:30:00 | 1054.60 | 1054.35 | 1048.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 13:15:00 | 1052.05 | 1053.82 | 1048.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 15:00:00 | 1052.35 | 1053.17 | 1049.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 1052.45 | 1052.43 | 1049.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 1061.05 | 1054.15 | 1050.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 10:45:00 | 1061.80 | 1055.57 | 1051.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 11:15:00 | 1063.75 | 1055.57 | 1051.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 09:30:00 | 1065.75 | 1070.04 | 1061.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 15:15:00 | 1080.40 | 1082.78 | 1083.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 15:15:00 | 1080.40 | 1082.78 | 1083.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 09:15:00 | 1079.85 | 1082.20 | 1082.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 10:15:00 | 1082.25 | 1082.21 | 1082.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 10:15:00 | 1082.25 | 1082.21 | 1082.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 1082.25 | 1082.21 | 1082.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:30:00 | 1083.70 | 1082.21 | 1082.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 1085.45 | 1082.86 | 1082.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 11:45:00 | 1085.55 | 1082.86 | 1082.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 1082.30 | 1082.74 | 1082.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 12:30:00 | 1084.15 | 1082.74 | 1082.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 13:15:00 | 1085.40 | 1083.28 | 1083.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 1087.80 | 1084.18 | 1083.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 1093.55 | 1099.93 | 1094.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 1093.55 | 1099.93 | 1094.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1093.55 | 1099.93 | 1094.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 1093.55 | 1099.93 | 1094.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 1092.30 | 1098.41 | 1094.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 12:45:00 | 1097.35 | 1097.49 | 1094.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 14:30:00 | 1101.00 | 1096.36 | 1094.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 1079.60 | 1092.11 | 1092.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1079.60 | 1092.11 | 1092.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 1044.20 | 1075.65 | 1083.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 13:15:00 | 1050.30 | 1049.98 | 1060.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 13:45:00 | 1050.90 | 1049.98 | 1060.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 1057.40 | 1051.47 | 1060.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 1061.50 | 1051.47 | 1060.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 1053.80 | 1051.93 | 1059.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 1045.40 | 1051.93 | 1059.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:15:00 | 993.13 | 1015.01 | 1024.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-17 09:15:00 | 980.00 | 978.22 | 992.28 | SL hit (close>ema200) qty=0.50 sl=978.22 alert=retest2 |

### Cycle 69 — BUY (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 11:15:00 | 973.40 | 963.12 | 962.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 12:15:00 | 982.00 | 966.89 | 963.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-24 15:15:00 | 966.50 | 968.71 | 965.70 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:15:00 | 976.80 | 968.71 | 965.70 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 976.70 | 970.60 | 967.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:30:00 | 968.30 | 970.60 | 967.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 976.00 | 977.54 | 973.64 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 954.40 | 972.09 | 972.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest1 |

### Cycle 70 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 954.40 | 972.09 | 972.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 931.85 | 955.84 | 961.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 949.00 | 948.16 | 955.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 13:45:00 | 947.95 | 948.16 | 955.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 928.55 | 943.71 | 951.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 10:15:00 | 925.30 | 943.71 | 951.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:15:00 | 912.00 | 935.41 | 943.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 12:15:00 | 945.20 | 937.49 | 936.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 12:15:00 | 945.20 | 937.49 | 936.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 09:15:00 | 951.55 | 941.68 | 939.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 09:15:00 | 946.95 | 950.78 | 946.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 09:15:00 | 946.95 | 950.78 | 946.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 946.95 | 950.78 | 946.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:00:00 | 946.95 | 950.78 | 946.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 944.70 | 949.56 | 946.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 11:00:00 | 944.70 | 949.56 | 946.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 11:15:00 | 950.00 | 949.65 | 946.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 12:45:00 | 955.95 | 950.25 | 947.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 15:15:00 | 957.00 | 950.83 | 947.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 957.15 | 952.64 | 949.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 14:45:00 | 953.95 | 954.59 | 951.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 950.40 | 953.75 | 951.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 976.75 | 953.75 | 951.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 1002.85 | 1008.14 | 1008.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 1002.85 | 1008.14 | 1008.66 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 1010.90 | 1006.51 | 1006.15 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 15:15:00 | 1003.90 | 1006.87 | 1007.18 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 1023.40 | 1010.18 | 1008.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 1028.30 | 1013.80 | 1010.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 1014.80 | 1016.74 | 1012.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 14:00:00 | 1014.80 | 1016.74 | 1012.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 1005.40 | 1014.47 | 1012.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:00:00 | 1005.40 | 1014.47 | 1012.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 1004.00 | 1012.38 | 1011.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 993.50 | 1012.38 | 1011.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 991.10 | 1008.12 | 1009.61 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 1017.00 | 1006.21 | 1005.36 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 11:15:00 | 1002.00 | 1005.16 | 1005.46 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 1007.70 | 1005.35 | 1005.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 11:15:00 | 1016.40 | 1007.56 | 1006.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 09:15:00 | 1010.00 | 1012.46 | 1009.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 10:00:00 | 1010.00 | 1012.46 | 1009.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 1013.55 | 1012.68 | 1009.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 14:30:00 | 1016.25 | 1014.05 | 1011.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 1022.95 | 1013.96 | 1011.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:45:00 | 1018.90 | 1014.26 | 1012.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-16 12:30:00 | 1175.40 | 2025-05-16 13:15:00 | 1163.60 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-06-12 11:00:00 | 1216.20 | 2025-06-18 10:15:00 | 1203.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-06-12 13:00:00 | 1217.10 | 2025-06-18 10:15:00 | 1203.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-06-13 10:30:00 | 1219.40 | 2025-06-18 10:15:00 | 1203.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-07-02 11:45:00 | 1287.10 | 2025-07-09 09:15:00 | 1290.50 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest2 | 2025-07-02 15:00:00 | 1285.90 | 2025-07-09 09:15:00 | 1290.50 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2025-07-25 15:00:00 | 1282.50 | 2025-07-28 10:15:00 | 1249.90 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-08-13 09:15:00 | 1291.50 | 2025-08-14 09:15:00 | 1227.40 | STOP_HIT | 1.00 | -4.96% |
| BUY | retest2 | 2025-08-13 13:00:00 | 1272.90 | 2025-08-14 09:15:00 | 1227.40 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest2 | 2025-08-13 13:45:00 | 1273.30 | 2025-08-14 09:15:00 | 1227.40 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2025-08-20 09:15:00 | 1221.80 | 2025-08-21 09:15:00 | 1242.60 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-08-20 10:30:00 | 1226.90 | 2025-08-21 09:15:00 | 1242.60 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-08-20 14:45:00 | 1226.80 | 2025-08-21 09:15:00 | 1242.60 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-09-17 10:30:00 | 1147.20 | 2025-09-18 13:15:00 | 1170.70 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1113.80 | 2025-10-03 10:15:00 | 1075.31 | PARTIAL | 0.50 | 3.46% |
| SELL | retest2 | 2025-09-29 09:30:00 | 1131.90 | 2025-10-03 10:15:00 | 1075.78 | PARTIAL | 0.50 | 4.96% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1113.80 | 2025-10-06 09:15:00 | 1108.90 | STOP_HIT | 0.50 | 0.44% |
| SELL | retest2 | 2025-09-29 09:30:00 | 1131.90 | 2025-10-06 09:15:00 | 1108.90 | STOP_HIT | 0.50 | 2.03% |
| SELL | retest2 | 2025-09-29 10:30:00 | 1132.40 | 2025-10-06 12:15:00 | 1131.10 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest1 | 2025-10-08 09:15:00 | 1135.00 | 2025-10-13 10:15:00 | 1142.30 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest1 | 2025-10-08 10:00:00 | 1133.00 | 2025-10-13 10:15:00 | 1142.30 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2025-10-09 10:15:00 | 1153.80 | 2025-10-13 15:15:00 | 1142.60 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-10-09 11:00:00 | 1155.00 | 2025-10-13 15:15:00 | 1142.60 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-10-10 11:00:00 | 1161.90 | 2025-10-13 15:15:00 | 1142.60 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-10-16 09:15:00 | 1167.40 | 2025-10-24 12:15:00 | 1191.80 | STOP_HIT | 1.00 | 2.09% |
| SELL | retest2 | 2025-11-07 10:15:00 | 1121.00 | 2025-11-17 12:15:00 | 1114.90 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest2 | 2025-11-10 12:45:00 | 1120.00 | 2025-11-17 12:15:00 | 1114.90 | STOP_HIT | 1.00 | 0.46% |
| BUY | retest2 | 2025-11-19 10:15:00 | 1147.00 | 2025-11-25 15:15:00 | 1152.00 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2025-12-10 13:30:00 | 1071.30 | 2025-12-11 11:15:00 | 1086.40 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-12-24 10:15:00 | 1082.40 | 2025-12-26 13:15:00 | 1072.80 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-12-24 11:45:00 | 1082.70 | 2025-12-26 13:15:00 | 1072.80 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-01-01 12:15:00 | 1049.20 | 2026-01-02 10:15:00 | 1055.40 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2026-01-02 10:30:00 | 1051.00 | 2026-01-02 11:15:00 | 1059.00 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2026-01-05 13:45:00 | 1057.10 | 2026-01-05 14:15:00 | 1050.80 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2026-01-06 14:15:00 | 1050.30 | 2026-01-12 09:15:00 | 997.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 14:15:00 | 1050.30 | 2026-01-12 14:15:00 | 1014.10 | STOP_HIT | 0.50 | 3.45% |
| BUY | retest2 | 2026-01-16 14:30:00 | 1038.40 | 2026-01-19 09:15:00 | 1022.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2026-01-22 10:15:00 | 997.20 | 2026-01-29 09:15:00 | 947.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 10:45:00 | 997.30 | 2026-01-29 09:15:00 | 947.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 12:45:00 | 997.10 | 2026-01-29 09:15:00 | 947.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 13:15:00 | 995.40 | 2026-01-29 09:15:00 | 945.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 10:15:00 | 997.20 | 2026-01-30 09:15:00 | 962.30 | STOP_HIT | 0.50 | 3.50% |
| SELL | retest2 | 2026-01-22 10:45:00 | 997.30 | 2026-01-30 09:15:00 | 962.30 | STOP_HIT | 0.50 | 3.51% |
| SELL | retest2 | 2026-01-22 12:45:00 | 997.10 | 2026-01-30 09:15:00 | 962.30 | STOP_HIT | 0.50 | 3.49% |
| SELL | retest2 | 2026-01-22 13:15:00 | 995.40 | 2026-01-30 09:15:00 | 962.30 | STOP_HIT | 0.50 | 3.33% |
| SELL | retest2 | 2026-01-23 15:00:00 | 988.30 | 2026-02-01 11:15:00 | 986.70 | STOP_HIT | 1.00 | 0.16% |
| BUY | retest2 | 2026-02-06 11:15:00 | 1028.65 | 2026-02-09 15:15:00 | 1010.50 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2026-02-13 11:30:00 | 1054.60 | 2026-02-24 15:15:00 | 1080.40 | STOP_HIT | 1.00 | 2.45% |
| BUY | retest2 | 2026-02-13 13:15:00 | 1052.05 | 2026-02-24 15:15:00 | 1080.40 | STOP_HIT | 1.00 | 2.69% |
| BUY | retest2 | 2026-02-13 15:00:00 | 1052.35 | 2026-02-24 15:15:00 | 1080.40 | STOP_HIT | 1.00 | 2.67% |
| BUY | retest2 | 2026-02-16 09:15:00 | 1052.45 | 2026-02-24 15:15:00 | 1080.40 | STOP_HIT | 1.00 | 2.66% |
| BUY | retest2 | 2026-02-16 10:45:00 | 1061.80 | 2026-02-24 15:15:00 | 1080.40 | STOP_HIT | 1.00 | 1.75% |
| BUY | retest2 | 2026-02-16 11:15:00 | 1063.75 | 2026-02-24 15:15:00 | 1080.40 | STOP_HIT | 1.00 | 1.57% |
| BUY | retest2 | 2026-02-17 09:30:00 | 1065.75 | 2026-02-24 15:15:00 | 1080.40 | STOP_HIT | 1.00 | 1.37% |
| BUY | retest2 | 2026-02-27 12:45:00 | 1097.35 | 2026-03-02 09:15:00 | 1079.60 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2026-02-27 14:30:00 | 1101.00 | 2026-03-02 09:15:00 | 1079.60 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-03-06 09:15:00 | 1045.40 | 2026-03-13 10:15:00 | 993.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 09:15:00 | 1045.40 | 2026-03-17 09:15:00 | 980.00 | STOP_HIT | 0.50 | 6.26% |
| BUY | retest1 | 2026-03-25 09:15:00 | 976.80 | 2026-03-30 09:15:00 | 954.40 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2026-04-06 10:15:00 | 925.30 | 2026-04-08 12:15:00 | 945.20 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2026-04-07 09:15:00 | 912.00 | 2026-04-08 12:15:00 | 945.20 | STOP_HIT | 1.00 | -3.64% |
| BUY | retest2 | 2026-04-10 12:45:00 | 955.95 | 2026-04-23 09:15:00 | 1002.85 | STOP_HIT | 1.00 | 4.91% |
| BUY | retest2 | 2026-04-10 15:15:00 | 957.00 | 2026-04-23 09:15:00 | 1002.85 | STOP_HIT | 1.00 | 4.79% |
| BUY | retest2 | 2026-04-13 10:45:00 | 957.15 | 2026-04-23 09:15:00 | 1002.85 | STOP_HIT | 1.00 | 4.77% |
| BUY | retest2 | 2026-04-13 14:45:00 | 953.95 | 2026-04-23 09:15:00 | 1002.85 | STOP_HIT | 1.00 | 5.13% |
| BUY | retest2 | 2026-04-15 09:15:00 | 976.75 | 2026-04-23 09:15:00 | 1002.85 | STOP_HIT | 1.00 | 2.67% |
