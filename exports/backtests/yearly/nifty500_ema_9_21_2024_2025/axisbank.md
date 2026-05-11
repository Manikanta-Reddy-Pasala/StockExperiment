# Axis Bank Ltd. (AXISBANK)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 1270.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 170 |
| ALERT1 | 104 |
| ALERT2 | 101 |
| ALERT2_SKIP | 44 |
| ALERT3 | 286 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 117 |
| PARTIAL | 12 |
| TARGET_HIT | 0 |
| STOP_HIT | 121 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 130 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 55 / 75
- **Target hits / Stop hits / Partials:** 0 / 118 / 12
- **Avg / median % per leg:** 0.61% / -0.15%
- **Sum % (uncompounded):** 79.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 48 | 17 | 35.4% | 0 | 48 | 0 | -0.16% | -7.8% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.70% | -1.7% |
| BUY @ 3rd Alert (retest2) | 47 | 17 | 36.2% | 0 | 47 | 0 | -0.13% | -6.1% |
| SELL (all) | 82 | 38 | 46.3% | 0 | 70 | 12 | 1.06% | 87.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 82 | 38 | 46.3% | 0 | 70 | 12 | 1.06% | 87.1% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.70% | -1.7% |
| retest2 (combined) | 129 | 55 | 42.6% | 0 | 117 | 12 | 0.63% | 81.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 15:15:00 | 1133.90 | 1124.47 | 1124.12 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 09:15:00 | 1120.30 | 1123.63 | 1123.77 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 13:15:00 | 1126.70 | 1123.60 | 1123.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 14:15:00 | 1128.95 | 1124.67 | 1123.97 | Break + close above crossover candle high |

### Cycle 4 — SELL (started 2024-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 10:15:00 | 1114.95 | 1123.64 | 1123.76 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 11:15:00 | 1132.35 | 1125.38 | 1124.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 14:15:00 | 1138.20 | 1127.23 | 1125.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 1131.05 | 1138.69 | 1135.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 1131.05 | 1138.69 | 1135.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 1131.05 | 1138.69 | 1135.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:00:00 | 1131.05 | 1138.69 | 1135.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 1139.65 | 1138.88 | 1135.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 11:15:00 | 1140.45 | 1138.88 | 1135.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-22 10:15:00 | 1124.60 | 1134.69 | 1135.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 10:15:00 | 1124.60 | 1134.69 | 1135.43 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 10:15:00 | 1149.80 | 1137.02 | 1135.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 11:15:00 | 1161.80 | 1141.97 | 1137.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 10:15:00 | 1182.60 | 1182.96 | 1173.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-28 10:30:00 | 1182.80 | 1182.96 | 1173.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 1169.30 | 1180.66 | 1176.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:00:00 | 1169.30 | 1180.66 | 1176.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 1165.25 | 1177.58 | 1175.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:45:00 | 1164.80 | 1177.58 | 1175.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 12:15:00 | 1167.80 | 1173.99 | 1174.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 13:15:00 | 1162.95 | 1171.78 | 1173.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 13:15:00 | 1169.35 | 1166.28 | 1168.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 13:15:00 | 1169.35 | 1166.28 | 1168.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 13:15:00 | 1169.35 | 1166.28 | 1168.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 13:45:00 | 1171.55 | 1166.28 | 1168.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 14:15:00 | 1167.40 | 1166.50 | 1168.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 15:15:00 | 1172.00 | 1166.50 | 1168.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 15:15:00 | 1172.00 | 1167.60 | 1169.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:15:00 | 1172.35 | 1167.60 | 1169.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 1164.45 | 1166.97 | 1168.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 10:15:00 | 1162.40 | 1166.97 | 1168.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 12:45:00 | 1163.95 | 1165.33 | 1167.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 14:00:00 | 1164.05 | 1165.08 | 1167.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 1207.95 | 1172.86 | 1170.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 1207.95 | 1172.86 | 1170.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 11:15:00 | 1234.00 | 1192.65 | 1179.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 1189.70 | 1206.52 | 1193.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 1189.70 | 1206.52 | 1193.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1189.70 | 1206.52 | 1193.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 1166.80 | 1206.52 | 1193.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 1144.75 | 1194.17 | 1188.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 1144.75 | 1194.17 | 1188.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 1124.95 | 1180.32 | 1182.97 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 1180.60 | 1167.37 | 1166.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 10:15:00 | 1194.85 | 1172.86 | 1169.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 12:15:00 | 1166.55 | 1171.82 | 1169.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 12:15:00 | 1166.55 | 1171.82 | 1169.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 12:15:00 | 1166.55 | 1171.82 | 1169.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 13:00:00 | 1166.55 | 1171.82 | 1169.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 13:15:00 | 1169.85 | 1171.43 | 1169.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 09:15:00 | 1177.50 | 1172.02 | 1170.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 10:30:00 | 1178.65 | 1173.84 | 1171.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-12 15:15:00 | 1186.00 | 1194.09 | 1194.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 15:15:00 | 1186.00 | 1194.09 | 1194.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-13 13:15:00 | 1179.90 | 1187.85 | 1191.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-14 14:15:00 | 1181.15 | 1180.31 | 1184.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-14 15:00:00 | 1181.15 | 1180.31 | 1184.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 1187.25 | 1181.81 | 1184.30 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2024-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 14:15:00 | 1193.00 | 1186.32 | 1185.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 09:15:00 | 1204.70 | 1190.58 | 1187.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 09:15:00 | 1230.40 | 1230.75 | 1218.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-21 09:45:00 | 1231.65 | 1230.75 | 1218.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 1230.30 | 1230.58 | 1225.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:45:00 | 1229.30 | 1230.58 | 1225.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 1248.25 | 1233.52 | 1228.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 12:15:00 | 1253.80 | 1239.18 | 1232.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 09:15:00 | 1267.05 | 1272.37 | 1272.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 09:15:00 | 1267.05 | 1272.37 | 1272.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 09:15:00 | 1250.55 | 1262.39 | 1266.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-02 13:15:00 | 1256.95 | 1256.18 | 1262.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 13:15:00 | 1256.95 | 1256.18 | 1262.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 1256.95 | 1256.18 | 1262.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 14:00:00 | 1256.95 | 1256.18 | 1262.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 1277.15 | 1258.71 | 1261.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 09:45:00 | 1277.70 | 1258.71 | 1261.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 1278.80 | 1262.73 | 1263.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 10:30:00 | 1281.55 | 1262.73 | 1263.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2024-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 11:15:00 | 1275.05 | 1265.19 | 1264.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 12:15:00 | 1282.80 | 1268.71 | 1265.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 10:15:00 | 1272.55 | 1275.75 | 1271.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 10:15:00 | 1272.55 | 1275.75 | 1271.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 1272.55 | 1275.75 | 1271.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 11:00:00 | 1272.55 | 1275.75 | 1271.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 1270.05 | 1274.61 | 1271.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 11:45:00 | 1269.45 | 1274.61 | 1271.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 12:15:00 | 1280.00 | 1275.69 | 1271.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 14:30:00 | 1281.50 | 1276.97 | 1273.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 09:15:00 | 1281.95 | 1277.14 | 1273.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 10:15:00 | 1283.85 | 1277.50 | 1274.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 09:30:00 | 1282.55 | 1287.71 | 1285.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 1283.05 | 1286.78 | 1285.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:45:00 | 1277.50 | 1286.78 | 1285.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 1286.10 | 1286.64 | 1285.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 12:15:00 | 1288.05 | 1286.64 | 1285.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-11 10:15:00 | 1279.90 | 1287.55 | 1286.75 | SL hit (close<static) qty=1.00 sl=1280.85 alert=retest2 |

### Cycle 16 — SELL (started 2024-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 11:15:00 | 1279.45 | 1285.93 | 1286.09 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 13:15:00 | 1294.85 | 1287.12 | 1286.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 15:15:00 | 1299.30 | 1291.08 | 1288.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 09:15:00 | 1309.85 | 1313.55 | 1305.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 10:00:00 | 1309.85 | 1313.55 | 1305.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 1313.65 | 1313.79 | 1306.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 11:45:00 | 1313.85 | 1313.79 | 1306.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 1307.20 | 1312.81 | 1307.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 15:00:00 | 1307.20 | 1312.81 | 1307.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 15:15:00 | 1308.45 | 1311.93 | 1308.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:15:00 | 1303.50 | 1311.93 | 1308.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 1303.20 | 1310.19 | 1307.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 10:00:00 | 1303.20 | 1310.19 | 1307.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 1304.00 | 1308.95 | 1307.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 11:00:00 | 1304.00 | 1308.95 | 1307.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 11:15:00 | 1302.20 | 1307.60 | 1306.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 11:45:00 | 1302.20 | 1307.60 | 1306.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 13:15:00 | 1304.00 | 1306.19 | 1306.26 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 13:15:00 | 1312.80 | 1306.86 | 1306.34 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 1302.05 | 1305.48 | 1305.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 13:15:00 | 1295.15 | 1301.53 | 1303.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 13:15:00 | 1178.20 | 1176.08 | 1198.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-26 14:00:00 | 1178.20 | 1176.08 | 1198.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 11:15:00 | 1177.65 | 1178.00 | 1191.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 11:45:00 | 1189.20 | 1178.00 | 1191.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 1166.90 | 1167.00 | 1171.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 09:30:00 | 1162.50 | 1168.13 | 1170.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 09:15:00 | 1150.95 | 1138.07 | 1137.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 09:15:00 | 1150.95 | 1138.07 | 1137.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 10:15:00 | 1159.65 | 1147.40 | 1143.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 12:15:00 | 1163.25 | 1164.03 | 1156.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 13:00:00 | 1163.25 | 1164.03 | 1156.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 1158.95 | 1162.84 | 1157.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:45:00 | 1157.00 | 1162.84 | 1157.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 1151.20 | 1160.16 | 1157.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 1153.80 | 1160.16 | 1157.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 1150.75 | 1158.28 | 1156.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:00:00 | 1150.75 | 1158.28 | 1156.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2024-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 13:15:00 | 1151.90 | 1155.01 | 1155.22 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 11:15:00 | 1160.70 | 1155.10 | 1155.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 12:15:00 | 1167.35 | 1157.55 | 1156.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 09:15:00 | 1155.75 | 1160.82 | 1158.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 09:15:00 | 1155.75 | 1160.82 | 1158.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 1155.75 | 1160.82 | 1158.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:00:00 | 1155.75 | 1160.82 | 1158.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 1157.90 | 1160.24 | 1158.49 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2024-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 13:15:00 | 1151.30 | 1156.99 | 1157.32 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 09:15:00 | 1169.90 | 1158.69 | 1157.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 10:15:00 | 1174.00 | 1161.76 | 1159.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 15:15:00 | 1165.50 | 1166.28 | 1163.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 09:15:00 | 1164.05 | 1166.28 | 1163.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 1164.90 | 1166.00 | 1163.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:30:00 | 1164.20 | 1166.00 | 1163.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 10:15:00 | 1166.00 | 1166.00 | 1163.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 11:00:00 | 1166.00 | 1166.00 | 1163.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 1169.90 | 1173.08 | 1170.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 15:00:00 | 1169.90 | 1173.08 | 1170.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 1170.00 | 1172.46 | 1170.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 1172.40 | 1172.46 | 1170.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 1169.15 | 1171.80 | 1170.10 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2024-08-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 15:15:00 | 1167.60 | 1169.32 | 1169.50 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 1173.25 | 1170.10 | 1169.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 11:15:00 | 1176.85 | 1172.15 | 1171.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 09:15:00 | 1174.50 | 1176.70 | 1174.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 09:15:00 | 1174.50 | 1176.70 | 1174.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 1174.50 | 1176.70 | 1174.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 10:00:00 | 1174.50 | 1176.70 | 1174.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 1172.00 | 1175.76 | 1174.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 11:00:00 | 1172.00 | 1175.76 | 1174.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 11:15:00 | 1170.90 | 1174.79 | 1173.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 12:00:00 | 1170.90 | 1174.79 | 1173.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 13:15:00 | 1171.70 | 1173.53 | 1173.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 13:45:00 | 1170.50 | 1173.53 | 1173.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2024-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 14:15:00 | 1170.65 | 1172.96 | 1173.07 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 11:15:00 | 1177.90 | 1173.95 | 1173.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 11:15:00 | 1178.95 | 1176.15 | 1174.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 14:15:00 | 1175.75 | 1176.58 | 1175.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 14:15:00 | 1175.75 | 1176.58 | 1175.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 1175.75 | 1176.58 | 1175.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 15:00:00 | 1175.75 | 1176.58 | 1175.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 15:15:00 | 1179.00 | 1177.06 | 1175.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 09:45:00 | 1181.15 | 1177.97 | 1176.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 11:30:00 | 1180.45 | 1183.16 | 1180.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 12:15:00 | 1175.35 | 1180.34 | 1181.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 12:15:00 | 1175.35 | 1180.34 | 1181.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 09:15:00 | 1166.35 | 1176.94 | 1178.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 13:15:00 | 1166.20 | 1161.08 | 1166.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 13:15:00 | 1166.20 | 1161.08 | 1166.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 1166.20 | 1161.08 | 1166.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:30:00 | 1162.25 | 1161.08 | 1166.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 1169.85 | 1162.83 | 1166.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 14:45:00 | 1172.75 | 1162.83 | 1166.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 1170.30 | 1164.32 | 1166.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 1184.65 | 1164.32 | 1166.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 10:15:00 | 1186.05 | 1171.84 | 1170.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 09:15:00 | 1190.75 | 1183.76 | 1177.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 12:15:00 | 1184.25 | 1185.28 | 1180.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 12:45:00 | 1184.40 | 1185.28 | 1180.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 1180.10 | 1184.70 | 1181.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:00:00 | 1180.10 | 1184.70 | 1181.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 1184.00 | 1184.56 | 1181.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 12:30:00 | 1185.75 | 1184.97 | 1182.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-24 14:15:00 | 1239.50 | 1241.78 | 1241.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 14:15:00 | 1239.50 | 1241.78 | 1241.97 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 10:15:00 | 1250.70 | 1243.52 | 1242.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-25 11:15:00 | 1257.55 | 1246.33 | 1244.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 12:15:00 | 1270.30 | 1272.10 | 1265.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-27 13:00:00 | 1270.30 | 1272.10 | 1265.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 1258.00 | 1269.28 | 1264.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 14:00:00 | 1258.00 | 1269.28 | 1264.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 1274.75 | 1270.37 | 1265.43 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 11:15:00 | 1245.80 | 1260.03 | 1261.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 12:15:00 | 1242.40 | 1256.50 | 1260.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 11:15:00 | 1199.10 | 1192.78 | 1208.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 12:00:00 | 1199.10 | 1192.78 | 1208.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 1181.65 | 1163.60 | 1170.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 1181.65 | 1163.60 | 1170.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 1178.95 | 1166.67 | 1171.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:45:00 | 1180.20 | 1166.67 | 1171.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 11:15:00 | 1170.05 | 1167.35 | 1171.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 13:00:00 | 1168.45 | 1167.57 | 1170.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 13:30:00 | 1168.95 | 1167.85 | 1170.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 14:00:00 | 1169.00 | 1167.85 | 1170.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-10 10:15:00 | 1180.00 | 1173.34 | 1172.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 10:15:00 | 1180.00 | 1173.34 | 1172.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 13:15:00 | 1186.55 | 1178.74 | 1175.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 10:15:00 | 1175.10 | 1180.50 | 1177.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 10:15:00 | 1175.10 | 1180.50 | 1177.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 1175.10 | 1180.50 | 1177.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:00:00 | 1175.10 | 1180.50 | 1177.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 1170.45 | 1178.49 | 1176.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:30:00 | 1171.75 | 1178.49 | 1176.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2024-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 13:15:00 | 1170.30 | 1175.58 | 1175.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 10:15:00 | 1164.70 | 1171.50 | 1173.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 15:15:00 | 1168.75 | 1167.38 | 1170.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 10:15:00 | 1164.15 | 1166.41 | 1169.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 1164.15 | 1166.41 | 1169.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:30:00 | 1171.00 | 1166.41 | 1169.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 1185.40 | 1146.33 | 1149.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 10:00:00 | 1185.40 | 1146.33 | 1149.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 10:15:00 | 1183.05 | 1153.67 | 1152.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 11:15:00 | 1194.45 | 1161.83 | 1156.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 14:15:00 | 1189.60 | 1190.05 | 1179.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-21 15:00:00 | 1189.60 | 1190.05 | 1179.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 11:15:00 | 1186.20 | 1189.39 | 1182.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 11:45:00 | 1185.30 | 1189.39 | 1182.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 12:15:00 | 1181.00 | 1187.71 | 1182.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 13:00:00 | 1181.00 | 1187.71 | 1182.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 13:15:00 | 1187.25 | 1187.62 | 1183.05 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2024-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 10:15:00 | 1168.35 | 1179.25 | 1180.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 11:15:00 | 1165.90 | 1176.58 | 1178.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 11:15:00 | 1169.70 | 1167.09 | 1171.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-24 12:00:00 | 1169.70 | 1167.09 | 1171.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 12:15:00 | 1169.60 | 1167.59 | 1171.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 12:30:00 | 1172.55 | 1167.59 | 1171.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 1181.60 | 1170.51 | 1171.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 11:30:00 | 1175.35 | 1171.89 | 1172.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-25 12:15:00 | 1175.45 | 1172.60 | 1172.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-10-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-25 12:15:00 | 1175.45 | 1172.60 | 1172.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-25 14:15:00 | 1190.95 | 1176.58 | 1174.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-28 10:15:00 | 1177.50 | 1179.48 | 1176.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-28 10:15:00 | 1177.50 | 1179.48 | 1176.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 1177.50 | 1179.48 | 1176.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:00:00 | 1177.50 | 1179.48 | 1176.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 1184.40 | 1180.46 | 1177.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:30:00 | 1176.15 | 1180.46 | 1177.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 12:15:00 | 1179.35 | 1180.24 | 1177.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-28 12:45:00 | 1179.70 | 1180.24 | 1177.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 13:15:00 | 1173.40 | 1178.87 | 1177.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-28 14:00:00 | 1173.40 | 1178.87 | 1177.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 14:15:00 | 1172.50 | 1177.60 | 1176.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-28 14:45:00 | 1168.65 | 1177.60 | 1176.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 09:15:00 | 1164.10 | 1174.32 | 1175.26 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 13:15:00 | 1186.60 | 1177.62 | 1176.47 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 11:15:00 | 1170.45 | 1176.45 | 1176.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 12:15:00 | 1164.95 | 1171.27 | 1173.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 15:15:00 | 1167.00 | 1166.79 | 1170.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 17:15:00 | 1170.90 | 1167.61 | 1170.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 1170.90 | 1167.61 | 1170.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 1170.90 | 1167.61 | 1170.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 1169.40 | 1167.97 | 1170.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-04 09:15:00 | 1154.15 | 1167.97 | 1170.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1149.65 | 1164.31 | 1168.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 11:00:00 | 1145.80 | 1160.61 | 1166.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 14:00:00 | 1143.50 | 1154.02 | 1161.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 10:30:00 | 1138.40 | 1146.08 | 1154.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 15:15:00 | 1170.15 | 1157.18 | 1157.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 15:15:00 | 1170.15 | 1157.18 | 1157.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 1177.45 | 1161.24 | 1158.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-06 14:15:00 | 1165.75 | 1166.93 | 1163.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-06 14:45:00 | 1166.25 | 1166.93 | 1163.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 1152.80 | 1164.44 | 1162.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 1152.80 | 1164.44 | 1162.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 1156.00 | 1162.75 | 1162.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 1152.00 | 1162.75 | 1162.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 11:15:00 | 1156.60 | 1161.52 | 1161.63 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 12:15:00 | 1162.60 | 1161.74 | 1161.72 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 14:15:00 | 1160.45 | 1161.49 | 1161.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 10:15:00 | 1156.70 | 1160.10 | 1160.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 12:15:00 | 1163.45 | 1160.46 | 1160.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 12:15:00 | 1163.45 | 1160.46 | 1160.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 1163.45 | 1160.46 | 1160.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:45:00 | 1163.00 | 1160.46 | 1160.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-08 13:15:00 | 1164.80 | 1161.32 | 1161.26 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 1154.35 | 1160.00 | 1160.69 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 10:15:00 | 1170.75 | 1162.15 | 1161.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-11 11:15:00 | 1172.45 | 1164.21 | 1162.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-12 12:15:00 | 1161.40 | 1169.51 | 1167.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 12:15:00 | 1161.40 | 1169.51 | 1167.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 12:15:00 | 1161.40 | 1169.51 | 1167.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 12:45:00 | 1162.05 | 1169.51 | 1167.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 13:15:00 | 1156.70 | 1166.95 | 1166.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 14:15:00 | 1155.40 | 1166.95 | 1166.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2024-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 14:15:00 | 1157.75 | 1165.11 | 1165.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 10:15:00 | 1149.70 | 1159.65 | 1162.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 10:15:00 | 1131.90 | 1129.46 | 1135.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-19 11:00:00 | 1131.90 | 1129.46 | 1135.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 13:15:00 | 1135.70 | 1130.39 | 1134.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 14:00:00 | 1135.70 | 1130.39 | 1134.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 1135.40 | 1131.40 | 1134.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 15:15:00 | 1133.45 | 1131.40 | 1134.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 1133.45 | 1131.81 | 1134.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 09:15:00 | 1126.85 | 1131.81 | 1134.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-21 10:15:00 | 1144.40 | 1134.53 | 1135.38 | SL hit (close>static) qty=1.00 sl=1138.45 alert=retest2 |

### Cycle 51 — BUY (started 2024-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 11:15:00 | 1148.90 | 1137.41 | 1136.61 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 11:15:00 | 1134.10 | 1136.45 | 1136.77 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 1141.80 | 1137.61 | 1137.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 1154.20 | 1142.16 | 1139.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 12:15:00 | 1145.25 | 1146.14 | 1142.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 13:00:00 | 1145.25 | 1146.14 | 1142.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 13:15:00 | 1147.50 | 1146.41 | 1142.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 14:30:00 | 1153.85 | 1148.59 | 1144.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 10:00:00 | 1153.55 | 1150.88 | 1146.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 10:30:00 | 1150.00 | 1150.79 | 1146.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 12:15:00 | 1151.45 | 1147.02 | 1146.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 1150.25 | 1149.02 | 1147.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 15:00:00 | 1150.25 | 1149.02 | 1147.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 1151.25 | 1149.78 | 1148.14 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-28 10:15:00 | 1140.90 | 1148.00 | 1147.48 | SL hit (close<static) qty=1.00 sl=1142.00 alert=retest2 |

### Cycle 54 — SELL (started 2024-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 12:15:00 | 1135.75 | 1144.91 | 1146.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 13:15:00 | 1133.75 | 1142.68 | 1145.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 13:15:00 | 1137.15 | 1135.98 | 1139.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 13:15:00 | 1137.15 | 1135.98 | 1139.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 1137.15 | 1135.98 | 1139.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 14:00:00 | 1137.15 | 1135.98 | 1139.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 1135.95 | 1135.97 | 1139.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 14:30:00 | 1139.20 | 1135.97 | 1139.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 1137.60 | 1133.63 | 1136.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 14:30:00 | 1139.55 | 1133.63 | 1136.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 15:15:00 | 1136.35 | 1134.18 | 1136.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:15:00 | 1143.20 | 1134.18 | 1136.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 1146.50 | 1136.64 | 1137.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:00:00 | 1146.50 | 1136.64 | 1137.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2024-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 10:15:00 | 1153.40 | 1139.99 | 1138.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 11:15:00 | 1158.70 | 1143.73 | 1140.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 13:15:00 | 1156.10 | 1157.13 | 1151.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 14:00:00 | 1156.10 | 1157.13 | 1151.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 1154.25 | 1157.08 | 1153.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:45:00 | 1151.10 | 1157.08 | 1153.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 1166.90 | 1159.05 | 1154.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 12:15:00 | 1168.45 | 1159.05 | 1154.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 13:00:00 | 1168.90 | 1161.02 | 1155.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 13:30:00 | 1170.50 | 1163.47 | 1157.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 15:15:00 | 1168.50 | 1164.16 | 1158.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 1185.05 | 1168.83 | 1161.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 12:15:00 | 1190.10 | 1173.06 | 1164.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 13:00:00 | 1190.80 | 1176.61 | 1166.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 13:30:00 | 1190.25 | 1179.34 | 1169.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 09:15:00 | 1163.35 | 1169.67 | 1169.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 09:15:00 | 1163.35 | 1169.67 | 1169.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 12:15:00 | 1154.30 | 1163.48 | 1166.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 12:15:00 | 1153.05 | 1151.70 | 1155.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-12 13:00:00 | 1153.05 | 1151.70 | 1155.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 1149.05 | 1140.29 | 1145.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 1149.05 | 1140.29 | 1145.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 1151.90 | 1142.61 | 1146.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 09:15:00 | 1143.00 | 1142.61 | 1146.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 10:15:00 | 1146.00 | 1144.08 | 1146.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 10:45:00 | 1144.70 | 1143.99 | 1146.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 13:15:00 | 1146.45 | 1144.49 | 1146.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 1149.15 | 1145.42 | 1146.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 14:00:00 | 1149.15 | 1145.42 | 1146.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 1150.50 | 1146.44 | 1146.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 15:00:00 | 1150.50 | 1146.44 | 1146.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 1141.50 | 1145.80 | 1146.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 11:30:00 | 1133.95 | 1143.14 | 1145.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 13:00:00 | 1134.65 | 1141.44 | 1144.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 09:15:00 | 1085.85 | 1106.91 | 1117.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 09:15:00 | 1088.70 | 1106.91 | 1117.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 09:15:00 | 1087.46 | 1106.91 | 1117.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 09:15:00 | 1089.13 | 1106.91 | 1117.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 13:15:00 | 1077.25 | 1092.47 | 1106.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 13:15:00 | 1077.92 | 1092.47 | 1106.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-24 10:15:00 | 1081.65 | 1079.65 | 1087.97 | SL hit (close>ema200) qty=0.50 sl=1079.65 alert=retest2 |

### Cycle 57 — BUY (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 11:15:00 | 1092.25 | 1082.00 | 1081.63 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 13:15:00 | 1066.45 | 1078.50 | 1080.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 09:15:00 | 1062.00 | 1072.73 | 1076.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 15:15:00 | 1068.00 | 1067.52 | 1071.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-01 09:15:00 | 1062.95 | 1067.52 | 1071.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 12:15:00 | 1069.50 | 1067.90 | 1070.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 12:45:00 | 1069.30 | 1067.90 | 1070.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 13:15:00 | 1075.20 | 1069.36 | 1071.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 14:00:00 | 1075.20 | 1069.36 | 1071.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 14:15:00 | 1071.50 | 1069.79 | 1071.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 09:15:00 | 1068.30 | 1070.28 | 1071.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 10:15:00 | 1075.65 | 1071.82 | 1071.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 10:15:00 | 1075.65 | 1071.82 | 1071.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 12:15:00 | 1081.40 | 1074.13 | 1072.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 13:15:00 | 1086.10 | 1086.24 | 1081.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 14:00:00 | 1086.10 | 1086.24 | 1081.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 1081.20 | 1085.16 | 1081.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:30:00 | 1076.40 | 1084.92 | 1081.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 1072.35 | 1082.40 | 1080.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 1072.35 | 1082.40 | 1080.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 1070.10 | 1079.94 | 1079.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 1066.15 | 1079.94 | 1079.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 12:15:00 | 1064.90 | 1076.93 | 1078.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 1061.85 | 1073.92 | 1077.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 13:15:00 | 1069.50 | 1067.79 | 1071.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 13:45:00 | 1067.65 | 1067.79 | 1071.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 1078.15 | 1069.92 | 1071.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 10:00:00 | 1078.15 | 1069.92 | 1071.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 1067.85 | 1069.50 | 1071.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 12:00:00 | 1064.95 | 1068.59 | 1070.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 12:30:00 | 1065.00 | 1066.88 | 1069.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 09:15:00 | 1072.45 | 1071.08 | 1071.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 09:15:00 | 1072.45 | 1071.08 | 1071.08 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 10:15:00 | 1058.60 | 1068.59 | 1069.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 1048.65 | 1061.22 | 1065.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-13 09:15:00 | 1055.60 | 1049.13 | 1055.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-13 09:15:00 | 1055.60 | 1049.13 | 1055.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 1055.60 | 1049.13 | 1055.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-13 10:00:00 | 1055.60 | 1049.13 | 1055.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 10:15:00 | 1060.05 | 1051.31 | 1056.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-13 10:45:00 | 1058.50 | 1051.31 | 1056.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 11:15:00 | 1054.75 | 1052.00 | 1056.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 12:45:00 | 1052.90 | 1051.98 | 1055.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 12:00:00 | 1053.85 | 1053.13 | 1054.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 13:00:00 | 1053.40 | 1053.18 | 1054.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 14:30:00 | 1052.80 | 1053.34 | 1054.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 1055.50 | 1053.77 | 1054.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 09:15:00 | 1042.00 | 1053.77 | 1054.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 09:15:00 | 1000.25 | 1026.92 | 1035.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 09:15:00 | 1001.16 | 1026.92 | 1035.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 09:15:00 | 1000.73 | 1026.92 | 1035.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 09:15:00 | 1000.16 | 1026.92 | 1035.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 09:15:00 | 989.90 | 1026.92 | 1035.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-20 13:15:00 | 991.90 | 991.34 | 1004.25 | SL hit (close>ema200) qty=0.50 sl=991.34 alert=retest2 |

### Cycle 63 — BUY (started 2025-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 10:15:00 | 977.95 | 956.72 | 955.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-28 11:15:00 | 991.15 | 963.61 | 959.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-29 15:15:00 | 981.45 | 981.95 | 975.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 09:15:00 | 977.10 | 981.95 | 975.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 985.20 | 982.60 | 976.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:30:00 | 980.20 | 982.60 | 976.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 983.35 | 984.93 | 980.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 09:15:00 | 989.95 | 985.58 | 982.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 12:45:00 | 991.65 | 989.32 | 985.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 13:15:00 | 995.05 | 989.32 | 985.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 13:00:00 | 991.50 | 991.22 | 989.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 13:15:00 | 989.70 | 990.92 | 989.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 13:45:00 | 990.00 | 990.92 | 989.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 14:15:00 | 989.20 | 990.57 | 989.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 15:00:00 | 989.20 | 990.57 | 989.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 15:15:00 | 989.45 | 990.35 | 989.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:15:00 | 988.75 | 990.35 | 989.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 1008.35 | 993.95 | 990.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 12:45:00 | 1014.45 | 1002.26 | 995.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 10:45:00 | 1010.60 | 1016.99 | 1015.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 11:30:00 | 1010.20 | 1015.48 | 1014.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 12:15:00 | 1013.80 | 1015.48 | 1014.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 13:15:00 | 1010.85 | 1014.05 | 1014.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 13:15:00 | 1010.85 | 1014.05 | 1014.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 12:15:00 | 1005.65 | 1011.39 | 1012.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 1004.45 | 1002.84 | 1007.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:45:00 | 1004.25 | 1002.84 | 1007.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 1009.10 | 1004.09 | 1007.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:45:00 | 1008.70 | 1004.09 | 1007.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 1002.00 | 1003.68 | 1006.94 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2025-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 10:15:00 | 1013.75 | 1008.28 | 1008.25 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 09:15:00 | 1004.60 | 1007.77 | 1008.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 10:15:00 | 997.10 | 1005.64 | 1007.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 13:15:00 | 988.45 | 988.11 | 994.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 14:00:00 | 988.45 | 988.11 | 994.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 995.00 | 990.34 | 994.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 990.70 | 990.34 | 994.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 989.80 | 990.23 | 994.31 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 1006.00 | 995.13 | 994.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 11:15:00 | 1012.20 | 1000.27 | 997.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 1009.85 | 1013.04 | 1008.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 1009.85 | 1013.04 | 1008.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 1009.55 | 1012.35 | 1008.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 1009.55 | 1012.35 | 1008.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 1011.55 | 1012.19 | 1008.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 12:15:00 | 1012.80 | 1012.19 | 1008.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-21 12:15:00 | 1005.50 | 1010.85 | 1008.42 | SL hit (close<static) qty=1.00 sl=1008.05 alert=retest2 |

### Cycle 68 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 1003.00 | 1006.56 | 1007.03 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 14:15:00 | 1011.05 | 1007.83 | 1007.49 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 09:15:00 | 1003.40 | 1007.35 | 1007.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 10:15:00 | 999.95 | 1005.87 | 1006.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 13:15:00 | 1011.45 | 1006.62 | 1006.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 13:15:00 | 1011.45 | 1006.62 | 1006.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 13:15:00 | 1011.45 | 1006.62 | 1006.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 13:45:00 | 1009.65 | 1006.62 | 1006.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 14:15:00 | 1009.30 | 1007.16 | 1007.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-27 09:15:00 | 1016.00 | 1009.18 | 1007.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 12:15:00 | 1015.70 | 1019.76 | 1015.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 12:15:00 | 1015.70 | 1019.76 | 1015.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 12:15:00 | 1015.70 | 1019.76 | 1015.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 13:00:00 | 1015.70 | 1019.76 | 1015.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 13:15:00 | 1020.35 | 1019.88 | 1016.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 13:30:00 | 1017.25 | 1019.88 | 1016.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 1015.60 | 1019.02 | 1016.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 15:00:00 | 1015.60 | 1019.02 | 1016.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 1015.60 | 1018.34 | 1016.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:15:00 | 1002.50 | 1018.34 | 1016.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2025-03-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 09:15:00 | 995.00 | 1013.67 | 1014.21 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 1015.85 | 1009.03 | 1008.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 1026.00 | 1014.38 | 1011.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 12:15:00 | 1034.65 | 1035.50 | 1027.61 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 09:15:00 | 1045.05 | 1036.39 | 1029.99 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 1036.50 | 1039.98 | 1035.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:30:00 | 1036.55 | 1039.98 | 1035.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 1035.30 | 1039.05 | 1035.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 1029.90 | 1039.05 | 1035.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 1027.25 | 1036.69 | 1034.45 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-11 09:15:00 | 1027.25 | 1036.69 | 1034.45 | SL hit (close<ema400) qty=1.00 sl=1034.45 alert=retest1 |

### Cycle 74 — SELL (started 2025-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 11:15:00 | 1026.40 | 1033.08 | 1033.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 09:15:00 | 1009.75 | 1025.38 | 1029.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 10:15:00 | 1013.35 | 1012.73 | 1018.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-13 10:30:00 | 1014.30 | 1012.73 | 1018.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 1014.20 | 1011.99 | 1015.50 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 11:15:00 | 1033.65 | 1018.17 | 1017.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 13:15:00 | 1036.50 | 1024.04 | 1020.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 1050.00 | 1050.40 | 1043.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 1050.00 | 1050.40 | 1043.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 12:15:00 | 1093.00 | 1103.17 | 1095.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:00:00 | 1093.00 | 1103.17 | 1095.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 1094.60 | 1101.45 | 1095.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:30:00 | 1093.45 | 1101.45 | 1095.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 1098.40 | 1100.84 | 1095.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 14:45:00 | 1092.95 | 1100.84 | 1095.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 15:15:00 | 1094.00 | 1099.47 | 1095.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:15:00 | 1101.00 | 1099.47 | 1095.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 1103.30 | 1100.24 | 1096.26 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 09:15:00 | 1082.75 | 1096.86 | 1097.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 10:15:00 | 1073.70 | 1092.23 | 1095.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 09:15:00 | 1091.80 | 1087.58 | 1091.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 1091.80 | 1087.58 | 1091.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 1091.80 | 1087.58 | 1091.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 09:30:00 | 1092.50 | 1087.58 | 1091.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 1093.45 | 1088.75 | 1091.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 11:15:00 | 1090.30 | 1088.75 | 1091.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 12:30:00 | 1090.35 | 1087.89 | 1089.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 14:15:00 | 1087.80 | 1088.43 | 1089.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 11:15:00 | 1090.25 | 1087.79 | 1088.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 11:15:00 | 1096.00 | 1089.43 | 1089.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 11:15:00 | 1096.00 | 1089.43 | 1089.26 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 1046.65 | 1082.51 | 1086.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 11:15:00 | 1034.15 | 1066.49 | 1078.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 1065.50 | 1056.91 | 1067.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 1065.50 | 1056.91 | 1067.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1065.50 | 1056.91 | 1067.69 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2025-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 15:15:00 | 1078.00 | 1072.78 | 1072.32 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 09:15:00 | 1068.10 | 1071.84 | 1071.94 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 1100.50 | 1074.29 | 1071.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 10:15:00 | 1106.90 | 1080.81 | 1074.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 13:15:00 | 1216.60 | 1218.76 | 1200.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 14:00:00 | 1216.60 | 1218.76 | 1200.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 1198.60 | 1213.10 | 1203.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 1199.10 | 1213.10 | 1203.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 1201.70 | 1210.82 | 1203.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 13:30:00 | 1206.40 | 1209.15 | 1203.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 15:15:00 | 1208.50 | 1209.33 | 1206.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 1162.50 | 1199.83 | 1202.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 1162.50 | 1199.83 | 1202.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 1150.50 | 1189.97 | 1198.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 1182.60 | 1174.52 | 1184.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 10:00:00 | 1182.60 | 1174.52 | 1184.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 1179.40 | 1175.50 | 1184.29 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 1201.40 | 1188.96 | 1188.06 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 09:15:00 | 1184.30 | 1188.95 | 1189.09 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 09:15:00 | 1207.40 | 1191.06 | 1189.68 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-05-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 14:15:00 | 1183.80 | 1189.07 | 1189.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-05 09:15:00 | 1177.20 | 1185.34 | 1187.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 13:15:00 | 1163.40 | 1162.70 | 1168.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 14:00:00 | 1163.40 | 1162.70 | 1168.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1172.50 | 1164.41 | 1167.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:30:00 | 1174.60 | 1164.41 | 1167.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 1172.10 | 1165.95 | 1168.03 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2025-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 12:15:00 | 1177.30 | 1169.28 | 1169.25 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 1159.60 | 1168.31 | 1168.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 14:15:00 | 1150.30 | 1159.35 | 1163.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 1194.70 | 1165.40 | 1165.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 1194.70 | 1165.40 | 1165.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1194.70 | 1165.40 | 1165.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 1194.70 | 1165.40 | 1165.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1199.10 | 1172.14 | 1168.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 1201.20 | 1184.41 | 1175.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 1197.60 | 1197.62 | 1187.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:00:00 | 1197.60 | 1197.62 | 1187.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 1196.30 | 1196.03 | 1189.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:30:00 | 1200.80 | 1193.88 | 1191.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:00:00 | 1204.90 | 1196.47 | 1193.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 10:30:00 | 1200.60 | 1204.95 | 1204.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 12:15:00 | 1201.90 | 1204.11 | 1204.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 1201.90 | 1204.11 | 1204.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 1198.90 | 1203.07 | 1203.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 1201.40 | 1200.94 | 1202.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 1201.40 | 1200.94 | 1202.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1201.40 | 1200.94 | 1202.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:45:00 | 1199.40 | 1200.94 | 1202.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1199.40 | 1200.64 | 1202.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 11:15:00 | 1197.10 | 1200.64 | 1202.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 10:15:00 | 1197.00 | 1189.78 | 1192.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 10:15:00 | 1208.30 | 1193.48 | 1194.24 | SL hit (close>static) qty=1.00 sl=1204.90 alert=retest2 |

### Cycle 91 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 1202.60 | 1195.31 | 1195.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 1221.90 | 1207.40 | 1201.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 15:15:00 | 1213.00 | 1213.01 | 1207.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 09:15:00 | 1197.50 | 1213.01 | 1207.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1201.60 | 1210.73 | 1207.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:30:00 | 1200.50 | 1210.73 | 1207.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1204.70 | 1209.52 | 1206.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:00:00 | 1204.70 | 1209.52 | 1206.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 1202.20 | 1207.88 | 1206.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 12:30:00 | 1204.60 | 1207.88 | 1206.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 1196.90 | 1205.68 | 1205.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:00:00 | 1196.90 | 1205.68 | 1205.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 14:15:00 | 1196.80 | 1203.91 | 1204.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 09:15:00 | 1187.70 | 1193.20 | 1194.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 11:15:00 | 1173.90 | 1171.58 | 1177.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-05 12:00:00 | 1173.90 | 1171.58 | 1177.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 1192.40 | 1168.54 | 1172.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:45:00 | 1187.80 | 1168.54 | 1172.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 1196.60 | 1174.15 | 1174.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:45:00 | 1200.20 | 1174.15 | 1174.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 12:15:00 | 1194.90 | 1178.30 | 1176.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 1224.10 | 1194.03 | 1185.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 1230.20 | 1230.27 | 1221.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 13:45:00 | 1231.20 | 1230.27 | 1221.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 1225.70 | 1229.47 | 1223.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:45:00 | 1224.10 | 1229.47 | 1223.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 1224.00 | 1228.37 | 1223.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 1224.00 | 1228.37 | 1223.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1220.90 | 1226.88 | 1223.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:00:00 | 1220.90 | 1226.88 | 1223.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1213.90 | 1224.28 | 1222.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 1213.90 | 1224.28 | 1222.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 1211.30 | 1219.85 | 1220.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 1204.00 | 1216.68 | 1219.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 1212.30 | 1208.54 | 1212.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 10:15:00 | 1212.30 | 1208.54 | 1212.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 1212.30 | 1208.54 | 1212.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 1212.30 | 1208.54 | 1212.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 1213.00 | 1209.43 | 1212.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:45:00 | 1211.90 | 1209.43 | 1212.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 1212.50 | 1210.05 | 1212.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:30:00 | 1213.40 | 1210.05 | 1212.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 1216.90 | 1211.42 | 1212.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 1216.90 | 1211.42 | 1212.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1215.60 | 1212.25 | 1213.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:15:00 | 1215.40 | 1212.25 | 1213.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 1215.40 | 1212.88 | 1213.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 1222.60 | 1212.88 | 1213.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 09:15:00 | 1221.40 | 1214.59 | 1214.13 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 09:15:00 | 1215.60 | 1218.40 | 1218.52 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 10:15:00 | 1224.00 | 1219.52 | 1219.02 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 09:15:00 | 1207.90 | 1217.13 | 1218.15 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 1227.10 | 1217.51 | 1217.38 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 11:15:00 | 1214.70 | 1218.72 | 1218.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-25 14:15:00 | 1213.00 | 1216.45 | 1217.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 09:15:00 | 1219.30 | 1216.14 | 1217.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 09:15:00 | 1219.30 | 1216.14 | 1217.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 1219.30 | 1216.14 | 1217.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 1219.30 | 1216.14 | 1217.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 1221.70 | 1217.26 | 1217.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 1221.70 | 1217.26 | 1217.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 11:15:00 | 1232.00 | 1220.20 | 1218.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 13:15:00 | 1236.00 | 1224.10 | 1221.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 09:15:00 | 1224.80 | 1226.55 | 1223.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 10:00:00 | 1224.80 | 1226.55 | 1223.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 1225.90 | 1226.42 | 1223.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:45:00 | 1224.90 | 1226.42 | 1223.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 1222.80 | 1225.70 | 1223.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 11:30:00 | 1222.50 | 1225.70 | 1223.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 1223.00 | 1225.16 | 1223.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:00:00 | 1223.00 | 1225.16 | 1223.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 1226.70 | 1225.47 | 1223.60 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2025-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 10:15:00 | 1213.50 | 1221.72 | 1222.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 12:15:00 | 1206.30 | 1217.54 | 1220.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 1179.40 | 1177.19 | 1184.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 10:00:00 | 1179.40 | 1177.19 | 1184.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 1178.60 | 1172.81 | 1176.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 1178.60 | 1172.81 | 1176.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 1175.50 | 1173.35 | 1176.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 1172.30 | 1173.35 | 1176.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:15:00 | 1174.00 | 1172.87 | 1175.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 12:00:00 | 1174.00 | 1173.10 | 1175.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 12:45:00 | 1174.30 | 1173.48 | 1175.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 1176.20 | 1174.02 | 1175.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 13:45:00 | 1175.90 | 1174.02 | 1175.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 1175.20 | 1174.26 | 1175.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 15:15:00 | 1173.10 | 1174.26 | 1175.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:00:00 | 1170.30 | 1173.28 | 1174.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:30:00 | 1172.80 | 1173.05 | 1174.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 11:30:00 | 1173.30 | 1172.92 | 1174.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 1167.50 | 1168.47 | 1170.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 15:00:00 | 1164.10 | 1167.30 | 1169.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 14:45:00 | 1164.60 | 1167.45 | 1168.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 11:15:00 | 1171.40 | 1168.87 | 1169.02 | SL hit (close>static) qty=1.00 sl=1171.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 12:15:00 | 1171.20 | 1169.33 | 1169.22 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 11:15:00 | 1168.50 | 1169.58 | 1169.62 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 1173.00 | 1170.05 | 1169.79 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 13:15:00 | 1163.50 | 1169.35 | 1169.80 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 13:15:00 | 1171.10 | 1169.59 | 1169.48 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 14:15:00 | 1168.20 | 1169.31 | 1169.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 15:15:00 | 1165.50 | 1168.55 | 1169.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 1105.70 | 1102.32 | 1116.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 10:00:00 | 1105.70 | 1102.32 | 1116.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 1106.70 | 1103.38 | 1107.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 14:15:00 | 1105.80 | 1104.07 | 1107.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 10:15:00 | 1078.10 | 1069.98 | 1069.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 10:15:00 | 1078.10 | 1069.98 | 1069.04 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 11:15:00 | 1067.10 | 1069.85 | 1069.99 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 12:15:00 | 1071.30 | 1070.14 | 1070.11 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 13:15:00 | 1069.70 | 1070.05 | 1070.07 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 1076.60 | 1071.36 | 1070.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 15:15:00 | 1079.50 | 1072.99 | 1071.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 09:15:00 | 1064.40 | 1071.27 | 1070.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 09:15:00 | 1064.40 | 1071.27 | 1070.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1064.40 | 1071.27 | 1070.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:00:00 | 1064.40 | 1071.27 | 1070.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 10:15:00 | 1061.50 | 1069.32 | 1069.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 1055.80 | 1063.46 | 1066.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 1063.40 | 1062.75 | 1065.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 10:00:00 | 1063.40 | 1062.75 | 1065.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 1065.90 | 1063.38 | 1065.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:00:00 | 1065.90 | 1063.38 | 1065.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 1065.40 | 1063.78 | 1065.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:15:00 | 1067.80 | 1063.78 | 1065.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 1069.10 | 1064.85 | 1066.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:45:00 | 1069.60 | 1064.85 | 1066.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 1072.40 | 1066.36 | 1066.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 13:45:00 | 1072.80 | 1066.36 | 1066.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 1073.80 | 1067.85 | 1067.28 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 11:15:00 | 1065.60 | 1068.28 | 1068.52 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 1086.40 | 1071.57 | 1069.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 10:15:00 | 1089.90 | 1082.70 | 1077.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 13:15:00 | 1084.10 | 1085.33 | 1080.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 14:00:00 | 1084.10 | 1085.33 | 1080.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 1079.90 | 1083.78 | 1080.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:30:00 | 1084.10 | 1082.76 | 1081.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 15:15:00 | 1077.10 | 1080.64 | 1080.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 1077.10 | 1080.64 | 1080.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 09:15:00 | 1073.30 | 1079.17 | 1080.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 11:15:00 | 1072.90 | 1072.73 | 1075.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 12:00:00 | 1072.90 | 1072.73 | 1075.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 1057.10 | 1054.31 | 1057.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 10:45:00 | 1058.20 | 1054.31 | 1057.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 1058.70 | 1055.19 | 1058.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 1058.70 | 1055.19 | 1058.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 1055.80 | 1055.31 | 1057.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 13:15:00 | 1055.00 | 1055.31 | 1057.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 11:30:00 | 1053.50 | 1053.21 | 1055.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 14:15:00 | 1061.80 | 1055.70 | 1055.91 | SL hit (close>static) qty=1.00 sl=1059.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 1062.90 | 1057.14 | 1056.54 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 13:15:00 | 1050.90 | 1056.59 | 1056.67 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 1064.00 | 1057.79 | 1057.17 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 11:15:00 | 1053.20 | 1056.50 | 1056.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 12:15:00 | 1046.20 | 1054.44 | 1055.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 14:15:00 | 1054.20 | 1053.86 | 1055.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-03 15:00:00 | 1054.20 | 1053.86 | 1055.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 1055.10 | 1054.11 | 1055.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 1057.40 | 1054.11 | 1055.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1056.80 | 1054.65 | 1055.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:45:00 | 1060.10 | 1054.65 | 1055.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 1056.30 | 1054.98 | 1055.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 11:30:00 | 1051.80 | 1054.50 | 1055.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 09:30:00 | 1052.20 | 1051.57 | 1053.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 14:00:00 | 1054.40 | 1051.91 | 1052.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 14:30:00 | 1054.70 | 1052.91 | 1053.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 15:15:00 | 1056.00 | 1053.53 | 1053.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2025-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 15:15:00 | 1056.00 | 1053.53 | 1053.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 11:15:00 | 1059.10 | 1055.89 | 1054.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 14:15:00 | 1055.70 | 1056.58 | 1055.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 14:15:00 | 1055.70 | 1056.58 | 1055.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 1055.70 | 1056.58 | 1055.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 15:00:00 | 1055.70 | 1056.58 | 1055.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 1057.20 | 1056.70 | 1055.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:15:00 | 1055.30 | 1056.70 | 1055.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1051.70 | 1055.70 | 1055.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:00:00 | 1051.70 | 1055.70 | 1055.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 1053.40 | 1055.24 | 1055.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:30:00 | 1050.40 | 1055.24 | 1055.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 11:15:00 | 1050.90 | 1054.37 | 1054.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 12:15:00 | 1049.60 | 1053.42 | 1054.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 15:15:00 | 1053.40 | 1052.90 | 1053.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 15:15:00 | 1053.40 | 1052.90 | 1053.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 1053.40 | 1052.90 | 1053.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 1060.20 | 1052.90 | 1053.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 1061.90 | 1054.70 | 1054.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 1069.90 | 1057.74 | 1055.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 13:15:00 | 1101.40 | 1103.83 | 1094.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 14:00:00 | 1101.40 | 1103.83 | 1094.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1160.20 | 1159.52 | 1154.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:45:00 | 1157.70 | 1159.52 | 1154.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 1161.40 | 1159.89 | 1155.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:30:00 | 1159.80 | 1159.89 | 1155.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 1162.80 | 1164.31 | 1160.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-26 10:30:00 | 1168.20 | 1164.51 | 1160.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 12:15:00 | 1151.20 | 1161.06 | 1159.59 | SL hit (close<static) qty=1.00 sl=1155.10 alert=retest2 |

### Cycle 126 — SELL (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 14:15:00 | 1154.10 | 1157.96 | 1158.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 09:15:00 | 1137.00 | 1153.14 | 1156.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 12:15:00 | 1133.60 | 1133.58 | 1140.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 13:00:00 | 1133.60 | 1133.58 | 1140.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1140.60 | 1134.23 | 1138.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:00:00 | 1140.60 | 1134.23 | 1138.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 1143.90 | 1136.16 | 1139.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:00:00 | 1143.90 | 1136.16 | 1139.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 1155.70 | 1140.07 | 1140.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:45:00 | 1163.80 | 1140.07 | 1140.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 1159.50 | 1143.96 | 1142.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 1181.60 | 1157.72 | 1149.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 09:15:00 | 1197.60 | 1200.32 | 1185.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 09:45:00 | 1195.00 | 1200.32 | 1185.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 1184.80 | 1193.65 | 1187.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:00:00 | 1184.80 | 1193.65 | 1187.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 1188.00 | 1192.52 | 1187.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:15:00 | 1187.90 | 1192.52 | 1187.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1180.20 | 1190.06 | 1186.93 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 13:15:00 | 1181.00 | 1184.31 | 1184.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 1174.80 | 1181.24 | 1183.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 1182.60 | 1173.39 | 1176.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 1182.60 | 1173.39 | 1176.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1182.60 | 1173.39 | 1176.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 1181.00 | 1173.39 | 1176.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 1184.90 | 1175.69 | 1177.65 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 1189.40 | 1180.58 | 1179.66 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 1169.20 | 1180.18 | 1181.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 12:15:00 | 1168.40 | 1177.82 | 1179.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 15:15:00 | 1171.00 | 1170.17 | 1173.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-16 09:15:00 | 1203.50 | 1170.17 | 1173.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1193.70 | 1174.88 | 1175.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:15:00 | 1187.40 | 1174.88 | 1175.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 10:15:00 | 1192.70 | 1178.44 | 1177.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 1192.70 | 1178.44 | 1177.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 13:15:00 | 1195.80 | 1186.07 | 1181.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 1194.00 | 1197.01 | 1189.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 13:00:00 | 1194.00 | 1197.01 | 1189.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1245.40 | 1250.13 | 1246.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:00:00 | 1245.40 | 1250.13 | 1246.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 1244.60 | 1249.02 | 1246.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:15:00 | 1242.40 | 1249.02 | 1246.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 1243.20 | 1247.86 | 1245.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:30:00 | 1241.90 | 1247.86 | 1245.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 1245.10 | 1247.31 | 1245.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:15:00 | 1242.70 | 1247.31 | 1245.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 1241.00 | 1246.05 | 1245.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:45:00 | 1239.60 | 1246.05 | 1245.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 1247.00 | 1246.25 | 1245.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 1243.50 | 1246.25 | 1245.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1243.50 | 1245.70 | 1245.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:30:00 | 1240.00 | 1245.70 | 1245.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 10:15:00 | 1241.40 | 1244.84 | 1244.96 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 1246.10 | 1245.07 | 1245.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 13:15:00 | 1249.20 | 1245.89 | 1245.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 1239.50 | 1245.57 | 1245.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 1239.50 | 1245.57 | 1245.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1239.50 | 1245.57 | 1245.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 1239.50 | 1245.57 | 1245.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 10:15:00 | 1243.30 | 1245.12 | 1245.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 13:15:00 | 1234.90 | 1239.48 | 1241.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 10:15:00 | 1239.30 | 1236.68 | 1239.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 10:15:00 | 1239.30 | 1236.68 | 1239.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 1239.30 | 1236.68 | 1239.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:00:00 | 1239.30 | 1236.68 | 1239.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 1234.00 | 1236.14 | 1238.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 1228.10 | 1235.50 | 1237.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:00:00 | 1229.50 | 1229.09 | 1232.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 12:00:00 | 1230.90 | 1229.50 | 1232.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 13:45:00 | 1230.30 | 1230.42 | 1232.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 1228.20 | 1229.97 | 1231.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 14:30:00 | 1230.10 | 1229.97 | 1231.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1226.20 | 1225.15 | 1227.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 1228.90 | 1225.15 | 1227.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1224.40 | 1225.00 | 1227.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 11:30:00 | 1221.30 | 1224.72 | 1226.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:45:00 | 1222.00 | 1224.48 | 1226.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 11:15:00 | 1226.30 | 1223.40 | 1223.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 11:15:00 | 1226.30 | 1223.40 | 1223.21 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 15:15:00 | 1216.50 | 1222.98 | 1223.22 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 10:15:00 | 1227.10 | 1223.45 | 1223.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 11:15:00 | 1231.20 | 1225.00 | 1224.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 1225.30 | 1225.54 | 1224.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 13:15:00 | 1225.30 | 1225.54 | 1224.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 1225.30 | 1225.54 | 1224.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 1225.30 | 1225.54 | 1224.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 1224.20 | 1225.27 | 1224.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 1224.20 | 1225.27 | 1224.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 1228.30 | 1225.88 | 1224.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 1232.90 | 1225.88 | 1224.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1230.70 | 1226.84 | 1225.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:45:00 | 1238.40 | 1229.39 | 1226.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:00:00 | 1238.10 | 1231.14 | 1227.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 10:15:00 | 1271.60 | 1274.69 | 1275.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2025-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 10:15:00 | 1271.60 | 1274.69 | 1275.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 14:15:00 | 1265.60 | 1272.03 | 1273.60 | Break + close below crossover candle low |

### Cycle 139 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 1287.50 | 1274.34 | 1274.33 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 13:15:00 | 1276.20 | 1281.36 | 1281.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 11:15:00 | 1270.70 | 1278.50 | 1280.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 11:15:00 | 1268.40 | 1264.68 | 1269.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 12:00:00 | 1268.40 | 1264.68 | 1269.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 1268.80 | 1265.50 | 1269.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 13:00:00 | 1268.80 | 1265.50 | 1269.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 1266.50 | 1265.70 | 1269.00 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 10:15:00 | 1278.30 | 1271.87 | 1271.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 1283.10 | 1278.73 | 1276.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 1270.30 | 1277.92 | 1276.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 1270.30 | 1277.92 | 1276.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 1270.30 | 1277.92 | 1276.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 1270.30 | 1277.92 | 1276.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 1266.80 | 1275.70 | 1275.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:00:00 | 1266.80 | 1275.70 | 1275.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 1268.90 | 1274.34 | 1274.72 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 12:15:00 | 1278.50 | 1274.31 | 1274.19 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 12:15:00 | 1271.70 | 1274.65 | 1274.70 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 13:15:00 | 1277.50 | 1275.22 | 1274.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 14:15:00 | 1279.80 | 1276.13 | 1275.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 15:15:00 | 1275.00 | 1275.91 | 1275.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 15:15:00 | 1275.00 | 1275.91 | 1275.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 1275.00 | 1275.91 | 1275.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:30:00 | 1273.00 | 1276.61 | 1275.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 1276.50 | 1276.58 | 1275.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:00:00 | 1276.50 | 1276.58 | 1275.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 1278.00 | 1276.87 | 1276.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:45:00 | 1276.20 | 1276.87 | 1276.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 1272.50 | 1275.99 | 1275.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:00:00 | 1272.50 | 1275.99 | 1275.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 1273.80 | 1275.56 | 1275.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 14:15:00 | 1277.00 | 1275.56 | 1275.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 14:15:00 | 1272.30 | 1274.90 | 1275.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 14:15:00 | 1272.30 | 1274.90 | 1275.22 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 1284.70 | 1277.17 | 1276.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 11:15:00 | 1289.60 | 1280.91 | 1278.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1283.20 | 1284.15 | 1281.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 1283.20 | 1284.15 | 1281.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1283.20 | 1284.15 | 1281.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:30:00 | 1281.50 | 1284.15 | 1281.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 13:15:00 | 1284.50 | 1285.32 | 1282.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 13:45:00 | 1283.00 | 1285.32 | 1282.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 1284.40 | 1285.13 | 1283.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 1254.20 | 1285.13 | 1283.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 1237.30 | 1275.56 | 1278.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 1230.30 | 1254.99 | 1267.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 09:15:00 | 1231.60 | 1230.27 | 1241.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 10:00:00 | 1231.60 | 1230.27 | 1241.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1234.90 | 1231.93 | 1237.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 1239.70 | 1231.93 | 1237.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 1238.00 | 1233.15 | 1237.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 1239.30 | 1233.15 | 1237.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 1233.80 | 1233.28 | 1236.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 14:45:00 | 1230.20 | 1232.78 | 1235.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 10:00:00 | 1231.40 | 1232.28 | 1235.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 09:45:00 | 1230.80 | 1231.58 | 1233.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 11:00:00 | 1230.60 | 1231.39 | 1232.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1232.80 | 1229.51 | 1231.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:45:00 | 1233.30 | 1229.51 | 1231.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 1233.00 | 1230.21 | 1231.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:15:00 | 1234.60 | 1230.21 | 1231.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 1231.30 | 1230.43 | 1231.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:30:00 | 1232.50 | 1230.43 | 1231.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 1228.20 | 1229.98 | 1231.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 13:15:00 | 1226.20 | 1229.98 | 1231.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 14:45:00 | 1227.10 | 1228.88 | 1230.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 11:15:00 | 1226.30 | 1227.58 | 1229.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 1221.00 | 1227.12 | 1228.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1220.80 | 1225.86 | 1227.60 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-29 14:15:00 | 1231.80 | 1228.13 | 1228.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 14:15:00 | 1231.80 | 1228.13 | 1228.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 15:15:00 | 1233.00 | 1229.11 | 1228.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 13:15:00 | 1267.60 | 1272.08 | 1265.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 13:15:00 | 1267.60 | 1272.08 | 1265.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 1267.60 | 1272.08 | 1265.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:45:00 | 1266.10 | 1272.08 | 1265.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 1266.40 | 1270.94 | 1266.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 15:00:00 | 1266.40 | 1270.94 | 1266.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 1269.00 | 1270.55 | 1266.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 1284.00 | 1270.55 | 1266.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 09:15:00 | 1282.20 | 1287.85 | 1287.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 1282.20 | 1287.85 | 1287.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 11:15:00 | 1272.50 | 1283.57 | 1285.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 1273.80 | 1272.76 | 1277.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 1275.00 | 1272.76 | 1277.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1270.30 | 1272.53 | 1276.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:30:00 | 1266.10 | 1270.04 | 1274.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 09:15:00 | 1288.10 | 1270.30 | 1272.49 | SL hit (close>static) qty=1.00 sl=1284.00 alert=retest2 |

### Cycle 151 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 1300.00 | 1276.24 | 1274.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 09:15:00 | 1304.10 | 1295.61 | 1290.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 1301.70 | 1302.22 | 1296.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-20 10:00:00 | 1301.70 | 1302.22 | 1296.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 1301.90 | 1302.16 | 1297.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:00:00 | 1301.90 | 1302.16 | 1297.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 1293.30 | 1300.36 | 1298.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 15:00:00 | 1293.30 | 1300.36 | 1298.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 1297.00 | 1299.69 | 1297.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:15:00 | 1291.80 | 1299.69 | 1297.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2026-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 09:15:00 | 1282.40 | 1296.23 | 1296.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 10:15:00 | 1273.90 | 1291.77 | 1294.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1290.60 | 1286.31 | 1289.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 1290.60 | 1286.31 | 1289.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1290.60 | 1286.31 | 1289.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 1294.10 | 1286.31 | 1289.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1283.30 | 1285.71 | 1289.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:30:00 | 1278.90 | 1284.12 | 1287.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 09:15:00 | 1322.50 | 1281.88 | 1283.79 | SL hit (close>static) qty=1.00 sl=1292.10 alert=retest2 |

### Cycle 153 — BUY (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 10:15:00 | 1327.10 | 1290.93 | 1287.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 09:15:00 | 1337.60 | 1313.87 | 1301.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-28 13:15:00 | 1315.80 | 1320.38 | 1309.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-28 13:45:00 | 1316.00 | 1320.38 | 1309.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 1319.80 | 1320.26 | 1310.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 09:30:00 | 1323.50 | 1321.17 | 1312.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 10:00:00 | 1329.80 | 1321.17 | 1312.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 09:15:00 | 1332.40 | 1349.33 | 1349.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 1332.40 | 1349.33 | 1349.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 1323.20 | 1344.11 | 1347.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1348.80 | 1327.57 | 1335.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 1348.80 | 1327.57 | 1335.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1348.80 | 1327.57 | 1335.35 | EMA400 retest candle locked (from downside) |

### Cycle 155 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 1356.80 | 1342.01 | 1340.80 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2026-02-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 15:15:00 | 1340.00 | 1344.12 | 1344.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 09:15:00 | 1328.20 | 1340.94 | 1342.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 15:15:00 | 1331.60 | 1330.34 | 1335.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 09:15:00 | 1334.60 | 1330.34 | 1335.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1333.60 | 1330.99 | 1335.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:30:00 | 1328.00 | 1330.99 | 1335.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1338.80 | 1332.55 | 1335.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:00:00 | 1338.80 | 1332.55 | 1335.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 1341.30 | 1334.30 | 1336.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:30:00 | 1344.60 | 1334.30 | 1336.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 1337.90 | 1335.02 | 1336.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:45:00 | 1343.20 | 1335.02 | 1336.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 1341.40 | 1337.43 | 1337.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 1345.30 | 1339.88 | 1338.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 1351.10 | 1351.88 | 1347.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 10:00:00 | 1351.10 | 1351.88 | 1347.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 1348.20 | 1350.85 | 1347.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:30:00 | 1347.90 | 1350.85 | 1347.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 1347.40 | 1350.16 | 1347.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:45:00 | 1347.90 | 1350.16 | 1347.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 1348.30 | 1349.79 | 1347.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 14:30:00 | 1347.50 | 1349.79 | 1347.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 1347.00 | 1349.23 | 1347.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 1353.50 | 1349.23 | 1347.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 1343.60 | 1348.11 | 1347.49 | SL hit (close<static) qty=1.00 sl=1346.20 alert=retest2 |

### Cycle 158 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 1339.90 | 1346.46 | 1346.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 11:15:00 | 1337.90 | 1344.75 | 1345.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 13:15:00 | 1334.80 | 1332.58 | 1337.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 14:00:00 | 1334.80 | 1332.58 | 1337.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 1341.00 | 1333.97 | 1337.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:00:00 | 1341.00 | 1333.97 | 1337.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 1348.80 | 1336.93 | 1338.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:45:00 | 1350.00 | 1336.93 | 1338.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 11:15:00 | 1349.70 | 1339.49 | 1339.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 12:15:00 | 1358.40 | 1343.27 | 1340.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 13:15:00 | 1353.40 | 1353.52 | 1348.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-17 13:30:00 | 1353.00 | 1353.52 | 1348.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 1365.60 | 1368.61 | 1363.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:45:00 | 1362.50 | 1368.61 | 1363.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 1359.60 | 1366.81 | 1362.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:00:00 | 1359.60 | 1366.81 | 1362.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 1358.50 | 1365.15 | 1362.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:45:00 | 1355.60 | 1365.15 | 1362.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 1400.20 | 1397.65 | 1391.39 | EMA400 retest candle locked (from upside) |

### Cycle 160 — SELL (started 2026-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 12:15:00 | 1384.80 | 1390.38 | 1391.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 1383.60 | 1388.96 | 1390.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 1374.30 | 1373.27 | 1380.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 15:00:00 | 1374.30 | 1373.27 | 1380.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 1344.70 | 1346.12 | 1354.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 1354.80 | 1346.12 | 1354.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 1309.80 | 1296.89 | 1309.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 11:00:00 | 1309.80 | 1296.89 | 1309.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 1311.00 | 1299.72 | 1309.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 11:45:00 | 1314.20 | 1299.72 | 1309.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 1312.80 | 1302.33 | 1309.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 13:00:00 | 1312.80 | 1302.33 | 1309.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 1316.70 | 1305.21 | 1310.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:00:00 | 1316.70 | 1305.21 | 1310.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 1315.30 | 1307.22 | 1311.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 15:00:00 | 1315.30 | 1307.22 | 1311.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 1318.10 | 1309.40 | 1311.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 09:15:00 | 1303.60 | 1309.40 | 1311.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 1238.42 | 1268.77 | 1286.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 14:15:00 | 1216.50 | 1204.61 | 1219.94 | SL hit (close>ema200) qty=0.50 sl=1204.61 alert=retest2 |

### Cycle 161 — BUY (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 15:15:00 | 1228.00 | 1222.74 | 1222.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 1242.50 | 1226.69 | 1224.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1212.20 | 1238.97 | 1234.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1212.20 | 1238.97 | 1234.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1212.20 | 1238.97 | 1234.38 | EMA400 retest candle locked (from upside) |

### Cycle 162 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 1221.40 | 1230.10 | 1230.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 1210.80 | 1226.24 | 1229.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1196.10 | 1182.20 | 1191.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 1196.10 | 1182.20 | 1191.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1196.10 | 1182.20 | 1191.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 1194.80 | 1182.20 | 1191.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1200.00 | 1185.76 | 1192.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 1200.00 | 1185.76 | 1192.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 1194.80 | 1188.29 | 1192.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 1214.50 | 1188.29 | 1192.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1213.70 | 1193.37 | 1194.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:45:00 | 1216.80 | 1193.37 | 1194.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 1227.00 | 1200.10 | 1197.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 1234.30 | 1206.94 | 1200.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1206.40 | 1215.39 | 1208.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1206.40 | 1215.39 | 1208.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1206.40 | 1215.39 | 1208.55 | EMA400 retest candle locked (from upside) |

### Cycle 164 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 1173.80 | 1202.10 | 1204.93 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 15:15:00 | 1200.00 | 1193.07 | 1192.24 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 1168.20 | 1188.09 | 1190.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 11:15:00 | 1155.00 | 1177.46 | 1184.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 1190.10 | 1178.47 | 1183.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 13:15:00 | 1190.10 | 1178.47 | 1183.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 13:15:00 | 1190.10 | 1178.47 | 1183.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 14:00:00 | 1190.10 | 1178.47 | 1183.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 1199.70 | 1182.72 | 1185.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 15:00:00 | 1199.70 | 1182.72 | 1185.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — BUY (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 09:15:00 | 1204.80 | 1189.60 | 1188.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 11:15:00 | 1220.10 | 1199.23 | 1192.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 1223.60 | 1224.38 | 1210.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 1223.60 | 1224.38 | 1210.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1329.30 | 1341.59 | 1324.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1336.70 | 1341.59 | 1324.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 09:15:00 | 1360.80 | 1371.15 | 1371.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 1360.80 | 1371.15 | 1371.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 1353.60 | 1367.64 | 1369.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 1366.20 | 1363.62 | 1366.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 14:15:00 | 1366.20 | 1363.62 | 1366.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 1366.20 | 1363.62 | 1366.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:45:00 | 1368.40 | 1363.62 | 1366.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 1362.00 | 1363.29 | 1366.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 09:15:00 | 1308.90 | 1363.29 | 1366.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 1300.60 | 1277.24 | 1274.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 1300.60 | 1277.24 | 1274.60 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 1270.10 | 1279.49 | 1279.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 13:15:00 | 1265.50 | 1276.69 | 1278.64 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 09:15:00 | 1113.70 | 2024-05-13 13:15:00 | 1130.20 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-05-13 10:15:00 | 1110.55 | 2024-05-13 15:15:00 | 1133.90 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2024-05-21 11:15:00 | 1140.45 | 2024-05-22 10:15:00 | 1124.60 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-05-31 10:15:00 | 1162.40 | 2024-06-03 09:15:00 | 1207.95 | STOP_HIT | 1.00 | -3.92% |
| SELL | retest2 | 2024-05-31 12:45:00 | 1163.95 | 2024-06-03 09:15:00 | 1207.95 | STOP_HIT | 1.00 | -3.78% |
| SELL | retest2 | 2024-05-31 14:00:00 | 1164.05 | 2024-06-03 09:15:00 | 1207.95 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest2 | 2024-06-07 09:15:00 | 1177.50 | 2024-06-12 15:15:00 | 1186.00 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2024-06-07 10:30:00 | 1178.65 | 2024-06-12 15:15:00 | 1186.00 | STOP_HIT | 1.00 | 0.62% |
| BUY | retest2 | 2024-06-25 12:15:00 | 1253.80 | 2024-07-01 09:15:00 | 1267.05 | STOP_HIT | 1.00 | 1.06% |
| BUY | retest2 | 2024-07-04 14:30:00 | 1281.50 | 2024-07-11 10:15:00 | 1279.90 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2024-07-05 09:15:00 | 1281.95 | 2024-07-11 11:15:00 | 1279.45 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2024-07-05 10:15:00 | 1283.85 | 2024-07-11 11:15:00 | 1279.45 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2024-07-10 09:30:00 | 1282.55 | 2024-07-11 11:15:00 | 1279.45 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2024-07-10 12:15:00 | 1288.05 | 2024-07-11 11:15:00 | 1279.45 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-08-02 09:30:00 | 1162.50 | 2024-08-09 09:15:00 | 1150.95 | STOP_HIT | 1.00 | 0.99% |
| BUY | retest2 | 2024-09-02 09:45:00 | 1181.15 | 2024-09-04 12:15:00 | 1175.35 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2024-09-03 11:30:00 | 1180.45 | 2024-09-04 12:15:00 | 1175.35 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2024-09-12 12:30:00 | 1185.75 | 2024-09-24 14:15:00 | 1239.50 | STOP_HIT | 1.00 | 4.53% |
| SELL | retest2 | 2024-10-09 13:00:00 | 1168.45 | 2024-10-10 10:15:00 | 1180.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-10-09 13:30:00 | 1168.95 | 2024-10-10 10:15:00 | 1180.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-10-09 14:00:00 | 1169.00 | 2024-10-10 10:15:00 | 1180.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-10-25 11:30:00 | 1175.35 | 2024-10-25 12:15:00 | 1175.45 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2024-11-04 11:00:00 | 1145.80 | 2024-11-05 15:15:00 | 1170.15 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-11-04 14:00:00 | 1143.50 | 2024-11-05 15:15:00 | 1170.15 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2024-11-05 10:30:00 | 1138.40 | 2024-11-05 15:15:00 | 1170.15 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2024-11-21 09:15:00 | 1126.85 | 2024-11-21 10:15:00 | 1144.40 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-11-25 14:30:00 | 1153.85 | 2024-11-28 10:15:00 | 1140.90 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-11-26 10:00:00 | 1153.55 | 2024-11-28 10:15:00 | 1140.90 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-11-26 10:30:00 | 1150.00 | 2024-11-28 10:15:00 | 1140.90 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-11-27 12:15:00 | 1151.45 | 2024-11-28 10:15:00 | 1140.90 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-12-05 12:15:00 | 1168.45 | 2024-12-10 09:15:00 | 1163.35 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2024-12-05 13:00:00 | 1168.90 | 2024-12-10 09:15:00 | 1163.35 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-12-05 13:30:00 | 1170.50 | 2024-12-10 09:15:00 | 1163.35 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-12-05 15:15:00 | 1168.50 | 2024-12-10 09:15:00 | 1163.35 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2024-12-06 12:15:00 | 1190.10 | 2024-12-10 09:15:00 | 1163.35 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2024-12-06 13:00:00 | 1190.80 | 2024-12-10 09:15:00 | 1163.35 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2024-12-06 13:30:00 | 1190.25 | 2024-12-10 09:15:00 | 1163.35 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2024-12-16 09:15:00 | 1143.00 | 2024-12-20 09:15:00 | 1085.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 10:15:00 | 1146.00 | 2024-12-20 09:15:00 | 1088.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 10:45:00 | 1144.70 | 2024-12-20 09:15:00 | 1087.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 13:15:00 | 1146.45 | 2024-12-20 09:15:00 | 1089.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 11:30:00 | 1133.95 | 2024-12-20 13:15:00 | 1077.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 13:00:00 | 1134.65 | 2024-12-20 13:15:00 | 1077.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 09:15:00 | 1143.00 | 2024-12-24 10:15:00 | 1081.65 | STOP_HIT | 0.50 | 5.37% |
| SELL | retest2 | 2024-12-16 10:15:00 | 1146.00 | 2024-12-24 10:15:00 | 1081.65 | STOP_HIT | 0.50 | 5.62% |
| SELL | retest2 | 2024-12-16 10:45:00 | 1144.70 | 2024-12-24 10:15:00 | 1081.65 | STOP_HIT | 0.50 | 5.51% |
| SELL | retest2 | 2024-12-16 13:15:00 | 1146.45 | 2024-12-24 10:15:00 | 1081.65 | STOP_HIT | 0.50 | 5.65% |
| SELL | retest2 | 2024-12-17 11:30:00 | 1133.95 | 2024-12-24 10:15:00 | 1081.65 | STOP_HIT | 0.50 | 4.61% |
| SELL | retest2 | 2024-12-17 13:00:00 | 1134.65 | 2024-12-24 10:15:00 | 1081.65 | STOP_HIT | 0.50 | 4.67% |
| SELL | retest2 | 2025-01-02 09:15:00 | 1068.30 | 2025-01-02 10:15:00 | 1075.65 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-01-08 12:00:00 | 1064.95 | 2025-01-09 09:15:00 | 1072.45 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-01-08 12:30:00 | 1065.00 | 2025-01-09 09:15:00 | 1072.45 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-01-13 12:45:00 | 1052.90 | 2025-01-17 09:15:00 | 1000.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-14 12:00:00 | 1053.85 | 2025-01-17 09:15:00 | 1001.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-14 13:00:00 | 1053.40 | 2025-01-17 09:15:00 | 1000.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-14 14:30:00 | 1052.80 | 2025-01-17 09:15:00 | 1000.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-15 09:15:00 | 1042.00 | 2025-01-17 09:15:00 | 989.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-13 12:45:00 | 1052.90 | 2025-01-20 13:15:00 | 991.90 | STOP_HIT | 0.50 | 5.79% |
| SELL | retest2 | 2025-01-14 12:00:00 | 1053.85 | 2025-01-20 13:15:00 | 991.90 | STOP_HIT | 0.50 | 5.88% |
| SELL | retest2 | 2025-01-14 13:00:00 | 1053.40 | 2025-01-20 13:15:00 | 991.90 | STOP_HIT | 0.50 | 5.84% |
| SELL | retest2 | 2025-01-14 14:30:00 | 1052.80 | 2025-01-20 13:15:00 | 991.90 | STOP_HIT | 0.50 | 5.78% |
| SELL | retest2 | 2025-01-15 09:15:00 | 1042.00 | 2025-01-20 13:15:00 | 991.90 | STOP_HIT | 0.50 | 4.81% |
| BUY | retest2 | 2025-02-01 09:15:00 | 989.95 | 2025-02-10 13:15:00 | 1010.85 | STOP_HIT | 1.00 | 2.11% |
| BUY | retest2 | 2025-02-01 12:45:00 | 991.65 | 2025-02-10 13:15:00 | 1010.85 | STOP_HIT | 1.00 | 1.94% |
| BUY | retest2 | 2025-02-01 13:15:00 | 995.05 | 2025-02-10 13:15:00 | 1010.85 | STOP_HIT | 1.00 | 1.59% |
| BUY | retest2 | 2025-02-03 13:00:00 | 991.50 | 2025-02-10 13:15:00 | 1010.85 | STOP_HIT | 1.00 | 1.95% |
| BUY | retest2 | 2025-02-04 12:45:00 | 1014.45 | 2025-02-10 13:15:00 | 1010.85 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2025-02-10 10:45:00 | 1010.60 | 2025-02-10 13:15:00 | 1010.85 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2025-02-10 11:30:00 | 1010.20 | 2025-02-10 13:15:00 | 1010.85 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-02-10 12:15:00 | 1013.80 | 2025-02-10 13:15:00 | 1010.85 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-02-21 12:15:00 | 1012.80 | 2025-02-21 12:15:00 | 1005.50 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest1 | 2025-03-10 09:15:00 | 1045.05 | 2025-03-11 09:15:00 | 1027.25 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-04-02 11:15:00 | 1090.30 | 2025-04-04 11:15:00 | 1096.00 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-04-03 12:30:00 | 1090.35 | 2025-04-04 11:15:00 | 1096.00 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-04-03 14:15:00 | 1087.80 | 2025-04-04 11:15:00 | 1096.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-04-04 11:15:00 | 1090.25 | 2025-04-04 11:15:00 | 1096.00 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-04-23 13:30:00 | 1206.40 | 2025-04-25 09:15:00 | 1162.50 | STOP_HIT | 1.00 | -3.64% |
| BUY | retest2 | 2025-04-24 15:15:00 | 1208.50 | 2025-04-25 09:15:00 | 1162.50 | STOP_HIT | 1.00 | -3.81% |
| BUY | retest2 | 2025-05-15 10:30:00 | 1200.80 | 2025-05-20 12:15:00 | 1201.90 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-05-15 13:00:00 | 1204.90 | 2025-05-20 12:15:00 | 1201.90 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-05-20 10:30:00 | 1200.60 | 2025-05-20 12:15:00 | 1201.90 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2025-05-21 11:15:00 | 1197.10 | 2025-05-23 10:15:00 | 1208.30 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-05-23 10:15:00 | 1197.00 | 2025-05-23 10:15:00 | 1208.30 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-07-07 09:15:00 | 1172.30 | 2025-07-11 11:15:00 | 1171.40 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2025-07-07 11:15:00 | 1174.00 | 2025-07-11 11:15:00 | 1171.40 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2025-07-07 12:00:00 | 1174.00 | 2025-07-11 12:15:00 | 1171.20 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2025-07-07 12:45:00 | 1174.30 | 2025-07-11 12:15:00 | 1171.20 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2025-07-07 15:15:00 | 1173.10 | 2025-07-11 12:15:00 | 1171.20 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2025-07-08 10:00:00 | 1170.30 | 2025-07-11 12:15:00 | 1171.20 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-07-08 10:30:00 | 1172.80 | 2025-07-11 12:15:00 | 1171.20 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2025-07-08 11:30:00 | 1173.30 | 2025-07-11 12:15:00 | 1171.20 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2025-07-09 15:00:00 | 1164.10 | 2025-07-11 12:15:00 | 1171.20 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-07-10 14:45:00 | 1164.60 | 2025-07-11 12:15:00 | 1171.20 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-07-23 14:15:00 | 1105.80 | 2025-08-05 10:15:00 | 1078.10 | STOP_HIT | 1.00 | 2.50% |
| BUY | retest2 | 2025-08-21 09:30:00 | 1084.10 | 2025-08-21 15:15:00 | 1077.10 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-08-29 13:15:00 | 1055.00 | 2025-09-01 14:15:00 | 1061.80 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-09-01 11:30:00 | 1053.50 | 2025-09-01 14:15:00 | 1061.80 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-09-04 11:30:00 | 1051.80 | 2025-09-05 15:15:00 | 1056.00 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-09-05 09:30:00 | 1052.20 | 2025-09-05 15:15:00 | 1056.00 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-09-05 14:00:00 | 1054.40 | 2025-09-05 15:15:00 | 1056.00 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-09-05 14:30:00 | 1054.70 | 2025-09-05 15:15:00 | 1056.00 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-09-26 10:30:00 | 1168.20 | 2025-09-26 12:15:00 | 1151.20 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-10-16 10:15:00 | 1187.40 | 2025-10-16 10:15:00 | 1192.70 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-11-04 09:15:00 | 1228.10 | 2025-11-12 11:15:00 | 1226.30 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2025-11-06 10:00:00 | 1229.50 | 2025-11-12 11:15:00 | 1226.30 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2025-11-06 12:00:00 | 1230.90 | 2025-11-12 11:15:00 | 1226.30 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2025-11-06 13:45:00 | 1230.30 | 2025-11-12 11:15:00 | 1226.30 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2025-11-10 11:30:00 | 1221.30 | 2025-11-12 11:15:00 | 1226.30 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-11-10 13:45:00 | 1222.00 | 2025-11-12 11:15:00 | 1226.30 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2025-11-14 10:45:00 | 1238.40 | 2025-11-25 10:15:00 | 1271.60 | STOP_HIT | 1.00 | 2.68% |
| BUY | retest2 | 2025-11-14 12:00:00 | 1238.10 | 2025-11-25 10:15:00 | 1271.60 | STOP_HIT | 1.00 | 2.71% |
| BUY | retest2 | 2025-12-11 14:15:00 | 1277.00 | 2025-12-11 14:15:00 | 1272.30 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-12-19 14:45:00 | 1230.20 | 2025-12-29 14:15:00 | 1231.80 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-12-22 10:00:00 | 1231.40 | 2025-12-29 14:15:00 | 1231.80 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-12-23 09:45:00 | 1230.80 | 2025-12-29 14:15:00 | 1231.80 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-12-23 11:00:00 | 1230.60 | 2025-12-29 14:15:00 | 1231.80 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-12-24 13:15:00 | 1226.20 | 2025-12-29 14:15:00 | 1231.80 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-12-24 14:45:00 | 1227.10 | 2025-12-29 14:15:00 | 1231.80 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-12-26 11:15:00 | 1226.30 | 2025-12-29 14:15:00 | 1231.80 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-12-29 09:15:00 | 1221.00 | 2025-12-29 14:15:00 | 1231.80 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2026-01-05 09:15:00 | 1284.00 | 2026-01-09 09:15:00 | 1282.20 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2026-01-13 12:30:00 | 1266.10 | 2026-01-14 09:15:00 | 1288.10 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2026-01-23 11:30:00 | 1278.90 | 2026-01-27 09:15:00 | 1322.50 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest2 | 2026-01-29 09:30:00 | 1323.50 | 2026-02-02 09:15:00 | 1332.40 | STOP_HIT | 1.00 | 0.67% |
| BUY | retest2 | 2026-01-29 10:00:00 | 1329.80 | 2026-02-02 09:15:00 | 1332.40 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2026-02-12 09:15:00 | 1353.50 | 2026-02-12 09:15:00 | 1343.60 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2026-03-11 09:15:00 | 1303.60 | 2026-03-12 09:15:00 | 1238.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 09:15:00 | 1303.60 | 2026-03-16 14:15:00 | 1216.50 | STOP_HIT | 0.50 | 6.68% |
| BUY | retest2 | 2026-04-13 10:15:00 | 1336.70 | 2026-04-24 09:15:00 | 1360.80 | STOP_HIT | 1.00 | 1.80% |
| SELL | retest2 | 2026-04-27 09:15:00 | 1308.90 | 2026-05-06 15:15:00 | 1300.60 | STOP_HIT | 1.00 | 0.63% |
