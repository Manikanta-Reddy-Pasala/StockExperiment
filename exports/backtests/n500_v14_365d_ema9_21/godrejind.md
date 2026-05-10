# Godrej Industries Ltd. (GODREJIND)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 1202.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 79 |
| ALERT1 | 43 |
| ALERT2 | 42 |
| ALERT2_SKIP | 17 |
| ALERT3 | 107 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 72 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 76 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 86 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 61
- **Target hits / Stop hits / Partials:** 3 / 76 / 7
- **Avg / median % per leg:** 0.12% / -0.70%
- **Sum % (uncompounded):** 10.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 28 | 8 | 28.6% | 1 | 27 | 0 | -0.67% | -18.7% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -4.86% | -4.9% |
| BUY @ 3rd Alert (retest2) | 27 | 8 | 29.6% | 1 | 26 | 0 | -0.51% | -13.8% |
| SELL (all) | 58 | 17 | 29.3% | 2 | 49 | 7 | 0.50% | 29.3% |
| SELL @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 0 | 6 | 2 | 0.78% | 6.3% |
| SELL @ 3rd Alert (retest2) | 50 | 13 | 26.0% | 2 | 43 | 5 | 0.46% | 23.0% |
| retest1 (combined) | 9 | 4 | 44.4% | 0 | 7 | 2 | 0.16% | 1.4% |
| retest2 (combined) | 77 | 21 | 27.3% | 3 | 69 | 5 | 0.12% | 9.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1122.50 | 1101.40 | 1100.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1139.00 | 1119.63 | 1111.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 1132.50 | 1156.45 | 1144.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 09:15:00 | 1132.50 | 1156.45 | 1144.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 1132.50 | 1156.45 | 1144.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:45:00 | 1133.10 | 1156.45 | 1144.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 1131.50 | 1151.46 | 1143.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 10:45:00 | 1133.20 | 1151.46 | 1143.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 14:15:00 | 1130.10 | 1138.62 | 1139.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 15:15:00 | 1124.70 | 1135.84 | 1137.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 09:15:00 | 1143.80 | 1137.43 | 1138.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 09:15:00 | 1143.80 | 1137.43 | 1138.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 1143.80 | 1137.43 | 1138.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-16 10:30:00 | 1137.10 | 1137.36 | 1138.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-16 15:15:00 | 1140.00 | 1137.87 | 1138.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 15:15:00 | 1140.00 | 1138.30 | 1138.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-16 15:15:00 | 1140.00 | 1138.30 | 1138.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 15:15:00 | 1140.00 | 1138.30 | 1138.24 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 09:15:00 | 1135.10 | 1137.66 | 1137.95 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 1145.10 | 1139.15 | 1138.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 13:15:00 | 1150.00 | 1142.39 | 1140.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 12:15:00 | 1170.20 | 1170.77 | 1164.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-23 13:00:00 | 1170.20 | 1170.77 | 1164.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 1170.00 | 1170.61 | 1165.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 13:30:00 | 1171.50 | 1170.61 | 1165.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 1173.30 | 1170.99 | 1166.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 13:30:00 | 1180.00 | 1174.49 | 1171.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 1181.00 | 1175.47 | 1172.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 09:45:00 | 1180.80 | 1180.68 | 1178.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 10:15:00 | 1182.10 | 1180.68 | 1178.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1180.50 | 1180.64 | 1178.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 14:15:00 | 1185.60 | 1181.36 | 1179.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 15:15:00 | 1193.00 | 1182.09 | 1179.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 09:30:00 | 1186.60 | 1184.24 | 1181.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 11:15:00 | 1187.00 | 1183.59 | 1181.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 1192.00 | 1185.27 | 1182.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 12:15:00 | 1195.80 | 1185.27 | 1182.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 14:15:00 | 1194.50 | 1186.43 | 1183.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 10:00:00 | 1195.00 | 1190.53 | 1186.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 15:00:00 | 1194.40 | 1197.07 | 1192.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 1189.30 | 1195.52 | 1191.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:15:00 | 1189.00 | 1195.52 | 1191.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 1182.90 | 1192.99 | 1190.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 1182.90 | 1192.99 | 1190.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 1186.30 | 1191.66 | 1190.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:30:00 | 1183.80 | 1191.66 | 1190.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-04 12:15:00 | 1186.10 | 1189.58 | 1189.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-04 12:15:00 | 1186.10 | 1189.58 | 1189.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-04 12:15:00 | 1186.10 | 1189.58 | 1189.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-04 12:15:00 | 1186.10 | 1189.58 | 1189.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-04 12:15:00 | 1186.10 | 1189.58 | 1189.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-04 12:15:00 | 1186.10 | 1189.58 | 1189.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-04 12:15:00 | 1186.10 | 1189.58 | 1189.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-04 12:15:00 | 1186.10 | 1189.58 | 1189.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-04 12:15:00 | 1186.10 | 1189.58 | 1189.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-04 12:15:00 | 1186.10 | 1189.58 | 1189.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-04 12:15:00 | 1186.10 | 1189.58 | 1189.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-04 12:15:00 | 1186.10 | 1189.58 | 1189.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 12:15:00 | 1186.10 | 1189.58 | 1189.72 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 10:15:00 | 1195.60 | 1189.80 | 1189.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 09:15:00 | 1226.00 | 1197.70 | 1193.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 1339.10 | 1340.55 | 1304.19 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 10:30:00 | 1357.00 | 1342.44 | 1308.35 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 1323.80 | 1334.31 | 1317.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:15:00 | 1307.70 | 1334.31 | 1317.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1291.00 | 1325.65 | 1314.80 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-11 09:15:00 | 1291.00 | 1325.65 | 1314.80 | SL hit (close<ema400) qty=1.00 sl=1314.80 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-06-11 10:00:00 | 1291.00 | 1325.65 | 1314.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 1291.40 | 1318.80 | 1312.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:15:00 | 1288.40 | 1318.80 | 1312.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 12:15:00 | 1289.10 | 1306.57 | 1307.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 1273.10 | 1285.91 | 1292.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 1279.60 | 1278.49 | 1286.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 13:45:00 | 1280.50 | 1278.49 | 1286.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 1278.70 | 1271.99 | 1279.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 1278.70 | 1271.99 | 1279.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 1280.50 | 1273.69 | 1279.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 1280.50 | 1273.69 | 1279.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 1298.40 | 1278.64 | 1281.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:45:00 | 1290.50 | 1278.64 | 1281.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1303.00 | 1283.51 | 1283.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:45:00 | 1307.10 | 1283.51 | 1283.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-06-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 15:15:00 | 1306.60 | 1288.13 | 1285.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 1319.40 | 1294.38 | 1288.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 11:15:00 | 1293.00 | 1295.63 | 1290.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 12:00:00 | 1293.00 | 1295.63 | 1290.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 1329.10 | 1340.62 | 1331.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 10:00:00 | 1349.90 | 1342.47 | 1332.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 12:30:00 | 1365.00 | 1348.54 | 1338.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 1312.20 | 1341.14 | 1339.16 | SL hit (close<static) qty=1.00 sl=1315.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 1312.20 | 1341.14 | 1339.16 | SL hit (close<static) qty=1.00 sl=1315.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 11:15:00 | 1297.70 | 1332.45 | 1335.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-23 12:15:00 | 1287.30 | 1323.42 | 1331.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 15:15:00 | 1256.00 | 1252.66 | 1269.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-26 09:15:00 | 1259.40 | 1252.66 | 1269.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1248.40 | 1247.73 | 1257.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:30:00 | 1249.20 | 1247.73 | 1257.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 1247.30 | 1247.64 | 1256.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 11:00:00 | 1247.30 | 1247.64 | 1256.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 1253.20 | 1248.30 | 1253.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 1264.00 | 1248.30 | 1253.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1256.10 | 1249.86 | 1253.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 10:45:00 | 1251.30 | 1251.07 | 1253.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 11:15:00 | 1248.80 | 1251.07 | 1253.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 10:45:00 | 1248.40 | 1248.46 | 1250.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 09:15:00 | 1188.73 | 1217.69 | 1233.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 09:15:00 | 1186.36 | 1217.69 | 1233.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 09:15:00 | 1185.98 | 1217.69 | 1233.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-07-03 09:15:00 | 1126.17 | 1167.26 | 1196.00 | Target hit (10%) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-03 12:15:00 | 1161.00 | 1160.53 | 1185.28 | SL hit (close>ema200) qty=0.50 sl=1160.53 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-03 12:15:00 | 1161.00 | 1160.53 | 1185.28 | SL hit (close>ema200) qty=0.50 sl=1160.53 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 09:15:00 | 1164.80 | 1120.32 | 1115.07 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 09:15:00 | 1133.90 | 1141.17 | 1141.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 10:15:00 | 1126.60 | 1138.25 | 1140.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 11:15:00 | 1138.30 | 1138.26 | 1139.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 12:00:00 | 1138.30 | 1138.26 | 1139.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 1131.60 | 1136.93 | 1139.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 12:45:00 | 1128.00 | 1136.09 | 1138.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 11:45:00 | 1129.20 | 1130.70 | 1133.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 15:15:00 | 1127.80 | 1131.81 | 1133.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 1140.70 | 1132.17 | 1131.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 1140.70 | 1132.17 | 1131.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 1140.70 | 1132.17 | 1131.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 10:15:00 | 1140.70 | 1132.17 | 1131.34 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 15:15:00 | 1124.00 | 1130.77 | 1131.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 09:15:00 | 1102.20 | 1125.06 | 1128.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 14:15:00 | 1117.60 | 1112.95 | 1119.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 15:00:00 | 1117.60 | 1112.95 | 1119.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 1119.60 | 1114.28 | 1119.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 1122.60 | 1114.28 | 1119.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1120.10 | 1115.45 | 1119.92 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 14:15:00 | 1137.90 | 1123.41 | 1122.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 12:15:00 | 1144.80 | 1129.48 | 1125.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 15:15:00 | 1133.10 | 1135.48 | 1129.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 09:15:00 | 1137.40 | 1135.48 | 1129.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1134.50 | 1135.28 | 1130.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 1134.50 | 1135.28 | 1130.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 1130.00 | 1133.65 | 1130.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:30:00 | 1129.00 | 1133.65 | 1130.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 1136.50 | 1134.22 | 1130.99 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 1114.90 | 1129.42 | 1129.52 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 14:15:00 | 1135.90 | 1126.41 | 1125.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 15:15:00 | 1140.10 | 1129.15 | 1126.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 09:15:00 | 1128.00 | 1134.69 | 1132.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 1128.00 | 1134.69 | 1132.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 1128.00 | 1134.69 | 1132.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:00:00 | 1128.00 | 1134.69 | 1132.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 1096.20 | 1126.99 | 1128.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 09:15:00 | 1091.20 | 1100.78 | 1108.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 1102.00 | 1096.28 | 1101.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 1102.00 | 1096.28 | 1101.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 1102.00 | 1096.28 | 1101.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 13:15:00 | 1092.30 | 1096.89 | 1100.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 14:00:00 | 1093.80 | 1096.27 | 1100.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 09:15:00 | 1118.40 | 1100.59 | 1101.17 | SL hit (close>static) qty=1.00 sl=1106.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-13 09:15:00 | 1118.40 | 1100.59 | 1101.17 | SL hit (close>static) qty=1.00 sl=1106.50 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 10:15:00 | 1115.00 | 1103.47 | 1102.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 11:15:00 | 1123.50 | 1107.48 | 1104.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 13:15:00 | 1154.30 | 1154.65 | 1138.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 13:30:00 | 1153.70 | 1154.65 | 1138.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1285.90 | 1286.80 | 1281.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 15:15:00 | 1295.00 | 1286.23 | 1283.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 10:15:00 | 1271.40 | 1283.30 | 1282.85 | SL hit (close<static) qty=1.00 sl=1279.20 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 1270.30 | 1280.70 | 1281.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 1265.60 | 1272.81 | 1277.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 14:15:00 | 1202.00 | 1199.63 | 1209.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-03 15:00:00 | 1202.00 | 1199.63 | 1209.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1198.50 | 1199.08 | 1207.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 11:15:00 | 1186.20 | 1197.26 | 1206.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 11:15:00 | 1187.60 | 1194.32 | 1199.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 09:15:00 | 1203.60 | 1200.26 | 1200.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-08 09:15:00 | 1203.60 | 1200.26 | 1200.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 1203.60 | 1200.26 | 1200.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 14:15:00 | 1218.00 | 1206.62 | 1204.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 10:15:00 | 1207.90 | 1208.01 | 1205.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 10:15:00 | 1207.90 | 1208.01 | 1205.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 1207.90 | 1208.01 | 1205.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:45:00 | 1204.40 | 1208.01 | 1205.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 1201.70 | 1206.75 | 1205.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 1201.70 | 1206.75 | 1205.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 1210.50 | 1207.50 | 1205.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 14:45:00 | 1213.00 | 1207.82 | 1206.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 1213.80 | 1207.33 | 1206.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 11:15:00 | 1197.40 | 1206.01 | 1205.91 | SL hit (close<static) qty=1.00 sl=1201.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-11 11:15:00 | 1197.40 | 1206.01 | 1205.91 | SL hit (close<static) qty=1.00 sl=1201.70 alert=retest2 |

### Cycle 22 — SELL (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 12:15:00 | 1202.40 | 1205.29 | 1205.59 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 15:15:00 | 1212.00 | 1206.51 | 1206.04 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 10:15:00 | 1202.00 | 1205.37 | 1205.58 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 12:15:00 | 1209.00 | 1206.20 | 1205.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 1232.50 | 1213.53 | 1209.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 09:15:00 | 1236.50 | 1236.98 | 1225.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 09:30:00 | 1237.20 | 1236.98 | 1225.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1226.90 | 1234.97 | 1226.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:00:00 | 1226.90 | 1234.97 | 1226.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 1225.00 | 1232.97 | 1225.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:45:00 | 1221.80 | 1232.97 | 1225.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 1223.00 | 1230.98 | 1225.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:45:00 | 1222.10 | 1230.98 | 1225.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 09:15:00 | 1209.40 | 1222.77 | 1223.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 10:15:00 | 1198.50 | 1217.92 | 1220.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 13:15:00 | 1215.00 | 1211.19 | 1216.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-17 14:00:00 | 1215.00 | 1211.19 | 1216.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 1218.40 | 1212.63 | 1216.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 15:00:00 | 1218.40 | 1212.63 | 1216.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 1217.00 | 1213.50 | 1216.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 1231.20 | 1213.50 | 1216.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1223.90 | 1215.58 | 1217.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 12:45:00 | 1207.30 | 1215.66 | 1217.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 09:30:00 | 1214.50 | 1214.32 | 1216.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 13:15:00 | 1194.60 | 1188.88 | 1188.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-24 13:15:00 | 1194.60 | 1188.88 | 1188.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 13:15:00 | 1194.60 | 1188.88 | 1188.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 10:15:00 | 1219.20 | 1196.69 | 1192.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 15:15:00 | 1210.20 | 1211.55 | 1202.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 09:15:00 | 1206.50 | 1211.55 | 1202.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 1195.70 | 1208.38 | 1202.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:00:00 | 1195.70 | 1208.38 | 1202.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 1199.70 | 1206.64 | 1201.97 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 15:15:00 | 1199.00 | 1199.74 | 1199.78 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 15:15:00 | 1203.20 | 1199.08 | 1198.54 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 09:15:00 | 1172.70 | 1193.80 | 1196.19 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 1210.60 | 1198.91 | 1198.07 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 14:15:00 | 1186.80 | 1196.65 | 1197.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-03 10:15:00 | 1172.90 | 1189.03 | 1193.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 11:15:00 | 1072.70 | 1069.27 | 1084.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 12:00:00 | 1072.70 | 1069.27 | 1084.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 1076.90 | 1071.35 | 1081.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 15:00:00 | 1076.90 | 1071.35 | 1081.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 1062.10 | 1050.78 | 1060.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:00:00 | 1062.10 | 1050.78 | 1060.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 1058.60 | 1052.35 | 1060.33 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 13:15:00 | 1070.00 | 1062.64 | 1062.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 15:15:00 | 1072.00 | 1065.69 | 1063.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 10:15:00 | 1065.00 | 1065.89 | 1064.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 10:15:00 | 1065.00 | 1065.89 | 1064.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 1065.00 | 1065.89 | 1064.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:00:00 | 1065.00 | 1065.89 | 1064.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 1072.50 | 1067.21 | 1064.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 12:15:00 | 1086.40 | 1067.21 | 1064.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 11:30:00 | 1077.70 | 1077.19 | 1072.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 12:00:00 | 1077.90 | 1077.19 | 1072.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 14:45:00 | 1080.50 | 1075.51 | 1072.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 1097.90 | 1093.55 | 1085.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 1095.40 | 1093.55 | 1085.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1088.80 | 1094.29 | 1087.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 1088.80 | 1094.29 | 1087.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 1098.60 | 1095.15 | 1088.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 12:00:00 | 1103.50 | 1096.82 | 1089.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 10:15:00 | 1077.00 | 1091.44 | 1090.22 | SL hit (close<static) qty=1.00 sl=1085.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-24 12:15:00 | 1080.10 | 1089.01 | 1089.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-24 12:15:00 | 1080.10 | 1089.01 | 1089.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-24 12:15:00 | 1080.10 | 1089.01 | 1089.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-24 12:15:00 | 1080.10 | 1089.01 | 1089.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 1080.10 | 1089.01 | 1089.32 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 14:15:00 | 1093.10 | 1088.34 | 1087.93 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 1074.70 | 1085.29 | 1086.69 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 1094.90 | 1086.61 | 1086.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 09:15:00 | 1108.00 | 1095.05 | 1091.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 1104.60 | 1108.04 | 1100.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 15:00:00 | 1104.60 | 1108.04 | 1100.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 1114.00 | 1109.23 | 1101.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 1107.80 | 1109.23 | 1101.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1108.90 | 1109.17 | 1102.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 11:00:00 | 1117.00 | 1110.73 | 1103.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 1085.10 | 1105.81 | 1107.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 1085.10 | 1105.81 | 1107.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 15:15:00 | 1080.20 | 1093.17 | 1100.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 12:15:00 | 1065.20 | 1061.01 | 1068.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 12:15:00 | 1065.20 | 1061.01 | 1068.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 1065.20 | 1061.01 | 1068.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 12:30:00 | 1061.30 | 1061.01 | 1068.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 1063.80 | 1061.57 | 1068.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:30:00 | 1066.60 | 1061.57 | 1068.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 1069.50 | 1063.15 | 1068.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:30:00 | 1072.00 | 1063.15 | 1068.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 1065.00 | 1063.52 | 1068.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 09:15:00 | 1044.50 | 1063.52 | 1068.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 11:45:00 | 1064.60 | 1058.95 | 1064.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 11:15:00 | 1072.30 | 1066.27 | 1065.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-13 11:15:00 | 1072.30 | 1066.27 | 1065.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 11:15:00 | 1072.30 | 1066.27 | 1065.88 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 1062.10 | 1065.34 | 1065.63 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 11:15:00 | 1071.60 | 1066.70 | 1066.20 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 14:15:00 | 1062.80 | 1065.68 | 1065.87 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-11-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 15:15:00 | 1067.30 | 1066.01 | 1066.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 09:15:00 | 1070.30 | 1066.87 | 1066.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 13:15:00 | 1068.00 | 1068.17 | 1067.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 13:15:00 | 1068.00 | 1068.17 | 1067.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 1068.00 | 1068.17 | 1067.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:00:00 | 1068.00 | 1068.17 | 1067.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 1074.90 | 1069.52 | 1067.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:45:00 | 1073.40 | 1069.52 | 1067.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1058.30 | 1067.67 | 1067.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:45:00 | 1058.10 | 1067.67 | 1067.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 1054.60 | 1065.06 | 1066.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 1046.00 | 1058.12 | 1062.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 14:15:00 | 1058.30 | 1052.10 | 1056.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 14:15:00 | 1058.30 | 1052.10 | 1056.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 1058.30 | 1052.10 | 1056.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 15:00:00 | 1058.30 | 1052.10 | 1056.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 1070.00 | 1055.68 | 1058.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 09:15:00 | 1055.80 | 1055.68 | 1058.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 14:15:00 | 1058.40 | 1033.83 | 1032.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 1058.40 | 1033.83 | 1032.86 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 11:15:00 | 1035.00 | 1042.50 | 1043.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 1030.80 | 1037.89 | 1040.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 12:15:00 | 1039.80 | 1037.81 | 1039.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 12:15:00 | 1039.80 | 1037.81 | 1039.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 1039.80 | 1037.81 | 1039.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 12:45:00 | 1038.10 | 1037.81 | 1039.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 1040.80 | 1038.41 | 1039.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 14:00:00 | 1040.80 | 1038.41 | 1039.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 1041.30 | 1038.99 | 1039.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 1041.30 | 1038.99 | 1039.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1044.40 | 1039.98 | 1040.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 1044.40 | 1039.98 | 1040.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 1038.00 | 1039.58 | 1039.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:15:00 | 1034.80 | 1038.73 | 1039.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 1045.70 | 1021.28 | 1018.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 1045.70 | 1021.28 | 1018.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 10:15:00 | 1049.80 | 1026.98 | 1021.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 09:15:00 | 1026.90 | 1030.21 | 1025.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 1026.90 | 1030.21 | 1025.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 1026.90 | 1030.21 | 1025.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:45:00 | 1023.30 | 1030.21 | 1025.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 1022.40 | 1028.64 | 1025.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:00:00 | 1022.40 | 1028.64 | 1025.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 1018.10 | 1026.54 | 1024.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:45:00 | 1018.00 | 1026.54 | 1024.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 13:15:00 | 1016.60 | 1022.40 | 1023.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 15:15:00 | 1010.00 | 1018.62 | 1021.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 11:15:00 | 1017.20 | 1015.19 | 1018.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 11:15:00 | 1017.20 | 1015.19 | 1018.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 1017.20 | 1015.19 | 1018.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:45:00 | 1014.00 | 1015.19 | 1018.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 1017.00 | 1015.55 | 1018.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 13:15:00 | 1016.30 | 1015.55 | 1018.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 1017.60 | 1015.96 | 1018.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:15:00 | 1017.50 | 1015.96 | 1018.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 1017.20 | 1016.21 | 1018.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 09:15:00 | 1012.80 | 1016.57 | 1018.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 10:00:00 | 1012.00 | 1015.65 | 1017.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 1015.40 | 1016.97 | 1017.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 10:30:00 | 1016.00 | 1016.69 | 1017.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 1017.50 | 1016.86 | 1017.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:00:00 | 1017.50 | 1016.86 | 1017.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-16 12:15:00 | 1020.80 | 1017.64 | 1017.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-16 12:15:00 | 1020.80 | 1017.64 | 1017.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-16 12:15:00 | 1020.80 | 1017.64 | 1017.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-16 12:15:00 | 1020.80 | 1017.64 | 1017.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 12:15:00 | 1020.80 | 1017.64 | 1017.52 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 1015.70 | 1017.12 | 1017.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 15:15:00 | 1011.40 | 1015.98 | 1016.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 13:15:00 | 1013.50 | 1012.24 | 1014.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 13:15:00 | 1013.50 | 1012.24 | 1014.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 1013.50 | 1012.24 | 1014.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:00:00 | 1013.50 | 1012.24 | 1014.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 1017.80 | 1013.35 | 1014.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 1017.80 | 1013.35 | 1014.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 1015.70 | 1013.82 | 1014.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:30:00 | 1000.80 | 1011.68 | 1013.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 10:45:00 | 1003.00 | 1009.84 | 1012.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 11:30:00 | 1001.50 | 1008.33 | 1011.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 12:00:00 | 1002.30 | 1008.33 | 1011.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 1010.00 | 1007.49 | 1010.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:00:00 | 1010.00 | 1007.49 | 1010.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 1010.90 | 1008.17 | 1010.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 1010.90 | 1008.17 | 1010.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 1010.00 | 1008.54 | 1010.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:30:00 | 1006.00 | 1005.45 | 1008.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:15:00 | 996.50 | 1006.18 | 1008.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 14:15:00 | 1004.80 | 1005.46 | 1008.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 12:30:00 | 1007.20 | 1001.39 | 1001.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 13:15:00 | 1009.90 | 1003.09 | 1002.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 13:15:00 | 1009.90 | 1003.09 | 1002.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 13:15:00 | 1009.90 | 1003.09 | 1002.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 13:15:00 | 1009.90 | 1003.09 | 1002.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 13:15:00 | 1009.90 | 1003.09 | 1002.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 13:15:00 | 1009.90 | 1003.09 | 1002.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 13:15:00 | 1009.90 | 1003.09 | 1002.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-23 13:15:00 | 1009.90 | 1003.09 | 1002.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 13:15:00 | 1009.90 | 1003.09 | 1002.55 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 1000.90 | 1003.43 | 1003.55 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 15:15:00 | 1007.00 | 1003.81 | 1003.59 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 1001.90 | 1003.43 | 1003.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 997.70 | 1002.28 | 1002.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 15:15:00 | 1004.90 | 1000.25 | 1001.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 15:15:00 | 1004.90 | 1000.25 | 1001.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 1004.90 | 1000.25 | 1001.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 981.80 | 1000.25 | 1001.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 12:15:00 | 1008.00 | 981.87 | 984.96 | SL hit (close>static) qty=1.00 sl=1004.90 alert=retest2 |

### Cycle 55 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 1007.90 | 987.08 | 987.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 12:15:00 | 1021.00 | 1001.28 | 994.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 14:15:00 | 1019.80 | 1019.96 | 1011.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 14:30:00 | 1021.20 | 1019.96 | 1011.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 1011.20 | 1020.42 | 1015.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 1011.20 | 1020.42 | 1015.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 1011.20 | 1018.58 | 1015.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:15:00 | 1005.10 | 1018.58 | 1015.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 1018.70 | 1017.83 | 1016.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 15:15:00 | 1022.00 | 1018.28 | 1016.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:15:00 | 1027.20 | 1018.30 | 1016.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 09:15:00 | 1011.10 | 1027.47 | 1027.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-09 09:15:00 | 1011.10 | 1027.47 | 1027.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 1011.10 | 1027.47 | 1027.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 15:15:00 | 1003.00 | 1011.37 | 1016.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 12:15:00 | 1011.00 | 1007.34 | 1012.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 13:00:00 | 1011.00 | 1007.34 | 1012.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 1000.60 | 1005.99 | 1011.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:15:00 | 997.00 | 1004.79 | 1010.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:00:00 | 998.00 | 1001.98 | 1004.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 13:15:00 | 1013.50 | 1004.28 | 1005.02 | SL hit (close>static) qty=1.00 sl=1011.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-16 13:15:00 | 1013.50 | 1004.28 | 1005.02 | SL hit (close>static) qty=1.00 sl=1011.50 alert=retest2 |

### Cycle 57 — BUY (started 2026-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 14:15:00 | 1017.80 | 1006.99 | 1006.18 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 1001.00 | 1005.40 | 1005.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 13:15:00 | 999.50 | 1004.22 | 1005.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-20 09:15:00 | 1011.10 | 1003.61 | 1004.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 1011.10 | 1003.61 | 1004.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 1011.10 | 1003.61 | 1004.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 1010.50 | 1003.61 | 1004.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 1007.90 | 1004.47 | 1004.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 11:45:00 | 1001.00 | 1004.07 | 1004.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 12:30:00 | 1000.00 | 1003.56 | 1004.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 13:00:00 | 1001.50 | 1003.56 | 1004.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 14:00:00 | 997.80 | 1002.41 | 1003.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 987.60 | 999.45 | 1002.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 14:30:00 | 1001.60 | 999.45 | 1002.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 1000.00 | 999.56 | 1002.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 10:30:00 | 984.80 | 994.22 | 999.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 15:15:00 | 1004.80 | 998.87 | 998.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 15:15:00 | 1004.80 | 998.87 | 998.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 15:15:00 | 1004.80 | 998.87 | 998.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 15:15:00 | 1004.80 | 998.87 | 998.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 15:15:00 | 1004.80 | 998.87 | 998.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 1004.80 | 998.87 | 998.07 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 992.90 | 997.22 | 997.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 11:15:00 | 991.20 | 996.01 | 996.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 971.60 | 968.44 | 978.10 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 09:15:00 | 964.90 | 968.95 | 977.45 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 10:00:00 | 965.20 | 968.20 | 976.34 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 11:00:00 | 965.00 | 967.56 | 975.31 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-28 12:45:00 | 965.70 | 967.15 | 973.75 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 990.50 | 971.94 | 974.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 990.50 | 971.94 | 974.79 | SL hit (close>ema400) qty=1.00 sl=974.79 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 990.50 | 971.94 | 974.79 | SL hit (close>ema400) qty=1.00 sl=974.79 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 990.50 | 971.94 | 974.79 | SL hit (close>ema400) qty=1.00 sl=974.79 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 990.50 | 971.94 | 974.79 | SL hit (close>ema400) qty=1.00 sl=974.79 alert=retest1 |
| ALERT3_SIDEWAYS | 2026-01-28 15:00:00 | 990.50 | 971.94 | 974.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 991.50 | 975.85 | 976.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 981.10 | 975.85 | 976.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 10:15:00 | 977.00 | 976.63 | 976.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 10:15:00 | 977.00 | 976.63 | 976.62 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 11:15:00 | 973.40 | 975.99 | 976.32 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 13:15:00 | 995.40 | 979.76 | 977.97 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 15:15:00 | 970.00 | 978.29 | 978.90 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 982.50 | 979.49 | 979.36 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 971.30 | 977.97 | 978.70 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 15:15:00 | 982.00 | 979.52 | 979.29 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 965.20 | 976.66 | 978.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 955.65 | 972.45 | 975.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 11:15:00 | 975.30 | 973.02 | 975.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 11:30:00 | 975.00 | 973.02 | 975.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 971.85 | 972.79 | 975.54 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 987.60 | 979.10 | 977.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 1001.00 | 983.48 | 980.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 14:15:00 | 1007.55 | 1007.68 | 995.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-03 15:00:00 | 1007.55 | 1007.68 | 995.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 999.20 | 1008.17 | 1003.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 999.20 | 1008.17 | 1003.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 1007.85 | 1008.10 | 1004.08 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 988.75 | 1000.46 | 1001.56 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 1011.00 | 1001.63 | 1000.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 13:15:00 | 1023.20 | 1009.64 | 1004.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 1022.90 | 1031.59 | 1023.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 1022.90 | 1031.59 | 1023.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1022.90 | 1031.59 | 1023.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 1022.90 | 1031.59 | 1023.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1024.50 | 1030.17 | 1023.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:45:00 | 1023.00 | 1030.17 | 1023.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 1019.00 | 1027.19 | 1023.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:45:00 | 1019.05 | 1027.19 | 1023.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 1014.15 | 1024.58 | 1022.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:45:00 | 1013.00 | 1024.58 | 1022.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2026-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 15:15:00 | 1013.00 | 1021.21 | 1021.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 997.25 | 1016.42 | 1019.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 11:15:00 | 1014.10 | 1013.51 | 1017.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-12 11:45:00 | 1013.55 | 1013.51 | 1017.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 976.80 | 977.08 | 980.56 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 09:15:00 | 1021.50 | 988.78 | 984.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 10:15:00 | 1058.85 | 1002.79 | 991.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 15:15:00 | 1041.85 | 1044.38 | 1033.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-24 09:15:00 | 1034.05 | 1044.38 | 1033.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 1037.15 | 1042.93 | 1033.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 15:15:00 | 1053.50 | 1040.94 | 1035.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 1024.70 | 1034.63 | 1035.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 09:15:00 | 1024.70 | 1034.63 | 1035.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 1003.00 | 1023.74 | 1029.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 09:15:00 | 936.85 | 935.42 | 948.98 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 929.50 | 934.43 | 947.29 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-09 09:15:00 | 929.10 | 937.50 | 944.10 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 922.60 | 921.10 | 929.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 10:30:00 | 915.00 | 919.69 | 928.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 09:30:00 | 918.85 | 915.64 | 922.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 883.02 | 897.20 | 908.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 882.64 | 897.20 | 908.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 872.91 | 897.20 | 908.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 11:15:00 | 897.95 | 895.91 | 906.15 | SL hit (close>ema200) qty=0.50 sl=895.91 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-12 11:15:00 | 897.95 | 895.91 | 906.15 | SL hit (close>ema200) qty=0.50 sl=895.91 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-12 11:15:00 | 897.95 | 895.91 | 906.15 | SL hit (close>ema200) qty=0.50 sl=895.91 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:15:00 | 869.25 | 885.09 | 896.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-16 09:15:00 | 823.50 | 852.43 | 873.22 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 75 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 812.20 | 806.29 | 805.83 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 786.35 | 802.68 | 804.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 784.05 | 798.95 | 802.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 836.25 | 775.35 | 780.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 836.25 | 775.35 | 780.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 836.25 | 775.35 | 780.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 836.25 | 775.35 | 780.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 873.50 | 794.98 | 788.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 886.90 | 857.13 | 843.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 875.00 | 877.94 | 863.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 875.00 | 877.94 | 863.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 876.00 | 877.65 | 870.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 891.20 | 877.65 | 870.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-23 09:15:00 | 980.32 | 948.83 | 936.51 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 10:15:00 | 959.20 | 980.69 | 982.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 955.30 | 964.74 | 972.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 11:15:00 | 967.85 | 964.25 | 970.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-30 11:30:00 | 965.10 | 964.25 | 970.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 967.55 | 964.91 | 970.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 15:00:00 | 961.60 | 964.46 | 969.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 09:30:00 | 963.60 | 960.94 | 966.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:00:00 | 957.15 | 960.94 | 966.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 977.00 | 964.15 | 967.80 | SL hit (close>static) qty=1.00 sl=975.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 977.00 | 964.15 | 967.80 | SL hit (close>static) qty=1.00 sl=975.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 977.00 | 964.15 | 967.80 | SL hit (close>static) qty=1.00 sl=975.80 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 15:15:00 | 959.50 | 966.07 | 967.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 15:15:00 | 959.50 | 964.76 | 967.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:15:00 | 968.65 | 964.76 | 967.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 975.25 | 966.86 | 967.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:30:00 | 981.15 | 966.86 | 967.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-05-05 10:15:00 | 983.10 | 970.10 | 969.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 983.10 | 970.10 | 969.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 11:15:00 | 994.05 | 974.89 | 971.46 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-16 10:30:00 | 1137.10 | 2025-05-16 15:15:00 | 1140.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-05-16 15:15:00 | 1140.00 | 2025-05-16 15:15:00 | 1140.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-05-28 13:30:00 | 1180.00 | 2025-06-04 12:15:00 | 1186.10 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2025-05-29 09:15:00 | 1181.00 | 2025-06-04 12:15:00 | 1186.10 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest2 | 2025-05-30 09:45:00 | 1180.80 | 2025-06-04 12:15:00 | 1186.10 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest2 | 2025-05-30 10:15:00 | 1182.10 | 2025-06-04 12:15:00 | 1186.10 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest2 | 2025-05-30 14:15:00 | 1185.60 | 2025-06-04 12:15:00 | 1186.10 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2025-05-30 15:15:00 | 1193.00 | 2025-06-04 12:15:00 | 1186.10 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-06-02 09:30:00 | 1186.60 | 2025-06-04 12:15:00 | 1186.10 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-06-02 11:15:00 | 1187.00 | 2025-06-04 12:15:00 | 1186.10 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-06-02 12:15:00 | 1195.80 | 2025-06-04 12:15:00 | 1186.10 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-06-02 14:15:00 | 1194.50 | 2025-06-04 12:15:00 | 1186.10 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-06-03 10:00:00 | 1195.00 | 2025-06-04 12:15:00 | 1186.10 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-06-03 15:00:00 | 1194.40 | 2025-06-04 12:15:00 | 1186.10 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest1 | 2025-06-10 10:30:00 | 1357.00 | 2025-06-11 09:15:00 | 1291.00 | STOP_HIT | 1.00 | -4.86% |
| BUY | retest2 | 2025-06-20 10:00:00 | 1349.90 | 2025-06-23 10:15:00 | 1312.20 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2025-06-20 12:30:00 | 1365.00 | 2025-06-23 10:15:00 | 1312.20 | STOP_HIT | 1.00 | -3.87% |
| SELL | retest2 | 2025-06-30 10:45:00 | 1251.30 | 2025-07-02 09:15:00 | 1188.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-30 11:15:00 | 1248.80 | 2025-07-02 09:15:00 | 1186.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-01 10:45:00 | 1248.40 | 2025-07-02 09:15:00 | 1185.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-30 10:45:00 | 1251.30 | 2025-07-03 09:15:00 | 1126.17 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-06-30 11:15:00 | 1248.80 | 2025-07-03 12:15:00 | 1161.00 | STOP_HIT | 0.50 | 7.03% |
| SELL | retest2 | 2025-07-01 10:45:00 | 1248.40 | 2025-07-03 12:15:00 | 1161.00 | STOP_HIT | 0.50 | 7.00% |
| SELL | retest2 | 2025-07-23 12:45:00 | 1128.00 | 2025-07-28 10:15:00 | 1140.70 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-07-24 11:45:00 | 1129.20 | 2025-07-28 10:15:00 | 1140.70 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-07-24 15:15:00 | 1127.80 | 2025-07-28 10:15:00 | 1140.70 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-08-12 13:15:00 | 1092.30 | 2025-08-13 09:15:00 | 1118.40 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2025-08-12 14:00:00 | 1093.80 | 2025-08-13 09:15:00 | 1118.40 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-08-25 15:15:00 | 1295.00 | 2025-08-26 10:15:00 | 1271.40 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-09-04 11:15:00 | 1186.20 | 2025-09-08 09:15:00 | 1203.60 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-09-05 11:15:00 | 1187.60 | 2025-09-08 09:15:00 | 1203.60 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-09-10 14:45:00 | 1213.00 | 2025-09-11 11:15:00 | 1197.40 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-09-11 09:15:00 | 1213.80 | 2025-09-11 11:15:00 | 1197.40 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-09-18 12:45:00 | 1207.30 | 2025-09-24 13:15:00 | 1194.60 | STOP_HIT | 1.00 | 1.05% |
| SELL | retest2 | 2025-09-19 09:30:00 | 1214.50 | 2025-09-24 13:15:00 | 1194.60 | STOP_HIT | 1.00 | 1.64% |
| BUY | retest2 | 2025-10-16 12:15:00 | 1086.40 | 2025-10-24 10:15:00 | 1077.00 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-10-17 11:30:00 | 1077.70 | 2025-10-24 12:15:00 | 1080.10 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-10-17 12:00:00 | 1077.90 | 2025-10-24 12:15:00 | 1080.10 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-10-17 14:45:00 | 1080.50 | 2025-10-24 12:15:00 | 1080.10 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-10-23 12:00:00 | 1103.50 | 2025-10-24 12:15:00 | 1080.10 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-11-03 11:00:00 | 1117.00 | 2025-11-06 09:15:00 | 1085.10 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2025-11-12 09:15:00 | 1044.50 | 2025-11-13 11:15:00 | 1072.30 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-11-12 11:45:00 | 1064.60 | 2025-11-13 11:15:00 | 1072.30 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-11-20 09:15:00 | 1055.80 | 2025-11-26 14:15:00 | 1058.40 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-12-04 13:15:00 | 1034.80 | 2025-12-10 09:15:00 | 1045.70 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-12-15 09:15:00 | 1012.80 | 2025-12-16 12:15:00 | 1020.80 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-12-15 10:00:00 | 1012.00 | 2025-12-16 12:15:00 | 1020.80 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-12-16 09:15:00 | 1015.40 | 2025-12-16 12:15:00 | 1020.80 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-12-16 10:30:00 | 1016.00 | 2025-12-16 12:15:00 | 1020.80 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-12-18 09:30:00 | 1000.80 | 2025-12-23 13:15:00 | 1009.90 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-12-18 10:45:00 | 1003.00 | 2025-12-23 13:15:00 | 1009.90 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-12-18 11:30:00 | 1001.50 | 2025-12-23 13:15:00 | 1009.90 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-12-18 12:00:00 | 1002.30 | 2025-12-23 13:15:00 | 1009.90 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-12-19 10:30:00 | 1006.00 | 2025-12-23 13:15:00 | 1009.90 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-12-19 12:15:00 | 996.50 | 2025-12-23 13:15:00 | 1009.90 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-12-19 14:15:00 | 1004.80 | 2025-12-23 13:15:00 | 1009.90 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-12-23 12:30:00 | 1007.20 | 2025-12-23 13:15:00 | 1009.90 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-12-30 09:15:00 | 981.80 | 2025-12-31 12:15:00 | 1008.00 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2026-01-06 15:15:00 | 1022.00 | 2026-01-09 09:15:00 | 1011.10 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2026-01-07 10:15:00 | 1027.20 | 2026-01-09 09:15:00 | 1011.10 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2026-01-13 15:15:00 | 997.00 | 2026-01-16 13:15:00 | 1013.50 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2026-01-16 13:00:00 | 998.00 | 2026-01-16 13:15:00 | 1013.50 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-01-20 11:45:00 | 1001.00 | 2026-01-22 15:15:00 | 1004.80 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2026-01-20 12:30:00 | 1000.00 | 2026-01-22 15:15:00 | 1004.80 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2026-01-20 13:00:00 | 1001.50 | 2026-01-22 15:15:00 | 1004.80 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2026-01-20 14:00:00 | 997.80 | 2026-01-22 15:15:00 | 1004.80 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2026-01-21 10:30:00 | 984.80 | 2026-01-22 15:15:00 | 1004.80 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest1 | 2026-01-28 09:15:00 | 964.90 | 2026-01-28 14:15:00 | 990.50 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest1 | 2026-01-28 10:00:00 | 965.20 | 2026-01-28 14:15:00 | 990.50 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest1 | 2026-01-28 11:00:00 | 965.00 | 2026-01-28 14:15:00 | 990.50 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest1 | 2026-01-28 12:45:00 | 965.70 | 2026-01-28 14:15:00 | 990.50 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2026-01-29 09:15:00 | 981.10 | 2026-01-29 10:15:00 | 977.00 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2026-02-24 15:15:00 | 1053.50 | 2026-02-26 09:15:00 | 1024.70 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest1 | 2026-03-06 10:45:00 | 929.50 | 2026-03-12 09:15:00 | 883.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-03-09 09:15:00 | 929.10 | 2026-03-12 09:15:00 | 882.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-10 10:30:00 | 915.00 | 2026-03-12 09:15:00 | 872.91 | PARTIAL | 0.50 | 4.60% |
| SELL | retest1 | 2026-03-06 10:45:00 | 929.50 | 2026-03-12 11:15:00 | 897.95 | STOP_HIT | 0.50 | 3.39% |
| SELL | retest1 | 2026-03-09 09:15:00 | 929.10 | 2026-03-12 11:15:00 | 897.95 | STOP_HIT | 0.50 | 3.35% |
| SELL | retest2 | 2026-03-10 10:30:00 | 915.00 | 2026-03-12 11:15:00 | 897.95 | STOP_HIT | 0.50 | 1.86% |
| SELL | retest2 | 2026-03-11 09:30:00 | 918.85 | 2026-03-13 10:15:00 | 869.25 | PARTIAL | 0.50 | 5.40% |
| SELL | retest2 | 2026-03-11 09:30:00 | 918.85 | 2026-03-16 09:15:00 | 823.50 | TARGET_HIT | 0.50 | 10.38% |
| BUY | retest2 | 2026-04-10 09:15:00 | 891.20 | 2026-04-23 09:15:00 | 980.32 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-30 15:00:00 | 961.60 | 2026-05-04 10:15:00 | 977.00 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2026-05-04 09:30:00 | 963.60 | 2026-05-04 10:15:00 | 977.00 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-05-04 10:00:00 | 957.15 | 2026-05-04 10:15:00 | 977.00 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2026-05-04 15:15:00 | 959.50 | 2026-05-05 10:15:00 | 983.10 | STOP_HIT | 1.00 | -2.46% |
