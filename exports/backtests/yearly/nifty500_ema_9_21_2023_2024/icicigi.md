# ICICI Lombard General Insurance Company Ltd. (ICICIGI)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1820.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 238 |
| ALERT1 | 155 |
| ALERT2 | 154 |
| ALERT2_SKIP | 81 |
| ALERT3 | 465 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 180 |
| PARTIAL | 14 |
| TARGET_HIT | 2 |
| STOP_HIT | 184 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 200 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 61 / 139
- **Target hits / Stop hits / Partials:** 2 / 184 / 14
- **Avg / median % per leg:** 0.16% / -0.69%
- **Sum % (uncompounded):** 31.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 109 | 20 | 18.3% | 2 | 107 | 0 | -0.37% | -40.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 109 | 20 | 18.3% | 2 | 107 | 0 | -0.37% | -40.1% |
| SELL (all) | 91 | 41 | 45.1% | 0 | 77 | 14 | 0.78% | 71.3% |
| SELL @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 6 | 0 | -0.90% | -5.4% |
| SELL @ 3rd Alert (retest2) | 85 | 40 | 47.1% | 0 | 71 | 14 | 0.90% | 76.8% |
| retest1 (combined) | 6 | 1 | 16.7% | 0 | 6 | 0 | -0.90% | -5.4% |
| retest2 (combined) | 194 | 60 | 30.9% | 2 | 178 | 14 | 0.19% | 36.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 10:15:00 | 1111.15 | 1115.99 | 1116.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-17 11:15:00 | 1106.70 | 1114.13 | 1115.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-18 09:15:00 | 1112.05 | 1108.37 | 1111.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-18 09:15:00 | 1112.05 | 1108.37 | 1111.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 09:15:00 | 1112.05 | 1108.37 | 1111.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-18 09:30:00 | 1112.50 | 1108.37 | 1111.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 10:15:00 | 1118.80 | 1110.46 | 1112.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-18 11:00:00 | 1118.80 | 1110.46 | 1112.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 11:15:00 | 1113.55 | 1111.07 | 1112.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-18 12:15:00 | 1111.80 | 1111.07 | 1112.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-26 10:15:00 | 1089.70 | 1082.19 | 1081.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2023-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 10:15:00 | 1089.70 | 1082.19 | 1081.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 13:15:00 | 1098.05 | 1087.63 | 1084.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-31 10:15:00 | 1178.35 | 1181.16 | 1164.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-31 11:00:00 | 1178.35 | 1181.16 | 1164.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 09:15:00 | 1215.40 | 1187.36 | 1174.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-01 11:00:00 | 1219.75 | 1193.84 | 1178.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-09 11:15:00 | 1229.40 | 1244.58 | 1244.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2023-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 11:15:00 | 1229.40 | 1244.58 | 1244.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 12:15:00 | 1221.00 | 1239.86 | 1242.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-13 09:15:00 | 1221.25 | 1215.27 | 1222.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 09:15:00 | 1221.25 | 1215.27 | 1222.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 09:15:00 | 1221.25 | 1215.27 | 1222.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-14 09:15:00 | 1210.70 | 1215.94 | 1219.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-14 09:45:00 | 1208.55 | 1214.36 | 1218.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-14 10:30:00 | 1210.40 | 1213.35 | 1217.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-15 09:15:00 | 1207.00 | 1212.17 | 1215.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 10:15:00 | 1213.70 | 1212.44 | 1215.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-15 10:45:00 | 1219.05 | 1212.44 | 1215.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 12:15:00 | 1213.75 | 1212.57 | 1214.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-15 12:45:00 | 1214.85 | 1212.57 | 1214.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 13:15:00 | 1215.25 | 1213.10 | 1214.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-15 13:45:00 | 1215.20 | 1213.10 | 1214.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 14:15:00 | 1210.15 | 1212.51 | 1214.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-15 14:30:00 | 1213.50 | 1212.51 | 1214.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-06-16 09:15:00 | 1231.50 | 1216.16 | 1215.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2023-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 09:15:00 | 1231.50 | 1216.16 | 1215.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-16 10:15:00 | 1242.00 | 1221.33 | 1218.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 12:15:00 | 1301.60 | 1303.63 | 1291.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-22 13:00:00 | 1301.60 | 1303.63 | 1291.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 13:15:00 | 1295.10 | 1301.93 | 1291.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 13:30:00 | 1295.75 | 1301.93 | 1291.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 14:15:00 | 1279.10 | 1297.36 | 1290.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 14:30:00 | 1280.00 | 1297.36 | 1290.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 15:15:00 | 1272.05 | 1292.30 | 1288.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 09:15:00 | 1268.90 | 1292.30 | 1288.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2023-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 10:15:00 | 1263.75 | 1283.25 | 1284.99 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 09:15:00 | 1300.10 | 1279.30 | 1278.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 12:15:00 | 1318.10 | 1293.48 | 1285.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 10:15:00 | 1336.20 | 1338.45 | 1328.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-03 11:00:00 | 1336.20 | 1338.45 | 1328.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 1334.45 | 1337.02 | 1332.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 09:45:00 | 1330.90 | 1337.02 | 1332.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 10:15:00 | 1333.35 | 1336.29 | 1332.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 10:45:00 | 1332.55 | 1336.29 | 1332.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 11:15:00 | 1334.20 | 1335.87 | 1332.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 11:45:00 | 1331.85 | 1335.87 | 1332.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 12:15:00 | 1326.10 | 1333.91 | 1331.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 12:45:00 | 1328.45 | 1333.91 | 1331.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2023-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 13:15:00 | 1314.45 | 1330.02 | 1330.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-04 15:15:00 | 1311.05 | 1323.52 | 1327.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-05 10:15:00 | 1323.40 | 1323.21 | 1326.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-05 11:00:00 | 1323.40 | 1323.21 | 1326.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 14:15:00 | 1329.00 | 1323.93 | 1325.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-05 15:00:00 | 1329.00 | 1323.93 | 1325.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 15:15:00 | 1330.00 | 1325.14 | 1326.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-06 09:15:00 | 1330.85 | 1325.14 | 1326.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2023-07-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 09:15:00 | 1333.05 | 1326.72 | 1326.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 10:15:00 | 1340.00 | 1329.38 | 1327.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 12:15:00 | 1339.45 | 1341.93 | 1336.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-07 12:45:00 | 1339.80 | 1341.93 | 1336.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 13:15:00 | 1334.95 | 1340.53 | 1336.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 14:00:00 | 1334.95 | 1340.53 | 1336.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 14:15:00 | 1327.70 | 1337.97 | 1335.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 15:00:00 | 1327.70 | 1337.97 | 1335.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 15:15:00 | 1332.85 | 1336.94 | 1335.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 09:15:00 | 1327.65 | 1336.94 | 1335.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 10:15:00 | 1331.30 | 1335.35 | 1334.97 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 11:15:00 | 1326.90 | 1333.66 | 1334.24 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 12:15:00 | 1343.50 | 1335.63 | 1335.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 09:15:00 | 1354.05 | 1341.65 | 1338.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-11 12:15:00 | 1342.40 | 1343.04 | 1339.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 12:15:00 | 1342.40 | 1343.04 | 1339.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 12:15:00 | 1342.40 | 1343.04 | 1339.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-11 12:30:00 | 1343.90 | 1343.04 | 1339.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 13:15:00 | 1342.55 | 1342.95 | 1340.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-11 13:45:00 | 1339.50 | 1342.95 | 1340.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 09:15:00 | 1342.00 | 1343.19 | 1340.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 10:00:00 | 1342.00 | 1343.19 | 1340.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 10:15:00 | 1344.45 | 1343.44 | 1341.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-12 14:00:00 | 1350.75 | 1344.18 | 1342.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-18 11:15:00 | 1347.50 | 1361.87 | 1362.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2023-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 11:15:00 | 1347.50 | 1361.87 | 1362.03 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 11:15:00 | 1377.50 | 1362.36 | 1361.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-20 13:15:00 | 1386.70 | 1370.84 | 1366.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-24 10:15:00 | 1387.25 | 1398.67 | 1389.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 10:15:00 | 1387.25 | 1398.67 | 1389.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 10:15:00 | 1387.25 | 1398.67 | 1389.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 11:00:00 | 1387.25 | 1398.67 | 1389.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 11:15:00 | 1388.60 | 1396.66 | 1389.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 11:45:00 | 1385.00 | 1396.66 | 1389.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 12:15:00 | 1387.45 | 1394.82 | 1389.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 12:30:00 | 1386.55 | 1394.82 | 1389.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 13:15:00 | 1383.25 | 1392.50 | 1388.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 13:30:00 | 1381.10 | 1392.50 | 1388.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 14:15:00 | 1387.45 | 1391.49 | 1388.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 15:15:00 | 1387.00 | 1391.49 | 1388.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 15:15:00 | 1387.00 | 1390.59 | 1388.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-25 09:15:00 | 1396.50 | 1390.59 | 1388.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-25 12:15:00 | 1373.05 | 1386.39 | 1387.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2023-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 12:15:00 | 1373.05 | 1386.39 | 1387.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-26 12:15:00 | 1369.10 | 1377.67 | 1381.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-27 15:15:00 | 1363.60 | 1363.59 | 1370.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-28 09:15:00 | 1362.10 | 1363.59 | 1370.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 11:15:00 | 1362.90 | 1362.76 | 1368.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-28 11:30:00 | 1362.30 | 1362.76 | 1368.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 14:15:00 | 1373.85 | 1365.32 | 1367.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-28 14:45:00 | 1373.65 | 1365.32 | 1367.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 15:15:00 | 1370.00 | 1366.26 | 1368.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-31 09:15:00 | 1355.00 | 1366.26 | 1368.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 1361.25 | 1365.26 | 1367.48 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 11:15:00 | 1378.65 | 1369.57 | 1369.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 14:15:00 | 1387.25 | 1375.40 | 1372.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 09:15:00 | 1376.30 | 1377.00 | 1373.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-01 09:15:00 | 1376.30 | 1377.00 | 1373.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 09:15:00 | 1376.30 | 1377.00 | 1373.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 10:00:00 | 1376.30 | 1377.00 | 1373.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 10:15:00 | 1371.00 | 1375.80 | 1373.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 10:30:00 | 1365.95 | 1375.80 | 1373.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 11:15:00 | 1362.40 | 1373.12 | 1372.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 12:00:00 | 1362.40 | 1373.12 | 1372.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2023-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-01 12:15:00 | 1361.00 | 1370.70 | 1371.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 12:15:00 | 1340.40 | 1356.57 | 1361.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 15:15:00 | 1353.00 | 1351.56 | 1357.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-04 09:15:00 | 1357.90 | 1351.56 | 1357.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 1350.85 | 1351.41 | 1356.96 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 09:15:00 | 1392.50 | 1363.57 | 1360.38 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-08-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 10:15:00 | 1360.00 | 1384.61 | 1387.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-10 11:15:00 | 1357.00 | 1379.09 | 1384.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-11 10:15:00 | 1382.40 | 1369.34 | 1375.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 10:15:00 | 1382.40 | 1369.34 | 1375.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 10:15:00 | 1382.40 | 1369.34 | 1375.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-11 11:00:00 | 1382.40 | 1369.34 | 1375.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 11:15:00 | 1382.00 | 1371.87 | 1376.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-11 14:30:00 | 1376.85 | 1374.09 | 1376.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-24 12:15:00 | 1338.05 | 1331.19 | 1330.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2023-08-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-24 12:15:00 | 1338.05 | 1331.19 | 1330.54 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 13:15:00 | 1327.75 | 1331.12 | 1331.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-28 11:15:00 | 1318.05 | 1327.08 | 1329.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 13:15:00 | 1329.05 | 1326.76 | 1328.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 13:15:00 | 1329.05 | 1326.76 | 1328.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 13:15:00 | 1329.05 | 1326.76 | 1328.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 13:30:00 | 1329.20 | 1326.76 | 1328.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 14:15:00 | 1328.20 | 1327.05 | 1328.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 14:45:00 | 1327.45 | 1327.05 | 1328.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 15:15:00 | 1329.75 | 1327.59 | 1328.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 09:15:00 | 1325.30 | 1327.59 | 1328.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 09:15:00 | 1330.20 | 1328.11 | 1328.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-29 12:15:00 | 1319.50 | 1326.96 | 1328.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-29 12:45:00 | 1315.00 | 1324.46 | 1326.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-29 14:45:00 | 1319.25 | 1322.15 | 1325.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-30 10:00:00 | 1318.85 | 1321.28 | 1324.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 11:15:00 | 1324.90 | 1322.28 | 1324.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 12:00:00 | 1324.90 | 1322.28 | 1324.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 12:15:00 | 1328.50 | 1323.52 | 1324.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 13:00:00 | 1328.50 | 1323.52 | 1324.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 13:15:00 | 1323.55 | 1323.53 | 1324.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-30 14:15:00 | 1321.95 | 1323.53 | 1324.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-31 09:15:00 | 1316.95 | 1323.37 | 1324.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-31 13:00:00 | 1320.35 | 1319.70 | 1322.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-01 10:15:00 | 1339.35 | 1323.50 | 1322.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2023-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 10:15:00 | 1339.35 | 1323.50 | 1322.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 11:15:00 | 1343.55 | 1327.51 | 1324.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 09:15:00 | 1352.45 | 1353.64 | 1345.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-05 09:45:00 | 1352.70 | 1353.64 | 1345.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 10:15:00 | 1341.80 | 1351.27 | 1344.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-05 11:00:00 | 1341.80 | 1351.27 | 1344.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 11:15:00 | 1339.50 | 1348.92 | 1344.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-05 12:00:00 | 1339.50 | 1348.92 | 1344.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 12:15:00 | 1347.00 | 1348.53 | 1344.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-05 13:15:00 | 1352.65 | 1348.53 | 1344.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-06 09:30:00 | 1350.10 | 1349.40 | 1346.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-06 14:45:00 | 1347.50 | 1347.46 | 1346.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-07 09:15:00 | 1351.90 | 1347.17 | 1346.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 10:15:00 | 1347.00 | 1346.64 | 1346.40 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-09-07 11:15:00 | 1343.00 | 1345.92 | 1346.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2023-09-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 11:15:00 | 1343.00 | 1345.92 | 1346.09 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-09-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 14:15:00 | 1349.45 | 1346.73 | 1346.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-08 09:15:00 | 1355.85 | 1349.05 | 1347.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-08 12:15:00 | 1346.50 | 1349.88 | 1348.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 12:15:00 | 1346.50 | 1349.88 | 1348.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 12:15:00 | 1346.50 | 1349.88 | 1348.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-08 13:00:00 | 1346.50 | 1349.88 | 1348.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 13:15:00 | 1347.35 | 1349.38 | 1348.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-08 13:30:00 | 1347.35 | 1349.38 | 1348.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 14:15:00 | 1349.00 | 1349.30 | 1348.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-08 14:45:00 | 1347.00 | 1349.30 | 1348.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 15:15:00 | 1349.20 | 1349.28 | 1348.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-11 09:15:00 | 1352.00 | 1349.28 | 1348.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 09:15:00 | 1356.35 | 1350.70 | 1349.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-11 10:30:00 | 1360.00 | 1352.30 | 1350.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-11 12:30:00 | 1361.00 | 1355.90 | 1352.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-11 14:00:00 | 1361.45 | 1357.01 | 1352.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-12 11:15:00 | 1363.35 | 1361.76 | 1356.96 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 11:15:00 | 1350.50 | 1359.50 | 1356.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 12:00:00 | 1350.50 | 1359.50 | 1356.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 12:15:00 | 1356.35 | 1358.87 | 1356.37 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-09-12 15:15:00 | 1352.00 | 1354.92 | 1354.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2023-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 15:15:00 | 1352.00 | 1354.92 | 1354.95 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-09-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 12:15:00 | 1356.45 | 1354.58 | 1354.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 09:15:00 | 1361.15 | 1356.76 | 1355.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-14 12:15:00 | 1355.35 | 1357.45 | 1356.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 12:15:00 | 1355.35 | 1357.45 | 1356.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 12:15:00 | 1355.35 | 1357.45 | 1356.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-14 13:00:00 | 1355.35 | 1357.45 | 1356.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 13:15:00 | 1357.05 | 1357.37 | 1356.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-14 13:30:00 | 1356.00 | 1357.37 | 1356.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 14:15:00 | 1369.50 | 1359.79 | 1357.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-14 15:15:00 | 1374.45 | 1359.79 | 1357.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-15 13:15:00 | 1370.00 | 1369.01 | 1363.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-15 14:00:00 | 1369.75 | 1369.15 | 1364.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-15 14:30:00 | 1375.00 | 1370.59 | 1365.38 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 12:15:00 | 1369.35 | 1373.87 | 1369.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 12:45:00 | 1368.90 | 1373.87 | 1369.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 13:15:00 | 1374.70 | 1374.03 | 1369.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-18 13:30:00 | 1365.60 | 1374.03 | 1369.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 1372.55 | 1374.41 | 1371.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-20 12:00:00 | 1385.95 | 1376.55 | 1372.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-21 12:15:00 | 1351.00 | 1369.54 | 1371.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2023-09-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 12:15:00 | 1351.00 | 1369.54 | 1371.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-22 09:15:00 | 1336.00 | 1357.54 | 1364.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 12:15:00 | 1353.30 | 1353.28 | 1360.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-22 12:45:00 | 1350.70 | 1353.28 | 1360.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 12:15:00 | 1295.50 | 1285.17 | 1292.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 13:00:00 | 1295.50 | 1285.17 | 1292.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 13:15:00 | 1310.85 | 1290.31 | 1294.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 14:00:00 | 1310.85 | 1290.31 | 1294.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 14:15:00 | 1308.10 | 1293.87 | 1295.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 14:45:00 | 1310.00 | 1293.87 | 1295.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2023-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 10:15:00 | 1306.05 | 1297.47 | 1297.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 11:15:00 | 1311.35 | 1300.25 | 1298.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 15:15:00 | 1302.65 | 1302.99 | 1300.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-04 09:15:00 | 1293.55 | 1302.99 | 1300.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 1286.65 | 1299.72 | 1299.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 10:00:00 | 1286.65 | 1299.72 | 1299.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2023-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 10:15:00 | 1289.00 | 1297.58 | 1298.36 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-10-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-04 14:15:00 | 1301.65 | 1298.66 | 1298.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-05 09:15:00 | 1309.25 | 1301.20 | 1299.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-05 13:15:00 | 1297.75 | 1302.96 | 1301.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 13:15:00 | 1297.75 | 1302.96 | 1301.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 13:15:00 | 1297.75 | 1302.96 | 1301.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-05 14:00:00 | 1297.75 | 1302.96 | 1301.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 14:15:00 | 1295.05 | 1301.38 | 1300.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-05 15:00:00 | 1295.05 | 1301.38 | 1300.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 1305.25 | 1302.08 | 1301.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-06 10:15:00 | 1316.55 | 1302.08 | 1301.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-09 13:15:00 | 1297.35 | 1307.69 | 1307.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2023-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 13:15:00 | 1297.35 | 1307.69 | 1307.90 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 12:15:00 | 1315.55 | 1308.35 | 1307.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 09:15:00 | 1320.80 | 1313.66 | 1311.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-12 13:15:00 | 1318.25 | 1319.15 | 1315.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-12 14:00:00 | 1318.25 | 1319.15 | 1315.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 12:15:00 | 1319.50 | 1323.52 | 1319.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 12:45:00 | 1321.00 | 1323.52 | 1319.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 13:15:00 | 1316.95 | 1322.20 | 1319.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 14:00:00 | 1316.95 | 1322.20 | 1319.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 14:15:00 | 1317.10 | 1321.18 | 1319.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-13 14:45:00 | 1317.10 | 1321.18 | 1319.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 15:15:00 | 1312.90 | 1319.53 | 1318.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 09:15:00 | 1323.65 | 1319.53 | 1318.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-26 09:15:00 | 1359.25 | 1381.80 | 1384.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2023-10-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 09:15:00 | 1359.25 | 1381.80 | 1384.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-26 13:15:00 | 1351.05 | 1368.76 | 1376.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 1365.90 | 1364.65 | 1372.43 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-30 09:15:00 | 1342.75 | 1360.92 | 1367.16 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 09:15:00 | 1363.90 | 1354.53 | 1358.80 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-10-31 09:15:00 | 1363.90 | 1354.53 | 1358.80 | SL hit (close>ema400) qty=1.00 sl=1358.80 alert=retest1 |

### Cycle 32 — BUY (started 2023-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 13:15:00 | 1369.00 | 1361.29 | 1360.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 14:15:00 | 1375.05 | 1364.04 | 1362.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 11:15:00 | 1367.80 | 1367.83 | 1365.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-01 12:15:00 | 1367.20 | 1367.83 | 1365.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 12:15:00 | 1361.70 | 1366.61 | 1364.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 13:00:00 | 1361.70 | 1366.61 | 1364.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 13:15:00 | 1365.60 | 1366.41 | 1364.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 09:15:00 | 1371.20 | 1363.96 | 1363.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 11:15:00 | 1369.95 | 1364.74 | 1364.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 13:45:00 | 1372.95 | 1366.55 | 1365.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-03 12:15:00 | 1362.00 | 1364.43 | 1364.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 12:15:00 | 1362.00 | 1364.43 | 1364.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-06 09:15:00 | 1359.40 | 1362.79 | 1363.80 | Break + close below crossover candle low |

### Cycle 34 — BUY (started 2023-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-06 11:15:00 | 1373.80 | 1364.37 | 1364.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-07 13:15:00 | 1379.40 | 1369.74 | 1367.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-08 09:15:00 | 1368.05 | 1372.42 | 1369.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-08 09:15:00 | 1368.05 | 1372.42 | 1369.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 09:15:00 | 1368.05 | 1372.42 | 1369.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 09:30:00 | 1368.00 | 1372.42 | 1369.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 10:15:00 | 1367.00 | 1371.34 | 1369.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 11:00:00 | 1367.00 | 1371.34 | 1369.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 11:15:00 | 1366.85 | 1370.44 | 1369.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 11:45:00 | 1364.75 | 1370.44 | 1369.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 12:15:00 | 1367.55 | 1369.86 | 1369.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 12:30:00 | 1364.50 | 1369.86 | 1369.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 13:15:00 | 1364.30 | 1368.75 | 1368.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 14:00:00 | 1364.30 | 1368.75 | 1368.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2023-11-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-08 14:15:00 | 1367.75 | 1368.55 | 1368.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 11:15:00 | 1361.65 | 1365.76 | 1367.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 15:15:00 | 1355.35 | 1353.71 | 1357.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-12 18:15:00 | 1360.75 | 1353.71 | 1357.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 1360.00 | 1354.97 | 1358.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-12 18:30:00 | 1368.55 | 1354.97 | 1358.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 09:15:00 | 1347.85 | 1353.55 | 1357.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 10:15:00 | 1341.70 | 1353.55 | 1357.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-15 09:15:00 | 1368.20 | 1357.65 | 1357.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2023-11-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 09:15:00 | 1368.20 | 1357.65 | 1357.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 10:15:00 | 1388.75 | 1363.87 | 1360.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-20 09:15:00 | 1434.55 | 1439.23 | 1420.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-20 09:15:00 | 1434.55 | 1439.23 | 1420.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 09:15:00 | 1434.55 | 1439.23 | 1420.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 09:15:00 | 1456.55 | 1443.87 | 1431.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-23 09:15:00 | 1471.85 | 1454.36 | 1451.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-23 11:15:00 | 1442.95 | 1450.24 | 1450.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2023-11-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-23 11:15:00 | 1442.95 | 1450.24 | 1450.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-23 13:15:00 | 1439.95 | 1446.81 | 1448.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 14:15:00 | 1438.75 | 1437.85 | 1442.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-24 14:45:00 | 1438.80 | 1437.85 | 1442.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 09:15:00 | 1442.45 | 1438.15 | 1441.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-28 11:00:00 | 1435.20 | 1437.56 | 1440.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-28 15:15:00 | 1447.50 | 1442.46 | 1442.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2023-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 15:15:00 | 1447.50 | 1442.46 | 1442.10 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-29 11:15:00 | 1438.90 | 1441.52 | 1441.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-29 12:15:00 | 1434.30 | 1440.07 | 1441.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-29 14:15:00 | 1442.10 | 1440.47 | 1441.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 14:15:00 | 1442.10 | 1440.47 | 1441.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 14:15:00 | 1442.10 | 1440.47 | 1441.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 15:00:00 | 1442.10 | 1440.47 | 1441.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 15:15:00 | 1441.00 | 1440.57 | 1441.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-30 09:15:00 | 1452.70 | 1440.57 | 1441.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2023-11-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 09:15:00 | 1460.95 | 1444.65 | 1442.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 11:15:00 | 1470.65 | 1452.95 | 1447.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 10:15:00 | 1463.30 | 1468.22 | 1459.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-01 11:00:00 | 1463.30 | 1468.22 | 1459.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 11:15:00 | 1459.55 | 1466.48 | 1459.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-01 12:00:00 | 1459.55 | 1466.48 | 1459.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 12:15:00 | 1455.30 | 1464.25 | 1458.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-01 13:00:00 | 1455.30 | 1464.25 | 1458.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 13:15:00 | 1452.85 | 1461.97 | 1458.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-01 14:00:00 | 1452.85 | 1461.97 | 1458.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 15:15:00 | 1456.00 | 1460.21 | 1458.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 09:15:00 | 1459.70 | 1460.21 | 1458.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-04 09:15:00 | 1451.60 | 1458.49 | 1457.45 | SL hit (close<static) qty=1.00 sl=1454.00 alert=retest2 |

### Cycle 41 — SELL (started 2023-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-04 12:15:00 | 1452.05 | 1456.57 | 1456.77 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 14:15:00 | 1470.05 | 1458.98 | 1457.81 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2023-12-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 10:15:00 | 1446.10 | 1460.20 | 1460.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 09:15:00 | 1442.35 | 1446.85 | 1450.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-08 10:15:00 | 1447.65 | 1447.01 | 1450.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-08 11:00:00 | 1447.65 | 1447.01 | 1450.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 11:15:00 | 1449.95 | 1447.60 | 1450.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-08 11:30:00 | 1447.25 | 1447.60 | 1450.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 12:15:00 | 1442.60 | 1446.60 | 1449.69 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2023-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 09:15:00 | 1459.65 | 1449.06 | 1448.99 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2023-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 09:15:00 | 1434.35 | 1450.60 | 1450.83 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-12-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 09:15:00 | 1458.75 | 1451.77 | 1450.90 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2023-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 15:15:00 | 1450.00 | 1453.54 | 1453.89 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2023-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-18 10:15:00 | 1461.45 | 1454.94 | 1454.45 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2023-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-18 14:15:00 | 1448.15 | 1454.27 | 1454.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-19 09:15:00 | 1439.10 | 1450.23 | 1452.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-20 10:15:00 | 1448.40 | 1444.57 | 1447.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 10:15:00 | 1448.40 | 1444.57 | 1447.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 10:15:00 | 1448.40 | 1444.57 | 1447.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-20 11:00:00 | 1448.40 | 1444.57 | 1447.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 11:15:00 | 1447.00 | 1445.06 | 1447.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-20 11:30:00 | 1447.05 | 1445.06 | 1447.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 12:15:00 | 1441.00 | 1444.24 | 1446.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-20 13:15:00 | 1440.00 | 1444.24 | 1446.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-27 10:15:00 | 1425.15 | 1418.70 | 1418.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2023-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 10:15:00 | 1425.15 | 1418.70 | 1418.62 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2023-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-27 13:15:00 | 1414.00 | 1417.82 | 1418.29 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2023-12-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-28 11:15:00 | 1419.60 | 1418.47 | 1418.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-28 12:15:00 | 1428.30 | 1420.43 | 1419.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 09:15:00 | 1421.80 | 1426.10 | 1422.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 09:15:00 | 1421.80 | 1426.10 | 1422.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 1421.80 | 1426.10 | 1422.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 10:15:00 | 1429.60 | 1423.04 | 1422.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 13:15:00 | 1432.00 | 1424.43 | 1423.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 10:00:00 | 1430.05 | 1430.09 | 1426.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 10:30:00 | 1430.45 | 1429.27 | 1426.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 11:15:00 | 1424.05 | 1428.22 | 1426.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 11:30:00 | 1424.55 | 1428.22 | 1426.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 12:15:00 | 1425.35 | 1427.65 | 1426.27 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-01-02 14:15:00 | 1421.00 | 1425.19 | 1425.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2024-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 14:15:00 | 1421.00 | 1425.19 | 1425.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-03 09:15:00 | 1399.90 | 1419.01 | 1422.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-04 10:15:00 | 1390.70 | 1390.59 | 1402.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-04 11:00:00 | 1390.70 | 1390.59 | 1402.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 09:15:00 | 1390.00 | 1391.20 | 1397.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-08 09:30:00 | 1384.00 | 1392.41 | 1395.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-08 11:15:00 | 1400.15 | 1393.97 | 1395.38 | SL hit (close>static) qty=1.00 sl=1398.55 alert=retest2 |

### Cycle 54 — BUY (started 2024-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-08 13:15:00 | 1398.85 | 1396.45 | 1396.36 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 14:15:00 | 1395.15 | 1396.19 | 1396.25 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 09:15:00 | 1406.50 | 1398.05 | 1397.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-09 10:15:00 | 1415.45 | 1401.53 | 1398.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-09 12:15:00 | 1395.50 | 1400.51 | 1398.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 12:15:00 | 1395.50 | 1400.51 | 1398.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 12:15:00 | 1395.50 | 1400.51 | 1398.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-09 13:00:00 | 1395.50 | 1400.51 | 1398.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 13:15:00 | 1400.65 | 1400.54 | 1398.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-09 14:15:00 | 1389.70 | 1400.54 | 1398.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 14:15:00 | 1392.40 | 1398.91 | 1398.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-09 14:30:00 | 1389.90 | 1398.91 | 1398.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 15:15:00 | 1400.00 | 1399.13 | 1398.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 09:15:00 | 1390.60 | 1399.13 | 1398.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2024-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 09:15:00 | 1387.40 | 1396.78 | 1397.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-10 12:15:00 | 1384.60 | 1392.38 | 1395.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-10 15:15:00 | 1390.55 | 1389.88 | 1393.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-11 09:15:00 | 1395.80 | 1389.88 | 1393.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 1390.25 | 1389.96 | 1392.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-12 09:30:00 | 1383.10 | 1389.17 | 1391.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-12 15:15:00 | 1395.95 | 1392.47 | 1392.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2024-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 15:15:00 | 1395.95 | 1392.47 | 1392.13 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 11:15:00 | 1387.20 | 1391.74 | 1391.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-15 13:15:00 | 1377.15 | 1388.77 | 1390.53 | Break + close below crossover candle low |

### Cycle 60 — BUY (started 2024-01-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-17 09:15:00 | 1448.55 | 1390.11 | 1386.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 13:15:00 | 1481.00 | 1457.95 | 1443.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 13:15:00 | 1470.00 | 1470.03 | 1458.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-20 14:00:00 | 1470.00 | 1470.03 | 1458.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 12:15:00 | 1457.35 | 1472.60 | 1465.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 13:00:00 | 1457.35 | 1472.60 | 1465.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 13:15:00 | 1473.90 | 1472.86 | 1466.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-23 15:00:00 | 1480.05 | 1474.30 | 1467.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-24 09:15:00 | 1487.90 | 1474.30 | 1468.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-02 15:15:00 | 1496.00 | 1501.85 | 1501.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2024-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 15:15:00 | 1496.00 | 1501.85 | 1501.85 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2024-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-05 09:15:00 | 1517.95 | 1505.07 | 1503.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-05 12:15:00 | 1520.00 | 1510.43 | 1506.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-08 14:15:00 | 1622.35 | 1623.00 | 1600.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-08 14:45:00 | 1620.75 | 1623.00 | 1600.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 12:15:00 | 1622.00 | 1629.92 | 1620.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 13:00:00 | 1622.00 | 1629.92 | 1620.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 13:15:00 | 1620.00 | 1627.94 | 1620.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 13:30:00 | 1622.00 | 1627.94 | 1620.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 14:15:00 | 1615.95 | 1625.54 | 1620.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 14:30:00 | 1620.15 | 1625.54 | 1620.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 15:15:00 | 1615.05 | 1623.44 | 1619.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-13 09:15:00 | 1618.45 | 1623.44 | 1619.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 13:15:00 | 1632.95 | 1630.89 | 1625.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-13 13:45:00 | 1628.30 | 1630.89 | 1625.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 14:15:00 | 1629.05 | 1630.52 | 1625.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-13 15:00:00 | 1629.05 | 1630.52 | 1625.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 15:15:00 | 1619.95 | 1628.41 | 1625.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-14 09:15:00 | 1616.50 | 1628.41 | 1625.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 09:15:00 | 1620.70 | 1626.87 | 1624.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-14 11:45:00 | 1627.15 | 1625.56 | 1624.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-15 12:15:00 | 1621.90 | 1626.33 | 1626.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2024-02-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-15 12:15:00 | 1621.90 | 1626.33 | 1626.35 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-02-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 13:15:00 | 1642.40 | 1629.55 | 1627.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 14:15:00 | 1647.65 | 1633.17 | 1629.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 12:15:00 | 1641.25 | 1643.30 | 1636.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-16 13:00:00 | 1641.25 | 1643.30 | 1636.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 09:15:00 | 1639.00 | 1642.79 | 1638.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 09:45:00 | 1636.75 | 1642.79 | 1638.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 10:15:00 | 1646.75 | 1643.58 | 1639.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 10:30:00 | 1649.90 | 1644.69 | 1641.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-20 11:15:00 | 1632.55 | 1642.26 | 1641.11 | SL hit (close<static) qty=1.00 sl=1636.25 alert=retest2 |

### Cycle 65 — SELL (started 2024-02-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 09:15:00 | 1631.55 | 1643.47 | 1643.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 10:15:00 | 1624.55 | 1639.69 | 1642.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 13:15:00 | 1639.35 | 1636.36 | 1639.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 13:15:00 | 1639.35 | 1636.36 | 1639.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 13:15:00 | 1639.35 | 1636.36 | 1639.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 14:00:00 | 1639.35 | 1636.36 | 1639.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 1645.00 | 1638.09 | 1640.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 15:00:00 | 1645.00 | 1638.09 | 1640.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 15:15:00 | 1652.60 | 1640.99 | 1641.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 09:15:00 | 1639.10 | 1640.99 | 1641.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 11:15:00 | 1646.00 | 1638.88 | 1639.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 12:00:00 | 1646.00 | 1638.88 | 1639.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 12:15:00 | 1645.80 | 1640.26 | 1640.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 12:45:00 | 1646.25 | 1640.26 | 1640.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 14:15:00 | 1640.40 | 1639.91 | 1640.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 15:00:00 | 1640.40 | 1639.91 | 1640.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 15:15:00 | 1640.00 | 1639.93 | 1640.26 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2024-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-26 09:15:00 | 1642.85 | 1640.51 | 1640.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-26 10:15:00 | 1651.70 | 1642.75 | 1641.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-28 09:15:00 | 1710.35 | 1710.40 | 1687.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-28 09:30:00 | 1689.05 | 1710.40 | 1687.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 09:15:00 | 1708.70 | 1718.34 | 1703.97 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2024-03-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-01 11:15:00 | 1669.05 | 1697.46 | 1700.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-01 12:15:00 | 1653.05 | 1688.58 | 1696.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-04 13:15:00 | 1649.00 | 1648.91 | 1663.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-04 13:45:00 | 1645.60 | 1648.91 | 1663.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 09:15:00 | 1651.00 | 1649.66 | 1659.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-05 09:45:00 | 1654.50 | 1649.66 | 1659.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 12:15:00 | 1650.35 | 1650.06 | 1657.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-05 12:30:00 | 1653.95 | 1650.06 | 1657.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 13:15:00 | 1655.50 | 1651.15 | 1657.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-05 13:30:00 | 1655.75 | 1651.15 | 1657.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 15:15:00 | 1659.80 | 1653.31 | 1657.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 09:15:00 | 1664.10 | 1653.31 | 1657.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 1649.75 | 1652.60 | 1656.62 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-03-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 13:15:00 | 1665.50 | 1656.45 | 1655.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-11 09:15:00 | 1694.80 | 1666.44 | 1660.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 13:15:00 | 1668.35 | 1670.82 | 1665.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-11 14:00:00 | 1668.35 | 1670.82 | 1665.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 14:15:00 | 1654.85 | 1667.63 | 1664.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 15:00:00 | 1654.85 | 1667.63 | 1664.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 15:15:00 | 1649.75 | 1664.05 | 1662.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-12 09:15:00 | 1662.85 | 1664.05 | 1662.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-12 10:15:00 | 1650.05 | 1661.59 | 1662.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2024-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 10:15:00 | 1650.05 | 1661.59 | 1662.02 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-03-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 14:15:00 | 1660.00 | 1654.34 | 1654.06 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-03-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-15 11:15:00 | 1651.55 | 1653.62 | 1653.84 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 13:15:00 | 1655.00 | 1653.60 | 1653.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-18 15:15:00 | 1658.95 | 1655.09 | 1654.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 10:15:00 | 1649.60 | 1654.89 | 1654.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-19 10:15:00 | 1649.60 | 1654.89 | 1654.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 10:15:00 | 1649.60 | 1654.89 | 1654.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 11:00:00 | 1649.60 | 1654.89 | 1654.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 11:15:00 | 1656.70 | 1655.25 | 1654.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 11:30:00 | 1648.00 | 1655.25 | 1654.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 12:15:00 | 1651.85 | 1654.57 | 1654.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 12:45:00 | 1650.35 | 1654.57 | 1654.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 13:15:00 | 1659.45 | 1655.55 | 1654.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-19 14:45:00 | 1663.05 | 1655.54 | 1654.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-20 09:15:00 | 1634.45 | 1651.23 | 1652.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2024-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 09:15:00 | 1634.45 | 1651.23 | 1652.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 10:15:00 | 1629.65 | 1646.92 | 1650.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 09:15:00 | 1661.55 | 1646.43 | 1648.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 09:15:00 | 1661.55 | 1646.43 | 1648.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 1661.55 | 1646.43 | 1648.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 10:00:00 | 1661.55 | 1646.43 | 1648.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 10:15:00 | 1659.85 | 1649.11 | 1649.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 10:45:00 | 1657.80 | 1649.11 | 1649.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2024-03-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 11:15:00 | 1653.55 | 1650.00 | 1649.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 14:15:00 | 1675.20 | 1656.73 | 1652.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 13:15:00 | 1660.05 | 1665.88 | 1660.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-22 13:15:00 | 1660.05 | 1665.88 | 1660.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 13:15:00 | 1660.05 | 1665.88 | 1660.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-22 13:30:00 | 1663.55 | 1665.88 | 1660.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 14:15:00 | 1669.90 | 1666.68 | 1661.18 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2024-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 10:15:00 | 1634.15 | 1654.54 | 1656.67 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 12:15:00 | 1659.15 | 1654.38 | 1654.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-27 13:15:00 | 1667.50 | 1657.01 | 1655.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 15:15:00 | 1657.75 | 1657.95 | 1656.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-28 09:15:00 | 1667.95 | 1657.95 | 1656.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 1672.90 | 1660.94 | 1657.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 12:00:00 | 1688.70 | 1669.15 | 1662.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-04 10:15:00 | 1675.05 | 1688.26 | 1689.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2024-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 10:15:00 | 1675.05 | 1688.26 | 1689.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-04 11:15:00 | 1654.90 | 1681.59 | 1686.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-05 09:15:00 | 1666.05 | 1659.89 | 1671.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 09:15:00 | 1666.05 | 1659.89 | 1671.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 1666.05 | 1659.89 | 1671.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-05 09:30:00 | 1679.30 | 1659.89 | 1671.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 10:15:00 | 1672.35 | 1662.39 | 1671.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-05 10:45:00 | 1672.15 | 1662.39 | 1671.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 11:15:00 | 1679.80 | 1665.87 | 1672.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-05 12:00:00 | 1679.80 | 1665.87 | 1672.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 12:15:00 | 1680.00 | 1668.69 | 1673.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-05 12:45:00 | 1680.00 | 1668.69 | 1673.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2024-04-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 14:15:00 | 1715.00 | 1681.77 | 1678.57 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-04-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 13:15:00 | 1690.10 | 1695.03 | 1695.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-10 14:15:00 | 1688.40 | 1693.71 | 1694.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 1656.10 | 1643.25 | 1657.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 10:00:00 | 1656.10 | 1643.25 | 1657.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 1667.30 | 1648.06 | 1658.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:00:00 | 1667.30 | 1648.06 | 1658.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 11:15:00 | 1672.40 | 1652.93 | 1660.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:45:00 | 1677.30 | 1652.93 | 1660.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2024-04-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 09:15:00 | 1713.00 | 1667.13 | 1664.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 11:15:00 | 1736.60 | 1687.09 | 1674.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 09:15:00 | 1696.45 | 1702.36 | 1688.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-19 09:15:00 | 1696.45 | 1702.36 | 1688.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 1696.45 | 1702.36 | 1688.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 11:30:00 | 1734.95 | 1709.47 | 1694.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 14:15:00 | 1685.95 | 1692.27 | 1692.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-22 14:15:00 | 1685.95 | 1692.27 | 1692.34 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 10:15:00 | 1695.55 | 1692.61 | 1692.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 12:15:00 | 1700.65 | 1695.01 | 1693.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 14:15:00 | 1695.75 | 1696.21 | 1694.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-23 14:15:00 | 1695.75 | 1696.21 | 1694.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 14:15:00 | 1695.75 | 1696.21 | 1694.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 15:00:00 | 1695.75 | 1696.21 | 1694.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 15:15:00 | 1698.20 | 1696.61 | 1694.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-24 09:15:00 | 1713.60 | 1696.61 | 1694.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 09:15:00 | 1716.85 | 1700.66 | 1696.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-24 11:15:00 | 1722.00 | 1704.70 | 1698.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-24 12:15:00 | 1722.60 | 1707.77 | 1700.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-26 13:15:00 | 1692.30 | 1703.15 | 1703.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2024-04-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 13:15:00 | 1692.30 | 1703.15 | 1703.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-26 15:15:00 | 1688.20 | 1698.22 | 1701.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-29 11:15:00 | 1694.30 | 1692.13 | 1697.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-29 12:00:00 | 1694.30 | 1692.13 | 1697.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 13:15:00 | 1692.95 | 1692.83 | 1696.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-29 13:30:00 | 1698.15 | 1692.83 | 1696.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 14:15:00 | 1708.65 | 1696.00 | 1697.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-29 15:00:00 | 1708.65 | 1696.00 | 1697.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 15:15:00 | 1703.00 | 1697.40 | 1698.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 09:15:00 | 1706.50 | 1697.40 | 1698.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2024-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 09:15:00 | 1718.25 | 1701.57 | 1700.05 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2024-05-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 15:15:00 | 1697.00 | 1706.17 | 1706.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 09:15:00 | 1687.80 | 1702.50 | 1704.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-03 14:15:00 | 1673.85 | 1672.89 | 1686.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-03 15:00:00 | 1673.85 | 1672.89 | 1686.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 13:15:00 | 1681.50 | 1673.75 | 1680.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 13:45:00 | 1684.00 | 1673.75 | 1680.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 14:15:00 | 1682.30 | 1675.46 | 1680.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 15:15:00 | 1682.00 | 1675.46 | 1680.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 15:15:00 | 1682.00 | 1676.77 | 1680.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 09:15:00 | 1703.05 | 1676.77 | 1680.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 11:15:00 | 1685.25 | 1682.99 | 1683.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 11:30:00 | 1689.45 | 1682.99 | 1683.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 12:15:00 | 1669.15 | 1680.22 | 1681.83 | EMA400 retest candle locked (from downside) |

### Cycle 86 — BUY (started 2024-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-08 09:15:00 | 1689.60 | 1683.57 | 1682.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 11:15:00 | 1694.45 | 1687.13 | 1684.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 09:15:00 | 1700.45 | 1703.34 | 1694.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-09 10:00:00 | 1700.45 | 1703.34 | 1694.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 10:15:00 | 1689.40 | 1700.55 | 1694.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 11:00:00 | 1689.40 | 1700.55 | 1694.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 11:15:00 | 1692.00 | 1698.84 | 1694.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 12:00:00 | 1692.00 | 1698.84 | 1694.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 12:15:00 | 1675.20 | 1694.11 | 1692.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 13:00:00 | 1675.20 | 1694.11 | 1692.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2024-05-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 13:15:00 | 1662.90 | 1687.87 | 1689.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 14:15:00 | 1651.95 | 1680.69 | 1686.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 15:15:00 | 1663.95 | 1663.80 | 1672.76 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 09:30:00 | 1656.35 | 1662.56 | 1671.38 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-13 12:45:00 | 1658.00 | 1662.27 | 1669.11 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 14:15:00 | 1684.40 | 1667.34 | 1670.27 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-13 14:15:00 | 1684.40 | 1667.34 | 1670.27 | SL hit (close>ema400) qty=1.00 sl=1670.27 alert=retest1 |

### Cycle 88 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 1680.05 | 1673.60 | 1672.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 12:15:00 | 1692.05 | 1678.59 | 1675.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 09:15:00 | 1657.35 | 1676.71 | 1675.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 09:15:00 | 1657.35 | 1676.71 | 1675.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 09:15:00 | 1657.35 | 1676.71 | 1675.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:00:00 | 1657.35 | 1676.71 | 1675.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2024-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 10:15:00 | 1659.30 | 1673.23 | 1674.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 14:15:00 | 1652.00 | 1663.07 | 1668.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 11:15:00 | 1671.20 | 1662.49 | 1666.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 11:15:00 | 1671.20 | 1662.49 | 1666.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 1671.20 | 1662.49 | 1666.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 11:45:00 | 1672.35 | 1662.49 | 1666.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 1670.90 | 1664.18 | 1666.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:15:00 | 1674.10 | 1664.18 | 1666.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 1675.45 | 1666.97 | 1667.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 15:00:00 | 1675.45 | 1666.97 | 1667.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2024-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 15:15:00 | 1678.95 | 1669.37 | 1668.61 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-18 09:15:00 | 1665.00 | 1668.30 | 1668.63 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 11:15:00 | 1677.35 | 1670.11 | 1669.43 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 09:15:00 | 1646.60 | 1664.75 | 1667.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 10:15:00 | 1629.80 | 1657.76 | 1663.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-21 14:15:00 | 1648.70 | 1647.38 | 1655.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-21 15:00:00 | 1648.70 | 1647.38 | 1655.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 1651.90 | 1647.71 | 1650.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 10:15:00 | 1641.80 | 1647.71 | 1650.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 14:15:00 | 1662.00 | 1653.45 | 1652.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2024-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 14:15:00 | 1662.00 | 1653.45 | 1652.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 10:15:00 | 1670.75 | 1658.04 | 1655.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 14:15:00 | 1661.55 | 1662.09 | 1658.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 14:15:00 | 1661.55 | 1662.09 | 1658.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 1661.55 | 1662.09 | 1658.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 15:00:00 | 1661.55 | 1662.09 | 1658.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 1650.40 | 1659.75 | 1657.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:15:00 | 1647.80 | 1659.75 | 1657.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2024-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 09:15:00 | 1637.65 | 1655.33 | 1655.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 10:15:00 | 1635.10 | 1651.28 | 1653.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 14:15:00 | 1637.05 | 1626.76 | 1634.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 14:15:00 | 1637.05 | 1626.76 | 1634.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 1637.05 | 1626.76 | 1634.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 15:00:00 | 1637.05 | 1626.76 | 1634.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 1634.00 | 1628.20 | 1634.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 09:15:00 | 1625.45 | 1628.20 | 1634.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 10:00:00 | 1628.00 | 1628.16 | 1633.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 10:15:00 | 1544.18 | 1586.20 | 1592.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 10:15:00 | 1546.60 | 1586.20 | 1592.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-05 09:15:00 | 1579.95 | 1558.64 | 1571.64 | SL hit (close>ema200) qty=0.50 sl=1558.64 alert=retest2 |

### Cycle 96 — BUY (started 2024-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 12:15:00 | 1604.95 | 1582.88 | 1580.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 14:15:00 | 1630.05 | 1596.68 | 1587.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 09:15:00 | 1661.40 | 1667.67 | 1653.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-11 09:15:00 | 1661.40 | 1667.67 | 1653.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 1661.40 | 1667.67 | 1653.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:15:00 | 1703.00 | 1661.20 | 1658.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-05 14:15:00 | 1873.30 | 1850.50 | 1839.26 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 11:15:00 | 1841.90 | 1846.26 | 1846.48 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2024-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 12:15:00 | 1869.95 | 1851.00 | 1848.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 14:15:00 | 1878.45 | 1858.28 | 1852.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 12:15:00 | 1867.50 | 1868.45 | 1860.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-11 13:00:00 | 1867.50 | 1868.45 | 1860.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 1862.10 | 1869.47 | 1863.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 10:00:00 | 1862.10 | 1869.47 | 1863.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 1873.00 | 1870.17 | 1864.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 10:30:00 | 1861.65 | 1870.17 | 1864.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 13:15:00 | 1865.20 | 1869.46 | 1865.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 13:45:00 | 1864.10 | 1869.46 | 1865.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 14:15:00 | 1864.45 | 1868.46 | 1865.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 14:30:00 | 1866.90 | 1868.46 | 1865.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 1868.85 | 1868.53 | 1865.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:15:00 | 1860.15 | 1868.53 | 1865.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 1868.95 | 1868.62 | 1866.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 11:00:00 | 1873.95 | 1869.68 | 1866.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 11:45:00 | 1887.85 | 1877.20 | 1872.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 15:15:00 | 1878.95 | 1878.79 | 1874.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 12:00:00 | 1876.25 | 1881.42 | 1877.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 1867.85 | 1878.70 | 1876.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 13:00:00 | 1867.85 | 1878.70 | 1876.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 1901.15 | 1883.19 | 1879.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 13:45:00 | 1871.75 | 1883.19 | 1879.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 1890.65 | 1892.37 | 1885.01 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-23 12:15:00 | 1874.10 | 1887.49 | 1888.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 12:15:00 | 1874.10 | 1887.49 | 1888.66 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 09:15:00 | 1912.50 | 1890.45 | 1889.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 12:15:00 | 1918.95 | 1901.74 | 1895.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 1898.65 | 1904.44 | 1899.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 09:15:00 | 1898.65 | 1904.44 | 1899.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 1898.65 | 1904.44 | 1899.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 12:00:00 | 1918.30 | 1906.24 | 1900.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 15:15:00 | 1984.35 | 1990.23 | 1990.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2024-08-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 15:15:00 | 1984.35 | 1990.23 | 1990.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 1968.45 | 1985.87 | 1988.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1952.75 | 1950.00 | 1964.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 1952.75 | 1950.00 | 1964.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1952.75 | 1950.00 | 1964.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 10:15:00 | 1943.60 | 1950.00 | 1964.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 11:45:00 | 1945.85 | 1948.14 | 1960.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 12:45:00 | 1943.65 | 1948.79 | 1959.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:30:00 | 1943.85 | 1943.62 | 1956.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 1942.85 | 1936.83 | 1948.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 12:00:00 | 1934.00 | 1936.26 | 1946.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 15:00:00 | 1934.95 | 1938.05 | 1945.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 09:15:00 | 1967.95 | 1943.70 | 1946.57 | SL hit (close>static) qty=1.00 sl=1952.75 alert=retest2 |

### Cycle 102 — BUY (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 11:15:00 | 1969.40 | 1952.22 | 1950.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 09:15:00 | 1971.65 | 1963.87 | 1957.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 14:15:00 | 1966.35 | 1970.24 | 1963.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 14:15:00 | 1966.35 | 1970.24 | 1963.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 1966.35 | 1970.24 | 1963.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 15:00:00 | 1966.35 | 1970.24 | 1963.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 1959.25 | 1968.80 | 1964.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:00:00 | 1959.25 | 1968.80 | 1964.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 1969.95 | 1969.03 | 1964.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:30:00 | 1971.00 | 1969.03 | 1964.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 11:15:00 | 1966.05 | 1968.44 | 1964.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 12:00:00 | 1966.05 | 1968.44 | 1964.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 12:15:00 | 1965.05 | 1967.76 | 1964.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 12:45:00 | 1965.55 | 1967.76 | 1964.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 13:15:00 | 1961.25 | 1966.46 | 1964.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 14:00:00 | 1961.25 | 1966.46 | 1964.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 1952.45 | 1963.66 | 1963.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 15:00:00 | 1952.45 | 1963.66 | 1963.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2024-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 15:15:00 | 1952.55 | 1961.43 | 1962.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 09:15:00 | 1940.90 | 1957.33 | 1960.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-13 11:15:00 | 1959.95 | 1957.20 | 1959.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 11:15:00 | 1959.95 | 1957.20 | 1959.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 1959.95 | 1957.20 | 1959.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 11:45:00 | 1960.75 | 1957.20 | 1959.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 1954.50 | 1956.66 | 1959.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:30:00 | 1958.90 | 1956.66 | 1959.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 1971.80 | 1956.84 | 1958.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 10:00:00 | 1971.80 | 1956.84 | 1958.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 10:15:00 | 1974.80 | 1960.43 | 1959.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 09:15:00 | 1997.70 | 1976.20 | 1968.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 10:15:00 | 2013.00 | 2016.92 | 1998.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 11:00:00 | 2013.00 | 2016.92 | 1998.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 12:15:00 | 1996.70 | 2010.71 | 1998.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 13:00:00 | 1996.70 | 2010.71 | 1998.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 13:15:00 | 2000.10 | 2008.59 | 1998.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 13:45:00 | 1998.20 | 2008.59 | 1998.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 14:15:00 | 2014.55 | 2009.78 | 2000.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 15:15:00 | 2019.90 | 2009.78 | 2000.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-04 10:15:00 | 2221.89 | 2192.86 | 2179.10 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2024-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 12:15:00 | 2221.40 | 2236.77 | 2238.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-10 10:15:00 | 2201.00 | 2225.13 | 2232.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-13 09:15:00 | 2130.05 | 2128.56 | 2148.99 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-13 11:30:00 | 2116.40 | 2126.00 | 2144.27 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 2099.60 | 2098.49 | 2113.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:45:00 | 2099.65 | 2098.49 | 2113.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 13:15:00 | 2103.75 | 2100.57 | 2109.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 13:30:00 | 2108.95 | 2100.57 | 2109.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 2102.55 | 2101.72 | 2108.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-18 09:15:00 | 2100.55 | 2101.72 | 2108.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 11:15:00 | 2128.80 | 2107.31 | 2109.39 | SL hit (close>ema400) qty=1.00 sl=2109.39 alert=retest1 |

### Cycle 106 — BUY (started 2024-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 12:15:00 | 2131.00 | 2112.05 | 2111.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-18 14:15:00 | 2143.55 | 2121.72 | 2116.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 10:15:00 | 2229.00 | 2251.20 | 2229.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 10:15:00 | 2229.00 | 2251.20 | 2229.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 2229.00 | 2251.20 | 2229.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:00:00 | 2229.00 | 2251.20 | 2229.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 2228.50 | 2246.66 | 2229.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 12:15:00 | 2225.35 | 2246.66 | 2229.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 12:15:00 | 2226.30 | 2242.59 | 2229.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 12:45:00 | 2218.60 | 2242.59 | 2229.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 13:15:00 | 2243.35 | 2242.74 | 2230.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 14:15:00 | 2222.50 | 2242.74 | 2230.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 2227.90 | 2239.77 | 2230.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 14:30:00 | 2220.90 | 2239.77 | 2230.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 2229.90 | 2237.80 | 2230.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:15:00 | 2233.90 | 2237.80 | 2230.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 2231.35 | 2236.51 | 2230.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 11:15:00 | 2251.00 | 2237.09 | 2231.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 12:00:00 | 2243.00 | 2238.27 | 2232.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 13:00:00 | 2245.40 | 2239.70 | 2233.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 10:15:00 | 2220.20 | 2239.02 | 2240.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2024-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 10:15:00 | 2220.20 | 2239.02 | 2240.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 12:15:00 | 2206.05 | 2228.27 | 2235.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 14:15:00 | 2240.80 | 2229.35 | 2234.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 14:15:00 | 2240.80 | 2229.35 | 2234.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 2240.80 | 2229.35 | 2234.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 2240.80 | 2229.35 | 2234.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 15:15:00 | 2235.55 | 2230.59 | 2234.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 09:15:00 | 2215.70 | 2230.59 | 2234.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 2104.91 | 2129.73 | 2144.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 14:15:00 | 2100.85 | 2092.06 | 2109.62 | SL hit (close>ema200) qty=0.50 sl=2092.06 alert=retest2 |

### Cycle 108 — BUY (started 2024-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 15:15:00 | 2090.00 | 2083.98 | 2083.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 09:15:00 | 2091.40 | 2085.46 | 2084.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 14:15:00 | 2101.85 | 2106.13 | 2096.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 14:15:00 | 2101.85 | 2106.13 | 2096.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 2101.85 | 2106.13 | 2096.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 14:45:00 | 2098.45 | 2106.13 | 2096.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 2101.05 | 2106.66 | 2099.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 10:45:00 | 2103.35 | 2106.66 | 2099.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 2100.95 | 2105.51 | 2099.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 14:00:00 | 2112.00 | 2106.22 | 2100.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 14:15:00 | 2076.80 | 2100.33 | 2098.75 | SL hit (close<static) qty=1.00 sl=2093.85 alert=retest2 |

### Cycle 109 — SELL (started 2024-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 15:15:00 | 2076.00 | 2095.47 | 2096.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 10:15:00 | 2052.60 | 2082.76 | 2090.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 09:15:00 | 2053.10 | 2035.71 | 2050.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 09:15:00 | 2053.10 | 2035.71 | 2050.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 2053.10 | 2035.71 | 2050.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:15:00 | 2017.00 | 2034.23 | 2048.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:00:00 | 2017.55 | 2030.90 | 2045.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 15:15:00 | 2020.95 | 2024.93 | 2038.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 10:15:00 | 2018.50 | 2025.94 | 2036.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 14:15:00 | 1919.90 | 1936.96 | 1955.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 14:15:00 | 1917.57 | 1936.96 | 1955.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 09:15:00 | 1916.15 | 1930.71 | 1949.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 09:15:00 | 1916.67 | 1930.71 | 1949.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-29 12:15:00 | 1905.60 | 1899.18 | 1915.96 | SL hit (close>ema200) qty=0.50 sl=1899.18 alert=retest2 |

### Cycle 110 — BUY (started 2024-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 11:15:00 | 1953.00 | 1926.60 | 1923.57 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 09:15:00 | 1905.70 | 1922.52 | 1922.96 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 15:15:00 | 1952.50 | 1922.78 | 1921.45 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 1880.35 | 1913.40 | 1917.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 1869.15 | 1904.55 | 1913.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 1930.45 | 1882.70 | 1890.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 1930.45 | 1882.70 | 1890.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 1930.45 | 1882.70 | 1890.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:00:00 | 1930.45 | 1882.70 | 1890.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 1914.45 | 1889.05 | 1892.41 | EMA400 retest candle locked (from downside) |

### Cycle 114 — BUY (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 15:15:00 | 1918.60 | 1894.96 | 1894.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 1923.00 | 1900.57 | 1897.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 1910.25 | 1923.95 | 1914.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 1910.25 | 1923.95 | 1914.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 1910.25 | 1923.95 | 1914.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 1910.25 | 1923.95 | 1914.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 1913.55 | 1921.87 | 1913.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 1909.90 | 1921.87 | 1913.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 1908.15 | 1919.12 | 1913.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:00:00 | 1908.15 | 1919.12 | 1913.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 1916.45 | 1918.59 | 1913.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 14:00:00 | 1921.80 | 1919.23 | 1914.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 14:30:00 | 1924.60 | 1918.70 | 1914.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 15:15:00 | 1921.00 | 1918.70 | 1914.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 11:15:00 | 1904.50 | 1914.43 | 1913.92 | SL hit (close<static) qty=1.00 sl=1907.00 alert=retest2 |

### Cycle 115 — SELL (started 2024-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 12:15:00 | 1903.40 | 1912.23 | 1912.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 1899.25 | 1907.09 | 1910.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 1917.85 | 1909.25 | 1910.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 10:15:00 | 1917.85 | 1909.25 | 1910.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 1917.85 | 1909.25 | 1910.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:00:00 | 1917.85 | 1909.25 | 1910.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 1907.80 | 1908.96 | 1910.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:30:00 | 1916.85 | 1908.96 | 1910.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 1913.70 | 1909.91 | 1910.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 13:00:00 | 1913.70 | 1909.91 | 1910.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 13:15:00 | 1916.85 | 1911.29 | 1911.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 13:45:00 | 1919.00 | 1911.29 | 1911.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2024-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 14:15:00 | 1921.05 | 1913.25 | 1912.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-11 15:15:00 | 1926.55 | 1915.91 | 1913.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-12 09:15:00 | 1897.00 | 1912.12 | 1912.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 1897.00 | 1912.12 | 1912.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 1897.00 | 1912.12 | 1912.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:30:00 | 1890.35 | 1912.12 | 1912.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 10:15:00 | 1886.00 | 1906.90 | 1909.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 1866.90 | 1883.29 | 1893.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 1891.95 | 1882.42 | 1890.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 1891.95 | 1882.42 | 1890.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 1891.95 | 1882.42 | 1890.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 1891.95 | 1882.42 | 1890.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 1871.85 | 1880.31 | 1889.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 11:15:00 | 1869.05 | 1880.31 | 1889.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 12:45:00 | 1868.55 | 1877.59 | 1886.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 13:30:00 | 1867.00 | 1875.76 | 1884.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 13:15:00 | 1839.95 | 1835.67 | 1835.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2024-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 13:15:00 | 1839.95 | 1835.67 | 1835.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 09:15:00 | 1847.05 | 1838.36 | 1836.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 09:15:00 | 1853.35 | 1853.37 | 1847.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 09:15:00 | 1853.35 | 1853.37 | 1847.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 1853.35 | 1853.37 | 1847.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:15:00 | 1844.50 | 1853.37 | 1847.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 1837.95 | 1850.29 | 1846.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:45:00 | 1833.55 | 1850.29 | 1846.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 1841.05 | 1848.44 | 1845.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 12:15:00 | 1846.00 | 1848.44 | 1845.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 10:15:00 | 1833.55 | 1856.74 | 1859.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 10:15:00 | 1833.55 | 1856.74 | 1859.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 12:15:00 | 1827.10 | 1847.11 | 1854.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 1854.60 | 1843.05 | 1849.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 09:15:00 | 1854.60 | 1843.05 | 1849.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 1854.60 | 1843.05 | 1849.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:30:00 | 1856.10 | 1843.05 | 1849.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 1860.80 | 1846.60 | 1850.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:30:00 | 1858.95 | 1846.60 | 1850.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2024-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 12:15:00 | 1893.95 | 1859.60 | 1855.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 13:15:00 | 1895.35 | 1866.75 | 1859.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 14:15:00 | 1935.65 | 1941.43 | 1921.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 14:15:00 | 1935.65 | 1941.43 | 1921.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 1935.65 | 1941.43 | 1921.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:45:00 | 1922.75 | 1941.43 | 1921.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 1951.65 | 1946.82 | 1936.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:30:00 | 1940.05 | 1946.82 | 1936.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 1955.85 | 1958.19 | 1948.73 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 11:15:00 | 1935.75 | 1945.19 | 1945.65 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2024-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 11:15:00 | 1949.55 | 1946.24 | 1945.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-12 13:15:00 | 1957.25 | 1949.03 | 1947.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-16 10:15:00 | 1961.60 | 1967.93 | 1960.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 10:15:00 | 1961.60 | 1967.93 | 1960.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 1961.60 | 1967.93 | 1960.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:00:00 | 1961.60 | 1967.93 | 1960.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 1970.00 | 1968.34 | 1961.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 12:30:00 | 1971.60 | 1969.36 | 1962.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 09:15:00 | 1940.95 | 1961.08 | 1960.87 | SL hit (close<static) qty=1.00 sl=1958.90 alert=retest2 |

### Cycle 123 — SELL (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 10:15:00 | 1940.45 | 1956.95 | 1959.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 11:15:00 | 1923.00 | 1950.16 | 1955.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 12:15:00 | 1875.40 | 1874.54 | 1897.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 13:00:00 | 1875.40 | 1874.54 | 1897.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 1896.95 | 1879.02 | 1897.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:00:00 | 1896.95 | 1879.02 | 1897.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 1901.10 | 1883.44 | 1897.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 15:00:00 | 1901.10 | 1883.44 | 1897.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 1902.15 | 1887.18 | 1898.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:15:00 | 1901.65 | 1887.18 | 1898.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 1899.30 | 1889.60 | 1898.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 09:15:00 | 1860.45 | 1901.96 | 1902.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 10:15:00 | 1886.60 | 1900.57 | 1901.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 13:15:00 | 1792.27 | 1818.18 | 1836.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-30 15:15:00 | 1825.00 | 1818.16 | 1833.58 | SL hit (close>ema200) qty=0.50 sl=1818.16 alert=retest2 |

### Cycle 124 — BUY (started 2025-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 12:15:00 | 1827.10 | 1811.32 | 1810.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 13:15:00 | 1828.10 | 1814.68 | 1811.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 10:15:00 | 1811.50 | 1821.95 | 1816.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 10:15:00 | 1811.50 | 1821.95 | 1816.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 1811.50 | 1821.95 | 1816.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 11:00:00 | 1811.50 | 1821.95 | 1816.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 11:15:00 | 1805.65 | 1818.69 | 1815.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 12:00:00 | 1805.65 | 1818.69 | 1815.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 12:15:00 | 1826.40 | 1820.23 | 1816.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 13:45:00 | 1830.10 | 1822.59 | 1818.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 10:00:00 | 1830.30 | 1827.90 | 1822.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 12:45:00 | 1832.30 | 1825.29 | 1822.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 15:15:00 | 1808.40 | 1819.12 | 1819.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2025-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 15:15:00 | 1808.40 | 1819.12 | 1819.83 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2025-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 10:15:00 | 1832.75 | 1821.37 | 1820.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 11:15:00 | 1848.80 | 1826.86 | 1823.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 11:15:00 | 1856.45 | 1860.86 | 1846.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-08 12:00:00 | 1856.45 | 1860.86 | 1846.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 1899.30 | 1867.62 | 1854.72 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2025-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 12:15:00 | 1850.40 | 1866.72 | 1868.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 14:15:00 | 1830.70 | 1856.85 | 1863.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 1847.35 | 1846.29 | 1856.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 1847.35 | 1846.29 | 1856.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 13:15:00 | 1855.15 | 1834.73 | 1841.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 14:00:00 | 1855.15 | 1834.73 | 1841.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2025-01-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 14:15:00 | 1897.00 | 1847.18 | 1846.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 1912.30 | 1866.95 | 1855.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 09:15:00 | 1920.85 | 1934.74 | 1912.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-20 10:15:00 | 1917.75 | 1934.74 | 1912.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 1919.80 | 1931.76 | 1913.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 09:15:00 | 1939.95 | 1925.65 | 1916.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 10:00:00 | 1933.50 | 1927.22 | 1918.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 14:15:00 | 1902.45 | 1918.04 | 1916.92 | SL hit (close<static) qty=1.00 sl=1910.00 alert=retest2 |

### Cycle 129 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 1884.10 | 1909.97 | 1913.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 1875.25 | 1898.21 | 1907.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-24 11:15:00 | 1839.90 | 1833.35 | 1853.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-24 12:00:00 | 1839.90 | 1833.35 | 1853.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 15:15:00 | 1772.00 | 1775.97 | 1794.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 09:15:00 | 1761.55 | 1775.97 | 1794.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 11:15:00 | 1801.75 | 1784.47 | 1793.65 | SL hit (close>static) qty=1.00 sl=1796.00 alert=retest2 |

### Cycle 130 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 1834.95 | 1799.63 | 1799.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 1839.60 | 1807.63 | 1802.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 13:15:00 | 1853.95 | 1855.63 | 1840.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 14:00:00 | 1853.95 | 1855.63 | 1840.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1807.95 | 1857.05 | 1849.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 1807.95 | 1857.05 | 1849.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1860.00 | 1857.64 | 1850.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:45:00 | 1868.40 | 1861.91 | 1852.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 10:15:00 | 1870.85 | 1868.10 | 1857.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 13:30:00 | 1874.80 | 1869.04 | 1861.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 13:30:00 | 1866.75 | 1873.71 | 1868.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 1874.25 | 1873.82 | 1869.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 10:45:00 | 1883.65 | 1874.21 | 1870.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-05 13:15:00 | 1863.90 | 1872.37 | 1870.68 | SL hit (close<static) qty=1.00 sl=1868.35 alert=retest2 |

### Cycle 131 — SELL (started 2025-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 14:15:00 | 1856.65 | 1869.22 | 1869.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-05 15:15:00 | 1850.90 | 1865.56 | 1867.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 10:15:00 | 1840.70 | 1839.87 | 1849.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 11:00:00 | 1840.70 | 1839.87 | 1849.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 1827.30 | 1829.59 | 1839.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 09:45:00 | 1806.45 | 1818.15 | 1828.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 1716.13 | 1741.42 | 1759.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 1729.70 | 1727.19 | 1744.34 | SL hit (close>ema200) qty=0.50 sl=1727.19 alert=retest2 |

### Cycle 132 — BUY (started 2025-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 13:15:00 | 1677.70 | 1666.05 | 1665.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 1698.70 | 1674.79 | 1670.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 09:15:00 | 1689.15 | 1693.32 | 1684.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 09:15:00 | 1689.15 | 1693.32 | 1684.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 1689.15 | 1693.32 | 1684.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:30:00 | 1685.75 | 1693.32 | 1684.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 1670.85 | 1688.23 | 1683.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:00:00 | 1670.85 | 1688.23 | 1683.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 1677.25 | 1686.04 | 1682.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 13:15:00 | 1682.35 | 1686.04 | 1682.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:30:00 | 1680.65 | 1684.13 | 1683.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 11:15:00 | 1676.60 | 1681.61 | 1682.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 11:15:00 | 1676.60 | 1681.61 | 1682.15 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2025-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-10 12:15:00 | 1692.85 | 1683.86 | 1683.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-10 14:15:00 | 1700.15 | 1688.57 | 1685.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-11 12:15:00 | 1677.00 | 1693.42 | 1690.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 12:15:00 | 1677.00 | 1693.42 | 1690.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 12:15:00 | 1677.00 | 1693.42 | 1690.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 13:00:00 | 1677.00 | 1693.42 | 1690.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 13:15:00 | 1697.15 | 1694.16 | 1690.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 14:15:00 | 1702.75 | 1694.16 | 1690.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 11:15:00 | 1673.00 | 1688.41 | 1689.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 11:15:00 | 1673.00 | 1688.41 | 1689.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 13:15:00 | 1665.25 | 1681.32 | 1685.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 13:15:00 | 1676.70 | 1673.40 | 1678.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-13 13:45:00 | 1675.05 | 1673.40 | 1678.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 1679.20 | 1674.56 | 1678.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 15:00:00 | 1679.20 | 1674.56 | 1678.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 1680.50 | 1675.75 | 1678.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:15:00 | 1689.60 | 1675.75 | 1678.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 1684.50 | 1677.50 | 1679.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 1689.30 | 1677.50 | 1679.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 1685.95 | 1679.19 | 1679.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:45:00 | 1685.00 | 1679.19 | 1679.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 11:15:00 | 1714.65 | 1686.28 | 1683.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 13:15:00 | 1726.10 | 1698.50 | 1689.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 1747.70 | 1749.54 | 1736.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 10:15:00 | 1740.35 | 1747.70 | 1737.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 1740.35 | 1747.70 | 1737.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 11:00:00 | 1740.35 | 1747.70 | 1737.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 11:15:00 | 1731.95 | 1744.55 | 1736.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 11:30:00 | 1729.20 | 1744.55 | 1736.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 12:15:00 | 1726.20 | 1740.88 | 1735.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 13:00:00 | 1726.20 | 1740.88 | 1735.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2025-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-20 15:15:00 | 1725.00 | 1732.70 | 1732.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-21 09:15:00 | 1707.90 | 1727.74 | 1730.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 14:15:00 | 1723.75 | 1719.89 | 1724.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-21 15:00:00 | 1723.75 | 1719.89 | 1724.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 1725.20 | 1720.95 | 1724.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:15:00 | 1746.80 | 1720.95 | 1724.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 1738.75 | 1724.51 | 1726.08 | EMA400 retest candle locked (from downside) |

### Cycle 138 — BUY (started 2025-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 10:15:00 | 1749.70 | 1729.55 | 1728.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 12:15:00 | 1756.75 | 1738.25 | 1732.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 14:15:00 | 1777.70 | 1783.65 | 1765.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 15:00:00 | 1777.70 | 1783.65 | 1765.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 12:15:00 | 1762.60 | 1777.37 | 1769.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:00:00 | 1762.60 | 1777.37 | 1769.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 1780.00 | 1777.90 | 1770.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 09:15:00 | 1780.80 | 1775.44 | 1770.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 14:15:00 | 1781.20 | 1774.97 | 1772.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 14:45:00 | 1784.45 | 1779.16 | 1774.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 09:30:00 | 1800.95 | 1789.59 | 1784.04 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 1793.95 | 1790.40 | 1785.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:30:00 | 1786.80 | 1790.40 | 1785.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 1791.85 | 1790.82 | 1786.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 15:15:00 | 1796.00 | 1790.91 | 1786.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 09:30:00 | 1797.05 | 1792.40 | 1788.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 1754.80 | 1811.83 | 1814.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 1754.80 | 1811.83 | 1814.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 11:15:00 | 1730.85 | 1784.40 | 1800.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 11:15:00 | 1774.00 | 1759.29 | 1775.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 11:15:00 | 1774.00 | 1759.29 | 1775.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 11:15:00 | 1774.00 | 1759.29 | 1775.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:00:00 | 1774.00 | 1759.29 | 1775.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 1773.55 | 1762.14 | 1775.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:30:00 | 1778.70 | 1762.14 | 1775.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 1774.50 | 1764.61 | 1775.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 13:30:00 | 1776.85 | 1764.61 | 1775.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 1773.90 | 1766.47 | 1775.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 14:30:00 | 1779.85 | 1766.47 | 1775.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 15:15:00 | 1775.80 | 1768.34 | 1775.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-09 09:15:00 | 1784.60 | 1768.34 | 1775.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 1782.25 | 1771.12 | 1775.93 | EMA400 retest candle locked (from downside) |

### Cycle 140 — BUY (started 2025-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 12:15:00 | 1790.10 | 1778.54 | 1778.43 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-11 09:15:00 | 1762.35 | 1777.75 | 1778.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-11 13:15:00 | 1734.85 | 1758.72 | 1768.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 09:15:00 | 1759.30 | 1745.06 | 1758.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-15 09:15:00 | 1759.30 | 1745.06 | 1758.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 09:15:00 | 1759.30 | 1745.06 | 1758.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 09:45:00 | 1770.30 | 1745.06 | 1758.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 10:15:00 | 1773.00 | 1750.65 | 1759.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 11:00:00 | 1773.00 | 1750.65 | 1759.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 11:15:00 | 1788.80 | 1758.28 | 1762.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 12:00:00 | 1788.80 | 1758.28 | 1762.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — BUY (started 2025-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 12:15:00 | 1809.70 | 1768.56 | 1766.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 14:15:00 | 1825.00 | 1786.77 | 1775.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 10:15:00 | 1793.10 | 1797.06 | 1784.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 11:00:00 | 1793.10 | 1797.06 | 1784.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1790.30 | 1799.66 | 1791.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 09:45:00 | 1790.00 | 1799.66 | 1791.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 1780.00 | 1795.73 | 1790.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 11:00:00 | 1780.00 | 1795.73 | 1790.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 11:15:00 | 1784.00 | 1793.38 | 1789.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 11:30:00 | 1780.00 | 1793.38 | 1789.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 13:15:00 | 1793.10 | 1792.93 | 1790.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:30:00 | 1807.80 | 1795.02 | 1791.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 10:00:00 | 1806.30 | 1795.02 | 1791.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 14:15:00 | 1852.10 | 1857.61 | 1857.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2025-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 14:15:00 | 1852.10 | 1857.61 | 1857.88 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 09:15:00 | 1862.20 | 1858.46 | 1858.22 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-28 10:15:00 | 1840.00 | 1854.77 | 1856.56 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 09:15:00 | 1879.20 | 1857.88 | 1855.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 09:15:00 | 1886.30 | 1873.91 | 1866.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 12:15:00 | 1863.30 | 1873.79 | 1868.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 12:15:00 | 1863.30 | 1873.79 | 1868.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 12:15:00 | 1863.30 | 1873.79 | 1868.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 13:00:00 | 1863.30 | 1873.79 | 1868.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 13:15:00 | 1864.60 | 1871.95 | 1868.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 13:30:00 | 1865.30 | 1871.95 | 1868.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 14:15:00 | 1866.80 | 1870.92 | 1867.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 14:45:00 | 1860.60 | 1870.92 | 1867.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 15:15:00 | 1870.00 | 1870.74 | 1868.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:15:00 | 1867.20 | 1870.74 | 1868.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 1875.50 | 1871.69 | 1868.76 | EMA400 retest candle locked (from upside) |

### Cycle 147 — SELL (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-05 11:15:00 | 1852.60 | 1865.38 | 1866.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-05 12:15:00 | 1842.50 | 1860.80 | 1864.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 1811.80 | 1811.43 | 1829.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 11:00:00 | 1811.80 | 1811.43 | 1829.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 1824.60 | 1813.07 | 1824.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:45:00 | 1823.00 | 1813.07 | 1824.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 1826.70 | 1815.80 | 1824.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 1822.20 | 1815.80 | 1824.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1817.10 | 1816.06 | 1824.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 10:30:00 | 1809.00 | 1814.25 | 1822.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 11:15:00 | 1828.40 | 1801.27 | 1800.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1828.40 | 1801.27 | 1800.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 1833.00 | 1807.61 | 1803.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 1843.60 | 1851.54 | 1836.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 09:45:00 | 1844.20 | 1851.54 | 1836.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 1840.10 | 1850.02 | 1839.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 1839.50 | 1850.02 | 1839.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 1849.00 | 1849.82 | 1840.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 09:15:00 | 1874.80 | 1850.89 | 1842.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 15:15:00 | 1857.90 | 1863.23 | 1859.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 10:00:00 | 1856.20 | 1860.97 | 1858.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 1868.60 | 1877.05 | 1877.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 1868.60 | 1877.05 | 1877.43 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 1893.60 | 1878.30 | 1876.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 12:15:00 | 1901.10 | 1882.86 | 1878.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 09:15:00 | 1877.50 | 1886.39 | 1882.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 1877.50 | 1886.39 | 1882.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 1877.50 | 1886.39 | 1882.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 10:00:00 | 1877.50 | 1886.39 | 1882.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 1852.80 | 1879.67 | 1879.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:00:00 | 1852.80 | 1879.67 | 1879.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — SELL (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 11:15:00 | 1843.80 | 1872.50 | 1876.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 13:15:00 | 1839.40 | 1862.28 | 1870.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 12:15:00 | 1854.20 | 1844.00 | 1850.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 12:15:00 | 1854.20 | 1844.00 | 1850.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 1854.20 | 1844.00 | 1850.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:00:00 | 1854.20 | 1844.00 | 1850.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 1859.50 | 1847.10 | 1851.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:00:00 | 1859.50 | 1847.10 | 1851.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 1855.00 | 1848.14 | 1850.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:00:00 | 1855.00 | 1848.14 | 1850.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 1849.00 | 1848.32 | 1850.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:30:00 | 1854.40 | 1848.32 | 1850.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 1860.00 | 1850.65 | 1851.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:00:00 | 1860.00 | 1850.65 | 1851.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 1845.60 | 1849.64 | 1850.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:30:00 | 1860.00 | 1849.64 | 1850.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 1850.00 | 1849.71 | 1850.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:45:00 | 1850.00 | 1849.71 | 1850.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 1850.70 | 1849.91 | 1850.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 1850.70 | 1849.91 | 1850.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 1850.10 | 1849.95 | 1850.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 1859.30 | 1849.95 | 1850.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — BUY (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 09:15:00 | 1858.10 | 1851.58 | 1851.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 11:15:00 | 1872.20 | 1856.65 | 1853.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 10:15:00 | 1869.40 | 1870.23 | 1863.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-02 11:00:00 | 1869.40 | 1870.23 | 1863.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 1862.70 | 1868.73 | 1863.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:30:00 | 1860.90 | 1868.73 | 1863.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 1861.50 | 1867.28 | 1862.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 13:00:00 | 1861.50 | 1867.28 | 1862.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 1863.40 | 1866.50 | 1863.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 09:15:00 | 1871.30 | 1864.65 | 1862.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 13:15:00 | 1856.60 | 1862.53 | 1862.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — SELL (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 13:15:00 | 1856.60 | 1862.53 | 1862.68 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 14:15:00 | 1864.20 | 1862.86 | 1862.82 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 15:15:00 | 1860.00 | 1862.29 | 1862.56 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 1875.00 | 1863.83 | 1863.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 1881.00 | 1871.44 | 1867.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 14:15:00 | 1877.10 | 1877.17 | 1872.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 15:00:00 | 1877.10 | 1877.17 | 1872.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 1892.60 | 1880.56 | 1874.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 10:15:00 | 1899.80 | 1880.56 | 1874.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 13:15:00 | 1944.00 | 1963.18 | 1965.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 1944.00 | 1963.18 | 1965.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 1931.50 | 1950.29 | 1958.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 09:15:00 | 1926.10 | 1919.14 | 1935.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 10:00:00 | 1926.10 | 1919.14 | 1935.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 1925.10 | 1920.34 | 1934.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 10:45:00 | 1938.40 | 1920.34 | 1934.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 1938.70 | 1923.92 | 1931.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 1938.70 | 1923.92 | 1931.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 1943.30 | 1927.79 | 1932.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 1933.00 | 1927.79 | 1932.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 10:30:00 | 1931.30 | 1929.58 | 1932.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 13:15:00 | 1943.30 | 1935.58 | 1934.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 13:15:00 | 1943.30 | 1935.58 | 1934.90 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 1926.40 | 1934.16 | 1934.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 13:15:00 | 1920.00 | 1931.33 | 1933.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 11:15:00 | 1933.80 | 1926.96 | 1929.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 11:15:00 | 1933.80 | 1926.96 | 1929.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 1933.80 | 1926.96 | 1929.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:45:00 | 1931.50 | 1926.96 | 1929.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 1932.00 | 1927.97 | 1930.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:15:00 | 1932.20 | 1927.97 | 1930.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 1930.10 | 1928.40 | 1930.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:30:00 | 1935.00 | 1928.40 | 1930.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 1933.40 | 1929.40 | 1930.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 15:00:00 | 1933.40 | 1929.40 | 1930.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 1929.60 | 1929.44 | 1930.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 1934.70 | 1929.44 | 1930.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1933.90 | 1930.33 | 1930.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:30:00 | 1941.60 | 1930.33 | 1930.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 1931.00 | 1930.46 | 1930.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:15:00 | 1928.00 | 1930.46 | 1930.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 15:00:00 | 1928.00 | 1929.88 | 1930.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 1942.80 | 1931.57 | 1930.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — BUY (started 2025-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 09:15:00 | 1942.80 | 1931.57 | 1930.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 10:15:00 | 1962.30 | 1937.72 | 1933.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 15:15:00 | 2010.00 | 2010.68 | 2000.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 09:15:00 | 2009.50 | 2010.68 | 2000.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 2040.10 | 2050.99 | 2039.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 2040.10 | 2050.99 | 2039.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 2032.20 | 2047.23 | 2039.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:00:00 | 2032.20 | 2047.23 | 2039.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 2040.30 | 2045.85 | 2039.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 12:45:00 | 2042.60 | 2045.08 | 2039.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 14:30:00 | 2042.60 | 2043.40 | 2039.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 15:00:00 | 2042.60 | 2043.40 | 2039.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 09:45:00 | 2042.80 | 2043.99 | 2040.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 2045.20 | 2044.23 | 2040.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:45:00 | 2045.20 | 2044.23 | 2040.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 2041.40 | 2043.67 | 2040.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:00:00 | 2041.40 | 2043.67 | 2040.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 2036.90 | 2042.31 | 2040.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:00:00 | 2036.90 | 2042.31 | 2040.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 2035.10 | 2040.87 | 2040.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:00:00 | 2035.10 | 2040.87 | 2040.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-03 14:15:00 | 2033.50 | 2039.40 | 2039.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — SELL (started 2025-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 14:15:00 | 2033.50 | 2039.40 | 2039.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 2031.70 | 2037.15 | 2038.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 2037.80 | 2036.72 | 2037.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 14:15:00 | 2037.80 | 2036.72 | 2037.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 2037.80 | 2036.72 | 2037.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 2037.80 | 2036.72 | 2037.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 2035.00 | 2036.38 | 2037.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 2035.00 | 2036.38 | 2037.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — BUY (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 09:15:00 | 2049.00 | 2038.90 | 2038.63 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 11:15:00 | 2031.70 | 2037.72 | 2038.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 14:15:00 | 2026.70 | 2034.68 | 2036.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 11:15:00 | 2031.30 | 2030.05 | 2033.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 11:15:00 | 2031.30 | 2030.05 | 2033.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 2031.30 | 2030.05 | 2033.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:00:00 | 2031.30 | 2030.05 | 2033.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 2039.30 | 2031.90 | 2033.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:00:00 | 2039.30 | 2031.90 | 2033.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 2034.70 | 2032.46 | 2034.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 2030.70 | 2034.06 | 2034.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 10:15:00 | 2024.30 | 2014.86 | 2013.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — BUY (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 10:15:00 | 2024.30 | 2014.86 | 2013.79 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 12:15:00 | 1995.20 | 2010.47 | 2011.96 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 2014.00 | 2012.90 | 2012.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 2029.40 | 2016.20 | 2014.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 10:15:00 | 2012.90 | 2015.54 | 2014.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 10:15:00 | 2012.90 | 2015.54 | 2014.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 2012.90 | 2015.54 | 2014.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 2012.90 | 2015.54 | 2014.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — SELL (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 11:15:00 | 2000.00 | 2012.43 | 2012.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-15 12:15:00 | 1996.70 | 2009.28 | 2011.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 15:15:00 | 2007.80 | 2006.57 | 2009.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 15:15:00 | 2007.80 | 2006.57 | 2009.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 2007.80 | 2006.57 | 2009.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:15:00 | 2013.00 | 2006.57 | 2009.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 2026.00 | 2010.46 | 2010.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:30:00 | 2029.60 | 2010.46 | 2010.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — BUY (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 10:15:00 | 2042.00 | 2016.76 | 2013.74 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 13:15:00 | 1976.70 | 2010.72 | 2011.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 14:15:00 | 1969.50 | 2002.48 | 2008.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 10:15:00 | 1925.60 | 1925.32 | 1945.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 11:00:00 | 1925.60 | 1925.32 | 1945.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 1951.30 | 1935.06 | 1943.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:00:00 | 1951.30 | 1935.06 | 1943.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 1954.20 | 1938.89 | 1944.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 1958.10 | 1938.89 | 1944.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — BUY (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 11:15:00 | 1965.00 | 1949.25 | 1948.51 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 10:15:00 | 1940.80 | 1950.04 | 1950.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 1914.00 | 1936.05 | 1942.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 12:15:00 | 1912.20 | 1908.67 | 1920.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-25 13:00:00 | 1912.20 | 1908.67 | 1920.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 1921.20 | 1911.18 | 1920.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:00:00 | 1921.20 | 1911.18 | 1920.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 1921.10 | 1913.16 | 1920.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:30:00 | 1923.90 | 1913.16 | 1920.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 1916.00 | 1913.73 | 1920.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:45:00 | 1912.30 | 1917.18 | 1920.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 13:30:00 | 1915.00 | 1910.43 | 1913.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 15:15:00 | 1915.20 | 1911.84 | 1914.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 15:15:00 | 1923.00 | 1914.40 | 1913.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — BUY (started 2025-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 15:15:00 | 1923.00 | 1914.40 | 1913.70 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 1902.30 | 1911.98 | 1912.66 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 11:15:00 | 1919.40 | 1913.72 | 1913.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 12:15:00 | 1923.80 | 1915.74 | 1914.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 1909.20 | 1919.93 | 1917.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 1909.20 | 1919.93 | 1917.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1909.20 | 1919.93 | 1917.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 1909.20 | 1919.93 | 1917.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 1904.80 | 1916.91 | 1916.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:00:00 | 1904.80 | 1916.91 | 1916.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 1888.00 | 1911.12 | 1913.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 12:15:00 | 1877.60 | 1904.42 | 1910.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 13:15:00 | 1890.00 | 1880.26 | 1890.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 13:15:00 | 1890.00 | 1880.26 | 1890.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 1890.00 | 1880.26 | 1890.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 1890.00 | 1880.26 | 1890.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1896.20 | 1883.45 | 1891.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 1896.20 | 1883.45 | 1891.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 1895.00 | 1885.76 | 1891.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 09:15:00 | 1878.80 | 1885.76 | 1891.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 12:15:00 | 1901.10 | 1888.72 | 1890.93 | SL hit (close>static) qty=1.00 sl=1900.50 alert=retest2 |

### Cycle 176 — BUY (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 15:15:00 | 1897.30 | 1893.17 | 1892.63 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 1867.50 | 1888.04 | 1890.34 | EMA200 below EMA400 |

### Cycle 178 — BUY (started 2025-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 13:15:00 | 1900.00 | 1887.32 | 1886.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 14:15:00 | 1903.90 | 1890.64 | 1888.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 09:15:00 | 1889.90 | 1892.79 | 1889.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 09:15:00 | 1889.90 | 1892.79 | 1889.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1889.90 | 1892.79 | 1889.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:00:00 | 1889.90 | 1892.79 | 1889.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 1889.20 | 1892.07 | 1889.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:00:00 | 1889.20 | 1892.07 | 1889.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 1878.20 | 1889.30 | 1888.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:00:00 | 1878.20 | 1889.30 | 1888.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 1895.10 | 1890.46 | 1889.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 13:15:00 | 1899.60 | 1890.46 | 1889.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 14:30:00 | 1901.20 | 1892.83 | 1890.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 09:15:00 | 1899.70 | 1892.23 | 1890.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 10:00:00 | 1897.90 | 1893.36 | 1891.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 1894.70 | 1893.63 | 1891.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:15:00 | 1892.80 | 1893.63 | 1891.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 1890.10 | 1892.92 | 1891.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:45:00 | 1885.40 | 1892.92 | 1891.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 1896.20 | 1893.58 | 1891.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:30:00 | 1887.30 | 1893.58 | 1891.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 1907.70 | 1899.12 | 1895.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 11:00:00 | 1918.50 | 1905.36 | 1900.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 09:15:00 | 1916.60 | 1913.01 | 1906.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 14:15:00 | 1898.40 | 1904.00 | 1904.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — SELL (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 14:15:00 | 1898.40 | 1904.00 | 1904.42 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 1973.90 | 1916.81 | 1910.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 1978.00 | 1929.05 | 1916.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 14:15:00 | 1945.20 | 1949.92 | 1932.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 15:00:00 | 1945.20 | 1949.92 | 1932.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 1928.20 | 1944.79 | 1932.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:00:00 | 1928.20 | 1944.79 | 1932.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 1935.00 | 1942.83 | 1933.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 11:30:00 | 1942.40 | 1944.29 | 1934.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 11:00:00 | 1937.80 | 1962.67 | 1961.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 11:15:00 | 1937.10 | 1957.56 | 1958.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 1937.10 | 1957.56 | 1958.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 12:15:00 | 1916.00 | 1949.25 | 1955.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 11:15:00 | 1925.20 | 1924.20 | 1937.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 12:00:00 | 1925.20 | 1924.20 | 1937.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 1926.70 | 1925.93 | 1936.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:30:00 | 1934.30 | 1925.93 | 1936.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1833.50 | 1839.25 | 1851.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 11:00:00 | 1819.30 | 1835.26 | 1848.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 14:15:00 | 1819.40 | 1831.87 | 1843.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 15:15:00 | 1820.00 | 1830.38 | 1841.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 1873.90 | 1832.92 | 1833.75 | SL hit (close>static) qty=1.00 sl=1863.40 alert=retest2 |

### Cycle 182 — BUY (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 10:15:00 | 1859.70 | 1838.28 | 1836.11 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 15:15:00 | 1833.80 | 1844.19 | 1844.93 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 1848.10 | 1841.79 | 1840.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 13:15:00 | 1860.00 | 1848.08 | 1844.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 1846.90 | 1849.84 | 1846.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 1846.90 | 1849.84 | 1846.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 1846.90 | 1849.84 | 1846.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:45:00 | 1842.20 | 1849.84 | 1846.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1851.50 | 1850.17 | 1846.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 11:45:00 | 1860.60 | 1852.94 | 1848.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 13:15:00 | 1868.00 | 1891.46 | 1892.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 1868.00 | 1891.46 | 1892.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 10:15:00 | 1865.00 | 1877.71 | 1882.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 15:15:00 | 1871.60 | 1867.73 | 1874.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-22 09:15:00 | 1877.00 | 1867.73 | 1874.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 1884.80 | 1871.14 | 1875.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:30:00 | 1879.00 | 1871.14 | 1875.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 1884.10 | 1873.73 | 1876.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:30:00 | 1886.10 | 1873.73 | 1876.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 1886.40 | 1877.11 | 1877.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 13:00:00 | 1886.40 | 1877.11 | 1877.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — BUY (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 13:15:00 | 1886.60 | 1879.01 | 1878.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 10:15:00 | 1891.70 | 1883.81 | 1880.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 09:15:00 | 1884.80 | 1889.23 | 1885.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 1884.80 | 1889.23 | 1885.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1884.80 | 1889.23 | 1885.48 | EMA400 retest candle locked (from upside) |

### Cycle 187 — SELL (started 2025-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 12:15:00 | 1876.40 | 1882.15 | 1882.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 09:15:00 | 1864.70 | 1877.23 | 1880.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 11:15:00 | 1880.00 | 1877.59 | 1879.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 11:15:00 | 1880.00 | 1877.59 | 1879.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 1880.00 | 1877.59 | 1879.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:30:00 | 1879.00 | 1877.59 | 1879.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 12:15:00 | 1880.00 | 1878.07 | 1879.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:45:00 | 1880.00 | 1878.07 | 1879.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 1880.50 | 1878.56 | 1879.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:00:00 | 1880.50 | 1878.56 | 1879.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 1880.30 | 1878.90 | 1879.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:45:00 | 1882.90 | 1878.90 | 1879.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 1886.00 | 1880.32 | 1880.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 1899.20 | 1880.32 | 1880.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — BUY (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-26 09:15:00 | 1900.10 | 1884.28 | 1882.23 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 09:15:00 | 1873.80 | 1888.63 | 1889.96 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 1897.80 | 1890.38 | 1890.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 1906.10 | 1894.54 | 1892.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 13:15:00 | 1916.00 | 1924.71 | 1917.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 13:15:00 | 1916.00 | 1924.71 | 1917.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 1916.00 | 1924.71 | 1917.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:00:00 | 1916.00 | 1924.71 | 1917.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 1903.50 | 1920.47 | 1916.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:00:00 | 1903.50 | 1920.47 | 1916.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 1903.10 | 1917.00 | 1914.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:15:00 | 1904.60 | 1917.00 | 1914.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 1884.50 | 1908.02 | 1911.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 1876.40 | 1894.54 | 1903.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 12:15:00 | 1885.20 | 1880.89 | 1891.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 13:00:00 | 1885.20 | 1880.89 | 1891.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 1890.00 | 1882.18 | 1889.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 1894.50 | 1882.18 | 1889.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1891.10 | 1883.96 | 1889.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 12:15:00 | 1885.20 | 1885.68 | 1889.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 2000.00 | 1880.69 | 1869.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 2000.00 | 1880.69 | 1869.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 12:15:00 | 2006.70 | 1939.24 | 1901.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 1997.00 | 2002.40 | 1972.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 10:00:00 | 1997.00 | 2002.40 | 1972.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 2004.40 | 2007.49 | 2000.48 | EMA400 retest candle locked (from upside) |

### Cycle 193 — SELL (started 2025-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 13:15:00 | 1984.20 | 1996.96 | 1997.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 14:15:00 | 1978.20 | 1993.21 | 1995.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 12:15:00 | 1990.90 | 1986.23 | 1990.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 12:15:00 | 1990.90 | 1986.23 | 1990.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 1990.90 | 1986.23 | 1990.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:00:00 | 1990.90 | 1986.23 | 1990.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 1995.60 | 1988.11 | 1991.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:00:00 | 1995.60 | 1988.11 | 1991.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 1993.40 | 1989.16 | 1991.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:30:00 | 1992.80 | 1989.16 | 1991.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 1995.00 | 1990.33 | 1991.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 2003.50 | 1990.33 | 1991.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 2010.00 | 1994.26 | 1993.36 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 13:15:00 | 1987.20 | 1992.28 | 1992.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 15:15:00 | 1981.60 | 1989.09 | 1991.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 1991.30 | 1989.53 | 1991.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 1991.30 | 1989.53 | 1991.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1991.30 | 1989.53 | 1991.26 | EMA400 retest candle locked (from downside) |

### Cycle 196 — BUY (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 11:15:00 | 2006.10 | 1993.72 | 1992.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 12:15:00 | 2016.00 | 1998.18 | 1995.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 2014.20 | 2021.20 | 2012.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 2014.20 | 2021.20 | 2012.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 2014.20 | 2021.20 | 2012.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 2014.20 | 2021.20 | 2012.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 2015.10 | 2019.98 | 2013.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:30:00 | 2013.30 | 2019.98 | 2013.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 2013.10 | 2018.60 | 2013.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:45:00 | 2015.00 | 2018.60 | 2013.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 2012.60 | 2017.40 | 2012.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 12:30:00 | 2014.00 | 2017.40 | 2012.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 2008.90 | 2015.70 | 2012.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:45:00 | 2005.70 | 2015.70 | 2012.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 2015.00 | 2015.56 | 2012.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:30:00 | 2008.70 | 2015.56 | 2012.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 2015.00 | 2015.45 | 2013.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:15:00 | 2009.20 | 2015.45 | 2013.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 2013.50 | 2015.06 | 2013.07 | EMA400 retest candle locked (from upside) |

### Cycle 197 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 1992.60 | 2009.27 | 2011.03 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 09:15:00 | 2032.60 | 2011.34 | 2009.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 10:15:00 | 2036.20 | 2016.31 | 2012.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 2014.60 | 2030.17 | 2023.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 2014.60 | 2030.17 | 2023.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 2014.60 | 2030.17 | 2023.20 | EMA400 retest candle locked (from upside) |

### Cycle 199 — SELL (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 12:15:00 | 2003.80 | 2017.64 | 2018.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 2001.50 | 2012.26 | 2015.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 2004.60 | 2000.01 | 2007.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 12:15:00 | 2004.60 | 2000.01 | 2007.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 2004.60 | 2000.01 | 2007.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:45:00 | 2008.90 | 2000.01 | 2007.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 2018.60 | 2003.73 | 2008.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 2020.00 | 2003.73 | 2008.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 2020.10 | 2007.00 | 2009.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 2026.00 | 2007.00 | 2009.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 200 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 2024.00 | 2012.58 | 2011.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 12:15:00 | 2035.90 | 2018.78 | 2014.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 15:15:00 | 2021.40 | 2022.01 | 2017.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-11 09:15:00 | 2007.00 | 2022.01 | 2017.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1998.60 | 2017.32 | 2015.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 1998.60 | 2017.32 | 2015.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 1993.10 | 2012.48 | 2013.63 | EMA200 below EMA400 |

### Cycle 202 — BUY (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 12:15:00 | 2016.20 | 2011.29 | 2010.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 13:15:00 | 2020.80 | 2013.19 | 2011.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 2032.90 | 2035.56 | 2027.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 2032.90 | 2035.56 | 2027.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 2031.20 | 2034.60 | 2028.16 | EMA400 retest candle locked (from upside) |

### Cycle 203 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 2004.20 | 2022.30 | 2023.63 | EMA200 below EMA400 |

### Cycle 204 — BUY (started 2025-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 13:15:00 | 2028.60 | 2022.90 | 2022.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 14:15:00 | 2037.20 | 2025.76 | 2023.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 14:15:00 | 2024.00 | 2029.04 | 2026.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 14:15:00 | 2024.00 | 2029.04 | 2026.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 2024.00 | 2029.04 | 2026.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 2024.00 | 2029.04 | 2026.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 2030.50 | 2029.33 | 2027.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:15:00 | 2019.60 | 2029.33 | 2027.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 2022.40 | 2027.95 | 2026.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 10:30:00 | 2031.10 | 2029.38 | 2027.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 10:15:00 | 2016.60 | 2033.56 | 2035.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 2016.60 | 2033.56 | 2035.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 2002.30 | 2023.20 | 2029.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 2008.00 | 2001.90 | 2012.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 2008.00 | 2001.90 | 2012.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 2008.00 | 2001.90 | 2012.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:45:00 | 2008.50 | 2001.90 | 2012.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 2001.00 | 2001.72 | 2011.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:30:00 | 2018.20 | 2001.72 | 2011.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 2020.50 | 2001.18 | 2006.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 2020.50 | 2001.18 | 2006.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 2006.10 | 2002.17 | 2006.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 11:30:00 | 2004.00 | 2003.33 | 2006.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 12:15:00 | 2002.50 | 2003.33 | 2006.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 09:15:00 | 2004.10 | 2006.07 | 2006.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 15:15:00 | 1990.00 | 1983.26 | 1982.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — BUY (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 15:15:00 | 1990.00 | 1983.26 | 1982.75 | EMA200 above EMA400 |

### Cycle 207 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 1956.40 | 1977.89 | 1980.36 | EMA200 below EMA400 |

### Cycle 208 — BUY (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 14:15:00 | 1987.30 | 1980.39 | 1980.37 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 1955.50 | 1976.61 | 1978.73 | EMA200 below EMA400 |

### Cycle 210 — BUY (started 2025-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 12:15:00 | 1981.50 | 1976.37 | 1976.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 13:15:00 | 1984.20 | 1977.93 | 1977.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 11:15:00 | 1971.00 | 1978.89 | 1978.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 11:15:00 | 1971.00 | 1978.89 | 1978.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 1971.00 | 1978.89 | 1978.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:00:00 | 1971.00 | 1978.89 | 1978.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 1978.70 | 1978.85 | 1978.27 | EMA400 retest candle locked (from upside) |

### Cycle 211 — SELL (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 14:15:00 | 1974.30 | 1977.79 | 1977.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 15:15:00 | 1970.20 | 1976.27 | 1977.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 12:15:00 | 1954.00 | 1953.51 | 1961.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 12:15:00 | 1954.00 | 1953.51 | 1961.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 1954.00 | 1953.51 | 1961.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:15:00 | 1942.10 | 1953.19 | 1960.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 14:15:00 | 1953.50 | 1938.82 | 1938.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 212 — BUY (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 14:15:00 | 1953.50 | 1938.82 | 1938.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 10:15:00 | 1962.50 | 1948.10 | 1943.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 13:15:00 | 1947.80 | 1950.59 | 1945.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 14:00:00 | 1947.80 | 1950.59 | 1945.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 1952.90 | 1951.06 | 1946.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:30:00 | 1944.30 | 1951.06 | 1946.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 1946.70 | 1950.18 | 1946.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:15:00 | 1953.90 | 1950.18 | 1946.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 11:15:00 | 1934.60 | 1945.13 | 1944.93 | SL hit (close<static) qty=1.00 sl=1940.70 alert=retest2 |

### Cycle 213 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 1940.50 | 1944.21 | 1944.52 | EMA200 below EMA400 |

### Cycle 214 — BUY (started 2025-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 12:15:00 | 1953.90 | 1945.98 | 1944.90 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 11:15:00 | 1936.60 | 1944.89 | 1945.75 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2025-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 13:15:00 | 1951.90 | 1946.81 | 1946.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 15:15:00 | 1954.00 | 1948.92 | 1947.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 12:15:00 | 1956.10 | 1956.46 | 1953.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 12:15:00 | 1956.10 | 1956.46 | 1953.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 1956.10 | 1956.46 | 1953.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 12:45:00 | 1955.70 | 1956.46 | 1953.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 1959.40 | 1957.14 | 1954.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:30:00 | 1955.50 | 1957.14 | 1954.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1958.80 | 1957.93 | 1955.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:45:00 | 1951.70 | 1957.93 | 1955.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 1945.00 | 1955.34 | 1954.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 1945.00 | 1955.34 | 1954.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 217 — SELL (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 11:15:00 | 1943.30 | 1952.93 | 1953.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 13:15:00 | 1940.20 | 1948.51 | 1951.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 14:15:00 | 1950.40 | 1948.89 | 1951.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 15:00:00 | 1950.40 | 1948.89 | 1951.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 1953.00 | 1949.71 | 1951.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 1944.90 | 1949.71 | 1951.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1941.80 | 1948.13 | 1950.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:00:00 | 1938.80 | 1946.26 | 1949.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 1933.50 | 1943.99 | 1946.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 11:30:00 | 1937.50 | 1941.80 | 1944.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 12:00:00 | 1938.20 | 1941.80 | 1944.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 1940.30 | 1941.50 | 1944.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:30:00 | 1947.70 | 1941.50 | 1944.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 1950.00 | 1943.20 | 1945.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 1950.00 | 1943.20 | 1945.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1955.10 | 1945.58 | 1945.96 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-30 14:15:00 | 1955.10 | 1945.58 | 1945.96 | SL hit (close>static) qty=1.00 sl=1953.00 alert=retest2 |

### Cycle 218 — BUY (started 2025-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 15:15:00 | 1955.00 | 1947.47 | 1946.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 1979.90 | 1960.98 | 1956.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 11:15:00 | 1970.00 | 1971.43 | 1965.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 11:30:00 | 1969.60 | 1971.43 | 1965.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 1964.60 | 1970.07 | 1965.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 13:00:00 | 1964.60 | 1970.07 | 1965.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 1967.10 | 1969.47 | 1965.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 1967.10 | 1969.47 | 1965.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 1980.70 | 1971.72 | 1967.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 1984.60 | 1973.18 | 1968.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:45:00 | 1984.30 | 1975.22 | 1969.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 10:30:00 | 1988.00 | 1980.72 | 1972.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 15:15:00 | 1967.20 | 1981.28 | 1982.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 219 — SELL (started 2026-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 15:15:00 | 1967.20 | 1981.28 | 1982.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 1946.50 | 1974.32 | 1979.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 11:15:00 | 1896.80 | 1895.94 | 1915.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:00:00 | 1896.80 | 1895.94 | 1915.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1907.00 | 1899.85 | 1912.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 1912.00 | 1899.85 | 1912.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1913.00 | 1902.48 | 1912.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 1907.80 | 1902.48 | 1912.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1905.60 | 1903.10 | 1911.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:30:00 | 1890.90 | 1901.78 | 1910.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 15:15:00 | 1885.00 | 1874.96 | 1874.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 220 — BUY (started 2026-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 15:15:00 | 1885.00 | 1874.96 | 1874.83 | EMA200 above EMA400 |

### Cycle 221 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 1871.70 | 1874.31 | 1874.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 1860.90 | 1869.70 | 1872.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-23 10:15:00 | 1823.90 | 1823.83 | 1834.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-23 11:00:00 | 1823.90 | 1823.83 | 1834.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 1813.40 | 1802.72 | 1812.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 1825.90 | 1802.72 | 1812.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1822.50 | 1806.67 | 1812.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 1823.40 | 1806.67 | 1812.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 1834.00 | 1812.14 | 1814.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 1834.00 | 1812.14 | 1814.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 222 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 1820.00 | 1816.51 | 1816.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 14:15:00 | 1824.90 | 1819.18 | 1818.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 1819.50 | 1820.33 | 1818.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 1819.50 | 1820.33 | 1818.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1819.50 | 1820.33 | 1818.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:45:00 | 1812.50 | 1820.33 | 1818.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 1821.00 | 1820.47 | 1819.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:30:00 | 1818.60 | 1820.47 | 1819.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 1812.00 | 1819.26 | 1818.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:00:00 | 1812.00 | 1819.26 | 1818.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 223 — SELL (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 13:15:00 | 1814.90 | 1818.39 | 1818.43 | EMA200 below EMA400 |

### Cycle 224 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 1831.80 | 1817.83 | 1817.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 13:15:00 | 1841.00 | 1825.81 | 1821.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 1826.00 | 1830.16 | 1825.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 1826.00 | 1830.16 | 1825.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 1826.00 | 1830.16 | 1825.02 | EMA400 retest candle locked (from upside) |

### Cycle 225 — SELL (started 2026-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 13:15:00 | 1816.00 | 1821.42 | 1821.89 | EMA200 below EMA400 |

### Cycle 226 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 1830.50 | 1823.24 | 1822.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 15:15:00 | 1831.00 | 1824.79 | 1823.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 1849.30 | 1852.33 | 1842.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 10:30:00 | 1846.70 | 1852.33 | 1842.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 1843.90 | 1850.64 | 1843.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 12:00:00 | 1843.90 | 1850.64 | 1843.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 1840.40 | 1848.59 | 1842.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 12:30:00 | 1835.50 | 1848.59 | 1842.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 1851.20 | 1849.11 | 1843.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 14:15:00 | 1853.10 | 1849.11 | 1843.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 15:00:00 | 1853.10 | 1849.91 | 1844.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 10:15:00 | 1830.50 | 1845.54 | 1843.84 | SL hit (close<static) qty=1.00 sl=1840.00 alert=retest2 |

### Cycle 227 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 1908.60 | 1936.50 | 1940.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 13:15:00 | 1898.80 | 1924.93 | 1934.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 1939.30 | 1921.44 | 1929.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 1939.30 | 1921.44 | 1929.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 1939.30 | 1921.44 | 1929.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:00:00 | 1939.30 | 1921.44 | 1929.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 1928.30 | 1922.81 | 1929.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:00:00 | 1924.20 | 1923.09 | 1929.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 1955.00 | 1934.14 | 1932.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 228 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 1955.00 | 1934.14 | 1932.55 | EMA200 above EMA400 |

### Cycle 229 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 10:15:00 | 1908.50 | 1929.78 | 1932.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 12:15:00 | 1904.30 | 1920.89 | 1927.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 10:15:00 | 1911.10 | 1907.67 | 1917.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 11:00:00 | 1911.10 | 1907.67 | 1917.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 1871.40 | 1857.97 | 1871.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 1871.40 | 1857.97 | 1871.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 1879.40 | 1862.26 | 1872.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 1868.80 | 1862.26 | 1872.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:45:00 | 1859.70 | 1862.61 | 1871.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 13:15:00 | 1886.50 | 1871.71 | 1873.09 | SL hit (close>static) qty=1.00 sl=1880.90 alert=retest2 |

### Cycle 230 — BUY (started 2026-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 15:15:00 | 1878.50 | 1874.36 | 1874.13 | EMA200 above EMA400 |

### Cycle 231 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 1859.80 | 1871.45 | 1872.83 | EMA200 below EMA400 |

### Cycle 232 — BUY (started 2026-03-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 14:15:00 | 1888.40 | 1872.99 | 1872.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 09:15:00 | 1892.70 | 1878.26 | 1875.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-10 14:15:00 | 1888.80 | 1889.13 | 1882.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-10 15:00:00 | 1888.80 | 1889.13 | 1882.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 1872.40 | 1885.28 | 1881.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:30:00 | 1874.60 | 1885.28 | 1881.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 1884.20 | 1885.07 | 1882.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 11:15:00 | 1892.00 | 1885.07 | 1882.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 15:15:00 | 1870.00 | 1881.78 | 1882.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 233 — SELL (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 15:15:00 | 1870.00 | 1881.78 | 1882.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 1849.20 | 1875.26 | 1879.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 15:15:00 | 1860.00 | 1859.23 | 1867.63 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:15:00 | 1849.20 | 1859.23 | 1867.63 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 11:30:00 | 1852.20 | 1854.61 | 1863.22 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1850.40 | 1834.19 | 1843.82 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-16 14:15:00 | 1850.40 | 1834.19 | 1843.82 | SL hit (close>ema400) qty=1.00 sl=1843.82 alert=retest1 |

### Cycle 234 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 1773.80 | 1757.19 | 1755.97 | EMA200 above EMA400 |

### Cycle 235 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 1740.00 | 1755.74 | 1756.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1699.30 | 1738.42 | 1747.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1729.30 | 1719.36 | 1730.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1729.30 | 1719.36 | 1730.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1729.30 | 1719.36 | 1730.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 1725.70 | 1719.36 | 1730.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:45:00 | 1724.20 | 1720.33 | 1729.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 10:15:00 | 1639.41 | 1686.37 | 1707.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 11:15:00 | 1637.99 | 1675.92 | 1701.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 1690.20 | 1674.58 | 1693.73 | SL hit (close>ema200) qty=0.50 sl=1674.58 alert=retest2 |

### Cycle 236 — BUY (started 2026-04-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 15:15:00 | 1718.80 | 1701.19 | 1699.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 14:15:00 | 1723.70 | 1711.22 | 1705.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 1745.80 | 1748.39 | 1734.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 1745.80 | 1748.39 | 1734.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 1745.80 | 1748.39 | 1734.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 09:30:00 | 1734.10 | 1748.39 | 1734.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1768.30 | 1779.44 | 1768.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 15:15:00 | 1788.00 | 1780.04 | 1772.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 11:15:00 | 1852.20 | 1864.74 | 1865.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 237 — SELL (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 11:15:00 | 1852.20 | 1864.74 | 1865.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 12:15:00 | 1846.90 | 1861.18 | 1863.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 12:15:00 | 1830.90 | 1826.79 | 1836.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-23 13:00:00 | 1830.90 | 1826.79 | 1836.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 13:15:00 | 1814.00 | 1824.23 | 1834.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 13:30:00 | 1818.50 | 1824.23 | 1834.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 1780.00 | 1772.98 | 1781.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:30:00 | 1797.80 | 1778.96 | 1783.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 1797.30 | 1782.63 | 1784.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 11:15:00 | 1794.30 | 1782.63 | 1784.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 13:15:00 | 1772.60 | 1764.38 | 1764.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 238 — BUY (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 13:15:00 | 1772.60 | 1764.38 | 1764.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 14:15:00 | 1775.90 | 1766.69 | 1765.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 11:15:00 | 1835.00 | 1839.63 | 1822.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 12:00:00 | 1835.00 | 1839.63 | 1822.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 1821.00 | 1835.91 | 1822.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:45:00 | 1810.40 | 1835.91 | 1822.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 1822.10 | 1833.15 | 1822.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:45:00 | 1820.70 | 1833.15 | 1822.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 1826.40 | 1831.80 | 1822.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:15:00 | 1820.00 | 1831.80 | 1822.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 1820.00 | 1829.44 | 1822.20 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-18 12:15:00 | 1111.80 | 2023-05-26 10:15:00 | 1089.70 | STOP_HIT | 1.00 | 1.99% |
| BUY | retest2 | 2023-06-01 11:00:00 | 1219.75 | 2023-06-09 11:15:00 | 1229.40 | STOP_HIT | 1.00 | 0.79% |
| SELL | retest2 | 2023-06-14 09:15:00 | 1210.70 | 2023-06-16 09:15:00 | 1231.50 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2023-06-14 09:45:00 | 1208.55 | 2023-06-16 09:15:00 | 1231.50 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2023-06-14 10:30:00 | 1210.40 | 2023-06-16 09:15:00 | 1231.50 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2023-06-15 09:15:00 | 1207.00 | 2023-06-16 09:15:00 | 1231.50 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2023-07-12 14:00:00 | 1350.75 | 2023-07-18 11:15:00 | 1347.50 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2023-07-25 09:15:00 | 1396.50 | 2023-07-25 12:15:00 | 1373.05 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2023-08-11 14:30:00 | 1376.85 | 2023-08-24 12:15:00 | 1338.05 | STOP_HIT | 1.00 | 2.82% |
| SELL | retest2 | 2023-08-29 12:15:00 | 1319.50 | 2023-09-01 10:15:00 | 1339.35 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2023-08-29 12:45:00 | 1315.00 | 2023-09-01 10:15:00 | 1339.35 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2023-08-29 14:45:00 | 1319.25 | 2023-09-01 10:15:00 | 1339.35 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2023-08-30 10:00:00 | 1318.85 | 2023-09-01 10:15:00 | 1339.35 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2023-08-30 14:15:00 | 1321.95 | 2023-09-01 10:15:00 | 1339.35 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2023-08-31 09:15:00 | 1316.95 | 2023-09-01 10:15:00 | 1339.35 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2023-08-31 13:00:00 | 1320.35 | 2023-09-01 10:15:00 | 1339.35 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2023-09-05 13:15:00 | 1352.65 | 2023-09-07 11:15:00 | 1343.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2023-09-06 09:30:00 | 1350.10 | 2023-09-07 11:15:00 | 1343.00 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2023-09-06 14:45:00 | 1347.50 | 2023-09-07 11:15:00 | 1343.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2023-09-07 09:15:00 | 1351.90 | 2023-09-07 11:15:00 | 1343.00 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2023-09-11 10:30:00 | 1360.00 | 2023-09-12 15:15:00 | 1352.00 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2023-09-11 12:30:00 | 1361.00 | 2023-09-12 15:15:00 | 1352.00 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2023-09-11 14:00:00 | 1361.45 | 2023-09-12 15:15:00 | 1352.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2023-09-12 11:15:00 | 1363.35 | 2023-09-12 15:15:00 | 1352.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2023-09-14 15:15:00 | 1374.45 | 2023-09-21 12:15:00 | 1351.00 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2023-09-15 13:15:00 | 1370.00 | 2023-09-21 12:15:00 | 1351.00 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2023-09-15 14:00:00 | 1369.75 | 2023-09-21 12:15:00 | 1351.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2023-09-15 14:30:00 | 1375.00 | 2023-09-21 12:15:00 | 1351.00 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2023-09-20 12:00:00 | 1385.95 | 2023-09-21 12:15:00 | 1351.00 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2023-10-06 10:15:00 | 1316.55 | 2023-10-09 13:15:00 | 1297.35 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2023-10-16 09:15:00 | 1323.65 | 2023-10-26 09:15:00 | 1359.25 | STOP_HIT | 1.00 | 2.69% |
| SELL | retest1 | 2023-10-30 09:15:00 | 1342.75 | 2023-10-31 09:15:00 | 1363.90 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2023-11-02 09:15:00 | 1371.20 | 2023-11-03 12:15:00 | 1362.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2023-11-02 11:15:00 | 1369.95 | 2023-11-03 12:15:00 | 1362.00 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2023-11-02 13:45:00 | 1372.95 | 2023-11-03 12:15:00 | 1362.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2023-11-13 10:15:00 | 1341.70 | 2023-11-15 09:15:00 | 1368.20 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2023-11-21 09:15:00 | 1456.55 | 2023-11-23 11:15:00 | 1442.95 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2023-11-23 09:15:00 | 1471.85 | 2023-11-23 11:15:00 | 1442.95 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2023-11-28 11:00:00 | 1435.20 | 2023-11-28 15:15:00 | 1447.50 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2023-12-04 09:15:00 | 1459.70 | 2023-12-04 09:15:00 | 1451.60 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2023-12-20 13:15:00 | 1440.00 | 2023-12-27 10:15:00 | 1425.15 | STOP_HIT | 1.00 | 1.03% |
| BUY | retest2 | 2024-01-01 10:15:00 | 1429.60 | 2024-01-02 14:15:00 | 1421.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-01-01 13:15:00 | 1432.00 | 2024-01-02 14:15:00 | 1421.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-01-02 10:00:00 | 1430.05 | 2024-01-02 14:15:00 | 1421.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-01-02 10:30:00 | 1430.45 | 2024-01-02 14:15:00 | 1421.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-01-08 09:30:00 | 1384.00 | 2024-01-08 11:15:00 | 1400.15 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-01-12 09:30:00 | 1383.10 | 2024-01-12 15:15:00 | 1395.95 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-01-23 15:00:00 | 1480.05 | 2024-02-02 15:15:00 | 1496.00 | STOP_HIT | 1.00 | 1.08% |
| BUY | retest2 | 2024-01-24 09:15:00 | 1487.90 | 2024-02-02 15:15:00 | 1496.00 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest2 | 2024-02-14 11:45:00 | 1627.15 | 2024-02-15 12:15:00 | 1621.90 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2024-02-20 10:30:00 | 1649.90 | 2024-02-20 11:15:00 | 1632.55 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-02-21 09:15:00 | 1650.80 | 2024-02-22 09:15:00 | 1631.55 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2024-02-21 11:00:00 | 1652.85 | 2024-02-22 09:15:00 | 1631.55 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-02-21 12:45:00 | 1652.30 | 2024-02-22 09:15:00 | 1631.55 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-03-12 09:15:00 | 1662.85 | 2024-03-12 10:15:00 | 1650.05 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-03-19 14:45:00 | 1663.05 | 2024-03-20 09:15:00 | 1634.45 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-03-28 12:00:00 | 1688.70 | 2024-04-04 10:15:00 | 1675.05 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-04-19 11:30:00 | 1734.95 | 2024-04-22 14:15:00 | 1685.95 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2024-04-24 11:15:00 | 1722.00 | 2024-04-26 13:15:00 | 1692.30 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-04-24 12:15:00 | 1722.60 | 2024-04-26 13:15:00 | 1692.30 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest1 | 2024-05-13 09:30:00 | 1656.35 | 2024-05-13 14:15:00 | 1684.40 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest1 | 2024-05-13 12:45:00 | 1658.00 | 2024-05-13 14:15:00 | 1684.40 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-05-23 10:15:00 | 1641.80 | 2024-05-23 14:15:00 | 1662.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-05-29 09:15:00 | 1625.45 | 2024-06-04 10:15:00 | 1544.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-29 10:00:00 | 1628.00 | 2024-06-04 10:15:00 | 1546.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-29 09:15:00 | 1625.45 | 2024-06-05 09:15:00 | 1579.95 | STOP_HIT | 0.50 | 2.80% |
| SELL | retest2 | 2024-05-29 10:00:00 | 1628.00 | 2024-06-05 09:15:00 | 1579.95 | STOP_HIT | 0.50 | 2.95% |
| BUY | retest2 | 2024-06-14 09:15:00 | 1703.00 | 2024-07-05 14:15:00 | 1873.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-15 11:00:00 | 1873.95 | 2024-07-23 12:15:00 | 1874.10 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2024-07-16 11:45:00 | 1887.85 | 2024-07-23 12:15:00 | 1874.10 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-07-16 15:15:00 | 1878.95 | 2024-07-23 12:15:00 | 1874.10 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2024-07-18 12:00:00 | 1876.25 | 2024-07-23 12:15:00 | 1874.10 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2024-07-25 12:00:00 | 1918.30 | 2024-08-02 15:15:00 | 1984.35 | STOP_HIT | 1.00 | 3.44% |
| SELL | retest2 | 2024-08-06 10:15:00 | 1943.60 | 2024-08-08 09:15:00 | 1967.95 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-08-06 11:45:00 | 1945.85 | 2024-08-08 09:15:00 | 1967.95 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-08-06 12:45:00 | 1943.65 | 2024-08-08 11:15:00 | 1969.40 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-08-06 13:30:00 | 1943.85 | 2024-08-08 11:15:00 | 1969.40 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-08-07 12:00:00 | 1934.00 | 2024-08-08 11:15:00 | 1969.40 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-08-07 15:00:00 | 1934.95 | 2024-08-08 11:15:00 | 1969.40 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2024-08-19 15:15:00 | 2019.90 | 2024-09-04 10:15:00 | 2221.89 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2024-09-13 11:30:00 | 2116.40 | 2024-09-18 11:15:00 | 2128.80 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2024-09-18 09:15:00 | 2100.55 | 2024-09-18 11:15:00 | 2128.80 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-09-25 11:15:00 | 2251.00 | 2024-09-27 10:15:00 | 2220.20 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-09-25 12:00:00 | 2243.00 | 2024-09-27 10:15:00 | 2220.20 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-09-25 13:00:00 | 2245.40 | 2024-09-27 10:15:00 | 2220.20 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-09-30 09:15:00 | 2215.70 | 2024-10-07 10:15:00 | 2104.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 09:15:00 | 2215.70 | 2024-10-08 14:15:00 | 2100.85 | STOP_HIT | 0.50 | 5.18% |
| BUY | retest2 | 2024-10-16 14:00:00 | 2112.00 | 2024-10-16 14:15:00 | 2076.80 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2024-10-21 11:15:00 | 2017.00 | 2024-10-25 14:15:00 | 1919.90 | PARTIAL | 0.50 | 4.81% |
| SELL | retest2 | 2024-10-21 12:00:00 | 2017.55 | 2024-10-25 14:15:00 | 1917.57 | PARTIAL | 0.50 | 4.96% |
| SELL | retest2 | 2024-10-21 15:15:00 | 2020.95 | 2024-10-28 09:15:00 | 1916.15 | PARTIAL | 0.50 | 5.19% |
| SELL | retest2 | 2024-10-22 10:15:00 | 2018.50 | 2024-10-28 09:15:00 | 1916.67 | PARTIAL | 0.50 | 5.04% |
| SELL | retest2 | 2024-10-21 11:15:00 | 2017.00 | 2024-10-29 12:15:00 | 1905.60 | STOP_HIT | 0.50 | 5.52% |
| SELL | retest2 | 2024-10-21 12:00:00 | 2017.55 | 2024-10-29 12:15:00 | 1905.60 | STOP_HIT | 0.50 | 5.55% |
| SELL | retest2 | 2024-10-21 15:15:00 | 2020.95 | 2024-10-29 12:15:00 | 1905.60 | STOP_HIT | 0.50 | 5.71% |
| SELL | retest2 | 2024-10-22 10:15:00 | 2018.50 | 2024-10-29 12:15:00 | 1905.60 | STOP_HIT | 0.50 | 5.59% |
| BUY | retest2 | 2024-11-07 14:00:00 | 1921.80 | 2024-11-08 11:15:00 | 1904.50 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-11-07 14:30:00 | 1924.60 | 2024-11-08 11:15:00 | 1904.50 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-11-07 15:15:00 | 1921.00 | 2024-11-08 11:15:00 | 1904.50 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-11-14 11:15:00 | 1869.05 | 2024-11-25 13:15:00 | 1839.95 | STOP_HIT | 1.00 | 1.56% |
| SELL | retest2 | 2024-11-14 12:45:00 | 1868.55 | 2024-11-25 13:15:00 | 1839.95 | STOP_HIT | 1.00 | 1.53% |
| SELL | retest2 | 2024-11-14 13:30:00 | 1867.00 | 2024-11-25 13:15:00 | 1839.95 | STOP_HIT | 1.00 | 1.45% |
| BUY | retest2 | 2024-11-27 12:15:00 | 1846.00 | 2024-12-02 10:15:00 | 1833.55 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-12-16 12:30:00 | 1971.60 | 2024-12-17 09:15:00 | 1940.95 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-12-23 09:15:00 | 1860.45 | 2024-12-30 13:15:00 | 1792.27 | PARTIAL | 0.50 | 3.66% |
| SELL | retest2 | 2024-12-23 09:15:00 | 1860.45 | 2024-12-30 15:15:00 | 1825.00 | STOP_HIT | 0.50 | 1.91% |
| SELL | retest2 | 2024-12-23 10:15:00 | 1886.60 | 2024-12-31 15:15:00 | 1767.43 | PARTIAL | 0.50 | 6.32% |
| SELL | retest2 | 2024-12-23 10:15:00 | 1886.60 | 2025-01-01 11:15:00 | 1794.35 | STOP_HIT | 0.50 | 4.89% |
| BUY | retest2 | 2025-01-03 13:45:00 | 1830.10 | 2025-01-06 15:15:00 | 1808.40 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-01-06 10:00:00 | 1830.30 | 2025-01-06 15:15:00 | 1808.40 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-01-06 12:45:00 | 1832.30 | 2025-01-06 15:15:00 | 1808.40 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-01-21 09:15:00 | 1939.95 | 2025-01-21 14:15:00 | 1902.45 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-01-21 10:00:00 | 1933.50 | 2025-01-21 14:15:00 | 1902.45 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-01-29 09:15:00 | 1761.55 | 2025-01-29 11:15:00 | 1801.75 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2025-02-01 14:45:00 | 1868.40 | 2025-02-05 13:15:00 | 1863.90 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-02-03 10:15:00 | 1870.85 | 2025-02-05 14:15:00 | 1856.65 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-02-03 13:30:00 | 1874.80 | 2025-02-05 14:15:00 | 1856.65 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-02-04 13:30:00 | 1866.75 | 2025-02-05 14:15:00 | 1856.65 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-02-05 10:45:00 | 1883.65 | 2025-02-05 14:15:00 | 1856.65 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-02-11 09:45:00 | 1806.45 | 2025-02-17 09:15:00 | 1716.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 09:45:00 | 1806.45 | 2025-02-17 14:15:00 | 1729.70 | STOP_HIT | 0.50 | 4.25% |
| BUY | retest2 | 2025-03-07 13:15:00 | 1682.35 | 2025-03-10 11:15:00 | 1676.60 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-03-10 09:30:00 | 1680.65 | 2025-03-10 11:15:00 | 1676.60 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-03-11 14:15:00 | 1702.75 | 2025-03-12 11:15:00 | 1673.00 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-03-27 09:15:00 | 1780.80 | 2025-04-07 09:15:00 | 1754.80 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-03-27 14:15:00 | 1781.20 | 2025-04-07 09:15:00 | 1754.80 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-03-27 14:45:00 | 1784.45 | 2025-04-07 09:15:00 | 1754.80 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-04-01 09:30:00 | 1800.95 | 2025-04-07 09:15:00 | 1754.80 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2025-04-01 15:15:00 | 1796.00 | 2025-04-07 09:15:00 | 1754.80 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-04-02 09:30:00 | 1797.05 | 2025-04-07 09:15:00 | 1754.80 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-04-21 09:30:00 | 1807.80 | 2025-04-25 14:15:00 | 1852.10 | STOP_HIT | 1.00 | 2.45% |
| BUY | retest2 | 2025-04-21 10:00:00 | 1806.30 | 2025-04-25 14:15:00 | 1852.10 | STOP_HIT | 1.00 | 2.54% |
| SELL | retest2 | 2025-05-08 10:30:00 | 1809.00 | 2025-05-12 11:15:00 | 1828.40 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-05-15 09:15:00 | 1874.80 | 2025-05-22 09:15:00 | 1868.60 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-05-16 15:15:00 | 1857.90 | 2025-05-22 09:15:00 | 1868.60 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2025-05-19 10:00:00 | 1856.20 | 2025-05-22 09:15:00 | 1868.60 | STOP_HIT | 1.00 | 0.67% |
| BUY | retest2 | 2025-06-03 09:15:00 | 1871.30 | 2025-06-03 13:15:00 | 1856.60 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-06-06 10:15:00 | 1899.80 | 2025-06-11 13:15:00 | 1944.00 | STOP_HIT | 1.00 | 2.33% |
| SELL | retest2 | 2025-06-16 09:15:00 | 1933.00 | 2025-06-16 13:15:00 | 1943.30 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-06-16 10:30:00 | 1931.30 | 2025-06-16 13:15:00 | 1943.30 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-06-19 11:15:00 | 1928.00 | 2025-06-20 09:15:00 | 1942.80 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-06-19 15:00:00 | 1928.00 | 2025-06-20 09:15:00 | 1942.80 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-07-02 12:45:00 | 2042.60 | 2025-07-03 14:15:00 | 2033.50 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-07-02 14:30:00 | 2042.60 | 2025-07-03 14:15:00 | 2033.50 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-07-02 15:00:00 | 2042.60 | 2025-07-03 14:15:00 | 2033.50 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-07-03 09:45:00 | 2042.80 | 2025-07-03 14:15:00 | 2033.50 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-07-09 09:15:00 | 2030.70 | 2025-07-14 10:15:00 | 2024.30 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2025-07-28 12:45:00 | 1912.30 | 2025-07-30 15:15:00 | 1923.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-07-29 13:30:00 | 1915.00 | 2025-07-30 15:15:00 | 1923.00 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-07-29 15:15:00 | 1915.20 | 2025-07-30 15:15:00 | 1923.00 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-08-05 09:15:00 | 1878.80 | 2025-08-05 12:15:00 | 1901.10 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-08-08 13:15:00 | 1899.60 | 2025-08-14 14:15:00 | 1898.40 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2025-08-08 14:30:00 | 1901.20 | 2025-08-14 14:15:00 | 1898.40 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2025-08-11 09:15:00 | 1899.70 | 2025-08-14 14:15:00 | 1898.40 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-08-11 10:00:00 | 1897.90 | 2025-08-14 14:15:00 | 1898.40 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2025-08-13 11:00:00 | 1918.50 | 2025-08-14 14:15:00 | 1898.40 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-08-14 09:15:00 | 1916.60 | 2025-08-14 14:15:00 | 1898.40 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-08-19 11:30:00 | 1942.40 | 2025-08-22 11:15:00 | 1937.10 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-08-22 11:00:00 | 1937.80 | 2025-08-22 11:15:00 | 1937.10 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-09-02 11:00:00 | 1819.30 | 2025-09-04 09:15:00 | 1873.90 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-09-02 14:15:00 | 1819.40 | 2025-09-04 09:15:00 | 1873.90 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-09-02 15:15:00 | 1820.00 | 2025-09-04 09:15:00 | 1873.90 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2025-09-11 11:45:00 | 1860.60 | 2025-09-17 13:15:00 | 1868.00 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2025-10-10 12:15:00 | 1885.20 | 2025-10-15 09:15:00 | 2000.00 | STOP_HIT | 1.00 | -6.09% |
| BUY | retest2 | 2025-11-19 10:30:00 | 2031.10 | 2025-11-21 10:15:00 | 2016.60 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-11-26 11:30:00 | 2004.00 | 2025-12-01 15:15:00 | 1990.00 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2025-11-26 12:15:00 | 2002.50 | 2025-12-01 15:15:00 | 1990.00 | STOP_HIT | 1.00 | 0.62% |
| SELL | retest2 | 2025-11-27 09:15:00 | 2004.10 | 2025-12-01 15:15:00 | 1990.00 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2025-12-09 14:15:00 | 1942.10 | 2025-12-15 14:15:00 | 1953.50 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-12-17 09:15:00 | 1953.90 | 2025-12-17 11:15:00 | 1934.60 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-12-29 11:00:00 | 1938.80 | 2025-12-30 14:15:00 | 1955.10 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-12-30 09:15:00 | 1933.50 | 2025-12-30 14:15:00 | 1955.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-12-30 11:30:00 | 1937.50 | 2025-12-30 14:15:00 | 1955.10 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-12-30 12:00:00 | 1938.20 | 2025-12-30 14:15:00 | 1955.10 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2026-01-06 09:15:00 | 1984.60 | 2026-01-07 15:15:00 | 1967.20 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2026-01-06 09:45:00 | 1984.30 | 2026-01-07 15:15:00 | 1967.20 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-01-06 10:30:00 | 1988.00 | 2026-01-07 15:15:00 | 1967.20 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2026-01-13 10:30:00 | 1890.90 | 2026-01-19 15:15:00 | 1885.00 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2026-02-04 14:15:00 | 1853.10 | 2026-02-05 10:15:00 | 1830.50 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2026-02-04 15:00:00 | 1853.10 | 2026-02-05 10:15:00 | 1830.50 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2026-02-05 13:15:00 | 1865.20 | 2026-02-24 11:15:00 | 1908.60 | STOP_HIT | 1.00 | 2.33% |
| BUY | retest2 | 2026-02-06 11:00:00 | 1853.20 | 2026-02-24 11:15:00 | 1908.60 | STOP_HIT | 1.00 | 2.99% |
| BUY | retest2 | 2026-02-09 12:00:00 | 1874.70 | 2026-02-24 11:15:00 | 1908.60 | STOP_HIT | 1.00 | 1.81% |
| BUY | retest2 | 2026-02-10 09:45:00 | 1874.10 | 2026-02-24 11:15:00 | 1908.60 | STOP_HIT | 1.00 | 1.84% |
| SELL | retest2 | 2026-02-25 12:00:00 | 1924.20 | 2026-02-26 09:15:00 | 1955.00 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2026-03-06 09:15:00 | 1868.80 | 2026-03-06 13:15:00 | 1886.50 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-03-06 09:45:00 | 1859.70 | 2026-03-06 13:15:00 | 1886.50 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2026-03-11 11:15:00 | 1892.00 | 2026-03-11 15:15:00 | 1870.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest1 | 2026-03-13 09:15:00 | 1849.20 | 2026-03-16 14:15:00 | 1850.40 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest1 | 2026-03-13 11:30:00 | 1852.20 | 2026-03-16 14:15:00 | 1850.40 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2026-03-18 09:30:00 | 1824.90 | 2026-03-23 09:15:00 | 1733.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 14:45:00 | 1830.40 | 2026-03-23 09:15:00 | 1738.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 09:30:00 | 1824.90 | 2026-03-24 11:15:00 | 1731.10 | STOP_HIT | 0.50 | 5.14% |
| SELL | retest2 | 2026-03-18 14:45:00 | 1830.40 | 2026-03-24 11:15:00 | 1731.10 | STOP_HIT | 0.50 | 5.43% |
| SELL | retest2 | 2026-04-01 10:15:00 | 1725.70 | 2026-04-02 10:15:00 | 1639.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 10:45:00 | 1724.20 | 2026-04-02 11:15:00 | 1637.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 10:15:00 | 1725.70 | 2026-04-02 14:15:00 | 1690.20 | STOP_HIT | 0.50 | 2.06% |
| SELL | retest2 | 2026-04-01 10:45:00 | 1724.20 | 2026-04-02 14:15:00 | 1690.20 | STOP_HIT | 0.50 | 1.97% |
| BUY | retest2 | 2026-04-13 15:15:00 | 1788.00 | 2026-04-21 11:15:00 | 1852.20 | STOP_HIT | 1.00 | 3.59% |
| SELL | retest2 | 2026-04-29 11:15:00 | 1794.30 | 2026-05-05 13:15:00 | 1772.60 | STOP_HIT | 1.00 | 1.21% |
