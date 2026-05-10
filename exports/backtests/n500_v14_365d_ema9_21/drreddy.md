# Dr. Reddy's Laboratories Ltd. (DRREDDY)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 1294.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 78 |
| ALERT1 | 52 |
| ALERT2 | 51 |
| ALERT2_SKIP | 30 |
| ALERT3 | 148 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 84 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 91 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 91 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 71
- **Target hits / Stop hits / Partials:** 0 / 88 / 3
- **Avg / median % per leg:** -0.52% / -0.78%
- **Sum % (uncompounded):** -47.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 59 | 14 | 23.7% | 0 | 59 | 0 | -0.62% | -36.7% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.95% | -2.9% |
| BUY @ 3rd Alert (retest2) | 56 | 14 | 25.0% | 0 | 56 | 0 | -0.60% | -33.9% |
| SELL (all) | 32 | 6 | 18.8% | 0 | 29 | 3 | -0.33% | -10.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.45% | -0.4% |
| SELL @ 3rd Alert (retest2) | 31 | 6 | 19.4% | 0 | 28 | 3 | -0.33% | -10.1% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.83% | -3.3% |
| retest2 (combined) | 87 | 20 | 23.0% | 0 | 84 | 3 | -0.51% | -44.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 1183.20 | 1159.05 | 1157.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 1188.50 | 1168.85 | 1162.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 1205.40 | 1213.70 | 1204.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 09:15:00 | 1205.40 | 1213.70 | 1204.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 1205.40 | 1213.70 | 1204.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:45:00 | 1206.50 | 1213.70 | 1204.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 1209.10 | 1212.78 | 1205.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:00:00 | 1226.00 | 1215.12 | 1207.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 12:15:00 | 1217.30 | 1225.81 | 1221.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 09:15:00 | 1222.30 | 1221.20 | 1220.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 1214.10 | 1224.24 | 1224.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 1214.10 | 1224.24 | 1224.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 1214.10 | 1224.24 | 1224.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 1214.10 | 1224.24 | 1224.95 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 1231.20 | 1224.93 | 1224.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 1240.40 | 1230.60 | 1228.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 15:15:00 | 1242.00 | 1242.56 | 1238.70 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 09:15:00 | 1247.40 | 1242.56 | 1238.70 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 10:30:00 | 1246.40 | 1243.68 | 1239.92 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 1242.50 | 1243.44 | 1240.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:00:00 | 1242.50 | 1243.44 | 1240.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 1238.70 | 1242.49 | 1240.02 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-28 12:15:00 | 1238.70 | 1242.49 | 1240.02 | SL hit (close<ema400) qty=1.00 sl=1240.02 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-05-28 12:15:00 | 1238.70 | 1242.49 | 1240.02 | SL hit (close<ema400) qty=1.00 sl=1240.02 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-05-28 13:00:00 | 1238.70 | 1242.49 | 1240.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 1240.30 | 1242.05 | 1240.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:30:00 | 1238.90 | 1242.05 | 1240.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 1243.30 | 1242.30 | 1240.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 1244.80 | 1241.64 | 1240.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 10:45:00 | 1247.80 | 1243.94 | 1241.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 10:30:00 | 1245.70 | 1246.90 | 1244.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 11:00:00 | 1248.30 | 1246.90 | 1244.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 1248.70 | 1247.75 | 1245.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:45:00 | 1248.60 | 1247.75 | 1245.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 1251.00 | 1248.88 | 1246.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 1240.70 | 1248.88 | 1246.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 1242.60 | 1247.63 | 1246.22 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-03 10:15:00 | 1245.70 | 1245.82 | 1245.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-03 10:15:00 | 1245.70 | 1245.82 | 1245.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-03 10:15:00 | 1245.70 | 1245.82 | 1245.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-03 10:15:00 | 1245.70 | 1245.82 | 1245.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 10:15:00 | 1245.70 | 1245.82 | 1245.83 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 11:15:00 | 1249.10 | 1246.48 | 1246.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 09:15:00 | 1256.50 | 1249.14 | 1247.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 14:15:00 | 1251.70 | 1252.64 | 1250.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 14:15:00 | 1251.70 | 1252.64 | 1250.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 1251.70 | 1252.64 | 1250.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:45:00 | 1251.50 | 1252.64 | 1250.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 1251.00 | 1252.31 | 1250.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 1288.60 | 1252.31 | 1250.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 14:15:00 | 1347.00 | 1351.62 | 1351.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 14:15:00 | 1347.00 | 1351.62 | 1351.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 09:15:00 | 1334.70 | 1347.98 | 1350.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 1318.60 | 1318.44 | 1327.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-19 10:00:00 | 1318.60 | 1318.44 | 1327.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 1329.90 | 1321.30 | 1327.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 1329.90 | 1321.30 | 1327.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 1326.90 | 1322.42 | 1327.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 14:30:00 | 1322.80 | 1324.11 | 1327.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 15:15:00 | 1331.90 | 1325.67 | 1327.72 | SL hit (close>static) qty=1.00 sl=1331.60 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 10:45:00 | 1323.00 | 1325.94 | 1327.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 11:15:00 | 1321.90 | 1325.94 | 1327.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 1320.70 | 1325.57 | 1327.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 1329.80 | 1325.91 | 1327.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:00:00 | 1329.80 | 1325.91 | 1327.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 1323.10 | 1325.35 | 1326.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 1312.50 | 1324.84 | 1326.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 1335.80 | 1323.03 | 1323.48 | SL hit (close>static) qty=1.00 sl=1331.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 1335.80 | 1323.03 | 1323.48 | SL hit (close>static) qty=1.00 sl=1331.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 1335.80 | 1323.03 | 1323.48 | SL hit (close>static) qty=1.00 sl=1331.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 1335.80 | 1323.03 | 1323.48 | SL hit (close>static) qty=1.00 sl=1330.60 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 1338.30 | 1326.08 | 1324.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 11:15:00 | 1344.30 | 1329.73 | 1326.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 1327.50 | 1337.10 | 1334.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 09:15:00 | 1327.50 | 1337.10 | 1334.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 1327.50 | 1337.10 | 1334.82 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 1316.60 | 1330.04 | 1331.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 09:15:00 | 1312.50 | 1322.25 | 1326.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 15:15:00 | 1282.00 | 1281.25 | 1289.98 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 09:15:00 | 1275.50 | 1281.25 | 1289.98 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 1281.20 | 1275.51 | 1280.41 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 1281.20 | 1275.51 | 1280.41 | SL hit (close>ema400) qty=1.00 sl=1280.41 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-07-03 11:00:00 | 1281.20 | 1275.51 | 1280.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 1284.90 | 1277.39 | 1280.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:45:00 | 1285.00 | 1277.39 | 1280.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 14:15:00 | 1288.70 | 1282.80 | 1282.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 15:15:00 | 1295.00 | 1285.24 | 1283.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 09:15:00 | 1287.80 | 1301.87 | 1297.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 09:15:00 | 1287.80 | 1301.87 | 1297.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1287.80 | 1301.87 | 1297.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:15:00 | 1285.50 | 1301.87 | 1297.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 1277.30 | 1296.96 | 1296.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:00:00 | 1277.30 | 1296.96 | 1296.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 11:15:00 | 1279.50 | 1293.47 | 1294.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 12:15:00 | 1271.40 | 1279.49 | 1285.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 12:15:00 | 1269.00 | 1268.14 | 1275.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 12:45:00 | 1269.80 | 1268.14 | 1275.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 1256.30 | 1252.27 | 1257.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:45:00 | 1256.50 | 1252.27 | 1257.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 1256.50 | 1253.12 | 1257.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:30:00 | 1256.40 | 1253.12 | 1257.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 1260.50 | 1255.14 | 1257.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 15:00:00 | 1260.50 | 1255.14 | 1257.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 1257.90 | 1255.69 | 1257.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 1256.40 | 1255.69 | 1257.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 13:45:00 | 1257.20 | 1256.46 | 1257.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 15:15:00 | 1259.50 | 1257.57 | 1257.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-16 15:15:00 | 1259.50 | 1257.57 | 1257.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 15:15:00 | 1259.50 | 1257.57 | 1257.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 1260.00 | 1258.06 | 1257.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 1261.30 | 1263.26 | 1261.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 09:15:00 | 1261.30 | 1263.26 | 1261.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 1261.30 | 1263.26 | 1261.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:45:00 | 1259.90 | 1263.26 | 1261.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 1251.90 | 1260.99 | 1260.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:00:00 | 1251.90 | 1260.99 | 1260.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 1257.30 | 1260.25 | 1260.10 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 12:15:00 | 1257.10 | 1259.62 | 1259.83 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 14:15:00 | 1261.20 | 1260.05 | 1259.99 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 1258.30 | 1259.70 | 1259.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 09:15:00 | 1252.80 | 1258.32 | 1259.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 11:15:00 | 1257.10 | 1257.02 | 1258.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 12:00:00 | 1257.10 | 1257.02 | 1258.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 1257.70 | 1256.99 | 1258.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:30:00 | 1257.50 | 1256.99 | 1258.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 1259.50 | 1257.49 | 1258.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:00:00 | 1259.50 | 1257.49 | 1258.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 1256.10 | 1257.21 | 1258.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 1251.60 | 1257.21 | 1258.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1247.50 | 1255.27 | 1257.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:30:00 | 1244.90 | 1253.36 | 1256.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 11:15:00 | 1242.50 | 1253.36 | 1256.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 13:45:00 | 1242.70 | 1247.68 | 1252.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 1276.90 | 1249.86 | 1248.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 1276.90 | 1249.86 | 1248.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 1276.90 | 1249.86 | 1248.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 09:15:00 | 1276.90 | 1249.86 | 1248.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 09:15:00 | 1288.60 | 1282.52 | 1276.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 09:15:00 | 1278.70 | 1289.31 | 1283.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 09:15:00 | 1278.70 | 1289.31 | 1283.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1278.70 | 1289.31 | 1283.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:00:00 | 1278.70 | 1289.31 | 1283.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 1287.90 | 1289.03 | 1284.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 11:15:00 | 1292.00 | 1289.03 | 1284.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 14:45:00 | 1289.50 | 1288.61 | 1285.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 1266.50 | 1284.73 | 1284.21 | SL hit (close<static) qty=1.00 sl=1278.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 1266.50 | 1284.73 | 1284.21 | SL hit (close<static) qty=1.00 sl=1278.30 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 1277.40 | 1283.26 | 1283.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 1243.40 | 1270.12 | 1276.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 1202.00 | 1194.62 | 1203.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 1202.00 | 1194.62 | 1203.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1198.00 | 1195.29 | 1202.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:45:00 | 1197.60 | 1196.99 | 1202.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 14:15:00 | 1210.70 | 1203.80 | 1204.15 | SL hit (close>static) qty=1.00 sl=1210.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 15:15:00 | 1215.00 | 1206.04 | 1205.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 09:15:00 | 1224.60 | 1209.75 | 1206.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 09:15:00 | 1212.40 | 1217.25 | 1213.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 1212.40 | 1217.25 | 1213.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 1212.40 | 1217.25 | 1213.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 10:30:00 | 1215.80 | 1216.52 | 1213.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 11:15:00 | 1215.60 | 1216.52 | 1213.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 09:15:00 | 1241.40 | 1248.61 | 1249.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-20 09:15:00 | 1241.40 | 1248.61 | 1249.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 09:15:00 | 1241.40 | 1248.61 | 1249.01 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 1261.00 | 1250.06 | 1249.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 11:15:00 | 1269.70 | 1253.99 | 1251.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 15:15:00 | 1275.00 | 1275.70 | 1268.32 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 09:30:00 | 1284.10 | 1277.66 | 1269.88 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1264.20 | 1278.89 | 1275.20 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 1264.20 | 1278.89 | 1275.20 | SL hit (close<ema400) qty=1.00 sl=1275.20 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 1261.80 | 1278.89 | 1275.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 1268.10 | 1276.73 | 1274.55 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 12:15:00 | 1264.90 | 1272.18 | 1272.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 15:15:00 | 1261.10 | 1267.34 | 1270.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 14:15:00 | 1260.10 | 1259.42 | 1263.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 15:00:00 | 1260.10 | 1259.42 | 1263.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 1247.50 | 1257.74 | 1262.47 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 1274.40 | 1263.63 | 1262.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 12:15:00 | 1280.00 | 1268.08 | 1265.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 09:15:00 | 1269.90 | 1272.87 | 1268.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 1269.90 | 1272.87 | 1268.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1269.90 | 1272.87 | 1268.80 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 14:15:00 | 1256.70 | 1266.62 | 1267.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 15:15:00 | 1250.00 | 1263.30 | 1265.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 11:15:00 | 1262.10 | 1261.63 | 1264.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 11:15:00 | 1262.10 | 1261.63 | 1264.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 1262.10 | 1261.63 | 1264.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:30:00 | 1264.70 | 1261.63 | 1264.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 1264.20 | 1262.14 | 1264.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:00:00 | 1264.20 | 1262.14 | 1264.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 1262.50 | 1262.21 | 1263.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 14:45:00 | 1259.10 | 1262.17 | 1263.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 11:15:00 | 1266.90 | 1263.84 | 1264.11 | SL hit (close>static) qty=1.00 sl=1265.70 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 12:15:00 | 1269.90 | 1265.05 | 1264.64 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 1258.60 | 1263.76 | 1264.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 1252.80 | 1261.57 | 1263.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 1266.50 | 1261.38 | 1262.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 1266.50 | 1261.38 | 1262.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1266.50 | 1261.38 | 1262.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 1266.50 | 1261.38 | 1262.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 1266.80 | 1262.46 | 1263.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:00:00 | 1266.80 | 1262.46 | 1263.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 1260.10 | 1261.99 | 1262.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:30:00 | 1265.00 | 1261.99 | 1262.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 1262.70 | 1262.13 | 1262.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:00:00 | 1262.70 | 1262.13 | 1262.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 1265.00 | 1262.70 | 1262.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 1265.00 | 1262.70 | 1262.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 14:15:00 | 1267.90 | 1263.74 | 1263.41 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 11:15:00 | 1256.10 | 1262.11 | 1262.88 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 10:15:00 | 1276.80 | 1263.77 | 1262.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 13:15:00 | 1282.80 | 1272.12 | 1267.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 1288.90 | 1296.46 | 1287.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 1288.90 | 1296.46 | 1287.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 1288.90 | 1296.46 | 1287.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:45:00 | 1290.40 | 1296.46 | 1287.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1294.50 | 1296.07 | 1288.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:30:00 | 1289.40 | 1296.07 | 1288.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1299.80 | 1308.93 | 1302.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:30:00 | 1315.00 | 1308.64 | 1305.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 12:30:00 | 1312.30 | 1309.82 | 1307.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 09:15:00 | 1312.10 | 1310.58 | 1308.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 10:45:00 | 1312.20 | 1310.84 | 1308.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1316.10 | 1316.73 | 1312.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:00:00 | 1316.10 | 1316.73 | 1312.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 1312.00 | 1318.48 | 1316.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:45:00 | 1313.50 | 1318.48 | 1316.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 1313.40 | 1317.47 | 1315.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:00:00 | 1313.40 | 1317.47 | 1315.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 1311.40 | 1315.73 | 1315.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:30:00 | 1313.00 | 1315.73 | 1315.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-22 13:15:00 | 1303.40 | 1313.26 | 1314.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 13:15:00 | 1303.40 | 1313.26 | 1314.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 13:15:00 | 1303.40 | 1313.26 | 1314.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 13:15:00 | 1303.40 | 1313.26 | 1314.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 13:15:00 | 1303.40 | 1313.26 | 1314.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 1302.00 | 1311.01 | 1313.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 11:15:00 | 1308.70 | 1307.51 | 1310.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 12:00:00 | 1308.70 | 1307.51 | 1310.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 1307.00 | 1307.45 | 1309.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:00:00 | 1307.00 | 1307.45 | 1309.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 1308.10 | 1307.58 | 1309.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:30:00 | 1311.00 | 1307.58 | 1309.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 1305.30 | 1307.12 | 1309.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 1301.90 | 1307.12 | 1309.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 12:00:00 | 1300.00 | 1304.86 | 1307.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 13:15:00 | 1236.81 | 1250.23 | 1263.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 13:15:00 | 1235.00 | 1250.23 | 1263.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 1232.90 | 1231.89 | 1243.42 | SL hit (close>ema200) qty=0.50 sl=1231.89 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 1232.90 | 1231.89 | 1243.42 | SL hit (close>ema200) qty=0.50 sl=1231.89 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 12:15:00 | 1247.90 | 1244.87 | 1244.54 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 13:15:00 | 1240.20 | 1243.93 | 1244.15 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 1247.80 | 1244.71 | 1244.48 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 09:15:00 | 1239.30 | 1244.10 | 1244.27 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 11:15:00 | 1245.50 | 1244.53 | 1244.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 15:15:00 | 1249.80 | 1246.81 | 1245.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 12:15:00 | 1248.90 | 1249.69 | 1247.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 12:15:00 | 1248.90 | 1249.69 | 1247.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 1248.90 | 1249.69 | 1247.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:45:00 | 1249.00 | 1249.69 | 1247.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 1248.90 | 1249.53 | 1247.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:30:00 | 1247.70 | 1249.53 | 1247.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 1247.50 | 1249.13 | 1247.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:45:00 | 1248.10 | 1249.13 | 1247.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 1248.00 | 1248.90 | 1247.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:15:00 | 1242.70 | 1248.90 | 1247.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1244.00 | 1247.92 | 1247.40 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 1239.10 | 1246.16 | 1246.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 11:15:00 | 1235.10 | 1243.95 | 1245.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 1259.20 | 1243.09 | 1244.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 1259.20 | 1243.09 | 1244.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1259.20 | 1243.09 | 1244.03 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 1259.90 | 1246.45 | 1245.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 11:15:00 | 1266.70 | 1254.09 | 1249.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 1251.50 | 1257.73 | 1253.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 1251.50 | 1257.73 | 1253.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1251.50 | 1257.73 | 1253.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:45:00 | 1253.30 | 1257.73 | 1253.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 1250.10 | 1256.20 | 1253.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 1250.10 | 1256.20 | 1253.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 1254.30 | 1255.82 | 1253.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 14:15:00 | 1258.90 | 1255.15 | 1253.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 10:15:00 | 1240.80 | 1252.94 | 1253.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 1240.80 | 1252.94 | 1253.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 13:15:00 | 1239.50 | 1247.10 | 1250.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 11:15:00 | 1236.30 | 1235.91 | 1240.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-16 11:30:00 | 1236.10 | 1235.91 | 1240.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 1238.70 | 1236.47 | 1240.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:00:00 | 1238.70 | 1236.47 | 1240.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 1237.50 | 1236.67 | 1239.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:30:00 | 1240.00 | 1236.67 | 1239.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 1241.50 | 1237.64 | 1240.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 15:00:00 | 1241.50 | 1237.64 | 1240.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 1240.00 | 1238.11 | 1240.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 1241.80 | 1238.11 | 1240.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 1242.10 | 1238.91 | 1240.19 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 11:15:00 | 1249.90 | 1242.24 | 1241.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 12:15:00 | 1250.10 | 1243.81 | 1242.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 09:15:00 | 1276.10 | 1279.32 | 1268.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 10:00:00 | 1276.10 | 1279.32 | 1268.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1278.70 | 1279.68 | 1273.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 12:00:00 | 1288.60 | 1283.03 | 1278.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 12:30:00 | 1289.20 | 1285.05 | 1280.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 13:45:00 | 1288.20 | 1285.80 | 1281.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 15:00:00 | 1290.10 | 1284.34 | 1282.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1289.60 | 1286.92 | 1283.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:30:00 | 1287.00 | 1286.92 | 1283.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 1286.30 | 1286.73 | 1284.32 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-29 13:15:00 | 1252.00 | 1279.81 | 1281.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 13:15:00 | 1252.00 | 1279.81 | 1281.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 13:15:00 | 1252.00 | 1279.81 | 1281.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 13:15:00 | 1252.00 | 1279.81 | 1281.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 13:15:00 | 1252.00 | 1279.81 | 1281.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 09:15:00 | 1194.10 | 1256.03 | 1269.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 13:15:00 | 1199.80 | 1198.70 | 1210.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 13:30:00 | 1199.70 | 1198.70 | 1210.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 1198.20 | 1198.06 | 1203.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:00:00 | 1194.50 | 1197.35 | 1202.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 11:15:00 | 1197.40 | 1197.64 | 1202.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 12:15:00 | 1209.00 | 1200.40 | 1202.61 | SL hit (close>static) qty=1.00 sl=1204.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-06 12:15:00 | 1209.00 | 1200.40 | 1202.61 | SL hit (close>static) qty=1.00 sl=1204.30 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 11:15:00 | 1210.00 | 1204.32 | 1203.78 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 09:15:00 | 1199.00 | 1203.73 | 1203.77 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 10:15:00 | 1208.40 | 1203.05 | 1202.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 12:15:00 | 1212.80 | 1205.90 | 1204.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 1229.90 | 1231.55 | 1224.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 14:00:00 | 1229.90 | 1231.55 | 1224.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1237.60 | 1243.18 | 1239.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 1237.60 | 1243.18 | 1239.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1235.00 | 1241.54 | 1239.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 1235.80 | 1241.54 | 1239.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 1242.00 | 1242.22 | 1240.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:15:00 | 1244.60 | 1242.22 | 1240.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 1239.50 | 1241.68 | 1240.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:00:00 | 1239.50 | 1241.68 | 1240.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 1238.30 | 1241.00 | 1240.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:00:00 | 1238.30 | 1241.00 | 1240.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1244.10 | 1245.60 | 1243.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 1244.10 | 1245.60 | 1243.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1242.20 | 1244.92 | 1243.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:15:00 | 1245.80 | 1244.92 | 1243.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 1246.90 | 1245.32 | 1243.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 12:30:00 | 1248.60 | 1245.91 | 1243.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 12:00:00 | 1247.80 | 1246.13 | 1245.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 13:15:00 | 1248.00 | 1246.32 | 1245.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 14:00:00 | 1248.70 | 1246.80 | 1245.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 1242.50 | 1245.94 | 1245.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 1242.50 | 1245.94 | 1245.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 1243.50 | 1245.45 | 1245.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 1250.30 | 1245.45 | 1245.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 13:15:00 | 1233.30 | 1245.31 | 1245.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 13:15:00 | 1233.30 | 1245.31 | 1245.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 13:15:00 | 1233.30 | 1245.31 | 1245.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 13:15:00 | 1233.30 | 1245.31 | 1245.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 13:15:00 | 1233.30 | 1245.31 | 1245.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 13:15:00 | 1233.30 | 1245.31 | 1245.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 14:15:00 | 1226.20 | 1241.49 | 1244.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 1242.60 | 1238.99 | 1242.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 1242.60 | 1238.99 | 1242.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 1242.60 | 1238.99 | 1242.33 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 1249.60 | 1243.85 | 1243.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 1250.30 | 1245.14 | 1243.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 1244.20 | 1246.21 | 1244.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 10:15:00 | 1244.20 | 1246.21 | 1244.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 1244.20 | 1246.21 | 1244.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 1244.20 | 1246.21 | 1244.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 1244.40 | 1245.85 | 1244.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 13:45:00 | 1246.70 | 1245.68 | 1244.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 15:00:00 | 1249.50 | 1246.45 | 1245.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 11:00:00 | 1246.50 | 1246.91 | 1245.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 11:30:00 | 1249.40 | 1247.47 | 1246.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1249.40 | 1251.99 | 1249.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:00:00 | 1249.40 | 1251.99 | 1249.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1259.50 | 1253.49 | 1250.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 11:45:00 | 1260.50 | 1254.70 | 1251.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 12:15:00 | 1260.30 | 1254.70 | 1251.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 12:45:00 | 1260.40 | 1256.12 | 1252.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 1264.70 | 1257.87 | 1253.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1272.70 | 1276.72 | 1271.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 15:15:00 | 1279.80 | 1275.55 | 1272.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 10:30:00 | 1279.00 | 1277.75 | 1274.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 12:15:00 | 1268.60 | 1273.51 | 1274.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 12:15:00 | 1268.60 | 1273.51 | 1274.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 12:15:00 | 1268.60 | 1273.51 | 1274.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 12:15:00 | 1268.60 | 1273.51 | 1274.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 12:15:00 | 1268.60 | 1273.51 | 1274.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 12:15:00 | 1268.60 | 1273.51 | 1274.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 12:15:00 | 1268.60 | 1273.51 | 1274.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 12:15:00 | 1268.60 | 1273.51 | 1274.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 12:15:00 | 1268.60 | 1273.51 | 1274.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 12:15:00 | 1268.60 | 1273.51 | 1274.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 1268.60 | 1273.51 | 1274.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 1266.40 | 1272.09 | 1273.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 1259.60 | 1253.06 | 1257.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 1259.60 | 1253.06 | 1257.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 1259.60 | 1253.06 | 1257.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:15:00 | 1258.80 | 1253.06 | 1257.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 1261.40 | 1254.73 | 1257.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:30:00 | 1265.60 | 1254.73 | 1257.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 1275.20 | 1261.19 | 1260.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 11:15:00 | 1277.30 | 1270.34 | 1265.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1268.00 | 1273.43 | 1269.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 1268.00 | 1273.43 | 1269.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1268.00 | 1273.43 | 1269.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 12:00:00 | 1275.10 | 1273.50 | 1270.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 09:30:00 | 1275.80 | 1276.50 | 1273.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 10:30:00 | 1277.10 | 1276.40 | 1273.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 11:15:00 | 1276.10 | 1274.51 | 1274.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 1272.90 | 1274.76 | 1274.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 13:00:00 | 1272.90 | 1274.76 | 1274.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-17 13:15:00 | 1269.60 | 1273.73 | 1273.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 13:15:00 | 1269.60 | 1273.73 | 1273.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 13:15:00 | 1269.60 | 1273.73 | 1273.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 13:15:00 | 1269.60 | 1273.73 | 1273.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 13:15:00 | 1269.60 | 1273.73 | 1273.93 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 11:15:00 | 1276.70 | 1273.81 | 1273.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 15:15:00 | 1280.00 | 1276.33 | 1275.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 11:15:00 | 1274.40 | 1278.37 | 1276.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 11:15:00 | 1274.40 | 1278.37 | 1276.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 1274.40 | 1278.37 | 1276.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:30:00 | 1273.10 | 1278.37 | 1276.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 1271.40 | 1276.97 | 1276.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:00:00 | 1271.40 | 1276.97 | 1276.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 1280.10 | 1276.93 | 1276.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 1283.80 | 1276.93 | 1276.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 10:45:00 | 1282.20 | 1278.09 | 1276.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 11:15:00 | 1281.60 | 1278.09 | 1276.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 12:30:00 | 1280.80 | 1279.02 | 1277.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 1280.20 | 1282.03 | 1280.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:00:00 | 1280.20 | 1282.03 | 1280.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 1279.70 | 1281.56 | 1279.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:15:00 | 1279.20 | 1281.56 | 1279.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 1282.00 | 1281.65 | 1280.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 14:15:00 | 1283.50 | 1281.65 | 1280.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 15:00:00 | 1284.70 | 1282.26 | 1280.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 1259.50 | 1277.75 | 1278.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 1259.50 | 1277.75 | 1278.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 1259.50 | 1277.75 | 1278.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 1259.50 | 1277.75 | 1278.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 1259.50 | 1277.75 | 1278.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 1259.50 | 1277.75 | 1278.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 09:15:00 | 1259.50 | 1277.75 | 1278.82 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 1270.10 | 1267.12 | 1266.98 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 09:15:00 | 1254.10 | 1265.99 | 1266.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 13:15:00 | 1250.40 | 1257.14 | 1261.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 13:15:00 | 1253.50 | 1252.56 | 1256.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-02 14:00:00 | 1253.50 | 1252.56 | 1256.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 1257.60 | 1254.08 | 1256.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 1251.60 | 1254.08 | 1256.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 11:15:00 | 1255.00 | 1254.73 | 1256.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 11:15:00 | 1260.70 | 1255.92 | 1256.87 | SL hit (close>static) qty=1.00 sl=1257.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 11:15:00 | 1260.70 | 1255.92 | 1256.87 | SL hit (close>static) qty=1.00 sl=1257.60 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 11:45:00 | 1255.30 | 1255.92 | 1256.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 12:15:00 | 1260.40 | 1256.82 | 1257.19 | SL hit (close>static) qty=1.00 sl=1257.60 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 13:45:00 | 1255.10 | 1255.79 | 1256.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1250.40 | 1252.76 | 1254.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 12:15:00 | 1245.80 | 1251.81 | 1254.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 1263.20 | 1255.02 | 1254.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 1263.20 | 1255.02 | 1254.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 1263.20 | 1255.02 | 1254.78 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 11:15:00 | 1244.60 | 1253.57 | 1254.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 1229.80 | 1244.73 | 1249.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 1210.80 | 1210.17 | 1218.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:45:00 | 1211.90 | 1210.17 | 1218.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1215.00 | 1211.96 | 1217.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 1217.50 | 1211.96 | 1217.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1215.00 | 1212.57 | 1217.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:15:00 | 1198.40 | 1212.57 | 1217.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 1222.20 | 1173.61 | 1173.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 1222.20 | 1173.61 | 1173.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 1243.50 | 1214.20 | 1197.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-27 13:15:00 | 1236.00 | 1236.98 | 1224.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-27 14:00:00 | 1236.00 | 1236.98 | 1224.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 1224.00 | 1233.60 | 1227.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 1224.00 | 1233.60 | 1227.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 1226.40 | 1232.16 | 1227.12 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 1193.10 | 1219.70 | 1222.67 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 1228.40 | 1218.34 | 1217.34 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 1199.00 | 1216.44 | 1216.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 1181.00 | 1209.35 | 1213.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1221.90 | 1193.10 | 1198.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 1221.90 | 1193.10 | 1198.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1221.90 | 1193.10 | 1198.72 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 1236.10 | 1207.87 | 1204.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 1243.80 | 1226.99 | 1216.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 1228.20 | 1240.89 | 1235.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 1228.20 | 1240.89 | 1235.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1228.20 | 1240.89 | 1235.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 1228.20 | 1240.89 | 1235.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1231.40 | 1238.99 | 1235.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:30:00 | 1226.00 | 1238.99 | 1235.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 1230.40 | 1236.36 | 1234.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:30:00 | 1228.30 | 1236.36 | 1234.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 1240.90 | 1237.02 | 1235.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 14:30:00 | 1236.30 | 1237.02 | 1235.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 1261.80 | 1261.36 | 1255.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:30:00 | 1257.50 | 1261.36 | 1255.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 1255.90 | 1260.27 | 1255.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:30:00 | 1255.00 | 1260.27 | 1255.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 1255.70 | 1259.35 | 1255.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 1261.20 | 1259.35 | 1255.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 10:15:00 | 1265.70 | 1267.81 | 1267.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 10:15:00 | 1265.70 | 1267.81 | 1267.89 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 1269.80 | 1268.21 | 1268.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 1277.80 | 1270.13 | 1268.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 1282.90 | 1284.70 | 1281.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 15:00:00 | 1282.90 | 1284.70 | 1281.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 1277.60 | 1283.28 | 1280.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 1282.50 | 1283.28 | 1280.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 1285.00 | 1283.62 | 1281.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 10:00:00 | 1296.80 | 1285.93 | 1283.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 11:00:00 | 1294.00 | 1287.54 | 1284.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 13:15:00 | 1299.90 | 1289.71 | 1286.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 10:15:00 | 1294.20 | 1299.00 | 1296.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 1300.60 | 1304.68 | 1300.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:00:00 | 1300.60 | 1304.68 | 1300.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 1307.00 | 1305.14 | 1300.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 1313.40 | 1305.17 | 1301.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 1297.00 | 1310.34 | 1307.33 | SL hit (close<static) qty=1.00 sl=1300.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-27 11:15:00 | 1283.90 | 1302.35 | 1304.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-27 11:15:00 | 1283.90 | 1302.35 | 1304.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-27 11:15:00 | 1283.90 | 1302.35 | 1304.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-27 11:15:00 | 1283.90 | 1302.35 | 1304.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 1283.90 | 1302.35 | 1304.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 1276.60 | 1290.51 | 1296.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 1293.80 | 1286.73 | 1292.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 14:15:00 | 1293.80 | 1286.73 | 1292.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 1293.80 | 1286.73 | 1292.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 14:30:00 | 1293.50 | 1286.73 | 1292.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 1288.10 | 1287.00 | 1291.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 1281.70 | 1287.00 | 1291.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:45:00 | 1278.30 | 1285.94 | 1290.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 11:15:00 | 1282.60 | 1286.37 | 1290.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 13:00:00 | 1287.60 | 1286.16 | 1289.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 13:15:00 | 1290.70 | 1287.07 | 1289.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:00:00 | 1290.70 | 1287.07 | 1289.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 14:15:00 | 1289.40 | 1287.53 | 1289.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:45:00 | 1298.70 | 1287.53 | 1289.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 15:15:00 | 1291.30 | 1288.29 | 1289.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:15:00 | 1303.10 | 1288.29 | 1289.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 1306.20 | 1291.87 | 1291.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 1306.20 | 1291.87 | 1291.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 1306.20 | 1291.87 | 1291.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 1306.20 | 1291.87 | 1291.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 09:15:00 | 1306.20 | 1291.87 | 1291.38 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 1284.00 | 1298.04 | 1299.35 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 1317.10 | 1299.98 | 1298.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 11:15:00 | 1319.30 | 1303.84 | 1300.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 1310.30 | 1319.93 | 1314.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 1310.30 | 1319.93 | 1314.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1310.30 | 1319.93 | 1314.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 12:15:00 | 1327.60 | 1319.78 | 1315.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 1296.50 | 1316.18 | 1315.91 | SL hit (close<static) qty=1.00 sl=1303.60 alert=retest2 |

### Cycle 64 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 1296.00 | 1312.14 | 1314.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 1283.30 | 1304.34 | 1310.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 1280.50 | 1279.26 | 1289.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:15:00 | 1279.60 | 1279.26 | 1289.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 1284.80 | 1280.10 | 1285.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:30:00 | 1285.00 | 1280.10 | 1285.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 1281.90 | 1280.46 | 1285.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:15:00 | 1291.50 | 1280.46 | 1285.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1292.50 | 1282.87 | 1286.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 1293.80 | 1282.87 | 1286.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 1298.40 | 1285.97 | 1287.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 1298.40 | 1285.97 | 1287.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 1294.50 | 1288.96 | 1288.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 1296.60 | 1290.49 | 1289.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 15:15:00 | 1286.90 | 1290.86 | 1289.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 15:15:00 | 1286.90 | 1290.86 | 1289.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 1286.90 | 1290.86 | 1289.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:15:00 | 1272.50 | 1290.86 | 1289.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 1276.30 | 1287.95 | 1288.40 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 11:15:00 | 1295.90 | 1286.69 | 1285.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 13:15:00 | 1302.30 | 1291.70 | 1288.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 1270.10 | 1289.04 | 1287.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 09:15:00 | 1270.10 | 1289.04 | 1287.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 1270.10 | 1289.04 | 1287.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 1270.10 | 1289.04 | 1287.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 1279.80 | 1287.19 | 1287.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 13:15:00 | 1267.50 | 1280.38 | 1283.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 09:15:00 | 1291.50 | 1266.05 | 1270.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 09:15:00 | 1291.50 | 1266.05 | 1270.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1291.50 | 1266.05 | 1270.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 1291.50 | 1266.05 | 1270.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 1291.90 | 1271.22 | 1272.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 11:00:00 | 1291.90 | 1271.22 | 1272.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 1297.30 | 1276.44 | 1274.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 1299.40 | 1281.03 | 1276.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 12:15:00 | 1292.40 | 1295.50 | 1288.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 13:00:00 | 1292.40 | 1295.50 | 1288.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 1279.40 | 1291.37 | 1287.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 1279.40 | 1291.37 | 1287.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 1281.10 | 1289.32 | 1286.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 1265.00 | 1289.32 | 1286.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 1260.40 | 1283.53 | 1284.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 15:15:00 | 1252.00 | 1265.03 | 1273.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1266.00 | 1265.23 | 1272.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1266.00 | 1265.23 | 1272.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1266.00 | 1265.23 | 1272.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 1256.10 | 1265.23 | 1272.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 1193.29 | 1222.89 | 1245.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 1218.80 | 1217.21 | 1233.42 | SL hit (close>ema200) qty=0.50 sl=1217.21 alert=retest2 |

### Cycle 71 — BUY (started 2026-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-09 12:15:00 | 1212.80 | 1202.72 | 1202.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 09:15:00 | 1222.20 | 1209.80 | 1206.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 1212.90 | 1223.08 | 1216.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 1212.90 | 1223.08 | 1216.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1212.90 | 1223.08 | 1216.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 09:45:00 | 1213.80 | 1223.08 | 1216.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 1226.50 | 1223.77 | 1217.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:30:00 | 1229.40 | 1225.06 | 1219.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 14:15:00 | 1216.00 | 1219.80 | 1219.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-15 14:15:00 | 1216.00 | 1219.80 | 1219.98 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 10:15:00 | 1222.20 | 1220.35 | 1220.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 13:15:00 | 1223.90 | 1221.60 | 1220.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 14:15:00 | 1221.20 | 1221.52 | 1220.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 14:15:00 | 1221.20 | 1221.52 | 1220.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 14:15:00 | 1221.20 | 1221.52 | 1220.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 15:00:00 | 1221.20 | 1221.52 | 1220.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 15:15:00 | 1220.80 | 1221.38 | 1220.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 09:15:00 | 1218.70 | 1221.38 | 1220.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 1223.40 | 1221.78 | 1221.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 11:45:00 | 1225.50 | 1222.90 | 1221.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:30:00 | 1224.60 | 1229.72 | 1229.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 11:15:00 | 1220.00 | 1227.77 | 1228.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-21 11:15:00 | 1220.00 | 1227.77 | 1228.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 11:15:00 | 1220.00 | 1227.77 | 1228.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 1212.80 | 1221.54 | 1224.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 12:15:00 | 1225.30 | 1221.73 | 1223.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 12:15:00 | 1225.30 | 1221.73 | 1223.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 12:15:00 | 1225.30 | 1221.73 | 1223.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 13:00:00 | 1225.30 | 1221.73 | 1223.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 1219.00 | 1221.19 | 1223.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 15:00:00 | 1214.80 | 1219.91 | 1222.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 1306.10 | 1236.71 | 1229.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 1306.10 | 1236.71 | 1229.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 12:15:00 | 1321.60 | 1272.08 | 1249.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 15:15:00 | 1313.00 | 1314.51 | 1293.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:15:00 | 1319.30 | 1314.51 | 1293.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 1342.70 | 1347.75 | 1334.46 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 1317.70 | 1331.53 | 1332.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 09:15:00 | 1301.50 | 1321.49 | 1326.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 1283.00 | 1280.17 | 1292.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-06 10:15:00 | 1286.40 | 1280.17 | 1292.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 1283.20 | 1280.78 | 1292.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:30:00 | 1289.60 | 1280.78 | 1292.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 1295.10 | 1283.64 | 1292.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:45:00 | 1297.50 | 1283.64 | 1292.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 1311.30 | 1289.17 | 1294.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 13:00:00 | 1311.30 | 1289.17 | 1294.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 13:15:00 | 1325.00 | 1296.34 | 1296.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 14:00:00 | 1325.00 | 1296.34 | 1296.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 1310.00 | 1299.07 | 1298.03 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 1294.40 | 1300.60 | 1301.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 14:15:00 | 1292.50 | 1298.93 | 1300.47 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-15 13:00:00 | 1226.00 | 2025-05-22 09:15:00 | 1214.10 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-05-19 12:15:00 | 1217.30 | 2025-05-22 09:15:00 | 1214.10 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-05-20 09:15:00 | 1222.30 | 2025-05-22 09:15:00 | 1214.10 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest1 | 2025-05-28 09:15:00 | 1247.40 | 2025-05-28 12:15:00 | 1238.70 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest1 | 2025-05-28 10:30:00 | 1246.40 | 2025-05-28 12:15:00 | 1238.70 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-05-29 09:15:00 | 1244.80 | 2025-06-03 10:15:00 | 1245.70 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2025-05-29 10:45:00 | 1247.80 | 2025-06-03 10:15:00 | 1245.70 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2025-05-30 10:30:00 | 1245.70 | 2025-06-03 10:15:00 | 1245.70 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-05-30 11:00:00 | 1248.30 | 2025-06-03 10:15:00 | 1245.70 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-06-05 09:15:00 | 1288.60 | 2025-06-16 14:15:00 | 1347.00 | STOP_HIT | 1.00 | 4.53% |
| SELL | retest2 | 2025-06-19 14:30:00 | 1322.80 | 2025-06-19 15:15:00 | 1331.90 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-06-20 10:45:00 | 1323.00 | 2025-06-24 09:15:00 | 1335.80 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-06-20 11:15:00 | 1321.90 | 2025-06-24 09:15:00 | 1335.80 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-06-20 12:15:00 | 1320.70 | 2025-06-24 09:15:00 | 1335.80 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-06-23 09:15:00 | 1312.50 | 2025-06-24 09:15:00 | 1335.80 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest1 | 2025-07-02 09:15:00 | 1275.50 | 2025-07-03 10:15:00 | 1281.20 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-07-16 09:15:00 | 1256.40 | 2025-07-16 15:15:00 | 1259.50 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-07-16 13:45:00 | 1257.20 | 2025-07-16 15:15:00 | 1259.50 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-07-22 10:30:00 | 1244.90 | 2025-07-24 09:15:00 | 1276.90 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-07-22 11:15:00 | 1242.50 | 2025-07-24 09:15:00 | 1276.90 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2025-07-22 13:45:00 | 1242.70 | 2025-07-24 09:15:00 | 1276.90 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-07-30 11:15:00 | 1292.00 | 2025-07-31 09:15:00 | 1266.50 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-07-30 14:45:00 | 1289.50 | 2025-07-31 09:15:00 | 1266.50 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-08-08 10:45:00 | 1197.60 | 2025-08-08 14:15:00 | 1210.70 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-08-12 10:30:00 | 1215.80 | 2025-08-20 09:15:00 | 1241.40 | STOP_HIT | 1.00 | 2.11% |
| BUY | retest2 | 2025-08-12 11:15:00 | 1215.60 | 2025-08-20 09:15:00 | 1241.40 | STOP_HIT | 1.00 | 2.12% |
| BUY | retest1 | 2025-08-25 09:30:00 | 1284.10 | 2025-08-26 09:15:00 | 1264.20 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-09-03 14:45:00 | 1259.10 | 2025-09-04 11:15:00 | 1266.90 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-09-17 09:30:00 | 1315.00 | 2025-09-22 13:15:00 | 1303.40 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-09-17 12:30:00 | 1312.30 | 2025-09-22 13:15:00 | 1303.40 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-09-18 09:15:00 | 1312.10 | 2025-09-22 13:15:00 | 1303.40 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-09-18 10:45:00 | 1312.20 | 2025-09-22 13:15:00 | 1303.40 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-09-24 09:15:00 | 1301.90 | 2025-09-29 13:15:00 | 1236.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 12:00:00 | 1300.00 | 2025-09-29 13:15:00 | 1235.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 09:15:00 | 1301.90 | 2025-10-01 09:15:00 | 1232.90 | STOP_HIT | 0.50 | 5.30% |
| SELL | retest2 | 2025-09-24 12:00:00 | 1300.00 | 2025-10-01 09:15:00 | 1232.90 | STOP_HIT | 0.50 | 5.16% |
| BUY | retest2 | 2025-10-13 14:15:00 | 1258.90 | 2025-10-14 10:15:00 | 1240.80 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-10-27 12:00:00 | 1288.60 | 2025-10-29 13:15:00 | 1252.00 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2025-10-27 12:30:00 | 1289.20 | 2025-10-29 13:15:00 | 1252.00 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2025-10-27 13:45:00 | 1288.20 | 2025-10-29 13:15:00 | 1252.00 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2025-10-28 15:00:00 | 1290.10 | 2025-10-29 13:15:00 | 1252.00 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2025-11-06 10:00:00 | 1194.50 | 2025-11-06 12:15:00 | 1209.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-11-06 11:15:00 | 1197.40 | 2025-11-06 12:15:00 | 1209.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-11-20 12:30:00 | 1248.60 | 2025-11-24 13:15:00 | 1233.30 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-11-21 12:00:00 | 1247.80 | 2025-11-24 13:15:00 | 1233.30 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-11-21 13:15:00 | 1248.00 | 2025-11-24 13:15:00 | 1233.30 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-11-21 14:00:00 | 1248.70 | 2025-11-24 13:15:00 | 1233.30 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-11-24 09:15:00 | 1250.30 | 2025-11-24 13:15:00 | 1233.30 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-11-27 13:45:00 | 1246.70 | 2025-12-08 12:15:00 | 1268.60 | STOP_HIT | 1.00 | 1.76% |
| BUY | retest2 | 2025-11-27 15:00:00 | 1249.50 | 2025-12-08 12:15:00 | 1268.60 | STOP_HIT | 1.00 | 1.53% |
| BUY | retest2 | 2025-11-28 11:00:00 | 1246.50 | 2025-12-08 12:15:00 | 1268.60 | STOP_HIT | 1.00 | 1.77% |
| BUY | retest2 | 2025-11-28 11:30:00 | 1249.40 | 2025-12-08 12:15:00 | 1268.60 | STOP_HIT | 1.00 | 1.54% |
| BUY | retest2 | 2025-12-01 11:45:00 | 1260.50 | 2025-12-08 12:15:00 | 1268.60 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2025-12-01 12:15:00 | 1260.30 | 2025-12-08 12:15:00 | 1268.60 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest2 | 2025-12-01 12:45:00 | 1260.40 | 2025-12-08 12:15:00 | 1268.60 | STOP_HIT | 1.00 | 0.65% |
| BUY | retest2 | 2025-12-02 09:15:00 | 1264.70 | 2025-12-08 12:15:00 | 1268.60 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2025-12-04 15:15:00 | 1279.80 | 2025-12-08 12:15:00 | 1268.60 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-12-05 10:30:00 | 1279.00 | 2025-12-08 12:15:00 | 1268.60 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-12-15 12:00:00 | 1275.10 | 2025-12-17 13:15:00 | 1269.60 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-12-16 09:30:00 | 1275.80 | 2025-12-17 13:15:00 | 1269.60 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-12-16 10:30:00 | 1277.10 | 2025-12-17 13:15:00 | 1269.60 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-12-17 11:15:00 | 1276.10 | 2025-12-17 13:15:00 | 1269.60 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-12-22 09:15:00 | 1283.80 | 2025-12-24 09:15:00 | 1259.50 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-12-22 10:45:00 | 1282.20 | 2025-12-24 09:15:00 | 1259.50 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-12-22 11:15:00 | 1281.60 | 2025-12-24 09:15:00 | 1259.50 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-12-22 12:30:00 | 1280.80 | 2025-12-24 09:15:00 | 1259.50 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-12-23 14:15:00 | 1283.50 | 2025-12-24 09:15:00 | 1259.50 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-12-23 15:00:00 | 1284.70 | 2025-12-24 09:15:00 | 1259.50 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2026-01-05 09:15:00 | 1251.60 | 2026-01-05 11:15:00 | 1260.70 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2026-01-05 11:15:00 | 1255.00 | 2026-01-05 11:15:00 | 1260.70 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2026-01-05 11:45:00 | 1255.30 | 2026-01-05 12:15:00 | 1260.40 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2026-01-05 13:45:00 | 1255.10 | 2026-01-07 09:15:00 | 1263.20 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2026-01-06 12:15:00 | 1245.80 | 2026-01-07 09:15:00 | 1263.20 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-01-13 09:15:00 | 1198.40 | 2026-01-22 09:15:00 | 1222.20 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2026-02-11 09:15:00 | 1261.20 | 2026-02-17 10:15:00 | 1265.70 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2026-02-23 10:00:00 | 1296.80 | 2026-02-27 09:15:00 | 1297.00 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2026-02-23 11:00:00 | 1294.00 | 2026-02-27 11:15:00 | 1283.90 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2026-02-23 13:15:00 | 1299.90 | 2026-02-27 11:15:00 | 1283.90 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2026-02-25 10:15:00 | 1294.20 | 2026-02-27 11:15:00 | 1283.90 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2026-02-26 09:15:00 | 1313.40 | 2026-02-27 11:15:00 | 1283.90 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2026-03-04 09:15:00 | 1281.70 | 2026-03-05 09:15:00 | 1306.20 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2026-03-04 09:45:00 | 1278.30 | 2026-03-05 09:15:00 | 1306.20 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2026-03-04 11:15:00 | 1282.60 | 2026-03-05 09:15:00 | 1306.20 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2026-03-04 13:00:00 | 1287.60 | 2026-03-05 09:15:00 | 1306.20 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2026-03-12 12:15:00 | 1327.60 | 2026-03-13 09:15:00 | 1296.50 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2026-04-01 10:15:00 | 1256.10 | 2026-04-02 09:15:00 | 1193.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 10:15:00 | 1256.10 | 2026-04-02 14:15:00 | 1218.80 | STOP_HIT | 0.50 | 2.97% |
| BUY | retest2 | 2026-04-13 12:30:00 | 1229.40 | 2026-04-15 14:15:00 | 1216.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-04-17 11:45:00 | 1225.50 | 2026-04-21 11:15:00 | 1220.00 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2026-04-21 10:30:00 | 1224.60 | 2026-04-21 11:15:00 | 1220.00 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2026-04-22 15:00:00 | 1214.80 | 2026-04-23 09:15:00 | 1306.10 | STOP_HIT | 1.00 | -7.52% |
