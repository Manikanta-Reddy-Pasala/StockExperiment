# TBO Tek Ltd. (TBOTEK)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 1227.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 69 |
| ALERT1 | 48 |
| ALERT2 | 47 |
| ALERT2_SKIP | 22 |
| ALERT3 | 122 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 62 |
| PARTIAL | 6 |
| TARGET_HIT | 4 |
| STOP_HIT | 62 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 71 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 51
- **Target hits / Stop hits / Partials:** 3 / 62 / 6
- **Avg / median % per leg:** -0.40% / -1.68%
- **Sum % (uncompounded):** -28.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 38 | 4 | 10.5% | 2 | 36 | 0 | -1.00% | -38.0% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.16% | -3.5% |
| BUY @ 3rd Alert (retest2) | 35 | 4 | 11.4% | 2 | 33 | 0 | -0.99% | -34.5% |
| SELL (all) | 33 | 16 | 48.5% | 1 | 26 | 6 | 0.29% | 9.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 33 | 16 | 48.5% | 1 | 26 | 6 | 0.29% | 9.6% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.16% | -3.5% |
| retest2 (combined) | 68 | 20 | 29.4% | 3 | 59 | 6 | -0.37% | -24.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 10:15:00 | 1199.50 | 1212.98 | 1213.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 12:15:00 | 1185.00 | 1204.97 | 1209.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 12:15:00 | 1191.80 | 1185.80 | 1194.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 12:15:00 | 1191.80 | 1185.80 | 1194.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 1191.80 | 1185.80 | 1194.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 13:00:00 | 1191.80 | 1185.80 | 1194.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 13:15:00 | 1213.00 | 1191.24 | 1196.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 14:00:00 | 1213.00 | 1191.24 | 1196.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 1233.50 | 1199.69 | 1199.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 15:00:00 | 1233.50 | 1199.69 | 1199.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 15:15:00 | 1239.00 | 1207.55 | 1203.47 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 15:15:00 | 1197.70 | 1203.91 | 1204.02 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 1241.50 | 1211.43 | 1207.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 12:15:00 | 1308.10 | 1249.15 | 1227.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 1298.80 | 1303.03 | 1281.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 10:00:00 | 1298.80 | 1303.03 | 1281.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 1287.50 | 1297.40 | 1282.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:45:00 | 1276.80 | 1297.40 | 1282.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 1300.00 | 1294.50 | 1286.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:45:00 | 1286.50 | 1294.50 | 1286.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 1294.30 | 1299.40 | 1293.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:45:00 | 1293.30 | 1299.40 | 1293.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 1309.80 | 1301.48 | 1295.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 11:30:00 | 1321.00 | 1305.16 | 1297.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 09:15:00 | 1316.60 | 1303.06 | 1298.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 12:00:00 | 1318.10 | 1306.46 | 1301.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 1312.20 | 1310.04 | 1305.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 1296.10 | 1307.26 | 1304.54 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-02 11:15:00 | 1286.60 | 1301.20 | 1302.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-02 11:15:00 | 1286.60 | 1301.20 | 1302.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-02 11:15:00 | 1286.60 | 1301.20 | 1302.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-02 11:15:00 | 1286.60 | 1301.20 | 1302.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 11:15:00 | 1286.60 | 1301.20 | 1302.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 13:15:00 | 1281.10 | 1294.79 | 1298.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 14:15:00 | 1297.00 | 1295.23 | 1298.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 15:00:00 | 1297.00 | 1295.23 | 1298.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 1299.00 | 1295.99 | 1298.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 1296.50 | 1295.99 | 1298.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 1296.80 | 1296.15 | 1298.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 12:00:00 | 1284.20 | 1294.52 | 1297.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 09:15:00 | 1332.00 | 1298.96 | 1297.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 09:15:00 | 1332.00 | 1298.96 | 1297.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 10:15:00 | 1352.20 | 1309.61 | 1302.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 12:15:00 | 1340.00 | 1352.22 | 1336.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 13:00:00 | 1340.00 | 1352.22 | 1336.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 1334.00 | 1348.58 | 1336.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 13:30:00 | 1326.70 | 1348.58 | 1336.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 1333.00 | 1345.46 | 1335.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 15:00:00 | 1333.00 | 1345.46 | 1335.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 1332.00 | 1342.77 | 1335.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:15:00 | 1323.90 | 1342.77 | 1335.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 1334.30 | 1340.87 | 1335.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:45:00 | 1334.10 | 1340.87 | 1335.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 1334.50 | 1339.60 | 1335.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:15:00 | 1332.00 | 1339.60 | 1335.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 1330.70 | 1337.82 | 1335.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:30:00 | 1330.20 | 1337.82 | 1335.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2025-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 14:15:00 | 1323.00 | 1333.52 | 1333.70 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 11:15:00 | 1341.20 | 1335.05 | 1334.25 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-06-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 14:15:00 | 1320.30 | 1332.19 | 1333.16 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-06-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 09:15:00 | 1364.00 | 1336.76 | 1334.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 10:15:00 | 1398.00 | 1349.01 | 1340.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 14:15:00 | 1340.60 | 1357.06 | 1348.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 14:15:00 | 1340.60 | 1357.06 | 1348.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 1340.60 | 1357.06 | 1348.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 1340.60 | 1357.06 | 1348.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 1344.00 | 1354.45 | 1347.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 1348.40 | 1354.45 | 1347.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 11:00:00 | 1347.70 | 1350.63 | 1347.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 13:15:00 | 1329.80 | 1344.34 | 1345.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-11 13:15:00 | 1329.80 | 1344.34 | 1345.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 1329.80 | 1344.34 | 1345.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 10:15:00 | 1313.30 | 1332.93 | 1339.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 14:15:00 | 1289.90 | 1289.68 | 1298.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 15:00:00 | 1289.90 | 1289.68 | 1298.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 1280.10 | 1287.68 | 1296.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:45:00 | 1278.00 | 1285.21 | 1294.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 14:30:00 | 1277.60 | 1285.42 | 1291.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:00:00 | 1273.80 | 1282.55 | 1288.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 14:15:00 | 1272.80 | 1263.07 | 1262.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 14:15:00 | 1272.80 | 1263.07 | 1262.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-24 14:15:00 | 1272.80 | 1263.07 | 1262.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2025-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 14:15:00 | 1272.80 | 1263.07 | 1262.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 1280.10 | 1268.39 | 1264.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 1377.80 | 1393.04 | 1368.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 10:00:00 | 1377.80 | 1393.04 | 1368.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1417.50 | 1436.20 | 1419.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 1417.50 | 1436.20 | 1419.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 1409.70 | 1430.90 | 1418.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:00:00 | 1409.70 | 1430.90 | 1418.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 1415.00 | 1425.27 | 1418.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:30:00 | 1415.10 | 1425.27 | 1418.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 1415.30 | 1423.28 | 1417.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 13:45:00 | 1415.00 | 1423.28 | 1417.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 1425.80 | 1423.78 | 1418.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 15:15:00 | 1406.00 | 1423.78 | 1418.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 1406.00 | 1420.23 | 1417.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:15:00 | 1427.40 | 1420.23 | 1417.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 11:15:00 | 1426.30 | 1420.93 | 1418.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 12:00:00 | 1425.30 | 1421.80 | 1418.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 14:15:00 | 1425.50 | 1422.95 | 1419.96 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 1424.00 | 1423.49 | 1420.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 1433.50 | 1423.49 | 1420.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1417.80 | 1422.35 | 1420.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 1417.80 | 1422.35 | 1420.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-07 10:15:00 | 1399.00 | 1417.68 | 1418.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-07 10:15:00 | 1399.00 | 1417.68 | 1418.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-07 10:15:00 | 1399.00 | 1417.68 | 1418.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-07 10:15:00 | 1399.00 | 1417.68 | 1418.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 10:15:00 | 1399.00 | 1417.68 | 1418.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 11:15:00 | 1396.60 | 1413.47 | 1416.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 1383.80 | 1375.98 | 1388.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 1383.80 | 1375.98 | 1388.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1383.80 | 1375.98 | 1388.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:30:00 | 1384.20 | 1375.98 | 1388.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1355.00 | 1344.94 | 1355.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 13:30:00 | 1341.40 | 1348.09 | 1354.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:30:00 | 1341.60 | 1340.18 | 1347.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 14:15:00 | 1345.90 | 1341.47 | 1346.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 15:00:00 | 1346.30 | 1342.43 | 1346.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 1350.80 | 1344.11 | 1346.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:15:00 | 1354.80 | 1344.11 | 1346.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 1373.40 | 1349.97 | 1349.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 1373.40 | 1349.97 | 1349.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 1373.40 | 1349.97 | 1349.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 1373.40 | 1349.97 | 1349.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 1373.40 | 1349.97 | 1349.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 10:15:00 | 1382.00 | 1356.37 | 1352.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 1397.70 | 1403.60 | 1390.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 09:45:00 | 1397.90 | 1403.60 | 1390.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 1413.30 | 1415.79 | 1404.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:45:00 | 1412.50 | 1415.79 | 1404.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 1420.20 | 1416.67 | 1405.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 1413.80 | 1416.67 | 1405.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1405.90 | 1414.47 | 1409.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:45:00 | 1405.60 | 1414.47 | 1409.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 1407.80 | 1413.14 | 1409.53 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 13:15:00 | 1394.00 | 1404.97 | 1406.31 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 1427.60 | 1407.23 | 1406.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 13:15:00 | 1456.40 | 1437.64 | 1426.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 15:15:00 | 1437.00 | 1438.20 | 1428.31 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 09:15:00 | 1448.30 | 1438.20 | 1428.31 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 10:30:00 | 1449.80 | 1442.06 | 1431.85 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-24 13:15:00 | 1448.20 | 1444.29 | 1434.71 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 1437.10 | 1442.85 | 1434.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:00:00 | 1437.10 | 1442.85 | 1434.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 1431.90 | 1440.66 | 1434.66 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-24 14:15:00 | 1431.90 | 1440.66 | 1434.66 | SL hit (close<ema400) qty=1.00 sl=1434.66 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-24 14:15:00 | 1431.90 | 1440.66 | 1434.66 | SL hit (close<ema400) qty=1.00 sl=1434.66 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-24 14:15:00 | 1431.90 | 1440.66 | 1434.66 | SL hit (close<ema400) qty=1.00 sl=1434.66 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-07-24 15:00:00 | 1431.90 | 1440.66 | 1434.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 1440.00 | 1440.53 | 1435.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 1419.90 | 1440.53 | 1435.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 1413.90 | 1435.20 | 1433.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:15:00 | 1404.00 | 1435.20 | 1433.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 1400.00 | 1428.16 | 1430.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 11:15:00 | 1393.80 | 1421.29 | 1426.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 09:15:00 | 1374.30 | 1372.64 | 1387.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 15:15:00 | 1390.00 | 1376.00 | 1382.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 1390.00 | 1376.00 | 1382.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 1385.20 | 1376.00 | 1382.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1376.50 | 1376.10 | 1382.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 11:30:00 | 1368.20 | 1374.43 | 1380.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 15:00:00 | 1352.80 | 1360.87 | 1367.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 10:15:00 | 1417.80 | 1372.11 | 1370.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 10:15:00 | 1417.80 | 1372.11 | 1370.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2025-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 10:15:00 | 1417.80 | 1372.11 | 1370.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-01 13:15:00 | 1422.00 | 1393.06 | 1381.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 11:15:00 | 1399.20 | 1403.38 | 1392.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-04 12:00:00 | 1399.20 | 1403.38 | 1392.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1399.80 | 1401.59 | 1394.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 1399.80 | 1401.59 | 1394.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 1388.00 | 1398.87 | 1393.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 1327.90 | 1398.87 | 1393.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 1339.60 | 1387.01 | 1388.60 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 09:15:00 | 1397.50 | 1371.94 | 1371.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 1428.10 | 1400.40 | 1388.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 14:15:00 | 1448.00 | 1449.99 | 1431.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 15:00:00 | 1448.00 | 1449.99 | 1431.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 1437.00 | 1447.39 | 1431.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 09:15:00 | 1480.40 | 1447.39 | 1431.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 1461.00 | 1453.69 | 1444.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 11:15:00 | 1461.70 | 1456.56 | 1453.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 12:00:00 | 1453.00 | 1455.82 | 1454.70 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 1462.90 | 1457.24 | 1455.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 14:00:00 | 1464.00 | 1458.59 | 1456.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 14:30:00 | 1464.00 | 1459.47 | 1456.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:15:00 | 1474.00 | 1459.88 | 1457.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 12:30:00 | 1464.70 | 1460.72 | 1458.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 1456.50 | 1459.87 | 1458.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:15:00 | 1456.80 | 1459.87 | 1458.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 1456.00 | 1459.10 | 1458.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 1463.60 | 1458.66 | 1458.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 1437.10 | 1454.58 | 1456.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 1437.10 | 1454.58 | 1456.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 1437.10 | 1454.58 | 1456.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 1437.10 | 1454.58 | 1456.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 1437.10 | 1454.58 | 1456.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 1437.10 | 1454.58 | 1456.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 1437.10 | 1454.58 | 1456.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 1437.10 | 1454.58 | 1456.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 1437.10 | 1454.58 | 1456.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 1437.10 | 1454.58 | 1456.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 09:15:00 | 1420.10 | 1440.17 | 1447.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 10:15:00 | 1337.80 | 1331.31 | 1351.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 10:45:00 | 1338.40 | 1331.31 | 1351.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 1362.70 | 1339.38 | 1350.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 1362.70 | 1339.38 | 1350.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 1363.00 | 1344.10 | 1351.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 1363.00 | 1344.10 | 1351.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 1401.00 | 1357.85 | 1356.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 1546.30 | 1411.34 | 1385.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 11:15:00 | 1526.00 | 1539.01 | 1487.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 12:00:00 | 1526.00 | 1539.01 | 1487.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 1566.10 | 1596.33 | 1567.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 15:00:00 | 1566.10 | 1596.33 | 1567.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 1578.00 | 1592.67 | 1568.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 1630.10 | 1592.67 | 1568.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 09:45:00 | 1628.60 | 1598.71 | 1573.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 1583.30 | 1600.81 | 1602.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 1583.30 | 1600.81 | 1602.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 12:15:00 | 1583.30 | 1600.81 | 1602.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 13:15:00 | 1569.20 | 1594.49 | 1599.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 11:15:00 | 1580.30 | 1564.24 | 1573.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 11:15:00 | 1580.30 | 1564.24 | 1573.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 1580.30 | 1564.24 | 1573.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:00:00 | 1580.30 | 1564.24 | 1573.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 1593.00 | 1569.99 | 1574.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:00:00 | 1593.00 | 1569.99 | 1574.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 1590.60 | 1574.11 | 1576.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:30:00 | 1592.20 | 1574.11 | 1576.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 15:15:00 | 1580.00 | 1577.99 | 1577.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 10:15:00 | 1598.80 | 1584.70 | 1581.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 15:15:00 | 1581.50 | 1589.86 | 1585.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 15:15:00 | 1581.50 | 1589.86 | 1585.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 1581.50 | 1589.86 | 1585.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 1596.10 | 1589.86 | 1585.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 1608.60 | 1593.61 | 1587.91 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 09:15:00 | 1575.30 | 1593.85 | 1594.11 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 15:15:00 | 1670.00 | 1606.50 | 1598.44 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 14:15:00 | 1596.00 | 1613.62 | 1614.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 1583.90 | 1605.50 | 1610.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 10:15:00 | 1526.50 | 1517.48 | 1539.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 11:00:00 | 1526.50 | 1517.48 | 1539.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 1495.70 | 1494.20 | 1515.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:30:00 | 1515.30 | 1494.20 | 1515.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 1511.60 | 1497.68 | 1514.91 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 1528.50 | 1519.23 | 1518.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 1543.90 | 1524.91 | 1521.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 09:15:00 | 1579.20 | 1587.51 | 1572.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 10:00:00 | 1579.20 | 1587.51 | 1572.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 1571.70 | 1584.35 | 1572.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 1571.70 | 1584.35 | 1572.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 1562.00 | 1579.88 | 1571.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:45:00 | 1563.20 | 1579.88 | 1571.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 1555.10 | 1574.92 | 1569.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:45:00 | 1554.00 | 1574.92 | 1569.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 14:15:00 | 1531.40 | 1563.08 | 1565.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 12:15:00 | 1530.10 | 1539.70 | 1547.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 1560.00 | 1541.45 | 1545.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 09:15:00 | 1560.00 | 1541.45 | 1545.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1560.00 | 1541.45 | 1545.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 1560.00 | 1541.45 | 1545.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 1558.70 | 1544.90 | 1546.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 12:00:00 | 1552.00 | 1546.32 | 1547.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 13:15:00 | 1555.20 | 1549.20 | 1548.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 1555.20 | 1549.20 | 1548.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 14:15:00 | 1560.10 | 1551.38 | 1549.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 10:15:00 | 1538.10 | 1550.76 | 1549.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 10:15:00 | 1538.10 | 1550.76 | 1549.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 1538.10 | 1550.76 | 1549.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 1538.10 | 1550.76 | 1549.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 1536.50 | 1547.91 | 1548.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 14:15:00 | 1531.80 | 1540.64 | 1544.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 10:15:00 | 1541.90 | 1538.97 | 1542.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 10:15:00 | 1541.90 | 1538.97 | 1542.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 1541.90 | 1538.97 | 1542.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:45:00 | 1544.90 | 1538.97 | 1542.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 1513.00 | 1533.78 | 1540.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 12:15:00 | 1504.30 | 1533.78 | 1540.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 10:15:00 | 1557.30 | 1522.40 | 1528.56 | SL hit (close>static) qty=1.00 sl=1542.20 alert=retest2 |

### Cycle 32 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 1551.00 | 1532.25 | 1532.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 1575.20 | 1547.94 | 1540.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 14:15:00 | 1586.00 | 1588.51 | 1577.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 15:00:00 | 1586.00 | 1588.51 | 1577.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 1575.80 | 1585.97 | 1577.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 1567.90 | 1582.10 | 1576.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1602.00 | 1586.08 | 1579.16 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 11:15:00 | 1585.00 | 1597.37 | 1597.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 15:15:00 | 1578.60 | 1589.47 | 1593.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 10:15:00 | 1495.00 | 1490.55 | 1509.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 10:45:00 | 1501.60 | 1490.55 | 1509.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 34 — BUY (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 09:15:00 | 1646.50 | 1524.58 | 1516.99 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 10:15:00 | 1580.00 | 1595.82 | 1596.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 11:15:00 | 1566.10 | 1589.88 | 1593.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 10:15:00 | 1578.30 | 1572.16 | 1580.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-14 11:00:00 | 1578.30 | 1572.16 | 1580.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1564.50 | 1570.68 | 1576.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 10:45:00 | 1555.70 | 1570.72 | 1575.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 11:15:00 | 1590.60 | 1574.70 | 1577.21 | SL hit (close>static) qty=1.00 sl=1584.00 alert=retest2 |

### Cycle 36 — BUY (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 12:15:00 | 1599.70 | 1579.70 | 1579.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 13:15:00 | 1616.80 | 1587.12 | 1582.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 09:15:00 | 1612.40 | 1613.62 | 1603.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 12:15:00 | 1594.20 | 1611.28 | 1605.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 1594.20 | 1611.28 | 1605.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:30:00 | 1595.60 | 1611.28 | 1605.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 1599.20 | 1608.86 | 1604.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:30:00 | 1598.20 | 1608.86 | 1604.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1693.70 | 1626.22 | 1613.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 10:15:00 | 1708.00 | 1626.22 | 1613.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 1751.00 | 1662.72 | 1640.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 10:00:00 | 1705.50 | 1722.28 | 1704.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 12:30:00 | 1702.50 | 1714.05 | 1705.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 1709.60 | 1713.16 | 1705.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 09:15:00 | 1717.80 | 1710.34 | 1705.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 11:15:00 | 1680.40 | 1705.71 | 1704.69 | SL hit (close<static) qty=1.00 sl=1701.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 13:15:00 | 1693.90 | 1702.43 | 1703.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 13:15:00 | 1693.90 | 1702.43 | 1703.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 13:15:00 | 1693.90 | 1702.43 | 1703.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 13:15:00 | 1693.90 | 1702.43 | 1703.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 13:15:00 | 1693.90 | 1702.43 | 1703.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 12:15:00 | 1664.20 | 1688.52 | 1694.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 1678.50 | 1673.92 | 1684.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 09:30:00 | 1673.60 | 1673.92 | 1684.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1687.90 | 1676.72 | 1684.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:45:00 | 1695.00 | 1676.72 | 1684.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 1675.00 | 1676.37 | 1683.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:30:00 | 1679.40 | 1676.37 | 1683.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 1683.50 | 1677.80 | 1683.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:00:00 | 1683.50 | 1677.80 | 1683.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 1687.00 | 1679.64 | 1684.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:15:00 | 1686.30 | 1679.64 | 1684.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 1685.00 | 1680.71 | 1684.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 15:15:00 | 1680.00 | 1680.71 | 1684.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 10:30:00 | 1681.80 | 1680.22 | 1683.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 13:15:00 | 1693.00 | 1679.18 | 1681.58 | SL hit (close>static) qty=1.00 sl=1690.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-02 13:15:00 | 1693.00 | 1679.18 | 1681.58 | SL hit (close>static) qty=1.00 sl=1690.80 alert=retest2 |

### Cycle 38 — BUY (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 14:15:00 | 1700.00 | 1683.35 | 1683.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 15:15:00 | 1704.00 | 1687.48 | 1685.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 11:15:00 | 1684.30 | 1690.59 | 1687.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 11:15:00 | 1684.30 | 1690.59 | 1687.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 1684.30 | 1690.59 | 1687.52 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2025-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 13:15:00 | 1671.70 | 1687.13 | 1687.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 09:15:00 | 1634.90 | 1670.84 | 1679.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 13:15:00 | 1618.90 | 1612.66 | 1633.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-08 14:00:00 | 1618.90 | 1612.66 | 1633.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 1605.70 | 1611.27 | 1631.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 15:15:00 | 1600.00 | 1611.27 | 1631.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 09:30:00 | 1590.90 | 1607.77 | 1626.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 1635.80 | 1619.07 | 1627.26 | SL hit (close>static) qty=1.00 sl=1633.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 1635.80 | 1619.07 | 1627.26 | SL hit (close>static) qty=1.00 sl=1633.40 alert=retest2 |

### Cycle 40 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 1672.50 | 1633.86 | 1632.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 09:15:00 | 1689.10 | 1661.43 | 1654.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 14:15:00 | 1676.30 | 1680.16 | 1668.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 15:00:00 | 1676.30 | 1680.16 | 1668.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 1675.90 | 1679.30 | 1669.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 1671.20 | 1682.66 | 1671.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 1683.80 | 1697.73 | 1687.26 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 1638.00 | 1675.90 | 1680.34 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 1687.50 | 1674.09 | 1673.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 1698.90 | 1683.43 | 1678.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 09:15:00 | 1684.00 | 1687.47 | 1681.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 09:15:00 | 1684.00 | 1687.47 | 1681.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 1684.00 | 1687.47 | 1681.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:30:00 | 1679.10 | 1687.47 | 1681.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 1680.40 | 1686.06 | 1681.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 11:00:00 | 1680.40 | 1686.06 | 1681.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 11:15:00 | 1679.70 | 1684.79 | 1681.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 12:15:00 | 1687.00 | 1684.79 | 1681.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 11:15:00 | 1687.00 | 1695.92 | 1696.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 1687.00 | 1695.92 | 1696.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 1659.60 | 1681.39 | 1688.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 1672.20 | 1662.18 | 1671.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 10:15:00 | 1672.20 | 1662.18 | 1671.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1672.20 | 1662.18 | 1671.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:30:00 | 1670.70 | 1662.18 | 1671.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 1660.90 | 1661.93 | 1670.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 13:30:00 | 1658.20 | 1662.83 | 1669.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:30:00 | 1659.00 | 1662.85 | 1667.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 10:00:00 | 1657.00 | 1662.85 | 1667.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 09:15:00 | 1575.29 | 1640.66 | 1645.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 1646.40 | 1640.66 | 1645.52 | SL hit (close>static) qty=0.50 sl=1640.66 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 09:15:00 | 1576.05 | 1640.66 | 1645.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 1646.40 | 1640.66 | 1645.52 | SL hit (close>static) qty=0.50 sl=1640.66 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 09:15:00 | 1574.15 | 1640.66 | 1645.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 1646.40 | 1640.66 | 1645.52 | SL hit (close>static) qty=0.50 sl=1640.66 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 12:45:00 | 1658.30 | 1647.67 | 1647.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 13:15:00 | 1656.90 | 1649.51 | 1648.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 13:15:00 | 1656.90 | 1649.51 | 1648.54 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 1627.70 | 1645.94 | 1647.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 1622.70 | 1641.29 | 1644.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 1526.50 | 1510.65 | 1526.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 1526.50 | 1510.65 | 1526.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1526.50 | 1510.65 | 1526.63 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 11:15:00 | 1545.90 | 1531.57 | 1530.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 13:15:00 | 1558.40 | 1539.33 | 1534.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 1529.40 | 1546.54 | 1541.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 13:15:00 | 1529.40 | 1546.54 | 1541.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 1529.40 | 1546.54 | 1541.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:00:00 | 1529.40 | 1546.54 | 1541.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 1523.90 | 1542.01 | 1540.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 15:00:00 | 1523.90 | 1542.01 | 1540.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 1512.50 | 1536.11 | 1537.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 1490.00 | 1526.89 | 1533.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 14:15:00 | 1511.00 | 1502.23 | 1516.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-19 14:45:00 | 1512.00 | 1502.23 | 1516.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 1512.80 | 1502.20 | 1510.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 13:00:00 | 1512.80 | 1502.20 | 1510.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 13:15:00 | 1516.80 | 1505.12 | 1511.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 14:00:00 | 1516.80 | 1505.12 | 1511.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 1519.90 | 1508.08 | 1511.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 14:30:00 | 1524.40 | 1508.08 | 1511.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 1480.80 | 1501.01 | 1507.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 10:15:00 | 1473.50 | 1501.01 | 1507.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 13:30:00 | 1470.50 | 1484.29 | 1496.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 1399.83 | 1437.32 | 1454.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 1396.97 | 1437.32 | 1454.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 1421.90 | 1412.87 | 1430.08 | SL hit (close>ema200) qty=0.50 sl=1412.87 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 1421.90 | 1412.87 | 1430.08 | SL hit (close>ema200) qty=0.50 sl=1412.87 alert=retest2 |

### Cycle 48 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1445.40 | 1394.57 | 1389.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 10:15:00 | 1455.00 | 1406.66 | 1395.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 09:15:00 | 1434.90 | 1440.63 | 1421.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 1434.90 | 1440.63 | 1421.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 1434.90 | 1440.63 | 1421.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 09:30:00 | 1427.20 | 1440.63 | 1421.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1436.80 | 1447.10 | 1435.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 1436.80 | 1447.10 | 1435.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 1428.80 | 1443.44 | 1434.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:45:00 | 1425.50 | 1443.44 | 1434.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 1450.60 | 1444.87 | 1435.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 13:00:00 | 1453.30 | 1446.56 | 1437.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:00:00 | 1453.80 | 1451.36 | 1443.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-11 09:15:00 | 1598.63 | 1560.45 | 1529.96 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-02-11 09:15:00 | 1599.18 | 1560.45 | 1529.96 | Target hit (10%) qty=1.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 09:30:00 | 1469.90 | 1536.53 | 1532.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 1475.00 | 1524.22 | 1527.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 1475.00 | 1524.22 | 1527.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 1403.00 | 1472.48 | 1497.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 12:15:00 | 1464.90 | 1462.78 | 1485.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 13:00:00 | 1464.90 | 1462.78 | 1485.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 13:15:00 | 1495.10 | 1469.25 | 1486.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 14:00:00 | 1495.10 | 1469.25 | 1486.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 1538.30 | 1483.06 | 1491.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 15:00:00 | 1538.30 | 1483.06 | 1491.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 15:15:00 | 1483.00 | 1483.05 | 1490.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 1458.00 | 1483.05 | 1490.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 14:15:00 | 1385.10 | 1404.05 | 1426.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-24 09:15:00 | 1312.20 | 1338.42 | 1355.53 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 50 — BUY (started 2026-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 14:15:00 | 1185.00 | 1161.07 | 1160.04 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 1141.90 | 1159.24 | 1159.51 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 10:15:00 | 1170.00 | 1161.39 | 1160.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-09 11:15:00 | 1183.80 | 1165.88 | 1162.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 1197.00 | 1205.22 | 1192.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 11:00:00 | 1197.00 | 1205.22 | 1192.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 1190.80 | 1202.34 | 1192.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:45:00 | 1188.60 | 1202.34 | 1192.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 1182.80 | 1198.43 | 1191.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 1182.80 | 1198.43 | 1191.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 1193.90 | 1197.52 | 1191.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 14:15:00 | 1199.00 | 1197.52 | 1191.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 14:45:00 | 1204.10 | 1199.04 | 1193.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:00:00 | 1201.70 | 1201.33 | 1195.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 10:15:00 | 1174.30 | 1204.71 | 1203.89 | SL hit (close<static) qty=1.00 sl=1174.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 10:15:00 | 1174.30 | 1204.71 | 1203.89 | SL hit (close<static) qty=1.00 sl=1174.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 10:15:00 | 1174.30 | 1204.71 | 1203.89 | SL hit (close<static) qty=1.00 sl=1174.70 alert=retest2 |

### Cycle 53 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 1174.90 | 1198.75 | 1201.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 1166.30 | 1192.26 | 1198.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 13:15:00 | 1160.90 | 1160.37 | 1174.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:00:00 | 1160.90 | 1160.37 | 1174.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1180.30 | 1164.36 | 1175.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 1185.60 | 1164.36 | 1175.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 1168.00 | 1165.09 | 1174.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:45:00 | 1181.40 | 1168.87 | 1175.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1178.10 | 1170.72 | 1175.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:30:00 | 1170.80 | 1172.19 | 1175.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 14:30:00 | 1176.60 | 1174.74 | 1176.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 15:15:00 | 1166.60 | 1174.74 | 1176.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 10:15:00 | 1197.90 | 1178.81 | 1177.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-18 10:15:00 | 1197.90 | 1178.81 | 1177.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-18 10:15:00 | 1197.90 | 1178.81 | 1177.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 1197.90 | 1178.81 | 1177.58 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2026-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 15:15:00 | 1171.00 | 1178.20 | 1178.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 1142.20 | 1171.00 | 1174.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 1047.70 | 1035.01 | 1063.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:00:00 | 1047.70 | 1035.01 | 1063.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 1068.50 | 1041.70 | 1064.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 11:45:00 | 1068.90 | 1041.70 | 1064.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1069.20 | 1047.20 | 1064.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 1071.40 | 1047.20 | 1064.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1059.90 | 1049.74 | 1064.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 1069.90 | 1049.74 | 1064.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 1055.00 | 1052.32 | 1062.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 1099.50 | 1052.32 | 1062.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1119.60 | 1065.78 | 1068.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:45:00 | 1115.80 | 1065.78 | 1068.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 1117.90 | 1076.20 | 1072.63 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2026-03-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 11:15:00 | 1072.70 | 1084.40 | 1084.86 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 1089.00 | 1083.17 | 1082.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 15:15:00 | 1100.00 | 1089.53 | 1085.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 1069.90 | 1085.60 | 1084.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 1069.90 | 1085.60 | 1084.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 1069.90 | 1085.60 | 1084.35 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 11:15:00 | 1072.90 | 1081.67 | 1082.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-06 09:15:00 | 1067.60 | 1079.77 | 1081.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 11:15:00 | 1086.00 | 1080.59 | 1081.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 11:15:00 | 1086.00 | 1080.59 | 1081.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 11:15:00 | 1086.00 | 1080.59 | 1081.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 11:45:00 | 1084.60 | 1080.59 | 1081.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 1094.60 | 1083.39 | 1082.69 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2026-04-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 13:15:00 | 1081.70 | 1084.73 | 1084.82 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 1162.10 | 1099.23 | 1091.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 11:15:00 | 1193.10 | 1130.86 | 1107.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 1143.50 | 1149.82 | 1128.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 1143.50 | 1149.82 | 1128.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1163.90 | 1170.34 | 1158.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:30:00 | 1195.90 | 1180.92 | 1168.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 12:15:00 | 1248.10 | 1272.72 | 1275.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 12:15:00 | 1248.10 | 1272.72 | 1275.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 13:15:00 | 1243.50 | 1266.88 | 1273.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1256.10 | 1247.74 | 1255.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1256.10 | 1247.74 | 1255.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1256.10 | 1247.74 | 1255.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1260.00 | 1247.74 | 1255.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1266.90 | 1251.58 | 1256.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 1268.30 | 1251.58 | 1256.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 1266.20 | 1254.50 | 1257.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:15:00 | 1271.40 | 1254.50 | 1257.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 1260.10 | 1256.93 | 1257.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:30:00 | 1266.10 | 1256.93 | 1257.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 1265.90 | 1258.73 | 1258.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 15:15:00 | 1271.00 | 1261.18 | 1259.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 11:15:00 | 1255.50 | 1261.19 | 1260.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 11:15:00 | 1255.50 | 1261.19 | 1260.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 1255.50 | 1261.19 | 1260.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:45:00 | 1250.50 | 1261.19 | 1260.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 12:15:00 | 1252.00 | 1259.35 | 1259.48 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 1275.20 | 1261.33 | 1259.98 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 13:15:00 | 1260.50 | 1263.25 | 1263.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 14:15:00 | 1255.10 | 1261.62 | 1262.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 1275.80 | 1261.81 | 1262.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 1275.80 | 1261.81 | 1262.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1275.80 | 1261.81 | 1262.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:45:00 | 1276.30 | 1261.81 | 1262.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 1282.80 | 1266.01 | 1264.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 13:15:00 | 1286.50 | 1273.38 | 1268.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 14:15:00 | 1282.20 | 1284.11 | 1278.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 14:15:00 | 1282.20 | 1284.11 | 1278.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 1282.20 | 1284.11 | 1278.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:30:00 | 1279.80 | 1284.11 | 1278.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 1276.00 | 1282.49 | 1277.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 10:15:00 | 1302.60 | 1280.32 | 1278.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 13:15:00 | 1269.00 | 1283.11 | 1281.10 | SL hit (close<static) qty=1.00 sl=1276.00 alert=retest2 |

### Cycle 69 — SELL (started 2026-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 14:15:00 | 1265.40 | 1279.57 | 1279.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 09:15:00 | 1246.90 | 1270.70 | 1275.48 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-29 11:30:00 | 1321.00 | 2025-06-02 11:15:00 | 1286.60 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2025-05-30 09:15:00 | 1316.60 | 2025-06-02 11:15:00 | 1286.60 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2025-05-30 12:00:00 | 1318.10 | 2025-06-02 11:15:00 | 1286.60 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2025-06-02 09:15:00 | 1312.20 | 2025-06-02 11:15:00 | 1286.60 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-06-03 12:00:00 | 1284.20 | 2025-06-04 09:15:00 | 1332.00 | STOP_HIT | 1.00 | -3.72% |
| BUY | retest2 | 2025-06-11 09:15:00 | 1348.40 | 2025-06-11 13:15:00 | 1329.80 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-06-11 11:00:00 | 1347.70 | 2025-06-11 13:15:00 | 1329.80 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-06-17 10:45:00 | 1278.00 | 2025-06-24 14:15:00 | 1272.80 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2025-06-17 14:30:00 | 1277.60 | 2025-06-24 14:15:00 | 1272.80 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2025-06-18 11:00:00 | 1273.80 | 2025-06-24 14:15:00 | 1272.80 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2025-07-04 09:15:00 | 1427.40 | 2025-07-07 10:15:00 | 1399.00 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-07-04 11:15:00 | 1426.30 | 2025-07-07 10:15:00 | 1399.00 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-07-04 12:00:00 | 1425.30 | 2025-07-07 10:15:00 | 1399.00 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-07-04 14:15:00 | 1425.50 | 2025-07-07 10:15:00 | 1399.00 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-07-11 13:30:00 | 1341.40 | 2025-07-15 09:15:00 | 1373.40 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2025-07-14 11:30:00 | 1341.60 | 2025-07-15 09:15:00 | 1373.40 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-07-14 14:15:00 | 1345.90 | 2025-07-15 09:15:00 | 1373.40 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-07-14 15:00:00 | 1346.30 | 2025-07-15 09:15:00 | 1373.40 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest1 | 2025-07-24 09:15:00 | 1448.30 | 2025-07-24 14:15:00 | 1431.90 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest1 | 2025-07-24 10:30:00 | 1449.80 | 2025-07-24 14:15:00 | 1431.90 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest1 | 2025-07-24 13:15:00 | 1448.20 | 2025-07-24 14:15:00 | 1431.90 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-07-30 11:30:00 | 1368.20 | 2025-08-01 10:15:00 | 1417.80 | STOP_HIT | 1.00 | -3.63% |
| SELL | retest2 | 2025-07-31 15:00:00 | 1352.80 | 2025-08-01 10:15:00 | 1417.80 | STOP_HIT | 1.00 | -4.80% |
| BUY | retest2 | 2025-08-14 09:15:00 | 1480.40 | 2025-08-22 10:15:00 | 1437.10 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2025-08-18 09:15:00 | 1461.00 | 2025-08-22 10:15:00 | 1437.10 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-08-19 11:15:00 | 1461.70 | 2025-08-22 10:15:00 | 1437.10 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-08-20 12:00:00 | 1453.00 | 2025-08-22 10:15:00 | 1437.10 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-08-20 14:00:00 | 1464.00 | 2025-08-22 10:15:00 | 1437.10 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-08-20 14:30:00 | 1464.00 | 2025-08-22 10:15:00 | 1437.10 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-08-21 09:15:00 | 1474.00 | 2025-08-22 10:15:00 | 1437.10 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-08-21 12:30:00 | 1464.70 | 2025-08-22 10:15:00 | 1437.10 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-08-22 09:15:00 | 1463.60 | 2025-08-22 10:15:00 | 1437.10 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-09-09 09:15:00 | 1630.10 | 2025-09-11 12:15:00 | 1583.30 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2025-09-09 09:45:00 | 1628.60 | 2025-09-11 12:15:00 | 1583.30 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2025-10-10 12:00:00 | 1552.00 | 2025-10-10 13:15:00 | 1555.20 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-10-14 12:15:00 | 1504.30 | 2025-10-15 10:15:00 | 1557.30 | STOP_HIT | 1.00 | -3.52% |
| SELL | retest2 | 2025-11-17 10:45:00 | 1555.70 | 2025-11-17 11:15:00 | 1590.60 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-11-20 10:15:00 | 1708.00 | 2025-11-26 11:15:00 | 1680.40 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-11-21 09:15:00 | 1751.00 | 2025-11-26 13:15:00 | 1693.90 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2025-11-25 10:00:00 | 1705.50 | 2025-11-26 13:15:00 | 1693.90 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-11-25 12:30:00 | 1702.50 | 2025-11-26 13:15:00 | 1693.90 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-11-26 09:15:00 | 1717.80 | 2025-11-26 13:15:00 | 1693.90 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-12-01 15:15:00 | 1680.00 | 2025-12-02 13:15:00 | 1693.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-12-02 10:30:00 | 1681.80 | 2025-12-02 13:15:00 | 1693.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-12-08 15:15:00 | 1600.00 | 2025-12-09 12:15:00 | 1635.80 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-12-09 09:30:00 | 1590.90 | 2025-12-09 12:15:00 | 1635.80 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2025-12-22 12:15:00 | 1687.00 | 2025-12-29 11:15:00 | 1687.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-12-31 13:30:00 | 1658.20 | 2026-01-05 09:15:00 | 1575.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-31 13:30:00 | 1658.20 | 2026-01-05 09:15:00 | 1646.40 | STOP_HIT | 0.50 | 0.71% |
| SELL | retest2 | 2026-01-01 09:30:00 | 1659.00 | 2026-01-05 09:15:00 | 1576.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-01 09:30:00 | 1659.00 | 2026-01-05 09:15:00 | 1646.40 | STOP_HIT | 0.50 | 0.76% |
| SELL | retest2 | 2026-01-01 10:00:00 | 1657.00 | 2026-01-05 09:15:00 | 1574.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-01 10:00:00 | 1657.00 | 2026-01-05 09:15:00 | 1646.40 | STOP_HIT | 0.50 | 0.64% |
| SELL | retest2 | 2026-01-05 12:45:00 | 1658.30 | 2026-01-05 13:15:00 | 1656.90 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2026-01-21 10:15:00 | 1473.50 | 2026-01-27 09:15:00 | 1399.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-21 13:30:00 | 1470.50 | 2026-01-27 09:15:00 | 1396.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-21 10:15:00 | 1473.50 | 2026-01-28 09:15:00 | 1421.90 | STOP_HIT | 0.50 | 3.50% |
| SELL | retest2 | 2026-01-21 13:30:00 | 1470.50 | 2026-01-28 09:15:00 | 1421.90 | STOP_HIT | 0.50 | 3.30% |
| BUY | retest2 | 2026-02-05 13:00:00 | 1453.30 | 2026-02-11 09:15:00 | 1598.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-06 11:00:00 | 1453.80 | 2026-02-11 09:15:00 | 1599.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-12 09:30:00 | 1469.90 | 2026-02-12 10:15:00 | 1475.00 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2026-02-16 09:15:00 | 1458.00 | 2026-02-18 14:15:00 | 1385.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-16 09:15:00 | 1458.00 | 2026-02-24 09:15:00 | 1312.20 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-03-11 14:15:00 | 1199.00 | 2026-03-13 10:15:00 | 1174.30 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2026-03-11 14:45:00 | 1204.10 | 2026-03-13 10:15:00 | 1174.30 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2026-03-12 10:00:00 | 1201.70 | 2026-03-13 10:15:00 | 1174.30 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2026-03-17 12:30:00 | 1170.80 | 2026-03-18 10:15:00 | 1197.90 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2026-03-17 14:30:00 | 1176.60 | 2026-03-18 10:15:00 | 1197.90 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2026-03-17 15:15:00 | 1166.60 | 2026-03-18 10:15:00 | 1197.90 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2026-04-15 09:30:00 | 1195.90 | 2026-04-23 12:15:00 | 1248.10 | STOP_HIT | 1.00 | 4.36% |
| BUY | retest2 | 2026-05-07 10:15:00 | 1302.60 | 2026-05-07 13:15:00 | 1269.00 | STOP_HIT | 1.00 | -2.58% |
