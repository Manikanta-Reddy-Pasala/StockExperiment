# Kajaria Ceramics Ltd. (KAJARIACER)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1105.00
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
| ALERT2 | 8 |
| ALERT2_SKIP | 0 |
| ALERT3 | 52 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 48 |
| PARTIAL | 6 |
| TARGET_HIT | 1 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 54 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 42
- **Target hits / Stop hits / Partials:** 1 / 47 / 6
- **Avg / median % per leg:** -0.71% / -1.27%
- **Sum % (uncompounded):** -38.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 1 | 3.8% | 1 | 25 | 0 | -2.05% | -53.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 26 | 1 | 3.8% | 1 | 25 | 0 | -2.05% | -53.3% |
| SELL (all) | 28 | 11 | 39.3% | 0 | 22 | 6 | 0.53% | 14.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 28 | 11 | 39.3% | 0 | 22 | 6 | 0.53% | 14.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 54 | 12 | 22.2% | 1 | 47 | 6 | -0.71% | -38.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 15:15:00 | 1279.00 | 1371.26 | 1371.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 11:15:00 | 1269.60 | 1343.15 | 1355.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-07 11:15:00 | 1302.50 | 1291.19 | 1318.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-07 11:45:00 | 1302.95 | 1291.19 | 1318.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 12:15:00 | 1299.05 | 1289.66 | 1316.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-09 12:45:00 | 1314.65 | 1289.66 | 1316.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 12:15:00 | 1296.40 | 1283.84 | 1305.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 12:30:00 | 1301.90 | 1283.84 | 1305.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 1310.30 | 1284.47 | 1305.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 10:00:00 | 1310.30 | 1284.47 | 1305.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 10:15:00 | 1312.00 | 1284.74 | 1305.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-24 11:15:00 | 1302.60 | 1284.74 | 1305.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-28 09:30:00 | 1309.40 | 1285.99 | 1305.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-28 10:00:00 | 1309.20 | 1285.99 | 1305.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-28 10:15:00 | 1323.40 | 1286.36 | 1305.64 | SL hit (close>static) qty=1.00 sl=1318.90 alert=retest2 |

### Cycle 2 — BUY (started 2023-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 10:15:00 | 1380.95 | 1319.11 | 1319.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 14:15:00 | 1387.15 | 1321.63 | 1320.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 12:15:00 | 1351.15 | 1353.62 | 1340.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-27 13:00:00 | 1351.15 | 1353.62 | 1340.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 13:15:00 | 1340.80 | 1353.17 | 1340.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 14:00:00 | 1340.80 | 1353.17 | 1340.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 14:15:00 | 1325.50 | 1352.89 | 1340.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 15:00:00 | 1325.50 | 1352.89 | 1340.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 15:15:00 | 1326.00 | 1352.63 | 1340.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 09:15:00 | 1338.00 | 1352.63 | 1340.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 10:15:00 | 1340.15 | 1352.43 | 1340.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-29 11:15:00 | 1321.05 | 1351.96 | 1340.01 | SL hit (close<static) qty=1.00 sl=1322.75 alert=retest2 |

### Cycle 3 — SELL (started 2024-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 11:15:00 | 1255.85 | 1345.23 | 1345.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 15:15:00 | 1249.65 | 1341.60 | 1343.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 14:15:00 | 1226.45 | 1221.21 | 1257.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-01 15:00:00 | 1226.45 | 1221.21 | 1257.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 09:15:00 | 1271.05 | 1221.71 | 1257.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-02 10:00:00 | 1271.05 | 1221.71 | 1257.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 10:15:00 | 1287.65 | 1222.36 | 1257.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-02 10:30:00 | 1289.50 | 1222.36 | 1257.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 12:15:00 | 1259.55 | 1229.04 | 1258.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-04 12:30:00 | 1262.00 | 1229.04 | 1258.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 13:15:00 | 1253.85 | 1229.29 | 1258.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-04 14:45:00 | 1245.80 | 1229.47 | 1258.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-05 15:15:00 | 1251.10 | 1230.74 | 1258.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-08 09:15:00 | 1262.25 | 1231.25 | 1258.02 | SL hit (close>static) qty=1.00 sl=1259.95 alert=retest2 |

### Cycle 4 — BUY (started 2024-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 14:15:00 | 1317.00 | 1238.14 | 1238.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 14:15:00 | 1369.50 | 1249.47 | 1244.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 09:15:00 | 1357.05 | 1381.63 | 1332.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 10:00:00 | 1357.05 | 1381.63 | 1332.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 1383.40 | 1421.34 | 1382.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 12:15:00 | 1389.80 | 1420.98 | 1382.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 09:45:00 | 1391.50 | 1420.15 | 1383.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 11:30:00 | 1386.50 | 1419.46 | 1383.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 15:15:00 | 1387.35 | 1418.59 | 1383.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 15:15:00 | 1387.35 | 1418.28 | 1383.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:30:00 | 1381.60 | 1417.92 | 1383.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 1383.55 | 1417.58 | 1383.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:45:00 | 1382.25 | 1417.58 | 1383.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 1380.80 | 1417.21 | 1383.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-16 11:45:00 | 1379.40 | 1417.21 | 1383.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 12:15:00 | 1381.30 | 1416.85 | 1383.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-16 13:00:00 | 1381.30 | 1416.85 | 1383.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 13:15:00 | 1380.75 | 1416.49 | 1383.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-16 13:30:00 | 1385.40 | 1416.49 | 1383.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 1379.70 | 1416.13 | 1383.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:45:00 | 1377.30 | 1416.13 | 1383.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 15:15:00 | 1381.00 | 1415.78 | 1383.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:15:00 | 1381.05 | 1415.78 | 1383.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 1378.95 | 1415.41 | 1383.13 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-19 14:15:00 | 1370.15 | 1413.35 | 1382.89 | SL hit (close<static) qty=1.00 sl=1372.40 alert=retest2 |

### Cycle 5 — SELL (started 2024-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 11:15:00 | 1277.70 | 1411.08 | 1411.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 12:15:00 | 1265.00 | 1409.63 | 1410.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 14:15:00 | 1172.80 | 1171.58 | 1218.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-01 15:00:00 | 1172.80 | 1171.58 | 1218.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 855.95 | 829.01 | 870.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 10:15:00 | 855.85 | 829.01 | 870.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-14 09:30:00 | 854.95 | 831.70 | 868.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-14 11:15:00 | 890.00 | 832.80 | 868.85 | SL hit (close>static) qty=1.00 sl=884.65 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 13:15:00 | 1047.65 | 894.90 | 894.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 09:15:00 | 1050.95 | 916.59 | 905.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-28 11:15:00 | 1223.90 | 1228.54 | 1168.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-28 12:00:00 | 1223.90 | 1228.54 | 1168.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 1204.00 | 1229.13 | 1195.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 12:45:00 | 1221.00 | 1224.74 | 1196.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 13:15:00 | 1221.90 | 1224.74 | 1196.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 11:15:00 | 1182.60 | 1221.90 | 1196.43 | SL hit (close<static) qty=1.00 sl=1186.50 alert=retest2 |

### Cycle 7 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 1115.00 | 1198.35 | 1198.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 14:15:00 | 1106.60 | 1188.43 | 1193.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 10:15:00 | 1092.50 | 1091.89 | 1127.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-15 10:45:00 | 1093.00 | 1091.89 | 1127.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 1007.60 | 952.62 | 991.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:45:00 | 1001.30 | 952.62 | 991.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 1003.55 | 953.13 | 991.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 12:30:00 | 1000.05 | 954.11 | 992.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 09:30:00 | 998.80 | 960.26 | 991.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:30:00 | 996.00 | 960.64 | 991.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 14:15:00 | 950.05 | 962.69 | 991.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 14:15:00 | 948.86 | 962.69 | 991.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 14:15:00 | 946.20 | 962.69 | 991.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 962.50 | 961.98 | 989.76 | SL hit (close>ema200) qty=0.50 sl=961.98 alert=retest2 |

### Cycle 8 — BUY (started 2026-04-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 15:15:00 | 1115.05 | 974.83 | 974.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 13:15:00 | 1127.40 | 981.76 | 978.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 13:15:00 | 1107.50 | 1109.44 | 1057.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-04 14:00:00 | 1107.50 | 1109.44 | 1057.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 1056.80 | 1107.53 | 1057.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:30:00 | 1055.80 | 1107.53 | 1057.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 1065.70 | 1107.11 | 1058.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:45:00 | 1055.10 | 1107.11 | 1058.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-11-24 11:15:00 | 1302.60 | 2023-11-28 10:15:00 | 1323.40 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2023-11-28 09:30:00 | 1309.40 | 2023-11-28 10:15:00 | 1323.40 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2023-11-28 10:00:00 | 1309.20 | 2023-11-28 10:15:00 | 1323.40 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2023-11-29 09:30:00 | 1303.25 | 2023-11-30 09:15:00 | 1320.95 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2023-12-29 09:15:00 | 1338.00 | 2023-12-29 11:15:00 | 1321.05 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2023-12-29 10:15:00 | 1340.15 | 2023-12-29 11:15:00 | 1321.05 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-01-05 13:30:00 | 1336.50 | 2024-01-24 10:15:00 | 1310.00 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-01-05 14:30:00 | 1337.50 | 2024-01-24 10:15:00 | 1310.00 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2024-01-25 09:15:00 | 1346.00 | 2024-01-30 15:15:00 | 1340.00 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-01-29 12:45:00 | 1350.40 | 2024-02-02 12:15:00 | 1344.05 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-01-29 13:15:00 | 1346.10 | 2024-02-02 12:15:00 | 1344.05 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2024-01-30 09:15:00 | 1368.70 | 2024-02-02 12:15:00 | 1344.05 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2024-01-30 11:00:00 | 1393.85 | 2024-02-05 09:15:00 | 1319.55 | STOP_HIT | 1.00 | -5.33% |
| BUY | retest2 | 2024-01-31 12:15:00 | 1397.95 | 2024-02-05 09:15:00 | 1319.55 | STOP_HIT | 1.00 | -5.61% |
| BUY | retest2 | 2024-01-31 14:15:00 | 1393.90 | 2024-02-05 09:15:00 | 1319.55 | STOP_HIT | 1.00 | -5.33% |
| BUY | retest2 | 2024-02-01 09:45:00 | 1399.30 | 2024-02-05 09:15:00 | 1319.55 | STOP_HIT | 1.00 | -5.70% |
| SELL | retest2 | 2024-04-04 14:45:00 | 1245.80 | 2024-04-08 09:15:00 | 1262.25 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-04-05 15:15:00 | 1251.10 | 2024-04-08 09:15:00 | 1262.25 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-04-08 13:30:00 | 1251.70 | 2024-04-09 11:15:00 | 1261.65 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-04-08 14:00:00 | 1250.40 | 2024-04-09 11:15:00 | 1261.65 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-04-10 09:30:00 | 1246.95 | 2024-05-03 09:15:00 | 1184.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-18 13:15:00 | 1244.85 | 2024-05-03 09:15:00 | 1182.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-10 09:30:00 | 1246.95 | 2024-05-15 09:15:00 | 1212.75 | STOP_HIT | 0.50 | 2.74% |
| SELL | retest2 | 2024-04-18 13:15:00 | 1244.85 | 2024-05-15 09:15:00 | 1212.75 | STOP_HIT | 0.50 | 2.58% |
| SELL | retest2 | 2024-05-16 13:15:00 | 1242.30 | 2024-05-17 12:15:00 | 1271.75 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2024-05-17 09:30:00 | 1248.85 | 2024-05-17 12:15:00 | 1271.75 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-05-31 14:15:00 | 1236.40 | 2024-06-04 10:15:00 | 1174.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-31 14:15:00 | 1236.40 | 2024-06-07 09:15:00 | 1245.90 | STOP_HIT | 0.50 | -0.77% |
| SELL | retest2 | 2024-06-07 14:15:00 | 1236.60 | 2024-06-10 15:15:00 | 1248.00 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2024-06-07 15:15:00 | 1237.00 | 2024-06-10 15:15:00 | 1248.00 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-06-10 09:45:00 | 1232.75 | 2024-06-10 15:15:00 | 1248.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-08-13 12:15:00 | 1389.80 | 2024-08-19 14:15:00 | 1370.15 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-08-14 09:45:00 | 1391.50 | 2024-08-19 14:15:00 | 1370.15 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-08-14 11:30:00 | 1386.50 | 2024-08-19 14:15:00 | 1370.15 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-08-14 15:15:00 | 1387.35 | 2024-08-19 14:15:00 | 1370.15 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-09-04 09:15:00 | 1425.00 | 2024-09-24 10:15:00 | 1567.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-11 10:00:00 | 1407.00 | 2024-10-16 13:15:00 | 1373.10 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2024-10-15 10:30:00 | 1396.70 | 2024-10-16 13:15:00 | 1373.10 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-10-15 11:15:00 | 1397.55 | 2024-10-16 13:15:00 | 1373.10 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-05-12 10:15:00 | 855.85 | 2025-05-14 11:15:00 | 890.00 | STOP_HIT | 1.00 | -3.99% |
| SELL | retest2 | 2025-05-14 09:30:00 | 854.95 | 2025-05-14 11:15:00 | 890.00 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest2 | 2025-09-24 12:45:00 | 1221.00 | 2025-09-26 11:15:00 | 1182.60 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2025-09-24 13:15:00 | 1221.90 | 2025-09-26 11:15:00 | 1182.60 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest2 | 2025-10-10 09:45:00 | 1234.20 | 2025-11-03 09:15:00 | 1175.50 | STOP_HIT | 1.00 | -4.76% |
| BUY | retest2 | 2025-10-15 10:00:00 | 1220.70 | 2025-11-03 09:15:00 | 1175.50 | STOP_HIT | 1.00 | -3.70% |
| BUY | retest2 | 2025-10-24 14:15:00 | 1211.00 | 2025-11-03 09:15:00 | 1175.50 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2025-10-27 11:00:00 | 1208.90 | 2025-11-03 09:15:00 | 1175.50 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2026-02-18 12:30:00 | 1000.05 | 2026-02-24 14:15:00 | 950.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 09:30:00 | 998.80 | 2026-02-24 14:15:00 | 948.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 10:30:00 | 996.00 | 2026-02-24 14:15:00 | 946.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 12:30:00 | 1000.05 | 2026-02-26 09:15:00 | 962.50 | STOP_HIT | 0.50 | 3.75% |
| SELL | retest2 | 2026-02-23 09:30:00 | 998.80 | 2026-02-26 09:15:00 | 962.50 | STOP_HIT | 0.50 | 3.63% |
| SELL | retest2 | 2026-02-23 10:30:00 | 996.00 | 2026-02-26 09:15:00 | 962.50 | STOP_HIT | 0.50 | 3.36% |
| SELL | retest2 | 2026-04-07 14:15:00 | 1000.10 | 2026-04-08 09:15:00 | 1061.25 | STOP_HIT | 1.00 | -6.11% |
