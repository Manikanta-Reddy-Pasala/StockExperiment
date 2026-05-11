# Action Construction Equipment Ltd. (ACE)

## Backtest Summary

- **Window:** 2022-04-07 13:15:00 → 2026-05-08 15:15:00 (7050 bars)
- **Last close:** 949.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 1 |
| ALERT3 | 34 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 9 |
| ENTRY2 | 32 |
| PARTIAL | 12 |
| TARGET_HIT | 6 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 53 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 35
- **Target hits / Stop hits / Partials:** 6 / 35 / 12
- **Avg / median % per leg:** 0.27% / -2.17%
- **Sum % (uncompounded):** 14.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 1 | 4 | 1 | 0.78% | 4.7% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 1 | 4 | 1 | 0.78% | 4.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 47 | 16 | 34.0% | 5 | 31 | 11 | 0.20% | 9.6% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.32% | -17.3% |
| SELL @ 3rd Alert (retest2) | 43 | 16 | 37.2% | 5 | 27 | 11 | 0.63% | 26.9% |
| retest1 (combined) | 10 | 2 | 20.0% | 1 | 8 | 1 | -1.26% | -12.6% |
| retest2 (combined) | 43 | 16 | 37.2% | 5 | 27 | 11 | 0.63% | 26.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 12:15:00 | 1347.75 | 1425.42 | 1425.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 1314.90 | 1422.02 | 1423.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 13:15:00 | 1315.40 | 1292.54 | 1329.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-16 13:45:00 | 1317.50 | 1292.54 | 1329.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 1360.95 | 1293.82 | 1329.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 1360.95 | 1293.82 | 1329.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 1351.60 | 1294.40 | 1329.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 09:15:00 | 1336.00 | 1302.39 | 1331.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 11:15:00 | 1373.00 | 1305.16 | 1331.95 | SL hit (close>static) qty=1.00 sl=1369.05 alert=retest2 |

### Cycle 2 — BUY (started 2024-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-17 13:15:00 | 1387.60 | 1350.54 | 1350.53 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 11:15:00 | 1270.15 | 1350.60 | 1350.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 1244.20 | 1341.67 | 1346.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 15:15:00 | 1321.05 | 1310.91 | 1329.04 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-31 09:15:00 | 1292.75 | 1310.91 | 1329.04 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-31 10:00:00 | 1291.25 | 1310.72 | 1328.86 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-31 11:00:00 | 1292.45 | 1310.53 | 1328.67 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-31 13:15:00 | 1290.00 | 1310.36 | 1328.41 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 1347.45 | 1310.56 | 1328.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-31 14:15:00 | 1347.45 | 1310.56 | 1328.33 | SL hit (close>ema400) qty=1.00 sl=1328.33 alert=retest1 |

### Cycle 4 — BUY (started 2024-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 15:15:00 | 1424.00 | 1319.40 | 1319.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 09:15:00 | 1426.20 | 1320.46 | 1319.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 15:15:00 | 1332.00 | 1335.16 | 1327.91 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 09:15:00 | 1371.60 | 1335.16 | 1327.91 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 09:15:00 | 1440.18 | 1339.72 | 1330.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-12-18 09:15:00 | 1508.76 | 1348.09 | 1335.08 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 5 — SELL (started 2025-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 13:15:00 | 1248.70 | 1370.15 | 1370.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 15:15:00 | 1238.95 | 1356.39 | 1363.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 11:15:00 | 1180.25 | 1175.19 | 1234.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 12:00:00 | 1180.25 | 1175.19 | 1234.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 1221.80 | 1175.26 | 1223.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 09:30:00 | 1221.00 | 1175.26 | 1223.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 1227.20 | 1175.78 | 1223.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:30:00 | 1224.90 | 1175.78 | 1223.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 11:15:00 | 1224.05 | 1176.26 | 1223.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-20 15:15:00 | 1215.00 | 1177.73 | 1223.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-21 09:15:00 | 1247.50 | 1178.79 | 1223.45 | SL hit (close>static) qty=1.00 sl=1227.30 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 1277.00 | 1227.16 | 1227.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 14:15:00 | 1294.00 | 1229.39 | 1228.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 11:15:00 | 1246.00 | 1247.94 | 1239.83 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 09:30:00 | 1263.00 | 1247.93 | 1240.03 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 12:30:00 | 1256.20 | 1248.25 | 1240.30 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 14:30:00 | 1262.10 | 1248.46 | 1240.49 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 13:00:00 | 1256.40 | 1249.08 | 1241.00 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 1242.90 | 1249.58 | 1241.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 1242.90 | 1249.58 | 1241.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1245.40 | 1249.54 | 1241.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:15:00 | 1241.60 | 1249.54 | 1241.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1227.00 | 1249.31 | 1241.72 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 1227.00 | 1249.31 | 1241.72 | SL hit (close<ema400) qty=1.00 sl=1241.72 alert=retest1 |

### Cycle 7 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 11:15:00 | 1179.00 | 1234.98 | 1235.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 1172.00 | 1234.35 | 1234.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 09:15:00 | 1224.10 | 1223.55 | 1229.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 09:15:00 | 1224.10 | 1223.55 | 1229.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 1224.10 | 1223.55 | 1229.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 09:45:00 | 1231.00 | 1223.55 | 1229.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1225.70 | 1222.94 | 1228.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:30:00 | 1227.20 | 1222.94 | 1228.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1229.70 | 1222.74 | 1228.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:45:00 | 1233.30 | 1222.74 | 1228.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 1230.30 | 1222.82 | 1228.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:30:00 | 1232.50 | 1222.82 | 1228.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 1215.10 | 1222.54 | 1227.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 14:15:00 | 1209.00 | 1222.38 | 1227.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 14:15:00 | 1148.55 | 1200.82 | 1213.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-07-29 09:15:00 | 1088.10 | 1171.18 | 1193.97 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 8 — BUY (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 15:15:00 | 916.55 | 879.60 | 879.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 11:15:00 | 934.75 | 886.21 | 883.32 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-09-19 09:15:00 | 1336.00 | 2024-09-20 11:15:00 | 1373.00 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2024-10-04 09:15:00 | 1336.90 | 2024-10-07 15:15:00 | 1278.61 | PARTIAL | 0.50 | 4.36% |
| SELL | retest2 | 2024-10-04 11:15:00 | 1345.90 | 2024-10-08 09:15:00 | 1270.06 | PARTIAL | 0.50 | 5.64% |
| SELL | retest2 | 2024-10-04 09:15:00 | 1336.90 | 2024-10-09 09:15:00 | 1346.15 | STOP_HIT | 0.50 | -0.69% |
| SELL | retest2 | 2024-10-04 11:15:00 | 1345.90 | 2024-10-09 09:15:00 | 1346.15 | STOP_HIT | 0.50 | -0.02% |
| SELL | retest2 | 2024-10-09 09:30:00 | 1345.75 | 2024-10-16 09:15:00 | 1406.55 | STOP_HIT | 1.00 | -4.52% |
| SELL | retest2 | 2024-10-11 14:45:00 | 1341.95 | 2024-10-16 09:15:00 | 1406.55 | STOP_HIT | 1.00 | -4.81% |
| SELL | retest2 | 2024-10-14 09:30:00 | 1338.85 | 2024-10-16 09:15:00 | 1406.55 | STOP_HIT | 1.00 | -5.06% |
| SELL | retest2 | 2024-10-15 10:00:00 | 1338.20 | 2024-10-16 09:15:00 | 1406.55 | STOP_HIT | 1.00 | -5.11% |
| SELL | retest2 | 2024-10-15 14:45:00 | 1340.00 | 2024-10-16 09:15:00 | 1406.55 | STOP_HIT | 1.00 | -4.97% |
| SELL | retest1 | 2024-10-31 09:15:00 | 1292.75 | 2024-10-31 14:15:00 | 1347.45 | STOP_HIT | 1.00 | -4.23% |
| SELL | retest1 | 2024-10-31 10:00:00 | 1291.25 | 2024-10-31 14:15:00 | 1347.45 | STOP_HIT | 1.00 | -4.35% |
| SELL | retest1 | 2024-10-31 11:00:00 | 1292.45 | 2024-10-31 14:15:00 | 1347.45 | STOP_HIT | 1.00 | -4.26% |
| SELL | retest1 | 2024-10-31 13:15:00 | 1290.00 | 2024-10-31 14:15:00 | 1347.45 | STOP_HIT | 1.00 | -4.45% |
| SELL | retest2 | 2024-11-01 18:45:00 | 1342.40 | 2024-11-08 15:15:00 | 1375.00 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-11-11 09:15:00 | 1329.45 | 2024-11-13 09:15:00 | 1262.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 10:30:00 | 1339.80 | 2024-11-13 09:15:00 | 1272.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 15:00:00 | 1335.00 | 2024-11-13 09:15:00 | 1268.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 09:15:00 | 1329.45 | 2024-11-18 15:15:00 | 1205.82 | TARGET_HIT | 0.50 | 9.30% |
| SELL | retest2 | 2024-11-11 10:30:00 | 1339.80 | 2024-11-18 15:15:00 | 1201.50 | TARGET_HIT | 0.50 | 10.32% |
| SELL | retest2 | 2024-11-11 15:00:00 | 1335.00 | 2024-11-21 09:15:00 | 1196.51 | TARGET_HIT | 0.50 | 10.37% |
| SELL | retest2 | 2024-11-28 13:00:00 | 1286.15 | 2024-12-02 09:15:00 | 1327.05 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2024-11-28 14:45:00 | 1288.60 | 2024-12-02 09:15:00 | 1327.05 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2024-11-29 09:30:00 | 1286.25 | 2024-12-02 09:15:00 | 1327.05 | STOP_HIT | 1.00 | -3.17% |
| SELL | retest2 | 2024-11-29 10:00:00 | 1285.50 | 2024-12-02 09:15:00 | 1327.05 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2024-12-02 09:15:00 | 1298.85 | 2024-12-02 09:15:00 | 1327.05 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest1 | 2024-12-16 09:15:00 | 1371.60 | 2024-12-17 09:15:00 | 1440.18 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-12-16 09:15:00 | 1371.60 | 2024-12-18 09:15:00 | 1508.76 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-20 15:15:00 | 1215.00 | 2025-03-21 09:15:00 | 1247.50 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-03-25 11:00:00 | 1219.45 | 2025-03-27 12:15:00 | 1251.75 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-04-07 09:15:00 | 1203.00 | 2025-04-07 09:15:00 | 1082.70 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-08 09:30:00 | 1208.00 | 2025-04-11 12:15:00 | 1233.50 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-04-25 09:45:00 | 1202.70 | 2025-05-07 09:15:00 | 1142.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-25 15:00:00 | 1203.80 | 2025-05-07 09:15:00 | 1143.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-28 11:45:00 | 1205.70 | 2025-05-07 09:15:00 | 1145.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-28 12:15:00 | 1206.70 | 2025-05-07 09:15:00 | 1146.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 09:15:00 | 1204.00 | 2025-05-07 09:15:00 | 1143.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-25 09:45:00 | 1202.70 | 2025-05-08 10:15:00 | 1221.00 | STOP_HIT | 0.50 | -1.52% |
| SELL | retest2 | 2025-04-25 15:00:00 | 1203.80 | 2025-05-08 10:15:00 | 1221.00 | STOP_HIT | 0.50 | -1.43% |
| SELL | retest2 | 2025-04-28 11:45:00 | 1205.70 | 2025-05-08 10:15:00 | 1221.00 | STOP_HIT | 0.50 | -1.27% |
| SELL | retest2 | 2025-04-28 12:15:00 | 1206.70 | 2025-05-08 10:15:00 | 1221.00 | STOP_HIT | 0.50 | -1.19% |
| SELL | retest2 | 2025-04-30 09:15:00 | 1204.00 | 2025-05-08 10:15:00 | 1221.00 | STOP_HIT | 0.50 | -1.41% |
| SELL | retest2 | 2025-04-30 10:15:00 | 1198.70 | 2025-05-13 13:15:00 | 1229.00 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2025-05-08 13:30:00 | 1198.20 | 2025-05-13 13:15:00 | 1229.00 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-05-12 09:30:00 | 1202.00 | 2025-05-16 09:15:00 | 1257.80 | STOP_HIT | 1.00 | -4.64% |
| SELL | retest2 | 2025-05-12 14:00:00 | 1207.10 | 2025-05-16 09:15:00 | 1257.80 | STOP_HIT | 1.00 | -4.20% |
| SELL | retest2 | 2025-05-13 09:15:00 | 1200.20 | 2025-05-16 09:15:00 | 1257.80 | STOP_HIT | 1.00 | -4.80% |
| BUY | retest1 | 2025-06-09 09:30:00 | 1263.00 | 2025-06-12 13:15:00 | 1227.00 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest1 | 2025-06-09 12:30:00 | 1256.20 | 2025-06-12 13:15:00 | 1227.00 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest1 | 2025-06-09 14:30:00 | 1262.10 | 2025-06-12 13:15:00 | 1227.00 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest1 | 2025-06-10 13:00:00 | 1256.40 | 2025-06-12 13:15:00 | 1227.00 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-07-01 14:15:00 | 1209.00 | 2025-07-17 14:15:00 | 1148.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-01 14:15:00 | 1209.00 | 2025-07-29 09:15:00 | 1088.10 | TARGET_HIT | 0.50 | 10.00% |
