# Eris Lifesciences Ltd. (ERIS)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1389.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 33 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 26 |
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 17
- **Target hits / Stop hits / Partials:** 3 / 23 / 6
- **Avg / median % per leg:** 0.35% / -0.37%
- **Sum % (uncompounded):** 11.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 3 | 30.0% | 3 | 7 | 0 | 0.94% | 9.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 3 | 30.0% | 3 | 7 | 0 | 0.94% | 9.4% |
| SELL (all) | 22 | 12 | 54.5% | 0 | 16 | 6 | 0.08% | 1.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 22 | 12 | 54.5% | 0 | 16 | 6 | 0.08% | 1.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 32 | 15 | 46.9% | 3 | 23 | 6 | 0.35% | 11.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 13:15:00 | 907.40 | 872.79 | 872.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 09:15:00 | 919.55 | 888.15 | 881.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 10:15:00 | 1010.30 | 1010.36 | 973.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 10:45:00 | 1008.25 | 1010.36 | 973.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 1253.55 | 1328.06 | 1249.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:30:00 | 1249.95 | 1328.06 | 1249.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 1303.90 | 1348.50 | 1295.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:45:00 | 1305.75 | 1348.50 | 1295.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 11:15:00 | 1290.00 | 1347.92 | 1295.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 12:00:00 | 1290.00 | 1347.92 | 1295.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 12:15:00 | 1290.50 | 1347.34 | 1295.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 13:00:00 | 1290.50 | 1347.34 | 1295.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 13:15:00 | 1293.05 | 1346.80 | 1295.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 14:15:00 | 1289.60 | 1346.80 | 1295.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 1289.90 | 1343.56 | 1295.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 12:00:00 | 1289.90 | 1343.56 | 1295.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 1288.50 | 1343.01 | 1295.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 12:30:00 | 1284.55 | 1343.01 | 1295.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 1286.90 | 1340.10 | 1294.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-24 11:00:00 | 1286.90 | 1340.10 | 1294.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 11:15:00 | 1289.15 | 1339.59 | 1294.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-29 15:15:00 | 1335.15 | 1327.21 | 1293.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-30 13:15:00 | 1300.30 | 1326.06 | 1293.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-30 14:15:00 | 1301.00 | 1325.77 | 1293.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 15:15:00 | 1282.10 | 1324.93 | 1293.58 | SL hit (close<static) qty=1.00 sl=1282.35 alert=retest2 |

### Cycle 2 — SELL (started 2025-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 11:15:00 | 1284.40 | 1360.58 | 1360.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 1253.85 | 1356.63 | 1358.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 1297.10 | 1265.49 | 1300.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 09:15:00 | 1297.10 | 1265.49 | 1300.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 1297.10 | 1265.49 | 1300.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:00:00 | 1297.10 | 1265.49 | 1300.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 1357.60 | 1266.41 | 1300.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 11:00:00 | 1357.60 | 1266.41 | 1300.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 11:15:00 | 1351.15 | 1267.25 | 1300.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 11:45:00 | 1353.75 | 1267.25 | 1300.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 1396.05 | 1273.43 | 1302.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:00:00 | 1396.05 | 1273.43 | 1302.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 1313.85 | 1303.25 | 1315.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 10:45:00 | 1314.00 | 1303.25 | 1315.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 1311.50 | 1303.33 | 1315.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:45:00 | 1316.85 | 1303.33 | 1315.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 1300.20 | 1303.20 | 1314.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 09:15:00 | 1292.15 | 1303.20 | 1314.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 13:15:00 | 1227.54 | 1298.50 | 1311.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-19 15:15:00 | 1290.25 | 1288.62 | 1304.96 | SL hit (close>ema200) qty=0.50 sl=1288.62 alert=retest2 |

### Cycle 3 — BUY (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 11:15:00 | 1376.85 | 1294.17 | 1294.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 1406.10 | 1298.48 | 1296.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 1317.05 | 1319.34 | 1307.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-04 10:00:00 | 1317.05 | 1319.34 | 1307.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 1355.60 | 1319.70 | 1308.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 11:15:00 | 1367.70 | 1319.70 | 1308.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 1268.10 | 1319.72 | 1308.40 | SL hit (close<static) qty=1.00 sl=1305.15 alert=retest2 |

### Cycle 4 — SELL (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 09:15:00 | 1647.10 | 1697.67 | 1697.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 1591.70 | 1692.80 | 1695.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 14:15:00 | 1627.00 | 1620.33 | 1646.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 15:00:00 | 1627.00 | 1620.33 | 1646.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 1616.50 | 1619.78 | 1645.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 11:45:00 | 1608.20 | 1619.84 | 1643.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 10:15:00 | 1609.00 | 1619.81 | 1643.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 13:00:00 | 1607.60 | 1618.21 | 1641.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:45:00 | 1607.10 | 1617.45 | 1640.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1594.20 | 1612.65 | 1635.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 11:00:00 | 1584.20 | 1612.13 | 1635.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:30:00 | 1584.30 | 1610.77 | 1633.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 15:15:00 | 1579.00 | 1609.31 | 1632.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 12:00:00 | 1584.20 | 1608.41 | 1631.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 13:15:00 | 1527.79 | 1603.90 | 1628.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 13:15:00 | 1528.55 | 1603.90 | 1628.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 13:15:00 | 1527.22 | 1603.90 | 1628.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 13:15:00 | 1526.74 | 1603.90 | 1628.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 12:15:00 | 1599.10 | 1592.77 | 1619.27 | SL hit (close>ema200) qty=0.50 sl=1592.77 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-15 10:15:00 | 870.30 | 2024-05-15 14:15:00 | 881.85 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-10-29 15:15:00 | 1335.15 | 2024-10-30 15:15:00 | 1282.10 | STOP_HIT | 1.00 | -3.97% |
| BUY | retest2 | 2024-10-30 13:15:00 | 1300.30 | 2024-10-30 15:15:00 | 1282.10 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-10-30 14:15:00 | 1301.00 | 2024-10-30 15:15:00 | 1282.10 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-10-31 09:15:00 | 1299.80 | 2024-11-07 15:15:00 | 1295.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2024-11-06 09:15:00 | 1323.60 | 2024-11-07 15:15:00 | 1295.00 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2024-11-06 10:45:00 | 1322.10 | 2024-11-07 15:15:00 | 1295.00 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2024-11-06 11:45:00 | 1323.00 | 2024-11-19 09:15:00 | 1429.78 | TARGET_HIT | 1.00 | 8.07% |
| BUY | retest2 | 2024-11-11 14:30:00 | 1323.75 | 2024-11-25 14:15:00 | 1456.13 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-13 09:15:00 | 1292.15 | 2025-02-14 13:15:00 | 1227.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 09:15:00 | 1292.15 | 2025-02-19 15:15:00 | 1290.25 | STOP_HIT | 0.50 | 0.15% |
| SELL | retest2 | 2025-03-06 10:15:00 | 1284.60 | 2025-03-19 12:15:00 | 1323.75 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-03-12 14:30:00 | 1280.00 | 2025-03-19 12:15:00 | 1323.75 | STOP_HIT | 1.00 | -3.42% |
| SELL | retest2 | 2025-03-19 10:00:00 | 1293.70 | 2025-03-19 12:15:00 | 1323.75 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-04-04 11:15:00 | 1367.70 | 2025-04-07 09:15:00 | 1268.10 | STOP_HIT | 1.00 | -7.28% |
| BUY | retest2 | 2025-04-11 13:00:00 | 1366.30 | 2025-04-24 09:15:00 | 1502.93 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-30 11:45:00 | 1608.20 | 2025-11-13 13:15:00 | 1527.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 10:15:00 | 1609.00 | 2025-11-13 13:15:00 | 1528.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 13:00:00 | 1607.60 | 2025-11-13 13:15:00 | 1527.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 09:45:00 | 1607.10 | 2025-11-13 13:15:00 | 1526.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 11:45:00 | 1608.20 | 2025-11-19 12:15:00 | 1599.10 | STOP_HIT | 0.50 | 0.57% |
| SELL | retest2 | 2025-10-31 10:15:00 | 1609.00 | 2025-11-19 12:15:00 | 1599.10 | STOP_HIT | 0.50 | 0.62% |
| SELL | retest2 | 2025-11-03 13:00:00 | 1607.60 | 2025-11-19 12:15:00 | 1599.10 | STOP_HIT | 0.50 | 0.53% |
| SELL | retest2 | 2025-11-04 09:45:00 | 1607.10 | 2025-11-19 12:15:00 | 1599.10 | STOP_HIT | 0.50 | 0.50% |
| SELL | retest2 | 2025-11-10 11:00:00 | 1584.20 | 2025-11-20 12:15:00 | 1650.10 | STOP_HIT | 1.00 | -4.16% |
| SELL | retest2 | 2025-11-11 09:30:00 | 1584.30 | 2025-11-20 12:15:00 | 1650.10 | STOP_HIT | 1.00 | -4.15% |
| SELL | retest2 | 2025-11-11 15:15:00 | 1579.00 | 2025-11-20 12:15:00 | 1650.10 | STOP_HIT | 1.00 | -4.50% |
| SELL | retest2 | 2025-11-12 12:00:00 | 1584.20 | 2025-11-20 12:15:00 | 1650.10 | STOP_HIT | 1.00 | -4.16% |
| SELL | retest2 | 2025-11-28 12:15:00 | 1594.60 | 2025-12-09 12:15:00 | 1640.00 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2025-11-28 15:15:00 | 1593.00 | 2025-12-09 12:15:00 | 1640.00 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2025-12-10 15:15:00 | 1593.00 | 2025-12-26 09:15:00 | 1513.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-10 15:15:00 | 1593.00 | 2026-01-05 11:15:00 | 1558.00 | STOP_HIT | 0.50 | 2.20% |
