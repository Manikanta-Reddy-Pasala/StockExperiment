# Zen Technologies Ltd. (ZENTEC)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1626.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 35 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 56 |
| PARTIAL | 19 |
| TARGET_HIT | 18 |
| STOP_HIT | 42 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 75 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 44 / 31
- **Target hits / Stop hits / Partials:** 18 / 38 / 19
- **Avg / median % per leg:** 2.35% / 2.93%
- **Sum % (uncompounded):** 176.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 6 | 33.3% | 6 | 12 | 0 | 0.87% | 15.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 6 | 33.3% | 6 | 12 | 0 | 0.87% | 15.7% |
| SELL (all) | 57 | 38 | 66.7% | 12 | 26 | 19 | 2.82% | 160.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 57 | 38 | 66.7% | 12 | 26 | 19 | 2.82% | 160.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 75 | 44 | 58.7% | 18 | 38 | 19 | 2.35% | 176.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 11:15:00 | 1884.50 | 1489.35 | 1489.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 1930.30 | 1509.08 | 1499.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 1887.00 | 1900.23 | 1774.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 12:45:00 | 1883.70 | 1900.23 | 1774.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1850.30 | 1922.41 | 1841.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 14:45:00 | 1900.80 | 1893.46 | 1841.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 09:45:00 | 1900.40 | 1893.54 | 1842.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 15:00:00 | 1899.40 | 1893.11 | 1843.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 14:45:00 | 1901.00 | 1892.52 | 1844.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 1826.30 | 1890.49 | 1845.97 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 1826.30 | 1890.49 | 1845.97 | SL hit (close<static) qty=1.00 sl=1840.40 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 15:15:00 | 1457.00 | 1807.82 | 1808.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 1439.00 | 1747.15 | 1776.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 11:15:00 | 1533.40 | 1507.02 | 1579.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-17 12:00:00 | 1533.40 | 1507.02 | 1579.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1611.10 | 1509.45 | 1579.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 15:00:00 | 1540.70 | 1534.97 | 1582.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:30:00 | 1552.10 | 1535.11 | 1582.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 10:00:00 | 1551.80 | 1535.11 | 1582.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 10:30:00 | 1553.40 | 1535.34 | 1582.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 13:15:00 | 1474.49 | 1532.69 | 1578.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 13:15:00 | 1474.21 | 1532.69 | 1578.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 13:15:00 | 1475.73 | 1532.69 | 1578.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 10:15:00 | 1463.66 | 1530.93 | 1576.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-10-09 13:15:00 | 1396.89 | 1496.40 | 1546.87 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 1433.90 | 1355.54 | 1355.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 13:15:00 | 1438.00 | 1357.91 | 1356.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 15:15:00 | 1365.00 | 1367.54 | 1362.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 15:15:00 | 1365.00 | 1367.54 | 1362.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 1365.00 | 1367.54 | 1362.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:15:00 | 1340.60 | 1367.54 | 1362.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 1347.10 | 1367.34 | 1361.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 14:45:00 | 1407.80 | 1366.60 | 1361.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-23 14:30:00 | 1386.10 | 1380.19 | 1370.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 09:15:00 | 1374.30 | 1379.87 | 1369.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 09:45:00 | 1372.10 | 1379.68 | 1369.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 1362.20 | 1379.51 | 1369.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-24 10:30:00 | 1358.30 | 1379.51 | 1369.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1375.00 | 1379.32 | 1369.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 13:30:00 | 1377.80 | 1379.30 | 1369.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 10:30:00 | 1382.50 | 1378.96 | 1369.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 12:15:00 | 1386.00 | 1378.90 | 1369.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 09:15:00 | 1351.40 | 1378.92 | 1370.14 | SL hit (close<static) qty=1.00 sl=1360.20 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-21 14:45:00 | 1900.80 | 2025-07-25 09:15:00 | 1826.30 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest2 | 2025-07-22 09:45:00 | 1900.40 | 2025-07-25 09:15:00 | 1826.30 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest2 | 2025-07-22 15:00:00 | 1899.40 | 2025-07-25 09:15:00 | 1826.30 | STOP_HIT | 1.00 | -3.85% |
| BUY | retest2 | 2025-07-23 14:45:00 | 1901.00 | 2025-07-25 09:15:00 | 1826.30 | STOP_HIT | 1.00 | -3.93% |
| SELL | retest2 | 2025-09-24 15:00:00 | 1540.70 | 2025-09-26 13:15:00 | 1474.49 | PARTIAL | 0.50 | 4.30% |
| SELL | retest2 | 2025-09-25 09:30:00 | 1552.10 | 2025-09-26 13:15:00 | 1474.21 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2025-09-25 10:00:00 | 1551.80 | 2025-09-26 13:15:00 | 1475.73 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2025-09-25 10:30:00 | 1553.40 | 2025-09-29 10:15:00 | 1463.66 | PARTIAL | 0.50 | 5.78% |
| SELL | retest2 | 2025-09-24 15:00:00 | 1540.70 | 2025-10-09 13:15:00 | 1396.89 | TARGET_HIT | 0.50 | 9.33% |
| SELL | retest2 | 2025-09-25 09:30:00 | 1552.10 | 2025-10-09 13:15:00 | 1396.62 | TARGET_HIT | 0.50 | 10.02% |
| SELL | retest2 | 2025-09-25 10:00:00 | 1551.80 | 2025-10-09 13:15:00 | 1398.06 | TARGET_HIT | 0.50 | 9.91% |
| SELL | retest2 | 2025-09-25 10:30:00 | 1553.40 | 2025-10-13 12:15:00 | 1386.63 | TARGET_HIT | 0.50 | 10.74% |
| SELL | retest2 | 2025-11-14 15:15:00 | 1433.90 | 2025-11-17 09:15:00 | 1474.90 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2025-11-18 09:15:00 | 1424.90 | 2025-12-08 11:15:00 | 1362.30 | PARTIAL | 0.50 | 4.39% |
| SELL | retest2 | 2025-11-20 11:00:00 | 1434.00 | 2025-12-08 12:15:00 | 1353.65 | PARTIAL | 0.50 | 5.60% |
| SELL | retest2 | 2025-11-21 09:15:00 | 1411.50 | 2025-12-09 09:15:00 | 1340.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 12:30:00 | 1406.90 | 2025-12-09 09:15:00 | 1336.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 15:00:00 | 1405.50 | 2025-12-09 09:15:00 | 1335.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 13:15:00 | 1403.50 | 2025-12-09 09:15:00 | 1333.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 15:00:00 | 1406.00 | 2025-12-09 09:15:00 | 1335.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 09:15:00 | 1424.90 | 2025-12-12 14:15:00 | 1392.00 | STOP_HIT | 0.50 | 2.31% |
| SELL | retest2 | 2025-11-20 11:00:00 | 1434.00 | 2025-12-12 14:15:00 | 1392.00 | STOP_HIT | 0.50 | 2.93% |
| SELL | retest2 | 2025-11-21 09:15:00 | 1411.50 | 2025-12-12 14:15:00 | 1392.00 | STOP_HIT | 0.50 | 1.38% |
| SELL | retest2 | 2025-11-26 12:30:00 | 1406.90 | 2025-12-12 14:15:00 | 1392.00 | STOP_HIT | 0.50 | 1.06% |
| SELL | retest2 | 2025-11-26 15:00:00 | 1405.50 | 2025-12-12 14:15:00 | 1392.00 | STOP_HIT | 0.50 | 0.96% |
| SELL | retest2 | 2025-11-27 13:15:00 | 1403.50 | 2025-12-12 14:15:00 | 1392.00 | STOP_HIT | 0.50 | 0.82% |
| SELL | retest2 | 2025-11-27 15:00:00 | 1406.00 | 2025-12-12 14:15:00 | 1392.00 | STOP_HIT | 0.50 | 1.00% |
| SELL | retest2 | 2025-12-22 15:00:00 | 1390.50 | 2026-01-07 09:15:00 | 1325.44 | PARTIAL | 0.50 | 4.68% |
| SELL | retest2 | 2025-12-23 11:30:00 | 1395.20 | 2026-01-07 09:15:00 | 1326.58 | PARTIAL | 0.50 | 4.92% |
| SELL | retest2 | 2025-12-23 12:15:00 | 1396.40 | 2026-01-07 15:15:00 | 1320.97 | PARTIAL | 0.50 | 5.40% |
| SELL | retest2 | 2025-12-23 15:00:00 | 1389.30 | 2026-01-07 15:15:00 | 1319.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-26 14:30:00 | 1385.20 | 2026-01-07 15:15:00 | 1315.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-29 09:30:00 | 1385.90 | 2026-01-07 15:15:00 | 1316.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-30 09:15:00 | 1374.90 | 2026-01-09 09:15:00 | 1310.33 | PARTIAL | 0.50 | 4.70% |
| SELL | retest2 | 2026-01-05 09:45:00 | 1379.30 | 2026-01-09 13:15:00 | 1306.15 | PARTIAL | 0.50 | 5.30% |
| SELL | retest2 | 2025-12-22 15:00:00 | 1390.50 | 2026-01-12 09:15:00 | 1251.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-23 11:30:00 | 1395.20 | 2026-01-12 09:15:00 | 1255.68 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-23 12:15:00 | 1396.40 | 2026-01-12 09:15:00 | 1256.76 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-23 15:00:00 | 1389.30 | 2026-01-12 09:15:00 | 1250.37 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-26 14:30:00 | 1385.20 | 2026-01-12 09:15:00 | 1246.68 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-29 09:30:00 | 1385.90 | 2026-01-12 09:15:00 | 1247.31 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-30 09:15:00 | 1374.90 | 2026-01-12 09:15:00 | 1241.37 | TARGET_HIT | 0.50 | 9.71% |
| SELL | retest2 | 2026-01-05 09:45:00 | 1379.30 | 2026-01-12 11:15:00 | 1237.41 | TARGET_HIT | 0.50 | 10.29% |
| SELL | retest2 | 2026-02-01 12:15:00 | 1330.50 | 2026-03-02 09:15:00 | 1366.30 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2026-02-01 13:15:00 | 1330.80 | 2026-03-02 09:15:00 | 1366.30 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2026-02-01 13:45:00 | 1333.00 | 2026-03-02 09:15:00 | 1366.30 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2026-02-01 15:00:00 | 1330.50 | 2026-03-02 09:15:00 | 1366.30 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2026-02-02 10:45:00 | 1317.80 | 2026-03-02 09:15:00 | 1366.30 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest2 | 2026-02-02 11:45:00 | 1319.00 | 2026-03-02 09:15:00 | 1366.30 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2026-02-05 09:30:00 | 1321.20 | 2026-03-02 09:15:00 | 1366.30 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2026-02-06 09:30:00 | 1319.30 | 2026-03-02 09:15:00 | 1366.30 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2026-02-12 15:15:00 | 1343.90 | 2026-03-02 10:15:00 | 1369.90 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2026-02-17 10:30:00 | 1348.00 | 2026-03-02 10:15:00 | 1369.90 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-02-17 13:45:00 | 1346.50 | 2026-03-02 10:15:00 | 1369.90 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2026-02-20 10:00:00 | 1341.00 | 2026-03-02 10:15:00 | 1369.90 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2026-02-26 12:15:00 | 1348.00 | 2026-03-04 09:15:00 | 1426.00 | STOP_HIT | 1.00 | -5.79% |
| SELL | retest2 | 2026-02-26 15:15:00 | 1350.00 | 2026-03-04 09:15:00 | 1426.00 | STOP_HIT | 1.00 | -5.63% |
| SELL | retest2 | 2026-02-27 10:45:00 | 1347.60 | 2026-03-04 09:15:00 | 1426.00 | STOP_HIT | 1.00 | -5.82% |
| SELL | retest2 | 2026-02-27 12:30:00 | 1350.70 | 2026-03-04 09:15:00 | 1426.00 | STOP_HIT | 1.00 | -5.57% |
| SELL | retest2 | 2026-03-04 11:15:00 | 1386.40 | 2026-03-05 09:15:00 | 1432.20 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2026-03-04 12:00:00 | 1385.10 | 2026-03-05 09:15:00 | 1432.20 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2026-03-16 14:45:00 | 1407.80 | 2026-03-27 09:15:00 | 1351.40 | STOP_HIT | 1.00 | -4.01% |
| BUY | retest2 | 2026-03-23 14:30:00 | 1386.10 | 2026-03-27 09:15:00 | 1351.40 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2026-03-24 09:15:00 | 1374.30 | 2026-03-27 09:15:00 | 1351.40 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2026-03-24 09:45:00 | 1372.10 | 2026-03-30 09:15:00 | 1321.90 | STOP_HIT | 1.00 | -3.66% |
| BUY | retest2 | 2026-03-24 13:30:00 | 1377.80 | 2026-03-30 09:15:00 | 1321.90 | STOP_HIT | 1.00 | -4.06% |
| BUY | retest2 | 2026-03-25 10:30:00 | 1382.50 | 2026-03-30 09:15:00 | 1321.90 | STOP_HIT | 1.00 | -4.38% |
| BUY | retest2 | 2026-03-25 12:15:00 | 1386.00 | 2026-03-30 09:15:00 | 1321.90 | STOP_HIT | 1.00 | -4.62% |
| BUY | retest2 | 2026-04-01 09:45:00 | 1381.90 | 2026-04-01 10:15:00 | 1329.50 | STOP_HIT | 1.00 | -3.79% |
| BUY | retest2 | 2026-04-06 10:30:00 | 1374.70 | 2026-04-09 09:15:00 | 1512.17 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 11:15:00 | 1377.00 | 2026-04-09 09:15:00 | 1514.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 12:15:00 | 1374.30 | 2026-04-09 09:15:00 | 1511.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 13:00:00 | 1375.80 | 2026-04-09 09:15:00 | 1513.38 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 11:45:00 | 1428.90 | 2026-04-09 11:15:00 | 1571.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-08 09:15:00 | 1441.30 | 2026-04-17 14:15:00 | 1585.43 | TARGET_HIT | 1.00 | 10.00% |
