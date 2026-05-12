# Dr. Lal Path Labs Ltd. (LALPATHLAB)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1655.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 148 |
| ALERT1 | 94 |
| ALERT2 | 92 |
| ALERT2_SKIP | 55 |
| ALERT3 | 287 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 128 |
| PARTIAL | 20 |
| TARGET_HIT | 6 |
| STOP_HIT | 124 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 150 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 63 / 87
- **Target hits / Stop hits / Partials:** 6 / 124 / 20
- **Avg / median % per leg:** 0.92% / -0.57%
- **Sum % (uncompounded):** 138.54%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 66 | 16 | 24.2% | 1 | 65 | 0 | -0.55% | -36.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 66 | 16 | 24.2% | 1 | 65 | 0 | -0.55% | -36.6% |
| SELL (all) | 84 | 47 | 56.0% | 5 | 59 | 20 | 2.09% | 175.1% |
| SELL @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 0 | 3 | 1 | 2.67% | 10.7% |
| SELL @ 3rd Alert (retest2) | 80 | 44 | 55.0% | 5 | 56 | 19 | 2.06% | 164.5% |
| retest1 (combined) | 4 | 3 | 75.0% | 0 | 3 | 1 | 2.67% | 10.7% |
| retest2 (combined) | 146 | 60 | 41.1% | 6 | 121 | 19 | 0.88% | 127.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 1293.22 | 1317.90 | 1319.04 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 14:15:00 | 1324.65 | 1319.88 | 1319.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 09:15:00 | 1357.13 | 1327.45 | 1323.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 09:15:00 | 1366.85 | 1369.13 | 1351.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 09:30:00 | 1364.88 | 1369.13 | 1351.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 11:15:00 | 1366.50 | 1368.10 | 1353.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:45:00 | 1371.50 | 1368.10 | 1353.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 1386.33 | 1395.65 | 1387.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:15:00 | 1401.73 | 1395.65 | 1387.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 1404.23 | 1397.36 | 1389.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 13:15:00 | 1409.95 | 1399.68 | 1392.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-12 10:15:00 | 1382.05 | 1393.71 | 1392.23 | SL hit (close<static) qty=1.00 sl=1385.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 12:15:00 | 1371.08 | 1387.54 | 1389.58 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 10:15:00 | 1415.00 | 1392.08 | 1390.26 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-06-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 10:15:00 | 1379.98 | 1389.29 | 1390.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 11:15:00 | 1362.18 | 1383.87 | 1387.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-14 15:15:00 | 1377.50 | 1377.39 | 1382.95 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-18 09:15:00 | 1369.65 | 1377.39 | 1382.95 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 1356.78 | 1350.48 | 1357.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 13:45:00 | 1350.85 | 1351.81 | 1356.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 14:30:00 | 1349.35 | 1350.98 | 1355.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 09:15:00 | 1347.40 | 1350.98 | 1355.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 09:45:00 | 1350.08 | 1350.64 | 1354.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 1350.70 | 1350.65 | 1354.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 14:15:00 | 1335.70 | 1346.95 | 1351.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 15:15:00 | 1337.03 | 1345.23 | 1350.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 14:15:00 | 1334.95 | 1342.06 | 1346.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 14:15:00 | 1347.95 | 1343.23 | 1346.29 | SL hit (close>ema400) qty=1.00 sl=1346.29 alert=retest1 |

### Cycle 6 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 1363.58 | 1347.99 | 1347.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 09:15:00 | 1394.35 | 1372.04 | 1363.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 12:15:00 | 1369.38 | 1372.83 | 1366.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 12:15:00 | 1369.38 | 1372.83 | 1366.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 1369.38 | 1372.83 | 1366.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:00:00 | 1369.38 | 1372.83 | 1366.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 1374.33 | 1373.13 | 1366.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:45:00 | 1368.65 | 1373.13 | 1366.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 1397.10 | 1410.03 | 1401.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:00:00 | 1397.10 | 1410.03 | 1401.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 1394.43 | 1406.91 | 1400.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 1394.43 | 1406.91 | 1400.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 1392.05 | 1403.94 | 1400.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:45:00 | 1390.55 | 1403.94 | 1400.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 11:15:00 | 1406.18 | 1405.99 | 1402.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 11:45:00 | 1406.88 | 1405.99 | 1402.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 12:15:00 | 1417.68 | 1408.33 | 1403.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 12:45:00 | 1411.05 | 1408.33 | 1403.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 1408.18 | 1408.96 | 1404.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 14:45:00 | 1405.78 | 1408.96 | 1404.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 1450.10 | 1444.86 | 1435.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:30:00 | 1441.05 | 1444.86 | 1435.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 1451.35 | 1461.32 | 1455.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 09:45:00 | 1451.25 | 1461.32 | 1455.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 1449.30 | 1458.92 | 1455.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 1446.83 | 1458.92 | 1455.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 1474.95 | 1468.94 | 1462.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 10:30:00 | 1483.98 | 1471.41 | 1464.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 11:00:00 | 1481.30 | 1471.41 | 1464.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 09:15:00 | 1521.23 | 1474.94 | 1471.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 14:00:00 | 1485.03 | 1491.73 | 1487.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 1497.80 | 1492.94 | 1488.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 09:15:00 | 1518.08 | 1492.95 | 1488.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 09:45:00 | 1508.30 | 1497.07 | 1494.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 09:15:00 | 1523.00 | 1494.74 | 1494.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 12:30:00 | 1517.58 | 1524.09 | 1516.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 1520.15 | 1523.30 | 1516.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:30:00 | 1505.43 | 1523.30 | 1516.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 1529.38 | 1524.52 | 1517.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:30:00 | 1519.53 | 1524.52 | 1517.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 1510.38 | 1522.97 | 1518.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:30:00 | 1502.63 | 1522.97 | 1518.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 1515.65 | 1521.51 | 1517.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 11:45:00 | 1522.85 | 1522.07 | 1518.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 13:30:00 | 1521.58 | 1520.16 | 1518.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 14:30:00 | 1518.45 | 1519.41 | 1518.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-25 09:15:00 | 1489.40 | 1512.96 | 1515.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 09:15:00 | 1489.40 | 1512.96 | 1515.35 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 10:15:00 | 1529.00 | 1513.93 | 1512.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 11:15:00 | 1533.08 | 1517.76 | 1514.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 12:15:00 | 1546.80 | 1548.86 | 1539.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 13:00:00 | 1546.80 | 1548.86 | 1539.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 13:15:00 | 1538.40 | 1546.77 | 1539.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 14:00:00 | 1538.40 | 1546.77 | 1539.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 1539.80 | 1545.38 | 1539.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 15:15:00 | 1534.00 | 1545.38 | 1539.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 1534.00 | 1543.10 | 1538.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:15:00 | 1563.50 | 1543.10 | 1538.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 11:00:00 | 1566.50 | 1551.27 | 1543.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 13:15:00 | 1560.50 | 1553.62 | 1546.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:45:00 | 1560.78 | 1553.52 | 1548.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 1547.50 | 1552.20 | 1548.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 13:00:00 | 1547.50 | 1552.20 | 1548.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 13:15:00 | 1541.28 | 1550.01 | 1548.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 13:45:00 | 1541.98 | 1550.01 | 1548.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 15:15:00 | 1559.00 | 1551.81 | 1549.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:15:00 | 1549.10 | 1551.81 | 1549.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 1548.43 | 1551.14 | 1549.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 10:30:00 | 1559.08 | 1552.51 | 1550.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 13:30:00 | 1554.50 | 1552.25 | 1550.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-05 10:00:00 | 1553.93 | 1552.31 | 1550.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 11:15:00 | 1545.15 | 1549.94 | 1549.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2024-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 11:15:00 | 1545.15 | 1549.94 | 1549.99 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-05 13:15:00 | 1573.20 | 1553.73 | 1551.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-05 14:15:00 | 1579.50 | 1558.89 | 1554.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 09:15:00 | 1613.50 | 1614.57 | 1594.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 09:15:00 | 1613.50 | 1614.57 | 1594.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 1613.50 | 1614.57 | 1594.87 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 09:15:00 | 1561.03 | 1596.13 | 1596.93 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2024-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 12:15:00 | 1601.00 | 1597.27 | 1597.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 13:15:00 | 1602.25 | 1598.27 | 1597.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 13:15:00 | 1622.80 | 1624.00 | 1613.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 14:00:00 | 1622.80 | 1624.00 | 1613.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 1614.00 | 1622.46 | 1614.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 1612.13 | 1620.71 | 1614.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 1618.35 | 1620.24 | 1614.94 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2024-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 15:15:00 | 1602.50 | 1613.24 | 1613.42 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 09:15:00 | 1628.53 | 1616.30 | 1614.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 10:15:00 | 1648.33 | 1622.71 | 1617.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 13:15:00 | 1656.63 | 1656.94 | 1649.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 13:45:00 | 1655.40 | 1656.94 | 1649.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 13:15:00 | 1650.55 | 1664.70 | 1658.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 13:45:00 | 1650.88 | 1664.70 | 1658.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 14:15:00 | 1642.60 | 1660.28 | 1657.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 15:00:00 | 1642.60 | 1660.28 | 1657.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 1658.00 | 1658.04 | 1656.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 11:00:00 | 1658.00 | 1658.04 | 1656.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 11:15:00 | 1658.73 | 1658.18 | 1656.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 11:30:00 | 1645.00 | 1658.18 | 1656.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 12:15:00 | 1657.08 | 1657.96 | 1656.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 13:00:00 | 1657.08 | 1657.96 | 1656.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 13:15:00 | 1657.60 | 1657.89 | 1656.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 13:30:00 | 1657.45 | 1657.89 | 1656.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 1655.20 | 1657.35 | 1656.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 14:45:00 | 1650.78 | 1657.35 | 1656.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2024-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 15:15:00 | 1652.00 | 1656.28 | 1656.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 12:15:00 | 1647.73 | 1652.86 | 1654.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 15:15:00 | 1652.50 | 1652.28 | 1653.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 15:15:00 | 1652.50 | 1652.28 | 1653.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 1652.50 | 1652.28 | 1653.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 1659.80 | 1652.28 | 1653.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 1657.68 | 1653.36 | 1654.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:45:00 | 1663.10 | 1653.36 | 1654.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 1657.43 | 1654.17 | 1654.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:00:00 | 1657.43 | 1654.17 | 1654.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 11:15:00 | 1652.38 | 1653.81 | 1654.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:30:00 | 1656.40 | 1653.81 | 1654.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 1650.50 | 1653.15 | 1653.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 12:30:00 | 1650.63 | 1653.15 | 1653.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 13:15:00 | 1653.28 | 1653.18 | 1653.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 14:00:00 | 1653.28 | 1653.18 | 1653.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2024-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 14:15:00 | 1661.50 | 1654.84 | 1654.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 15:15:00 | 1662.50 | 1656.37 | 1655.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 12:15:00 | 1676.75 | 1676.97 | 1669.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 12:15:00 | 1676.75 | 1676.97 | 1669.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 12:15:00 | 1676.75 | 1676.97 | 1669.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 12:45:00 | 1674.80 | 1676.97 | 1669.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 1671.68 | 1677.12 | 1672.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 12:15:00 | 1683.70 | 1676.94 | 1672.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 09:45:00 | 1686.18 | 1679.36 | 1675.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 14:15:00 | 1686.85 | 1699.51 | 1700.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2024-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 14:15:00 | 1686.85 | 1699.51 | 1700.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-10 11:15:00 | 1678.15 | 1688.13 | 1691.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 14:15:00 | 1689.20 | 1686.28 | 1689.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 14:15:00 | 1689.20 | 1686.28 | 1689.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 14:15:00 | 1689.20 | 1686.28 | 1689.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 15:00:00 | 1689.20 | 1686.28 | 1689.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 15:15:00 | 1691.13 | 1687.25 | 1690.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 09:15:00 | 1696.98 | 1687.25 | 1690.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 1693.53 | 1688.51 | 1690.37 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2024-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 11:15:00 | 1699.05 | 1691.96 | 1691.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 09:15:00 | 1708.33 | 1698.78 | 1695.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 09:15:00 | 1699.50 | 1705.19 | 1701.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 09:15:00 | 1699.50 | 1705.19 | 1701.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 1699.50 | 1705.19 | 1701.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 10:00:00 | 1699.50 | 1705.19 | 1701.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 10:15:00 | 1705.00 | 1705.15 | 1701.63 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2024-09-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 14:15:00 | 1690.50 | 1698.47 | 1699.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 12:15:00 | 1687.55 | 1692.93 | 1695.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 1625.60 | 1625.35 | 1643.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 1625.60 | 1625.35 | 1643.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 1625.60 | 1625.35 | 1643.48 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2024-09-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 15:15:00 | 1643.65 | 1641.59 | 1641.47 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 12:15:00 | 1636.43 | 1640.84 | 1641.32 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2024-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 13:15:00 | 1646.00 | 1641.87 | 1641.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 09:15:00 | 1660.35 | 1646.43 | 1643.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 11:15:00 | 1668.73 | 1672.33 | 1662.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 12:00:00 | 1668.73 | 1672.33 | 1662.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 1668.33 | 1670.85 | 1663.35 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 10:15:00 | 1622.58 | 1652.97 | 1656.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 12:15:00 | 1615.80 | 1640.26 | 1650.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 14:15:00 | 1648.05 | 1637.75 | 1647.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 14:15:00 | 1648.05 | 1637.75 | 1647.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 1648.05 | 1637.75 | 1647.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:00:00 | 1648.05 | 1637.75 | 1647.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 1645.48 | 1639.30 | 1646.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 09:15:00 | 1636.80 | 1639.30 | 1646.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 10:00:00 | 1638.58 | 1639.15 | 1646.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 11:15:00 | 1638.53 | 1639.57 | 1645.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 12:45:00 | 1632.80 | 1638.26 | 1644.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 13:15:00 | 1637.28 | 1638.06 | 1643.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 13:45:00 | 1643.00 | 1638.06 | 1643.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 15:15:00 | 1637.83 | 1637.50 | 1642.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:15:00 | 1638.93 | 1637.50 | 1642.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 1645.70 | 1639.14 | 1642.52 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-30 11:15:00 | 1650.25 | 1641.50 | 1643.02 | SL hit (close>static) qty=1.00 sl=1649.98 alert=retest2 |

### Cycle 24 — BUY (started 2024-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 13:15:00 | 1656.00 | 1646.27 | 1645.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 09:15:00 | 1666.00 | 1651.47 | 1647.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 09:15:00 | 1695.08 | 1730.88 | 1715.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-07 09:15:00 | 1695.08 | 1730.88 | 1715.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 1695.08 | 1730.88 | 1715.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 10:00:00 | 1695.08 | 1730.88 | 1715.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 1717.20 | 1728.15 | 1716.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 10:30:00 | 1704.58 | 1728.15 | 1716.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 11:15:00 | 1723.20 | 1727.16 | 1716.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-07 13:15:00 | 1726.70 | 1726.17 | 1717.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-07 13:45:00 | 1729.00 | 1727.14 | 1718.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-07 14:30:00 | 1728.03 | 1727.74 | 1719.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-08 09:30:00 | 1744.65 | 1734.64 | 1724.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 1771.48 | 1789.67 | 1776.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 12:00:00 | 1771.48 | 1789.67 | 1776.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 12:15:00 | 1754.75 | 1782.69 | 1774.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 13:00:00 | 1754.75 | 1782.69 | 1774.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-11 10:15:00 | 1750.85 | 1766.39 | 1768.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2024-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 10:15:00 | 1750.85 | 1766.39 | 1768.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 09:15:00 | 1714.50 | 1748.75 | 1758.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 15:15:00 | 1702.25 | 1700.92 | 1716.58 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 09:15:00 | 1679.23 | 1700.92 | 1716.58 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 15:15:00 | 1665.90 | 1657.33 | 1671.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 09:15:00 | 1674.98 | 1657.33 | 1671.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 1682.98 | 1662.46 | 1672.60 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-18 09:15:00 | 1682.98 | 1662.46 | 1672.60 | SL hit (close>ema400) qty=1.00 sl=1672.60 alert=retest1 |

### Cycle 26 — BUY (started 2024-10-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 10:15:00 | 1681.53 | 1676.18 | 1675.79 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2024-10-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 12:15:00 | 1650.18 | 1670.58 | 1673.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 15:15:00 | 1642.10 | 1658.30 | 1666.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-22 11:15:00 | 1667.30 | 1659.63 | 1664.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-22 11:15:00 | 1667.30 | 1659.63 | 1664.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 11:15:00 | 1667.30 | 1659.63 | 1664.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 12:00:00 | 1667.30 | 1659.63 | 1664.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 12:15:00 | 1662.53 | 1660.21 | 1664.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 15:00:00 | 1651.55 | 1658.96 | 1663.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 10:00:00 | 1650.98 | 1656.81 | 1661.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 11:00:00 | 1653.88 | 1656.23 | 1660.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 09:15:00 | 1568.97 | 1628.26 | 1644.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 09:15:00 | 1568.43 | 1628.26 | 1644.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 09:15:00 | 1571.19 | 1628.26 | 1644.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-25 12:15:00 | 1562.33 | 1552.51 | 1580.52 | SL hit (close>ema200) qty=0.50 sl=1552.51 alert=retest2 |

### Cycle 28 — BUY (started 2024-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 15:15:00 | 1565.00 | 1540.58 | 1538.46 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2024-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 13:15:00 | 1535.38 | 1538.08 | 1538.24 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-04 15:15:00 | 1542.50 | 1538.39 | 1538.32 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 09:15:00 | 1535.18 | 1537.75 | 1538.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 10:15:00 | 1528.93 | 1535.98 | 1537.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 1538.85 | 1533.21 | 1535.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 1538.85 | 1533.21 | 1535.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 1538.85 | 1533.21 | 1535.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 13:45:00 | 1541.73 | 1533.21 | 1535.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 1533.98 | 1533.36 | 1535.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 1533.98 | 1533.36 | 1535.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 1531.88 | 1533.07 | 1534.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:15:00 | 1547.63 | 1533.07 | 1534.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 1551.03 | 1536.66 | 1536.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 12:15:00 | 1554.33 | 1543.30 | 1539.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 1530.65 | 1543.03 | 1541.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 1530.65 | 1543.03 | 1541.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 1530.65 | 1543.03 | 1541.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 1530.65 | 1543.03 | 1541.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 1531.50 | 1540.73 | 1540.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 1533.00 | 1540.73 | 1540.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2024-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 11:15:00 | 1527.80 | 1538.14 | 1539.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 1523.00 | 1531.18 | 1534.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 1531.48 | 1515.51 | 1522.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 1531.48 | 1515.51 | 1522.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 1531.48 | 1515.51 | 1522.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:00:00 | 1531.48 | 1515.51 | 1522.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 1523.43 | 1517.09 | 1522.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 13:30:00 | 1519.98 | 1520.41 | 1523.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 1515.58 | 1493.05 | 1492.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 1515.58 | 1493.05 | 1492.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 10:15:00 | 1518.53 | 1498.15 | 1495.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 15:15:00 | 1510.50 | 1513.79 | 1505.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-21 09:15:00 | 1516.28 | 1513.79 | 1505.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 12:15:00 | 1509.35 | 1514.94 | 1509.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 12:45:00 | 1507.93 | 1514.94 | 1509.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 13:15:00 | 1511.80 | 1514.31 | 1509.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 13:30:00 | 1507.95 | 1514.31 | 1509.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 14:15:00 | 1514.50 | 1514.35 | 1509.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 15:00:00 | 1514.50 | 1514.35 | 1509.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 1522.88 | 1515.77 | 1511.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 15:15:00 | 1530.98 | 1522.72 | 1516.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-27 12:15:00 | 1521.83 | 1536.03 | 1536.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2024-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 12:15:00 | 1521.83 | 1536.03 | 1536.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 13:15:00 | 1512.80 | 1531.38 | 1534.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 10:15:00 | 1504.50 | 1499.46 | 1510.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-29 11:00:00 | 1504.50 | 1499.46 | 1510.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 1508.38 | 1502.44 | 1507.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:45:00 | 1510.75 | 1502.44 | 1507.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 1502.25 | 1502.40 | 1506.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 09:15:00 | 1488.00 | 1507.16 | 1507.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-06 09:15:00 | 1513.30 | 1496.12 | 1494.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 09:15:00 | 1513.30 | 1496.12 | 1494.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 10:15:00 | 1528.33 | 1502.56 | 1497.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 15:15:00 | 1575.73 | 1576.64 | 1558.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-11 09:15:00 | 1575.05 | 1576.64 | 1558.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 1560.00 | 1572.94 | 1566.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 09:30:00 | 1559.00 | 1572.94 | 1566.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 1542.58 | 1566.87 | 1564.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:30:00 | 1548.60 | 1566.87 | 1564.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2024-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 11:15:00 | 1531.00 | 1559.70 | 1561.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 12:15:00 | 1526.00 | 1552.96 | 1558.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 15:15:00 | 1499.45 | 1493.77 | 1508.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-17 09:15:00 | 1501.90 | 1493.77 | 1508.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 1519.00 | 1498.82 | 1509.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:00:00 | 1519.00 | 1498.82 | 1509.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 1510.90 | 1501.23 | 1509.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:30:00 | 1518.10 | 1501.23 | 1509.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 11:15:00 | 1514.83 | 1503.95 | 1510.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:00:00 | 1514.83 | 1503.95 | 1510.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 1513.10 | 1505.78 | 1510.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:30:00 | 1515.00 | 1505.78 | 1510.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 1512.50 | 1507.13 | 1510.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:00:00 | 1512.50 | 1507.13 | 1510.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 1514.00 | 1508.50 | 1510.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 15:00:00 | 1514.00 | 1508.50 | 1510.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 15:15:00 | 1511.85 | 1509.17 | 1511.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:15:00 | 1516.50 | 1509.17 | 1511.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 1520.43 | 1511.42 | 1511.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 10:00:00 | 1520.43 | 1511.42 | 1511.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2024-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 10:15:00 | 1531.83 | 1515.50 | 1513.68 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2024-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 11:15:00 | 1509.23 | 1513.43 | 1513.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 11:15:00 | 1496.78 | 1507.96 | 1510.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 10:15:00 | 1501.93 | 1492.41 | 1499.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-23 10:15:00 | 1501.93 | 1492.41 | 1499.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 1501.93 | 1492.41 | 1499.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:00:00 | 1501.93 | 1492.41 | 1499.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 1487.38 | 1491.40 | 1498.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 12:15:00 | 1484.00 | 1491.40 | 1498.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 10:00:00 | 1485.75 | 1474.81 | 1475.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 10:15:00 | 1484.95 | 1476.55 | 1475.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 10:15:00 | 1484.95 | 1476.55 | 1475.96 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2024-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 15:15:00 | 1470.03 | 1475.08 | 1475.54 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2024-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 11:15:00 | 1490.00 | 1477.39 | 1476.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 12:15:00 | 1496.40 | 1481.19 | 1478.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 11:15:00 | 1496.50 | 1496.72 | 1488.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-01 12:00:00 | 1496.50 | 1496.72 | 1488.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 1497.83 | 1510.71 | 1499.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:30:00 | 1497.63 | 1510.71 | 1499.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 1491.28 | 1506.82 | 1499.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 11:00:00 | 1491.28 | 1506.82 | 1499.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 12:15:00 | 1491.75 | 1502.43 | 1498.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 13:00:00 | 1491.75 | 1502.43 | 1498.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 13:15:00 | 1498.93 | 1501.73 | 1498.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 15:00:00 | 1500.45 | 1501.48 | 1498.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-03 10:15:00 | 1485.48 | 1496.94 | 1497.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2025-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 10:15:00 | 1485.48 | 1496.94 | 1497.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 11:15:00 | 1472.98 | 1492.15 | 1494.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-06 09:15:00 | 1488.13 | 1480.70 | 1486.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 1488.13 | 1480.70 | 1486.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 1488.13 | 1480.70 | 1486.88 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-06 12:15:00 | 1515.23 | 1494.24 | 1492.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-07 10:15:00 | 1556.30 | 1512.49 | 1501.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 09:15:00 | 1523.00 | 1538.05 | 1522.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 09:15:00 | 1523.00 | 1538.05 | 1522.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 1523.00 | 1538.05 | 1522.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:45:00 | 1527.90 | 1538.05 | 1522.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 1507.55 | 1531.95 | 1521.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 11:00:00 | 1507.55 | 1531.95 | 1521.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 11:15:00 | 1502.05 | 1525.97 | 1519.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 11:45:00 | 1503.50 | 1525.97 | 1519.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2025-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 13:15:00 | 1497.73 | 1515.45 | 1515.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 1481.70 | 1498.91 | 1504.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-16 09:15:00 | 1396.03 | 1389.99 | 1407.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-16 14:15:00 | 1405.20 | 1396.95 | 1404.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 14:15:00 | 1405.20 | 1396.95 | 1404.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 15:00:00 | 1405.20 | 1396.95 | 1404.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 15:15:00 | 1409.45 | 1399.45 | 1404.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:15:00 | 1396.95 | 1399.45 | 1404.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 1402.98 | 1400.15 | 1404.65 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2025-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 14:15:00 | 1412.55 | 1404.95 | 1404.82 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 1389.20 | 1403.66 | 1404.45 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 10:15:00 | 1414.48 | 1404.69 | 1404.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-22 14:15:00 | 1424.13 | 1412.32 | 1408.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-23 15:15:00 | 1432.60 | 1437.97 | 1427.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 09:15:00 | 1418.50 | 1437.97 | 1427.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 1417.80 | 1433.94 | 1426.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:15:00 | 1412.60 | 1433.94 | 1426.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 1408.18 | 1428.79 | 1424.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:45:00 | 1409.15 | 1428.79 | 1424.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2025-01-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 13:15:00 | 1408.60 | 1420.25 | 1421.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 1386.05 | 1412.07 | 1417.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 1375.33 | 1361.67 | 1376.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 1375.33 | 1361.67 | 1376.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1375.33 | 1361.67 | 1376.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 1375.33 | 1361.67 | 1376.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 1378.90 | 1365.11 | 1377.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:30:00 | 1380.95 | 1365.11 | 1377.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 1389.25 | 1369.94 | 1378.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:00:00 | 1389.25 | 1369.94 | 1378.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 1382.50 | 1372.45 | 1378.64 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 1407.00 | 1383.04 | 1382.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 11:15:00 | 1412.13 | 1392.53 | 1386.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 14:15:00 | 1422.90 | 1424.26 | 1411.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 15:00:00 | 1422.90 | 1424.26 | 1411.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 1414.80 | 1421.69 | 1412.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 09:30:00 | 1416.48 | 1421.69 | 1412.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 1403.85 | 1418.12 | 1411.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 10:30:00 | 1406.60 | 1418.12 | 1411.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 1417.88 | 1418.07 | 1412.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 1413.03 | 1418.07 | 1412.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1439.00 | 1421.72 | 1414.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:15:00 | 1444.48 | 1421.72 | 1414.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 1405.05 | 1419.70 | 1415.86 | SL hit (close<static) qty=1.00 sl=1407.25 alert=retest2 |

### Cycle 51 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 1401.20 | 1412.37 | 1412.97 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2025-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 14:15:00 | 1415.25 | 1411.77 | 1411.54 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 09:15:00 | 1399.03 | 1409.74 | 1410.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 09:15:00 | 1394.28 | 1399.69 | 1404.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 11:15:00 | 1401.35 | 1399.74 | 1403.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 11:15:00 | 1401.35 | 1399.74 | 1403.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 1401.35 | 1399.74 | 1403.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:45:00 | 1403.05 | 1399.74 | 1403.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 1400.48 | 1399.89 | 1403.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:30:00 | 1406.00 | 1399.89 | 1403.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 1400.63 | 1400.04 | 1403.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:30:00 | 1403.73 | 1400.04 | 1403.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 1404.45 | 1400.92 | 1403.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 15:00:00 | 1404.45 | 1400.92 | 1403.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 1405.00 | 1401.74 | 1403.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:15:00 | 1391.33 | 1401.74 | 1403.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1392.88 | 1399.97 | 1402.39 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2025-02-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 15:15:00 | 1407.00 | 1403.35 | 1403.09 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 1393.53 | 1401.38 | 1402.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 1387.23 | 1397.59 | 1400.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-14 15:15:00 | 1314.10 | 1314.02 | 1325.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 09:15:00 | 1309.08 | 1314.02 | 1325.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 1315.93 | 1310.23 | 1318.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 14:45:00 | 1316.88 | 1310.23 | 1318.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 1308.63 | 1309.91 | 1316.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 09:15:00 | 1292.78 | 1309.96 | 1313.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 09:15:00 | 1299.85 | 1308.76 | 1311.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 09:45:00 | 1301.05 | 1307.31 | 1310.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 10:30:00 | 1300.75 | 1305.85 | 1309.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 13:15:00 | 1311.83 | 1303.84 | 1307.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 14:00:00 | 1311.83 | 1303.84 | 1307.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 14:15:00 | 1312.55 | 1305.58 | 1307.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 09:15:00 | 1297.55 | 1306.97 | 1308.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 10:15:00 | 1228.14 | 1276.65 | 1289.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 10:15:00 | 1234.86 | 1276.65 | 1289.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 10:15:00 | 1236.00 | 1276.65 | 1289.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 10:15:00 | 1235.71 | 1276.65 | 1289.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 10:15:00 | 1232.67 | 1276.65 | 1289.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-27 11:15:00 | 1170.94 | 1207.49 | 1241.04 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 56 — BUY (started 2025-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 14:15:00 | 1193.63 | 1183.73 | 1183.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 09:15:00 | 1198.65 | 1188.22 | 1185.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 14:15:00 | 1245.78 | 1249.25 | 1235.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 15:00:00 | 1245.78 | 1249.25 | 1235.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 1258.90 | 1277.50 | 1262.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 10:00:00 | 1258.90 | 1277.50 | 1262.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 1265.40 | 1275.08 | 1262.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 15:15:00 | 1272.55 | 1268.90 | 1263.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 09:45:00 | 1271.97 | 1270.40 | 1265.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 10:30:00 | 1271.63 | 1270.85 | 1265.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 09:15:00 | 1255.47 | 1263.19 | 1263.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2025-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 09:15:00 | 1255.47 | 1263.19 | 1263.59 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2025-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 09:15:00 | 1282.68 | 1266.69 | 1264.82 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2025-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-20 11:15:00 | 1263.08 | 1274.31 | 1274.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-20 14:15:00 | 1257.63 | 1267.89 | 1271.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 10:15:00 | 1268.35 | 1265.83 | 1269.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-21 10:45:00 | 1266.90 | 1265.83 | 1269.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 1270.45 | 1263.60 | 1267.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 15:00:00 | 1270.45 | 1263.60 | 1267.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 1258.63 | 1262.61 | 1266.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:15:00 | 1250.47 | 1262.61 | 1266.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 1250.25 | 1260.14 | 1264.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 09:30:00 | 1229.93 | 1247.38 | 1252.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 09:15:00 | 1238.70 | 1237.24 | 1238.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 09:15:00 | 1249.90 | 1239.77 | 1239.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 1249.90 | 1239.77 | 1239.34 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2025-03-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 15:15:00 | 1230.00 | 1239.95 | 1240.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 10:15:00 | 1218.80 | 1234.90 | 1237.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 12:15:00 | 1235.38 | 1212.94 | 1220.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 12:15:00 | 1235.38 | 1212.94 | 1220.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 1235.38 | 1212.94 | 1220.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:00:00 | 1235.38 | 1212.94 | 1220.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 1241.55 | 1218.66 | 1221.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 14:15:00 | 1243.45 | 1218.66 | 1221.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 1248.30 | 1224.59 | 1224.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 14:15:00 | 1257.10 | 1246.11 | 1237.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 1243.08 | 1246.69 | 1239.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 1243.08 | 1246.69 | 1239.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1243.08 | 1246.69 | 1239.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:15:00 | 1240.25 | 1246.69 | 1239.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 1251.50 | 1247.65 | 1240.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 12:00:00 | 1259.05 | 1249.93 | 1242.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 14:45:00 | 1261.60 | 1253.74 | 1245.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 1222.13 | 1248.02 | 1244.76 | SL hit (close<static) qty=1.00 sl=1232.50 alert=retest2 |

### Cycle 63 — SELL (started 2025-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 11:15:00 | 1222.20 | 1238.72 | 1240.84 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2025-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 10:15:00 | 1259.38 | 1243.56 | 1241.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 11:15:00 | 1266.78 | 1248.20 | 1243.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 1364.05 | 1375.70 | 1361.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 1364.05 | 1375.70 | 1361.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1364.05 | 1375.70 | 1361.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 09:30:00 | 1364.25 | 1375.70 | 1361.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 11:15:00 | 1362.50 | 1371.20 | 1361.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 11:45:00 | 1362.65 | 1371.20 | 1361.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 12:15:00 | 1362.50 | 1369.46 | 1361.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 12:30:00 | 1363.05 | 1369.46 | 1361.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 13:15:00 | 1365.70 | 1368.71 | 1362.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 13:30:00 | 1362.50 | 1368.71 | 1362.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 14:15:00 | 1358.55 | 1366.67 | 1361.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 15:00:00 | 1358.55 | 1366.67 | 1361.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 1353.55 | 1364.05 | 1361.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:15:00 | 1354.60 | 1364.05 | 1361.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 10:15:00 | 1364.70 | 1363.32 | 1361.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 09:30:00 | 1371.10 | 1365.78 | 1363.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 11:30:00 | 1378.60 | 1376.50 | 1372.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 10:15:00 | 1373.45 | 1390.66 | 1385.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 12:00:00 | 1371.05 | 1383.44 | 1382.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 12:15:00 | 1370.75 | 1380.90 | 1381.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2025-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 12:15:00 | 1370.75 | 1380.90 | 1381.55 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2025-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 14:15:00 | 1450.00 | 1393.87 | 1387.28 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2025-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 12:15:00 | 1382.50 | 1398.50 | 1398.51 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2025-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 12:15:00 | 1401.05 | 1396.71 | 1396.41 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2025-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 14:15:00 | 1384.55 | 1395.06 | 1395.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 15:15:00 | 1375.00 | 1391.05 | 1393.87 | Break + close below crossover candle low |

### Cycle 70 — BUY (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 09:15:00 | 1415.15 | 1395.87 | 1395.81 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2025-05-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 12:15:00 | 1391.55 | 1395.01 | 1395.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-05 09:15:00 | 1374.00 | 1390.66 | 1393.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-06 09:15:00 | 1397.80 | 1384.28 | 1387.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 1397.80 | 1384.28 | 1387.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 1397.80 | 1384.28 | 1387.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:00:00 | 1397.80 | 1384.28 | 1387.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 1398.40 | 1387.11 | 1388.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:30:00 | 1401.15 | 1387.11 | 1388.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 11:15:00 | 1400.00 | 1389.69 | 1389.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 12:15:00 | 1405.80 | 1392.91 | 1390.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 11:15:00 | 1418.40 | 1422.49 | 1412.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 12:00:00 | 1418.40 | 1422.49 | 1412.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 12:15:00 | 1413.00 | 1420.60 | 1412.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 12:45:00 | 1410.50 | 1420.60 | 1412.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 1393.35 | 1415.15 | 1410.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 1393.35 | 1415.15 | 1410.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 1380.55 | 1408.23 | 1407.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 1380.55 | 1408.23 | 1407.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 1387.00 | 1403.98 | 1405.85 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 1411.80 | 1402.20 | 1401.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 1418.85 | 1407.39 | 1404.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 1404.95 | 1407.48 | 1404.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 10:15:00 | 1404.95 | 1407.48 | 1404.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 10:15:00 | 1404.95 | 1407.48 | 1404.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 10:30:00 | 1407.50 | 1407.48 | 1404.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 11:15:00 | 1388.05 | 1403.59 | 1403.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 11:30:00 | 1389.30 | 1403.59 | 1403.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2025-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 12:15:00 | 1376.00 | 1398.07 | 1400.83 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2025-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 14:15:00 | 1404.00 | 1399.86 | 1399.65 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 15:15:00 | 1397.50 | 1399.39 | 1399.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 09:15:00 | 1393.65 | 1398.24 | 1398.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 13:15:00 | 1397.75 | 1396.83 | 1397.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 13:15:00 | 1397.75 | 1396.83 | 1397.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 1397.75 | 1396.83 | 1397.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:00:00 | 1397.75 | 1396.83 | 1397.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 1396.15 | 1396.70 | 1397.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-15 15:15:00 | 1394.00 | 1396.70 | 1397.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-16 09:30:00 | 1391.05 | 1393.03 | 1395.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 14:15:00 | 1403.05 | 1391.04 | 1393.21 | SL hit (close>static) qty=1.00 sl=1398.95 alert=retest2 |

### Cycle 78 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 1397.90 | 1395.17 | 1394.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 13:15:00 | 1405.85 | 1399.77 | 1397.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-21 11:15:00 | 1406.65 | 1410.18 | 1404.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-21 12:00:00 | 1406.65 | 1410.18 | 1404.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 1413.75 | 1410.90 | 1405.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:45:00 | 1411.65 | 1410.90 | 1405.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 15:15:00 | 1419.00 | 1413.90 | 1408.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:15:00 | 1404.95 | 1413.90 | 1408.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 1418.70 | 1414.86 | 1409.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 09:30:00 | 1430.00 | 1417.08 | 1412.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 11:15:00 | 1424.90 | 1418.34 | 1413.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 12:15:00 | 1424.65 | 1419.33 | 1414.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 15:00:00 | 1427.30 | 1421.23 | 1416.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 1426.55 | 1431.76 | 1424.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:45:00 | 1425.75 | 1431.76 | 1424.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 1427.80 | 1430.97 | 1424.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 1413.45 | 1422.34 | 1423.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 09:15:00 | 1413.45 | 1422.34 | 1423.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 10:15:00 | 1405.75 | 1419.02 | 1421.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 1417.50 | 1415.13 | 1418.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 1417.50 | 1415.13 | 1418.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 1417.50 | 1415.13 | 1418.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 14:45:00 | 1407.50 | 1411.13 | 1414.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 09:30:00 | 1400.30 | 1397.12 | 1402.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 12:15:00 | 1413.55 | 1405.69 | 1405.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2025-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 12:15:00 | 1413.55 | 1405.69 | 1405.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 09:15:00 | 1421.00 | 1410.47 | 1407.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 14:15:00 | 1424.05 | 1428.84 | 1425.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 14:15:00 | 1424.05 | 1428.84 | 1425.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 1424.05 | 1428.84 | 1425.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 1424.05 | 1428.84 | 1425.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 1424.20 | 1427.91 | 1425.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:15:00 | 1430.05 | 1427.91 | 1425.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 09:30:00 | 1425.90 | 1436.34 | 1431.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 1453.10 | 1473.85 | 1475.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2025-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 09:15:00 | 1453.10 | 1473.85 | 1475.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 1445.90 | 1454.20 | 1459.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 11:15:00 | 1452.65 | 1445.59 | 1450.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 11:15:00 | 1452.65 | 1445.59 | 1450.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1452.65 | 1445.59 | 1450.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 1452.65 | 1445.59 | 1450.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 1448.45 | 1446.16 | 1450.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 1424.05 | 1450.83 | 1451.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 09:15:00 | 1400.60 | 1386.36 | 1385.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2025-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 09:15:00 | 1400.60 | 1386.36 | 1385.72 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 14:15:00 | 1384.90 | 1389.94 | 1389.98 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 15:15:00 | 1395.00 | 1390.95 | 1390.43 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 1386.95 | 1389.46 | 1389.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 09:15:00 | 1376.15 | 1385.38 | 1387.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 1394.45 | 1381.93 | 1383.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 1394.45 | 1381.93 | 1383.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 1394.45 | 1381.93 | 1383.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:00:00 | 1394.45 | 1381.93 | 1383.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 10:15:00 | 1414.00 | 1388.34 | 1386.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 14:15:00 | 1424.70 | 1403.80 | 1395.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 10:15:00 | 1417.35 | 1422.03 | 1412.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-08 11:00:00 | 1417.35 | 1422.03 | 1412.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1457.70 | 1465.18 | 1450.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:30:00 | 1458.00 | 1465.18 | 1450.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1463.90 | 1463.95 | 1456.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:30:00 | 1459.95 | 1463.95 | 1456.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 1453.00 | 1460.97 | 1457.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 14:45:00 | 1464.50 | 1461.45 | 1457.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 10:15:00 | 1441.75 | 1456.01 | 1456.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 10:15:00 | 1441.75 | 1456.01 | 1456.14 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 1460.75 | 1455.13 | 1454.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 12:15:00 | 1463.15 | 1456.73 | 1455.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 11:15:00 | 1482.85 | 1486.46 | 1478.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 12:00:00 | 1482.85 | 1486.46 | 1478.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 1500.40 | 1491.39 | 1484.33 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2025-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 14:15:00 | 1480.95 | 1486.36 | 1486.76 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 10:15:00 | 1499.25 | 1488.66 | 1487.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 11:15:00 | 1501.60 | 1491.25 | 1488.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 12:15:00 | 1504.05 | 1504.13 | 1498.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-23 12:30:00 | 1506.70 | 1504.13 | 1498.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 1525.50 | 1534.75 | 1529.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:30:00 | 1529.35 | 1534.75 | 1529.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 1522.10 | 1532.22 | 1528.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 12:00:00 | 1522.10 | 1532.22 | 1528.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2025-07-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 13:15:00 | 1508.90 | 1523.88 | 1525.10 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 1554.90 | 1523.55 | 1522.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 13:15:00 | 1572.00 | 1542.63 | 1537.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 09:15:00 | 1562.35 | 1575.77 | 1563.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 1562.35 | 1575.77 | 1563.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1562.35 | 1575.77 | 1563.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:45:00 | 1566.70 | 1575.77 | 1563.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 1570.00 | 1574.61 | 1564.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:45:00 | 1559.80 | 1574.61 | 1564.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 1573.90 | 1574.47 | 1564.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:30:00 | 1571.40 | 1574.47 | 1564.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 1567.10 | 1573.00 | 1565.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 1567.10 | 1573.00 | 1565.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 1564.55 | 1571.31 | 1565.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:30:00 | 1565.95 | 1571.31 | 1565.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1549.75 | 1567.00 | 1563.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 1549.75 | 1567.00 | 1563.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 1550.25 | 1563.65 | 1562.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 09:15:00 | 1587.80 | 1563.65 | 1562.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-19 09:15:00 | 1746.58 | 1707.75 | 1680.97 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 13:15:00 | 1691.75 | 1701.88 | 1703.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 09:15:00 | 1682.60 | 1691.08 | 1695.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 09:15:00 | 1683.85 | 1683.41 | 1688.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-26 10:00:00 | 1683.85 | 1683.41 | 1688.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 1688.00 | 1684.33 | 1688.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 11:15:00 | 1682.00 | 1684.33 | 1688.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 10:00:00 | 1675.20 | 1666.14 | 1671.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 11:15:00 | 1664.50 | 1645.02 | 1643.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 11:15:00 | 1664.50 | 1645.02 | 1643.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 09:15:00 | 1697.50 | 1666.20 | 1655.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 1655.05 | 1674.51 | 1664.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 14:15:00 | 1655.05 | 1674.51 | 1664.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 1655.05 | 1674.51 | 1664.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 1655.05 | 1674.51 | 1664.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 1655.00 | 1670.61 | 1664.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 1658.65 | 1670.61 | 1664.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1659.90 | 1668.47 | 1663.69 | EMA400 retest candle locked (from upside) |

### Cycle 95 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 1630.50 | 1657.34 | 1659.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 12:15:00 | 1624.95 | 1650.86 | 1656.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 1596.85 | 1595.72 | 1608.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 1596.85 | 1595.72 | 1608.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1596.85 | 1595.72 | 1608.22 | EMA400 retest candle locked (from downside) |

### Cycle 96 — BUY (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 11:15:00 | 1624.10 | 1611.19 | 1610.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 13:15:00 | 1635.00 | 1617.04 | 1613.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 09:15:00 | 1639.95 | 1654.97 | 1641.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 09:15:00 | 1639.95 | 1654.97 | 1641.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1639.95 | 1654.97 | 1641.85 | EMA400 retest candle locked (from upside) |

### Cycle 97 — SELL (started 2025-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 15:15:00 | 1633.50 | 1644.02 | 1644.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 12:15:00 | 1625.00 | 1639.34 | 1642.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 09:15:00 | 1638.55 | 1635.78 | 1639.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 1638.55 | 1635.78 | 1639.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1638.55 | 1635.78 | 1639.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:45:00 | 1634.65 | 1635.78 | 1639.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 1641.30 | 1636.88 | 1639.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:45:00 | 1643.20 | 1636.88 | 1639.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 1641.00 | 1637.71 | 1639.60 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2025-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 09:15:00 | 1686.05 | 1649.14 | 1644.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 14:15:00 | 1734.75 | 1678.57 | 1661.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 12:15:00 | 1690.00 | 1690.36 | 1675.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 13:00:00 | 1690.00 | 1690.36 | 1675.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 1654.65 | 1684.32 | 1677.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:00:00 | 1654.65 | 1684.32 | 1677.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 1660.65 | 1679.59 | 1675.76 | EMA400 retest candle locked (from upside) |

### Cycle 99 — SELL (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 13:15:00 | 1665.05 | 1672.28 | 1672.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 1652.00 | 1666.18 | 1669.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 15:15:00 | 1657.70 | 1654.48 | 1661.03 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 09:15:00 | 1641.25 | 1654.48 | 1661.03 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 11:15:00 | 1559.19 | 1587.17 | 1609.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 1570.40 | 1565.47 | 1578.20 | SL hit (close>ema200) qty=0.50 sl=1565.47 alert=retest1 |

### Cycle 100 — BUY (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 14:15:00 | 1601.25 | 1575.73 | 1574.98 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 1568.70 | 1573.81 | 1574.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 12:15:00 | 1562.60 | 1571.57 | 1573.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 14:15:00 | 1572.50 | 1571.02 | 1572.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 14:15:00 | 1572.50 | 1571.02 | 1572.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 1572.50 | 1571.02 | 1572.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:00:00 | 1572.50 | 1571.02 | 1572.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 1573.00 | 1571.42 | 1572.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 1559.35 | 1571.42 | 1572.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 1569.90 | 1548.23 | 1546.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 09:15:00 | 1569.90 | 1548.23 | 1546.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 11:15:00 | 1578.35 | 1558.51 | 1551.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 09:15:00 | 1577.90 | 1583.99 | 1568.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 10:00:00 | 1577.90 | 1583.99 | 1568.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 1589.80 | 1587.84 | 1576.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:45:00 | 1586.90 | 1587.84 | 1576.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1565.50 | 1582.36 | 1575.99 | EMA400 retest candle locked (from upside) |

### Cycle 103 — SELL (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 14:15:00 | 1557.30 | 1569.76 | 1571.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 15:15:00 | 1555.00 | 1566.81 | 1569.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 11:15:00 | 1579.00 | 1566.61 | 1568.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 11:15:00 | 1579.00 | 1566.61 | 1568.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 1579.00 | 1566.61 | 1568.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:00:00 | 1579.00 | 1566.61 | 1568.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2025-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 12:15:00 | 1589.85 | 1571.26 | 1570.66 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 1562.25 | 1572.15 | 1572.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 14:15:00 | 1557.25 | 1569.17 | 1571.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 11:15:00 | 1564.30 | 1561.97 | 1566.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-20 11:45:00 | 1562.35 | 1561.97 | 1566.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 1587.50 | 1567.08 | 1568.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 12:45:00 | 1589.35 | 1567.08 | 1568.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 13:15:00 | 1580.95 | 1569.85 | 1569.45 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 09:15:00 | 1542.75 | 1564.43 | 1567.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 10:15:00 | 1536.00 | 1558.75 | 1564.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 14:15:00 | 1525.20 | 1522.70 | 1536.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-24 15:00:00 | 1525.20 | 1522.70 | 1536.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 1560.80 | 1529.98 | 1537.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:45:00 | 1585.50 | 1529.98 | 1537.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 1565.00 | 1536.98 | 1539.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:30:00 | 1564.95 | 1536.98 | 1539.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 12:15:00 | 1568.75 | 1547.15 | 1544.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 14:15:00 | 1572.50 | 1554.67 | 1548.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 1558.25 | 1558.56 | 1551.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 1558.25 | 1558.56 | 1551.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1558.25 | 1558.56 | 1551.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:00:00 | 1558.25 | 1558.56 | 1551.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 1540.95 | 1554.33 | 1550.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 1540.95 | 1554.33 | 1550.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 1550.00 | 1553.47 | 1550.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 13:15:00 | 1560.00 | 1553.47 | 1550.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 1534.65 | 1546.71 | 1548.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 1534.65 | 1546.71 | 1548.03 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 10:15:00 | 1553.85 | 1548.15 | 1547.77 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 1537.50 | 1546.02 | 1546.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 12:15:00 | 1533.85 | 1540.83 | 1543.53 | Break + close below crossover candle low |

### Cycle 112 — BUY (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 13:15:00 | 1569.90 | 1546.64 | 1545.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 09:15:00 | 1616.00 | 1566.86 | 1555.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 09:15:00 | 1591.05 | 1608.10 | 1587.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 09:30:00 | 1590.00 | 1608.10 | 1587.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 1573.90 | 1601.26 | 1586.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:00:00 | 1573.90 | 1601.26 | 1586.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 1582.10 | 1597.43 | 1586.29 | EMA400 retest candle locked (from upside) |

### Cycle 113 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 1560.30 | 1579.43 | 1580.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 10:15:00 | 1551.35 | 1573.82 | 1577.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 1573.70 | 1571.59 | 1575.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 12:15:00 | 1573.70 | 1571.59 | 1575.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 1573.70 | 1571.59 | 1575.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:45:00 | 1575.15 | 1571.59 | 1575.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 1575.00 | 1572.27 | 1575.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:30:00 | 1575.90 | 1572.27 | 1575.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 1546.30 | 1567.08 | 1573.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 14:30:00 | 1576.15 | 1567.08 | 1573.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1548.80 | 1546.98 | 1554.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 1548.80 | 1546.98 | 1554.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 1561.25 | 1549.84 | 1555.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:45:00 | 1564.65 | 1549.84 | 1555.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 1555.70 | 1551.01 | 1555.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:45:00 | 1546.50 | 1550.60 | 1554.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 10:15:00 | 1571.70 | 1545.81 | 1543.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 10:15:00 | 1571.70 | 1545.81 | 1543.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 11:15:00 | 1578.95 | 1552.44 | 1546.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 09:15:00 | 1571.15 | 1573.66 | 1566.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 1571.15 | 1573.66 | 1566.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1571.15 | 1573.66 | 1566.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 15:00:00 | 1580.00 | 1573.88 | 1569.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 10:15:00 | 1577.90 | 1575.21 | 1570.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 10:45:00 | 1578.85 | 1575.67 | 1571.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 11:15:00 | 1578.05 | 1575.67 | 1571.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1585.60 | 1603.51 | 1595.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:00:00 | 1585.60 | 1603.51 | 1595.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1568.15 | 1596.44 | 1593.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:00:00 | 1568.15 | 1596.44 | 1593.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 1565.05 | 1590.16 | 1590.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 11:15:00 | 1565.05 | 1590.16 | 1590.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 1549.35 | 1570.24 | 1579.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 13:15:00 | 1577.65 | 1567.22 | 1574.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 13:15:00 | 1577.65 | 1567.22 | 1574.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 1577.65 | 1567.22 | 1574.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 13:45:00 | 1575.60 | 1567.22 | 1574.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 1571.50 | 1568.08 | 1574.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 1562.20 | 1568.36 | 1573.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 11:15:00 | 1564.80 | 1567.52 | 1572.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 13:45:00 | 1566.60 | 1568.01 | 1571.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 15:15:00 | 1565.90 | 1557.86 | 1562.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 1565.90 | 1559.47 | 1562.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 09:15:00 | 1555.50 | 1559.47 | 1562.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 11:30:00 | 1555.55 | 1557.23 | 1560.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:15:00 | 1555.70 | 1553.36 | 1556.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:00:00 | 1555.00 | 1556.73 | 1557.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1555.00 | 1556.39 | 1557.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:30:00 | 1556.50 | 1556.39 | 1557.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 1557.20 | 1556.55 | 1557.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 15:00:00 | 1557.20 | 1556.55 | 1557.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 1560.00 | 1557.24 | 1557.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:15:00 | 1553.65 | 1557.24 | 1557.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1542.50 | 1554.29 | 1556.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 10:15:00 | 1531.15 | 1554.29 | 1556.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:15:00 | 1484.09 | 1512.98 | 1520.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:15:00 | 1486.56 | 1512.98 | 1520.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:15:00 | 1488.27 | 1512.98 | 1520.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:15:00 | 1487.61 | 1512.98 | 1520.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 1505.80 | 1504.82 | 1513.18 | SL hit (close>ema200) qty=0.50 sl=1504.82 alert=retest2 |

### Cycle 116 — BUY (started 2025-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 13:15:00 | 1416.20 | 1400.37 | 1398.57 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 12:15:00 | 1404.10 | 1405.41 | 1405.50 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 13:15:00 | 1420.10 | 1408.35 | 1406.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 14:15:00 | 1434.90 | 1419.08 | 1413.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 09:15:00 | 1484.90 | 1493.57 | 1477.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 10:15:00 | 1478.70 | 1493.57 | 1477.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 1470.60 | 1488.97 | 1476.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:45:00 | 1471.10 | 1488.97 | 1476.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 1471.10 | 1485.40 | 1476.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:00:00 | 1471.10 | 1485.40 | 1476.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 1476.90 | 1478.07 | 1474.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:45:00 | 1464.50 | 1478.07 | 1474.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1480.00 | 1477.98 | 1475.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 1474.00 | 1477.98 | 1475.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 1485.00 | 1491.58 | 1486.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:00:00 | 1485.00 | 1491.58 | 1486.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 1486.00 | 1490.47 | 1486.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 13:15:00 | 1481.90 | 1490.47 | 1486.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 1476.60 | 1487.69 | 1485.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 13:45:00 | 1473.50 | 1487.69 | 1485.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 1477.20 | 1485.59 | 1484.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 15:00:00 | 1477.20 | 1485.59 | 1484.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 09:15:00 | 1469.60 | 1481.50 | 1482.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 10:15:00 | 1455.30 | 1476.26 | 1480.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 13:15:00 | 1470.00 | 1468.61 | 1475.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 13:30:00 | 1475.20 | 1468.61 | 1475.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 1477.10 | 1470.31 | 1475.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:45:00 | 1476.20 | 1470.31 | 1475.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 1470.00 | 1470.25 | 1474.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 1457.10 | 1470.25 | 1474.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:00:00 | 1456.20 | 1467.44 | 1473.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 1384.24 | 1412.88 | 1431.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 12:15:00 | 1383.39 | 1406.92 | 1426.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 10:15:00 | 1405.50 | 1396.03 | 1412.67 | SL hit (close>ema200) qty=0.50 sl=1396.03 alert=retest2 |

### Cycle 120 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 1431.10 | 1415.11 | 1414.75 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2026-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 15:15:00 | 1408.00 | 1414.34 | 1414.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 09:15:00 | 1398.40 | 1411.16 | 1413.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 15:15:00 | 1404.30 | 1400.87 | 1406.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 15:15:00 | 1404.30 | 1400.87 | 1406.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 1404.30 | 1400.87 | 1406.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 1388.00 | 1400.87 | 1406.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1386.80 | 1398.06 | 1404.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 13:15:00 | 1380.10 | 1394.56 | 1401.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 14:00:00 | 1380.00 | 1391.65 | 1399.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 1366.00 | 1390.31 | 1397.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 15:00:00 | 1374.80 | 1376.93 | 1386.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 1359.20 | 1372.95 | 1382.72 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-22 10:15:00 | 1398.50 | 1385.55 | 1384.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — BUY (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 10:15:00 | 1398.50 | 1385.55 | 1384.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 12:15:00 | 1414.30 | 1393.97 | 1388.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 1400.20 | 1403.09 | 1395.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 1400.20 | 1403.09 | 1395.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 1400.20 | 1403.09 | 1395.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:15:00 | 1395.90 | 1403.09 | 1395.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 1403.80 | 1403.24 | 1396.17 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 1377.40 | 1390.35 | 1391.73 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 11:15:00 | 1405.40 | 1392.15 | 1391.87 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 11:15:00 | 1391.60 | 1392.95 | 1392.97 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 1393.70 | 1393.01 | 1392.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 1403.70 | 1395.15 | 1393.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 1387.20 | 1394.35 | 1393.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 1387.20 | 1394.35 | 1393.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 1387.20 | 1394.35 | 1393.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 1387.20 | 1394.35 | 1393.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 1399.60 | 1395.40 | 1394.38 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 12:15:00 | 1383.30 | 1392.26 | 1393.08 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 1395.60 | 1393.47 | 1393.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 10:15:00 | 1406.40 | 1396.06 | 1394.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 15:15:00 | 1398.90 | 1401.06 | 1398.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 15:15:00 | 1398.90 | 1401.06 | 1398.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 1398.90 | 1401.06 | 1398.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 1382.80 | 1401.06 | 1398.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 1382.10 | 1397.27 | 1396.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 11:00:00 | 1413.50 | 1400.52 | 1398.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 13:30:00 | 1405.50 | 1407.22 | 1406.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 14:30:00 | 1406.90 | 1409.10 | 1407.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 10:00:00 | 1406.30 | 1411.10 | 1408.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 10:15:00 | 1402.20 | 1409.32 | 1408.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-03 10:30:00 | 1400.00 | 1409.32 | 1408.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 11:15:00 | 1411.10 | 1409.68 | 1408.42 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-03 14:15:00 | 1405.10 | 1407.35 | 1407.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-03 14:15:00 | 1405.10 | 1407.35 | 1407.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 09:15:00 | 1398.50 | 1404.89 | 1406.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 11:15:00 | 1404.10 | 1403.63 | 1405.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-04 12:00:00 | 1404.10 | 1403.63 | 1405.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 1407.00 | 1404.30 | 1405.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 12:45:00 | 1405.00 | 1404.30 | 1405.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 1405.10 | 1404.46 | 1405.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 15:15:00 | 1398.50 | 1405.21 | 1405.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 13:15:00 | 1414.20 | 1405.53 | 1405.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 13:15:00 | 1414.20 | 1405.53 | 1405.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 09:15:00 | 1423.80 | 1410.48 | 1407.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-09 13:15:00 | 1435.50 | 1438.58 | 1429.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-09 14:00:00 | 1435.50 | 1438.58 | 1429.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 1447.40 | 1440.34 | 1430.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 14:45:00 | 1429.80 | 1440.34 | 1430.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 1426.50 | 1437.58 | 1430.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 09:45:00 | 1452.50 | 1438.76 | 1431.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 10:45:00 | 1458.70 | 1442.79 | 1434.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 14:45:00 | 1454.50 | 1450.09 | 1440.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 1462.10 | 1447.87 | 1440.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 1445.10 | 1451.17 | 1444.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:00:00 | 1445.10 | 1451.17 | 1444.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 1444.40 | 1449.82 | 1444.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 14:00:00 | 1444.40 | 1449.82 | 1444.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 1443.20 | 1448.49 | 1444.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 14:30:00 | 1441.70 | 1448.49 | 1444.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 1445.50 | 1447.89 | 1444.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 1445.40 | 1447.89 | 1444.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1447.30 | 1447.78 | 1445.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 10:15:00 | 1455.30 | 1447.78 | 1445.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 14:15:00 | 1432.80 | 1443.71 | 1444.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 14:15:00 | 1432.80 | 1443.71 | 1444.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 1416.00 | 1436.93 | 1440.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 1438.10 | 1417.85 | 1426.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 1438.10 | 1417.85 | 1426.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 1438.10 | 1417.85 | 1426.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:30:00 | 1449.00 | 1417.85 | 1426.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 1444.00 | 1423.08 | 1427.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:45:00 | 1442.60 | 1423.08 | 1427.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 1419.50 | 1424.62 | 1427.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:45:00 | 1420.20 | 1424.62 | 1427.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 1422.70 | 1424.24 | 1427.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 1411.60 | 1424.24 | 1427.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 10:15:00 | 1436.00 | 1426.38 | 1427.51 | SL hit (close>static) qty=1.00 sl=1434.90 alert=retest2 |

### Cycle 132 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 1446.40 | 1431.26 | 1429.59 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 11:15:00 | 1422.10 | 1429.71 | 1430.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 13:15:00 | 1406.00 | 1423.59 | 1427.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-19 11:15:00 | 1418.50 | 1416.70 | 1421.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 11:15:00 | 1418.50 | 1416.70 | 1421.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1418.50 | 1416.70 | 1421.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:45:00 | 1424.10 | 1416.70 | 1421.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 1419.90 | 1417.34 | 1421.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 13:15:00 | 1408.20 | 1417.34 | 1421.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 1422.20 | 1406.23 | 1404.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 1422.20 | 1406.23 | 1404.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 15:15:00 | 1429.50 | 1419.62 | 1412.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 09:15:00 | 1412.00 | 1418.10 | 1412.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 1412.00 | 1418.10 | 1412.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 1412.00 | 1418.10 | 1412.65 | EMA400 retest candle locked (from upside) |

### Cycle 135 — SELL (started 2026-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 12:15:00 | 1401.10 | 1409.74 | 1409.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 1393.70 | 1404.73 | 1407.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 12:15:00 | 1407.30 | 1401.56 | 1404.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 12:15:00 | 1407.30 | 1401.56 | 1404.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 1407.30 | 1401.56 | 1404.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:45:00 | 1403.70 | 1401.56 | 1404.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 1400.00 | 1401.25 | 1404.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 1390.30 | 1400.97 | 1403.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 10:15:00 | 1414.70 | 1404.52 | 1404.88 | SL hit (close>static) qty=1.00 sl=1408.00 alert=retest2 |

### Cycle 136 — BUY (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 11:15:00 | 1409.40 | 1405.50 | 1405.29 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 1379.30 | 1401.50 | 1403.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 1361.50 | 1389.58 | 1397.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 09:15:00 | 1356.60 | 1345.47 | 1360.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 09:15:00 | 1356.60 | 1345.47 | 1360.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1356.60 | 1345.47 | 1360.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:45:00 | 1355.10 | 1345.47 | 1360.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 1355.40 | 1347.46 | 1359.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:45:00 | 1357.90 | 1347.46 | 1359.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 1359.00 | 1352.44 | 1358.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 14:45:00 | 1360.00 | 1352.44 | 1358.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 1354.60 | 1352.87 | 1358.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 1332.10 | 1352.87 | 1358.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 15:15:00 | 1373.00 | 1357.73 | 1357.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — BUY (started 2026-03-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 15:15:00 | 1373.00 | 1357.73 | 1357.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 09:15:00 | 1379.70 | 1362.12 | 1359.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 15:15:00 | 1395.80 | 1396.89 | 1385.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 15:15:00 | 1395.80 | 1396.89 | 1385.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1395.80 | 1396.89 | 1385.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 1383.90 | 1396.89 | 1385.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1364.70 | 1390.45 | 1383.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:00:00 | 1364.70 | 1390.45 | 1383.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 1376.60 | 1387.68 | 1382.67 | EMA400 retest candle locked (from upside) |

### Cycle 139 — SELL (started 2026-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 13:15:00 | 1370.00 | 1379.71 | 1379.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 1355.40 | 1374.02 | 1377.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 14:15:00 | 1336.50 | 1336.34 | 1344.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 15:00:00 | 1336.50 | 1336.34 | 1344.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1339.40 | 1336.23 | 1343.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 1340.70 | 1336.23 | 1343.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 1340.10 | 1337.00 | 1342.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:30:00 | 1345.10 | 1337.00 | 1342.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 1340.30 | 1337.66 | 1342.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:45:00 | 1344.60 | 1337.66 | 1342.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 1343.80 | 1338.89 | 1342.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 12:45:00 | 1341.40 | 1338.89 | 1342.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 13:15:00 | 1348.30 | 1340.77 | 1343.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 14:00:00 | 1348.30 | 1340.77 | 1343.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 14:15:00 | 1347.00 | 1342.02 | 1343.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 15:15:00 | 1348.90 | 1342.02 | 1343.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 1348.90 | 1343.39 | 1344.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 1330.80 | 1343.39 | 1344.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 14:15:00 | 1327.10 | 1315.73 | 1314.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 1327.10 | 1315.73 | 1314.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 1342.60 | 1324.21 | 1319.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1315.20 | 1332.44 | 1326.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1315.20 | 1332.44 | 1326.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1315.20 | 1332.44 | 1326.59 | EMA400 retest candle locked (from upside) |

### Cycle 141 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 1304.60 | 1321.02 | 1322.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 1291.00 | 1313.17 | 1316.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 12:15:00 | 1316.00 | 1310.32 | 1313.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 12:15:00 | 1316.00 | 1310.32 | 1313.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 12:15:00 | 1316.00 | 1310.32 | 1313.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 13:00:00 | 1316.00 | 1310.32 | 1313.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 13:15:00 | 1322.80 | 1312.81 | 1314.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 13:30:00 | 1326.00 | 1312.81 | 1314.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 1334.70 | 1317.19 | 1316.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 10:15:00 | 1340.00 | 1322.34 | 1319.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 13:15:00 | 1359.50 | 1360.07 | 1345.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 13:45:00 | 1358.90 | 1360.07 | 1345.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1390.00 | 1399.37 | 1391.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 1401.10 | 1399.50 | 1391.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 15:15:00 | 1377.50 | 1389.59 | 1389.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 15:15:00 | 1377.50 | 1389.59 | 1389.68 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 1423.00 | 1396.28 | 1392.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 1435.00 | 1406.98 | 1398.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 09:15:00 | 1421.00 | 1423.97 | 1411.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 09:15:00 | 1421.00 | 1423.97 | 1411.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 1421.00 | 1423.97 | 1411.73 | EMA400 retest candle locked (from upside) |

### Cycle 145 — SELL (started 2026-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 15:15:00 | 1404.40 | 1409.78 | 1410.26 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2026-04-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 13:15:00 | 1419.70 | 1411.94 | 1411.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 10:15:00 | 1428.10 | 1418.36 | 1414.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 09:15:00 | 1449.00 | 1451.65 | 1440.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 09:15:00 | 1449.00 | 1451.65 | 1440.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 1449.00 | 1451.65 | 1440.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:30:00 | 1445.90 | 1451.65 | 1440.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 1443.10 | 1449.97 | 1442.80 | EMA400 retest candle locked (from upside) |

### Cycle 147 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 1406.30 | 1434.94 | 1437.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 11:15:00 | 1402.30 | 1424.60 | 1432.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1420.00 | 1417.11 | 1424.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1420.00 | 1417.11 | 1424.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1420.00 | 1417.11 | 1424.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1425.30 | 1417.11 | 1424.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 1411.20 | 1415.06 | 1419.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 10:45:00 | 1402.60 | 1411.00 | 1415.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 14:45:00 | 1400.90 | 1402.91 | 1409.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1487.70 | 1401.87 | 1401.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 1487.70 | 1401.87 | 1401.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 10:15:00 | 1629.50 | 1447.40 | 1421.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 12:15:00 | 1541.50 | 1543.48 | 1502.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 12:45:00 | 1541.70 | 1543.48 | 1502.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 15:15:00 | 1546.10 | 1573.29 | 1560.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 1611.40 | 1573.29 | 1560.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-21 13:30:00 | 1275.25 | 2024-06-04 11:15:00 | 1293.22 | STOP_HIT | 1.00 | 1.41% |
| BUY | retest2 | 2024-05-22 09:45:00 | 1275.33 | 2024-06-04 11:15:00 | 1293.22 | STOP_HIT | 1.00 | 1.40% |
| BUY | retest2 | 2024-05-23 11:00:00 | 1276.95 | 2024-06-04 11:15:00 | 1293.22 | STOP_HIT | 1.00 | 1.27% |
| BUY | retest2 | 2024-05-23 11:30:00 | 1280.22 | 2024-06-04 11:15:00 | 1293.22 | STOP_HIT | 1.00 | 1.02% |
| BUY | retest2 | 2024-05-24 10:30:00 | 1297.70 | 2024-06-04 11:15:00 | 1293.22 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2024-05-29 09:30:00 | 1298.58 | 2024-06-04 11:15:00 | 1293.22 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-06-11 13:15:00 | 1409.95 | 2024-06-12 10:15:00 | 1382.05 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest1 | 2024-06-18 09:15:00 | 1369.65 | 2024-06-24 14:15:00 | 1347.95 | STOP_HIT | 1.00 | 1.58% |
| SELL | retest2 | 2024-06-20 13:45:00 | 1350.85 | 2024-06-25 09:15:00 | 1363.58 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-06-20 14:30:00 | 1349.35 | 2024-06-25 09:15:00 | 1363.58 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-06-21 09:15:00 | 1347.40 | 2024-06-25 09:15:00 | 1363.58 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-06-21 09:45:00 | 1350.08 | 2024-06-25 09:15:00 | 1363.58 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-06-21 14:15:00 | 1335.70 | 2024-06-25 09:15:00 | 1363.58 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2024-06-21 15:15:00 | 1337.03 | 2024-06-25 09:15:00 | 1363.58 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2024-06-24 14:15:00 | 1334.95 | 2024-06-25 09:15:00 | 1363.58 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2024-07-11 10:30:00 | 1483.98 | 2024-07-25 09:15:00 | 1489.40 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2024-07-11 11:00:00 | 1481.30 | 2024-07-25 09:15:00 | 1489.40 | STOP_HIT | 1.00 | 0.55% |
| BUY | retest2 | 2024-07-15 09:15:00 | 1521.23 | 2024-07-25 09:15:00 | 1489.40 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2024-07-16 14:00:00 | 1485.03 | 2024-07-25 09:15:00 | 1489.40 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2024-07-18 09:15:00 | 1518.08 | 2024-07-25 09:15:00 | 1489.40 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-07-19 09:45:00 | 1508.30 | 2024-07-25 09:15:00 | 1489.40 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-07-22 09:15:00 | 1523.00 | 2024-07-25 09:15:00 | 1489.40 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2024-07-23 12:30:00 | 1517.58 | 2024-07-25 09:15:00 | 1489.40 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-07-24 11:45:00 | 1522.85 | 2024-07-25 09:15:00 | 1489.40 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2024-07-24 13:30:00 | 1521.58 | 2024-07-25 09:15:00 | 1489.40 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2024-07-24 14:30:00 | 1518.45 | 2024-07-25 09:15:00 | 1489.40 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2024-07-31 09:15:00 | 1563.50 | 2024-08-05 11:15:00 | 1545.15 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2024-07-31 11:00:00 | 1566.50 | 2024-08-05 11:15:00 | 1545.15 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-07-31 13:15:00 | 1560.50 | 2024-08-05 11:15:00 | 1545.15 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-08-01 09:45:00 | 1560.78 | 2024-08-05 11:15:00 | 1545.15 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-08-02 10:30:00 | 1559.08 | 2024-08-05 11:15:00 | 1545.15 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-08-02 13:30:00 | 1554.50 | 2024-08-05 11:15:00 | 1545.15 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-08-05 10:00:00 | 1553.93 | 2024-08-05 11:15:00 | 1545.15 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-08-29 12:15:00 | 1683.70 | 2024-09-06 14:15:00 | 1686.85 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2024-08-30 09:45:00 | 1686.18 | 2024-09-06 14:15:00 | 1686.85 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2024-09-27 09:15:00 | 1636.80 | 2024-09-30 11:15:00 | 1650.25 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-09-27 10:00:00 | 1638.58 | 2024-09-30 11:15:00 | 1650.25 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-09-27 11:15:00 | 1638.53 | 2024-09-30 11:15:00 | 1650.25 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-09-27 12:45:00 | 1632.80 | 2024-09-30 11:15:00 | 1650.25 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-10-07 13:15:00 | 1726.70 | 2024-10-11 10:15:00 | 1750.85 | STOP_HIT | 1.00 | 1.40% |
| BUY | retest2 | 2024-10-07 13:45:00 | 1729.00 | 2024-10-11 10:15:00 | 1750.85 | STOP_HIT | 1.00 | 1.26% |
| BUY | retest2 | 2024-10-07 14:30:00 | 1728.03 | 2024-10-11 10:15:00 | 1750.85 | STOP_HIT | 1.00 | 1.32% |
| BUY | retest2 | 2024-10-08 09:30:00 | 1744.65 | 2024-10-11 10:15:00 | 1750.85 | STOP_HIT | 1.00 | 0.36% |
| SELL | retest1 | 2024-10-16 09:15:00 | 1679.23 | 2024-10-18 09:15:00 | 1682.98 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-10-22 15:00:00 | 1651.55 | 2024-10-24 09:15:00 | 1568.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-23 10:00:00 | 1650.98 | 2024-10-24 09:15:00 | 1568.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-23 11:00:00 | 1653.88 | 2024-10-24 09:15:00 | 1571.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 15:00:00 | 1651.55 | 2024-10-25 12:15:00 | 1562.33 | STOP_HIT | 0.50 | 5.40% |
| SELL | retest2 | 2024-10-23 10:00:00 | 1650.98 | 2024-10-25 12:15:00 | 1562.33 | STOP_HIT | 0.50 | 5.37% |
| SELL | retest2 | 2024-10-23 11:00:00 | 1653.88 | 2024-10-25 12:15:00 | 1562.33 | STOP_HIT | 0.50 | 5.54% |
| SELL | retest2 | 2024-11-12 13:30:00 | 1519.98 | 2024-11-19 09:15:00 | 1515.58 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2024-11-22 15:15:00 | 1530.98 | 2024-11-27 12:15:00 | 1521.83 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-12-03 09:15:00 | 1488.00 | 2024-12-06 09:15:00 | 1513.30 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-12-23 12:15:00 | 1484.00 | 2024-12-30 10:15:00 | 1484.95 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2024-12-27 10:00:00 | 1485.75 | 2024-12-30 10:15:00 | 1484.95 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2025-01-02 15:00:00 | 1500.45 | 2025-01-03 10:15:00 | 1485.48 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-02-01 14:15:00 | 1444.48 | 2025-02-03 09:15:00 | 1405.05 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-02-19 09:15:00 | 1292.78 | 2025-02-25 10:15:00 | 1228.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-20 09:15:00 | 1299.85 | 2025-02-25 10:15:00 | 1234.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-20 09:45:00 | 1301.05 | 2025-02-25 10:15:00 | 1236.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-20 10:30:00 | 1300.75 | 2025-02-25 10:15:00 | 1235.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-21 09:15:00 | 1297.55 | 2025-02-25 10:15:00 | 1232.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-19 09:15:00 | 1292.78 | 2025-02-27 11:15:00 | 1170.94 | TARGET_HIT | 0.50 | 9.42% |
| SELL | retest2 | 2025-02-20 09:15:00 | 1299.85 | 2025-02-27 12:15:00 | 1163.50 | TARGET_HIT | 0.50 | 10.49% |
| SELL | retest2 | 2025-02-20 09:45:00 | 1301.05 | 2025-02-27 12:15:00 | 1169.87 | TARGET_HIT | 0.50 | 10.08% |
| SELL | retest2 | 2025-02-20 10:30:00 | 1300.75 | 2025-02-27 12:15:00 | 1170.67 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-21 09:15:00 | 1297.55 | 2025-02-27 12:15:00 | 1167.80 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-03-11 15:15:00 | 1272.55 | 2025-03-13 09:15:00 | 1255.47 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-03-12 09:45:00 | 1271.97 | 2025-03-13 09:15:00 | 1255.47 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-03-12 10:30:00 | 1271.63 | 2025-03-13 09:15:00 | 1255.47 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-03-26 09:30:00 | 1229.93 | 2025-03-28 09:15:00 | 1249.90 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-03-28 09:15:00 | 1238.70 | 2025-03-28 09:15:00 | 1249.90 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-04-04 12:00:00 | 1259.05 | 2025-04-07 09:15:00 | 1222.13 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2025-04-04 14:45:00 | 1261.60 | 2025-04-07 09:15:00 | 1222.13 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2025-04-22 09:30:00 | 1371.10 | 2025-04-25 12:15:00 | 1370.75 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-04-23 11:30:00 | 1378.60 | 2025-04-25 12:15:00 | 1370.75 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-04-25 10:15:00 | 1373.45 | 2025-04-25 12:15:00 | 1370.75 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2025-04-25 12:00:00 | 1371.05 | 2025-04-25 12:15:00 | 1370.75 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-05-15 15:15:00 | 1394.00 | 2025-05-16 14:15:00 | 1403.05 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-05-16 09:30:00 | 1391.05 | 2025-05-16 14:15:00 | 1403.05 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-05-23 09:30:00 | 1430.00 | 2025-05-28 09:15:00 | 1413.45 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-05-23 11:15:00 | 1424.90 | 2025-05-28 09:15:00 | 1413.45 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-05-23 12:15:00 | 1424.65 | 2025-05-28 09:15:00 | 1413.45 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-05-23 15:00:00 | 1427.30 | 2025-05-28 09:15:00 | 1413.45 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-05-30 14:45:00 | 1407.50 | 2025-06-03 12:15:00 | 1413.55 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-06-03 09:30:00 | 1400.30 | 2025-06-03 12:15:00 | 1413.55 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-06-09 09:15:00 | 1430.05 | 2025-06-16 09:15:00 | 1453.10 | STOP_HIT | 1.00 | 1.61% |
| BUY | retest2 | 2025-06-10 09:30:00 | 1425.90 | 2025-06-16 09:15:00 | 1453.10 | STOP_HIT | 1.00 | 1.91% |
| SELL | retest2 | 2025-06-23 09:15:00 | 1424.05 | 2025-06-30 09:15:00 | 1400.60 | STOP_HIT | 1.00 | 1.65% |
| BUY | retest2 | 2025-07-11 14:45:00 | 1464.50 | 2025-07-14 10:15:00 | 1441.75 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-08-05 09:15:00 | 1587.80 | 2025-08-19 09:15:00 | 1746.58 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-26 11:15:00 | 1682.00 | 2025-09-03 11:15:00 | 1664.50 | STOP_HIT | 1.00 | 1.04% |
| SELL | retest2 | 2025-08-29 10:00:00 | 1675.20 | 2025-09-03 11:15:00 | 1664.50 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest1 | 2025-09-25 09:15:00 | 1641.25 | 2025-09-29 11:15:00 | 1559.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-09-25 09:15:00 | 1641.25 | 2025-10-01 09:15:00 | 1570.40 | STOP_HIT | 0.50 | 4.32% |
| SELL | retest2 | 2025-10-03 10:45:00 | 1565.00 | 2025-10-06 14:15:00 | 1601.25 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-10-03 13:00:00 | 1564.40 | 2025-10-06 14:15:00 | 1601.25 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-10-03 15:00:00 | 1565.00 | 2025-10-06 14:15:00 | 1601.25 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-10-06 09:45:00 | 1565.50 | 2025-10-06 14:15:00 | 1601.25 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-10-08 09:15:00 | 1559.35 | 2025-10-13 09:15:00 | 1569.90 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-10-28 13:15:00 | 1560.00 | 2025-10-29 09:15:00 | 1534.65 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-11-10 13:45:00 | 1546.50 | 2025-11-13 10:15:00 | 1571.70 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-11-17 15:00:00 | 1580.00 | 2025-11-20 11:15:00 | 1565.05 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-11-18 10:15:00 | 1577.90 | 2025-11-20 11:15:00 | 1565.05 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-11-18 10:45:00 | 1578.85 | 2025-11-20 11:15:00 | 1565.05 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-11-18 11:15:00 | 1578.05 | 2025-11-20 11:15:00 | 1565.05 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-11-24 09:15:00 | 1562.20 | 2025-12-03 11:15:00 | 1484.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-24 11:15:00 | 1564.80 | 2025-12-03 11:15:00 | 1486.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-24 13:45:00 | 1566.60 | 2025-12-03 11:15:00 | 1488.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-25 15:15:00 | 1565.90 | 2025-12-03 11:15:00 | 1487.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-24 09:15:00 | 1562.20 | 2025-12-04 09:15:00 | 1505.80 | STOP_HIT | 0.50 | 3.61% |
| SELL | retest2 | 2025-11-24 11:15:00 | 1564.80 | 2025-12-04 09:15:00 | 1505.80 | STOP_HIT | 0.50 | 3.77% |
| SELL | retest2 | 2025-11-24 13:45:00 | 1566.60 | 2025-12-04 09:15:00 | 1505.80 | STOP_HIT | 0.50 | 3.88% |
| SELL | retest2 | 2025-11-25 15:15:00 | 1565.90 | 2025-12-04 09:15:00 | 1505.80 | STOP_HIT | 0.50 | 3.84% |
| SELL | retest2 | 2025-11-26 09:15:00 | 1555.50 | 2025-12-08 15:15:00 | 1477.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 11:30:00 | 1555.55 | 2025-12-08 15:15:00 | 1477.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 10:15:00 | 1555.70 | 2025-12-08 15:15:00 | 1477.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 13:00:00 | 1555.00 | 2025-12-08 15:15:00 | 1477.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-28 10:15:00 | 1531.15 | 2025-12-09 09:15:00 | 1454.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 09:15:00 | 1555.50 | 2025-12-11 13:15:00 | 1434.85 | STOP_HIT | 0.50 | 7.76% |
| SELL | retest2 | 2025-11-26 11:30:00 | 1555.55 | 2025-12-11 13:15:00 | 1434.85 | STOP_HIT | 0.50 | 7.76% |
| SELL | retest2 | 2025-11-27 10:15:00 | 1555.70 | 2025-12-11 13:15:00 | 1434.85 | STOP_HIT | 0.50 | 7.77% |
| SELL | retest2 | 2025-11-27 13:00:00 | 1555.00 | 2025-12-11 13:15:00 | 1434.85 | STOP_HIT | 0.50 | 7.73% |
| SELL | retest2 | 2025-11-28 10:15:00 | 1531.15 | 2025-12-11 13:15:00 | 1434.85 | STOP_HIT | 0.50 | 6.29% |
| SELL | retest2 | 2026-01-08 09:15:00 | 1457.10 | 2026-01-12 11:15:00 | 1384.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:00:00 | 1456.20 | 2026-01-12 12:15:00 | 1383.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 1457.10 | 2026-01-13 10:15:00 | 1405.50 | STOP_HIT | 0.50 | 3.54% |
| SELL | retest2 | 2026-01-08 10:00:00 | 1456.20 | 2026-01-13 10:15:00 | 1405.50 | STOP_HIT | 0.50 | 3.48% |
| SELL | retest2 | 2026-01-19 13:15:00 | 1380.10 | 2026-01-22 10:15:00 | 1398.50 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2026-01-19 14:00:00 | 1380.00 | 2026-01-22 10:15:00 | 1398.50 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-01-20 09:15:00 | 1366.00 | 2026-01-22 10:15:00 | 1398.50 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2026-01-20 15:00:00 | 1374.80 | 2026-01-22 10:15:00 | 1398.50 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2026-02-01 11:00:00 | 1413.50 | 2026-02-03 14:15:00 | 1405.10 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2026-02-02 13:30:00 | 1405.50 | 2026-02-03 14:15:00 | 1405.10 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2026-02-02 14:30:00 | 1406.90 | 2026-02-03 14:15:00 | 1405.10 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2026-02-03 10:00:00 | 1406.30 | 2026-02-03 14:15:00 | 1405.10 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2026-02-04 15:15:00 | 1398.50 | 2026-02-05 13:15:00 | 1414.20 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2026-02-10 09:45:00 | 1452.50 | 2026-02-12 14:15:00 | 1432.80 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2026-02-10 10:45:00 | 1458.70 | 2026-02-12 14:15:00 | 1432.80 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2026-02-10 14:45:00 | 1454.50 | 2026-02-12 14:15:00 | 1432.80 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2026-02-11 09:15:00 | 1462.10 | 2026-02-12 14:15:00 | 1432.80 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2026-02-12 10:15:00 | 1455.30 | 2026-02-12 14:15:00 | 1432.80 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-02-17 09:15:00 | 1411.60 | 2026-02-17 10:15:00 | 1436.00 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2026-02-19 13:15:00 | 1408.20 | 2026-02-25 09:15:00 | 1422.20 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2026-03-02 09:15:00 | 1390.30 | 2026-03-02 10:15:00 | 1414.70 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1332.10 | 2026-03-09 15:15:00 | 1373.00 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2026-03-19 09:15:00 | 1330.80 | 2026-03-24 14:15:00 | 1327.10 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2026-04-13 10:45:00 | 1401.10 | 2026-04-13 15:15:00 | 1377.50 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-04-29 10:45:00 | 1402.60 | 2026-05-04 09:15:00 | 1487.70 | STOP_HIT | 1.00 | -6.07% |
| SELL | retest2 | 2026-04-29 14:45:00 | 1400.90 | 2026-05-04 09:15:00 | 1487.70 | STOP_HIT | 1.00 | -6.20% |
