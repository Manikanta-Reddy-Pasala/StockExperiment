# Ipca Laboratories Ltd. (IPCALAB)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1554.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 90 |
| ALERT1 | 55 |
| ALERT2 | 54 |
| ALERT2_SKIP | 39 |
| ALERT3 | 158 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 84 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 94 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 88 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 60
- **Target hits / Stop hits / Partials:** 0 / 88 / 0
- **Avg / median % per leg:** -0.53% / -0.99%
- **Sum % (uncompounded):** -46.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 51 | 11 | 21.6% | 0 | 51 | 0 | -0.38% | -19.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 51 | 11 | 21.6% | 0 | 51 | 0 | -0.38% | -19.6% |
| SELL (all) | 37 | 17 | 45.9% | 0 | 37 | 0 | -0.72% | -26.8% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 4 | 0 | 1.30% | 5.2% |
| SELL @ 3rd Alert (retest2) | 33 | 13 | 39.4% | 0 | 33 | 0 | -0.97% | -32.0% |
| retest1 (combined) | 4 | 4 | 100.0% | 0 | 4 | 0 | 1.30% | 5.2% |
| retest2 (combined) | 84 | 24 | 28.6% | 0 | 84 | 0 | -0.61% | -51.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 14:15:00 | 1373.00 | 1360.63 | 1359.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1396.10 | 1367.80 | 1362.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 10:15:00 | 1404.90 | 1409.65 | 1398.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 10:45:00 | 1407.20 | 1409.65 | 1398.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 1402.60 | 1406.79 | 1399.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 12:45:00 | 1396.30 | 1406.79 | 1399.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 1402.00 | 1405.83 | 1399.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:45:00 | 1405.20 | 1405.83 | 1399.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 1397.20 | 1404.41 | 1400.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:00:00 | 1397.20 | 1404.41 | 1400.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 1391.60 | 1401.85 | 1399.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 11:00:00 | 1391.60 | 1401.85 | 1399.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 11:15:00 | 1390.50 | 1399.58 | 1398.86 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 12:15:00 | 1392.00 | 1398.06 | 1398.23 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 09:15:00 | 1452.60 | 1406.19 | 1401.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 1472.10 | 1436.95 | 1421.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 15:15:00 | 1444.00 | 1450.44 | 1436.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 15:15:00 | 1444.00 | 1450.44 | 1436.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 1444.00 | 1450.44 | 1436.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:15:00 | 1475.30 | 1450.44 | 1436.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 14:45:00 | 1476.20 | 1460.88 | 1449.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 10:00:00 | 1470.90 | 1453.67 | 1451.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 11:15:00 | 1467.50 | 1455.74 | 1452.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 1455.40 | 1457.70 | 1453.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:30:00 | 1455.00 | 1457.70 | 1453.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 1454.70 | 1457.10 | 1453.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 13:30:00 | 1449.30 | 1457.10 | 1453.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 1460.30 | 1457.74 | 1454.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 1479.50 | 1458.43 | 1455.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 10:30:00 | 1464.00 | 1460.41 | 1456.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 12:15:00 | 1450.70 | 1457.57 | 1456.03 | SL hit (close<static) qty=1.00 sl=1453.40 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 09:15:00 | 1450.40 | 1457.18 | 1457.85 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 10:15:00 | 1465.90 | 1458.92 | 1458.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 12:15:00 | 1495.50 | 1467.34 | 1462.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 14:15:00 | 1469.10 | 1481.81 | 1475.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 14:15:00 | 1469.10 | 1481.81 | 1475.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 1469.10 | 1481.81 | 1475.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 1469.10 | 1481.81 | 1475.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 1477.90 | 1481.03 | 1475.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 1455.00 | 1481.03 | 1475.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1442.50 | 1473.32 | 1472.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:45:00 | 1444.50 | 1473.32 | 1472.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 10:15:00 | 1423.20 | 1463.30 | 1468.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 11:15:00 | 1410.60 | 1452.76 | 1463.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 14:15:00 | 1420.90 | 1406.19 | 1425.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 14:15:00 | 1420.90 | 1406.19 | 1425.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 1420.90 | 1406.19 | 1425.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 1420.90 | 1406.19 | 1425.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 1401.00 | 1405.15 | 1423.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 09:30:00 | 1399.00 | 1402.14 | 1420.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 11:30:00 | 1400.90 | 1400.80 | 1416.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 14:15:00 | 1394.00 | 1404.05 | 1409.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 12:15:00 | 1387.80 | 1375.97 | 1375.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 12:15:00 | 1387.80 | 1375.97 | 1375.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 09:15:00 | 1416.60 | 1389.08 | 1382.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 1381.40 | 1393.59 | 1387.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 13:15:00 | 1381.40 | 1393.59 | 1387.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1381.40 | 1393.59 | 1387.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 1381.40 | 1393.59 | 1387.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 1381.90 | 1391.25 | 1386.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:00:00 | 1381.90 | 1391.25 | 1386.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 1381.70 | 1389.34 | 1386.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 1359.60 | 1389.34 | 1386.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 1367.20 | 1384.91 | 1384.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:30:00 | 1354.00 | 1384.91 | 1384.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 10:15:00 | 1369.10 | 1381.75 | 1383.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 1361.00 | 1370.75 | 1376.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 1368.30 | 1365.50 | 1371.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 1368.30 | 1365.50 | 1371.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 1357.50 | 1357.59 | 1363.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 1354.50 | 1357.59 | 1363.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 1364.10 | 1358.89 | 1363.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:15:00 | 1346.20 | 1358.49 | 1361.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:15:00 | 1348.00 | 1356.75 | 1360.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 09:15:00 | 1347.80 | 1358.10 | 1359.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 15:15:00 | 1383.00 | 1350.80 | 1352.84 | SL hit (close>static) qty=1.00 sl=1366.50 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 14:15:00 | 1359.90 | 1346.93 | 1345.31 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 13:15:00 | 1338.40 | 1344.18 | 1344.89 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 10:15:00 | 1361.40 | 1347.81 | 1346.32 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 15:15:00 | 1330.00 | 1344.80 | 1346.11 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 11:15:00 | 1348.80 | 1346.86 | 1346.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 13:15:00 | 1379.30 | 1353.85 | 1350.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 10:15:00 | 1360.00 | 1365.75 | 1358.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 10:15:00 | 1360.00 | 1365.75 | 1358.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 1360.00 | 1365.75 | 1358.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 1360.00 | 1365.75 | 1358.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 1363.00 | 1365.20 | 1358.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:45:00 | 1358.00 | 1365.20 | 1358.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 1370.60 | 1365.36 | 1360.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 15:15:00 | 1372.90 | 1365.36 | 1360.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 10:45:00 | 1378.70 | 1370.10 | 1363.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 15:00:00 | 1375.10 | 1372.12 | 1366.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 14:15:00 | 1445.00 | 1446.98 | 1447.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 14:15:00 | 1445.00 | 1446.98 | 1447.16 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 15:15:00 | 1450.00 | 1447.59 | 1447.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 09:15:00 | 1454.40 | 1448.95 | 1448.05 | Break + close above crossover candle high |

### Cycle 16 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 1439.80 | 1447.12 | 1447.30 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 12:15:00 | 1455.10 | 1448.30 | 1447.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 1468.40 | 1455.67 | 1451.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 10:15:00 | 1459.40 | 1465.22 | 1460.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 10:15:00 | 1459.40 | 1465.22 | 1460.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 1459.40 | 1465.22 | 1460.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:00:00 | 1459.40 | 1465.22 | 1460.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 1462.70 | 1464.71 | 1460.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 1475.00 | 1460.81 | 1459.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 12:00:00 | 1468.70 | 1463.41 | 1461.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 10:00:00 | 1474.20 | 1470.16 | 1465.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 11:45:00 | 1471.30 | 1469.20 | 1466.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1470.00 | 1470.28 | 1467.92 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-21 13:15:00 | 1461.90 | 1466.07 | 1466.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 13:15:00 | 1461.90 | 1466.07 | 1466.52 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 09:15:00 | 1520.50 | 1470.76 | 1465.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 14:15:00 | 1537.90 | 1510.17 | 1489.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 10:15:00 | 1525.00 | 1525.50 | 1502.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 10:45:00 | 1519.70 | 1525.50 | 1502.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 1514.90 | 1518.97 | 1506.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:30:00 | 1504.60 | 1518.97 | 1506.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 1511.00 | 1517.37 | 1506.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 09:15:00 | 1518.70 | 1517.37 | 1506.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 10:00:00 | 1520.00 | 1517.90 | 1508.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 1494.10 | 1513.14 | 1506.86 | SL hit (close<static) qty=1.00 sl=1496.30 alert=retest2 |

### Cycle 20 — SELL (started 2025-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 12:15:00 | 1481.60 | 1501.72 | 1502.44 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 1531.60 | 1497.08 | 1495.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 10:15:00 | 1546.00 | 1506.86 | 1500.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 1484.70 | 1514.95 | 1509.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 1484.70 | 1514.95 | 1509.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1484.70 | 1514.95 | 1509.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:00:00 | 1484.70 | 1514.95 | 1509.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 1482.10 | 1508.38 | 1506.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:30:00 | 1481.90 | 1508.38 | 1506.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 11:15:00 | 1482.10 | 1503.12 | 1504.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 1472.80 | 1497.03 | 1501.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 15:15:00 | 1395.20 | 1391.13 | 1404.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-08 09:15:00 | 1396.40 | 1391.13 | 1404.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1398.30 | 1392.56 | 1403.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:00:00 | 1398.30 | 1392.56 | 1403.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 1330.60 | 1365.26 | 1380.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 13:30:00 | 1370.00 | 1365.26 | 1380.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 1381.00 | 1359.44 | 1370.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 12:00:00 | 1381.00 | 1359.44 | 1370.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 1381.80 | 1363.91 | 1371.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 12:30:00 | 1380.10 | 1363.91 | 1371.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 1376.70 | 1369.76 | 1373.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 10:15:00 | 1369.60 | 1373.57 | 1374.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 10:30:00 | 1371.90 | 1366.56 | 1366.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 11:15:00 | 1370.80 | 1367.40 | 1367.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 11:15:00 | 1370.80 | 1367.40 | 1367.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 12:15:00 | 1377.40 | 1369.40 | 1368.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 1372.50 | 1377.09 | 1372.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 1372.50 | 1377.09 | 1372.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 1372.50 | 1377.09 | 1372.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:00:00 | 1372.50 | 1377.09 | 1372.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 1375.00 | 1376.67 | 1372.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 14:30:00 | 1381.70 | 1376.04 | 1373.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 15:15:00 | 1369.00 | 1377.94 | 1376.95 | SL hit (close<static) qty=1.00 sl=1371.10 alert=retest2 |

### Cycle 24 — SELL (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 09:15:00 | 1364.70 | 1375.29 | 1375.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 1357.20 | 1369.36 | 1372.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 10:15:00 | 1379.80 | 1365.47 | 1368.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 10:15:00 | 1379.80 | 1365.47 | 1368.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1379.80 | 1365.47 | 1368.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 1379.80 | 1365.47 | 1368.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 11:15:00 | 1395.30 | 1371.44 | 1371.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 12:15:00 | 1417.80 | 1380.71 | 1375.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 11:15:00 | 1392.80 | 1397.04 | 1388.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-25 11:45:00 | 1392.20 | 1397.04 | 1388.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1387.00 | 1395.34 | 1390.77 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 12:15:00 | 1378.70 | 1386.85 | 1387.58 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 14:15:00 | 1435.00 | 1395.32 | 1391.23 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 12:15:00 | 1390.00 | 1401.12 | 1401.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 14:15:00 | 1381.00 | 1395.32 | 1398.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 15:15:00 | 1374.10 | 1373.34 | 1382.73 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-02 09:15:00 | 1366.10 | 1373.34 | 1382.73 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-02 11:30:00 | 1369.50 | 1371.97 | 1379.73 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-02 12:00:00 | 1368.50 | 1371.97 | 1379.73 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-02 13:30:00 | 1368.00 | 1371.20 | 1378.03 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 1345.70 | 1346.38 | 1356.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:45:00 | 1351.40 | 1346.38 | 1356.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1334.70 | 1342.02 | 1351.80 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 1350.30 | 1342.41 | 1350.20 | SL hit (close>ema400) qty=1.00 sl=1350.20 alert=retest1 |

### Cycle 29 — BUY (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 09:15:00 | 1322.00 | 1313.88 | 1313.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 11:15:00 | 1325.00 | 1317.02 | 1314.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 09:15:00 | 1388.00 | 1391.27 | 1373.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-23 09:30:00 | 1384.40 | 1391.27 | 1373.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 1372.30 | 1387.47 | 1373.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 1372.30 | 1387.47 | 1373.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 1370.30 | 1384.04 | 1373.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 1369.60 | 1384.04 | 1373.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 1391.40 | 1385.51 | 1374.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:30:00 | 1388.70 | 1385.51 | 1374.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1372.80 | 1385.61 | 1378.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:15:00 | 1371.00 | 1385.61 | 1378.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 1364.10 | 1381.31 | 1377.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 1364.10 | 1381.31 | 1377.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 12:15:00 | 1356.00 | 1373.04 | 1374.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 1353.10 | 1366.73 | 1370.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 10:15:00 | 1358.00 | 1352.56 | 1358.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 10:15:00 | 1358.00 | 1352.56 | 1358.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 1358.00 | 1352.56 | 1358.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:00:00 | 1358.00 | 1352.56 | 1358.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 1359.40 | 1353.93 | 1358.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:45:00 | 1364.00 | 1353.93 | 1358.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 12:15:00 | 1374.50 | 1358.04 | 1359.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 13:00:00 | 1374.50 | 1358.04 | 1359.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 1372.90 | 1361.02 | 1361.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:15:00 | 1375.00 | 1361.02 | 1361.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-26 14:15:00 | 1371.50 | 1363.11 | 1361.99 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-09-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 10:15:00 | 1347.50 | 1359.30 | 1360.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 11:15:00 | 1334.80 | 1354.40 | 1358.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 09:15:00 | 1348.70 | 1338.95 | 1347.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 09:15:00 | 1348.70 | 1338.95 | 1347.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 1348.70 | 1338.95 | 1347.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:00:00 | 1348.70 | 1338.95 | 1347.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 1348.90 | 1340.94 | 1347.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 11:15:00 | 1350.00 | 1340.94 | 1347.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 1356.20 | 1343.99 | 1348.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 11:45:00 | 1356.90 | 1343.99 | 1348.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 1342.00 | 1343.59 | 1348.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:15:00 | 1335.90 | 1343.59 | 1348.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 1330.70 | 1341.47 | 1345.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 10:15:00 | 1376.20 | 1333.56 | 1327.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 10:15:00 | 1376.20 | 1333.56 | 1327.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 12:15:00 | 1388.40 | 1351.21 | 1337.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 1352.90 | 1354.82 | 1343.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 1349.80 | 1351.70 | 1347.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1349.80 | 1351.70 | 1347.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:00:00 | 1349.80 | 1351.70 | 1347.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 1346.00 | 1350.56 | 1347.14 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 13:15:00 | 1339.60 | 1345.13 | 1345.21 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 1352.20 | 1345.18 | 1344.92 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 12:15:00 | 1332.60 | 1342.66 | 1343.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 1326.70 | 1337.04 | 1340.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1311.80 | 1311.72 | 1320.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:00:00 | 1311.80 | 1311.72 | 1320.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 1308.60 | 1310.11 | 1316.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:30:00 | 1312.20 | 1310.11 | 1316.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 1304.30 | 1305.21 | 1312.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:30:00 | 1311.10 | 1305.21 | 1312.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 1309.90 | 1306.14 | 1312.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:45:00 | 1309.70 | 1306.14 | 1312.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 1279.20 | 1277.53 | 1284.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 1280.10 | 1277.53 | 1284.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 1284.30 | 1278.89 | 1284.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 1284.30 | 1278.89 | 1284.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1281.10 | 1279.33 | 1284.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 12:45:00 | 1269.90 | 1276.19 | 1279.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 15:00:00 | 1269.90 | 1273.75 | 1278.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 1287.90 | 1276.27 | 1278.48 | SL hit (close>static) qty=1.00 sl=1287.40 alert=retest2 |

### Cycle 37 — BUY (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 11:15:00 | 1290.80 | 1281.45 | 1280.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 13:15:00 | 1294.40 | 1284.96 | 1282.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 11:15:00 | 1281.50 | 1289.53 | 1286.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 11:15:00 | 1281.50 | 1289.53 | 1286.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 1281.50 | 1289.53 | 1286.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 1281.50 | 1289.53 | 1286.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 1289.00 | 1289.42 | 1286.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 15:15:00 | 1291.00 | 1287.87 | 1286.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 10:45:00 | 1293.30 | 1288.51 | 1286.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 10:15:00 | 1278.20 | 1286.92 | 1287.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 10:15:00 | 1278.20 | 1286.92 | 1287.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 1272.00 | 1278.17 | 1281.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 1283.10 | 1277.85 | 1280.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 1283.10 | 1277.85 | 1280.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1283.10 | 1277.85 | 1280.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:30:00 | 1286.20 | 1277.85 | 1280.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 1287.60 | 1279.80 | 1281.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:30:00 | 1293.30 | 1279.80 | 1281.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 11:15:00 | 1300.20 | 1283.88 | 1283.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 12:15:00 | 1312.40 | 1289.58 | 1285.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 10:15:00 | 1297.10 | 1299.54 | 1292.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 10:15:00 | 1297.10 | 1299.54 | 1292.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 1297.10 | 1299.54 | 1292.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:45:00 | 1299.80 | 1299.54 | 1292.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 1294.00 | 1300.48 | 1295.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:00:00 | 1294.00 | 1300.48 | 1295.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 1294.40 | 1299.26 | 1295.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 15:15:00 | 1291.30 | 1299.26 | 1295.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 1291.30 | 1297.67 | 1294.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:15:00 | 1300.60 | 1297.67 | 1294.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1294.00 | 1296.93 | 1294.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 12:00:00 | 1305.00 | 1297.98 | 1295.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 14:15:00 | 1309.20 | 1299.33 | 1296.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 1288.00 | 1297.87 | 1296.78 | SL hit (close<static) qty=1.00 sl=1289.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 10:15:00 | 1284.40 | 1295.18 | 1295.65 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 11:15:00 | 1309.50 | 1298.04 | 1296.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 12:15:00 | 1331.00 | 1304.63 | 1300.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 09:15:00 | 1307.60 | 1311.32 | 1305.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 09:15:00 | 1307.60 | 1311.32 | 1305.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1307.60 | 1311.32 | 1305.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 13:00:00 | 1326.80 | 1312.29 | 1307.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 10:45:00 | 1327.60 | 1328.08 | 1318.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 12:45:00 | 1333.40 | 1327.78 | 1320.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 09:15:00 | 1327.90 | 1325.48 | 1320.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 1342.90 | 1328.97 | 1322.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 10:15:00 | 1346.50 | 1328.97 | 1322.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 1351.90 | 1339.23 | 1332.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-13 14:15:00 | 1301.40 | 1327.18 | 1329.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 14:15:00 | 1301.40 | 1327.18 | 1329.69 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 10:15:00 | 1380.50 | 1337.29 | 1333.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 12:15:00 | 1405.00 | 1357.26 | 1343.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1449.50 | 1451.86 | 1418.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 14:15:00 | 1423.80 | 1438.18 | 1423.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 1423.80 | 1438.18 | 1423.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 1423.80 | 1438.18 | 1423.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 1420.00 | 1434.54 | 1423.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:00:00 | 1411.20 | 1429.87 | 1422.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 1415.50 | 1427.00 | 1421.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:00:00 | 1415.50 | 1427.00 | 1421.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 1421.30 | 1425.86 | 1421.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:00:00 | 1421.30 | 1425.86 | 1421.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 1426.10 | 1425.91 | 1422.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 13:45:00 | 1430.80 | 1426.73 | 1422.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 14:15:00 | 1430.90 | 1426.73 | 1422.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 1433.20 | 1438.43 | 1433.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 1442.50 | 1439.92 | 1437.38 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 1440.30 | 1439.99 | 1437.64 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-24 12:15:00 | 1425.00 | 1435.52 | 1436.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 12:15:00 | 1425.00 | 1435.52 | 1436.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 15:15:00 | 1418.20 | 1429.88 | 1433.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 1435.70 | 1431.04 | 1433.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 1435.70 | 1431.04 | 1433.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 1435.70 | 1431.04 | 1433.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:45:00 | 1433.60 | 1431.04 | 1433.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 1435.10 | 1431.85 | 1433.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:15:00 | 1439.70 | 1431.85 | 1433.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 1429.70 | 1431.42 | 1433.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 12:30:00 | 1425.50 | 1430.14 | 1432.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 10:00:00 | 1414.90 | 1403.51 | 1407.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 11:00:00 | 1424.80 | 1407.77 | 1409.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 11:15:00 | 1438.50 | 1413.91 | 1411.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 11:15:00 | 1438.50 | 1413.91 | 1411.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 12:15:00 | 1441.00 | 1419.33 | 1414.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 11:15:00 | 1418.80 | 1432.25 | 1425.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 11:15:00 | 1418.80 | 1432.25 | 1425.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 1418.80 | 1432.25 | 1425.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 1418.80 | 1432.25 | 1425.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 1425.00 | 1430.80 | 1425.28 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 1408.70 | 1422.21 | 1422.58 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 14:15:00 | 1434.20 | 1423.10 | 1422.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 11:15:00 | 1441.00 | 1429.48 | 1425.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 15:15:00 | 1432.00 | 1435.66 | 1430.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 15:15:00 | 1432.00 | 1435.66 | 1430.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 1432.00 | 1435.66 | 1430.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 09:30:00 | 1447.00 | 1442.15 | 1433.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 09:15:00 | 1424.00 | 1445.92 | 1441.23 | SL hit (close<static) qty=1.00 sl=1430.20 alert=retest2 |

### Cycle 48 — SELL (started 2025-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 12:15:00 | 1423.70 | 1436.58 | 1437.72 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 14:15:00 | 1457.40 | 1438.34 | 1438.17 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 1428.60 | 1437.46 | 1437.87 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 11:15:00 | 1454.60 | 1439.21 | 1438.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-08 12:15:00 | 1460.00 | 1443.37 | 1440.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 15:15:00 | 1426.20 | 1442.39 | 1440.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 15:15:00 | 1426.20 | 1442.39 | 1440.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 15:15:00 | 1426.20 | 1442.39 | 1440.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:15:00 | 1412.00 | 1442.39 | 1440.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 1409.50 | 1435.81 | 1438.08 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 1449.70 | 1436.66 | 1436.39 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 12:15:00 | 1429.70 | 1436.42 | 1436.60 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 13:15:00 | 1468.90 | 1442.91 | 1439.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 14:15:00 | 1475.70 | 1449.47 | 1442.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 09:15:00 | 1434.90 | 1447.76 | 1443.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 1434.90 | 1447.76 | 1443.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 1434.90 | 1447.76 | 1443.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:45:00 | 1431.60 | 1447.76 | 1443.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 1468.80 | 1451.97 | 1445.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 15:15:00 | 1481.00 | 1462.34 | 1453.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:45:00 | 1480.30 | 1468.70 | 1457.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 11:15:00 | 1448.00 | 1457.68 | 1458.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 11:15:00 | 1448.00 | 1457.68 | 1458.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 12:15:00 | 1443.70 | 1454.88 | 1456.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 14:15:00 | 1399.70 | 1394.88 | 1409.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 15:00:00 | 1399.70 | 1394.88 | 1409.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 1430.00 | 1401.90 | 1411.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:45:00 | 1396.30 | 1403.53 | 1409.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 15:15:00 | 1430.50 | 1412.44 | 1412.52 | SL hit (close>static) qty=1.00 sl=1430.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 1430.40 | 1416.03 | 1414.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 10:15:00 | 1441.70 | 1421.17 | 1416.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 14:15:00 | 1426.90 | 1431.49 | 1423.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 14:15:00 | 1426.90 | 1431.49 | 1423.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 1426.90 | 1431.49 | 1423.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 15:00:00 | 1426.90 | 1431.49 | 1423.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 1432.00 | 1431.59 | 1424.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:45:00 | 1437.30 | 1432.33 | 1425.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 13:45:00 | 1437.30 | 1430.75 | 1426.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 10:15:00 | 1422.00 | 1429.13 | 1427.33 | SL hit (close<static) qty=1.00 sl=1423.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 12:15:00 | 1420.80 | 1425.36 | 1425.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 15:15:00 | 1412.90 | 1420.48 | 1423.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 14:15:00 | 1443.00 | 1417.16 | 1419.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 14:15:00 | 1443.00 | 1417.16 | 1419.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 1443.00 | 1417.16 | 1419.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 15:00:00 | 1443.00 | 1417.16 | 1419.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 15:15:00 | 1435.00 | 1420.73 | 1420.67 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 1406.60 | 1417.98 | 1419.43 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 15:15:00 | 1439.00 | 1422.36 | 1420.51 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 10:15:00 | 1396.00 | 1417.62 | 1418.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 11:15:00 | 1385.60 | 1411.22 | 1415.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 1424.60 | 1401.86 | 1407.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 10:15:00 | 1424.60 | 1401.86 | 1407.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1424.60 | 1401.86 | 1407.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 1424.60 | 1401.86 | 1407.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 1439.00 | 1409.29 | 1409.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:00:00 | 1439.00 | 1409.29 | 1409.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 1440.80 | 1415.59 | 1412.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 13:15:00 | 1449.40 | 1422.35 | 1416.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 15:15:00 | 1421.00 | 1422.59 | 1417.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 15:15:00 | 1421.00 | 1422.59 | 1417.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 1421.00 | 1422.59 | 1417.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 10:15:00 | 1432.70 | 1423.21 | 1418.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 11:15:00 | 1432.50 | 1423.71 | 1418.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 14:15:00 | 1397.00 | 1417.14 | 1417.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 14:15:00 | 1397.00 | 1417.14 | 1417.24 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 1440.00 | 1415.89 | 1413.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 10:15:00 | 1446.50 | 1422.01 | 1416.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-12 09:15:00 | 1511.30 | 1545.47 | 1526.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 09:15:00 | 1511.30 | 1545.47 | 1526.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 1511.30 | 1545.47 | 1526.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:45:00 | 1521.10 | 1545.47 | 1526.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 1517.40 | 1539.86 | 1525.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 10:30:00 | 1506.30 | 1539.86 | 1525.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1544.50 | 1531.47 | 1525.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 1513.60 | 1531.47 | 1525.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1505.30 | 1526.24 | 1523.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:30:00 | 1506.80 | 1526.24 | 1523.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 1513.80 | 1523.75 | 1522.98 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 11:15:00 | 1510.00 | 1521.00 | 1521.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 12:15:00 | 1507.90 | 1518.38 | 1520.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 11:15:00 | 1508.80 | 1508.25 | 1513.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 11:15:00 | 1508.80 | 1508.25 | 1513.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 1508.80 | 1508.25 | 1513.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 1498.10 | 1512.92 | 1514.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 1498.30 | 1509.17 | 1510.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 11:30:00 | 1503.00 | 1506.01 | 1508.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 13:15:00 | 1494.50 | 1484.05 | 1482.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 1494.50 | 1484.05 | 1482.91 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 1473.60 | 1482.47 | 1482.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 11:15:00 | 1467.30 | 1479.44 | 1481.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 11:15:00 | 1469.40 | 1468.31 | 1473.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 11:45:00 | 1469.40 | 1468.31 | 1473.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 12:15:00 | 1467.50 | 1468.15 | 1472.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 12:45:00 | 1469.00 | 1468.15 | 1472.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1460.50 | 1462.33 | 1468.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 12:15:00 | 1447.00 | 1461.58 | 1466.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 13:15:00 | 1444.20 | 1459.62 | 1465.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 12:15:00 | 1473.00 | 1459.47 | 1461.34 | SL hit (close>static) qty=1.00 sl=1469.80 alert=retest2 |

### Cycle 69 — BUY (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 13:15:00 | 1485.40 | 1464.66 | 1463.53 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 09:15:00 | 1446.20 | 1466.77 | 1467.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 1424.50 | 1455.84 | 1462.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 10:15:00 | 1435.60 | 1431.05 | 1444.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 10:15:00 | 1435.60 | 1431.05 | 1444.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 1435.60 | 1431.05 | 1444.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 1435.60 | 1431.05 | 1444.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 1404.70 | 1423.60 | 1437.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:45:00 | 1418.60 | 1423.60 | 1437.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 1434.00 | 1425.68 | 1437.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 14:45:00 | 1440.10 | 1425.68 | 1437.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 1445.20 | 1429.58 | 1438.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 1450.80 | 1429.58 | 1438.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1449.50 | 1433.57 | 1439.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 12:00:00 | 1442.20 | 1440.23 | 1441.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 12:45:00 | 1437.50 | 1439.82 | 1441.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 12:30:00 | 1434.30 | 1436.82 | 1438.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 15:15:00 | 1440.00 | 1417.07 | 1422.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 1440.00 | 1421.65 | 1424.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 09:15:00 | 1410.70 | 1421.65 | 1424.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 11:15:00 | 1434.20 | 1422.62 | 1422.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 1434.20 | 1422.62 | 1422.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 1444.00 | 1426.90 | 1424.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 10:15:00 | 1437.30 | 1439.30 | 1432.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 11:15:00 | 1432.10 | 1437.86 | 1432.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 1432.10 | 1437.86 | 1432.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 12:00:00 | 1432.10 | 1437.86 | 1432.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 1428.20 | 1435.93 | 1431.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:15:00 | 1428.10 | 1435.93 | 1431.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 1428.40 | 1434.42 | 1431.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:45:00 | 1430.00 | 1434.42 | 1431.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 1426.20 | 1432.78 | 1431.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:00:00 | 1426.20 | 1432.78 | 1431.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 1412.60 | 1427.02 | 1428.72 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 11:15:00 | 1451.80 | 1431.73 | 1430.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 12:15:00 | 1469.80 | 1439.34 | 1434.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 13:15:00 | 1436.00 | 1438.68 | 1434.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 13:15:00 | 1436.00 | 1438.68 | 1434.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 1436.00 | 1438.68 | 1434.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 14:00:00 | 1436.00 | 1438.68 | 1434.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 1427.90 | 1436.52 | 1433.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 15:00:00 | 1427.90 | 1436.52 | 1433.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 1434.90 | 1436.20 | 1433.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 1423.60 | 1436.20 | 1433.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1420.70 | 1433.10 | 1432.62 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 1413.10 | 1429.10 | 1430.85 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 14:15:00 | 1476.80 | 1433.66 | 1431.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 14:15:00 | 1499.30 | 1473.38 | 1461.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 13:15:00 | 1480.20 | 1486.70 | 1475.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-17 14:00:00 | 1480.20 | 1486.70 | 1475.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 1492.70 | 1487.90 | 1477.00 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 13:15:00 | 1466.90 | 1478.65 | 1479.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 1455.00 | 1472.04 | 1476.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 1471.60 | 1460.13 | 1465.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 1471.60 | 1460.13 | 1465.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1471.60 | 1460.13 | 1465.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:15:00 | 1481.30 | 1460.13 | 1465.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1488.90 | 1465.88 | 1467.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 1488.90 | 1465.88 | 1467.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 1483.20 | 1469.34 | 1469.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 12:15:00 | 1493.30 | 1474.14 | 1471.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 15:15:00 | 1469.70 | 1475.30 | 1472.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 15:15:00 | 1469.70 | 1475.30 | 1472.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 1469.70 | 1475.30 | 1472.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:15:00 | 1458.80 | 1475.30 | 1472.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 1466.60 | 1473.56 | 1472.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 13:00:00 | 1479.00 | 1473.58 | 1472.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 13:15:00 | 1510.00 | 1517.33 | 1517.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2026-03-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 13:15:00 | 1510.00 | 1517.33 | 1517.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 1489.00 | 1510.00 | 1514.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 1484.40 | 1481.46 | 1493.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 09:30:00 | 1484.90 | 1481.46 | 1493.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 1500.60 | 1483.61 | 1490.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:00:00 | 1500.60 | 1483.61 | 1490.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 1500.60 | 1487.00 | 1491.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 1504.20 | 1487.00 | 1491.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1512.30 | 1493.66 | 1494.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:45:00 | 1512.70 | 1493.66 | 1494.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 10:15:00 | 1507.80 | 1496.49 | 1495.26 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2026-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 15:15:00 | 1483.10 | 1494.65 | 1495.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 1458.60 | 1487.44 | 1491.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 14:15:00 | 1500.00 | 1480.69 | 1485.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 14:15:00 | 1500.00 | 1480.69 | 1485.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 14:15:00 | 1500.00 | 1480.69 | 1485.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 15:00:00 | 1500.00 | 1480.69 | 1485.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 15:15:00 | 1500.00 | 1484.55 | 1486.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:15:00 | 1508.10 | 1484.55 | 1486.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 1491.90 | 1488.69 | 1488.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 1497.10 | 1491.70 | 1490.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 09:15:00 | 1556.50 | 1559.58 | 1546.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 09:15:00 | 1556.50 | 1559.58 | 1546.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 1556.50 | 1559.58 | 1546.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:30:00 | 1546.90 | 1559.58 | 1546.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 1546.90 | 1557.05 | 1546.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:45:00 | 1550.20 | 1557.05 | 1546.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 1547.50 | 1555.14 | 1546.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 11:30:00 | 1546.60 | 1555.14 | 1546.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 1568.50 | 1557.81 | 1548.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 12:30:00 | 1545.10 | 1557.81 | 1548.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 1545.70 | 1566.69 | 1561.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:15:00 | 1553.00 | 1566.69 | 1561.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1538.60 | 1561.08 | 1559.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:00:00 | 1538.60 | 1561.08 | 1559.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 10:15:00 | 1538.20 | 1556.50 | 1557.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 1529.90 | 1544.81 | 1550.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-19 11:15:00 | 1565.20 | 1548.16 | 1550.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 11:15:00 | 1565.20 | 1548.16 | 1550.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 1565.20 | 1548.16 | 1550.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:45:00 | 1554.60 | 1548.16 | 1550.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 1556.70 | 1549.87 | 1551.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 13:45:00 | 1543.10 | 1547.90 | 1550.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 15:15:00 | 1527.00 | 1538.00 | 1542.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 15:00:00 | 1527.30 | 1520.95 | 1525.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1565.60 | 1530.21 | 1528.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1565.60 | 1530.21 | 1528.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 1592.00 | 1542.57 | 1534.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1555.00 | 1562.93 | 1550.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1555.00 | 1562.93 | 1550.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1555.00 | 1562.93 | 1550.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 13:00:00 | 1601.70 | 1585.97 | 1572.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 15:00:00 | 1602.00 | 1590.14 | 1576.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 13:15:00 | 1527.90 | 1569.14 | 1572.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 13:15:00 | 1527.90 | 1569.14 | 1572.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-01 14:15:00 | 1505.10 | 1556.33 | 1566.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 1500.30 | 1498.99 | 1527.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 14:00:00 | 1500.30 | 1498.99 | 1527.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 13:15:00 | 1454.10 | 1446.50 | 1460.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 14:00:00 | 1454.10 | 1446.50 | 1460.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 14:15:00 | 1445.70 | 1446.34 | 1458.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 14:30:00 | 1464.70 | 1446.34 | 1458.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 1456.30 | 1448.60 | 1457.72 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 1462.40 | 1456.44 | 1456.11 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-15 10:15:00 | 1448.10 | 1454.77 | 1455.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-15 11:15:00 | 1447.70 | 1453.36 | 1454.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 13:15:00 | 1464.00 | 1454.85 | 1455.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 13:15:00 | 1464.00 | 1454.85 | 1455.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 13:15:00 | 1464.00 | 1454.85 | 1455.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:00:00 | 1464.00 | 1454.85 | 1455.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2026-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 14:15:00 | 1471.40 | 1458.16 | 1456.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 1485.50 | 1465.92 | 1460.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 11:15:00 | 1464.30 | 1467.26 | 1462.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 12:00:00 | 1464.30 | 1467.26 | 1462.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 1454.60 | 1464.73 | 1461.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 13:00:00 | 1454.60 | 1464.73 | 1461.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 1456.00 | 1462.98 | 1460.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 13:30:00 | 1454.90 | 1462.98 | 1460.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 15:15:00 | 1468.00 | 1463.51 | 1461.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 09:15:00 | 1462.20 | 1463.51 | 1461.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 1462.70 | 1463.35 | 1461.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 09:30:00 | 1463.30 | 1463.35 | 1461.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 10:15:00 | 1466.80 | 1464.04 | 1462.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 14:30:00 | 1469.30 | 1464.35 | 1462.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 15:15:00 | 1472.90 | 1464.35 | 1462.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 09:30:00 | 1479.30 | 1468.59 | 1465.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 12:15:00 | 1469.20 | 1469.10 | 1466.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 12:15:00 | 1468.00 | 1468.88 | 1466.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 1471.70 | 1468.77 | 1466.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:00:00 | 1471.40 | 1469.30 | 1467.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 11:15:00 | 1483.30 | 1469.22 | 1467.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 12:15:00 | 1535.30 | 1543.69 | 1544.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 12:15:00 | 1535.30 | 1543.69 | 1544.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 13:15:00 | 1531.10 | 1541.17 | 1542.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 12:15:00 | 1537.70 | 1533.38 | 1537.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 12:15:00 | 1537.70 | 1533.38 | 1537.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 1537.70 | 1533.38 | 1537.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:45:00 | 1549.00 | 1533.38 | 1537.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 1539.30 | 1534.56 | 1537.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:15:00 | 1548.10 | 1534.56 | 1537.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 1556.00 | 1538.85 | 1539.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:45:00 | 1553.90 | 1538.85 | 1539.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 1536.60 | 1538.40 | 1538.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:15:00 | 1549.90 | 1538.40 | 1538.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 1552.20 | 1541.16 | 1540.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 10:15:00 | 1562.10 | 1545.35 | 1542.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 11:15:00 | 1561.00 | 1562.78 | 1555.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 12:00:00 | 1561.00 | 1562.78 | 1555.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 14:15:00 | 1556.40 | 1562.31 | 1557.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 15:00:00 | 1556.40 | 1562.31 | 1557.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 15:15:00 | 1556.80 | 1561.21 | 1557.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 1582.90 | 1561.21 | 1557.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 11:15:00 | 1551.00 | 1565.24 | 1560.65 | SL hit (close<static) qty=1.00 sl=1555.10 alert=retest2 |

### Cycle 90 — SELL (started 2026-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 15:15:00 | 1554.00 | 1557.37 | 1557.79 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-21 09:15:00 | 1475.30 | 2025-05-26 12:15:00 | 1450.70 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-05-21 14:45:00 | 1476.20 | 2025-05-26 12:15:00 | 1450.70 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-05-23 10:00:00 | 1470.90 | 2025-05-27 15:15:00 | 1453.00 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-05-23 11:15:00 | 1467.50 | 2025-05-27 15:15:00 | 1453.00 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-05-26 09:15:00 | 1479.50 | 2025-05-28 09:15:00 | 1450.40 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-05-26 10:30:00 | 1464.00 | 2025-05-28 09:15:00 | 1450.40 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-05-26 15:00:00 | 1464.80 | 2025-05-28 09:15:00 | 1450.40 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-05-27 09:30:00 | 1467.10 | 2025-05-28 09:15:00 | 1450.40 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-06-03 09:30:00 | 1399.00 | 2025-06-11 12:15:00 | 1387.80 | STOP_HIT | 1.00 | 0.80% |
| SELL | retest2 | 2025-06-03 11:30:00 | 1400.90 | 2025-06-11 12:15:00 | 1387.80 | STOP_HIT | 1.00 | 0.94% |
| SELL | retest2 | 2025-06-04 14:15:00 | 1394.00 | 2025-06-11 12:15:00 | 1387.80 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2025-06-19 10:15:00 | 1346.20 | 2025-06-20 15:15:00 | 1383.00 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-06-19 11:15:00 | 1348.00 | 2025-06-20 15:15:00 | 1383.00 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-06-20 09:15:00 | 1347.80 | 2025-06-20 15:15:00 | 1383.00 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2025-06-23 10:15:00 | 1347.70 | 2025-06-25 14:15:00 | 1359.90 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-06-24 12:00:00 | 1345.00 | 2025-06-25 14:15:00 | 1359.90 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-07-01 15:15:00 | 1372.90 | 2025-07-10 14:15:00 | 1445.00 | STOP_HIT | 1.00 | 5.25% |
| BUY | retest2 | 2025-07-02 10:45:00 | 1378.70 | 2025-07-10 14:15:00 | 1445.00 | STOP_HIT | 1.00 | 4.81% |
| BUY | retest2 | 2025-07-02 15:00:00 | 1375.10 | 2025-07-10 14:15:00 | 1445.00 | STOP_HIT | 1.00 | 5.08% |
| BUY | retest2 | 2025-07-17 09:15:00 | 1475.00 | 2025-07-21 13:15:00 | 1461.90 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-07-17 12:00:00 | 1468.70 | 2025-07-21 13:15:00 | 1461.90 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-07-18 10:00:00 | 1474.20 | 2025-07-21 13:15:00 | 1461.90 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-07-18 11:45:00 | 1471.30 | 2025-07-21 13:15:00 | 1461.90 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-07-28 09:15:00 | 1518.70 | 2025-07-28 10:15:00 | 1494.10 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-07-28 10:00:00 | 1520.00 | 2025-07-28 10:15:00 | 1494.10 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-08-13 10:15:00 | 1369.60 | 2025-08-18 11:15:00 | 1370.80 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-08-18 10:30:00 | 1371.90 | 2025-08-18 11:15:00 | 1370.80 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2025-08-19 14:30:00 | 1381.70 | 2025-08-20 15:15:00 | 1369.00 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest1 | 2025-09-02 09:15:00 | 1366.10 | 2025-09-05 11:15:00 | 1350.30 | STOP_HIT | 1.00 | 1.16% |
| SELL | retest1 | 2025-09-02 11:30:00 | 1369.50 | 2025-09-05 11:15:00 | 1350.30 | STOP_HIT | 1.00 | 1.40% |
| SELL | retest1 | 2025-09-02 12:00:00 | 1368.50 | 2025-09-05 11:15:00 | 1350.30 | STOP_HIT | 1.00 | 1.33% |
| SELL | retest1 | 2025-09-02 13:30:00 | 1368.00 | 2025-09-05 11:15:00 | 1350.30 | STOP_HIT | 1.00 | 1.29% |
| SELL | retest2 | 2025-09-08 11:00:00 | 1332.10 | 2025-09-18 09:15:00 | 1322.00 | STOP_HIT | 1.00 | 0.76% |
| SELL | retest2 | 2025-09-10 10:30:00 | 1322.10 | 2025-09-18 09:15:00 | 1322.00 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2025-09-30 13:15:00 | 1335.90 | 2025-10-07 10:15:00 | 1376.20 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2025-10-01 09:15:00 | 1330.70 | 2025-10-07 10:15:00 | 1376.20 | STOP_HIT | 1.00 | -3.42% |
| SELL | retest2 | 2025-10-24 12:45:00 | 1269.90 | 2025-10-27 09:15:00 | 1287.90 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-10-24 15:00:00 | 1269.90 | 2025-10-27 09:15:00 | 1287.90 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-10-28 15:15:00 | 1291.00 | 2025-10-30 10:15:00 | 1278.20 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-10-29 10:45:00 | 1293.30 | 2025-10-30 10:15:00 | 1278.20 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-11-06 12:00:00 | 1305.00 | 2025-11-07 09:15:00 | 1288.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-11-06 14:15:00 | 1309.20 | 2025-11-07 09:15:00 | 1288.00 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-11-10 13:00:00 | 1326.80 | 2025-11-13 14:15:00 | 1301.40 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-11-11 10:45:00 | 1327.60 | 2025-11-13 14:15:00 | 1301.40 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-11-11 12:45:00 | 1333.40 | 2025-11-13 14:15:00 | 1301.40 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-11-12 09:15:00 | 1327.90 | 2025-11-13 14:15:00 | 1301.40 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-11-12 10:15:00 | 1346.50 | 2025-11-13 14:15:00 | 1301.40 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest2 | 2025-11-13 09:15:00 | 1351.90 | 2025-11-13 14:15:00 | 1301.40 | STOP_HIT | 1.00 | -3.74% |
| BUY | retest2 | 2025-11-19 13:45:00 | 1430.80 | 2025-11-24 12:15:00 | 1425.00 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-11-19 14:15:00 | 1430.90 | 2025-11-24 12:15:00 | 1425.00 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-11-21 09:15:00 | 1433.20 | 2025-11-24 12:15:00 | 1425.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-11-24 09:15:00 | 1442.50 | 2025-11-24 12:15:00 | 1425.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-11-25 12:30:00 | 1425.50 | 2025-11-28 11:15:00 | 1438.50 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-11-28 10:00:00 | 1414.90 | 2025-11-28 11:15:00 | 1438.50 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-11-28 11:00:00 | 1424.80 | 2025-11-28 11:15:00 | 1438.50 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-12-04 09:30:00 | 1447.00 | 2025-12-05 09:15:00 | 1424.00 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-12-11 15:15:00 | 1481.00 | 2025-12-15 11:15:00 | 1448.00 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-12-12 09:45:00 | 1480.30 | 2025-12-15 11:15:00 | 1448.00 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-12-19 12:45:00 | 1396.30 | 2025-12-19 15:15:00 | 1430.50 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2025-12-23 09:45:00 | 1437.30 | 2025-12-24 10:15:00 | 1422.00 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-12-23 13:45:00 | 1437.30 | 2025-12-24 10:15:00 | 1422.00 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2026-01-01 10:15:00 | 1432.70 | 2026-01-01 14:15:00 | 1397.00 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2026-01-01 11:15:00 | 1432.50 | 2026-01-01 14:15:00 | 1397.00 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2026-01-16 09:15:00 | 1498.10 | 2026-01-22 13:15:00 | 1494.50 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2026-01-19 09:15:00 | 1498.30 | 2026-01-22 13:15:00 | 1494.50 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2026-01-19 11:30:00 | 1503.00 | 2026-01-22 13:15:00 | 1494.50 | STOP_HIT | 1.00 | 0.57% |
| SELL | retest2 | 2026-01-28 12:15:00 | 1447.00 | 2026-01-29 12:15:00 | 1473.00 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2026-01-28 13:15:00 | 1444.20 | 2026-01-29 12:15:00 | 1473.00 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2026-02-03 12:00:00 | 1442.20 | 2026-02-09 11:15:00 | 1434.20 | STOP_HIT | 1.00 | 0.55% |
| SELL | retest2 | 2026-02-03 12:45:00 | 1437.50 | 2026-02-09 11:15:00 | 1434.20 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2026-02-04 12:30:00 | 1434.30 | 2026-02-09 11:15:00 | 1434.20 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2026-02-05 15:15:00 | 1440.00 | 2026-02-09 11:15:00 | 1434.20 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2026-02-06 09:15:00 | 1410.70 | 2026-02-09 11:15:00 | 1434.20 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2026-02-24 13:00:00 | 1479.00 | 2026-03-02 13:15:00 | 1510.00 | STOP_HIT | 1.00 | 2.10% |
| SELL | retest2 | 2026-03-19 13:45:00 | 1543.10 | 2026-03-25 09:15:00 | 1565.60 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-03-20 15:15:00 | 1527.00 | 2026-03-25 09:15:00 | 1565.60 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2026-03-24 15:00:00 | 1527.30 | 2026-03-25 09:15:00 | 1565.60 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2026-03-30 13:00:00 | 1601.70 | 2026-04-01 13:15:00 | 1527.90 | STOP_HIT | 1.00 | -4.61% |
| BUY | retest2 | 2026-03-30 15:00:00 | 1602.00 | 2026-04-01 13:15:00 | 1527.90 | STOP_HIT | 1.00 | -4.63% |
| BUY | retest2 | 2026-04-17 14:30:00 | 1469.30 | 2026-05-04 12:15:00 | 1535.30 | STOP_HIT | 1.00 | 4.49% |
| BUY | retest2 | 2026-04-17 15:15:00 | 1472.90 | 2026-05-04 12:15:00 | 1535.30 | STOP_HIT | 1.00 | 4.24% |
| BUY | retest2 | 2026-04-20 09:30:00 | 1479.30 | 2026-05-04 12:15:00 | 1535.30 | STOP_HIT | 1.00 | 3.79% |
| BUY | retest2 | 2026-04-20 12:15:00 | 1469.20 | 2026-05-04 12:15:00 | 1535.30 | STOP_HIT | 1.00 | 4.50% |
| BUY | retest2 | 2026-04-21 09:15:00 | 1471.70 | 2026-05-04 12:15:00 | 1535.30 | STOP_HIT | 1.00 | 4.32% |
| BUY | retest2 | 2026-04-21 10:00:00 | 1471.40 | 2026-05-04 12:15:00 | 1535.30 | STOP_HIT | 1.00 | 4.34% |
| BUY | retest2 | 2026-04-21 11:15:00 | 1483.30 | 2026-05-04 12:15:00 | 1535.30 | STOP_HIT | 1.00 | 3.51% |
| BUY | retest2 | 2026-05-08 09:15:00 | 1582.90 | 2026-05-08 11:15:00 | 1551.00 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2026-05-08 13:30:00 | 1566.30 | 2026-05-08 14:15:00 | 1551.50 | STOP_HIT | 1.00 | -0.94% |
