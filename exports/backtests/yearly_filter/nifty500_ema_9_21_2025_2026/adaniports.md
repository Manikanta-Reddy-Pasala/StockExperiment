# Adani Ports and Special Economic Zone Ltd. (ADANIPORTS)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1760.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 71 |
| ALERT1 | 54 |
| ALERT2 | 54 |
| ALERT2_SKIP | 29 |
| ALERT3 | 142 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 54 |
| PARTIAL | 5 |
| TARGET_HIT | 5 |
| STOP_HIT | 44 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 54 (incl. partial bookings)
- **Trades open at end:** 5
- **Winners / losers:** 21 / 33
- **Target hits / Stop hits / Partials:** 5 / 44 / 5
- **Avg / median % per leg:** 1.21% / -0.45%
- **Sum % (uncompounded):** 65.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 8 | 38.1% | 5 | 16 | 0 | 1.95% | 40.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 8 | 38.1% | 5 | 16 | 0 | 1.95% | 40.9% |
| SELL (all) | 33 | 13 | 39.4% | 0 | 28 | 5 | 0.73% | 24.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 33 | 13 | 39.4% | 0 | 28 | 5 | 0.73% | 24.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 54 | 21 | 38.9% | 5 | 44 | 5 | 1.21% | 65.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1369.50 | 1332.16 | 1327.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 10:15:00 | 1373.50 | 1357.88 | 1345.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 13:15:00 | 1371.90 | 1372.44 | 1363.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 14:00:00 | 1371.90 | 1372.44 | 1363.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 1395.50 | 1401.94 | 1395.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 12:00:00 | 1395.50 | 1401.94 | 1395.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 1396.10 | 1400.77 | 1395.46 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 1387.80 | 1394.46 | 1394.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 1385.00 | 1392.56 | 1393.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 15:15:00 | 1387.60 | 1384.84 | 1388.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 15:15:00 | 1387.60 | 1384.84 | 1388.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 15:15:00 | 1387.60 | 1384.84 | 1388.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:15:00 | 1404.40 | 1384.84 | 1388.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 1383.30 | 1384.53 | 1387.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 10:15:00 | 1378.60 | 1384.53 | 1387.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 11:15:00 | 1395.30 | 1382.61 | 1382.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 1395.30 | 1382.61 | 1382.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 12:15:00 | 1396.90 | 1385.47 | 1383.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 15:15:00 | 1396.70 | 1398.94 | 1394.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 09:15:00 | 1392.00 | 1398.94 | 1394.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1392.00 | 1397.56 | 1393.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 1392.00 | 1397.56 | 1393.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1403.80 | 1398.80 | 1394.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 1411.90 | 1398.80 | 1394.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 15:15:00 | 1404.80 | 1401.52 | 1397.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 15:15:00 | 1434.00 | 1440.81 | 1441.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 15:15:00 | 1434.00 | 1440.81 | 1441.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 09:15:00 | 1426.10 | 1437.87 | 1439.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 13:15:00 | 1438.80 | 1437.20 | 1438.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 13:15:00 | 1438.80 | 1437.20 | 1438.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 1438.80 | 1437.20 | 1438.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:00:00 | 1438.80 | 1437.20 | 1438.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 1436.00 | 1436.96 | 1438.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 15:15:00 | 1434.30 | 1436.96 | 1438.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 09:15:00 | 1461.70 | 1441.48 | 1440.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 1461.70 | 1441.48 | 1440.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 09:15:00 | 1463.00 | 1454.30 | 1448.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 14:15:00 | 1467.60 | 1469.97 | 1464.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 15:00:00 | 1467.60 | 1469.97 | 1464.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1464.80 | 1473.63 | 1470.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:00:00 | 1464.80 | 1473.63 | 1470.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 1465.80 | 1472.07 | 1470.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:30:00 | 1464.00 | 1472.07 | 1470.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-06-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 12:15:00 | 1459.00 | 1467.63 | 1468.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 13:15:00 | 1454.00 | 1464.91 | 1467.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 12:15:00 | 1459.00 | 1457.63 | 1461.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 12:15:00 | 1459.00 | 1457.63 | 1461.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1459.00 | 1457.63 | 1461.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:45:00 | 1459.10 | 1457.63 | 1461.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 1358.90 | 1349.91 | 1355.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:45:00 | 1360.40 | 1349.91 | 1355.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 1360.00 | 1351.93 | 1355.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:30:00 | 1363.10 | 1351.93 | 1355.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 1354.00 | 1353.07 | 1355.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:30:00 | 1355.00 | 1353.07 | 1355.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 1351.10 | 1352.68 | 1355.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:15:00 | 1404.80 | 1352.68 | 1355.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 1402.20 | 1362.58 | 1359.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 12:15:00 | 1429.30 | 1406.61 | 1394.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 15:15:00 | 1444.10 | 1444.50 | 1434.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 09:15:00 | 1447.80 | 1444.50 | 1434.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 1448.00 | 1446.44 | 1441.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:30:00 | 1443.20 | 1446.44 | 1441.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 1449.00 | 1447.22 | 1443.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 1443.40 | 1447.22 | 1443.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 1445.00 | 1446.78 | 1443.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:30:00 | 1444.50 | 1446.78 | 1443.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 1442.00 | 1445.82 | 1443.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:15:00 | 1439.00 | 1445.82 | 1443.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1447.00 | 1446.06 | 1443.62 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 13:15:00 | 1433.20 | 1440.75 | 1441.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 15:15:00 | 1429.00 | 1436.99 | 1439.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 12:15:00 | 1428.90 | 1426.23 | 1430.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 12:15:00 | 1428.90 | 1426.23 | 1430.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 1428.90 | 1426.23 | 1430.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:45:00 | 1429.20 | 1426.23 | 1430.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 1430.70 | 1427.13 | 1430.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 14:00:00 | 1430.70 | 1427.13 | 1430.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 1436.40 | 1428.98 | 1431.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 15:00:00 | 1436.40 | 1428.98 | 1431.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 1435.60 | 1430.31 | 1431.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:15:00 | 1441.70 | 1430.31 | 1431.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 1432.40 | 1431.00 | 1431.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:45:00 | 1433.70 | 1431.00 | 1431.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 1433.30 | 1431.46 | 1431.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:00:00 | 1433.30 | 1431.46 | 1431.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 12:15:00 | 1437.80 | 1432.73 | 1432.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 13:15:00 | 1440.20 | 1434.22 | 1433.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 12:15:00 | 1440.30 | 1443.58 | 1439.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 13:00:00 | 1440.30 | 1443.58 | 1439.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 1445.70 | 1444.00 | 1439.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 1447.20 | 1442.66 | 1441.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 1433.80 | 1440.46 | 1440.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 1433.80 | 1440.46 | 1440.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 15:15:00 | 1428.30 | 1435.89 | 1438.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 1437.00 | 1436.11 | 1438.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 10:00:00 | 1437.00 | 1436.11 | 1438.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1438.90 | 1436.67 | 1438.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:30:00 | 1440.00 | 1436.67 | 1438.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 1435.60 | 1436.46 | 1437.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 12:45:00 | 1432.50 | 1435.94 | 1437.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 13:30:00 | 1431.40 | 1434.76 | 1436.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 15:00:00 | 1432.50 | 1434.30 | 1436.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 09:30:00 | 1430.80 | 1434.21 | 1436.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 1442.70 | 1435.91 | 1436.65 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 1442.70 | 1435.91 | 1436.65 | SL hit (close>static) qty=1.00 sl=1438.90 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 1442.50 | 1437.23 | 1437.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 15:15:00 | 1447.00 | 1441.52 | 1439.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 1452.00 | 1454.97 | 1449.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 1452.00 | 1454.97 | 1449.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1452.00 | 1454.97 | 1449.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:30:00 | 1452.50 | 1454.97 | 1449.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 1454.30 | 1454.84 | 1450.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 11:30:00 | 1457.40 | 1454.39 | 1450.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 12:45:00 | 1455.80 | 1453.87 | 1450.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 1441.80 | 1450.57 | 1449.99 | SL hit (close<static) qty=1.00 sl=1449.60 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 1433.50 | 1447.15 | 1448.49 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 15:15:00 | 1449.00 | 1445.35 | 1445.22 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 09:15:00 | 1443.40 | 1444.96 | 1445.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 11:15:00 | 1439.20 | 1443.61 | 1444.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 1388.90 | 1385.18 | 1393.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:00:00 | 1388.90 | 1385.18 | 1393.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 1392.80 | 1386.70 | 1393.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:00:00 | 1392.80 | 1386.70 | 1393.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 1395.20 | 1388.40 | 1393.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 1395.20 | 1388.40 | 1393.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 1397.00 | 1390.12 | 1393.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 1392.90 | 1390.12 | 1393.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1398.20 | 1391.74 | 1394.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:00:00 | 1398.20 | 1391.74 | 1394.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 1400.40 | 1393.47 | 1394.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:15:00 | 1401.90 | 1393.47 | 1394.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 1398.60 | 1394.50 | 1395.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 12:30:00 | 1397.20 | 1394.60 | 1395.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 13:00:00 | 1395.00 | 1394.60 | 1395.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 15:15:00 | 1395.00 | 1376.91 | 1375.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 1395.00 | 1376.91 | 1375.75 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 13:15:00 | 1359.60 | 1373.67 | 1374.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 1345.40 | 1363.06 | 1367.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 1345.00 | 1342.62 | 1353.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 14:45:00 | 1341.80 | 1342.62 | 1353.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1330.70 | 1341.01 | 1351.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:30:00 | 1327.30 | 1339.27 | 1349.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 12:00:00 | 1326.30 | 1336.68 | 1347.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:00:00 | 1329.70 | 1327.96 | 1337.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:30:00 | 1324.10 | 1327.94 | 1336.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 1336.10 | 1329.58 | 1336.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 13:00:00 | 1336.10 | 1329.58 | 1336.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 1340.40 | 1331.74 | 1336.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 13:30:00 | 1339.40 | 1331.74 | 1336.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 1340.40 | 1333.47 | 1337.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:45:00 | 1343.00 | 1333.47 | 1337.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 1345.70 | 1337.41 | 1338.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:00:00 | 1345.70 | 1337.41 | 1338.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 1334.30 | 1336.79 | 1338.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 14:45:00 | 1332.40 | 1336.33 | 1337.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 09:30:00 | 1329.10 | 1334.88 | 1336.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 10:30:00 | 1330.80 | 1332.93 | 1335.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 15:15:00 | 1330.00 | 1319.61 | 1318.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 15:15:00 | 1330.00 | 1319.61 | 1318.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 1335.40 | 1322.77 | 1320.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 12:15:00 | 1363.70 | 1366.41 | 1357.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 12:30:00 | 1362.80 | 1366.41 | 1357.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 1360.00 | 1363.75 | 1358.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 1346.90 | 1363.75 | 1358.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1357.20 | 1362.44 | 1358.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 1358.90 | 1362.44 | 1358.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1342.80 | 1358.51 | 1356.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 1342.80 | 1358.51 | 1356.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 1342.90 | 1355.39 | 1355.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 09:15:00 | 1338.80 | 1345.98 | 1350.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 13:15:00 | 1342.90 | 1341.76 | 1346.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 14:00:00 | 1342.90 | 1341.76 | 1346.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 1335.50 | 1326.01 | 1332.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:00:00 | 1335.50 | 1326.01 | 1332.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 1330.50 | 1326.91 | 1332.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:30:00 | 1337.10 | 1326.91 | 1332.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1328.10 | 1318.38 | 1322.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:30:00 | 1330.10 | 1318.38 | 1322.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1333.20 | 1321.34 | 1323.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 1333.20 | 1321.34 | 1323.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 1338.40 | 1324.75 | 1324.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 1342.00 | 1333.07 | 1329.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 1335.50 | 1336.96 | 1332.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 1335.50 | 1336.96 | 1332.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 1335.50 | 1336.96 | 1332.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 1335.50 | 1336.96 | 1332.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 1338.70 | 1337.27 | 1333.79 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 1329.50 | 1333.80 | 1334.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 1327.50 | 1332.54 | 1333.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1342.40 | 1327.41 | 1328.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 1342.40 | 1327.41 | 1328.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1342.40 | 1327.41 | 1328.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 1342.40 | 1327.41 | 1328.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 1346.70 | 1331.27 | 1330.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 11:15:00 | 1353.50 | 1335.72 | 1332.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 12:15:00 | 1393.60 | 1395.88 | 1383.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 13:00:00 | 1393.60 | 1395.88 | 1383.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 1395.30 | 1398.32 | 1394.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:30:00 | 1395.20 | 1398.32 | 1394.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 1395.00 | 1397.65 | 1394.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 15:15:00 | 1395.10 | 1397.65 | 1394.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 1395.10 | 1397.14 | 1394.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 1403.50 | 1397.14 | 1394.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 1433.90 | 1437.83 | 1437.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 09:15:00 | 1433.90 | 1437.83 | 1437.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 13:15:00 | 1418.40 | 1429.18 | 1433.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 15:15:00 | 1392.00 | 1391.11 | 1401.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 09:15:00 | 1390.40 | 1391.11 | 1401.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 1398.00 | 1395.97 | 1400.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:30:00 | 1396.00 | 1399.68 | 1401.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 12:15:00 | 1395.00 | 1399.30 | 1400.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 1415.30 | 1402.63 | 1402.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 1415.30 | 1402.63 | 1402.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 1421.90 | 1406.48 | 1403.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 1404.60 | 1413.96 | 1411.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 1404.60 | 1413.96 | 1411.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 1404.60 | 1413.96 | 1411.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:00:00 | 1404.60 | 1413.96 | 1411.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 1400.40 | 1411.25 | 1410.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 1399.20 | 1411.25 | 1410.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 11:15:00 | 1400.00 | 1409.00 | 1409.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 1387.00 | 1400.21 | 1403.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 12:15:00 | 1399.50 | 1398.60 | 1402.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-08 13:00:00 | 1399.50 | 1398.60 | 1402.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 1398.50 | 1398.58 | 1401.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 13:30:00 | 1400.70 | 1398.58 | 1401.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 1394.70 | 1394.66 | 1397.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:00:00 | 1394.70 | 1394.66 | 1397.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 1395.00 | 1394.73 | 1397.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 1395.00 | 1394.73 | 1397.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 1395.40 | 1394.87 | 1397.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 1408.10 | 1394.87 | 1397.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1414.50 | 1398.79 | 1398.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:30:00 | 1412.40 | 1398.79 | 1398.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 1411.30 | 1401.29 | 1399.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 10:15:00 | 1430.70 | 1413.01 | 1407.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 10:15:00 | 1427.10 | 1428.70 | 1419.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 11:00:00 | 1427.10 | 1428.70 | 1419.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 1470.00 | 1475.20 | 1469.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 12:30:00 | 1471.00 | 1475.20 | 1469.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 1465.10 | 1473.18 | 1468.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:45:00 | 1463.30 | 1473.18 | 1468.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 1466.60 | 1471.86 | 1468.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 1477.40 | 1471.33 | 1468.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 11:00:00 | 1469.80 | 1471.94 | 1469.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 11:15:00 | 1461.30 | 1469.81 | 1468.97 | SL hit (close<static) qty=1.00 sl=1463.80 alert=retest2 |

### Cycle 26 — SELL (started 2025-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 12:15:00 | 1462.00 | 1468.25 | 1468.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 13:15:00 | 1457.00 | 1466.00 | 1467.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 1423.00 | 1422.74 | 1432.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 1423.00 | 1422.74 | 1432.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1423.00 | 1422.74 | 1432.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 1431.30 | 1422.74 | 1432.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1435.70 | 1422.46 | 1427.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 1435.70 | 1422.46 | 1427.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1451.90 | 1428.35 | 1429.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 1451.90 | 1428.35 | 1429.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 1450.00 | 1432.68 | 1431.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 13:15:00 | 1462.70 | 1442.74 | 1436.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 09:15:00 | 1452.80 | 1454.94 | 1448.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 09:15:00 | 1452.80 | 1454.94 | 1448.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1452.80 | 1454.94 | 1448.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 12:45:00 | 1459.60 | 1453.25 | 1449.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 1443.00 | 1450.55 | 1449.33 | SL hit (close<static) qty=1.00 sl=1448.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 11:15:00 | 1440.00 | 1446.93 | 1447.80 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 12:15:00 | 1459.70 | 1449.28 | 1448.33 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 13:15:00 | 1440.30 | 1447.48 | 1447.60 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 10:15:00 | 1452.40 | 1448.34 | 1447.86 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 13:15:00 | 1440.80 | 1446.98 | 1447.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 1436.00 | 1444.79 | 1446.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 1450.80 | 1442.19 | 1444.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 10:15:00 | 1450.80 | 1442.19 | 1444.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 1450.80 | 1442.19 | 1444.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:00:00 | 1450.80 | 1442.19 | 1444.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 1452.60 | 1444.27 | 1445.18 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 12:15:00 | 1454.50 | 1446.32 | 1446.03 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 1441.90 | 1446.48 | 1447.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 10:15:00 | 1440.50 | 1445.28 | 1446.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 12:15:00 | 1445.00 | 1444.25 | 1445.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 12:15:00 | 1445.00 | 1444.25 | 1445.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 1445.00 | 1444.25 | 1445.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:00:00 | 1445.00 | 1444.25 | 1445.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 13:15:00 | 1463.80 | 1448.16 | 1447.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 14:15:00 | 1474.90 | 1453.51 | 1449.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 12:15:00 | 1498.00 | 1500.35 | 1486.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 12:45:00 | 1501.60 | 1500.35 | 1486.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1505.80 | 1507.86 | 1504.39 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 12:15:00 | 1495.20 | 1501.73 | 1502.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 10:15:00 | 1480.20 | 1494.12 | 1498.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 1499.80 | 1490.24 | 1493.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 1499.80 | 1490.24 | 1493.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1499.80 | 1490.24 | 1493.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:30:00 | 1498.00 | 1490.24 | 1493.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1498.50 | 1491.89 | 1494.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:30:00 | 1498.40 | 1491.89 | 1494.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 1493.30 | 1494.05 | 1494.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 1483.60 | 1493.32 | 1494.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 1515.40 | 1488.83 | 1486.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 1515.40 | 1488.83 | 1486.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 11:15:00 | 1529.80 | 1514.53 | 1506.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 14:15:00 | 1516.30 | 1518.45 | 1510.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 15:00:00 | 1516.30 | 1518.45 | 1510.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1517.00 | 1526.88 | 1521.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 1517.00 | 1526.88 | 1521.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 1520.60 | 1525.63 | 1521.45 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 1508.90 | 1517.62 | 1518.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 1505.00 | 1515.09 | 1517.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 1505.10 | 1504.94 | 1510.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 10:00:00 | 1505.10 | 1504.94 | 1510.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 1507.00 | 1502.98 | 1507.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:00:00 | 1507.00 | 1502.98 | 1507.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 1504.80 | 1503.34 | 1507.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 1501.00 | 1504.22 | 1507.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 13:15:00 | 1511.60 | 1504.87 | 1506.10 | SL hit (close>static) qty=1.00 sl=1509.10 alert=retest2 |

### Cycle 39 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 1512.80 | 1496.39 | 1495.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 1524.00 | 1514.06 | 1508.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1509.90 | 1515.68 | 1510.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 1509.90 | 1515.68 | 1510.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1509.90 | 1515.68 | 1510.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:00:00 | 1509.90 | 1515.68 | 1510.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 1514.60 | 1515.46 | 1511.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 12:00:00 | 1517.00 | 1515.77 | 1511.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 1504.30 | 1511.92 | 1511.35 | SL hit (close<static) qty=1.00 sl=1509.40 alert=retest2 |

### Cycle 40 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 1501.50 | 1509.83 | 1510.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 14:15:00 | 1500.60 | 1504.94 | 1507.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 1490.70 | 1488.00 | 1494.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:00:00 | 1490.70 | 1488.00 | 1494.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 1496.20 | 1490.27 | 1493.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 1496.20 | 1490.27 | 1493.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 1496.00 | 1491.42 | 1493.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 1502.00 | 1491.42 | 1493.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 1494.10 | 1493.45 | 1494.36 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 1497.00 | 1494.97 | 1494.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 1507.20 | 1497.41 | 1495.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 1505.00 | 1505.62 | 1502.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 11:00:00 | 1505.00 | 1505.62 | 1502.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 1501.80 | 1504.86 | 1502.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:45:00 | 1500.50 | 1504.86 | 1502.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 1498.50 | 1503.59 | 1501.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 1498.50 | 1503.59 | 1501.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 1501.00 | 1503.07 | 1501.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:30:00 | 1498.60 | 1503.07 | 1501.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 15:15:00 | 1492.80 | 1499.60 | 1500.23 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 10:15:00 | 1503.90 | 1500.68 | 1500.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-24 11:15:00 | 1506.50 | 1501.85 | 1501.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 13:15:00 | 1501.60 | 1502.40 | 1501.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 13:15:00 | 1501.60 | 1502.40 | 1501.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 1501.60 | 1502.40 | 1501.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:00:00 | 1501.60 | 1502.40 | 1501.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 1494.20 | 1500.76 | 1500.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 10:15:00 | 1488.90 | 1496.68 | 1498.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 12:15:00 | 1466.00 | 1464.43 | 1473.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 13:00:00 | 1466.00 | 1464.43 | 1473.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1467.50 | 1465.70 | 1471.27 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 10:15:00 | 1485.90 | 1474.71 | 1473.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 1494.50 | 1482.28 | 1478.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 1484.00 | 1486.95 | 1483.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 1484.00 | 1486.95 | 1483.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1484.00 | 1486.95 | 1483.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:00:00 | 1484.00 | 1486.95 | 1483.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1484.90 | 1486.54 | 1483.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:15:00 | 1482.50 | 1486.54 | 1483.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 1484.60 | 1486.15 | 1483.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 12:30:00 | 1488.90 | 1487.42 | 1484.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 1477.60 | 1487.06 | 1485.31 | SL hit (close<static) qty=1.00 sl=1481.10 alert=retest2 |

### Cycle 46 — SELL (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 12:15:00 | 1478.20 | 1483.38 | 1483.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 1475.60 | 1481.82 | 1483.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 1484.00 | 1471.63 | 1475.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 1484.00 | 1471.63 | 1475.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1484.00 | 1471.63 | 1475.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 1484.00 | 1471.63 | 1475.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 1483.10 | 1473.93 | 1475.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:30:00 | 1484.50 | 1473.93 | 1475.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 1469.50 | 1472.64 | 1474.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 14:15:00 | 1467.30 | 1472.27 | 1474.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 15:00:00 | 1466.70 | 1471.16 | 1473.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 1393.93 | 1403.23 | 1413.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 1393.37 | 1403.23 | 1413.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 12:15:00 | 1386.50 | 1379.09 | 1392.52 | SL hit (close>ema200) qty=0.50 sl=1379.09 alert=retest2 |

### Cycle 47 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 1410.00 | 1395.02 | 1394.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 15:15:00 | 1416.10 | 1403.92 | 1399.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 1396.90 | 1402.51 | 1399.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 1396.90 | 1402.51 | 1399.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 1396.90 | 1402.51 | 1399.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:45:00 | 1395.30 | 1402.51 | 1399.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 1400.30 | 1402.07 | 1399.23 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 1355.20 | 1391.07 | 1394.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 1302.10 | 1373.28 | 1386.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 1367.10 | 1354.54 | 1372.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 10:00:00 | 1367.10 | 1354.54 | 1372.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 1363.60 | 1352.50 | 1364.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 14:45:00 | 1366.20 | 1352.50 | 1364.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 1365.00 | 1355.00 | 1364.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 1374.00 | 1355.00 | 1364.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1377.10 | 1359.42 | 1365.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:45:00 | 1380.90 | 1359.42 | 1365.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 1377.30 | 1363.00 | 1366.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:45:00 | 1378.40 | 1363.00 | 1366.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 1380.30 | 1369.23 | 1369.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 12:15:00 | 1390.50 | 1380.18 | 1375.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 1407.50 | 1414.98 | 1405.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 1407.50 | 1414.98 | 1405.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1407.50 | 1414.98 | 1405.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1407.50 | 1414.98 | 1405.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1385.20 | 1409.02 | 1403.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1388.00 | 1409.02 | 1403.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 1372.50 | 1401.72 | 1400.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 1378.20 | 1401.72 | 1400.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 1351.60 | 1391.69 | 1396.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 1348.00 | 1382.95 | 1391.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 1387.50 | 1381.95 | 1388.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 12:15:00 | 1387.50 | 1381.95 | 1388.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 1387.50 | 1381.95 | 1388.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:00:00 | 1387.50 | 1381.95 | 1388.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 1391.50 | 1383.86 | 1388.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:45:00 | 1393.60 | 1383.86 | 1388.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 1404.70 | 1388.03 | 1390.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 1404.70 | 1388.03 | 1390.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 1402.50 | 1390.92 | 1391.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 1511.10 | 1390.92 | 1391.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1502.40 | 1413.22 | 1401.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 13:15:00 | 1526.20 | 1474.86 | 1437.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 1556.80 | 1559.99 | 1534.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 10:00:00 | 1556.80 | 1559.99 | 1534.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 1530.00 | 1549.35 | 1535.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:00:00 | 1530.00 | 1549.35 | 1535.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 1542.40 | 1547.96 | 1536.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:30:00 | 1534.80 | 1547.96 | 1536.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1550.40 | 1557.91 | 1550.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:00:00 | 1550.40 | 1557.91 | 1550.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 1547.00 | 1555.73 | 1550.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:30:00 | 1540.60 | 1555.73 | 1550.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 1556.10 | 1555.80 | 1550.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 12:15:00 | 1557.80 | 1555.80 | 1550.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 09:15:00 | 1544.70 | 1552.65 | 1551.27 | SL hit (close<static) qty=1.00 sl=1545.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 11:15:00 | 1546.30 | 1549.73 | 1550.08 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 14:15:00 | 1553.90 | 1550.90 | 1550.54 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 1544.50 | 1549.19 | 1549.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 10:15:00 | 1539.30 | 1547.21 | 1548.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 14:15:00 | 1543.20 | 1542.86 | 1545.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 14:15:00 | 1543.20 | 1542.86 | 1545.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 1543.20 | 1542.86 | 1545.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 15:00:00 | 1543.20 | 1542.86 | 1545.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 1510.70 | 1536.51 | 1542.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:15:00 | 1508.00 | 1526.59 | 1534.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 1557.30 | 1537.25 | 1534.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 1557.30 | 1537.25 | 1534.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 11:15:00 | 1564.50 | 1545.88 | 1539.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 1544.30 | 1554.88 | 1547.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 1544.30 | 1554.88 | 1547.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 1544.30 | 1554.88 | 1547.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:45:00 | 1541.90 | 1554.88 | 1547.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 1544.70 | 1552.85 | 1547.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:45:00 | 1546.80 | 1552.85 | 1547.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 1552.10 | 1552.70 | 1547.57 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 1533.90 | 1545.43 | 1546.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 1529.90 | 1542.33 | 1544.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 1550.30 | 1523.96 | 1528.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 1550.30 | 1523.96 | 1528.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1550.30 | 1523.96 | 1528.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:00:00 | 1550.30 | 1523.96 | 1528.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1553.00 | 1529.77 | 1530.31 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 1546.40 | 1533.10 | 1531.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 15:15:00 | 1561.90 | 1549.63 | 1543.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 11:15:00 | 1554.30 | 1557.15 | 1549.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 11:45:00 | 1553.80 | 1557.15 | 1549.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 1541.20 | 1553.96 | 1548.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:00:00 | 1541.20 | 1553.96 | 1548.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 1528.30 | 1548.83 | 1546.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:00:00 | 1528.30 | 1548.83 | 1546.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 14:15:00 | 1528.60 | 1544.78 | 1544.98 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 14:15:00 | 1551.70 | 1544.19 | 1543.93 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 1530.60 | 1542.08 | 1543.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 1519.90 | 1535.98 | 1539.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 1460.30 | 1448.71 | 1469.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 14:15:00 | 1495.50 | 1463.23 | 1468.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 1495.50 | 1463.23 | 1468.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 1495.50 | 1463.23 | 1468.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 1488.00 | 1468.19 | 1470.57 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 1490.20 | 1472.59 | 1472.35 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 1429.60 | 1469.52 | 1472.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 1395.10 | 1416.82 | 1428.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 11:15:00 | 1371.70 | 1370.55 | 1384.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 11:30:00 | 1373.00 | 1370.55 | 1384.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1373.00 | 1367.04 | 1379.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 15:00:00 | 1373.00 | 1367.04 | 1379.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1366.80 | 1367.85 | 1377.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 1353.80 | 1366.54 | 1376.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:15:00 | 1353.20 | 1365.09 | 1374.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 10:15:00 | 1386.00 | 1372.58 | 1374.23 | SL hit (close>static) qty=1.00 sl=1379.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 1399.60 | 1377.99 | 1376.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 1408.80 | 1388.28 | 1381.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1369.90 | 1388.63 | 1383.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1369.90 | 1388.63 | 1383.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1369.90 | 1388.63 | 1383.89 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 1367.60 | 1380.45 | 1380.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1352.40 | 1370.14 | 1375.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1378.70 | 1370.23 | 1374.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 1378.70 | 1370.23 | 1374.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1378.70 | 1370.23 | 1374.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 1377.00 | 1370.23 | 1374.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 1379.30 | 1372.04 | 1374.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 1377.40 | 1372.04 | 1374.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 1374.50 | 1372.53 | 1374.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 1368.40 | 1372.53 | 1374.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:45:00 | 1371.50 | 1372.57 | 1374.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:45:00 | 1369.40 | 1371.75 | 1374.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 1299.98 | 1356.34 | 1366.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 1302.92 | 1356.34 | 1366.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 1300.93 | 1356.34 | 1366.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 1325.70 | 1320.32 | 1338.60 | SL hit (close>ema200) qty=0.50 sl=1320.32 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1388.00 | 1346.70 | 1343.88 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 1340.60 | 1354.38 | 1354.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1323.00 | 1342.12 | 1348.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1377.60 | 1335.15 | 1339.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1377.60 | 1335.15 | 1339.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1377.60 | 1335.15 | 1339.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 1377.60 | 1335.15 | 1339.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 1378.20 | 1343.76 | 1342.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 11:15:00 | 1394.90 | 1353.99 | 1347.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 1344.90 | 1368.72 | 1359.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 1344.90 | 1368.72 | 1359.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 1344.90 | 1368.72 | 1359.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 13:15:00 | 1364.90 | 1359.64 | 1357.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:00:00 | 1360.70 | 1366.54 | 1361.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:30:00 | 1360.80 | 1363.93 | 1361.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 11:15:00 | 1360.40 | 1363.93 | 1361.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 11:15:00 | 1357.40 | 1362.63 | 1360.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 12:15:00 | 1362.60 | 1362.63 | 1360.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-08 10:15:00 | 1501.39 | 1406.69 | 1387.76 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 12:15:00 | 1560.00 | 1583.39 | 1586.27 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 1630.10 | 1592.25 | 1589.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 10:15:00 | 1648.80 | 1603.56 | 1594.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 1613.20 | 1645.64 | 1637.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 1613.20 | 1645.64 | 1637.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1613.20 | 1645.64 | 1637.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 1613.20 | 1645.64 | 1637.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 1629.30 | 1642.37 | 1636.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 11:15:00 | 1633.50 | 1642.37 | 1636.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 12:00:00 | 1633.80 | 1640.66 | 1636.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 12:45:00 | 1633.70 | 1638.13 | 1635.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 13:15:00 | 1600.00 | 1630.50 | 1632.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 13:15:00 | 1600.00 | 1630.50 | 1632.16 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 14:15:00 | 1661.80 | 1636.76 | 1634.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 15:15:00 | 1675.50 | 1644.51 | 1638.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 14:15:00 | 1726.00 | 1727.29 | 1705.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 15:00:00 | 1726.00 | 1727.29 | 1705.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 1718.30 | 1732.94 | 1721.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 10:00:00 | 1718.30 | 1732.94 | 1721.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 1727.30 | 1731.81 | 1722.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 12:00:00 | 1732.60 | 1731.97 | 1723.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 12:30:00 | 1735.80 | 1732.68 | 1724.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 13:45:00 | 1735.50 | 1732.94 | 1725.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 15:15:00 | 1734.80 | 1732.61 | 1725.69 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1771.20 | 1740.68 | 1730.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 10:15:00 | 1782.00 | 1740.68 | 1730.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-22 10:15:00 | 1378.60 | 2025-05-23 11:15:00 | 1395.30 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-05-27 11:15:00 | 1411.90 | 2025-06-03 15:15:00 | 1434.00 | STOP_HIT | 1.00 | 1.57% |
| BUY | retest2 | 2025-05-27 15:15:00 | 1404.80 | 2025-06-03 15:15:00 | 1434.00 | STOP_HIT | 1.00 | 2.08% |
| SELL | retest2 | 2025-06-04 15:15:00 | 1434.30 | 2025-06-05 09:15:00 | 1461.70 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-07-11 09:15:00 | 1447.20 | 2025-07-11 10:15:00 | 1433.80 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-07-14 12:45:00 | 1432.50 | 2025-07-15 10:15:00 | 1442.70 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-07-14 13:30:00 | 1431.40 | 2025-07-15 10:15:00 | 1442.70 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-07-14 15:00:00 | 1432.50 | 2025-07-15 10:15:00 | 1442.70 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-07-15 09:30:00 | 1430.80 | 2025-07-15 10:15:00 | 1442.70 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-07-17 11:30:00 | 1457.40 | 2025-07-18 09:15:00 | 1441.80 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-07-17 12:45:00 | 1455.80 | 2025-07-18 09:15:00 | 1441.80 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-07-30 12:30:00 | 1397.20 | 2025-08-04 15:15:00 | 1395.00 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2025-07-30 13:00:00 | 1395.00 | 2025-08-04 15:15:00 | 1395.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-08-08 10:30:00 | 1327.30 | 2025-08-18 15:15:00 | 1330.00 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-08-08 12:00:00 | 1326.30 | 2025-08-18 15:15:00 | 1330.00 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-08-11 11:00:00 | 1329.70 | 2025-08-18 15:15:00 | 1330.00 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-08-11 11:30:00 | 1324.10 | 2025-08-18 15:15:00 | 1330.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-08-12 14:45:00 | 1332.40 | 2025-08-18 15:15:00 | 1330.00 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2025-08-13 09:30:00 | 1329.10 | 2025-08-18 15:15:00 | 1330.00 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-08-13 10:30:00 | 1330.80 | 2025-08-18 15:15:00 | 1330.00 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-09-16 09:15:00 | 1403.50 | 2025-09-25 09:15:00 | 1433.90 | STOP_HIT | 1.00 | 2.17% |
| SELL | retest2 | 2025-10-01 10:30:00 | 1396.00 | 2025-10-01 13:15:00 | 1415.30 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-10-01 12:15:00 | 1395.00 | 2025-10-01 13:15:00 | 1415.30 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-10-21 13:45:00 | 1477.40 | 2025-10-23 11:15:00 | 1461.30 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-10-23 11:00:00 | 1469.80 | 2025-10-23 11:15:00 | 1461.30 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-10-23 11:30:00 | 1470.10 | 2025-10-23 12:15:00 | 1462.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-10-31 12:45:00 | 1459.60 | 2025-11-03 09:15:00 | 1443.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-11-21 09:15:00 | 1483.60 | 2025-11-26 09:15:00 | 1515.40 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-12-05 09:15:00 | 1501.00 | 2025-12-05 13:15:00 | 1511.60 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-12-08 09:45:00 | 1500.00 | 2025-12-10 09:15:00 | 1512.80 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-12-15 12:00:00 | 1517.00 | 2025-12-16 09:15:00 | 1504.30 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2026-01-05 12:30:00 | 1488.90 | 2026-01-06 09:15:00 | 1477.60 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-01-08 14:15:00 | 1467.30 | 2026-01-20 11:15:00 | 1393.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 15:00:00 | 1466.70 | 2026-01-20 11:15:00 | 1393.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 14:15:00 | 1467.30 | 2026-01-21 12:15:00 | 1386.50 | STOP_HIT | 0.50 | 5.51% |
| SELL | retest2 | 2026-01-08 15:00:00 | 1466.70 | 2026-01-21 12:15:00 | 1386.50 | STOP_HIT | 0.50 | 5.47% |
| BUY | retest2 | 2026-02-10 12:15:00 | 1557.80 | 2026-02-11 09:15:00 | 1544.70 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2026-02-13 15:15:00 | 1508.00 | 2026-02-17 09:15:00 | 1557.30 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2026-03-17 11:15:00 | 1353.80 | 2026-03-18 10:15:00 | 1386.00 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2026-03-17 12:15:00 | 1353.20 | 2026-03-18 10:15:00 | 1386.00 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2026-03-20 12:15:00 | 1368.40 | 2026-03-23 09:15:00 | 1299.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:45:00 | 1371.50 | 2026-03-23 09:15:00 | 1302.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 13:45:00 | 1369.40 | 2026-03-23 09:15:00 | 1300.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 1368.40 | 2026-03-24 09:15:00 | 1325.70 | STOP_HIT | 0.50 | 3.12% |
| SELL | retest2 | 2026-03-20 12:45:00 | 1371.50 | 2026-03-24 09:15:00 | 1325.70 | STOP_HIT | 0.50 | 3.34% |
| SELL | retest2 | 2026-03-20 13:45:00 | 1369.40 | 2026-03-24 09:15:00 | 1325.70 | STOP_HIT | 0.50 | 3.19% |
| BUY | retest2 | 2026-04-02 13:15:00 | 1364.90 | 2026-04-08 10:15:00 | 1501.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 10:00:00 | 1360.70 | 2026-04-08 10:15:00 | 1496.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 10:30:00 | 1360.80 | 2026-04-08 10:15:00 | 1496.88 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 11:15:00 | 1360.40 | 2026-04-08 10:15:00 | 1496.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 12:15:00 | 1362.60 | 2026-04-08 10:15:00 | 1498.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-30 11:15:00 | 1633.50 | 2026-04-30 13:15:00 | 1600.00 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2026-04-30 12:00:00 | 1633.80 | 2026-04-30 13:15:00 | 1600.00 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2026-04-30 12:45:00 | 1633.70 | 2026-04-30 13:15:00 | 1600.00 | STOP_HIT | 1.00 | -2.06% |
