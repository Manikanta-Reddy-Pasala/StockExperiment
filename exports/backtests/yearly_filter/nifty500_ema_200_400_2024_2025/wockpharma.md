# Wockhardt Ltd. (WOCKPHARMA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1611.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT2_SKIP | 3 |
| ALERT3 | 51 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 58 |
| PARTIAL | 14 |
| TARGET_HIT | 16 |
| STOP_HIT | 42 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 72 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 32 / 40
- **Target hits / Stop hits / Partials:** 16 / 42 / 14
- **Avg / median % per leg:** 1.67% / -0.40%
- **Sum % (uncompounded):** 120.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 4 | 11.8% | 4 | 30 | 0 | -1.64% | -55.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 34 | 4 | 11.8% | 4 | 30 | 0 | -1.64% | -55.8% |
| SELL (all) | 38 | 28 | 73.7% | 12 | 12 | 14 | 4.64% | 176.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 38 | 28 | 73.7% | 12 | 12 | 14 | 4.64% | 176.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 72 | 32 | 44.4% | 16 | 42 | 14 | 1.67% | 120.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-03-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 12:15:00 | 1203.05 | 1363.94 | 1364.49 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-03-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 15:15:00 | 1526.00 | 1355.93 | 1355.80 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 11:15:00 | 1190.95 | 1362.11 | 1362.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-09 13:15:00 | 1179.40 | 1358.59 | 1360.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 09:15:00 | 1394.80 | 1352.57 | 1357.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-15 09:15:00 | 1394.80 | 1352.57 | 1357.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 09:15:00 | 1394.80 | 1352.57 | 1357.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 09:45:00 | 1383.10 | 1352.57 | 1357.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 10:15:00 | 1383.90 | 1352.88 | 1357.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-15 12:00:00 | 1382.10 | 1353.17 | 1357.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-15 12:30:00 | 1377.50 | 1353.45 | 1357.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 14:15:00 | 1406.10 | 1354.23 | 1358.24 | SL hit (close>static) qty=1.00 sl=1404.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 12:15:00 | 1402.60 | 1361.96 | 1361.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 09:15:00 | 1418.20 | 1363.59 | 1362.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 13:15:00 | 1370.00 | 1379.96 | 1371.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 13:15:00 | 1370.00 | 1379.96 | 1371.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 13:15:00 | 1370.00 | 1379.96 | 1371.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 14:00:00 | 1370.00 | 1379.96 | 1371.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 1366.00 | 1379.82 | 1371.95 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 10:15:00 | 1264.90 | 1364.31 | 1364.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 11:15:00 | 1258.50 | 1363.26 | 1364.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 12:15:00 | 1337.50 | 1321.20 | 1339.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 12:15:00 | 1337.50 | 1321.20 | 1339.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 12:15:00 | 1337.50 | 1321.20 | 1339.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 13:00:00 | 1337.50 | 1321.20 | 1339.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 1381.90 | 1322.02 | 1339.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:00:00 | 1381.90 | 1322.02 | 1339.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 1374.40 | 1322.54 | 1339.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:45:00 | 1381.00 | 1322.54 | 1339.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 1325.90 | 1325.63 | 1340.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 12:30:00 | 1306.10 | 1326.00 | 1339.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 11:15:00 | 1315.30 | 1325.65 | 1339.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 15:15:00 | 1343.00 | 1326.31 | 1339.31 | SL hit (close>static) qty=1.00 sl=1341.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 10:15:00 | 1484.30 | 1349.49 | 1348.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 09:15:00 | 1512.00 | 1357.83 | 1353.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 13:15:00 | 1704.50 | 1705.39 | 1620.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 14:00:00 | 1704.50 | 1705.39 | 1620.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1633.00 | 1699.85 | 1632.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 1633.00 | 1699.85 | 1632.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 1644.00 | 1699.30 | 1632.06 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 13:15:00 | 1484.60 | 1589.76 | 1590.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 13:15:00 | 1483.60 | 1583.07 | 1586.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 1517.50 | 1513.87 | 1543.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 10:00:00 | 1517.50 | 1513.87 | 1543.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 1544.50 | 1513.75 | 1542.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:00:00 | 1544.50 | 1513.75 | 1542.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 1555.10 | 1514.17 | 1542.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:30:00 | 1570.30 | 1514.17 | 1542.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 1564.30 | 1514.66 | 1542.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:15:00 | 1543.60 | 1514.66 | 1542.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1535.40 | 1515.69 | 1541.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:30:00 | 1532.90 | 1515.69 | 1541.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 1537.60 | 1515.91 | 1541.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:30:00 | 1543.00 | 1515.91 | 1541.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1567.90 | 1516.74 | 1540.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:00:00 | 1567.90 | 1516.74 | 1540.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1549.90 | 1517.07 | 1540.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:45:00 | 1543.10 | 1517.36 | 1540.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 12:45:00 | 1545.30 | 1517.60 | 1540.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 10:30:00 | 1544.10 | 1519.08 | 1540.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 11:30:00 | 1544.90 | 1519.37 | 1540.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1542.00 | 1520.65 | 1540.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 11:15:00 | 1539.00 | 1520.89 | 1541.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 09:15:00 | 1465.94 | 1515.86 | 1535.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 09:15:00 | 1468.03 | 1515.86 | 1535.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 09:15:00 | 1466.89 | 1515.86 | 1535.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 09:15:00 | 1467.65 | 1515.86 | 1535.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 09:15:00 | 1462.05 | 1515.86 | 1535.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-09-26 09:15:00 | 1388.79 | 1510.32 | 1531.66 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 8 — BUY (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 15:15:00 | 1449.00 | 1392.73 | 1392.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 10:15:00 | 1461.50 | 1394.01 | 1393.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 12:15:00 | 1410.10 | 1410.12 | 1401.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-09 13:00:00 | 1410.10 | 1410.12 | 1401.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 1405.50 | 1410.05 | 1401.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:30:00 | 1413.30 | 1410.05 | 1401.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 1401.90 | 1409.97 | 1401.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:15:00 | 1365.30 | 1409.97 | 1401.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 1357.60 | 1409.45 | 1401.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:45:00 | 1362.10 | 1409.45 | 1401.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 1374.20 | 1409.10 | 1401.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 12:00:00 | 1386.80 | 1408.87 | 1401.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-13 12:15:00 | 1382.40 | 1407.69 | 1401.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-13 13:00:00 | 1382.40 | 1407.44 | 1401.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-13 14:30:00 | 1384.70 | 1406.95 | 1400.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 1408.50 | 1406.78 | 1400.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 09:45:00 | 1414.50 | 1406.64 | 1401.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 10:00:00 | 1412.80 | 1408.35 | 1402.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 10:30:00 | 1411.00 | 1409.17 | 1402.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 11:15:00 | 1381.20 | 1408.89 | 1402.64 | SL hit (close<static) qty=1.00 sl=1400.50 alert=retest2 |

### Cycle 9 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 1354.20 | 1396.97 | 1397.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 12:15:00 | 1348.50 | 1396.13 | 1396.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 11:15:00 | 1379.90 | 1379.16 | 1387.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-04 12:00:00 | 1379.90 | 1379.16 | 1387.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 1416.70 | 1379.53 | 1387.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:00:00 | 1416.70 | 1379.53 | 1387.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 1427.50 | 1380.01 | 1387.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 15:00:00 | 1413.70 | 1380.34 | 1387.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 09:15:00 | 1445.00 | 1382.33 | 1388.06 | SL hit (close>static) qty=1.00 sl=1440.00 alert=retest2 |

### Cycle 10 — BUY (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 12:15:00 | 1458.90 | 1329.93 | 1329.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 1588.00 | 1363.35 | 1348.31 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-14 11:45:00 | 556.00 | 2024-05-17 15:15:00 | 539.50 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2024-05-16 10:15:00 | 550.00 | 2024-05-17 15:15:00 | 539.50 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2024-05-16 13:45:00 | 549.65 | 2024-05-17 15:15:00 | 539.50 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-05-16 14:30:00 | 550.00 | 2024-05-17 15:15:00 | 539.50 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2024-05-16 15:15:00 | 553.20 | 2024-05-21 13:15:00 | 537.00 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2024-05-18 12:15:00 | 560.00 | 2024-05-21 15:15:00 | 535.00 | STOP_HIT | 1.00 | -4.46% |
| BUY | retest2 | 2024-05-23 11:30:00 | 551.80 | 2024-05-30 12:15:00 | 539.35 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2024-05-23 12:30:00 | 553.00 | 2024-05-30 12:15:00 | 539.35 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2024-05-23 13:00:00 | 554.00 | 2024-05-30 12:15:00 | 539.35 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2024-05-29 11:00:00 | 569.00 | 2024-05-30 12:15:00 | 539.35 | STOP_HIT | 1.00 | -5.21% |
| BUY | retest2 | 2024-05-29 14:15:00 | 569.00 | 2024-05-30 12:15:00 | 539.35 | STOP_HIT | 1.00 | -5.21% |
| BUY | retest2 | 2024-06-07 11:45:00 | 571.90 | 2024-06-26 09:15:00 | 629.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-10 10:00:00 | 577.10 | 2024-06-26 09:15:00 | 634.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-04 11:00:00 | 975.00 | 2024-10-07 10:15:00 | 917.25 | STOP_HIT | 1.00 | -5.92% |
| BUY | retest2 | 2024-10-04 13:15:00 | 978.00 | 2024-10-07 10:15:00 | 917.25 | STOP_HIT | 1.00 | -6.21% |
| BUY | retest2 | 2024-10-07 11:15:00 | 980.00 | 2024-10-07 12:15:00 | 930.00 | STOP_HIT | 1.00 | -5.10% |
| BUY | retest2 | 2024-10-09 09:15:00 | 980.00 | 2024-10-15 09:15:00 | 1078.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-14 14:00:00 | 1354.15 | 2025-01-22 09:15:00 | 1306.00 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2025-01-15 10:30:00 | 1357.95 | 2025-01-22 09:15:00 | 1306.00 | STOP_HIT | 1.00 | -3.83% |
| BUY | retest2 | 2025-01-15 11:15:00 | 1351.85 | 2025-01-22 09:15:00 | 1306.00 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2025-01-15 12:30:00 | 1352.50 | 2025-01-22 09:15:00 | 1306.00 | STOP_HIT | 1.00 | -3.44% |
| BUY | retest2 | 2025-01-16 09:15:00 | 1364.55 | 2025-01-22 09:15:00 | 1306.00 | STOP_HIT | 1.00 | -4.29% |
| BUY | retest2 | 2025-01-21 11:30:00 | 1339.70 | 2025-01-22 09:15:00 | 1306.00 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-01-21 13:15:00 | 1339.95 | 2025-01-22 09:15:00 | 1306.00 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2025-01-21 15:15:00 | 1343.80 | 2025-01-22 09:15:00 | 1306.00 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2025-01-30 09:30:00 | 1338.95 | 2025-01-30 12:15:00 | 1299.00 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2025-01-31 09:15:00 | 1416.90 | 2025-02-03 09:15:00 | 1558.59 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-15 12:00:00 | 1382.10 | 2025-04-15 14:15:00 | 1406.10 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-04-15 12:30:00 | 1377.50 | 2025-04-15 14:15:00 | 1406.10 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-04-16 09:30:00 | 1383.00 | 2025-04-17 11:15:00 | 1410.60 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-04-16 13:15:00 | 1366.60 | 2025-04-17 11:15:00 | 1410.60 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2025-05-22 12:30:00 | 1306.10 | 2025-05-23 15:15:00 | 1343.00 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-05-23 11:15:00 | 1315.30 | 2025-05-23 15:15:00 | 1343.00 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-09-16 11:45:00 | 1543.10 | 2025-09-24 09:15:00 | 1465.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-16 12:45:00 | 1545.30 | 2025-09-24 09:15:00 | 1468.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 10:30:00 | 1544.10 | 2025-09-24 09:15:00 | 1466.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 11:30:00 | 1544.90 | 2025-09-24 09:15:00 | 1467.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 11:15:00 | 1539.00 | 2025-09-24 09:15:00 | 1462.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-16 11:45:00 | 1543.10 | 2025-09-26 09:15:00 | 1388.79 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-16 12:45:00 | 1545.30 | 2025-09-26 09:15:00 | 1390.77 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-17 10:30:00 | 1544.10 | 2025-09-26 09:15:00 | 1389.69 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-17 11:30:00 | 1544.90 | 2025-09-26 09:15:00 | 1390.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-18 11:15:00 | 1539.00 | 2025-09-26 09:15:00 | 1385.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-30 09:15:00 | 1526.60 | 2025-10-06 09:15:00 | 1450.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-30 09:15:00 | 1526.60 | 2025-10-14 11:15:00 | 1373.94 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-02 10:00:00 | 1532.40 | 2025-12-03 09:15:00 | 1455.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 10:00:00 | 1532.40 | 2025-12-03 09:15:00 | 1438.00 | STOP_HIT | 0.50 | 6.16% |
| BUY | retest2 | 2026-01-12 12:00:00 | 1386.80 | 2026-01-20 11:15:00 | 1381.20 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2026-01-13 12:15:00 | 1382.40 | 2026-01-20 11:15:00 | 1381.20 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2026-01-13 13:00:00 | 1382.40 | 2026-01-20 11:15:00 | 1381.20 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2026-01-13 14:30:00 | 1384.70 | 2026-01-21 10:15:00 | 1353.40 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2026-01-16 09:45:00 | 1414.50 | 2026-01-21 10:15:00 | 1353.40 | STOP_HIT | 1.00 | -4.32% |
| BUY | retest2 | 2026-01-19 10:00:00 | 1412.80 | 2026-01-21 10:15:00 | 1353.40 | STOP_HIT | 1.00 | -4.20% |
| BUY | retest2 | 2026-01-20 10:30:00 | 1411.00 | 2026-01-21 10:15:00 | 1353.40 | STOP_HIT | 1.00 | -4.08% |
| SELL | retest2 | 2026-02-04 15:00:00 | 1413.70 | 2026-02-09 09:15:00 | 1445.00 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2026-02-09 13:45:00 | 1415.00 | 2026-02-18 11:15:00 | 1417.50 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2026-02-09 15:15:00 | 1402.10 | 2026-02-18 11:15:00 | 1417.50 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-02-11 09:15:00 | 1414.30 | 2026-03-02 09:15:00 | 1344.25 | PARTIAL | 0.50 | 4.95% |
| SELL | retest2 | 2026-02-16 11:15:00 | 1390.80 | 2026-03-02 09:15:00 | 1331.99 | PARTIAL | 0.50 | 4.23% |
| SELL | retest2 | 2026-02-16 14:45:00 | 1391.00 | 2026-03-02 09:15:00 | 1343.58 | PARTIAL | 0.50 | 3.41% |
| SELL | retest2 | 2026-02-19 15:15:00 | 1385.00 | 2026-03-02 09:15:00 | 1315.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 10:00:00 | 1391.60 | 2026-03-02 09:15:00 | 1322.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 11:30:00 | 1376.70 | 2026-03-02 09:15:00 | 1307.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 09:15:00 | 1414.30 | 2026-03-04 12:15:00 | 1273.50 | TARGET_HIT | 0.50 | 9.96% |
| SELL | retest2 | 2026-02-16 11:15:00 | 1390.80 | 2026-03-04 12:15:00 | 1272.87 | TARGET_HIT | 0.50 | 8.48% |
| SELL | retest2 | 2026-02-16 14:45:00 | 1391.00 | 2026-03-09 09:15:00 | 1261.89 | TARGET_HIT | 0.50 | 9.28% |
| SELL | retest2 | 2026-02-19 15:15:00 | 1385.00 | 2026-03-09 11:15:00 | 1252.44 | TARGET_HIT | 0.50 | 9.57% |
| SELL | retest2 | 2026-02-26 10:00:00 | 1391.60 | 2026-03-13 09:15:00 | 1246.50 | TARGET_HIT | 0.50 | 10.43% |
| SELL | retest2 | 2026-02-26 11:30:00 | 1376.70 | 2026-03-13 09:15:00 | 1239.03 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-09 14:30:00 | 1376.50 | 2026-04-13 09:15:00 | 1307.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-09 14:30:00 | 1376.50 | 2026-04-13 09:15:00 | 1323.30 | STOP_HIT | 0.50 | 3.86% |
| SELL | retest2 | 2026-04-20 09:30:00 | 1373.50 | 2026-04-20 10:15:00 | 1393.50 | STOP_HIT | 1.00 | -1.46% |
