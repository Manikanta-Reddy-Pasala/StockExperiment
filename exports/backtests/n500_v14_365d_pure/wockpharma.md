# Wockhardt Ltd. (WOCKPHARMA)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1611.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 28 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 27 |
| PARTIAL | 14 |
| TARGET_HIT | 12 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 41 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 28 / 13
- **Target hits / Stop hits / Partials:** 12 / 15 / 14
- **Avg / median % per leg:** 4.15% / 5.00%
- **Sum % (uncompounded):** 170.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.21% | -15.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.21% | -15.4% |
| SELL (all) | 34 | 28 | 82.4% | 12 | 8 | 14 | 5.45% | 185.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 34 | 28 | 82.4% | 12 | 8 | 14 | 5.45% | 185.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 41 | 28 | 68.3% | 12 | 15 | 14 | 4.15% | 170.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 10:15:00 | 1484.30 | 1349.49 | 1348.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 09:15:00 | 1512.00 | 1357.83 | 1353.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 13:15:00 | 1704.50 | 1705.39 | 1620.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 14:00:00 | 1704.50 | 1705.39 | 1620.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1633.00 | 1699.85 | 1632.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 1633.00 | 1699.85 | 1632.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 1644.00 | 1699.30 | 1632.06 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-08-20 13:15:00)

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
| Target hit | 2025-09-26 09:15:00 | 1390.77 | 1510.32 | 1531.66 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-09-26 09:15:00 | 1389.69 | 1510.32 | 1531.66 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-09-26 09:15:00 | 1390.41 | 1510.32 | 1531.66 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-09-26 09:15:00 | 1385.10 | 1510.32 | 1531.66 | Target hit (10%) qty=0.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 1526.60 | 1502.09 | 1526.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-06 09:15:00 | 1450.27 | 1499.63 | 1522.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-10-14 11:15:00 | 1373.94 | 1467.19 | 1499.96 | Target hit (10%) qty=0.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 10:00:00 | 1532.40 | 1339.95 | 1384.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 09:15:00 | 1455.78 | 1350.28 | 1387.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 1438.00 | 1350.28 | 1387.79 | SL hit (close>static) qty=0.50 sl=1350.28 alert=retest2 |

### Cycle 3 — BUY (started 2026-01-05 15:15:00)

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
| Stop hit — per-position SL triggered | 2026-01-20 11:15:00 | 1381.20 | 1408.89 | 1402.64 | SL hit (close<static) qty=1.00 sl=1400.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 11:15:00 | 1381.20 | 1408.89 | 1402.64 | SL hit (close<static) qty=1.00 sl=1400.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 10:15:00 | 1353.40 | 1406.65 | 1401.69 | SL hit (close<static) qty=1.00 sl=1356.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 10:15:00 | 1353.40 | 1406.65 | 1401.69 | SL hit (close<static) qty=1.00 sl=1356.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 10:15:00 | 1353.40 | 1406.65 | 1401.69 | SL hit (close<static) qty=1.00 sl=1356.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 10:15:00 | 1353.40 | 1406.65 | 1401.69 | SL hit (close<static) qty=1.00 sl=1356.20 alert=retest2 |

### Cycle 4 — SELL (started 2026-01-27 10:15:00)

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
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 13:45:00 | 1415.00 | 1384.35 | 1388.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 15:15:00 | 1402.10 | 1384.68 | 1389.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 1414.30 | 1386.28 | 1389.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 1393.00 | 1387.17 | 1389.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 11:00:00 | 1393.00 | 1387.17 | 1389.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 1407.40 | 1387.37 | 1390.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:00:00 | 1407.40 | 1387.37 | 1390.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 1413.50 | 1387.63 | 1390.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:30:00 | 1416.10 | 1387.63 | 1390.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 15:15:00 | 1391.40 | 1387.93 | 1390.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:15:00 | 1405.00 | 1387.93 | 1390.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 1393.20 | 1387.98 | 1390.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:30:00 | 1397.10 | 1387.98 | 1390.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 1395.10 | 1388.05 | 1390.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 11:15:00 | 1390.80 | 1388.05 | 1390.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 14:45:00 | 1391.00 | 1388.32 | 1390.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 11:15:00 | 1417.50 | 1389.57 | 1390.94 | SL hit (close>static) qty=1.00 sl=1411.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 11:15:00 | 1417.50 | 1389.57 | 1390.94 | SL hit (close>static) qty=1.00 sl=1411.90 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 15:15:00 | 1385.00 | 1391.87 | 1392.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 10:00:00 | 1391.60 | 1386.14 | 1388.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 1383.50 | 1386.12 | 1388.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:30:00 | 1376.70 | 1386.00 | 1388.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1344.25 | 1383.47 | 1387.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1331.99 | 1383.47 | 1387.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1343.58 | 1383.47 | 1387.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1315.75 | 1383.47 | 1387.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1322.02 | 1383.47 | 1387.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1307.87 | 1383.47 | 1387.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 12:15:00 | 1273.50 | 1375.93 | 1383.36 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-04 12:15:00 | 1272.87 | 1375.93 | 1383.36 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 1261.89 | 1363.76 | 1376.37 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-09 11:15:00 | 1252.44 | 1361.64 | 1375.18 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-13 09:15:00 | 1246.50 | 1348.26 | 1366.40 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-13 09:15:00 | 1239.03 | 1348.26 | 1366.40 | Target hit (10%) qty=0.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 14:30:00 | 1376.50 | 1275.26 | 1307.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-13 09:15:00 | 1307.67 | 1281.96 | 1309.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 1323.30 | 1281.96 | 1309.57 | SL hit (close>static) qty=0.50 sl=1281.96 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 09:30:00 | 1373.50 | 1302.09 | 1316.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 10:15:00 | 1393.50 | 1303.00 | 1317.02 | SL hit (close>static) qty=1.00 sl=1392.90 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 12:15:00 | 1458.90 | 1329.93 | 1329.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 1588.00 | 1363.35 | 1348.31 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
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
