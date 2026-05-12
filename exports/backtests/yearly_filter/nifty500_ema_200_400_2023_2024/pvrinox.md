# PVR INOX Ltd. (PVRINOX)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1075.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT2_SKIP | 3 |
| ALERT3 | 46 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 30 |
| PARTIAL | 7 |
| TARGET_HIT | 7 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 37 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 18
- **Target hits / Stop hits / Partials:** 7 / 23 / 7
- **Avg / median % per leg:** 2.64% / 3.75%
- **Sum % (uncompounded):** 97.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 5 | 31.2% | 5 | 11 | 0 | 2.18% | 34.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 5 | 31.2% | 5 | 11 | 0 | 2.18% | 34.8% |
| SELL (all) | 21 | 14 | 66.7% | 2 | 12 | 7 | 2.99% | 62.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 21 | 14 | 66.7% | 2 | 12 | 7 | 2.99% | 62.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 37 | 19 | 51.4% | 7 | 23 | 7 | 2.64% | 97.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 15:15:00 | 1546.10 | 1441.29 | 1441.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 10:15:00 | 1574.95 | 1449.54 | 1445.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 11:15:00 | 1727.30 | 1729.87 | 1651.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-18 12:00:00 | 1727.30 | 1729.87 | 1651.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 1681.00 | 1717.33 | 1671.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-09 09:30:00 | 1676.00 | 1717.33 | 1671.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 11:15:00 | 1673.95 | 1716.51 | 1671.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-09 12:00:00 | 1673.95 | 1716.51 | 1671.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 12:15:00 | 1665.75 | 1716.01 | 1671.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-09 13:00:00 | 1665.75 | 1716.01 | 1671.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 13:15:00 | 1668.90 | 1715.54 | 1671.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-09 15:00:00 | 1674.30 | 1715.13 | 1671.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-10 09:15:00 | 1679.15 | 1714.63 | 1671.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-23 09:15:00 | 1649.00 | 1724.99 | 1689.45 | SL hit (close<static) qty=1.00 sl=1665.00 alert=retest2 |

### Cycle 2 — SELL (started 2023-11-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-15 09:15:00 | 1644.00 | 1664.89 | 1664.89 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 15:15:00 | 1685.00 | 1664.74 | 1664.73 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-11-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 15:15:00 | 1655.00 | 1664.70 | 1664.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-23 11:15:00 | 1647.55 | 1664.26 | 1664.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 10:15:00 | 1664.00 | 1663.57 | 1664.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 10:15:00 | 1664.00 | 1663.57 | 1664.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 10:15:00 | 1664.00 | 1663.57 | 1664.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 11:00:00 | 1664.00 | 1663.57 | 1664.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 11:15:00 | 1662.80 | 1663.56 | 1664.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 12:00:00 | 1662.80 | 1663.56 | 1664.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 12:15:00 | 1653.75 | 1663.46 | 1664.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-24 13:15:00 | 1649.55 | 1663.46 | 1664.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-24 15:15:00 | 1650.00 | 1663.31 | 1663.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-28 12:15:00 | 1666.45 | 1663.20 | 1663.92 | SL hit (close>static) qty=1.00 sl=1664.45 alert=retest2 |

### Cycle 5 — BUY (started 2023-11-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 15:15:00 | 1699.00 | 1664.90 | 1664.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 09:15:00 | 1713.95 | 1665.39 | 1664.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-21 13:15:00 | 1717.70 | 1730.36 | 1704.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-21 14:00:00 | 1717.70 | 1730.36 | 1704.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 12:15:00 | 1706.00 | 1730.62 | 1705.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-22 12:30:00 | 1707.70 | 1730.62 | 1705.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 13:15:00 | 1699.00 | 1730.31 | 1705.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-22 13:45:00 | 1698.10 | 1730.31 | 1705.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 14:15:00 | 1701.80 | 1730.03 | 1705.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-22 14:45:00 | 1701.00 | 1730.03 | 1705.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 10:15:00 | 1664.00 | 1728.45 | 1705.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-26 11:00:00 | 1664.00 | 1728.45 | 1705.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 09:15:00 | 1665.65 | 1705.86 | 1697.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-04 09:45:00 | 1660.95 | 1705.86 | 1697.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 10:15:00 | 1669.30 | 1705.50 | 1697.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-04 10:45:00 | 1668.80 | 1705.50 | 1697.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-09 11:15:00 | 1583.20 | 1690.14 | 1690.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-09 12:15:00 | 1579.20 | 1689.04 | 1689.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 13:15:00 | 1417.00 | 1415.92 | 1478.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-07 14:00:00 | 1417.00 | 1415.92 | 1478.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 12:15:00 | 1408.00 | 1369.06 | 1411.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 12:45:00 | 1406.30 | 1369.06 | 1411.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 13:15:00 | 1417.45 | 1369.54 | 1411.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 14:00:00 | 1417.45 | 1369.54 | 1411.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 14:15:00 | 1407.65 | 1369.92 | 1411.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 10:15:00 | 1404.00 | 1370.76 | 1411.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 12:45:00 | 1406.90 | 1371.89 | 1411.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-15 09:15:00 | 1380.80 | 1373.11 | 1411.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-16 11:45:00 | 1401.80 | 1375.36 | 1410.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 10:15:00 | 1432.00 | 1376.73 | 1410.58 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-04-18 10:15:00 | 1432.00 | 1376.73 | 1410.58 | SL hit (close>static) qty=1.00 sl=1417.90 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 09:15:00 | 1455.90 | 1371.31 | 1371.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 11:15:00 | 1458.20 | 1373.01 | 1371.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 09:15:00 | 1414.00 | 1430.85 | 1408.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-18 10:00:00 | 1414.00 | 1430.85 | 1408.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 1409.95 | 1430.64 | 1408.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:00:00 | 1409.95 | 1430.64 | 1408.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 1408.75 | 1430.43 | 1408.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:30:00 | 1407.20 | 1430.43 | 1408.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 1409.45 | 1430.22 | 1408.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 12:30:00 | 1403.00 | 1430.22 | 1408.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 1414.80 | 1430.06 | 1408.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 09:15:00 | 1419.85 | 1429.73 | 1408.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 13:45:00 | 1424.00 | 1428.89 | 1408.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 14:15:00 | 1402.45 | 1428.63 | 1408.90 | SL hit (close<static) qty=1.00 sl=1407.95 alert=retest2 |

### Cycle 8 — SELL (started 2024-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 12:15:00 | 1470.05 | 1554.52 | 1554.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 14:15:00 | 1463.05 | 1547.36 | 1550.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 13:15:00 | 1512.95 | 1507.97 | 1526.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-27 13:30:00 | 1517.30 | 1507.97 | 1526.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 1533.50 | 1508.35 | 1526.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:00:00 | 1533.50 | 1508.35 | 1526.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 1514.00 | 1508.40 | 1526.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:30:00 | 1526.00 | 1508.40 | 1526.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 1521.15 | 1508.71 | 1526.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:15:00 | 1542.50 | 1508.71 | 1526.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 1532.00 | 1508.94 | 1526.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:30:00 | 1532.90 | 1508.94 | 1526.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 1534.90 | 1509.20 | 1526.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:45:00 | 1534.80 | 1509.20 | 1526.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 1541.40 | 1529.14 | 1534.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 11:00:00 | 1541.40 | 1529.14 | 1534.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 11:15:00 | 1539.00 | 1529.24 | 1534.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 11:30:00 | 1543.25 | 1529.24 | 1534.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 1523.35 | 1511.06 | 1523.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:45:00 | 1531.70 | 1511.06 | 1523.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 1520.75 | 1511.15 | 1523.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 12:00:00 | 1492.55 | 1510.97 | 1523.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 13:15:00 | 1417.92 | 1506.34 | 1520.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-26 09:15:00 | 1343.30 | 1474.47 | 1501.24 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 9 — BUY (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 14:15:00 | 1014.60 | 984.77 | 984.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 12:15:00 | 1024.45 | 986.27 | 985.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 11:15:00 | 987.35 | 989.51 | 987.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 11:15:00 | 987.35 | 989.51 | 987.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 987.35 | 989.51 | 987.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:45:00 | 988.50 | 989.51 | 987.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 973.65 | 989.35 | 987.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 13:00:00 | 973.65 | 989.35 | 987.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 13:15:00 | 976.75 | 989.22 | 987.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 09:15:00 | 983.00 | 989.04 | 987.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 13:00:00 | 982.05 | 988.66 | 986.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 14:15:00 | 984.90 | 988.58 | 986.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-07 09:15:00 | 1081.30 | 997.18 | 991.86 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 1088.00 | 1106.55 | 1106.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 15:15:00 | 1080.50 | 1106.13 | 1106.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 10:15:00 | 1119.40 | 1095.93 | 1100.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 10:15:00 | 1119.40 | 1095.93 | 1100.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 1119.40 | 1095.93 | 1100.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:45:00 | 1120.20 | 1095.93 | 1100.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 1100.10 | 1095.98 | 1100.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 13:15:00 | 1090.70 | 1096.01 | 1100.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 09:15:00 | 1036.16 | 1084.51 | 1093.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-12 09:15:00 | 981.63 | 1046.84 | 1067.24 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 11 — BUY (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 13:15:00 | 1066.90 | 1001.16 | 1000.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 1074.20 | 1003.13 | 1001.84 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-10-09 15:00:00 | 1674.30 | 2023-10-23 09:15:00 | 1649.00 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2023-10-10 09:15:00 | 1679.15 | 2023-10-23 09:15:00 | 1649.00 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2023-11-08 15:00:00 | 1673.10 | 2023-11-08 15:15:00 | 1664.85 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2023-11-09 12:15:00 | 1673.00 | 2023-11-09 12:15:00 | 1662.05 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2023-11-10 09:15:00 | 1672.00 | 2023-11-12 18:15:00 | 1657.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2023-11-10 11:30:00 | 1670.00 | 2023-11-12 18:15:00 | 1657.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2023-11-12 18:15:00 | 1675.35 | 2023-11-12 18:15:00 | 1657.00 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2023-11-24 13:15:00 | 1649.55 | 2023-11-28 12:15:00 | 1666.45 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2023-11-24 15:15:00 | 1650.00 | 2023-11-28 12:15:00 | 1666.45 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-04-12 10:15:00 | 1404.00 | 2024-04-18 10:15:00 | 1432.00 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2024-04-12 12:45:00 | 1406.90 | 2024-04-18 10:15:00 | 1432.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-04-15 09:15:00 | 1380.80 | 2024-04-18 10:15:00 | 1432.00 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2024-04-16 11:45:00 | 1401.80 | 2024-04-18 10:15:00 | 1432.00 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2024-04-26 10:15:00 | 1402.55 | 2024-05-03 09:15:00 | 1332.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-26 13:30:00 | 1403.10 | 2024-05-03 09:15:00 | 1332.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-26 14:00:00 | 1402.80 | 2024-05-03 09:15:00 | 1332.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-26 10:15:00 | 1402.55 | 2024-05-21 09:15:00 | 1349.90 | STOP_HIT | 0.50 | 3.75% |
| SELL | retest2 | 2024-04-26 13:30:00 | 1403.10 | 2024-05-21 09:15:00 | 1349.90 | STOP_HIT | 0.50 | 3.79% |
| SELL | retest2 | 2024-04-26 14:00:00 | 1402.80 | 2024-05-21 09:15:00 | 1349.90 | STOP_HIT | 0.50 | 3.77% |
| SELL | retest2 | 2024-06-18 15:15:00 | 1402.00 | 2024-06-21 09:15:00 | 1423.00 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-07-19 09:15:00 | 1419.85 | 2024-07-19 14:15:00 | 1402.45 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-07-19 13:45:00 | 1424.00 | 2024-07-19 14:15:00 | 1402.45 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-07-22 09:30:00 | 1428.25 | 2024-07-23 09:15:00 | 1392.95 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2024-07-22 10:00:00 | 1431.95 | 2024-07-23 09:15:00 | 1392.95 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2024-07-23 11:30:00 | 1404.30 | 2024-08-19 09:15:00 | 1544.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-23 12:45:00 | 1408.60 | 2024-08-19 09:15:00 | 1549.46 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-17 12:00:00 | 1492.55 | 2024-12-18 13:15:00 | 1417.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 12:00:00 | 1492.55 | 2024-12-26 09:15:00 | 1343.30 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-29 09:15:00 | 983.00 | 2025-08-07 09:15:00 | 1081.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-29 13:00:00 | 982.05 | 2025-08-07 09:15:00 | 1080.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-29 14:15:00 | 984.90 | 2025-08-07 09:15:00 | 1083.39 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-12-15 13:15:00 | 1090.70 | 2025-12-23 09:15:00 | 1036.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-15 13:15:00 | 1090.70 | 2026-01-12 09:15:00 | 981.63 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-12 09:15:00 | 1089.60 | 2026-02-16 09:15:00 | 1035.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 09:15:00 | 1089.60 | 2026-02-16 09:15:00 | 1039.80 | STOP_HIT | 0.50 | 4.57% |
| SELL | retest2 | 2026-02-12 11:15:00 | 1094.05 | 2026-02-16 09:15:00 | 1039.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 11:15:00 | 1094.05 | 2026-02-16 09:15:00 | 1039.80 | STOP_HIT | 0.50 | 4.96% |
