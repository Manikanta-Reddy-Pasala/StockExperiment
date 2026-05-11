# R R Kabel Ltd. (RRKABEL)

## Backtest Summary

- **Window:** 2023-09-20 09:15:00 → 2026-05-08 15:15:00 (4540 bars)
- **Last close:** 1945.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT2_SKIP | 3 |
| ALERT3 | 58 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 46 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 45 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 50 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 45
- **Target hits / Stop hits / Partials:** 2 / 45 / 3
- **Avg / median % per leg:** -2.30% / -2.47%
- **Sum % (uncompounded):** -115.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 1 | 5.0% | 1 | 19 | 0 | -2.58% | -51.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 20 | 1 | 5.0% | 1 | 19 | 0 | -2.58% | -51.6% |
| SELL (all) | 30 | 4 | 13.3% | 1 | 26 | 3 | -2.11% | -63.4% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| SELL @ 3rd Alert (retest2) | 28 | 2 | 7.1% | 0 | 26 | 2 | -2.80% | -78.4% |
| retest1 (combined) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| retest2 (combined) | 48 | 3 | 6.2% | 1 | 45 | 2 | -2.71% | -130.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-01-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-31 15:15:00 | 1444.00 | 1512.31 | 1512.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 09:15:00 | 1425.25 | 1511.45 | 1512.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-16 09:15:00 | 1501.85 | 1477.85 | 1491.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 09:15:00 | 1501.85 | 1477.85 | 1491.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 09:15:00 | 1501.85 | 1477.85 | 1491.95 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-04 09:15:00 | 1571.50 | 1470.71 | 1470.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 09:15:00 | 1587.60 | 1482.99 | 1476.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 11:15:00 | 1597.50 | 1599.95 | 1553.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-07 12:00:00 | 1597.50 | 1599.95 | 1553.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 1564.00 | 1701.64 | 1641.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 1564.00 | 1701.64 | 1641.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 1491.50 | 1699.55 | 1640.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 1491.50 | 1699.55 | 1640.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1685.30 | 1690.29 | 1639.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 09:15:00 | 1697.00 | 1689.75 | 1640.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 12:45:00 | 1696.55 | 1689.74 | 1641.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 14:00:00 | 1696.35 | 1689.81 | 1641.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-07 13:45:00 | 1703.20 | 1741.84 | 1726.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 15:15:00 | 1636.00 | 1731.30 | 1721.95 | SL hit (close<static) qty=1.00 sl=1638.05 alert=retest2 |

### Cycle 3 — SELL (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 10:15:00 | 1566.30 | 1712.86 | 1713.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 10:15:00 | 1554.35 | 1641.42 | 1668.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-11 14:15:00 | 1715.15 | 1630.50 | 1660.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 14:15:00 | 1715.15 | 1630.50 | 1660.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 1715.15 | 1630.50 | 1660.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 1715.15 | 1630.50 | 1660.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 1706.00 | 1631.25 | 1660.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 14:30:00 | 1650.55 | 1633.82 | 1661.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 15:15:00 | 1651.90 | 1633.82 | 1661.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 10:00:00 | 1653.95 | 1634.19 | 1660.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 14:15:00 | 1651.00 | 1634.89 | 1660.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 15:15:00 | 1652.00 | 1635.21 | 1660.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 09:15:00 | 1661.35 | 1635.21 | 1660.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 1649.50 | 1635.35 | 1660.58 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-20 14:15:00 | 1749.90 | 1644.92 | 1661.85 | SL hit (close>static) qty=1.00 sl=1729.95 alert=retest2 |

### Cycle 4 — BUY (started 2024-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-03 15:15:00 | 1723.50 | 1674.13 | 1674.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-04 10:15:00 | 1740.20 | 1675.26 | 1674.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-18 09:15:00 | 1699.75 | 1711.27 | 1695.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 09:15:00 | 1699.75 | 1711.27 | 1695.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 1699.75 | 1711.27 | 1695.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 14:15:00 | 1726.45 | 1711.46 | 1696.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 14:45:00 | 1724.10 | 1711.53 | 1696.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 09:15:00 | 1735.05 | 1711.60 | 1696.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 10:00:00 | 1726.10 | 1711.74 | 1696.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 14:15:00 | 1685.05 | 1711.19 | 1696.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 15:00:00 | 1685.05 | 1711.19 | 1696.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 15:15:00 | 1681.10 | 1710.89 | 1696.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:15:00 | 1670.45 | 1710.89 | 1696.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-22 09:15:00 | 1662.25 | 1710.41 | 1696.29 | SL hit (close<static) qty=1.00 sl=1666.05 alert=retest2 |

### Cycle 5 — SELL (started 2024-10-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 15:15:00 | 1510.00 | 1683.43 | 1683.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-28 09:15:00 | 1453.75 | 1681.14 | 1682.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 15:15:00 | 1328.00 | 1316.62 | 1387.82 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-03 09:15:00 | 1281.65 | 1316.62 | 1387.82 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-05 15:15:00 | 1217.57 | 1302.22 | 1373.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-02-11 10:15:00 | 1153.49 | 1279.71 | 1353.61 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 6 — BUY (started 2025-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 12:15:00 | 1318.00 | 1065.74 | 1064.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 1321.70 | 1091.14 | 1078.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 10:15:00 | 1310.00 | 1310.04 | 1238.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 11:00:00 | 1310.00 | 1310.04 | 1238.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 1334.60 | 1389.49 | 1340.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:45:00 | 1338.00 | 1389.49 | 1340.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 1337.00 | 1388.97 | 1340.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:15:00 | 1310.80 | 1388.97 | 1340.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 1300.50 | 1387.29 | 1340.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:45:00 | 1298.90 | 1387.29 | 1340.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2025-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 12:15:00 | 1215.60 | 1308.86 | 1309.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 1204.70 | 1284.45 | 1295.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 15:15:00 | 1243.00 | 1242.64 | 1266.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 09:15:00 | 1256.30 | 1242.64 | 1266.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 1257.60 | 1243.43 | 1265.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:30:00 | 1259.30 | 1243.43 | 1265.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1259.40 | 1244.09 | 1265.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:15:00 | 1252.20 | 1244.28 | 1265.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 14:15:00 | 1252.30 | 1244.46 | 1265.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 10:00:00 | 1251.00 | 1244.57 | 1264.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:15:00 | 1248.60 | 1244.44 | 1263.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 1256.00 | 1244.96 | 1263.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:45:00 | 1262.80 | 1244.96 | 1263.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 1272.70 | 1245.36 | 1263.17 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 1272.70 | 1245.36 | 1263.17 | SL hit (close>static) qty=1.00 sl=1270.40 alert=retest2 |

### Cycle 8 — BUY (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 13:15:00 | 1392.00 | 1266.07 | 1266.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 14:15:00 | 1401.70 | 1267.42 | 1266.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 10:15:00 | 1339.00 | 1341.13 | 1313.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 10:45:00 | 1337.80 | 1341.13 | 1313.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 1321.60 | 1344.88 | 1318.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:45:00 | 1319.60 | 1344.88 | 1318.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 1328.90 | 1344.73 | 1318.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 12:15:00 | 1331.20 | 1344.73 | 1318.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-12-15 13:15:00 | 1464.32 | 1380.09 | 1350.86 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2026-03-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 14:15:00 | 1318.60 | 1432.18 | 1432.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 15:15:00 | 1306.00 | 1413.82 | 1422.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 1397.40 | 1385.43 | 1405.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 11:00:00 | 1397.40 | 1385.43 | 1405.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 1402.00 | 1385.59 | 1405.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 10:30:00 | 1389.20 | 1386.23 | 1405.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 1365.00 | 1387.05 | 1404.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 12:30:00 | 1387.80 | 1386.83 | 1404.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 11:00:00 | 1382.50 | 1386.49 | 1403.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 1392.00 | 1386.67 | 1403.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:30:00 | 1398.40 | 1386.67 | 1403.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 1391.60 | 1386.76 | 1403.12 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-16 14:15:00 | 1454.30 | 1387.95 | 1403.31 | SL hit (close>static) qty=1.00 sl=1408.10 alert=retest2 |

### Cycle 10 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 1505.60 | 1416.39 | 1416.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 14:15:00 | 1510.10 | 1417.33 | 1416.57 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-07 09:15:00 | 1697.00 | 2024-08-09 15:15:00 | 1636.00 | STOP_HIT | 1.00 | -3.59% |
| BUY | retest2 | 2024-06-07 12:45:00 | 1696.55 | 2024-08-09 15:15:00 | 1636.00 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest2 | 2024-06-07 14:00:00 | 1696.35 | 2024-08-09 15:15:00 | 1636.00 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2024-08-07 13:45:00 | 1703.20 | 2024-08-09 15:15:00 | 1636.00 | STOP_HIT | 1.00 | -3.95% |
| SELL | retest2 | 2024-09-12 14:30:00 | 1650.55 | 2024-09-20 14:15:00 | 1749.90 | STOP_HIT | 1.00 | -6.02% |
| SELL | retest2 | 2024-09-12 15:15:00 | 1651.90 | 2024-09-20 14:15:00 | 1749.90 | STOP_HIT | 1.00 | -5.93% |
| SELL | retest2 | 2024-09-13 10:00:00 | 1653.95 | 2024-09-20 14:15:00 | 1749.90 | STOP_HIT | 1.00 | -5.80% |
| SELL | retest2 | 2024-09-13 14:15:00 | 1651.00 | 2024-09-20 14:15:00 | 1749.90 | STOP_HIT | 1.00 | -5.99% |
| BUY | retest2 | 2024-10-18 14:15:00 | 1726.45 | 2024-10-22 09:15:00 | 1662.25 | STOP_HIT | 1.00 | -3.72% |
| BUY | retest2 | 2024-10-18 14:45:00 | 1724.10 | 2024-10-22 09:15:00 | 1662.25 | STOP_HIT | 1.00 | -3.59% |
| BUY | retest2 | 2024-10-21 09:15:00 | 1735.05 | 2024-10-22 09:15:00 | 1662.25 | STOP_HIT | 1.00 | -4.20% |
| BUY | retest2 | 2024-10-21 10:00:00 | 1726.10 | 2024-10-22 09:15:00 | 1662.25 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest1 | 2025-02-03 09:15:00 | 1281.65 | 2025-02-05 15:15:00 | 1217.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-03 09:15:00 | 1281.65 | 2025-02-11 10:15:00 | 1153.49 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-16 11:15:00 | 1252.20 | 2025-09-22 09:15:00 | 1272.70 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-09-16 14:15:00 | 1252.30 | 2025-09-22 09:15:00 | 1272.70 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-09-17 10:00:00 | 1251.00 | 2025-09-22 09:15:00 | 1272.70 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-09-19 10:15:00 | 1248.60 | 2025-09-22 09:15:00 | 1272.70 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-09-25 09:15:00 | 1246.00 | 2025-09-29 09:15:00 | 1183.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1252.20 | 2025-09-29 09:15:00 | 1189.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 09:15:00 | 1246.00 | 2025-09-30 10:15:00 | 1256.60 | STOP_HIT | 0.50 | -0.85% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1252.20 | 2025-09-30 10:15:00 | 1256.60 | STOP_HIT | 0.50 | -0.35% |
| SELL | retest2 | 2025-09-30 11:30:00 | 1255.00 | 2025-10-01 14:15:00 | 1268.50 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-09-30 13:00:00 | 1254.90 | 2025-10-06 09:15:00 | 1266.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-10-01 09:15:00 | 1242.90 | 2025-10-10 09:15:00 | 1268.00 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-10-03 13:30:00 | 1251.80 | 2025-10-10 09:15:00 | 1268.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-10-08 15:00:00 | 1255.00 | 2025-10-10 12:15:00 | 1283.60 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-10-09 15:00:00 | 1255.00 | 2025-10-10 12:15:00 | 1283.60 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-10-14 10:30:00 | 1250.70 | 2025-10-16 14:15:00 | 1278.00 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-10-15 09:30:00 | 1253.60 | 2025-10-16 14:15:00 | 1278.00 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-10-15 13:15:00 | 1254.40 | 2025-10-16 14:15:00 | 1278.00 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-10-20 10:00:00 | 1254.30 | 2025-10-27 09:15:00 | 1335.90 | STOP_HIT | 1.00 | -6.51% |
| SELL | retest2 | 2025-10-23 12:15:00 | 1256.30 | 2025-10-27 09:15:00 | 1335.90 | STOP_HIT | 1.00 | -6.34% |
| SELL | retest2 | 2025-10-23 15:00:00 | 1252.60 | 2025-10-27 09:15:00 | 1335.90 | STOP_HIT | 1.00 | -6.65% |
| BUY | retest2 | 2025-11-24 12:15:00 | 1331.20 | 2025-12-15 13:15:00 | 1464.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-27 11:00:00 | 1334.00 | 2026-01-27 13:15:00 | 1306.50 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2026-01-27 14:45:00 | 1331.60 | 2026-01-29 13:15:00 | 1315.70 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-01-29 10:45:00 | 1330.00 | 2026-01-29 13:15:00 | 1315.70 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2026-02-11 10:30:00 | 1445.30 | 2026-02-13 09:15:00 | 1401.20 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2026-02-12 10:00:00 | 1442.20 | 2026-02-13 09:15:00 | 1401.20 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2026-02-18 12:30:00 | 1442.80 | 2026-02-19 15:15:00 | 1408.00 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2026-02-19 10:45:00 | 1443.70 | 2026-02-19 15:15:00 | 1408.00 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2026-02-20 09:45:00 | 1435.50 | 2026-03-13 09:15:00 | 1379.40 | STOP_HIT | 1.00 | -3.91% |
| BUY | retest2 | 2026-02-20 14:00:00 | 1444.00 | 2026-03-13 09:15:00 | 1379.40 | STOP_HIT | 1.00 | -4.47% |
| BUY | retest2 | 2026-03-11 10:00:00 | 1431.90 | 2026-03-13 09:15:00 | 1379.40 | STOP_HIT | 1.00 | -3.67% |
| BUY | retest2 | 2026-03-11 11:15:00 | 1445.10 | 2026-03-13 09:15:00 | 1379.40 | STOP_HIT | 1.00 | -4.55% |
| SELL | retest2 | 2026-04-09 10:30:00 | 1389.20 | 2026-04-16 14:15:00 | 1454.30 | STOP_HIT | 1.00 | -4.69% |
| SELL | retest2 | 2026-04-13 09:15:00 | 1365.00 | 2026-04-16 14:15:00 | 1454.30 | STOP_HIT | 1.00 | -6.54% |
| SELL | retest2 | 2026-04-13 12:30:00 | 1387.80 | 2026-04-16 14:15:00 | 1454.30 | STOP_HIT | 1.00 | -4.79% |
| SELL | retest2 | 2026-04-15 11:00:00 | 1382.50 | 2026-04-16 14:15:00 | 1454.30 | STOP_HIT | 1.00 | -5.19% |
