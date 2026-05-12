# Havells India Ltd. (HAVELLS)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1253.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 4 |
| ALERT3 | 47 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 49 |
| PARTIAL | 23 |
| TARGET_HIT | 10 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 72 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 46 / 26
- **Target hits / Stop hits / Partials:** 10 / 39 / 23
- **Avg / median % per leg:** 2.87% / 2.37%
- **Sum % (uncompounded):** 206.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 2 | 25.0% | 2 | 6 | 0 | 1.75% | 14.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 2 | 25.0% | 2 | 6 | 0 | 1.75% | 14.0% |
| SELL (all) | 64 | 44 | 68.8% | 8 | 33 | 23 | 3.01% | 192.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 64 | 44 | 68.8% | 8 | 33 | 23 | 3.01% | 192.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 72 | 46 | 63.9% | 10 | 39 | 23 | 2.87% | 207.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 10:15:00 | 1264.10 | 1351.02 | 1351.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 12:15:00 | 1257.80 | 1343.57 | 1347.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-21 10:15:00 | 1296.45 | 1294.55 | 1315.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-21 11:00:00 | 1296.45 | 1294.55 | 1315.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 1308.40 | 1295.70 | 1314.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 09:30:00 | 1312.05 | 1295.70 | 1314.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 09:15:00 | 1307.20 | 1295.00 | 1311.01 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-12-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 09:15:00 | 1383.60 | 1320.28 | 1320.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-03 11:15:00 | 1416.00 | 1341.22 | 1332.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-19 13:15:00 | 1376.05 | 1382.51 | 1360.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-19 14:00:00 | 1376.05 | 1382.51 | 1360.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 1342.20 | 1381.85 | 1360.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-14 09:30:00 | 1376.35 | 1348.38 | 1347.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-14 10:00:00 | 1378.45 | 1348.38 | 1347.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-28 09:15:00 | 1513.99 | 1383.84 | 1367.74 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 15:15:00 | 1724.10 | 1899.28 | 1899.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 1694.25 | 1897.24 | 1898.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 1724.85 | 1714.21 | 1778.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-25 10:00:00 | 1724.85 | 1714.21 | 1778.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 1749.30 | 1720.97 | 1765.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 11:15:00 | 1745.10 | 1721.24 | 1764.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 14:15:00 | 1744.55 | 1722.09 | 1764.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 12:15:00 | 1745.70 | 1721.95 | 1760.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 14:00:00 | 1743.40 | 1722.39 | 1760.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 1744.80 | 1723.19 | 1760.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 10:15:00 | 1742.50 | 1723.19 | 1760.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 11:15:00 | 1742.50 | 1725.64 | 1760.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 10:15:00 | 1768.00 | 1727.93 | 1760.20 | SL hit (close>static) qty=1.00 sl=1763.90 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 13:15:00 | 1599.30 | 1549.09 | 1548.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 14:15:00 | 1610.70 | 1549.71 | 1549.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 12:15:00 | 1550.50 | 1563.95 | 1557.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 12:15:00 | 1550.50 | 1563.95 | 1557.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 12:15:00 | 1550.50 | 1563.95 | 1557.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:00:00 | 1550.50 | 1563.95 | 1557.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 1542.20 | 1563.73 | 1557.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:45:00 | 1544.20 | 1563.73 | 1557.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 1559.50 | 1570.21 | 1562.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:45:00 | 1556.10 | 1570.21 | 1562.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 1564.90 | 1570.16 | 1562.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:15:00 | 1570.20 | 1570.16 | 1562.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 15:00:00 | 1566.00 | 1570.47 | 1563.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 09:30:00 | 1568.40 | 1570.40 | 1563.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 10:30:00 | 1567.70 | 1570.38 | 1563.22 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1564.00 | 1570.65 | 1563.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 1564.00 | 1570.65 | 1563.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1572.40 | 1570.66 | 1563.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 14:45:00 | 1573.80 | 1570.66 | 1563.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 15:15:00 | 1572.60 | 1570.66 | 1563.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 1556.70 | 1570.54 | 1564.00 | SL hit (close<static) qty=1.00 sl=1562.80 alert=retest2 |

### Cycle 5 — SELL (started 2025-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 15:15:00 | 1500.40 | 1558.72 | 1558.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 09:15:00 | 1497.80 | 1558.12 | 1558.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 09:15:00 | 1548.20 | 1547.44 | 1552.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 09:15:00 | 1548.20 | 1547.44 | 1552.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 1548.20 | 1547.44 | 1552.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 11:30:00 | 1528.50 | 1551.33 | 1554.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:45:00 | 1529.60 | 1550.68 | 1553.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 12:15:00 | 1554.20 | 1550.67 | 1553.67 | SL hit (close>static) qty=1.00 sl=1553.50 alert=retest2 |

### Cycle 6 — BUY (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 10:15:00 | 1585.60 | 1554.72 | 1554.63 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 13:15:00 | 1525.60 | 1554.40 | 1554.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 10:15:00 | 1515.00 | 1553.26 | 1553.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 10:15:00 | 1547.00 | 1541.25 | 1546.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 10:15:00 | 1547.00 | 1541.25 | 1546.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 1547.00 | 1541.25 | 1546.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:30:00 | 1556.40 | 1541.25 | 1546.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 1526.40 | 1541.10 | 1546.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:15:00 | 1545.10 | 1541.10 | 1546.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 1554.10 | 1541.23 | 1546.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:00:00 | 1554.10 | 1541.23 | 1546.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 1571.30 | 1541.53 | 1546.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:45:00 | 1571.40 | 1541.53 | 1546.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 1578.80 | 1541.90 | 1546.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 1578.80 | 1541.90 | 1546.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 1552.70 | 1544.64 | 1548.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 15:00:00 | 1552.70 | 1544.64 | 1548.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 1547.40 | 1544.67 | 1548.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:30:00 | 1538.90 | 1544.60 | 1548.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 1461.95 | 1521.12 | 1533.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 1546.70 | 1509.71 | 1525.60 | SL hit (close>ema200) qty=0.50 sl=1509.71 alert=retest2 |

### Cycle 8 — BUY (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 15:15:00 | 1572.00 | 1536.20 | 1536.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 09:15:00 | 1581.50 | 1536.65 | 1536.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 09:15:00 | 1548.40 | 1567.87 | 1555.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 1548.40 | 1567.87 | 1555.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1548.40 | 1567.87 | 1555.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:45:00 | 1545.90 | 1567.87 | 1555.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 1548.00 | 1567.68 | 1555.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 1548.00 | 1567.68 | 1555.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 1549.50 | 1566.32 | 1555.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:00:00 | 1549.50 | 1566.32 | 1555.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 12:15:00 | 1556.50 | 1566.22 | 1555.47 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 13:15:00 | 1473.00 | 1546.77 | 1546.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 1468.50 | 1530.64 | 1537.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 09:15:00 | 1510.60 | 1508.94 | 1523.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-24 10:00:00 | 1510.60 | 1508.94 | 1523.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1449.20 | 1427.60 | 1447.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:00:00 | 1449.20 | 1427.60 | 1447.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1455.70 | 1427.88 | 1447.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:30:00 | 1455.40 | 1427.88 | 1447.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 1456.60 | 1428.17 | 1447.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 14:00:00 | 1451.90 | 1428.69 | 1447.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 1462.50 | 1429.52 | 1447.89 | SL hit (close>static) qty=1.00 sl=1460.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-02-14 09:30:00 | 1376.35 | 2024-02-28 09:15:00 | 1513.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-14 10:00:00 | 1378.45 | 2024-02-28 09:15:00 | 1516.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-06 11:15:00 | 1745.10 | 2024-12-16 10:15:00 | 1768.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-12-06 14:15:00 | 1744.55 | 2024-12-16 10:15:00 | 1768.00 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-12-11 12:15:00 | 1745.70 | 2024-12-20 14:15:00 | 1657.84 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2024-12-11 14:00:00 | 1743.40 | 2024-12-20 14:15:00 | 1657.32 | PARTIAL | 0.50 | 4.94% |
| SELL | retest2 | 2024-12-12 10:15:00 | 1742.50 | 2024-12-20 14:15:00 | 1658.41 | PARTIAL | 0.50 | 4.83% |
| SELL | retest2 | 2024-12-13 11:15:00 | 1742.50 | 2024-12-20 14:15:00 | 1656.23 | PARTIAL | 0.50 | 4.95% |
| SELL | retest2 | 2024-12-17 12:30:00 | 1742.75 | 2024-12-20 14:15:00 | 1655.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 13:00:00 | 1736.80 | 2024-12-27 14:15:00 | 1649.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-11 12:15:00 | 1745.70 | 2025-01-02 14:15:00 | 1701.45 | STOP_HIT | 0.50 | 2.53% |
| SELL | retest2 | 2024-12-11 14:00:00 | 1743.40 | 2025-01-02 14:15:00 | 1701.45 | STOP_HIT | 0.50 | 2.41% |
| SELL | retest2 | 2024-12-12 10:15:00 | 1742.50 | 2025-01-02 14:15:00 | 1701.45 | STOP_HIT | 0.50 | 2.36% |
| SELL | retest2 | 2024-12-13 11:15:00 | 1742.50 | 2025-01-02 14:15:00 | 1701.45 | STOP_HIT | 0.50 | 2.36% |
| SELL | retest2 | 2024-12-17 12:30:00 | 1742.75 | 2025-01-02 14:15:00 | 1701.45 | STOP_HIT | 0.50 | 2.37% |
| SELL | retest2 | 2024-12-17 13:00:00 | 1736.80 | 2025-01-02 14:15:00 | 1701.45 | STOP_HIT | 0.50 | 2.04% |
| SELL | retest2 | 2025-03-28 13:45:00 | 1519.35 | 2025-04-07 09:15:00 | 1443.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 09:15:00 | 1509.40 | 2025-04-07 09:15:00 | 1433.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 12:30:00 | 1519.60 | 2025-04-07 09:15:00 | 1443.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 13:15:00 | 1517.20 | 2025-04-07 09:15:00 | 1441.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 13:45:00 | 1519.35 | 2025-04-11 09:15:00 | 1500.60 | STOP_HIT | 0.50 | 1.23% |
| SELL | retest2 | 2025-04-01 09:15:00 | 1509.40 | 2025-04-11 09:15:00 | 1500.60 | STOP_HIT | 0.50 | 0.58% |
| SELL | retest2 | 2025-04-03 12:30:00 | 1519.60 | 2025-04-11 09:15:00 | 1500.60 | STOP_HIT | 0.50 | 1.25% |
| SELL | retest2 | 2025-04-03 13:15:00 | 1517.20 | 2025-04-11 09:15:00 | 1500.60 | STOP_HIT | 0.50 | 1.09% |
| BUY | retest2 | 2025-05-21 09:15:00 | 1570.20 | 2025-05-28 09:15:00 | 1556.70 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-05-22 15:00:00 | 1566.00 | 2025-05-28 09:15:00 | 1556.70 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-05-23 09:30:00 | 1568.40 | 2025-05-28 10:15:00 | 1552.90 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-05-23 10:30:00 | 1567.70 | 2025-05-28 10:15:00 | 1552.90 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-05-27 14:45:00 | 1573.80 | 2025-05-28 10:15:00 | 1552.90 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-05-27 15:15:00 | 1572.60 | 2025-05-28 10:15:00 | 1552.90 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-06-13 11:30:00 | 1528.50 | 2025-06-16 12:15:00 | 1554.20 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-06-16 09:45:00 | 1529.60 | 2025-06-16 12:15:00 | 1554.20 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-06-17 15:00:00 | 1530.80 | 2025-06-23 12:15:00 | 1563.00 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-06-19 10:30:00 | 1522.00 | 2025-06-23 12:15:00 | 1563.00 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-06-23 11:15:00 | 1537.90 | 2025-06-23 12:15:00 | 1563.00 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-06-30 10:15:00 | 1540.00 | 2025-06-30 15:15:00 | 1552.10 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-06-30 11:15:00 | 1539.40 | 2025-06-30 15:15:00 | 1552.10 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-07-02 09:30:00 | 1538.30 | 2025-07-02 15:15:00 | 1552.00 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-07-25 09:30:00 | 1538.90 | 2025-08-11 09:15:00 | 1461.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 09:30:00 | 1538.90 | 2025-08-18 09:15:00 | 1546.70 | STOP_HIT | 0.50 | -0.51% |
| SELL | retest2 | 2025-08-18 10:15:00 | 1541.30 | 2025-08-18 10:15:00 | 1564.80 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-08-26 09:15:00 | 1542.60 | 2025-09-01 10:15:00 | 1549.70 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-08-26 10:15:00 | 1545.00 | 2025-09-01 10:15:00 | 1549.70 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-08-26 11:45:00 | 1535.20 | 2025-09-01 10:15:00 | 1549.70 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-08-26 12:45:00 | 1535.10 | 2025-09-01 12:15:00 | 1561.30 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-08-26 14:30:00 | 1531.20 | 2025-09-01 12:15:00 | 1561.30 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2026-01-05 14:00:00 | 1451.90 | 2026-01-06 09:15:00 | 1462.50 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2026-01-12 09:15:00 | 1451.80 | 2026-01-12 14:15:00 | 1454.10 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2026-01-12 10:30:00 | 1446.60 | 2026-01-20 11:15:00 | 1379.21 | PARTIAL | 0.50 | 4.66% |
| SELL | retest2 | 2026-01-12 12:15:00 | 1447.30 | 2026-01-20 11:15:00 | 1374.27 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2026-01-12 13:15:00 | 1443.40 | 2026-01-20 11:15:00 | 1374.93 | PARTIAL | 0.50 | 4.74% |
| SELL | retest2 | 2026-01-13 09:15:00 | 1441.70 | 2026-01-20 11:15:00 | 1369.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 12:30:00 | 1435.70 | 2026-01-20 11:15:00 | 1363.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 15:15:00 | 1436.00 | 2026-01-20 11:15:00 | 1364.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-12 10:30:00 | 1446.60 | 2026-01-21 09:15:00 | 1306.62 | TARGET_HIT | 0.50 | 9.68% |
| SELL | retest2 | 2026-01-12 12:15:00 | 1447.30 | 2026-01-21 09:15:00 | 1301.94 | TARGET_HIT | 0.50 | 10.04% |
| SELL | retest2 | 2026-01-12 13:15:00 | 1443.40 | 2026-01-21 09:15:00 | 1302.57 | TARGET_HIT | 0.50 | 9.76% |
| SELL | retest2 | 2026-01-13 09:15:00 | 1441.70 | 2026-01-21 09:15:00 | 1297.53 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-20 10:15:00 | 1399.00 | 2026-01-21 09:15:00 | 1329.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 12:30:00 | 1435.70 | 2026-01-23 11:15:00 | 1292.40 | TARGET_HIT | 0.50 | 9.98% |
| SELL | retest2 | 2026-01-19 15:15:00 | 1436.00 | 2026-01-23 12:15:00 | 1292.13 | TARGET_HIT | 0.50 | 10.02% |
| SELL | retest2 | 2026-01-20 10:15:00 | 1399.00 | 2026-01-29 10:15:00 | 1259.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-13 15:15:00 | 1400.00 | 2026-03-04 09:15:00 | 1330.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 14:00:00 | 1401.20 | 2026-03-04 09:15:00 | 1331.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-20 11:00:00 | 1401.80 | 2026-03-04 09:15:00 | 1331.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 1372.20 | 2026-03-09 09:15:00 | 1303.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 15:15:00 | 1400.00 | 2026-03-10 10:15:00 | 1379.00 | STOP_HIT | 0.50 | 1.50% |
| SELL | retest2 | 2026-02-19 14:00:00 | 1401.20 | 2026-03-10 10:15:00 | 1379.00 | STOP_HIT | 0.50 | 1.58% |
| SELL | retest2 | 2026-02-20 11:00:00 | 1401.80 | 2026-03-10 10:15:00 | 1379.00 | STOP_HIT | 0.50 | 1.63% |
| SELL | retest2 | 2026-03-02 09:15:00 | 1372.20 | 2026-03-10 10:15:00 | 1379.00 | STOP_HIT | 0.50 | -0.50% |
| SELL | retest2 | 2026-03-11 09:45:00 | 1387.70 | 2026-03-13 09:15:00 | 1318.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 09:45:00 | 1387.70 | 2026-03-23 09:15:00 | 1248.93 | TARGET_HIT | 0.50 | 10.00% |
