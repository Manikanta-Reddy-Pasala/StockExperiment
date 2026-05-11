# APL Apollo Tubes Ltd. (APLAPOLLO)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1950.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 11 |
| ALERT2 | 10 |
| ALERT2_SKIP | 4 |
| ALERT3 | 48 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 49 |
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 54 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 42
- **Target hits / Stop hits / Partials:** 2 / 47 / 5
- **Avg / median % per leg:** -0.69% / -1.52%
- **Sum % (uncompounded):** -37.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 2 | 6.9% | 2 | 27 | 0 | -1.34% | -38.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 29 | 2 | 6.9% | 2 | 27 | 0 | -1.34% | -38.8% |
| SELL (all) | 25 | 10 | 40.0% | 0 | 20 | 5 | 0.06% | 1.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 25 | 10 | 40.0% | 0 | 20 | 5 | 0.06% | 1.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 54 | 12 | 22.2% | 2 | 47 | 5 | -0.69% | -37.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 1546.85 | 1572.45 | 1572.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 11:15:00 | 1534.40 | 1572.07 | 1572.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 14:15:00 | 1445.00 | 1444.63 | 1485.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-27 15:00:00 | 1445.00 | 1444.63 | 1485.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 1474.95 | 1446.41 | 1484.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 15:00:00 | 1453.90 | 1451.43 | 1484.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 11:45:00 | 1465.00 | 1449.90 | 1480.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 12:30:00 | 1465.15 | 1450.05 | 1479.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 1391.75 | 1447.76 | 1477.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 1391.89 | 1447.76 | 1477.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-11 10:15:00 | 1444.90 | 1442.92 | 1472.53 | SL hit (close>ema200) qty=0.50 sl=1442.92 alert=retest2 |

### Cycle 2 — BUY (started 2024-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-03 14:15:00 | 1614.00 | 1481.12 | 1480.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-04 11:15:00 | 1620.00 | 1486.27 | 1483.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 09:15:00 | 1522.55 | 1528.43 | 1509.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-22 10:15:00 | 1515.05 | 1528.43 | 1509.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 1526.75 | 1528.42 | 1509.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:30:00 | 1506.15 | 1528.42 | 1509.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 12:15:00 | 1523.95 | 1528.21 | 1509.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-22 15:00:00 | 1538.40 | 1528.22 | 1509.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-23 11:15:00 | 1507.05 | 1527.75 | 1510.05 | SL hit (close<static) qty=1.00 sl=1507.35 alert=retest2 |

### Cycle 3 — SELL (started 2024-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 09:15:00 | 1422.00 | 1502.82 | 1502.84 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 12:15:00 | 1577.75 | 1501.76 | 1501.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 14:15:00 | 1581.50 | 1503.33 | 1502.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-23 09:15:00 | 1548.25 | 1552.08 | 1532.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-23 10:00:00 | 1548.25 | 1552.08 | 1532.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 1532.05 | 1551.55 | 1532.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 11:00:00 | 1554.80 | 1542.48 | 1530.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 14:00:00 | 1543.70 | 1557.62 | 1541.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-08 14:30:00 | 1542.30 | 1557.44 | 1541.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 10:15:00 | 1527.70 | 1556.70 | 1541.13 | SL hit (close<static) qty=1.00 sl=1530.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 14:15:00 | 1508.45 | 1534.23 | 1534.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 1457.90 | 1533.15 | 1533.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 09:15:00 | 1444.80 | 1436.66 | 1476.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-20 10:00:00 | 1444.80 | 1436.66 | 1476.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 14:15:00 | 1480.00 | 1437.75 | 1475.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 15:00:00 | 1480.00 | 1437.75 | 1475.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 15:15:00 | 1473.50 | 1438.11 | 1475.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 09:15:00 | 1523.00 | 1438.11 | 1475.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 1502.70 | 1438.75 | 1475.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 10:15:00 | 1485.55 | 1438.75 | 1475.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 10:45:00 | 1483.65 | 1439.10 | 1475.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 15:15:00 | 1483.00 | 1441.00 | 1476.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 1411.27 | 1444.97 | 1474.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 1409.47 | 1444.97 | 1474.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 1408.85 | 1444.97 | 1474.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 1464.20 | 1440.74 | 1469.28 | SL hit (close>ema200) qty=0.50 sl=1440.74 alert=retest2 |

### Cycle 6 — BUY (started 2025-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 13:15:00 | 1551.60 | 1475.57 | 1475.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 15:15:00 | 1562.50 | 1477.27 | 1476.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1431.00 | 1485.68 | 1480.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 1431.00 | 1485.68 | 1480.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 1431.00 | 1485.68 | 1480.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 09:30:00 | 1504.50 | 1480.16 | 1478.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-07 11:15:00 | 1654.95 | 1563.79 | 1530.67 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 15:15:00 | 1551.60 | 1716.09 | 1716.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 09:15:00 | 1497.50 | 1713.92 | 1715.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 1638.10 | 1630.49 | 1660.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 09:45:00 | 1641.20 | 1630.49 | 1660.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 1665.00 | 1630.71 | 1659.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:00:00 | 1665.00 | 1630.71 | 1659.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 1664.90 | 1631.05 | 1659.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:30:00 | 1659.90 | 1631.30 | 1659.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 11:30:00 | 1659.10 | 1631.58 | 1653.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 12:00:00 | 1659.30 | 1631.58 | 1653.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 13:15:00 | 1655.00 | 1631.89 | 1653.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 1655.00 | 1632.58 | 1653.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 1674.10 | 1632.58 | 1653.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 1668.00 | 1632.94 | 1653.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 11:15:00 | 1647.20 | 1633.14 | 1653.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 14:15:00 | 1677.80 | 1634.35 | 1654.05 | SL hit (close>static) qty=1.00 sl=1669.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 09:15:00 | 1669.70 | 1666.01 | 1666.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 09:15:00 | 1689.40 | 1667.86 | 1666.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 1667.90 | 1669.40 | 1667.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 13:15:00 | 1667.90 | 1669.40 | 1667.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 1667.90 | 1669.40 | 1667.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:15:00 | 1657.90 | 1669.40 | 1667.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 1664.00 | 1669.35 | 1667.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 09:15:00 | 1682.60 | 1669.22 | 1667.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-12-22 10:15:00 | 1850.86 | 1751.24 | 1740.43 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 12:15:00 | 1910.30 | 2018.31 | 2018.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 09:15:00 | 1880.00 | 2013.93 | 2016.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 12:15:00 | 2033.90 | 2006.43 | 2012.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 12:15:00 | 2033.90 | 2006.43 | 2012.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 12:15:00 | 2033.90 | 2006.43 | 2012.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 13:00:00 | 2033.90 | 2006.43 | 2012.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 13:15:00 | 2034.40 | 2006.71 | 2012.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 13:30:00 | 2037.00 | 2006.71 | 2012.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 2011.90 | 2012.20 | 2015.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 11:00:00 | 2011.90 | 2012.20 | 2015.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 11:15:00 | 2012.00 | 2012.20 | 2015.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 11:30:00 | 2015.20 | 2012.20 | 2015.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 12:15:00 | 2011.50 | 2012.19 | 2015.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 13:15:00 | 2002.50 | 2012.19 | 2015.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 11:15:00 | 2031.70 | 2011.61 | 2014.64 | SL hit (close>static) qty=1.00 sl=2027.70 alert=retest2 |

### Cycle 10 — BUY (started 2026-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 13:15:00 | 2106.90 | 2017.77 | 2017.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 10:15:00 | 2121.80 | 2021.48 | 2019.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 14:15:00 | 2023.10 | 2039.66 | 2029.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 14:15:00 | 2023.10 | 2039.66 | 2029.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 2023.10 | 2039.66 | 2029.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 15:00:00 | 2023.10 | 2039.66 | 2029.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 2028.80 | 2039.55 | 2029.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:30:00 | 2013.00 | 2039.25 | 2029.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 1986.90 | 2038.73 | 2029.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:00:00 | 1986.90 | 2038.73 | 2029.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 2012.50 | 2036.68 | 2028.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 2012.50 | 2036.68 | 2028.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 13:15:00 | 1902.30 | 2021.13 | 2021.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 09:15:00 | 1878.20 | 2017.46 | 2019.38 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-03 12:00:00 | 1564.00 | 2024-06-04 10:15:00 | 1476.05 | STOP_HIT | 1.00 | -5.62% |
| BUY | retest2 | 2024-06-06 10:30:00 | 1564.90 | 2024-06-13 11:15:00 | 1552.70 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-06-06 13:15:00 | 1568.90 | 2024-06-27 14:15:00 | 1531.75 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2024-06-06 14:45:00 | 1573.65 | 2024-06-27 14:15:00 | 1531.75 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2024-06-07 10:00:00 | 1581.75 | 2024-06-27 14:15:00 | 1531.75 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2024-06-18 11:00:00 | 1578.95 | 2024-06-28 12:15:00 | 1525.00 | STOP_HIT | 1.00 | -3.42% |
| BUY | retest2 | 2024-06-18 13:45:00 | 1582.00 | 2024-06-28 12:15:00 | 1525.00 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2024-06-18 15:15:00 | 1584.00 | 2024-06-28 12:15:00 | 1525.00 | STOP_HIT | 1.00 | -3.72% |
| BUY | retest2 | 2024-06-28 10:30:00 | 1562.25 | 2024-06-28 12:15:00 | 1525.00 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2024-06-28 15:15:00 | 1560.00 | 2024-07-18 10:15:00 | 1546.85 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-07-05 10:15:00 | 1577.30 | 2024-07-18 10:15:00 | 1546.85 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-07-05 13:30:00 | 1560.00 | 2024-07-18 10:15:00 | 1546.85 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-08-30 15:00:00 | 1453.90 | 2024-09-09 09:15:00 | 1391.75 | PARTIAL | 0.50 | 4.27% |
| SELL | retest2 | 2024-09-05 11:45:00 | 1465.00 | 2024-09-09 09:15:00 | 1391.89 | PARTIAL | 0.50 | 4.99% |
| SELL | retest2 | 2024-08-30 15:00:00 | 1453.90 | 2024-09-11 10:15:00 | 1444.90 | STOP_HIT | 0.50 | 0.62% |
| SELL | retest2 | 2024-09-05 11:45:00 | 1465.00 | 2024-09-11 10:15:00 | 1444.90 | STOP_HIT | 0.50 | 1.37% |
| SELL | retest2 | 2024-09-05 12:30:00 | 1465.15 | 2024-09-24 15:15:00 | 1479.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-09-13 15:00:00 | 1455.25 | 2024-09-24 15:15:00 | 1479.00 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-09-16 10:45:00 | 1451.30 | 2024-09-24 15:15:00 | 1479.00 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2024-09-16 11:45:00 | 1449.50 | 2024-09-24 15:15:00 | 1479.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2024-09-16 14:30:00 | 1446.40 | 2024-09-25 09:15:00 | 1498.45 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2024-09-17 09:15:00 | 1441.45 | 2024-09-25 09:15:00 | 1498.45 | STOP_HIT | 1.00 | -3.95% |
| BUY | retest2 | 2024-10-22 15:00:00 | 1538.40 | 2024-10-23 11:15:00 | 1507.05 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-10-31 10:15:00 | 1526.20 | 2024-11-04 12:15:00 | 1505.70 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-10-31 14:45:00 | 1528.80 | 2024-11-04 12:15:00 | 1505.70 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-11-01 18:00:00 | 1539.00 | 2024-11-04 12:15:00 | 1505.70 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2024-11-04 15:00:00 | 1515.00 | 2024-11-08 11:15:00 | 1490.05 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-11-06 09:15:00 | 1536.60 | 2024-11-08 11:15:00 | 1490.05 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2024-11-08 14:30:00 | 1524.80 | 2024-11-13 13:15:00 | 1498.00 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-11-11 09:15:00 | 1534.45 | 2024-11-13 13:15:00 | 1498.00 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2024-12-31 11:00:00 | 1554.80 | 2025-01-09 10:15:00 | 1527.70 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-01-08 14:00:00 | 1543.70 | 2025-01-09 10:15:00 | 1527.70 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-01-08 14:30:00 | 1542.30 | 2025-01-09 10:15:00 | 1527.70 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-01-09 10:30:00 | 1542.50 | 2025-01-10 09:15:00 | 1519.80 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-01-09 13:30:00 | 1543.20 | 2025-01-10 09:15:00 | 1519.80 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-01-16 14:15:00 | 1547.30 | 2025-01-27 09:15:00 | 1508.20 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2025-01-22 12:15:00 | 1543.95 | 2025-01-27 09:15:00 | 1508.20 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-02-21 10:15:00 | 1485.55 | 2025-02-28 09:15:00 | 1411.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-21 10:45:00 | 1483.65 | 2025-02-28 09:15:00 | 1409.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-21 15:15:00 | 1483.00 | 2025-02-28 09:15:00 | 1408.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-21 10:15:00 | 1485.55 | 2025-03-05 09:15:00 | 1464.20 | STOP_HIT | 0.50 | 1.44% |
| SELL | retest2 | 2025-02-21 10:45:00 | 1483.65 | 2025-03-05 09:15:00 | 1464.20 | STOP_HIT | 0.50 | 1.31% |
| SELL | retest2 | 2025-02-21 15:15:00 | 1483.00 | 2025-03-05 09:15:00 | 1464.20 | STOP_HIT | 0.50 | 1.27% |
| SELL | retest2 | 2025-03-19 15:15:00 | 1489.05 | 2025-03-24 09:15:00 | 1540.00 | STOP_HIT | 1.00 | -3.42% |
| BUY | retest2 | 2025-04-11 09:30:00 | 1504.50 | 2025-05-07 11:15:00 | 1654.95 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-21 12:30:00 | 1659.90 | 2025-09-03 14:15:00 | 1677.80 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-09-02 11:30:00 | 1659.10 | 2025-09-03 14:15:00 | 1677.80 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-09-02 12:00:00 | 1659.30 | 2025-09-03 14:15:00 | 1677.80 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-09-02 13:15:00 | 1655.00 | 2025-09-03 14:15:00 | 1677.80 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-09-03 11:15:00 | 1647.20 | 2025-09-08 10:15:00 | 1679.90 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-09-05 12:00:00 | 1653.30 | 2025-09-08 10:15:00 | 1679.90 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-09-05 13:00:00 | 1653.60 | 2025-09-08 10:15:00 | 1679.90 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-09-29 09:15:00 | 1682.60 | 2025-12-22 10:15:00 | 1850.86 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-13 13:15:00 | 2002.50 | 2026-04-15 11:15:00 | 2031.70 | STOP_HIT | 1.00 | -1.46% |
