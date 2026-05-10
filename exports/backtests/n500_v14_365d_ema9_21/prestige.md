# Prestige Estates Projects Ltd. (PRESTIGE)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 1495.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 69 |
| ALERT1 | 52 |
| ALERT2 | 49 |
| ALERT2_SKIP | 24 |
| ALERT3 | 120 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 59 |
| PARTIAL | 9 |
| TARGET_HIT | 9 |
| STOP_HIT | 55 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 71 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 30 / 41
- **Target hits / Stop hits / Partials:** 9 / 53 / 9
- **Avg / median % per leg:** 1.61% / -0.27%
- **Sum % (uncompounded):** 114.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 10 | 47.6% | 6 | 15 | 0 | 2.88% | 60.6% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.60% | -5.2% |
| BUY @ 3rd Alert (retest2) | 19 | 10 | 52.6% | 6 | 13 | 0 | 3.46% | 65.8% |
| SELL (all) | 50 | 20 | 40.0% | 3 | 38 | 9 | 1.08% | 54.0% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.32% | -1.0% |
| SELL @ 3rd Alert (retest2) | 47 | 20 | 42.6% | 3 | 35 | 9 | 1.17% | 54.9% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.23% | -6.2% |
| retest2 (combined) | 66 | 30 | 45.5% | 9 | 48 | 9 | 1.83% | 120.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 1338.50 | 1318.23 | 1316.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 1371.00 | 1328.78 | 1321.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 1347.80 | 1350.91 | 1339.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 1347.80 | 1350.91 | 1339.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 15:15:00 | 1348.00 | 1350.05 | 1341.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 1365.00 | 1350.05 | 1341.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 12:15:00 | 1415.00 | 1431.92 | 1432.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 12:15:00 | 1415.00 | 1431.92 | 1432.30 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 13:15:00 | 1431.60 | 1430.36 | 1430.33 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 09:15:00 | 1426.00 | 1430.44 | 1430.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 10:15:00 | 1414.70 | 1427.29 | 1429.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 13:15:00 | 1434.40 | 1426.61 | 1428.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 13:15:00 | 1434.40 | 1426.61 | 1428.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 1434.40 | 1426.61 | 1428.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:00:00 | 1434.40 | 1426.61 | 1428.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 14:15:00 | 1438.90 | 1429.07 | 1429.05 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 10:15:00 | 1425.30 | 1428.61 | 1428.95 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 13:15:00 | 1432.90 | 1429.55 | 1429.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 14:15:00 | 1446.20 | 1432.88 | 1430.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 13:15:00 | 1450.70 | 1451.02 | 1442.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 13:30:00 | 1451.50 | 1451.02 | 1442.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 1462.10 | 1482.14 | 1476.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 1462.10 | 1482.14 | 1476.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 1484.00 | 1482.51 | 1477.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 1524.10 | 1482.51 | 1477.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-06 10:15:00 | 1676.51 | 1636.23 | 1606.29 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 11:15:00 | 1664.20 | 1672.93 | 1673.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 1661.40 | 1667.10 | 1669.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 1660.30 | 1640.51 | 1648.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 14:15:00 | 1660.30 | 1640.51 | 1648.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 1660.30 | 1640.51 | 1648.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 1660.30 | 1640.51 | 1648.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 1656.20 | 1643.65 | 1648.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 1628.60 | 1643.65 | 1648.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 10:15:00 | 1666.00 | 1647.77 | 1649.86 | SL hit (close>static) qty=1.00 sl=1660.40 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 1688.50 | 1655.92 | 1653.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 14:15:00 | 1701.40 | 1672.92 | 1662.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 14:15:00 | 1695.00 | 1698.38 | 1683.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 14:45:00 | 1695.10 | 1698.38 | 1683.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 1701.90 | 1698.39 | 1685.96 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 13:15:00 | 1675.30 | 1686.87 | 1687.30 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 10:15:00 | 1713.80 | 1691.18 | 1688.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 13:15:00 | 1724.70 | 1706.50 | 1699.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 1728.00 | 1730.16 | 1717.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 13:45:00 | 1731.80 | 1730.16 | 1717.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 1723.40 | 1726.83 | 1719.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 10:30:00 | 1719.20 | 1726.83 | 1719.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 1709.50 | 1723.36 | 1718.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:00:00 | 1709.50 | 1723.36 | 1718.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 1718.00 | 1722.29 | 1718.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 13:30:00 | 1721.70 | 1722.33 | 1719.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 10:15:00 | 1707.70 | 1723.99 | 1721.64 | SL hit (close<static) qty=1.00 sl=1709.40 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 10:30:00 | 1719.90 | 1723.99 | 1721.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 11:15:00 | 1700.00 | 1719.19 | 1719.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 1700.00 | 1719.19 | 1719.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 09:15:00 | 1678.80 | 1702.35 | 1710.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 1620.70 | 1614.71 | 1632.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 10:00:00 | 1620.70 | 1614.71 | 1632.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 1614.60 | 1602.98 | 1613.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 1614.60 | 1602.98 | 1613.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 1618.00 | 1605.99 | 1613.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 1613.00 | 1605.99 | 1613.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1608.20 | 1606.43 | 1613.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 1613.90 | 1606.43 | 1613.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 1612.90 | 1607.72 | 1613.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:00:00 | 1612.90 | 1607.72 | 1613.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 1614.50 | 1609.08 | 1613.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:30:00 | 1616.60 | 1609.08 | 1613.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 1613.60 | 1609.98 | 1613.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:45:00 | 1611.90 | 1609.98 | 1613.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 1619.70 | 1611.93 | 1613.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 14:00:00 | 1619.70 | 1611.93 | 1613.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 14:15:00 | 1631.40 | 1615.82 | 1615.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 11:15:00 | 1632.00 | 1624.78 | 1620.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 10:15:00 | 1640.30 | 1642.92 | 1632.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 11:00:00 | 1640.30 | 1642.92 | 1632.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 1647.90 | 1643.46 | 1634.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:45:00 | 1633.80 | 1643.46 | 1634.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 1637.30 | 1642.23 | 1635.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 09:15:00 | 1667.80 | 1644.54 | 1637.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 10:15:00 | 1741.00 | 1775.47 | 1779.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 10:15:00 | 1741.00 | 1775.47 | 1779.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 1729.80 | 1752.59 | 1764.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 09:15:00 | 1633.90 | 1628.15 | 1657.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 09:45:00 | 1634.00 | 1628.15 | 1657.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 1626.70 | 1628.34 | 1641.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:30:00 | 1639.30 | 1628.34 | 1641.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 1621.60 | 1613.39 | 1624.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 14:00:00 | 1621.60 | 1613.39 | 1624.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 1626.40 | 1615.99 | 1624.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 14:30:00 | 1631.50 | 1615.99 | 1624.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 1625.00 | 1617.79 | 1624.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:15:00 | 1616.90 | 1617.79 | 1624.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1651.20 | 1624.47 | 1627.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 1651.20 | 1624.47 | 1627.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 1641.40 | 1627.86 | 1628.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:30:00 | 1652.50 | 1627.86 | 1628.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 1627.60 | 1627.22 | 1627.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 12:30:00 | 1623.30 | 1627.22 | 1627.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 1618.00 | 1625.38 | 1627.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 14:15:00 | 1604.30 | 1625.38 | 1627.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 13:15:00 | 1631.40 | 1618.27 | 1620.37 | SL hit (close>static) qty=1.00 sl=1630.30 alert=retest2 |

### Cycle 15 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 1628.00 | 1622.69 | 1622.16 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 1608.30 | 1619.81 | 1620.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 13:15:00 | 1602.20 | 1614.41 | 1617.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 12:15:00 | 1616.70 | 1599.15 | 1607.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 12:15:00 | 1616.70 | 1599.15 | 1607.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 1616.70 | 1599.15 | 1607.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 13:00:00 | 1616.70 | 1599.15 | 1607.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 1607.40 | 1600.80 | 1607.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 14:15:00 | 1604.40 | 1600.80 | 1607.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 15:15:00 | 1600.00 | 1601.84 | 1607.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 09:45:00 | 1604.00 | 1600.04 | 1605.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 1630.00 | 1611.56 | 1609.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 1630.00 | 1611.56 | 1609.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 1630.00 | 1611.56 | 1609.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 1630.00 | 1611.56 | 1609.25 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 14:15:00 | 1600.60 | 1607.79 | 1608.54 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 1614.70 | 1609.01 | 1608.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 11:15:00 | 1626.50 | 1612.51 | 1610.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 10:15:00 | 1628.80 | 1634.23 | 1624.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 11:00:00 | 1628.80 | 1634.23 | 1624.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 1634.80 | 1633.51 | 1627.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:15:00 | 1645.80 | 1633.51 | 1627.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 1635.00 | 1633.81 | 1628.46 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 1611.90 | 1628.56 | 1628.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 11:15:00 | 1602.80 | 1620.76 | 1625.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1639.10 | 1615.55 | 1619.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 1639.10 | 1615.55 | 1619.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1639.10 | 1615.55 | 1619.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 1639.10 | 1615.55 | 1619.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 1634.40 | 1619.32 | 1621.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 12:00:00 | 1623.20 | 1620.10 | 1621.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 13:15:00 | 1633.90 | 1624.33 | 1623.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 13:15:00 | 1633.90 | 1624.33 | 1623.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 09:15:00 | 1651.90 | 1631.09 | 1627.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 13:15:00 | 1636.20 | 1639.05 | 1633.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 13:15:00 | 1636.20 | 1639.05 | 1633.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 1636.20 | 1639.05 | 1633.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:00:00 | 1636.20 | 1639.05 | 1633.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 1628.00 | 1636.84 | 1632.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 1628.00 | 1636.84 | 1632.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 1637.00 | 1636.87 | 1633.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 10:15:00 | 1663.70 | 1639.62 | 1636.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 11:45:00 | 1660.50 | 1646.11 | 1640.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 12:45:00 | 1656.30 | 1648.23 | 1641.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 1625.20 | 1645.27 | 1642.56 | SL hit (close<static) qty=1.00 sl=1626.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 1625.20 | 1645.27 | 1642.56 | SL hit (close<static) qty=1.00 sl=1626.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 1625.20 | 1645.27 | 1642.56 | SL hit (close<static) qty=1.00 sl=1626.60 alert=retest2 |

### Cycle 22 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 1618.10 | 1639.84 | 1640.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 1613.00 | 1628.76 | 1634.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 15:15:00 | 1590.00 | 1589.87 | 1606.19 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 09:15:00 | 1566.20 | 1589.87 | 1606.19 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 11:30:00 | 1576.10 | 1580.75 | 1597.22 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-29 12:15:00 | 1572.40 | 1580.75 | 1597.22 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 1576.60 | 1566.73 | 1576.54 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-01 14:15:00 | 1576.60 | 1566.73 | 1576.54 | SL hit (close>ema400) qty=1.00 sl=1576.54 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-01 14:15:00 | 1576.60 | 1566.73 | 1576.54 | SL hit (close>ema400) qty=1.00 sl=1576.54 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-01 14:15:00 | 1576.60 | 1566.73 | 1576.54 | SL hit (close>ema400) qty=1.00 sl=1576.54 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 1576.60 | 1566.73 | 1576.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 1584.00 | 1570.19 | 1577.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 1568.20 | 1570.19 | 1577.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1585.90 | 1573.33 | 1578.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 14:00:00 | 1562.50 | 1571.82 | 1575.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 14:15:00 | 1585.10 | 1575.40 | 1575.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 14:15:00 | 1585.10 | 1575.40 | 1575.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 15:15:00 | 1589.60 | 1578.24 | 1576.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 11:15:00 | 1578.00 | 1583.82 | 1580.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 11:15:00 | 1578.00 | 1583.82 | 1580.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 1578.00 | 1583.82 | 1580.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:00:00 | 1578.00 | 1583.82 | 1580.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 1568.00 | 1580.65 | 1578.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:30:00 | 1569.90 | 1580.65 | 1578.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 1556.00 | 1575.72 | 1576.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 1553.80 | 1571.34 | 1574.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1540.80 | 1528.33 | 1544.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 1540.80 | 1528.33 | 1544.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1540.80 | 1528.33 | 1544.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:45:00 | 1540.00 | 1528.33 | 1544.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1539.20 | 1530.50 | 1544.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 1532.00 | 1537.36 | 1543.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1548.60 | 1543.63 | 1543.85 | SL hit (close>static) qty=1.00 sl=1548.40 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 1562.00 | 1547.31 | 1545.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 09:15:00 | 1570.40 | 1558.27 | 1552.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 13:15:00 | 1551.60 | 1558.82 | 1554.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 13:15:00 | 1551.60 | 1558.82 | 1554.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 1551.60 | 1558.82 | 1554.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 1551.60 | 1558.82 | 1554.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 1557.60 | 1558.58 | 1555.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 1564.50 | 1559.06 | 1555.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 10:00:00 | 1566.90 | 1560.63 | 1556.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 11:15:00 | 1544.00 | 1556.82 | 1555.59 | SL hit (close<static) qty=1.00 sl=1550.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-12 11:15:00 | 1544.00 | 1556.82 | 1555.59 | SL hit (close<static) qty=1.00 sl=1550.30 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 12:15:00 | 1536.90 | 1552.84 | 1553.89 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 11:15:00 | 1573.80 | 1556.40 | 1554.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 12:15:00 | 1593.60 | 1563.84 | 1558.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 11:15:00 | 1638.80 | 1646.10 | 1627.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 11:45:00 | 1636.50 | 1646.10 | 1627.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 1635.70 | 1644.02 | 1628.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:45:00 | 1634.80 | 1644.02 | 1628.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1628.20 | 1640.85 | 1628.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:00:00 | 1628.20 | 1640.85 | 1628.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 1631.10 | 1638.90 | 1628.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:15:00 | 1627.10 | 1638.90 | 1628.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 1627.10 | 1636.54 | 1628.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 1628.00 | 1636.54 | 1628.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1620.70 | 1633.37 | 1627.96 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 11:15:00 | 1601.30 | 1622.77 | 1623.82 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 10:15:00 | 1635.40 | 1625.43 | 1624.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 11:15:00 | 1640.00 | 1628.34 | 1625.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 1621.50 | 1630.43 | 1627.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 14:15:00 | 1621.50 | 1630.43 | 1627.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 1621.50 | 1630.43 | 1627.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:45:00 | 1619.40 | 1630.43 | 1627.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 1614.00 | 1627.14 | 1626.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:15:00 | 1612.60 | 1627.14 | 1626.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 09:15:00 | 1609.70 | 1623.66 | 1625.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 10:15:00 | 1580.00 | 1603.99 | 1613.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 10:15:00 | 1526.60 | 1521.56 | 1540.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:45:00 | 1527.70 | 1521.56 | 1540.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1524.60 | 1522.41 | 1534.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 1514.60 | 1522.11 | 1533.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:45:00 | 1518.80 | 1521.55 | 1532.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:00:00 | 1518.80 | 1510.99 | 1520.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:45:00 | 1518.60 | 1512.91 | 1520.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 1519.40 | 1514.21 | 1520.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 12:45:00 | 1514.50 | 1514.71 | 1519.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 1526.40 | 1517.04 | 1520.36 | SL hit (close>static) qty=1.00 sl=1523.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 1543.40 | 1522.32 | 1522.45 | SL hit (close>static) qty=1.00 sl=1540.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 1543.40 | 1522.32 | 1522.45 | SL hit (close>static) qty=1.00 sl=1540.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 1543.40 | 1522.32 | 1522.45 | SL hit (close>static) qty=1.00 sl=1540.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 1543.40 | 1522.32 | 1522.45 | SL hit (close>static) qty=1.00 sl=1540.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 1542.00 | 1526.25 | 1524.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 15:15:00 | 1548.10 | 1536.48 | 1532.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 10:15:00 | 1530.70 | 1535.98 | 1532.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 10:15:00 | 1530.70 | 1535.98 | 1532.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 1530.70 | 1535.98 | 1532.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 1530.70 | 1535.98 | 1532.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 1528.00 | 1534.39 | 1532.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:00:00 | 1528.00 | 1534.39 | 1532.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 1529.00 | 1533.31 | 1531.97 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 15:15:00 | 1530.00 | 1531.09 | 1531.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 1516.70 | 1528.21 | 1529.84 | Break + close below crossover candle low |

### Cycle 33 — BUY (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 09:15:00 | 1569.40 | 1526.64 | 1526.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 10:15:00 | 1626.20 | 1584.23 | 1560.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 12:15:00 | 1616.60 | 1620.31 | 1598.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 12:30:00 | 1615.80 | 1620.31 | 1598.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 1603.30 | 1613.65 | 1601.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:45:00 | 1601.80 | 1613.65 | 1601.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 1600.10 | 1610.94 | 1601.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:45:00 | 1602.40 | 1610.94 | 1601.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 1588.80 | 1606.51 | 1600.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:00:00 | 1588.80 | 1606.51 | 1600.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 1600.30 | 1604.23 | 1600.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 14:30:00 | 1606.00 | 1604.10 | 1600.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 15:15:00 | 1604.40 | 1604.10 | 1600.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-24 12:15:00 | 1764.84 | 1744.47 | 1730.02 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-10-27 09:15:00 | 1766.60 | 1756.57 | 1740.50 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 11:15:00 | 1747.30 | 1757.32 | 1758.18 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 1763.70 | 1756.32 | 1756.09 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 13:15:00 | 1751.90 | 1755.33 | 1755.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 1740.20 | 1752.30 | 1754.32 | Break + close below crossover candle low |

### Cycle 37 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 1772.50 | 1756.29 | 1755.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 10:15:00 | 1774.20 | 1759.87 | 1757.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 09:15:00 | 1774.20 | 1776.52 | 1768.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 10:00:00 | 1774.20 | 1776.52 | 1768.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 1765.50 | 1774.31 | 1768.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:30:00 | 1767.80 | 1774.31 | 1768.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 1759.50 | 1771.35 | 1767.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:30:00 | 1759.30 | 1771.35 | 1767.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 1763.30 | 1768.96 | 1767.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 13:45:00 | 1763.20 | 1768.96 | 1767.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 14:15:00 | 1742.90 | 1763.75 | 1765.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 1739.80 | 1756.12 | 1761.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 1740.10 | 1731.23 | 1742.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 12:00:00 | 1740.10 | 1731.23 | 1742.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 1728.80 | 1730.74 | 1740.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 1728.80 | 1730.74 | 1740.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 1737.30 | 1733.06 | 1740.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:15:00 | 1736.00 | 1733.06 | 1740.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1736.00 | 1733.65 | 1739.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 1737.90 | 1733.65 | 1739.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1736.80 | 1734.28 | 1739.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 1753.30 | 1734.28 | 1739.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1747.10 | 1736.84 | 1740.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 1743.30 | 1736.84 | 1740.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 1742.60 | 1737.99 | 1740.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:30:00 | 1751.60 | 1737.99 | 1740.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 1755.70 | 1742.98 | 1742.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 15:15:00 | 1765.00 | 1753.73 | 1748.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 10:15:00 | 1739.70 | 1751.76 | 1748.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 10:15:00 | 1739.70 | 1751.76 | 1748.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 1739.70 | 1751.76 | 1748.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:00:00 | 1739.70 | 1751.76 | 1748.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 1732.00 | 1747.81 | 1747.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 12:00:00 | 1732.00 | 1747.81 | 1747.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 12:15:00 | 1726.70 | 1743.59 | 1745.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 13:15:00 | 1712.70 | 1737.41 | 1742.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 09:15:00 | 1758.80 | 1730.55 | 1737.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 1758.80 | 1730.55 | 1737.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1758.80 | 1730.55 | 1737.08 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 12:15:00 | 1748.20 | 1741.60 | 1741.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 14:15:00 | 1756.70 | 1745.00 | 1742.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 1725.60 | 1742.22 | 1742.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 09:15:00 | 1725.60 | 1742.22 | 1742.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1725.60 | 1742.22 | 1742.04 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 1721.20 | 1738.02 | 1740.14 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-11-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 13:15:00 | 1749.00 | 1741.42 | 1741.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 13:15:00 | 1760.00 | 1749.77 | 1745.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1724.60 | 1745.74 | 1745.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 1724.60 | 1745.74 | 1745.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1724.60 | 1745.74 | 1745.15 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 1734.70 | 1743.54 | 1744.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 1720.00 | 1735.27 | 1739.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 13:15:00 | 1715.40 | 1714.36 | 1722.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 14:00:00 | 1715.40 | 1714.36 | 1722.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 1720.80 | 1715.64 | 1721.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 1720.80 | 1715.64 | 1721.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 1724.00 | 1717.32 | 1722.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:15:00 | 1708.60 | 1717.32 | 1722.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1703.00 | 1714.45 | 1720.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 11:00:00 | 1683.00 | 1708.16 | 1716.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 14:15:00 | 1691.00 | 1698.26 | 1709.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 13:15:00 | 1675.80 | 1671.15 | 1671.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-28 13:15:00 | 1675.80 | 1671.15 | 1671.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 13:15:00 | 1675.80 | 1671.15 | 1671.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 14:15:00 | 1677.10 | 1672.34 | 1671.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 10:15:00 | 1674.40 | 1675.18 | 1673.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 10:15:00 | 1674.40 | 1675.18 | 1673.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1674.40 | 1675.18 | 1673.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:00:00 | 1674.40 | 1675.18 | 1673.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 11:15:00 | 1651.50 | 1670.45 | 1671.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 1643.50 | 1652.61 | 1658.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 13:15:00 | 1648.40 | 1648.13 | 1654.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 14:00:00 | 1648.40 | 1648.13 | 1654.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1646.50 | 1643.83 | 1649.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 1646.50 | 1643.83 | 1649.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 1661.70 | 1647.41 | 1651.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:00:00 | 1661.70 | 1647.41 | 1651.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 1662.90 | 1650.50 | 1652.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:30:00 | 1663.60 | 1650.50 | 1652.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 14:15:00 | 1661.10 | 1654.40 | 1653.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 09:15:00 | 1691.90 | 1662.38 | 1657.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 1640.50 | 1675.08 | 1669.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 1640.50 | 1675.08 | 1669.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 1640.50 | 1675.08 | 1669.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 1640.50 | 1675.08 | 1669.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 1625.40 | 1665.15 | 1665.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 1621.10 | 1656.34 | 1661.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 1643.10 | 1626.14 | 1639.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 11:15:00 | 1643.10 | 1626.14 | 1639.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 1643.10 | 1626.14 | 1639.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 1643.10 | 1626.14 | 1639.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 1631.00 | 1627.11 | 1638.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:45:00 | 1629.10 | 1629.37 | 1637.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 14:00:00 | 1618.40 | 1632.23 | 1636.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 1650.50 | 1631.34 | 1634.41 | SL hit (close>static) qty=1.00 sl=1648.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 1650.50 | 1631.34 | 1634.41 | SL hit (close>static) qty=1.00 sl=1648.80 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 09:30:00 | 1627.40 | 1631.34 | 1634.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 10:15:00 | 1651.00 | 1635.27 | 1635.92 | SL hit (close>static) qty=1.00 sl=1648.80 alert=retest2 |

### Cycle 49 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 1645.40 | 1637.30 | 1636.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 1670.60 | 1649.44 | 1643.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 10:15:00 | 1648.80 | 1649.31 | 1643.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-12 10:45:00 | 1645.20 | 1649.31 | 1643.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 1645.90 | 1648.63 | 1644.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 13:15:00 | 1654.60 | 1647.79 | 1644.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 09:15:00 | 1641.80 | 1650.98 | 1647.19 | SL hit (close<static) qty=1.00 sl=1643.70 alert=retest2 |

### Cycle 50 — SELL (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 12:15:00 | 1635.50 | 1644.66 | 1644.93 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 14:15:00 | 1656.20 | 1646.28 | 1645.58 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 1630.00 | 1643.57 | 1644.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 1619.70 | 1636.77 | 1641.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 1614.70 | 1605.37 | 1612.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 1614.70 | 1605.37 | 1612.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1614.70 | 1605.37 | 1612.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 1619.80 | 1605.37 | 1612.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 1607.90 | 1605.87 | 1611.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 1625.00 | 1605.87 | 1611.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 1604.60 | 1605.24 | 1610.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:30:00 | 1606.80 | 1605.24 | 1610.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 1618.00 | 1607.79 | 1611.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 1618.00 | 1607.79 | 1611.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 1625.90 | 1611.41 | 1612.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:45:00 | 1617.50 | 1611.41 | 1612.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 1628.30 | 1615.76 | 1614.40 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 14:15:00 | 1605.00 | 1614.14 | 1614.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 11:15:00 | 1599.20 | 1610.86 | 1612.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 1610.60 | 1607.25 | 1609.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 09:15:00 | 1610.60 | 1607.25 | 1609.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1610.60 | 1607.25 | 1609.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 13:00:00 | 1595.80 | 1605.85 | 1608.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 15:15:00 | 1619.90 | 1611.52 | 1610.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 15:15:00 | 1619.90 | 1611.52 | 1610.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 09:15:00 | 1635.80 | 1616.38 | 1613.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 12:15:00 | 1614.70 | 1621.25 | 1616.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 12:15:00 | 1614.70 | 1621.25 | 1616.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 1614.70 | 1621.25 | 1616.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 13:00:00 | 1614.70 | 1621.25 | 1616.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 1624.60 | 1621.92 | 1617.30 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 1605.20 | 1614.07 | 1614.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 13:15:00 | 1593.30 | 1604.41 | 1609.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 1576.00 | 1575.97 | 1588.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:15:00 | 1574.30 | 1575.97 | 1588.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1588.10 | 1577.17 | 1586.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 1588.10 | 1577.17 | 1586.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 1589.80 | 1579.70 | 1587.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:30:00 | 1589.70 | 1579.70 | 1587.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 1595.70 | 1582.90 | 1587.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 13:30:00 | 1587.10 | 1583.98 | 1587.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 15:15:00 | 1597.50 | 1588.25 | 1589.25 | SL hit (close>static) qty=1.00 sl=1597.10 alert=retest2 |

### Cycle 57 — BUY (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 10:15:00 | 1606.00 | 1592.69 | 1591.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 13:15:00 | 1608.50 | 1599.48 | 1594.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 13:15:00 | 1659.20 | 1659.79 | 1642.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 14:00:00 | 1659.20 | 1659.79 | 1642.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 1649.00 | 1656.71 | 1645.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:45:00 | 1636.00 | 1656.71 | 1645.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 1635.80 | 1652.53 | 1644.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 1635.80 | 1652.53 | 1644.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 1622.30 | 1646.48 | 1642.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:00:00 | 1622.30 | 1646.48 | 1642.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 13:15:00 | 1619.10 | 1635.78 | 1638.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 1601.70 | 1621.89 | 1630.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 1518.60 | 1511.15 | 1531.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 15:00:00 | 1518.60 | 1511.15 | 1531.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1549.80 | 1507.10 | 1514.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:00:00 | 1549.80 | 1507.10 | 1514.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 1539.80 | 1513.64 | 1516.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:45:00 | 1533.00 | 1517.91 | 1518.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:15:00 | 1533.30 | 1517.91 | 1518.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 12:15:00 | 1533.40 | 1521.01 | 1519.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-16 12:15:00 | 1533.40 | 1521.01 | 1519.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 12:15:00 | 1533.40 | 1521.01 | 1519.92 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 1510.90 | 1519.70 | 1519.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 10:15:00 | 1498.60 | 1515.48 | 1517.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 14:15:00 | 1443.10 | 1430.98 | 1450.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 15:00:00 | 1443.10 | 1430.98 | 1450.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1439.60 | 1432.59 | 1448.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 1430.10 | 1432.59 | 1448.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 1358.59 | 1393.55 | 1410.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 14:15:00 | 1389.60 | 1378.54 | 1395.17 | SL hit (close>ema200) qty=0.50 sl=1378.54 alert=retest2 |

### Cycle 61 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 1410.00 | 1401.84 | 1401.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 1421.90 | 1405.85 | 1403.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 1410.30 | 1411.50 | 1407.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 10:15:00 | 1410.30 | 1411.50 | 1407.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 1410.30 | 1411.50 | 1407.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 1408.90 | 1411.50 | 1407.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 1406.80 | 1410.56 | 1407.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:00:00 | 1406.80 | 1410.56 | 1407.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 1418.90 | 1412.23 | 1408.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 13:15:00 | 1426.50 | 1412.23 | 1408.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 09:30:00 | 1426.50 | 1422.18 | 1414.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 12:30:00 | 1453.80 | 1455.69 | 1440.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-03 09:15:00 | 1569.15 | 1483.41 | 1465.83 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-02-03 09:15:00 | 1569.15 | 1483.41 | 1465.83 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-02-03 09:15:00 | 1599.18 | 1483.41 | 1465.83 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 14:15:00 | 1570.80 | 1576.03 | 1576.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 1491.10 | 1558.21 | 1567.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 1527.10 | 1522.52 | 1534.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 1527.10 | 1522.52 | 1534.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 1527.00 | 1523.42 | 1533.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 1516.50 | 1523.42 | 1533.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 1542.40 | 1527.18 | 1532.14 | SL hit (close>static) qty=1.00 sl=1534.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 15:15:00 | 1520.90 | 1527.80 | 1531.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:45:00 | 1514.00 | 1523.93 | 1529.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:30:00 | 1514.50 | 1518.18 | 1522.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 1497.20 | 1491.93 | 1501.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:00:00 | 1497.20 | 1491.93 | 1501.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 1486.60 | 1490.87 | 1499.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 12:00:00 | 1482.40 | 1489.01 | 1496.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 12:45:00 | 1482.60 | 1488.25 | 1495.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:30:00 | 1482.20 | 1486.74 | 1493.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 1467.30 | 1486.14 | 1492.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 11:15:00 | 1444.86 | 1470.64 | 1483.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 12:15:00 | 1438.30 | 1461.71 | 1477.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 12:15:00 | 1438.77 | 1461.71 | 1477.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 14:15:00 | 1408.47 | 1446.58 | 1467.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 1453.40 | 1445.45 | 1463.43 | SL hit (close>ema200) qty=0.50 sl=1445.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 1453.40 | 1445.45 | 1463.43 | SL hit (close>ema200) qty=0.50 sl=1445.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 1453.40 | 1445.45 | 1463.43 | SL hit (close>ema200) qty=0.50 sl=1445.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 1453.40 | 1445.45 | 1463.43 | SL hit (close>ema200) qty=0.50 sl=1445.45 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 1450.60 | 1446.48 | 1462.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:15:00 | 1445.90 | 1446.48 | 1462.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 1408.28 | 1424.92 | 1436.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 09:15:00 | 1408.09 | 1424.92 | 1436.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 11:15:00 | 1393.93 | 1417.53 | 1430.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-02 09:15:00 | 1334.16 | 1397.73 | 1414.87 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-02 09:15:00 | 1333.98 | 1397.73 | 1414.87 | Target hit (10%) qty=0.50 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1373.61 | 1397.73 | 1414.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 10:15:00 | 1320.57 | 1360.97 | 1384.32 | Target hit (10%) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 14:15:00 | 1352.90 | 1347.66 | 1369.10 | SL hit (close>ema200) qty=0.50 sl=1347.66 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 15:15:00 | 1272.00 | 1253.09 | 1252.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 1293.30 | 1264.00 | 1257.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1272.50 | 1294.83 | 1279.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1272.50 | 1294.83 | 1279.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1272.50 | 1294.83 | 1279.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 1272.50 | 1294.83 | 1279.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 1279.60 | 1291.79 | 1279.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 1290.40 | 1278.71 | 1276.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 13:15:00 | 1261.60 | 1274.21 | 1275.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 13:15:00 | 1261.60 | 1274.21 | 1275.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 14:15:00 | 1246.90 | 1268.75 | 1273.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1200.30 | 1198.21 | 1218.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:30:00 | 1208.50 | 1198.21 | 1218.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1238.00 | 1208.84 | 1217.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 1238.60 | 1208.84 | 1217.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 1231.70 | 1213.41 | 1218.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 11:15:00 | 1225.90 | 1213.41 | 1218.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 14:00:00 | 1227.20 | 1219.09 | 1220.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 14:15:00 | 1228.50 | 1220.97 | 1220.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-25 14:15:00 | 1228.50 | 1220.97 | 1220.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 14:15:00 | 1228.50 | 1220.97 | 1220.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 15:15:00 | 1234.00 | 1223.58 | 1222.13 | Break + close above crossover candle high |

### Cycle 66 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 1178.10 | 1214.48 | 1218.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 1174.90 | 1206.57 | 1214.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1158.60 | 1143.05 | 1163.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1158.60 | 1143.05 | 1163.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1158.60 | 1143.05 | 1163.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:45:00 | 1140.00 | 1142.52 | 1161.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:45:00 | 1141.80 | 1146.53 | 1157.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 1101.50 | 1146.23 | 1156.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 1139.40 | 1138.55 | 1144.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 1149.20 | 1144.14 | 1146.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:45:00 | 1157.40 | 1144.14 | 1146.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 1176.50 | 1152.51 | 1149.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 1176.50 | 1152.51 | 1149.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 1176.50 | 1152.51 | 1149.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 1176.50 | 1152.51 | 1149.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 1176.50 | 1152.51 | 1149.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 1193.40 | 1160.69 | 1153.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 1310.40 | 1315.17 | 1286.52 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 1329.50 | 1315.17 | 1286.52 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 15:00:00 | 1322.60 | 1323.31 | 1304.30 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1291.50 | 1317.32 | 1304.90 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 1291.50 | 1317.32 | 1304.90 | SL hit (close<ema400) qty=1.00 sl=1304.90 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 1291.50 | 1317.32 | 1304.90 | SL hit (close<ema400) qty=1.00 sl=1304.90 alert=retest1 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1303.50 | 1317.32 | 1304.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:15:00 | 1303.00 | 1313.51 | 1304.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 15:15:00 | 1380.00 | 1393.68 | 1393.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-23 15:15:00 | 1380.00 | 1393.68 | 1393.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 1380.00 | 1393.68 | 1393.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 1360.40 | 1384.90 | 1389.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 15:15:00 | 1375.00 | 1373.04 | 1380.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 09:15:00 | 1390.70 | 1373.04 | 1380.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1392.40 | 1376.91 | 1381.74 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 1399.60 | 1385.05 | 1384.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 1408.00 | 1389.64 | 1386.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 1394.30 | 1400.01 | 1395.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 13:15:00 | 1394.30 | 1400.01 | 1395.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 1394.30 | 1400.01 | 1395.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:00:00 | 1394.30 | 1400.01 | 1395.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 1402.90 | 1400.59 | 1396.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 1413.50 | 1401.67 | 1397.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 13:00:00 | 1408.30 | 1413.79 | 1411.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 09:15:00 | 1365.00 | 2025-05-22 12:15:00 | 1415.00 | STOP_HIT | 1.00 | 3.66% |
| BUY | retest2 | 2025-06-02 09:15:00 | 1524.10 | 2025-06-06 10:15:00 | 1676.51 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-16 09:15:00 | 1628.60 | 2025-06-16 10:15:00 | 1666.00 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-06-25 13:30:00 | 1721.70 | 2025-06-26 10:15:00 | 1707.70 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-06-26 10:30:00 | 1719.90 | 2025-06-26 11:15:00 | 1700.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-07-10 09:15:00 | 1667.80 | 2025-07-23 10:15:00 | 1741.00 | STOP_HIT | 1.00 | 4.39% |
| SELL | retest2 | 2025-08-01 14:15:00 | 1604.30 | 2025-08-04 13:15:00 | 1631.40 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-08-06 14:15:00 | 1604.40 | 2025-08-07 14:15:00 | 1630.00 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-08-06 15:15:00 | 1600.00 | 2025-08-07 14:15:00 | 1630.00 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-08-07 09:45:00 | 1604.00 | 2025-08-07 14:15:00 | 1630.00 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-08-18 12:00:00 | 1623.20 | 2025-08-18 13:15:00 | 1633.90 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-08-25 10:15:00 | 1663.70 | 2025-08-26 09:15:00 | 1625.20 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2025-08-25 11:45:00 | 1660.50 | 2025-08-26 09:15:00 | 1625.20 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-08-25 12:45:00 | 1656.30 | 2025-08-26 09:15:00 | 1625.20 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest1 | 2025-08-29 09:15:00 | 1566.20 | 2025-09-01 14:15:00 | 1576.60 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest1 | 2025-08-29 11:30:00 | 1576.10 | 2025-09-01 14:15:00 | 1576.60 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest1 | 2025-08-29 12:15:00 | 1572.40 | 2025-09-01 14:15:00 | 1576.60 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-09-02 14:00:00 | 1562.50 | 2025-09-03 14:15:00 | 1585.10 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-09-09 09:15:00 | 1532.00 | 2025-09-10 09:15:00 | 1548.60 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-09-12 09:15:00 | 1564.50 | 2025-09-12 11:15:00 | 1544.00 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-09-12 10:00:00 | 1566.90 | 2025-09-12 11:15:00 | 1544.00 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-09-30 09:15:00 | 1514.60 | 2025-10-01 13:15:00 | 1526.40 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-09-30 09:45:00 | 1518.80 | 2025-10-01 14:15:00 | 1543.40 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-10-01 10:00:00 | 1518.80 | 2025-10-01 14:15:00 | 1543.40 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-10-01 10:45:00 | 1518.60 | 2025-10-01 14:15:00 | 1543.40 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-10-01 12:45:00 | 1514.50 | 2025-10-01 14:15:00 | 1543.40 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-10-14 14:30:00 | 1606.00 | 2025-10-24 12:15:00 | 1764.84 | TARGET_HIT | 1.00 | 9.89% |
| BUY | retest2 | 2025-10-14 15:15:00 | 1604.40 | 2025-10-27 09:15:00 | 1766.60 | TARGET_HIT | 1.00 | 10.11% |
| SELL | retest2 | 2025-11-21 11:00:00 | 1683.00 | 2025-11-28 13:15:00 | 1675.80 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2025-11-21 14:15:00 | 1691.00 | 2025-11-28 13:15:00 | 1675.80 | STOP_HIT | 1.00 | 0.90% |
| SELL | retest2 | 2025-12-09 14:45:00 | 1629.10 | 2025-12-11 09:15:00 | 1650.50 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-12-10 14:00:00 | 1618.40 | 2025-12-11 09:15:00 | 1650.50 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-12-11 09:30:00 | 1627.40 | 2025-12-11 10:15:00 | 1651.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-12-12 13:15:00 | 1654.60 | 2025-12-15 09:15:00 | 1641.80 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-12-24 13:00:00 | 1595.80 | 2025-12-24 15:15:00 | 1619.90 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-12-31 13:30:00 | 1587.10 | 2025-12-31 15:15:00 | 1597.50 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2026-01-16 11:45:00 | 1533.00 | 2026-01-16 12:15:00 | 1533.40 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2026-01-16 12:15:00 | 1533.30 | 2026-01-16 12:15:00 | 1533.40 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2026-01-22 10:15:00 | 1430.10 | 2026-01-27 09:15:00 | 1358.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 10:15:00 | 1430.10 | 2026-01-27 14:15:00 | 1389.60 | STOP_HIT | 0.50 | 2.83% |
| BUY | retest2 | 2026-01-29 13:15:00 | 1426.50 | 2026-02-03 09:15:00 | 1569.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-30 09:30:00 | 1426.50 | 2026-02-03 09:15:00 | 1569.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-01 12:30:00 | 1453.80 | 2026-02-03 09:15:00 | 1599.18 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-17 09:15:00 | 1516.50 | 2026-02-17 12:15:00 | 1542.40 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-02-17 15:15:00 | 1520.90 | 2026-02-24 11:15:00 | 1444.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 09:45:00 | 1514.00 | 2026-02-24 12:15:00 | 1438.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 09:30:00 | 1514.50 | 2026-02-24 12:15:00 | 1438.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 12:00:00 | 1482.40 | 2026-02-24 14:15:00 | 1408.47 | PARTIAL | 0.50 | 4.99% |
| SELL | retest2 | 2026-02-17 15:15:00 | 1520.90 | 2026-02-25 09:15:00 | 1453.40 | STOP_HIT | 0.50 | 4.44% |
| SELL | retest2 | 2026-02-18 09:45:00 | 1514.00 | 2026-02-25 09:15:00 | 1453.40 | STOP_HIT | 0.50 | 4.00% |
| SELL | retest2 | 2026-02-19 09:30:00 | 1514.50 | 2026-02-25 09:15:00 | 1453.40 | STOP_HIT | 0.50 | 4.03% |
| SELL | retest2 | 2026-02-23 12:00:00 | 1482.40 | 2026-02-25 09:15:00 | 1453.40 | STOP_HIT | 0.50 | 1.96% |
| SELL | retest2 | 2026-02-23 12:45:00 | 1482.60 | 2026-02-27 09:15:00 | 1408.28 | PARTIAL | 0.50 | 5.01% |
| SELL | retest2 | 2026-02-23 13:30:00 | 1482.20 | 2026-02-27 09:15:00 | 1408.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1467.30 | 2026-02-27 11:15:00 | 1393.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 12:45:00 | 1482.60 | 2026-03-02 09:15:00 | 1334.16 | TARGET_HIT | 0.50 | 10.01% |
| SELL | retest2 | 2026-02-23 13:30:00 | 1482.20 | 2026-03-02 09:15:00 | 1333.98 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-25 11:15:00 | 1445.90 | 2026-03-02 09:15:00 | 1373.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1467.30 | 2026-03-04 10:15:00 | 1320.57 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-25 11:15:00 | 1445.90 | 2026-03-04 14:15:00 | 1352.90 | STOP_HIT | 0.50 | 6.43% |
| BUY | retest2 | 2026-03-20 09:15:00 | 1290.40 | 2026-03-20 13:15:00 | 1261.60 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2026-03-25 11:15:00 | 1225.90 | 2026-03-25 14:15:00 | 1228.50 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2026-03-25 14:00:00 | 1227.20 | 2026-03-25 14:15:00 | 1228.50 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2026-04-01 10:45:00 | 1140.00 | 2026-04-06 12:15:00 | 1176.50 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2026-04-01 14:45:00 | 1141.80 | 2026-04-06 12:15:00 | 1176.50 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2026-04-02 09:15:00 | 1101.50 | 2026-04-06 12:15:00 | 1176.50 | STOP_HIT | 1.00 | -6.81% |
| SELL | retest2 | 2026-04-06 09:15:00 | 1139.40 | 2026-04-06 12:15:00 | 1176.50 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest1 | 2026-04-10 09:15:00 | 1329.50 | 2026-04-13 09:15:00 | 1291.50 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest1 | 2026-04-10 15:00:00 | 1322.60 | 2026-04-13 09:15:00 | 1291.50 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2026-04-13 10:15:00 | 1303.50 | 2026-04-23 15:15:00 | 1380.00 | STOP_HIT | 1.00 | 5.87% |
| BUY | retest2 | 2026-04-13 11:15:00 | 1303.00 | 2026-04-23 15:15:00 | 1380.00 | STOP_HIT | 1.00 | 5.91% |
