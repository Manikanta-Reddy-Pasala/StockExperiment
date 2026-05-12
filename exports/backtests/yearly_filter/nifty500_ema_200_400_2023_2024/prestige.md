# Prestige Estates Projects Ltd. (PRESTIGE)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1495.50
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
| ALERT2 | 9 |
| ALERT2_SKIP | 4 |
| ALERT3 | 79 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 38 |
| PARTIAL | 1 |
| TARGET_HIT | 8 |
| STOP_HIT | 32 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 41 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 32
- **Target hits / Stop hits / Partials:** 8 / 32 / 1
- **Avg / median % per leg:** 0.32% / -1.51%
- **Sum % (uncompounded):** 13.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 38 | 7 | 18.4% | 7 | 31 | 0 | 0.11% | 4.4% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.50% | -7.0% |
| BUY @ 3rd Alert (retest2) | 36 | 7 | 19.4% | 7 | 29 | 0 | 0.32% | 11.4% |
| SELL (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 2.96% | 8.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 3 | 2 | 66.7% | 1 | 1 | 1 | 2.96% | 8.9% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.50% | -7.0% |
| retest2 (combined) | 39 | 9 | 23.1% | 8 | 30 | 1 | 0.52% | 20.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 11:15:00 | 1009.00 | 1141.44 | 1141.45 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 13:15:00 | 1250.30 | 1140.32 | 1140.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 14:15:00 | 1270.55 | 1141.62 | 1140.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 14:15:00 | 1188.85 | 1200.92 | 1175.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-12 15:00:00 | 1188.85 | 1200.92 | 1175.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 1219.00 | 1201.06 | 1175.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-15 10:15:00 | 1232.85 | 1201.06 | 1175.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-16 09:15:00 | 1236.00 | 1202.01 | 1176.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-18 11:15:00 | 1235.55 | 1202.65 | 1178.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 09:30:00 | 1232.45 | 1204.49 | 1179.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-26 14:15:00 | 1356.13 | 1231.69 | 1198.96 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 14:15:00 | 1655.35 | 1778.99 | 1779.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-28 15:15:00 | 1650.15 | 1777.71 | 1778.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 1685.10 | 1658.55 | 1702.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 09:15:00 | 1685.10 | 1658.55 | 1702.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 1685.10 | 1658.55 | 1702.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 10:00:00 | 1685.10 | 1658.55 | 1702.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 13:15:00 | 1706.85 | 1659.98 | 1702.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 14:00:00 | 1706.85 | 1659.98 | 1702.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 1728.30 | 1660.66 | 1702.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 15:00:00 | 1728.30 | 1660.66 | 1702.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 1734.80 | 1661.40 | 1702.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-26 09:15:00 | 1743.65 | 1661.40 | 1702.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 1705.60 | 1664.96 | 1703.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 09:15:00 | 1631.00 | 1664.96 | 1703.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 14:15:00 | 1730.75 | 1662.72 | 1697.21 | SL hit (close>static) qty=1.00 sl=1708.70 alert=retest2 |

### Cycle 4 — BUY (started 2024-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 15:15:00 | 1855.00 | 1717.90 | 1717.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-18 12:15:00 | 1884.85 | 1723.41 | 1720.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-23 11:15:00 | 1728.10 | 1742.37 | 1730.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-23 11:15:00 | 1728.10 | 1742.37 | 1730.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 1728.10 | 1742.37 | 1730.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 12:00:00 | 1728.10 | 1742.37 | 1730.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 12:15:00 | 1732.05 | 1742.27 | 1730.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 13:30:00 | 1740.25 | 1742.33 | 1731.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-26 09:15:00 | 1717.05 | 1743.32 | 1732.11 | SL hit (close<static) qty=1.00 sl=1725.65 alert=retest2 |

### Cycle 5 — SELL (started 2025-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 13:15:00 | 1658.90 | 1724.09 | 1724.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 15:15:00 | 1652.10 | 1722.72 | 1723.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 10:15:00 | 1212.45 | 1210.49 | 1316.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-20 11:00:00 | 1212.45 | 1210.49 | 1316.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 11:15:00 | 1245.40 | 1178.22 | 1246.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 12:00:00 | 1245.40 | 1178.22 | 1246.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 12:15:00 | 1249.00 | 1178.93 | 1246.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 12:30:00 | 1250.70 | 1178.93 | 1246.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 13:15:00 | 1250.60 | 1179.64 | 1246.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 13:45:00 | 1249.20 | 1179.64 | 1246.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 14:15:00 | 1269.60 | 1180.54 | 1247.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 15:00:00 | 1269.60 | 1180.54 | 1247.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 1257.40 | 1206.94 | 1253.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-25 12:00:00 | 1257.40 | 1206.94 | 1253.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 12:15:00 | 1265.90 | 1207.52 | 1253.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-25 12:45:00 | 1265.00 | 1207.52 | 1253.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 1296.60 | 1210.97 | 1253.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:30:00 | 1300.00 | 1210.97 | 1253.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 10:15:00 | 1274.70 | 1270.04 | 1277.43 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 11:15:00 | 1389.40 | 1284.38 | 1284.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 14:15:00 | 1394.40 | 1287.55 | 1285.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 10:15:00 | 1610.50 | 1612.98 | 1519.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 11:00:00 | 1610.50 | 1612.98 | 1519.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 1642.00 | 1683.90 | 1606.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:30:00 | 1623.70 | 1683.90 | 1606.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 1606.10 | 1681.98 | 1606.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 13:00:00 | 1606.10 | 1681.98 | 1606.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 13:15:00 | 1598.30 | 1681.15 | 1606.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 13:45:00 | 1597.30 | 1681.15 | 1606.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 14:15:00 | 1613.80 | 1680.48 | 1606.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 09:15:00 | 1622.90 | 1679.78 | 1606.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:45:00 | 1624.30 | 1669.66 | 1607.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 14:30:00 | 1631.50 | 1669.23 | 1608.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 12:15:00 | 1626.10 | 1663.97 | 1608.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1608.30 | 1662.05 | 1608.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:00:00 | 1608.30 | 1662.05 | 1608.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 1618.40 | 1661.62 | 1609.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:30:00 | 1622.00 | 1661.62 | 1609.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 1609.00 | 1660.68 | 1609.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:00:00 | 1609.00 | 1660.68 | 1609.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 1602.20 | 1660.10 | 1609.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 14:00:00 | 1602.20 | 1660.10 | 1609.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 1609.80 | 1659.60 | 1609.06 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 1572.80 | 1658.22 | 1608.87 | SL hit (close<static) qty=1.00 sl=1591.20 alert=retest2 |

### Cycle 7 — SELL (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 12:15:00 | 1555.00 | 1599.25 | 1599.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 13:15:00 | 1551.60 | 1596.32 | 1597.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 12:15:00 | 1593.60 | 1591.55 | 1595.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 12:15:00 | 1593.60 | 1591.55 | 1595.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 1593.60 | 1591.55 | 1595.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:00:00 | 1593.60 | 1591.55 | 1595.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 1592.10 | 1591.55 | 1595.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:30:00 | 1592.30 | 1591.55 | 1595.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 1592.30 | 1591.56 | 1595.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 15:00:00 | 1592.30 | 1591.56 | 1595.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 1603.20 | 1591.68 | 1595.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 1596.60 | 1591.68 | 1595.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1611.70 | 1591.87 | 1595.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:00:00 | 1611.70 | 1591.87 | 1595.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1608.90 | 1592.04 | 1595.52 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 11:15:00 | 1638.80 | 1598.77 | 1598.74 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 14:15:00 | 1522.90 | 1598.72 | 1598.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 12:15:00 | 1521.10 | 1595.25 | 1597.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 1569.40 | 1566.11 | 1580.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 1569.40 | 1566.11 | 1580.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1569.40 | 1566.11 | 1580.18 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 1710.00 | 1591.53 | 1591.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 11:15:00 | 1720.20 | 1594.03 | 1592.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 15:15:00 | 1695.80 | 1696.71 | 1658.41 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 09:15:00 | 1756.10 | 1696.71 | 1658.41 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 10:15:00 | 1707.40 | 1708.78 | 1671.39 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 1676.60 | 1707.46 | 1673.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:45:00 | 1673.80 | 1707.46 | 1673.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 1670.80 | 1707.09 | 1673.24 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 1670.80 | 1707.09 | 1673.24 | SL hit (close<ema400) qty=1.00 sl=1673.24 alert=retest1 |

### Cycle 11 — SELL (started 2025-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 14:15:00 | 1598.80 | 1659.09 | 1659.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 15:15:00 | 1588.90 | 1642.07 | 1649.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 10:15:00 | 1638.00 | 1630.95 | 1642.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 10:45:00 | 1629.00 | 1630.95 | 1642.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 1643.30 | 1631.08 | 1642.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:00:00 | 1643.30 | 1631.08 | 1642.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 1641.80 | 1631.18 | 1642.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:30:00 | 1646.30 | 1631.18 | 1642.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 1643.80 | 1631.31 | 1642.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 13:45:00 | 1641.60 | 1631.31 | 1642.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 1665.10 | 1631.65 | 1642.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 1665.10 | 1631.65 | 1642.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 1635.80 | 1634.40 | 1643.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 11:15:00 | 1632.10 | 1634.40 | 1643.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 1550.49 | 1630.53 | 1641.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 09:15:00 | 1468.89 | 1590.22 | 1617.01 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-15 10:15:00 | 1232.85 | 2024-04-26 14:15:00 | 1356.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-16 09:15:00 | 1236.00 | 2024-04-26 14:15:00 | 1359.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-18 11:15:00 | 1235.55 | 2024-04-26 14:15:00 | 1359.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-19 09:30:00 | 1232.45 | 2024-04-26 14:15:00 | 1355.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-12 11:15:00 | 1732.25 | 2024-08-23 11:15:00 | 1712.75 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-08-22 09:30:00 | 1734.15 | 2024-08-23 11:15:00 | 1712.75 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-08-22 10:30:00 | 1735.40 | 2024-08-23 11:15:00 | 1712.75 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-08-23 11:15:00 | 1730.80 | 2024-08-23 11:15:00 | 1712.75 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-08-26 13:45:00 | 1731.10 | 2024-08-27 12:15:00 | 1699.50 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2024-08-27 13:30:00 | 1724.75 | 2024-09-12 14:15:00 | 1897.23 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-28 09:15:00 | 1727.65 | 2024-09-12 14:15:00 | 1900.42 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-04 09:30:00 | 1716.70 | 2024-10-14 09:15:00 | 1888.37 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-08 14:15:00 | 1805.00 | 2024-10-18 09:15:00 | 1743.90 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2024-10-08 15:00:00 | 1817.85 | 2024-10-18 09:15:00 | 1743.90 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest2 | 2024-10-17 15:15:00 | 1806.65 | 2024-10-18 09:15:00 | 1743.90 | STOP_HIT | 1.00 | -3.47% |
| SELL | retest2 | 2024-11-27 09:15:00 | 1631.00 | 2024-12-02 14:15:00 | 1730.75 | STOP_HIT | 1.00 | -6.12% |
| BUY | retest2 | 2024-12-23 13:30:00 | 1740.25 | 2024-12-26 09:15:00 | 1717.05 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-12-26 14:45:00 | 1741.85 | 2024-12-30 13:15:00 | 1719.45 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-12-27 13:00:00 | 1745.90 | 2024-12-30 13:15:00 | 1719.45 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-07-29 09:15:00 | 1622.90 | 2025-08-06 09:15:00 | 1572.80 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2025-07-31 13:45:00 | 1624.30 | 2025-08-06 09:15:00 | 1572.80 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2025-07-31 14:30:00 | 1631.50 | 2025-08-06 09:15:00 | 1572.80 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2025-08-04 12:15:00 | 1626.10 | 2025-08-06 09:15:00 | 1572.80 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2025-08-06 13:00:00 | 1616.70 | 2025-08-07 09:15:00 | 1594.30 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-08-07 09:30:00 | 1620.00 | 2025-08-28 09:15:00 | 1585.00 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-08-07 12:00:00 | 1616.90 | 2025-08-28 09:15:00 | 1585.00 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-08-07 14:00:00 | 1618.40 | 2025-08-28 09:15:00 | 1585.00 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-08-11 09:15:00 | 1612.40 | 2025-08-28 09:15:00 | 1585.00 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-08-11 10:15:00 | 1617.00 | 2025-08-28 09:15:00 | 1585.00 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-08-14 10:00:00 | 1611.90 | 2025-08-28 09:15:00 | 1585.00 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-08-18 09:15:00 | 1627.40 | 2025-08-28 09:15:00 | 1585.00 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest1 | 2025-11-13 09:15:00 | 1756.10 | 2025-11-24 10:15:00 | 1670.80 | STOP_HIT | 1.00 | -4.86% |
| BUY | retest1 | 2025-11-20 10:15:00 | 1707.40 | 2025-11-24 10:15:00 | 1670.80 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-11-28 11:30:00 | 1672.50 | 2025-12-01 11:15:00 | 1651.50 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-11-28 13:30:00 | 1672.60 | 2025-12-01 11:15:00 | 1651.50 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-12-05 09:15:00 | 1673.00 | 2025-12-08 09:15:00 | 1640.50 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-12-12 09:15:00 | 1679.30 | 2025-12-12 10:15:00 | 1648.80 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-12-12 13:15:00 | 1654.60 | 2025-12-15 09:15:00 | 1641.80 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-12-15 14:30:00 | 1651.90 | 2025-12-16 09:15:00 | 1630.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2026-01-07 11:15:00 | 1632.10 | 2026-01-09 09:15:00 | 1550.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 11:15:00 | 1632.10 | 2026-01-20 09:15:00 | 1468.89 | TARGET_HIT | 0.50 | 10.00% |
