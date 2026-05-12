# Olectra Greentech Ltd. (OLECTRA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1345.00
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
| ALERT2_SKIP | 5 |
| ALERT3 | 50 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 43 |
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 40 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 48 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 37
- **Target hits / Stop hits / Partials:** 3 / 40 / 5
- **Avg / median % per leg:** -1.49% / -2.75%
- **Sum % (uncompounded):** -71.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 0 | 0.0% | 0 | 15 | 0 | -3.48% | -52.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 0 | 0.0% | 0 | 15 | 0 | -3.48% | -52.2% |
| SELL (all) | 33 | 11 | 33.3% | 3 | 25 | 5 | -0.58% | -19.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 33 | 11 | 33.3% | 3 | 25 | 5 | -0.58% | -19.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 48 | 11 | 22.9% | 3 | 40 | 5 | -1.49% | -71.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-08 15:15:00 | 1645.00 | 1769.87 | 1770.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 10:15:00 | 1633.05 | 1767.26 | 1768.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 09:15:00 | 1769.70 | 1724.62 | 1743.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 09:15:00 | 1769.70 | 1724.62 | 1743.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 1769.70 | 1724.62 | 1743.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 10:00:00 | 1769.70 | 1724.62 | 1743.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 1770.50 | 1725.07 | 1743.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 09:30:00 | 1746.40 | 1752.33 | 1754.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 1802.75 | 1752.24 | 1754.81 | SL hit (close>static) qty=1.00 sl=1794.90 alert=retest2 |

### Cycle 2 — BUY (started 2024-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 10:15:00 | 1831.00 | 1752.34 | 1752.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 11:15:00 | 1873.50 | 1753.55 | 1752.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 14:15:00 | 1802.10 | 1803.64 | 1783.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-12 15:00:00 | 1802.10 | 1803.64 | 1783.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 1787.10 | 1805.08 | 1785.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 1775.00 | 1805.08 | 1785.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 1769.65 | 1804.73 | 1785.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:00:00 | 1769.65 | 1804.73 | 1785.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 1776.75 | 1804.45 | 1785.26 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 12:15:00 | 1730.10 | 1770.99 | 1771.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 13:15:00 | 1723.00 | 1770.52 | 1770.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-13 09:15:00 | 1723.90 | 1692.87 | 1727.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 09:15:00 | 1723.90 | 1692.87 | 1727.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 1723.90 | 1692.87 | 1727.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 09:45:00 | 1722.10 | 1692.87 | 1727.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 1715.65 | 1693.10 | 1726.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:30:00 | 1728.05 | 1693.10 | 1726.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 1710.50 | 1693.27 | 1726.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 11:45:00 | 1717.60 | 1693.27 | 1726.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 1617.35 | 1602.48 | 1651.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:15:00 | 1598.55 | 1650.00 | 1659.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 09:15:00 | 1703.00 | 1645.72 | 1656.47 | SL hit (close>static) qty=1.00 sl=1672.95 alert=retest2 |

### Cycle 4 — BUY (started 2024-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 14:15:00 | 1724.05 | 1665.82 | 1665.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 15:15:00 | 1730.05 | 1666.46 | 1665.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-18 09:15:00 | 1662.45 | 1670.52 | 1668.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 09:15:00 | 1662.45 | 1670.52 | 1668.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 1662.45 | 1670.52 | 1668.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 11:45:00 | 1688.00 | 1670.58 | 1668.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 12:15:00 | 1680.05 | 1670.58 | 1668.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 13:15:00 | 1697.20 | 1670.62 | 1668.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-22 10:30:00 | 1680.95 | 1674.25 | 1670.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 12:15:00 | 1667.45 | 1674.19 | 1670.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 12:45:00 | 1665.70 | 1674.19 | 1670.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 13:15:00 | 1675.55 | 1674.20 | 1670.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 14:15:00 | 1656.50 | 1674.20 | 1670.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 14:15:00 | 1613.90 | 1673.60 | 1669.94 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-22 14:15:00 | 1613.90 | 1673.60 | 1669.94 | SL hit (close<static) qty=1.00 sl=1642.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-10-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 14:15:00 | 1594.90 | 1666.69 | 1666.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 1582.50 | 1646.02 | 1654.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 13:15:00 | 1559.95 | 1551.00 | 1595.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-27 13:30:00 | 1567.95 | 1551.00 | 1595.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 1618.00 | 1552.08 | 1595.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:00:00 | 1618.00 | 1552.08 | 1595.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 1600.00 | 1552.55 | 1595.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:30:00 | 1606.65 | 1552.55 | 1595.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 1600.65 | 1553.03 | 1595.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 11:45:00 | 1601.20 | 1553.03 | 1595.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 1578.55 | 1553.29 | 1595.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 13:30:00 | 1574.90 | 1553.52 | 1595.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 14:15:00 | 1575.35 | 1553.52 | 1595.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 11:45:00 | 1567.05 | 1554.70 | 1595.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 10:30:00 | 1575.25 | 1556.61 | 1595.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 1601.10 | 1559.40 | 1594.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:30:00 | 1609.45 | 1559.40 | 1594.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 1595.20 | 1559.75 | 1594.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 11:45:00 | 1584.40 | 1560.05 | 1594.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 14:15:00 | 1588.15 | 1560.67 | 1593.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 11:15:00 | 1583.00 | 1562.26 | 1593.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 13:15:00 | 1618.25 | 1564.84 | 1591.86 | SL hit (close>static) qty=1.00 sl=1602.55 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 1336.40 | 1222.10 | 1221.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 15:15:00 | 1350.00 | 1225.65 | 1223.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 09:15:00 | 1213.00 | 1228.87 | 1225.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 1213.00 | 1228.87 | 1225.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 1213.00 | 1228.87 | 1225.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:00:00 | 1213.00 | 1228.87 | 1225.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 1200.60 | 1228.59 | 1225.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 09:15:00 | 1235.00 | 1226.84 | 1224.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 14:15:00 | 1217.40 | 1228.33 | 1225.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:30:00 | 1216.00 | 1228.00 | 1225.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 1172.30 | 1223.82 | 1223.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 09:15:00 | 1172.30 | 1223.82 | 1223.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 1158.00 | 1213.80 | 1218.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 09:15:00 | 1215.00 | 1196.08 | 1207.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 09:15:00 | 1215.00 | 1196.08 | 1207.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 1215.00 | 1196.08 | 1207.13 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 13:15:00 | 1278.30 | 1213.39 | 1213.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 14:15:00 | 1284.20 | 1217.35 | 1215.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 12:15:00 | 1573.20 | 1578.42 | 1495.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 13:00:00 | 1573.20 | 1578.42 | 1495.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1489.00 | 1561.47 | 1509.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:00:00 | 1489.00 | 1561.47 | 1509.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 1484.20 | 1560.70 | 1509.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:45:00 | 1484.00 | 1560.70 | 1509.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 1516.50 | 1508.40 | 1495.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 11:30:00 | 1518.40 | 1508.48 | 1495.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 12:15:00 | 1526.00 | 1508.48 | 1495.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 14:45:00 | 1517.80 | 1508.79 | 1496.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 15:00:00 | 1519.10 | 1508.61 | 1496.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1492.20 | 1508.42 | 1496.66 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 1492.20 | 1508.42 | 1496.66 | SL hit (close<static) qty=1.00 sl=1494.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 09:15:00 | 1412.40 | 1486.59 | 1486.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 1391.10 | 1481.20 | 1484.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 10:15:00 | 1261.00 | 1257.13 | 1329.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-24 10:30:00 | 1267.70 | 1257.13 | 1329.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 14:15:00 | 1060.25 | 974.94 | 1048.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 15:00:00 | 1060.25 | 974.94 | 1048.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 1033.15 | 975.52 | 1047.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 1027.50 | 975.52 | 1047.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 09:30:00 | 1018.15 | 978.30 | 1046.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 10:15:00 | 1075.50 | 979.27 | 1046.71 | SL hit (close>static) qty=1.00 sl=1069.45 alert=retest2 |

### Cycle 10 — BUY (started 2026-04-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 12:15:00 | 1227.05 | 1071.79 | 1071.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 09:15:00 | 1232.60 | 1095.51 | 1084.09 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-31 09:30:00 | 1746.40 | 2024-06-03 09:15:00 | 1802.75 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2024-06-04 09:15:00 | 1672.60 | 2024-06-04 11:15:00 | 1505.34 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-10 09:30:00 | 1752.30 | 2024-06-11 14:15:00 | 1784.95 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-06-10 10:15:00 | 1754.55 | 2024-06-18 15:15:00 | 1800.00 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2024-06-11 09:15:00 | 1742.00 | 2024-06-18 15:15:00 | 1800.00 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2024-06-24 14:15:00 | 1744.65 | 2024-06-25 09:15:00 | 1818.10 | STOP_HIT | 1.00 | -4.21% |
| SELL | retest2 | 2024-10-07 10:15:00 | 1598.55 | 2024-10-09 09:15:00 | 1703.00 | STOP_HIT | 1.00 | -6.53% |
| BUY | retest2 | 2024-10-18 11:45:00 | 1688.00 | 2024-10-22 14:15:00 | 1613.90 | STOP_HIT | 1.00 | -4.39% |
| BUY | retest2 | 2024-10-18 12:15:00 | 1680.05 | 2024-10-22 14:15:00 | 1613.90 | STOP_HIT | 1.00 | -3.94% |
| BUY | retest2 | 2024-10-18 13:15:00 | 1697.20 | 2024-10-22 14:15:00 | 1613.90 | STOP_HIT | 1.00 | -4.91% |
| BUY | retest2 | 2024-10-22 10:30:00 | 1680.95 | 2024-10-22 14:15:00 | 1613.90 | STOP_HIT | 1.00 | -3.99% |
| BUY | retest2 | 2024-10-23 09:15:00 | 1670.00 | 2024-10-25 12:15:00 | 1598.00 | STOP_HIT | 1.00 | -4.31% |
| BUY | retest2 | 2024-10-23 13:30:00 | 1653.00 | 2024-10-25 12:15:00 | 1598.00 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2024-10-24 09:45:00 | 1667.05 | 2024-10-25 12:15:00 | 1598.00 | STOP_HIT | 1.00 | -4.14% |
| BUY | retest2 | 2024-10-24 15:15:00 | 1655.00 | 2024-10-25 12:15:00 | 1598.00 | STOP_HIT | 1.00 | -3.44% |
| SELL | retest2 | 2024-11-28 13:30:00 | 1574.90 | 2024-12-10 13:15:00 | 1618.25 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2024-11-28 14:15:00 | 1575.35 | 2024-12-10 13:15:00 | 1618.25 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2024-11-29 11:45:00 | 1567.05 | 2024-12-10 13:15:00 | 1618.25 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2024-12-02 10:30:00 | 1575.25 | 2024-12-10 13:15:00 | 1618.25 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2024-12-04 11:45:00 | 1584.40 | 2024-12-10 13:15:00 | 1618.25 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-12-04 14:15:00 | 1588.15 | 2024-12-10 13:15:00 | 1618.25 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-12-06 11:15:00 | 1583.00 | 2024-12-10 13:15:00 | 1618.25 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2024-12-12 14:45:00 | 1589.80 | 2024-12-18 13:15:00 | 1510.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 14:45:00 | 1589.80 | 2024-12-27 12:15:00 | 1430.82 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-01 11:30:00 | 1474.55 | 2025-02-03 09:15:00 | 1400.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-01 11:30:00 | 1474.55 | 2025-02-10 09:15:00 | 1327.10 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-05-30 09:15:00 | 1235.00 | 2025-06-16 09:15:00 | 1172.30 | STOP_HIT | 1.00 | -5.08% |
| BUY | retest2 | 2025-06-11 14:15:00 | 1217.40 | 2025-06-16 09:15:00 | 1172.30 | STOP_HIT | 1.00 | -3.70% |
| BUY | retest2 | 2025-06-12 09:30:00 | 1216.00 | 2025-06-16 09:15:00 | 1172.30 | STOP_HIT | 1.00 | -3.59% |
| BUY | retest2 | 2025-11-06 11:30:00 | 1518.40 | 2025-11-10 10:15:00 | 1492.20 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-11-06 12:15:00 | 1526.00 | 2025-11-10 10:15:00 | 1492.20 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-11-06 14:45:00 | 1517.80 | 2025-11-10 10:15:00 | 1492.20 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-11-07 15:00:00 | 1519.10 | 2025-11-10 10:15:00 | 1492.20 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2026-03-19 09:15:00 | 1027.50 | 2026-03-20 10:15:00 | 1075.50 | STOP_HIT | 1.00 | -4.67% |
| SELL | retest2 | 2026-03-20 09:30:00 | 1018.15 | 2026-03-20 10:15:00 | 1075.50 | STOP_HIT | 1.00 | -5.63% |
| SELL | retest2 | 2026-03-23 12:15:00 | 1028.05 | 2026-03-24 09:15:00 | 1071.00 | STOP_HIT | 1.00 | -4.18% |
| SELL | retest2 | 2026-03-23 12:45:00 | 1021.20 | 2026-03-24 09:15:00 | 1071.00 | STOP_HIT | 1.00 | -4.88% |
| SELL | retest2 | 2026-03-27 09:45:00 | 1038.25 | 2026-03-30 09:15:00 | 986.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-27 09:45:00 | 1038.25 | 2026-04-01 09:15:00 | 1033.00 | STOP_HIT | 0.50 | 0.51% |
| SELL | retest2 | 2026-04-01 09:45:00 | 1036.35 | 2026-04-02 09:15:00 | 987.52 | PARTIAL | 0.50 | 4.71% |
| SELL | retest2 | 2026-04-01 12:30:00 | 1039.50 | 2026-04-02 09:15:00 | 987.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 09:45:00 | 1036.35 | 2026-04-02 11:15:00 | 1003.50 | STOP_HIT | 0.50 | 3.17% |
| SELL | retest2 | 2026-04-01 12:30:00 | 1039.50 | 2026-04-02 11:15:00 | 1003.50 | STOP_HIT | 0.50 | 3.46% |
| SELL | retest2 | 2026-04-01 13:15:00 | 1039.50 | 2026-04-08 09:15:00 | 1063.25 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2026-04-01 14:45:00 | 1027.10 | 2026-04-08 09:15:00 | 1063.25 | STOP_HIT | 1.00 | -3.52% |
| SELL | retest2 | 2026-04-06 14:15:00 | 1026.95 | 2026-04-08 09:15:00 | 1063.25 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2026-04-07 09:30:00 | 1025.35 | 2026-04-08 09:15:00 | 1063.25 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2026-04-07 15:00:00 | 1026.75 | 2026-04-09 09:15:00 | 1119.55 | STOP_HIT | 1.00 | -9.04% |
