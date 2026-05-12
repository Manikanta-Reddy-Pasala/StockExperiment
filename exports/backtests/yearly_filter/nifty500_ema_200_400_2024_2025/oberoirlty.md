# Oberoi Realty Ltd. (OBEROIRLTY)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1710.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 1 |
| ALERT3 | 28 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 17 |
| PARTIAL | 1 |
| TARGET_HIT | 3 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 16
- **Target hits / Stop hits / Partials:** 0 / 17 / 1
- **Avg / median % per leg:** -1.35% / -1.53%
- **Sum % (uncompounded):** -24.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.71% | -13.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.71% | -13.6% |
| SELL (all) | 13 | 2 | 15.4% | 0 | 12 | 1 | -0.83% | -10.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 13 | 2 | 15.4% | 0 | 12 | 1 | -0.83% | -10.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 18 | 2 | 11.1% | 0 | 17 | 1 | -1.35% | -24.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 12:15:00 | 1789.70 | 2061.29 | 2061.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 1766.75 | 2058.36 | 2060.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 11:15:00 | 1624.10 | 1619.73 | 1728.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-20 11:30:00 | 1623.65 | 1619.73 | 1728.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 1699.00 | 1602.84 | 1665.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-22 10:00:00 | 1699.00 | 1602.84 | 1665.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 10:15:00 | 1700.30 | 1603.81 | 1666.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-22 10:30:00 | 1700.00 | 1603.81 | 1666.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 15:15:00 | 1675.00 | 1619.09 | 1668.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 09:15:00 | 1647.50 | 1619.09 | 1668.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 14:15:00 | 1565.12 | 1618.93 | 1658.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-12 14:15:00 | 1615.10 | 1606.42 | 1646.19 | SL hit (close>ema200) qty=0.50 sl=1606.42 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 13:15:00 | 1746.30 | 1667.39 | 1667.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 14:15:00 | 1757.80 | 1672.81 | 1669.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 1865.00 | 1865.82 | 1803.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 11:45:00 | 1862.30 | 1865.82 | 1803.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1805.00 | 1857.28 | 1810.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:00:00 | 1805.00 | 1857.28 | 1810.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 1813.40 | 1856.84 | 1810.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:30:00 | 1799.30 | 1856.84 | 1810.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 12:15:00 | 1807.30 | 1856.35 | 1810.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 13:00:00 | 1807.30 | 1856.35 | 1810.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 1804.10 | 1855.83 | 1810.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:00:00 | 1804.10 | 1855.83 | 1810.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 1801.50 | 1855.29 | 1810.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:30:00 | 1803.60 | 1855.29 | 1810.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1812.40 | 1852.43 | 1811.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:45:00 | 1807.60 | 1852.43 | 1811.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 1808.30 | 1851.99 | 1811.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:45:00 | 1811.50 | 1851.99 | 1811.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 1814.00 | 1851.62 | 1811.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:30:00 | 1804.20 | 1851.62 | 1811.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 1813.00 | 1850.34 | 1811.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 1835.30 | 1850.34 | 1811.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 10:45:00 | 1826.80 | 1847.00 | 1813.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 1760.90 | 1845.14 | 1813.80 | SL hit (close<static) qty=1.00 sl=1810.50 alert=retest2 |

### Cycle 3 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 1618.30 | 1789.48 | 1789.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 1605.60 | 1776.72 | 1783.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 10:15:00 | 1669.80 | 1668.36 | 1706.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-02 11:00:00 | 1669.80 | 1668.36 | 1706.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 1675.40 | 1647.75 | 1680.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:30:00 | 1679.10 | 1647.75 | 1680.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 1668.40 | 1647.96 | 1680.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 12:15:00 | 1664.20 | 1647.96 | 1680.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 13:45:00 | 1662.80 | 1648.22 | 1680.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 09:45:00 | 1661.00 | 1648.57 | 1680.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:30:00 | 1660.60 | 1648.57 | 1680.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 1675.00 | 1648.95 | 1679.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:00:00 | 1675.00 | 1648.95 | 1679.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 1686.00 | 1650.06 | 1679.86 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-22 10:15:00 | 1686.00 | 1650.06 | 1679.86 | SL hit (close>static) qty=1.00 sl=1683.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 12:15:00 | 1780.50 | 1666.56 | 1666.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 13:15:00 | 1784.00 | 1667.73 | 1666.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1710.20 | 1719.00 | 1697.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 10:00:00 | 1710.20 | 1719.00 | 1697.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 1700.20 | 1718.56 | 1698.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:30:00 | 1716.70 | 1718.15 | 1698.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 10:45:00 | 1719.50 | 1717.83 | 1698.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 11:30:00 | 1715.40 | 1717.79 | 1698.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 1683.40 | 1717.12 | 1699.06 | SL hit (close<static) qty=1.00 sl=1690.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 15:15:00 | 1633.00 | 1685.17 | 1685.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 1623.20 | 1678.62 | 1681.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 12:15:00 | 1661.80 | 1660.44 | 1670.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 12:15:00 | 1661.80 | 1660.44 | 1670.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 1661.80 | 1660.44 | 1670.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:45:00 | 1665.50 | 1660.44 | 1670.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1669.10 | 1660.41 | 1670.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 1673.10 | 1660.41 | 1670.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 1675.80 | 1660.51 | 1670.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 1675.80 | 1660.51 | 1670.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 1675.30 | 1660.66 | 1670.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 1687.90 | 1660.66 | 1670.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 13:15:00 | 1666.70 | 1661.27 | 1670.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 13:30:00 | 1671.20 | 1661.27 | 1670.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 1670.40 | 1661.46 | 1670.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 12:15:00 | 1659.10 | 1661.50 | 1670.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 12:45:00 | 1658.50 | 1661.50 | 1670.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 15:15:00 | 1658.50 | 1661.56 | 1670.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 1677.10 | 1661.69 | 1670.09 | SL hit (close>static) qty=1.00 sl=1676.10 alert=retest2 |

### Cycle 6 — BUY (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 10:15:00 | 1727.30 | 1675.46 | 1675.40 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 1645.00 | 1676.17 | 1676.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 1568.80 | 1673.48 | 1674.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 15:15:00 | 1566.40 | 1565.32 | 1605.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-10 09:15:00 | 1562.70 | 1565.32 | 1605.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1519.10 | 1477.70 | 1515.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:00:00 | 1519.10 | 1477.70 | 1515.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 1507.70 | 1478.00 | 1515.42 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 10:15:00 | 1709.50 | 1544.06 | 1543.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 14:15:00 | 1716.00 | 1550.51 | 1546.76 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-04-25 09:15:00 | 1647.50 | 2025-05-06 14:15:00 | 1565.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-25 09:15:00 | 1647.50 | 2025-05-12 14:15:00 | 1615.10 | STOP_HIT | 0.50 | 1.97% |
| SELL | retest2 | 2025-05-16 14:15:00 | 1673.50 | 2025-05-19 09:15:00 | 1713.10 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2025-07-17 09:15:00 | 1835.30 | 2025-07-23 09:15:00 | 1760.90 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest2 | 2025-07-22 10:45:00 | 1826.80 | 2025-07-23 09:15:00 | 1760.90 | STOP_HIT | 1.00 | -3.61% |
| SELL | retest2 | 2025-09-18 12:15:00 | 1664.20 | 2025-09-22 10:15:00 | 1686.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-09-18 13:45:00 | 1662.80 | 2025-09-22 10:15:00 | 1686.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-09-19 09:45:00 | 1661.00 | 2025-09-22 10:15:00 | 1686.00 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-09-19 10:30:00 | 1660.60 | 2025-09-22 10:15:00 | 1686.00 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-11-19 12:30:00 | 1716.70 | 2025-11-21 09:15:00 | 1683.40 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-11-20 10:45:00 | 1719.50 | 2025-11-21 09:15:00 | 1683.40 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-11-20 11:30:00 | 1715.40 | 2025-11-21 09:15:00 | 1683.40 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-12-23 12:15:00 | 1659.10 | 2025-12-24 09:15:00 | 1677.10 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-12-23 12:45:00 | 1658.50 | 2025-12-24 09:15:00 | 1677.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-12-23 15:15:00 | 1658.50 | 2025-12-24 09:15:00 | 1677.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-12-30 09:15:00 | 1656.70 | 2026-01-01 13:15:00 | 1685.00 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-12-30 12:15:00 | 1648.80 | 2026-01-01 13:15:00 | 1685.00 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-12-31 09:30:00 | 1645.00 | 2026-01-01 13:15:00 | 1685.00 | STOP_HIT | 1.00 | -2.43% |
