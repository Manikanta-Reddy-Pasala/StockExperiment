# Tata Communications Ltd. (TATACOMM)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1582.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 7 |
| ALERT2_SKIP | 1 |
| ALERT3 | 69 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 57 |
| PARTIAL | 15 |
| TARGET_HIT | 19 |
| STOP_HIT | 38 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 72 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 34 / 38
- **Target hits / Stop hits / Partials:** 19 / 38 / 15
- **Avg / median % per leg:** 2.64% / -0.74%
- **Sum % (uncompounded):** 190.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 32 | 4 | 12.5% | 4 | 28 | 0 | -0.21% | -6.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 32 | 4 | 12.5% | 4 | 28 | 0 | -0.21% | -6.6% |
| SELL (all) | 40 | 30 | 75.0% | 15 | 10 | 15 | 4.92% | 197.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 40 | 30 | 75.0% | 15 | 10 | 15 | 4.92% | 197.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 72 | 34 | 47.2% | 19 | 38 | 15 | 2.64% | 190.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 10:15:00 | 1865.15 | 1824.28 | 1824.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 14:15:00 | 1890.10 | 1835.26 | 1830.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 10:15:00 | 1835.05 | 1846.28 | 1837.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 10:15:00 | 1835.05 | 1846.28 | 1837.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 1835.05 | 1846.28 | 1837.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:45:00 | 1828.80 | 1846.28 | 1837.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 11:15:00 | 1843.50 | 1846.25 | 1837.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 12:30:00 | 1855.25 | 1846.39 | 1837.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 15:15:00 | 1849.40 | 1846.48 | 1837.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 11:30:00 | 1856.50 | 1846.46 | 1837.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 11:30:00 | 1851.35 | 1847.07 | 1838.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 1849.10 | 1850.74 | 1841.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 1843.45 | 1850.74 | 1841.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 1835.70 | 1850.59 | 1841.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:00:00 | 1835.70 | 1850.59 | 1841.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 1844.20 | 1850.53 | 1841.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:30:00 | 1833.70 | 1850.53 | 1841.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 1851.70 | 1850.54 | 1841.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 15:00:00 | 1876.90 | 1850.75 | 1841.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 09:15:00 | 1817.05 | 1850.64 | 1841.43 | SL hit (close<static) qty=1.00 sl=1832.25 alert=retest2 |

### Cycle 2 — SELL (started 2024-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 13:15:00 | 1793.15 | 1945.16 | 1945.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 14:15:00 | 1784.50 | 1943.56 | 1944.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 14:15:00 | 1796.00 | 1794.26 | 1840.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-02 15:00:00 | 1796.00 | 1794.26 | 1840.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 13:15:00 | 1833.95 | 1795.76 | 1831.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 13:30:00 | 1834.10 | 1795.76 | 1831.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 1841.15 | 1796.21 | 1831.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 15:00:00 | 1841.15 | 1796.21 | 1831.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 15:15:00 | 1844.00 | 1796.69 | 1831.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 09:15:00 | 1846.35 | 1796.69 | 1831.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 1847.00 | 1797.79 | 1831.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 11:45:00 | 1840.75 | 1798.15 | 1831.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 09:15:00 | 1826.90 | 1799.99 | 1832.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 15:15:00 | 1840.00 | 1801.08 | 1831.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 10:15:00 | 1836.75 | 1805.31 | 1832.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 13:15:00 | 1748.71 | 1800.83 | 1827.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 13:15:00 | 1748.00 | 1800.83 | 1827.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 13:15:00 | 1744.91 | 1800.83 | 1827.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 14:15:00 | 1735.56 | 1799.89 | 1826.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 12:15:00 | 1656.67 | 1743.62 | 1780.91 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 13:15:00 | 1644.00 | 1572.55 | 1572.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 10:15:00 | 1664.30 | 1579.42 | 1575.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 09:15:00 | 1665.60 | 1672.23 | 1638.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 10:00:00 | 1665.60 | 1672.23 | 1638.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 1645.00 | 1671.66 | 1639.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 1645.00 | 1671.66 | 1639.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1643.50 | 1670.24 | 1639.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:45:00 | 1644.00 | 1670.24 | 1639.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 1632.10 | 1669.86 | 1639.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:00:00 | 1632.10 | 1669.86 | 1639.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 1651.20 | 1669.67 | 1639.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:30:00 | 1640.90 | 1669.67 | 1639.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1646.60 | 1669.10 | 1639.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:45:00 | 1643.70 | 1669.10 | 1639.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 1647.50 | 1668.25 | 1639.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 13:45:00 | 1644.10 | 1668.25 | 1639.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 1643.50 | 1668.00 | 1639.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:45:00 | 1640.90 | 1668.00 | 1639.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 1652.70 | 1667.63 | 1639.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:45:00 | 1654.60 | 1667.63 | 1639.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 1691.30 | 1725.63 | 1698.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:00:00 | 1691.30 | 1725.63 | 1698.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 1670.80 | 1725.09 | 1698.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 12:00:00 | 1670.80 | 1725.09 | 1698.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 1678.90 | 1715.19 | 1695.54 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 1591.30 | 1682.11 | 1682.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 1570.60 | 1670.26 | 1676.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 12:15:00 | 1616.50 | 1611.09 | 1637.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 13:00:00 | 1616.50 | 1611.09 | 1637.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1638.30 | 1611.36 | 1637.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:45:00 | 1644.90 | 1611.36 | 1637.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 1652.60 | 1611.77 | 1637.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:00:00 | 1652.60 | 1611.77 | 1637.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 1660.00 | 1612.25 | 1637.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:45:00 | 1661.20 | 1612.25 | 1637.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 1647.10 | 1637.50 | 1646.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:30:00 | 1649.20 | 1637.50 | 1646.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 1654.90 | 1637.67 | 1646.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:00:00 | 1654.90 | 1637.67 | 1646.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 1655.30 | 1637.85 | 1646.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:45:00 | 1656.80 | 1637.85 | 1646.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 1651.50 | 1639.59 | 1647.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 15:15:00 | 1645.80 | 1640.37 | 1647.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 09:15:00 | 1667.00 | 1634.49 | 1642.80 | SL hit (close>static) qty=1.00 sl=1662.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 1868.90 | 1651.23 | 1650.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 13:15:00 | 1888.10 | 1653.59 | 1651.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 09:15:00 | 1823.00 | 1824.58 | 1768.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 10:00:00 | 1823.00 | 1824.58 | 1768.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 1814.90 | 1847.30 | 1811.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:30:00 | 1816.60 | 1847.30 | 1811.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 1809.50 | 1846.93 | 1811.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:45:00 | 1817.40 | 1846.93 | 1811.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 15:15:00 | 1800.00 | 1846.46 | 1811.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:00:00 | 1790.10 | 1845.90 | 1811.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 1803.00 | 1845.47 | 1811.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 11:30:00 | 1808.00 | 1845.11 | 1811.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 13:30:00 | 1820.70 | 1844.36 | 1811.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 1804.50 | 1843.57 | 1811.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 10:00:00 | 1805.20 | 1843.19 | 1811.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 1793.80 | 1842.70 | 1811.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:45:00 | 1795.40 | 1842.70 | 1811.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 1779.00 | 1839.94 | 1810.68 | SL hit (close<static) qty=1.00 sl=1790.00 alert=retest2 |

### Cycle 6 — SELL (started 2026-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 13:15:00 | 1725.90 | 1801.08 | 1801.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 1705.90 | 1782.54 | 1791.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 14:15:00 | 1678.00 | 1647.55 | 1703.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-10 15:00:00 | 1678.00 | 1647.55 | 1703.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1670.30 | 1648.17 | 1702.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 10:30:00 | 1662.10 | 1648.38 | 1702.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:30:00 | 1663.60 | 1653.19 | 1701.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:00:00 | 1652.00 | 1653.93 | 1700.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 09:30:00 | 1662.30 | 1653.44 | 1692.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 1678.00 | 1654.69 | 1692.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 1645.50 | 1654.69 | 1692.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1578.99 | 1645.29 | 1682.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1580.42 | 1645.29 | 1682.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1569.40 | 1645.29 | 1682.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1579.18 | 1645.29 | 1682.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1563.22 | 1645.29 | 1682.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 1495.89 | 1638.81 | 1677.45 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-28 12:00:00 | 1792.80 | 2024-06-04 09:15:00 | 1703.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-29 11:15:00 | 1794.00 | 2024-06-04 09:15:00 | 1704.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-29 11:45:00 | 1793.55 | 2024-06-04 09:15:00 | 1703.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-29 13:00:00 | 1794.05 | 2024-06-04 09:15:00 | 1704.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 11:15:00 | 1792.55 | 2024-06-04 09:15:00 | 1702.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 12:00:00 | 1795.00 | 2024-06-04 09:15:00 | 1705.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-28 12:00:00 | 1792.80 | 2024-06-04 11:15:00 | 1613.52 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-29 11:15:00 | 1794.00 | 2024-06-04 11:15:00 | 1614.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-29 11:45:00 | 1793.55 | 2024-06-04 11:15:00 | 1614.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-29 13:00:00 | 1794.05 | 2024-06-04 11:15:00 | 1614.64 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-03 11:15:00 | 1792.55 | 2024-06-04 11:15:00 | 1613.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-03 12:00:00 | 1795.00 | 2024-06-04 11:15:00 | 1615.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-07 11:15:00 | 1795.10 | 2024-06-10 09:15:00 | 1853.55 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2024-07-09 12:30:00 | 1855.25 | 2024-07-19 09:15:00 | 1817.05 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2024-07-09 15:15:00 | 1849.40 | 2024-07-19 09:15:00 | 1817.05 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-07-10 11:30:00 | 1856.50 | 2024-07-19 09:15:00 | 1817.05 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-07-11 11:30:00 | 1851.35 | 2024-07-19 09:15:00 | 1817.05 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-07-18 15:00:00 | 1876.90 | 2024-07-19 09:15:00 | 1817.05 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2024-07-24 12:30:00 | 1854.00 | 2024-07-25 13:15:00 | 1822.70 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-07-26 09:15:00 | 1867.55 | 2024-08-13 14:15:00 | 1848.95 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-08-07 09:15:00 | 1853.60 | 2024-08-13 14:15:00 | 1848.95 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2024-08-07 14:00:00 | 1874.25 | 2024-08-13 14:15:00 | 1848.95 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-08-07 14:45:00 | 1871.55 | 2024-08-13 14:15:00 | 1848.95 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-08-08 10:15:00 | 1874.95 | 2024-08-13 14:15:00 | 1848.95 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-08-08 12:00:00 | 1874.10 | 2024-08-13 14:15:00 | 1848.95 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-08-09 09:15:00 | 1873.30 | 2024-08-14 10:15:00 | 1825.00 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2024-08-12 10:15:00 | 1860.95 | 2024-08-14 10:15:00 | 1825.00 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-08-16 09:15:00 | 1856.55 | 2024-09-13 09:15:00 | 2042.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-16 11:15:00 | 1860.45 | 2024-09-13 09:15:00 | 2046.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-16 15:00:00 | 1869.55 | 2024-09-13 09:15:00 | 2056.51 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-19 09:15:00 | 1875.15 | 2024-09-13 09:15:00 | 2062.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-18 10:00:00 | 1862.60 | 2024-10-18 11:15:00 | 1847.40 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-10-18 13:45:00 | 1864.95 | 2024-10-21 14:15:00 | 1851.15 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-12-12 11:45:00 | 1840.75 | 2024-12-20 13:15:00 | 1748.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-13 09:15:00 | 1826.90 | 2024-12-20 13:15:00 | 1748.00 | PARTIAL | 0.50 | 4.32% |
| SELL | retest2 | 2024-12-13 15:15:00 | 1840.00 | 2024-12-20 13:15:00 | 1744.91 | PARTIAL | 0.50 | 5.17% |
| SELL | retest2 | 2024-12-17 10:15:00 | 1836.75 | 2024-12-20 14:15:00 | 1735.56 | PARTIAL | 0.50 | 5.51% |
| SELL | retest2 | 2024-12-12 11:45:00 | 1840.75 | 2025-01-13 12:15:00 | 1656.67 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-13 09:15:00 | 1826.90 | 2025-01-13 12:15:00 | 1644.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-13 15:15:00 | 1840.00 | 2025-01-13 12:15:00 | 1656.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-17 10:15:00 | 1836.75 | 2025-01-13 12:15:00 | 1653.08 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-07 09:30:00 | 1516.55 | 2025-04-08 11:15:00 | 1567.95 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2025-04-07 11:30:00 | 1540.00 | 2025-04-08 11:15:00 | 1567.95 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-04-09 09:30:00 | 1540.00 | 2025-04-11 09:15:00 | 1570.60 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-04-09 10:00:00 | 1535.40 | 2025-04-11 09:15:00 | 1570.60 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2025-04-15 11:45:00 | 1543.10 | 2025-04-15 14:15:00 | 1591.40 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2025-04-21 09:45:00 | 1542.00 | 2025-04-22 10:15:00 | 1594.20 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2025-04-25 11:15:00 | 1539.30 | 2025-04-29 09:15:00 | 1586.20 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-05-06 14:30:00 | 1544.40 | 2025-05-14 13:15:00 | 1613.00 | STOP_HIT | 1.00 | -4.44% |
| SELL | retest2 | 2025-09-25 15:15:00 | 1645.80 | 2025-10-07 09:15:00 | 1667.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-12-09 11:30:00 | 1808.00 | 2025-12-11 09:15:00 | 1779.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-12-09 13:30:00 | 1820.70 | 2025-12-11 09:15:00 | 1779.00 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-12-10 09:15:00 | 1804.50 | 2025-12-11 09:15:00 | 1779.00 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-12-10 10:00:00 | 1805.20 | 2025-12-11 09:15:00 | 1779.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-12-16 11:30:00 | 1838.90 | 2025-12-18 09:15:00 | 1798.60 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-12-17 09:15:00 | 1838.00 | 2025-12-18 09:15:00 | 1798.60 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-12-19 14:30:00 | 1833.50 | 2025-12-22 13:15:00 | 1800.30 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-12-22 09:45:00 | 1833.00 | 2025-12-22 13:15:00 | 1800.30 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-12-30 10:15:00 | 1802.20 | 2026-01-08 11:15:00 | 1772.60 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2026-01-02 10:45:00 | 1804.80 | 2026-01-08 11:15:00 | 1772.60 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2026-01-02 13:00:00 | 1800.00 | 2026-01-08 11:15:00 | 1772.60 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2026-01-02 13:30:00 | 1802.80 | 2026-01-08 11:15:00 | 1772.60 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-02-11 10:30:00 | 1662.10 | 2026-03-02 09:15:00 | 1578.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 09:30:00 | 1663.60 | 2026-03-02 09:15:00 | 1580.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 15:00:00 | 1652.00 | 2026-03-02 09:15:00 | 1569.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 09:30:00 | 1662.30 | 2026-03-02 09:15:00 | 1579.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1645.50 | 2026-03-02 09:15:00 | 1563.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 10:30:00 | 1662.10 | 2026-03-04 09:15:00 | 1495.89 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-13 09:30:00 | 1663.60 | 2026-03-04 09:15:00 | 1497.24 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-13 15:00:00 | 1652.00 | 2026-03-04 09:15:00 | 1496.07 | TARGET_HIT | 0.50 | 9.44% |
| SELL | retest2 | 2026-02-23 09:30:00 | 1662.30 | 2026-03-04 11:15:00 | 1486.80 | TARGET_HIT | 0.50 | 10.56% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1645.50 | 2026-03-04 14:15:00 | 1480.95 | TARGET_HIT | 0.50 | 10.00% |
