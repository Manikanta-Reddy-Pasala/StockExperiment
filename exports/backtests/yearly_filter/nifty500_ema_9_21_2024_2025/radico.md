# Radico Khaitan Ltd (RADICO)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-08 15:15:00 (3709 bars)
- **Last close:** 3481.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 139 |
| ALERT1 | 100 |
| ALERT2 | 96 |
| ALERT2_SKIP | 52 |
| ALERT3 | 268 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 119 |
| PARTIAL | 28 |
| TARGET_HIT | 8 |
| STOP_HIT | 115 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 151 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 71 / 80
- **Target hits / Stop hits / Partials:** 8 / 115 / 28
- **Avg / median % per leg:** 1.44% / -0.26%
- **Sum % (uncompounded):** 217.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 52 | 10 | 19.2% | 5 | 45 | 2 | 0.05% | 2.6% |
| BUY @ 2nd Alert (retest1) | 5 | 5 | 100.0% | 2 | 1 | 2 | 6.27% | 31.3% |
| BUY @ 3rd Alert (retest2) | 47 | 5 | 10.6% | 3 | 44 | 0 | -0.61% | -28.8% |
| SELL (all) | 99 | 61 | 61.6% | 3 | 70 | 26 | 2.17% | 214.6% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.02% | 0.0% |
| SELL @ 3rd Alert (retest2) | 98 | 60 | 61.2% | 3 | 69 | 26 | 2.19% | 214.6% |
| retest1 (combined) | 6 | 6 | 100.0% | 2 | 2 | 2 | 5.23% | 31.4% |
| retest2 (combined) | 145 | 65 | 44.8% | 6 | 113 | 26 | 1.28% | 185.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 11:15:00 | 1639.90 | 1628.05 | 1626.75 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 14:15:00 | 1618.65 | 1625.01 | 1625.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 15:15:00 | 1614.00 | 1622.81 | 1624.52 | Break + close below crossover candle low |

### Cycle 3 — BUY (started 2024-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 09:15:00 | 1638.65 | 1625.98 | 1625.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 10:15:00 | 1697.85 | 1640.35 | 1632.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-18 09:15:00 | 1714.75 | 1719.53 | 1696.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-18 09:15:00 | 1714.75 | 1719.53 | 1696.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 1714.75 | 1719.53 | 1696.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 09:45:00 | 1714.75 | 1719.53 | 1696.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 1701.55 | 1716.17 | 1700.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:30:00 | 1706.80 | 1716.17 | 1700.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 1701.50 | 1713.23 | 1700.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:30:00 | 1697.45 | 1713.23 | 1700.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 11:15:00 | 1695.50 | 1709.69 | 1700.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 12:00:00 | 1695.50 | 1709.69 | 1700.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 12:15:00 | 1694.40 | 1706.63 | 1699.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 10:00:00 | 1704.45 | 1700.66 | 1698.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 13:00:00 | 1699.15 | 1698.94 | 1698.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 15:00:00 | 1703.50 | 1699.85 | 1698.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 09:15:00 | 1694.60 | 1698.01 | 1698.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 09:15:00 | 1694.60 | 1698.01 | 1698.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 10:15:00 | 1688.75 | 1696.16 | 1697.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 14:15:00 | 1693.25 | 1690.55 | 1693.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-23 15:00:00 | 1693.25 | 1690.55 | 1693.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1598.85 | 1582.42 | 1592.78 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 13:15:00 | 1616.20 | 1600.23 | 1598.89 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 1537.45 | 1590.52 | 1595.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 1493.20 | 1571.06 | 1586.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 13:15:00 | 1568.40 | 1562.24 | 1579.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-04 13:45:00 | 1568.10 | 1562.24 | 1579.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 15:15:00 | 1569.30 | 1564.81 | 1577.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 09:15:00 | 1628.70 | 1564.81 | 1577.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 1635.40 | 1578.93 | 1582.62 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 10:15:00 | 1708.05 | 1604.75 | 1594.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 12:15:00 | 1710.00 | 1642.25 | 1613.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 09:15:00 | 1711.50 | 1712.68 | 1693.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 10:00:00 | 1711.50 | 1712.68 | 1693.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 12:15:00 | 1701.95 | 1706.42 | 1695.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 12:30:00 | 1698.00 | 1706.42 | 1695.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 1714.90 | 1707.72 | 1697.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:30:00 | 1697.00 | 1707.72 | 1697.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 1691.20 | 1703.76 | 1697.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 10:00:00 | 1691.20 | 1703.76 | 1697.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 10:15:00 | 1689.70 | 1700.95 | 1697.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 11:00:00 | 1689.70 | 1700.95 | 1697.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-11 14:15:00 | 1687.80 | 1694.02 | 1694.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-11 15:15:00 | 1684.95 | 1692.21 | 1693.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-12 10:15:00 | 1693.00 | 1691.15 | 1692.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-12 10:15:00 | 1693.00 | 1691.15 | 1692.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 10:15:00 | 1693.00 | 1691.15 | 1692.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-12 10:45:00 | 1696.00 | 1691.15 | 1692.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 11:15:00 | 1698.00 | 1692.52 | 1693.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-12 12:00:00 | 1698.00 | 1692.52 | 1693.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 12:15:00 | 1700.00 | 1694.02 | 1693.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-12 13:15:00 | 1715.55 | 1698.32 | 1695.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 13:15:00 | 1717.85 | 1719.40 | 1710.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 14:00:00 | 1717.85 | 1719.40 | 1710.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 1713.95 | 1719.87 | 1712.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 10:00:00 | 1713.95 | 1719.87 | 1712.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 10:15:00 | 1711.70 | 1718.24 | 1712.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 11:00:00 | 1711.70 | 1718.24 | 1712.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 11:15:00 | 1706.00 | 1715.79 | 1712.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 11:45:00 | 1708.10 | 1715.79 | 1712.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 14:15:00 | 1723.05 | 1714.77 | 1712.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 12:45:00 | 1743.35 | 1717.64 | 1714.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 14:15:00 | 1804.35 | 1808.54 | 1808.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 14:15:00 | 1804.35 | 1808.54 | 1808.67 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 09:15:00 | 1838.50 | 1813.49 | 1810.83 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 10:15:00 | 1799.95 | 1813.77 | 1814.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 11:15:00 | 1790.00 | 1809.02 | 1812.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 14:15:00 | 1783.60 | 1778.34 | 1790.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-01 14:45:00 | 1783.85 | 1778.34 | 1790.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 1773.75 | 1776.72 | 1786.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:45:00 | 1775.85 | 1776.72 | 1786.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 1780.45 | 1777.46 | 1785.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 11:45:00 | 1791.30 | 1777.46 | 1785.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 1662.30 | 1670.51 | 1682.79 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 09:15:00 | 1694.55 | 1671.85 | 1669.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 12:15:00 | 1702.50 | 1682.46 | 1675.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 13:15:00 | 1693.35 | 1700.05 | 1690.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 13:15:00 | 1693.35 | 1700.05 | 1690.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 1693.35 | 1700.05 | 1690.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:00:00 | 1693.35 | 1700.05 | 1690.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 1684.80 | 1697.00 | 1690.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 15:00:00 | 1684.80 | 1697.00 | 1690.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 15:15:00 | 1690.00 | 1695.60 | 1690.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:15:00 | 1675.50 | 1695.60 | 1690.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 1665.10 | 1689.50 | 1687.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:15:00 | 1668.80 | 1689.50 | 1687.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 1657.50 | 1683.10 | 1685.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 13:15:00 | 1649.35 | 1668.83 | 1677.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-19 14:15:00 | 1688.50 | 1672.76 | 1678.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 14:15:00 | 1688.50 | 1672.76 | 1678.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 14:15:00 | 1688.50 | 1672.76 | 1678.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 15:00:00 | 1688.50 | 1672.76 | 1678.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 15:15:00 | 1679.45 | 1674.10 | 1678.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 09:15:00 | 1679.00 | 1674.10 | 1678.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-22 12:15:00 | 1699.90 | 1681.81 | 1680.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 12:15:00 | 1699.90 | 1681.81 | 1680.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 14:15:00 | 1714.95 | 1687.20 | 1683.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 12:15:00 | 1697.50 | 1699.65 | 1692.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 12:15:00 | 1697.50 | 1699.65 | 1692.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 1697.50 | 1699.65 | 1692.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 1698.15 | 1699.65 | 1692.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 1698.15 | 1699.35 | 1692.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:30:00 | 1692.85 | 1699.35 | 1692.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 1715.10 | 1717.90 | 1709.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 12:00:00 | 1732.10 | 1720.05 | 1711.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 13:30:00 | 1745.00 | 1726.66 | 1716.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 15:15:00 | 1729.75 | 1732.15 | 1731.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 13:15:00 | 1729.15 | 1730.92 | 1730.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 13:15:00 | 1732.95 | 1731.32 | 1731.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:30:00 | 1738.05 | 1733.60 | 1732.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-31 14:15:00 | 1721.20 | 1736.26 | 1735.24 | SL hit (close<static) qty=1.00 sl=1724.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 09:15:00 | 1727.40 | 1733.17 | 1733.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 11:15:00 | 1718.20 | 1728.96 | 1731.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 14:15:00 | 1709.40 | 1699.57 | 1710.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-02 15:00:00 | 1709.40 | 1699.57 | 1710.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 15:15:00 | 1707.00 | 1701.06 | 1710.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 09:15:00 | 1655.90 | 1701.06 | 1710.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 11:00:00 | 1694.00 | 1679.83 | 1688.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 10:15:00 | 1699.35 | 1684.19 | 1683.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 10:15:00 | 1699.35 | 1684.19 | 1683.66 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 11:15:00 | 1677.45 | 1682.84 | 1683.10 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 14:15:00 | 1706.50 | 1687.09 | 1684.90 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 11:15:00 | 1680.00 | 1683.15 | 1683.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 12:15:00 | 1677.90 | 1682.10 | 1682.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 14:15:00 | 1689.90 | 1683.28 | 1683.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 14:15:00 | 1689.90 | 1683.28 | 1683.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 1689.90 | 1683.28 | 1683.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 15:00:00 | 1689.90 | 1683.28 | 1683.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 1683.00 | 1683.22 | 1683.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 1674.20 | 1683.22 | 1683.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 12:45:00 | 1666.55 | 1675.30 | 1676.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 13:45:00 | 1669.00 | 1673.44 | 1675.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 14:30:00 | 1664.35 | 1670.21 | 1673.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 1662.00 | 1653.28 | 1659.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:00:00 | 1662.00 | 1653.28 | 1659.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 1648.55 | 1652.33 | 1658.67 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-19 09:15:00 | 1666.90 | 1662.02 | 1661.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 1666.90 | 1662.02 | 1661.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 12:15:00 | 1695.20 | 1674.73 | 1668.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 09:15:00 | 1705.20 | 1707.63 | 1695.18 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 11:15:00 | 1712.15 | 1708.32 | 1696.62 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 11:45:00 | 1726.85 | 1711.93 | 1699.33 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 13:15:00 | 1797.76 | 1777.57 | 1753.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 14:15:00 | 1813.19 | 1790.91 | 1761.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-08-26 09:15:00 | 1883.37 | 1812.46 | 1777.21 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 22 — SELL (started 2024-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 15:15:00 | 1811.00 | 1819.39 | 1819.44 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 1822.00 | 1819.91 | 1819.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 10:15:00 | 1836.90 | 1823.31 | 1821.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 14:15:00 | 1989.60 | 1997.60 | 1963.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-03 15:00:00 | 1989.60 | 1997.60 | 1963.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 15:15:00 | 2000.00 | 2003.72 | 1995.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 09:15:00 | 2032.85 | 2003.72 | 1995.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 09:15:00 | 1959.25 | 1995.70 | 1997.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 09:15:00 | 1959.25 | 1995.70 | 1997.31 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 09:15:00 | 2039.95 | 1992.56 | 1992.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 12:15:00 | 2048.00 | 2026.87 | 2018.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 15:15:00 | 2015.00 | 2025.73 | 2020.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 15:15:00 | 2015.00 | 2025.73 | 2020.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 15:15:00 | 2015.00 | 2025.73 | 2020.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 10:00:00 | 2038.45 | 2028.27 | 2022.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-18 09:15:00 | 2242.30 | 2173.47 | 2129.59 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 15:15:00 | 2114.00 | 2171.07 | 2173.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 11:15:00 | 2083.20 | 2125.37 | 2137.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 09:15:00 | 2139.00 | 2120.80 | 2129.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 09:15:00 | 2139.00 | 2120.80 | 2129.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 2139.00 | 2120.80 | 2129.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:30:00 | 2138.45 | 2120.80 | 2129.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 2138.90 | 2124.42 | 2130.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 13:00:00 | 2125.85 | 2126.72 | 2130.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 15:15:00 | 2130.00 | 2129.27 | 2131.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 11:00:00 | 2129.70 | 2124.34 | 2128.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 11:45:00 | 2108.65 | 2121.91 | 2126.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 2065.40 | 2093.67 | 2110.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 12:30:00 | 2064.65 | 2081.70 | 2099.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 15:00:00 | 2050.00 | 2073.01 | 2092.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 15:15:00 | 2023.50 | 2063.41 | 2086.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-30 15:15:00 | 2023.21 | 2063.41 | 2086.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-01 09:15:00 | 2019.56 | 2062.03 | 2083.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 12:00:00 | 2060.00 | 2061.34 | 2079.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 12:15:00 | 2063.20 | 2061.71 | 2078.17 | SL hit (close>ema200) qty=0.50 sl=2061.71 alert=retest2 |

### Cycle 27 — BUY (started 2024-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 13:15:00 | 2030.00 | 2012.74 | 2010.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 14:15:00 | 2112.30 | 2032.65 | 2019.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 09:15:00 | 2196.00 | 2207.83 | 2176.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-16 09:45:00 | 2197.65 | 2207.83 | 2176.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 2174.00 | 2201.06 | 2176.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:00:00 | 2174.00 | 2201.06 | 2176.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 2151.05 | 2191.06 | 2173.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 12:00:00 | 2151.05 | 2191.06 | 2173.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 12:15:00 | 2185.00 | 2189.85 | 2174.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 13:15:00 | 2193.90 | 2189.85 | 2174.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 15:00:00 | 2200.95 | 2192.49 | 2178.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 12:30:00 | 2192.70 | 2202.30 | 2190.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 09:15:00 | 2138.55 | 2184.89 | 2186.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 09:15:00 | 2138.55 | 2184.89 | 2186.08 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-10-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 13:15:00 | 2211.55 | 2187.05 | 2185.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 14:15:00 | 2240.00 | 2197.64 | 2190.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 12:15:00 | 2190.10 | 2203.26 | 2196.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 12:15:00 | 2190.10 | 2203.26 | 2196.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 12:15:00 | 2190.10 | 2203.26 | 2196.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 13:00:00 | 2190.10 | 2203.26 | 2196.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 13:15:00 | 2197.25 | 2202.06 | 2197.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 14:15:00 | 2188.55 | 2202.06 | 2197.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 14:15:00 | 2187.15 | 2199.08 | 2196.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 15:00:00 | 2187.15 | 2199.08 | 2196.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 15:15:00 | 2187.20 | 2196.70 | 2195.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:15:00 | 2160.40 | 2196.70 | 2195.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-10-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 09:15:00 | 2142.15 | 2185.79 | 2190.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 10:15:00 | 2113.10 | 2171.25 | 2183.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-22 14:15:00 | 2184.95 | 2148.80 | 2166.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-22 14:15:00 | 2184.95 | 2148.80 | 2166.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 14:15:00 | 2184.95 | 2148.80 | 2166.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 15:00:00 | 2184.95 | 2148.80 | 2166.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 15:15:00 | 2175.95 | 2154.23 | 2167.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 09:15:00 | 2154.45 | 2154.23 | 2167.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 2169.30 | 2157.24 | 2167.46 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2024-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 11:15:00 | 2221.00 | 2173.89 | 2173.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-23 14:15:00 | 2235.50 | 2196.29 | 2184.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-25 11:15:00 | 2295.25 | 2304.12 | 2261.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 11:15:00 | 2295.25 | 2304.12 | 2261.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 11:15:00 | 2295.25 | 2304.12 | 2261.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 11:30:00 | 2285.85 | 2304.12 | 2261.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 12:15:00 | 2277.25 | 2298.75 | 2263.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 12:45:00 | 2285.05 | 2298.75 | 2263.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 13:15:00 | 2234.80 | 2285.96 | 2260.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 14:00:00 | 2234.80 | 2285.96 | 2260.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 14:15:00 | 2275.05 | 2283.78 | 2261.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 14:30:00 | 2212.40 | 2283.78 | 2261.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 15:15:00 | 2270.00 | 2281.02 | 2262.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-28 09:15:00 | 2305.95 | 2281.02 | 2262.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 2286.10 | 2282.04 | 2264.72 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 2214.85 | 2267.07 | 2268.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 12:15:00 | 2210.70 | 2247.08 | 2258.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 15:15:00 | 2244.70 | 2243.16 | 2253.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-30 09:15:00 | 2253.05 | 2243.16 | 2253.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 2274.65 | 2249.46 | 2255.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 10:00:00 | 2274.65 | 2249.46 | 2255.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 10:15:00 | 2270.00 | 2253.57 | 2256.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 11:15:00 | 2278.00 | 2253.57 | 2256.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2024-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 12:15:00 | 2310.00 | 2268.10 | 2262.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 13:15:00 | 2341.95 | 2282.87 | 2270.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 09:15:00 | 2391.30 | 2394.15 | 2373.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 09:15:00 | 2391.30 | 2394.15 | 2373.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 2391.30 | 2394.15 | 2373.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 11:00:00 | 2442.25 | 2401.45 | 2387.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 09:30:00 | 2429.00 | 2409.03 | 2398.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 10:30:00 | 2409.95 | 2409.45 | 2399.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-07 15:15:00 | 2368.65 | 2393.91 | 2395.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2024-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 15:15:00 | 2368.65 | 2393.91 | 2395.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 2344.90 | 2384.11 | 2390.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 2304.50 | 2299.15 | 2326.60 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-13 09:15:00 | 2266.15 | 2293.83 | 2311.84 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 2270.75 | 2256.56 | 2276.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 2270.75 | 2256.56 | 2276.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 2269.75 | 2263.07 | 2273.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 14:30:00 | 2257.20 | 2262.46 | 2272.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 15:15:00 | 2243.95 | 2262.46 | 2272.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 2265.75 | 2232.55 | 2244.52 | SL hit (close>ema400) qty=1.00 sl=2244.52 alert=retest1 |

### Cycle 35 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 2269.40 | 2252.16 | 2251.66 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 2216.20 | 2247.10 | 2249.73 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 11:15:00 | 2275.65 | 2256.06 | 2253.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-21 14:15:00 | 2309.40 | 2271.91 | 2261.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-22 14:15:00 | 2288.55 | 2299.19 | 2284.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 14:15:00 | 2288.55 | 2299.19 | 2284.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 2288.55 | 2299.19 | 2284.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 15:00:00 | 2288.55 | 2299.19 | 2284.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 15:15:00 | 2305.00 | 2300.35 | 2286.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 09:15:00 | 2372.00 | 2300.35 | 2286.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 10:15:00 | 2356.85 | 2411.20 | 2412.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 10:15:00 | 2356.85 | 2411.20 | 2412.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 12:15:00 | 2345.95 | 2370.26 | 2386.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 10:15:00 | 2369.95 | 2361.28 | 2374.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-05 10:30:00 | 2368.15 | 2361.28 | 2374.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 2366.25 | 2362.27 | 2373.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 12:15:00 | 2360.00 | 2362.27 | 2373.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 13:30:00 | 2362.25 | 2362.30 | 2371.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 09:15:00 | 2359.95 | 2362.61 | 2370.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 11:15:00 | 2383.75 | 2358.37 | 2360.46 | SL hit (close>static) qty=1.00 sl=2374.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 12:15:00 | 2392.75 | 2365.25 | 2363.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 09:15:00 | 2420.00 | 2388.29 | 2375.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 11:15:00 | 2420.40 | 2425.46 | 2408.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-11 11:45:00 | 2421.55 | 2425.46 | 2408.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 15:15:00 | 2419.35 | 2426.64 | 2414.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 09:15:00 | 2454.80 | 2426.64 | 2414.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 09:15:00 | 2397.70 | 2435.32 | 2430.86 | SL hit (close<static) qty=1.00 sl=2400.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-12-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 12:15:00 | 2422.15 | 2427.53 | 2427.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 13:15:00 | 2414.30 | 2424.88 | 2426.64 | Break + close below crossover candle low |

### Cycle 41 — BUY (started 2024-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 14:15:00 | 2440.90 | 2428.09 | 2427.93 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-12-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 15:15:00 | 2422.55 | 2426.98 | 2427.45 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 09:15:00 | 2441.00 | 2429.78 | 2428.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-18 10:15:00 | 2525.00 | 2464.54 | 2448.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 10:15:00 | 2516.70 | 2527.77 | 2496.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-19 10:30:00 | 2516.95 | 2527.77 | 2496.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 2518.70 | 2524.16 | 2507.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:15:00 | 2522.15 | 2524.16 | 2507.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 2522.10 | 2523.75 | 2508.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 11:45:00 | 2541.45 | 2525.41 | 2511.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 14:15:00 | 2476.75 | 2513.03 | 2509.36 | SL hit (close<static) qty=1.00 sl=2497.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 09:15:00 | 2462.00 | 2503.14 | 2505.52 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 14:15:00 | 2536.95 | 2488.83 | 2487.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 2560.00 | 2525.66 | 2509.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 10:15:00 | 2547.85 | 2549.83 | 2533.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-30 11:00:00 | 2547.85 | 2549.83 | 2533.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 2528.00 | 2556.28 | 2544.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 10:00:00 | 2528.00 | 2556.28 | 2544.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 2522.95 | 2549.61 | 2542.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 10:30:00 | 2538.45 | 2549.61 | 2542.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 11:15:00 | 2556.70 | 2551.03 | 2544.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 11:30:00 | 2547.05 | 2551.03 | 2544.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 11:15:00 | 2568.05 | 2579.75 | 2565.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 12:00:00 | 2568.05 | 2579.75 | 2565.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 12:15:00 | 2576.55 | 2579.11 | 2566.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 09:15:00 | 2590.05 | 2579.96 | 2570.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 11:45:00 | 2599.80 | 2586.00 | 2575.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 15:15:00 | 2592.00 | 2584.55 | 2577.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 09:15:00 | 2550.00 | 2585.49 | 2585.01 | SL hit (close<static) qty=1.00 sl=2564.50 alert=retest2 |

### Cycle 46 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 2505.00 | 2569.40 | 2577.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 2490.40 | 2553.60 | 2569.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 2535.00 | 2528.90 | 2549.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 2535.00 | 2528.90 | 2549.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 2535.00 | 2528.90 | 2549.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:45:00 | 2538.90 | 2528.90 | 2549.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 2159.25 | 2167.77 | 2215.08 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 12:15:00 | 2226.70 | 2210.31 | 2208.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 14:15:00 | 2273.40 | 2226.03 | 2215.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 10:15:00 | 2300.00 | 2312.54 | 2281.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:30:00 | 2310.20 | 2312.54 | 2281.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 2285.05 | 2301.69 | 2283.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 13:45:00 | 2277.75 | 2301.69 | 2283.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 2284.40 | 2298.23 | 2283.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 14:45:00 | 2283.05 | 2298.23 | 2283.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 2283.85 | 2295.36 | 2283.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 2240.10 | 2295.36 | 2283.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 2230.00 | 2282.29 | 2278.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:30:00 | 2234.95 | 2282.29 | 2278.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 2210.75 | 2267.98 | 2272.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 2205.60 | 2255.50 | 2266.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 10:15:00 | 2236.50 | 2221.50 | 2239.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 10:15:00 | 2236.50 | 2221.50 | 2239.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 2236.50 | 2221.50 | 2239.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:00:00 | 2236.50 | 2221.50 | 2239.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 2238.00 | 2224.80 | 2239.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:30:00 | 2246.95 | 2224.80 | 2239.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 2239.15 | 2227.67 | 2239.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 12:45:00 | 2238.60 | 2227.67 | 2239.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 2244.75 | 2231.08 | 2240.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 13:45:00 | 2244.15 | 2231.08 | 2240.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 2231.50 | 2231.17 | 2239.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:15:00 | 2222.35 | 2230.93 | 2238.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:45:00 | 2220.65 | 2227.59 | 2236.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:15:00 | 2111.23 | 2174.46 | 2202.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:15:00 | 2109.62 | 2174.46 | 2202.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 12:15:00 | 2118.65 | 2110.26 | 2144.94 | SL hit (close>ema200) qty=0.50 sl=2110.26 alert=retest2 |

### Cycle 49 — BUY (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 12:15:00 | 2182.25 | 2155.60 | 2152.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 13:15:00 | 2359.75 | 2219.14 | 2195.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-04 11:15:00 | 2314.60 | 2334.57 | 2298.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-04 12:00:00 | 2314.60 | 2334.57 | 2298.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 14:15:00 | 2304.00 | 2320.69 | 2300.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 15:15:00 | 2300.00 | 2320.69 | 2300.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 2300.00 | 2316.55 | 2300.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 2293.15 | 2316.55 | 2300.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 2288.85 | 2311.01 | 2299.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 15:15:00 | 2322.45 | 2305.64 | 2300.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-06 10:15:00 | 2269.55 | 2295.83 | 2297.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 10:15:00 | 2269.55 | 2295.83 | 2297.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 12:15:00 | 2260.40 | 2284.92 | 2291.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 14:15:00 | 2214.50 | 2210.32 | 2235.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-10 15:00:00 | 2214.50 | 2210.32 | 2235.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 14:15:00 | 2215.65 | 2160.70 | 2191.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 15:00:00 | 2215.65 | 2160.70 | 2191.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 15:15:00 | 2188.30 | 2166.22 | 2191.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 09:15:00 | 2138.55 | 2166.22 | 2191.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 09:15:00 | 2137.50 | 2160.47 | 2186.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 14:15:00 | 2123.35 | 2162.55 | 2176.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 15:00:00 | 2116.25 | 2153.29 | 2170.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 2109.00 | 2148.02 | 2166.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 2017.18 | 2051.47 | 2100.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 2010.44 | 2051.47 | 2100.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 2003.55 | 2051.47 | 2100.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-17 13:15:00 | 1911.01 | 1978.81 | 2046.53 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 51 — BUY (started 2025-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 15:15:00 | 2018.60 | 1988.47 | 1986.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 2100.15 | 2010.81 | 1996.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 2115.00 | 2115.55 | 2067.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 2115.00 | 2115.55 | 2067.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 2085.65 | 2112.97 | 2082.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 14:00:00 | 2085.65 | 2112.97 | 2082.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 2098.05 | 2109.98 | 2083.50 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2025-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 12:15:00 | 2034.80 | 2070.72 | 2072.67 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 15:15:00 | 2068.80 | 2060.12 | 2059.54 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 2016.35 | 2051.37 | 2055.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 11:15:00 | 2011.00 | 2043.53 | 2051.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 14:15:00 | 2084.00 | 2044.18 | 2048.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 14:15:00 | 2084.00 | 2044.18 | 2048.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 2084.00 | 2044.18 | 2048.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 14:45:00 | 2108.65 | 2044.18 | 2048.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 2049.20 | 2045.18 | 2048.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:15:00 | 2021.45 | 2045.18 | 2048.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 09:15:00 | 2028.70 | 2039.39 | 2041.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-04 09:15:00 | 2069.60 | 2045.43 | 2044.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 09:15:00 | 2069.60 | 2045.43 | 2044.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 2079.70 | 2062.76 | 2055.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 14:15:00 | 2187.10 | 2198.38 | 2159.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 15:00:00 | 2187.10 | 2198.38 | 2159.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 2176.15 | 2189.35 | 2176.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 11:15:00 | 2188.95 | 2188.16 | 2177.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 10:15:00 | 2163.00 | 2174.98 | 2175.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 10:15:00 | 2163.00 | 2174.98 | 2175.34 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 14:15:00 | 2195.00 | 2174.49 | 2174.42 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 15:15:00 | 2163.00 | 2172.19 | 2173.38 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 09:15:00 | 2200.00 | 2177.75 | 2175.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 11:15:00 | 2203.00 | 2186.36 | 2180.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 12:15:00 | 2169.05 | 2182.90 | 2179.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 12:15:00 | 2169.05 | 2182.90 | 2179.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 12:15:00 | 2169.05 | 2182.90 | 2179.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 13:00:00 | 2169.05 | 2182.90 | 2179.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 13:15:00 | 2190.45 | 2184.41 | 2180.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 09:15:00 | 2213.00 | 2186.73 | 2182.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-28 14:15:00 | 2434.30 | 2366.30 | 2352.81 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2025-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 10:15:00 | 2353.60 | 2359.67 | 2360.03 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 11:15:00 | 2395.60 | 2366.85 | 2363.26 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 11:15:00 | 2354.95 | 2363.61 | 2364.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 09:15:00 | 2313.00 | 2352.73 | 2358.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 2288.00 | 2267.45 | 2295.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 2288.00 | 2267.45 | 2295.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 2288.00 | 2267.45 | 2295.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:30:00 | 2326.55 | 2267.45 | 2295.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 2305.20 | 2275.00 | 2296.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 11:00:00 | 2305.20 | 2275.00 | 2296.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 11:15:00 | 2325.35 | 2285.07 | 2299.15 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 15:15:00 | 2328.00 | 2308.50 | 2307.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 2373.90 | 2327.65 | 2317.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-21 11:15:00 | 2456.00 | 2458.16 | 2438.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-21 13:15:00 | 2439.00 | 2454.54 | 2440.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 13:15:00 | 2439.00 | 2454.54 | 2440.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 13:45:00 | 2439.80 | 2454.54 | 2440.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 14:15:00 | 2435.00 | 2450.64 | 2439.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 14:30:00 | 2439.30 | 2450.64 | 2439.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 15:15:00 | 2436.10 | 2447.73 | 2439.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 09:15:00 | 2467.00 | 2447.73 | 2439.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 2394.40 | 2462.21 | 2469.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 2394.40 | 2462.21 | 2469.98 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 15:15:00 | 2474.00 | 2455.14 | 2453.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 13:15:00 | 2497.60 | 2469.56 | 2461.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 09:15:00 | 2456.70 | 2472.33 | 2465.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 09:15:00 | 2456.70 | 2472.33 | 2465.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 2456.70 | 2472.33 | 2465.43 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2025-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 14:15:00 | 2457.00 | 2461.83 | 2461.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 15:15:00 | 2441.00 | 2457.67 | 2460.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 13:15:00 | 2445.70 | 2443.36 | 2450.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-02 13:45:00 | 2447.10 | 2443.36 | 2450.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 14:15:00 | 2444.70 | 2443.62 | 2450.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 14:45:00 | 2448.70 | 2443.62 | 2450.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 15:15:00 | 2445.20 | 2443.94 | 2449.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:15:00 | 2505.50 | 2443.94 | 2449.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 2545.00 | 2464.15 | 2458.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 09:15:00 | 2547.00 | 2513.38 | 2490.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 2456.00 | 2516.07 | 2506.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 09:15:00 | 2456.00 | 2516.07 | 2506.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 2456.00 | 2516.07 | 2506.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 10:00:00 | 2456.00 | 2516.07 | 2506.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 2459.40 | 2504.73 | 2501.79 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 11:15:00 | 2432.70 | 2490.33 | 2495.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 2393.20 | 2455.69 | 2469.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 14:15:00 | 2448.50 | 2440.14 | 2454.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 15:00:00 | 2448.50 | 2440.14 | 2454.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 2448.80 | 2441.87 | 2453.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 2478.40 | 2441.87 | 2453.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 2481.10 | 2449.72 | 2456.40 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 2506.00 | 2465.56 | 2462.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 2526.00 | 2491.11 | 2476.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 12:15:00 | 2558.70 | 2574.96 | 2547.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 13:00:00 | 2558.70 | 2574.96 | 2547.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 2572.80 | 2572.15 | 2550.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:30:00 | 2563.30 | 2572.15 | 2550.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 2571.80 | 2579.11 | 2572.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 14:30:00 | 2572.60 | 2579.11 | 2572.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 15:15:00 | 2572.10 | 2577.71 | 2572.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:15:00 | 2564.00 | 2577.71 | 2572.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 2580.00 | 2583.66 | 2577.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 13:45:00 | 2576.90 | 2583.66 | 2577.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 2590.10 | 2584.95 | 2579.07 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 2562.10 | 2576.08 | 2576.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 09:15:00 | 2484.50 | 2553.42 | 2566.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 11:15:00 | 2474.60 | 2473.71 | 2504.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 12:00:00 | 2474.60 | 2473.71 | 2504.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 2484.50 | 2476.71 | 2494.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:30:00 | 2486.20 | 2476.71 | 2494.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 2485.70 | 2478.50 | 2493.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 2485.70 | 2478.50 | 2493.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 2479.70 | 2470.92 | 2482.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 13:15:00 | 2451.90 | 2472.21 | 2480.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 13:45:00 | 2448.50 | 2459.91 | 2467.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 10:15:00 | 2476.80 | 2463.91 | 2463.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 10:15:00 | 2476.80 | 2463.91 | 2463.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 12:15:00 | 2498.80 | 2473.03 | 2467.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 14:15:00 | 2706.00 | 2706.79 | 2673.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 14:30:00 | 2706.40 | 2706.79 | 2673.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 2669.40 | 2698.23 | 2674.98 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2025-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 13:15:00 | 2637.70 | 2661.30 | 2662.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 14:15:00 | 2628.10 | 2654.66 | 2659.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 14:15:00 | 2650.00 | 2644.98 | 2650.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 14:15:00 | 2650.00 | 2644.98 | 2650.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 2650.00 | 2644.98 | 2650.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 14:45:00 | 2650.00 | 2644.98 | 2650.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 2647.00 | 2645.38 | 2650.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:15:00 | 2656.10 | 2645.38 | 2650.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 2658.30 | 2647.97 | 2650.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 11:45:00 | 2645.50 | 2648.97 | 2650.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 13:45:00 | 2646.60 | 2648.14 | 2650.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 2630.30 | 2649.16 | 2650.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 11:15:00 | 2625.10 | 2604.49 | 2603.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 2625.10 | 2604.49 | 2603.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 2633.20 | 2616.79 | 2610.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 11:15:00 | 2608.00 | 2615.24 | 2610.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 11:15:00 | 2608.00 | 2615.24 | 2610.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 2608.00 | 2615.24 | 2610.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:00:00 | 2608.00 | 2615.24 | 2610.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 2599.90 | 2612.17 | 2609.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:45:00 | 2599.40 | 2612.17 | 2609.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 2600.00 | 2609.74 | 2608.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 13:45:00 | 2594.20 | 2609.74 | 2608.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 2610.00 | 2609.54 | 2608.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 09:15:00 | 2631.00 | 2609.54 | 2608.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 2638.60 | 2642.87 | 2643.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 09:15:00 | 2638.60 | 2642.87 | 2643.04 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 2646.60 | 2643.25 | 2643.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 2662.00 | 2649.10 | 2646.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 14:15:00 | 2689.40 | 2690.96 | 2676.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 15:00:00 | 2689.40 | 2690.96 | 2676.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 2677.40 | 2687.79 | 2677.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:30:00 | 2681.20 | 2687.79 | 2677.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 2677.00 | 2685.63 | 2677.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:45:00 | 2672.20 | 2685.63 | 2677.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 2680.30 | 2684.56 | 2677.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:30:00 | 2677.40 | 2684.56 | 2677.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 2678.60 | 2683.37 | 2677.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:30:00 | 2672.50 | 2683.37 | 2677.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 2680.20 | 2682.74 | 2677.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 13:45:00 | 2677.80 | 2682.74 | 2677.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 2673.30 | 2680.85 | 2677.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 15:00:00 | 2673.30 | 2680.85 | 2677.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 2677.20 | 2680.12 | 2677.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:15:00 | 2675.20 | 2680.12 | 2677.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 09:15:00 | 2653.00 | 2674.70 | 2675.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 11:15:00 | 2649.10 | 2665.95 | 2670.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 14:15:00 | 2617.00 | 2612.72 | 2632.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-30 15:00:00 | 2617.00 | 2612.72 | 2632.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 2613.10 | 2614.62 | 2629.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 10:15:00 | 2601.10 | 2614.62 | 2629.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 09:15:00 | 2634.80 | 2580.29 | 2574.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 09:15:00 | 2634.80 | 2580.29 | 2574.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 14:15:00 | 2679.00 | 2659.64 | 2641.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 15:15:00 | 2692.00 | 2696.98 | 2675.61 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-11 09:15:00 | 2702.10 | 2696.98 | 2675.61 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 2743.80 | 2745.03 | 2735.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:30:00 | 2737.70 | 2745.03 | 2735.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 2740.00 | 2743.66 | 2738.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:30:00 | 2739.80 | 2743.66 | 2738.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 2742.90 | 2743.51 | 2738.77 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-17 10:15:00 | 2738.40 | 2741.86 | 2738.80 | SL hit (close<ema400) qty=1.00 sl=2738.80 alert=retest1 |

### Cycle 78 — SELL (started 2025-07-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 15:15:00 | 2731.40 | 2736.28 | 2736.85 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 09:15:00 | 2751.20 | 2739.27 | 2738.16 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 14:15:00 | 2718.60 | 2738.18 | 2738.92 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 11:15:00 | 2751.10 | 2732.66 | 2732.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 12:15:00 | 2758.70 | 2737.87 | 2734.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 11:15:00 | 2747.00 | 2750.40 | 2743.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 11:15:00 | 2747.00 | 2750.40 | 2743.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 2747.00 | 2750.40 | 2743.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 12:00:00 | 2747.00 | 2750.40 | 2743.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 2746.90 | 2749.70 | 2743.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 12:45:00 | 2745.70 | 2749.70 | 2743.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 2748.60 | 2749.48 | 2744.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:30:00 | 2744.80 | 2749.48 | 2744.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 2763.80 | 2752.34 | 2746.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 2768.90 | 2752.97 | 2746.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 2737.70 | 2749.92 | 2746.13 | SL hit (close<static) qty=1.00 sl=2743.10 alert=retest2 |

### Cycle 82 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 2730.60 | 2742.87 | 2743.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 12:15:00 | 2722.00 | 2738.69 | 2741.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 2720.00 | 2711.81 | 2721.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 2720.00 | 2711.81 | 2721.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 2720.00 | 2711.81 | 2721.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 11:30:00 | 2701.00 | 2707.07 | 2717.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:45:00 | 2696.60 | 2703.67 | 2714.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 09:30:00 | 2700.10 | 2688.81 | 2703.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 12:45:00 | 2700.00 | 2693.55 | 2701.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 2685.00 | 2691.84 | 2700.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 15:00:00 | 2682.40 | 2689.95 | 2698.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 14:15:00 | 2711.30 | 2699.22 | 2699.60 | SL hit (close>static) qty=1.00 sl=2702.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 15:15:00 | 2707.00 | 2700.78 | 2700.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 09:15:00 | 2722.50 | 2705.12 | 2702.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 09:15:00 | 2826.00 | 2828.49 | 2789.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-04 09:30:00 | 2817.70 | 2828.49 | 2789.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 2787.20 | 2820.23 | 2788.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:00:00 | 2787.20 | 2820.23 | 2788.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 2801.90 | 2816.56 | 2790.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 09:15:00 | 2836.50 | 2806.03 | 2792.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 09:30:00 | 2844.70 | 2849.59 | 2842.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 10:15:00 | 2819.00 | 2836.44 | 2838.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 10:15:00 | 2819.00 | 2836.44 | 2838.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 15:15:00 | 2810.40 | 2829.20 | 2834.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 2843.70 | 2832.10 | 2834.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 09:15:00 | 2843.70 | 2832.10 | 2834.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 2843.70 | 2832.10 | 2834.93 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 2859.00 | 2837.48 | 2837.11 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-08-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 13:15:00 | 2833.10 | 2836.74 | 2836.90 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 2850.60 | 2839.51 | 2838.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 2874.90 | 2849.10 | 2842.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 2868.70 | 2880.21 | 2866.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 2868.70 | 2880.21 | 2866.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 2868.70 | 2880.21 | 2866.25 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 12:15:00 | 2815.50 | 2854.16 | 2856.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 13:15:00 | 2795.70 | 2842.47 | 2851.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 13:15:00 | 2862.90 | 2830.03 | 2838.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 13:15:00 | 2862.90 | 2830.03 | 2838.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 2862.90 | 2830.03 | 2838.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:00:00 | 2862.90 | 2830.03 | 2838.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 2861.90 | 2836.41 | 2840.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:30:00 | 2869.50 | 2836.41 | 2840.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 2879.50 | 2847.36 | 2844.71 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 11:15:00 | 2825.80 | 2847.65 | 2849.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 14:15:00 | 2821.00 | 2839.35 | 2844.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 12:15:00 | 2832.70 | 2821.57 | 2832.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 12:15:00 | 2832.70 | 2821.57 | 2832.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 2832.70 | 2821.57 | 2832.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:00:00 | 2832.70 | 2821.57 | 2832.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 2830.00 | 2823.25 | 2832.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:00:00 | 2830.00 | 2823.25 | 2832.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 2829.90 | 2824.58 | 2831.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:45:00 | 2837.80 | 2824.58 | 2831.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 2825.90 | 2824.85 | 2831.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:15:00 | 2872.00 | 2824.85 | 2831.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 2847.40 | 2829.36 | 2832.80 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 11:15:00 | 2861.30 | 2839.00 | 2836.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 14:15:00 | 2890.20 | 2857.45 | 2846.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 2858.30 | 2893.61 | 2884.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 2858.30 | 2893.61 | 2884.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 2858.30 | 2893.61 | 2884.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 2862.90 | 2893.61 | 2884.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 2899.50 | 2894.79 | 2885.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 12:00:00 | 2911.90 | 2898.21 | 2887.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 14:45:00 | 2913.20 | 2901.06 | 2891.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 11:15:00 | 2865.90 | 2883.76 | 2885.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 11:15:00 | 2865.90 | 2883.76 | 2885.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 12:15:00 | 2853.90 | 2877.79 | 2883.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 11:15:00 | 2872.70 | 2860.43 | 2869.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 11:15:00 | 2872.70 | 2860.43 | 2869.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 2872.70 | 2860.43 | 2869.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:45:00 | 2873.10 | 2860.43 | 2869.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 2833.30 | 2855.01 | 2866.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 12:30:00 | 2868.90 | 2855.01 | 2866.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 2849.10 | 2848.35 | 2861.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 14:30:00 | 2858.70 | 2848.35 | 2861.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 2855.50 | 2850.40 | 2859.83 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 2897.60 | 2866.06 | 2862.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 10:15:00 | 2940.60 | 2893.00 | 2879.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 14:15:00 | 2901.20 | 2905.53 | 2891.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 15:00:00 | 2901.20 | 2905.53 | 2891.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 2899.20 | 2904.26 | 2891.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 2872.70 | 2904.26 | 2891.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 2890.20 | 2901.45 | 2891.69 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 12:15:00 | 2840.90 | 2879.57 | 2883.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 2822.50 | 2868.15 | 2877.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 14:15:00 | 2756.70 | 2752.33 | 2779.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 15:00:00 | 2756.70 | 2752.33 | 2779.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 2721.50 | 2747.39 | 2772.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 10:15:00 | 2712.30 | 2747.39 | 2772.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 13:15:00 | 2791.40 | 2755.52 | 2767.74 | SL hit (close>static) qty=1.00 sl=2790.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 09:15:00 | 2830.00 | 2785.70 | 2779.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 11:15:00 | 2852.40 | 2806.96 | 2791.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 14:15:00 | 2870.50 | 2871.67 | 2844.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 14:45:00 | 2874.90 | 2871.67 | 2844.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 3011.00 | 3002.32 | 2980.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:30:00 | 2998.90 | 3002.32 | 2980.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 3034.20 | 3037.04 | 3022.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 13:15:00 | 3020.00 | 3037.04 | 3022.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 3020.40 | 3033.71 | 3022.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 13:45:00 | 3018.10 | 3033.71 | 3022.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 3030.00 | 3032.97 | 3023.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:30:00 | 3023.80 | 3032.97 | 3023.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 3024.90 | 3031.36 | 3023.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:15:00 | 3011.10 | 3031.36 | 3023.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 2992.20 | 3023.53 | 3020.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:45:00 | 2992.90 | 3023.53 | 3020.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 2992.40 | 3017.30 | 3018.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 11:15:00 | 2986.40 | 3011.12 | 3015.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 13:15:00 | 2968.00 | 2965.38 | 2983.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 13:15:00 | 2968.00 | 2965.38 | 2983.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 2968.00 | 2965.38 | 2983.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:45:00 | 2985.00 | 2965.38 | 2983.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 2999.80 | 2973.12 | 2982.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 12:45:00 | 2969.20 | 2975.32 | 2981.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 15:00:00 | 2956.00 | 2970.65 | 2978.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 2918.60 | 2903.21 | 2902.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 2918.60 | 2903.21 | 2902.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 2942.80 | 2911.13 | 2906.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 13:15:00 | 2917.70 | 2925.25 | 2916.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 13:15:00 | 2917.70 | 2925.25 | 2916.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 2917.70 | 2925.25 | 2916.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 13:45:00 | 2915.30 | 2925.25 | 2916.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 2961.20 | 2932.44 | 2920.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 11:30:00 | 2968.20 | 2945.79 | 2931.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 12:15:00 | 2965.90 | 2945.79 | 2931.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:15:00 | 2973.00 | 2953.12 | 2939.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 11:45:00 | 2966.90 | 2973.61 | 2970.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 2972.60 | 2973.41 | 2970.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:30:00 | 2966.70 | 2973.41 | 2970.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 2971.60 | 2973.05 | 2971.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:15:00 | 2966.60 | 2973.05 | 2971.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 2988.50 | 2976.14 | 2972.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:30:00 | 2974.60 | 2976.14 | 2972.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 3002.60 | 2981.09 | 2975.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 09:30:00 | 3026.40 | 2988.38 | 2982.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 2945.80 | 2984.70 | 2985.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 2945.80 | 2984.70 | 2985.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 2938.70 | 2975.50 | 2980.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 2944.10 | 2939.42 | 2957.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:00:00 | 2944.10 | 2939.42 | 2957.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 2950.00 | 2939.10 | 2949.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:45:00 | 2960.00 | 2939.10 | 2949.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 2946.00 | 2940.48 | 2949.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 2988.00 | 2940.48 | 2949.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 3001.10 | 2952.60 | 2954.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:30:00 | 3000.30 | 2952.60 | 2954.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 2994.10 | 2960.90 | 2957.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 11:15:00 | 3011.00 | 2970.92 | 2962.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 12:15:00 | 3272.90 | 3280.74 | 3211.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 13:00:00 | 3272.90 | 3280.74 | 3211.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 3239.70 | 3264.11 | 3240.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:30:00 | 3234.20 | 3264.11 | 3240.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 3218.40 | 3254.97 | 3238.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 15:00:00 | 3218.40 | 3254.97 | 3238.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 3219.90 | 3247.96 | 3236.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 3203.10 | 3247.96 | 3236.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 3222.50 | 3236.74 | 3233.15 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 12:15:00 | 3220.40 | 3230.64 | 3230.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 11:15:00 | 3209.70 | 3221.27 | 3225.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 3210.00 | 3206.35 | 3215.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 3210.00 | 3206.35 | 3215.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 3210.00 | 3206.35 | 3215.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 11:15:00 | 3182.50 | 3203.56 | 3213.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 14:30:00 | 3166.20 | 3194.72 | 3205.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 11:00:00 | 3178.00 | 3152.43 | 3160.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 12:00:00 | 3179.20 | 3157.79 | 3162.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 13:15:00 | 3192.00 | 3169.16 | 3167.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 13:15:00 | 3192.00 | 3169.16 | 3167.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 09:15:00 | 3197.40 | 3181.72 | 3174.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 14:15:00 | 3177.10 | 3211.56 | 3201.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 14:15:00 | 3177.10 | 3211.56 | 3201.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 3177.10 | 3211.56 | 3201.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:00:00 | 3177.10 | 3211.56 | 3201.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 3178.00 | 3204.84 | 3199.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:15:00 | 3175.10 | 3204.84 | 3199.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 3276.10 | 3274.06 | 3260.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:00:00 | 3276.10 | 3274.06 | 3260.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 3314.00 | 3296.62 | 3280.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 13:30:00 | 3304.70 | 3296.62 | 3280.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 3269.20 | 3292.62 | 3282.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:00:00 | 3269.20 | 3292.62 | 3282.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 3281.00 | 3290.30 | 3282.61 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2025-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 12:15:00 | 3219.90 | 3270.57 | 3274.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 13:15:00 | 3201.00 | 3256.66 | 3267.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 11:15:00 | 3241.30 | 3228.42 | 3246.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-14 12:00:00 | 3241.30 | 3228.42 | 3246.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 3247.90 | 3232.32 | 3246.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 12:30:00 | 3248.20 | 3232.32 | 3246.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 3248.60 | 3235.57 | 3246.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:45:00 | 3257.60 | 3235.57 | 3246.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 3263.40 | 3241.14 | 3248.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 15:00:00 | 3263.40 | 3241.14 | 3248.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 3260.00 | 3244.91 | 3249.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 3287.00 | 3244.91 | 3249.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 3291.40 | 3254.21 | 3253.08 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 12:15:00 | 3246.70 | 3272.70 | 3274.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 13:15:00 | 3229.60 | 3264.08 | 3270.12 | Break + close below crossover candle low |

### Cycle 105 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 3413.40 | 3284.13 | 3276.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 10:15:00 | 3570.70 | 3341.44 | 3303.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 13:15:00 | 3356.60 | 3360.77 | 3323.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-20 14:00:00 | 3356.60 | 3360.77 | 3323.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 3319.80 | 3363.37 | 3340.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:00:00 | 3319.80 | 3363.37 | 3340.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 3306.20 | 3351.94 | 3337.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:45:00 | 3300.00 | 3351.94 | 3337.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2025-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 14:15:00 | 3279.30 | 3327.48 | 3328.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 09:15:00 | 3237.30 | 3281.53 | 3289.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 10:15:00 | 3200.40 | 3199.43 | 3232.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 10:15:00 | 3200.40 | 3199.43 | 3232.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 3200.40 | 3199.43 | 3232.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:00:00 | 3200.40 | 3199.43 | 3232.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 3238.50 | 3210.07 | 3231.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 13:00:00 | 3238.50 | 3210.07 | 3231.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 3227.00 | 3213.46 | 3231.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 13:30:00 | 3238.80 | 3213.46 | 3231.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 3213.00 | 3213.36 | 3229.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 15:15:00 | 3207.00 | 3213.36 | 3229.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 12:30:00 | 3207.80 | 3213.18 | 3223.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:30:00 | 3208.00 | 3216.41 | 3223.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 09:45:00 | 3209.40 | 3218.42 | 3222.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 3219.00 | 3218.54 | 3222.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:45:00 | 3247.00 | 3218.54 | 3222.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 3208.00 | 3216.43 | 3221.09 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 3235.50 | 3216.31 | 3218.57 | SL hit (close>static) qty=1.00 sl=3232.20 alert=retest2 |

### Cycle 107 — BUY (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 10:15:00 | 3249.90 | 3223.03 | 3221.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 13:15:00 | 3257.90 | 3238.70 | 3229.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 11:15:00 | 3225.00 | 3242.61 | 3236.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 11:15:00 | 3225.00 | 3242.61 | 3236.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 3225.00 | 3242.61 | 3236.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:00:00 | 3225.00 | 3242.61 | 3236.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 3239.60 | 3242.00 | 3236.47 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2025-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 15:15:00 | 3210.00 | 3230.68 | 3232.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 09:15:00 | 3194.80 | 3223.50 | 3228.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 15:15:00 | 3207.70 | 3206.78 | 3216.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 15:15:00 | 3207.70 | 3206.78 | 3216.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 3207.70 | 3206.78 | 3216.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 3227.20 | 3206.78 | 3216.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 3236.80 | 3212.78 | 3218.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:30:00 | 3243.20 | 3212.78 | 3218.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 3205.70 | 3211.37 | 3217.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 11:30:00 | 3199.50 | 3208.91 | 3215.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:45:00 | 3200.00 | 3181.64 | 3187.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 3240.00 | 3197.69 | 3194.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 3240.00 | 3197.69 | 3194.42 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 13:15:00 | 3199.60 | 3205.62 | 3205.86 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 3223.00 | 3207.94 | 3206.72 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 3184.30 | 3217.87 | 3217.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 10:15:00 | 3151.00 | 3204.50 | 3211.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 15:15:00 | 3177.00 | 3174.28 | 3190.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 09:15:00 | 3175.30 | 3174.28 | 3190.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 3181.90 | 3175.81 | 3190.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:45:00 | 3195.00 | 3175.81 | 3190.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 3144.20 | 3117.53 | 3142.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 3144.20 | 3117.53 | 3142.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 3178.20 | 3129.66 | 3145.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 3178.20 | 3129.66 | 3145.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 3169.50 | 3137.63 | 3147.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:15:00 | 3154.00 | 3137.63 | 3147.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 09:15:00 | 3150.50 | 3146.30 | 3150.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 11:15:00 | 3180.90 | 3157.25 | 3154.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 3180.90 | 3157.25 | 3154.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 3202.90 | 3169.78 | 3160.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 12:15:00 | 3256.30 | 3260.34 | 3228.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 13:00:00 | 3256.30 | 3260.34 | 3228.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 3310.10 | 3272.87 | 3248.88 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 3228.80 | 3266.33 | 3270.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 10:15:00 | 3167.80 | 3231.19 | 3249.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 3278.90 | 3227.61 | 3242.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 13:15:00 | 3278.90 | 3227.61 | 3242.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 3278.90 | 3227.61 | 3242.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 3278.90 | 3227.61 | 3242.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 3378.80 | 3257.85 | 3254.78 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 13:15:00 | 3262.70 | 3273.10 | 3273.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 14:15:00 | 3247.00 | 3267.88 | 3271.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 14:15:00 | 3103.80 | 3092.10 | 3142.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 15:00:00 | 3103.80 | 3092.10 | 3142.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 3151.00 | 3111.89 | 3133.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 13:00:00 | 3151.00 | 3111.89 | 3133.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 3150.00 | 3119.51 | 3134.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 13:30:00 | 3162.00 | 3119.51 | 3134.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 3140.00 | 3124.87 | 3134.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 3146.10 | 3124.87 | 3134.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 3154.60 | 3130.82 | 3136.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:45:00 | 3164.70 | 3130.82 | 3136.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 3170.60 | 3138.77 | 3139.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:45:00 | 3167.00 | 3138.77 | 3139.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 11:15:00 | 3206.80 | 3152.38 | 3145.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 12:15:00 | 3231.10 | 3168.12 | 3153.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 3166.00 | 3184.45 | 3169.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 10:15:00 | 3166.00 | 3184.45 | 3169.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 3166.00 | 3184.45 | 3169.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 3166.00 | 3184.45 | 3169.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 3162.60 | 3180.08 | 3168.86 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 3096.40 | 3154.69 | 3158.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 3087.20 | 3141.19 | 3152.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 2899.90 | 2869.28 | 2942.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 10:15:00 | 2932.00 | 2881.83 | 2941.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 2932.00 | 2881.83 | 2941.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:30:00 | 2941.60 | 2881.83 | 2941.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 2951.80 | 2895.82 | 2942.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:00:00 | 2951.80 | 2895.82 | 2942.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 2940.00 | 2904.66 | 2942.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:15:00 | 2927.60 | 2904.66 | 2942.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:30:00 | 2937.10 | 2914.50 | 2940.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:00:00 | 2929.10 | 2914.50 | 2940.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 2909.40 | 2917.62 | 2937.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 2933.90 | 2920.14 | 2935.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:00:00 | 2933.90 | 2920.14 | 2935.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 2931.90 | 2922.49 | 2934.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:30:00 | 2936.50 | 2922.49 | 2934.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 2925.40 | 2923.07 | 2934.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:00:00 | 2903.30 | 2919.12 | 2931.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 10:15:00 | 2896.00 | 2914.19 | 2926.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 2790.24 | 2853.64 | 2880.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 2781.22 | 2841.89 | 2872.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 2782.64 | 2841.89 | 2872.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 14:15:00 | 2763.93 | 2802.44 | 2842.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 14:15:00 | 2758.14 | 2802.44 | 2842.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 15:15:00 | 2751.20 | 2791.95 | 2834.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 11:15:00 | 2792.10 | 2784.38 | 2819.69 | SL hit (close>ema200) qty=0.50 sl=2784.38 alert=retest2 |

### Cycle 119 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 2874.00 | 2835.73 | 2830.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 13:15:00 | 2942.10 | 2862.71 | 2844.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 14:15:00 | 2958.40 | 2960.40 | 2917.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 15:00:00 | 2958.40 | 2960.40 | 2917.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 2945.20 | 2957.61 | 2923.63 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 10:15:00 | 2820.20 | 2903.35 | 2912.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 11:15:00 | 2790.00 | 2880.68 | 2900.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 2759.60 | 2731.40 | 2780.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-30 10:00:00 | 2759.60 | 2731.40 | 2780.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 2806.70 | 2746.46 | 2782.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:45:00 | 2808.90 | 2746.46 | 2782.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 2807.50 | 2758.67 | 2784.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:00:00 | 2807.50 | 2758.67 | 2784.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 2795.00 | 2765.93 | 2785.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:30:00 | 2807.80 | 2765.93 | 2785.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 2748.70 | 2753.63 | 2769.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:30:00 | 2766.20 | 2753.63 | 2769.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 2757.00 | 2736.04 | 2749.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 14:45:00 | 2754.00 | 2736.04 | 2749.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 2774.90 | 2743.81 | 2751.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 2799.90 | 2743.81 | 2751.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2787.10 | 2752.47 | 2755.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 10:30:00 | 2776.20 | 2753.16 | 2755.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 14:15:00 | 2769.20 | 2757.79 | 2756.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 14:15:00 | 2769.20 | 2757.79 | 2756.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 2840.90 | 2776.36 | 2765.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 13:15:00 | 2773.20 | 2790.26 | 2777.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 13:15:00 | 2773.20 | 2790.26 | 2777.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 2773.20 | 2790.26 | 2777.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:45:00 | 2771.60 | 2790.26 | 2777.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 2755.00 | 2783.20 | 2775.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 14:45:00 | 2752.80 | 2783.20 | 2775.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 10:15:00 | 2754.60 | 2768.50 | 2769.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 11:15:00 | 2716.10 | 2758.02 | 2765.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 14:15:00 | 2773.30 | 2752.89 | 2760.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 14:15:00 | 2773.30 | 2752.89 | 2760.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 2773.30 | 2752.89 | 2760.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 15:00:00 | 2773.30 | 2752.89 | 2760.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 2769.00 | 2756.11 | 2761.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 09:15:00 | 2738.30 | 2756.11 | 2761.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 10:15:00 | 2782.20 | 2742.58 | 2744.81 | SL hit (close>static) qty=1.00 sl=2773.20 alert=retest2 |

### Cycle 123 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 2811.40 | 2756.34 | 2750.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 2822.90 | 2769.65 | 2757.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 09:15:00 | 2784.00 | 2795.39 | 2775.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 10:00:00 | 2784.00 | 2795.39 | 2775.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 2770.30 | 2788.64 | 2778.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:00:00 | 2770.30 | 2788.64 | 2778.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 2760.00 | 2782.91 | 2777.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:00:00 | 2760.00 | 2782.91 | 2777.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 2794.30 | 2801.63 | 2791.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 10:45:00 | 2814.20 | 2803.03 | 2793.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:00:00 | 2815.90 | 2805.60 | 2795.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 09:30:00 | 2814.00 | 2812.48 | 2804.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 12:15:00 | 2787.40 | 2799.86 | 2799.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 12:15:00 | 2787.40 | 2799.86 | 2799.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 13:15:00 | 2782.80 | 2796.45 | 2798.31 | Break + close below crossover candle low |

### Cycle 125 — BUY (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 09:15:00 | 2831.10 | 2799.80 | 2799.07 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 11:15:00 | 2801.00 | 2806.32 | 2806.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 14:15:00 | 2786.80 | 2800.77 | 2803.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 2734.90 | 2720.49 | 2741.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 2734.90 | 2720.49 | 2741.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 2734.90 | 2720.49 | 2741.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:00:00 | 2707.30 | 2717.85 | 2738.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:45:00 | 2707.90 | 2714.79 | 2731.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:30:00 | 2704.40 | 2715.96 | 2728.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 14:45:00 | 2706.60 | 2717.26 | 2720.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 2720.00 | 2717.81 | 2720.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 2684.50 | 2717.81 | 2720.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2571.93 | 2645.43 | 2669.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2572.51 | 2645.43 | 2669.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2569.18 | 2645.43 | 2669.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2571.27 | 2645.43 | 2669.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 2550.28 | 2645.43 | 2669.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 10:15:00 | 2549.10 | 2548.92 | 2581.24 | SL hit (close>ema200) qty=0.50 sl=2548.92 alert=retest2 |

### Cycle 127 — BUY (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 13:15:00 | 2705.50 | 2600.23 | 2586.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 14:15:00 | 2772.70 | 2634.73 | 2603.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 2818.50 | 2828.87 | 2782.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 2818.50 | 2828.87 | 2782.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 2818.50 | 2828.87 | 2782.56 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 11:15:00 | 2756.10 | 2802.35 | 2808.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-17 12:15:00 | 2746.00 | 2791.08 | 2802.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 2766.60 | 2766.29 | 2784.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 2766.60 | 2766.29 | 2784.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 2766.60 | 2766.29 | 2784.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 12:30:00 | 2731.10 | 2756.14 | 2775.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 13:15:00 | 2594.54 | 2655.17 | 2706.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-20 10:15:00 | 2646.70 | 2635.46 | 2678.55 | SL hit (close>ema200) qty=0.50 sl=2635.46 alert=retest2 |

### Cycle 129 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 2671.60 | 2636.10 | 2634.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 2711.90 | 2655.72 | 2644.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 2687.90 | 2703.82 | 2676.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 2687.90 | 2703.82 | 2676.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 2687.90 | 2703.82 | 2676.32 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 2638.40 | 2664.64 | 2665.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 2599.00 | 2651.52 | 2659.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 14:15:00 | 2641.60 | 2621.88 | 2638.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 14:15:00 | 2641.60 | 2621.88 | 2638.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 2641.60 | 2621.88 | 2638.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 15:00:00 | 2641.60 | 2621.88 | 2638.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 2604.00 | 2618.30 | 2635.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 2656.10 | 2618.30 | 2635.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 2629.10 | 2620.46 | 2635.14 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 2676.10 | 2649.19 | 2645.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 15:15:00 | 2690.00 | 2662.07 | 2652.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 2608.40 | 2651.34 | 2648.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 2608.40 | 2651.34 | 2648.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 2608.40 | 2651.34 | 2648.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 2608.40 | 2651.34 | 2648.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 2622.80 | 2645.63 | 2646.04 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 2686.00 | 2646.36 | 2642.62 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 11:15:00 | 2625.60 | 2638.90 | 2640.46 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 2727.60 | 2653.35 | 2645.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 12:15:00 | 2793.00 | 2753.11 | 2727.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 2770.80 | 2772.33 | 2746.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 2770.80 | 2772.33 | 2746.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2770.80 | 2772.33 | 2746.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 2784.90 | 2772.33 | 2746.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-17 09:15:00 | 3063.39 | 3012.18 | 2934.81 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 14:15:00 | 3198.50 | 3222.39 | 3223.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 3173.30 | 3204.33 | 3213.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 3274.80 | 3210.25 | 3212.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 3274.80 | 3210.25 | 3212.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 3274.80 | 3210.25 | 3212.23 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 3278.60 | 3223.92 | 3218.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 3321.10 | 3253.45 | 3233.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 3383.60 | 3386.44 | 3346.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 13:45:00 | 3372.70 | 3386.44 | 3346.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 3327.30 | 3397.22 | 3363.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 3327.30 | 3397.22 | 3363.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 3294.30 | 3376.64 | 3357.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:00:00 | 3294.30 | 3376.64 | 3357.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 3430.80 | 3373.87 | 3360.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 10:15:00 | 3447.00 | 3397.88 | 3374.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 11:15:00 | 3473.80 | 3406.58 | 3380.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 3346.00 | 3374.44 | 3377.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 12:15:00 | 3346.00 | 3374.44 | 3377.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 15:15:00 | 3332.00 | 3355.78 | 3367.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 3376.00 | 3359.82 | 3367.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 3376.00 | 3359.82 | 3367.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 3376.00 | 3359.82 | 3367.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:30:00 | 3375.00 | 3359.82 | 3367.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 3367.00 | 3361.26 | 3367.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:45:00 | 3374.00 | 3361.26 | 3367.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 3372.00 | 3363.41 | 3368.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 12:00:00 | 3372.00 | 3363.41 | 3368.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 3355.30 | 3361.79 | 3367.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 13:30:00 | 3348.60 | 3359.43 | 3365.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 15:00:00 | 3349.50 | 3357.44 | 3364.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 10:00:00 | 3350.60 | 3354.08 | 3361.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 10:15:00 | 3391.00 | 3361.47 | 3363.92 | SL hit (close>static) qty=1.00 sl=3372.90 alert=retest2 |

### Cycle 139 — BUY (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 11:15:00 | 3427.00 | 3374.57 | 3369.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 12:15:00 | 3458.20 | 3391.30 | 3377.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 13:15:00 | 3390.00 | 3391.04 | 3378.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 14:00:00 | 3390.00 | 3391.04 | 3378.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 14:15:00 | 3408.20 | 3394.47 | 3381.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 14:45:00 | 3391.60 | 3394.47 | 3381.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-22 10:00:00 | 1704.45 | 2024-05-23 09:15:00 | 1694.60 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-05-22 13:00:00 | 1699.15 | 2024-05-23 09:15:00 | 1694.60 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2024-05-22 15:00:00 | 1703.50 | 2024-05-23 09:15:00 | 1694.60 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2024-06-18 12:45:00 | 1743.35 | 2024-06-26 14:15:00 | 1804.35 | STOP_HIT | 1.00 | 3.50% |
| SELL | retest2 | 2024-07-22 09:15:00 | 1679.00 | 2024-07-22 12:15:00 | 1699.90 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-07-25 12:00:00 | 1732.10 | 2024-07-31 14:15:00 | 1721.20 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-07-25 13:30:00 | 1745.00 | 2024-08-01 09:15:00 | 1727.40 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-07-29 15:15:00 | 1729.75 | 2024-08-01 09:15:00 | 1727.40 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2024-07-30 13:15:00 | 1729.15 | 2024-08-01 09:15:00 | 1727.40 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2024-07-31 09:30:00 | 1738.05 | 2024-08-01 09:15:00 | 1727.40 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2024-08-05 09:15:00 | 1655.90 | 2024-08-08 10:15:00 | 1699.35 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2024-08-06 11:00:00 | 1694.00 | 2024-08-08 10:15:00 | 1699.35 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2024-08-12 09:15:00 | 1674.20 | 2024-08-19 09:15:00 | 1666.90 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2024-08-13 12:45:00 | 1666.55 | 2024-08-19 09:15:00 | 1666.90 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2024-08-13 13:45:00 | 1669.00 | 2024-08-19 09:15:00 | 1666.90 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2024-08-13 14:30:00 | 1664.35 | 2024-08-19 09:15:00 | 1666.90 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2024-08-21 11:15:00 | 1712.15 | 2024-08-23 13:15:00 | 1797.76 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-08-21 11:45:00 | 1726.85 | 2024-08-23 14:15:00 | 1813.19 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-08-21 11:15:00 | 1712.15 | 2024-08-26 09:15:00 | 1883.37 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2024-08-21 11:45:00 | 1726.85 | 2024-08-26 09:15:00 | 1899.54 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-08-28 09:15:00 | 1831.20 | 2024-08-29 15:15:00 | 1811.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-08-29 12:15:00 | 1818.90 | 2024-08-29 15:15:00 | 1811.00 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2024-08-29 15:00:00 | 1818.85 | 2024-08-29 15:15:00 | 1811.00 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2024-09-06 09:15:00 | 2032.85 | 2024-09-09 09:15:00 | 1959.25 | STOP_HIT | 1.00 | -3.62% |
| BUY | retest2 | 2024-09-13 10:00:00 | 2038.45 | 2024-09-18 09:15:00 | 2242.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-09-26 13:00:00 | 2125.85 | 2024-09-30 15:15:00 | 2023.50 | PARTIAL | 0.50 | 4.81% |
| SELL | retest2 | 2024-09-26 15:15:00 | 2130.00 | 2024-09-30 15:15:00 | 2023.21 | PARTIAL | 0.50 | 5.01% |
| SELL | retest2 | 2024-09-27 11:00:00 | 2129.70 | 2024-10-01 09:15:00 | 2019.56 | PARTIAL | 0.50 | 5.17% |
| SELL | retest2 | 2024-09-26 13:00:00 | 2125.85 | 2024-10-01 12:15:00 | 2063.20 | STOP_HIT | 0.50 | 2.95% |
| SELL | retest2 | 2024-09-26 15:15:00 | 2130.00 | 2024-10-01 12:15:00 | 2063.20 | STOP_HIT | 0.50 | 3.14% |
| SELL | retest2 | 2024-09-27 11:00:00 | 2129.70 | 2024-10-01 12:15:00 | 2063.20 | STOP_HIT | 0.50 | 3.12% |
| SELL | retest2 | 2024-09-27 11:45:00 | 2108.65 | 2024-10-04 09:15:00 | 2003.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 11:45:00 | 2108.65 | 2024-10-04 14:15:00 | 2033.70 | STOP_HIT | 0.50 | 3.55% |
| SELL | retest2 | 2024-09-30 12:30:00 | 2064.65 | 2024-10-07 10:15:00 | 1961.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 15:00:00 | 2050.00 | 2024-10-07 10:15:00 | 1947.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 12:00:00 | 2060.00 | 2024-10-07 10:15:00 | 1957.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 12:45:00 | 2065.00 | 2024-10-07 10:15:00 | 1961.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 09:15:00 | 2052.70 | 2024-10-07 10:15:00 | 1950.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 12:30:00 | 2064.65 | 2024-10-07 14:15:00 | 2015.80 | STOP_HIT | 0.50 | 2.37% |
| SELL | retest2 | 2024-09-30 15:00:00 | 2050.00 | 2024-10-07 14:15:00 | 2015.80 | STOP_HIT | 0.50 | 1.67% |
| SELL | retest2 | 2024-10-01 12:00:00 | 2060.00 | 2024-10-07 14:15:00 | 2015.80 | STOP_HIT | 0.50 | 2.15% |
| SELL | retest2 | 2024-10-01 12:45:00 | 2065.00 | 2024-10-07 14:15:00 | 2015.80 | STOP_HIT | 0.50 | 2.38% |
| SELL | retest2 | 2024-10-03 09:15:00 | 2052.70 | 2024-10-07 14:15:00 | 2015.80 | STOP_HIT | 0.50 | 1.80% |
| BUY | retest2 | 2024-10-16 13:15:00 | 2193.90 | 2024-10-18 09:15:00 | 2138.55 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2024-10-16 15:00:00 | 2200.95 | 2024-10-18 09:15:00 | 2138.55 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2024-10-17 12:30:00 | 2192.70 | 2024-10-18 09:15:00 | 2138.55 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2024-11-06 11:00:00 | 2442.25 | 2024-11-07 15:15:00 | 2368.65 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2024-11-07 09:30:00 | 2429.00 | 2024-11-07 15:15:00 | 2368.65 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2024-11-07 10:30:00 | 2409.95 | 2024-11-07 15:15:00 | 2368.65 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest1 | 2024-11-13 09:15:00 | 2266.15 | 2024-11-19 09:15:00 | 2265.75 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2024-11-14 14:30:00 | 2257.20 | 2024-11-19 10:15:00 | 2282.15 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-11-14 15:15:00 | 2243.95 | 2024-11-19 10:15:00 | 2282.15 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-11-25 09:15:00 | 2372.00 | 2024-12-03 10:15:00 | 2356.85 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-12-05 12:15:00 | 2360.00 | 2024-12-09 11:15:00 | 2383.75 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-12-05 13:30:00 | 2362.25 | 2024-12-09 11:15:00 | 2383.75 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-12-06 09:15:00 | 2359.95 | 2024-12-09 11:15:00 | 2383.75 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-12-12 09:15:00 | 2454.80 | 2024-12-13 09:15:00 | 2397.70 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2024-12-20 11:45:00 | 2541.45 | 2024-12-20 14:15:00 | 2476.75 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-01-02 09:15:00 | 2590.05 | 2025-01-06 09:15:00 | 2550.00 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-01-02 11:45:00 | 2599.80 | 2025-01-06 09:15:00 | 2550.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-01-02 15:15:00 | 2592.00 | 2025-01-06 09:15:00 | 2550.00 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-01-24 09:15:00 | 2222.35 | 2025-01-27 10:15:00 | 2111.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:45:00 | 2220.65 | 2025-01-27 10:15:00 | 2109.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:15:00 | 2222.35 | 2025-01-28 12:15:00 | 2118.65 | STOP_HIT | 0.50 | 4.67% |
| SELL | retest2 | 2025-01-24 09:45:00 | 2220.65 | 2025-01-28 12:15:00 | 2118.65 | STOP_HIT | 0.50 | 4.59% |
| BUY | retest2 | 2025-02-05 15:15:00 | 2322.45 | 2025-02-06 10:15:00 | 2269.55 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-02-13 14:15:00 | 2123.35 | 2025-02-17 09:15:00 | 2017.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 15:00:00 | 2116.25 | 2025-02-17 09:15:00 | 2010.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 09:15:00 | 2109.00 | 2025-02-17 09:15:00 | 2003.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 14:15:00 | 2123.35 | 2025-02-17 13:15:00 | 1911.01 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-13 15:00:00 | 2116.25 | 2025-02-17 13:15:00 | 1904.62 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-14 09:15:00 | 2109.00 | 2025-02-18 13:15:00 | 1898.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-03 09:15:00 | 2021.45 | 2025-03-04 09:15:00 | 2069.60 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2025-03-04 09:15:00 | 2028.70 | 2025-03-04 09:15:00 | 2069.60 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-03-11 11:15:00 | 2188.95 | 2025-03-12 10:15:00 | 2163.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-03-17 09:15:00 | 2213.00 | 2025-03-28 14:15:00 | 2434.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-22 09:15:00 | 2467.00 | 2025-04-25 09:15:00 | 2394.40 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2025-05-26 13:15:00 | 2451.90 | 2025-05-29 10:15:00 | 2476.80 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-05-27 13:45:00 | 2448.50 | 2025-05-29 10:15:00 | 2476.80 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-06-10 11:45:00 | 2645.50 | 2025-06-16 11:15:00 | 2625.10 | STOP_HIT | 1.00 | 0.77% |
| SELL | retest2 | 2025-06-10 13:45:00 | 2646.60 | 2025-06-16 11:15:00 | 2625.10 | STOP_HIT | 1.00 | 0.81% |
| SELL | retest2 | 2025-06-11 09:15:00 | 2630.30 | 2025-06-16 11:15:00 | 2625.10 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-06-18 09:15:00 | 2631.00 | 2025-06-23 09:15:00 | 2638.60 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest2 | 2025-07-01 10:15:00 | 2601.10 | 2025-07-04 09:15:00 | 2634.80 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest1 | 2025-07-11 09:15:00 | 2702.10 | 2025-07-17 10:15:00 | 2738.40 | STOP_HIT | 1.00 | 1.34% |
| BUY | retest2 | 2025-07-24 09:15:00 | 2768.90 | 2025-07-24 09:15:00 | 2737.70 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-07-28 11:30:00 | 2701.00 | 2025-07-30 14:15:00 | 2711.30 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-07-28 12:45:00 | 2696.60 | 2025-07-30 15:15:00 | 2707.00 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-07-29 09:30:00 | 2700.10 | 2025-07-30 15:15:00 | 2707.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-07-29 12:45:00 | 2700.00 | 2025-07-30 15:15:00 | 2707.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-07-29 15:00:00 | 2682.40 | 2025-07-30 15:15:00 | 2707.00 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-08-05 09:15:00 | 2836.50 | 2025-08-08 10:15:00 | 2819.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-08-07 09:30:00 | 2844.70 | 2025-08-08 10:15:00 | 2819.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-08-26 12:00:00 | 2911.90 | 2025-08-28 11:15:00 | 2865.90 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-08-26 14:45:00 | 2913.20 | 2025-08-28 11:15:00 | 2865.90 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-09-10 10:15:00 | 2712.30 | 2025-09-10 13:15:00 | 2791.40 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-09-25 12:45:00 | 2969.20 | 2025-10-01 15:15:00 | 2918.60 | STOP_HIT | 1.00 | 1.70% |
| SELL | retest2 | 2025-09-25 15:00:00 | 2956.00 | 2025-10-01 15:15:00 | 2918.60 | STOP_HIT | 1.00 | 1.27% |
| BUY | retest2 | 2025-10-06 11:30:00 | 2968.20 | 2025-10-14 09:15:00 | 2945.80 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-10-06 12:15:00 | 2965.90 | 2025-10-14 09:15:00 | 2945.80 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-10-07 09:15:00 | 2973.00 | 2025-10-14 09:15:00 | 2945.80 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-10-09 11:45:00 | 2966.90 | 2025-10-14 09:15:00 | 2945.80 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-10-13 09:30:00 | 3026.40 | 2025-10-14 09:15:00 | 2945.80 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-10-29 11:15:00 | 3182.50 | 2025-11-03 13:15:00 | 3192.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-10-29 14:30:00 | 3166.20 | 2025-11-03 13:15:00 | 3192.00 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-11-03 11:00:00 | 3178.00 | 2025-11-03 13:15:00 | 3192.00 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-11-03 12:00:00 | 3179.20 | 2025-11-03 13:15:00 | 3192.00 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-11-28 15:15:00 | 3207.00 | 2025-12-03 09:15:00 | 3235.50 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-12-01 12:30:00 | 3207.80 | 2025-12-03 09:15:00 | 3235.50 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-12-01 14:30:00 | 3208.00 | 2025-12-03 09:15:00 | 3235.50 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-12-02 09:45:00 | 3209.40 | 2025-12-03 09:15:00 | 3235.50 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-12-08 11:30:00 | 3199.50 | 2025-12-10 09:15:00 | 3240.00 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-12-09 14:45:00 | 3200.00 | 2025-12-10 09:15:00 | 3240.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-12-18 14:15:00 | 3154.00 | 2025-12-19 11:15:00 | 3180.90 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-12-19 09:15:00 | 3150.50 | 2025-12-19 11:15:00 | 3180.90 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2026-01-13 13:15:00 | 2927.60 | 2026-01-20 09:15:00 | 2790.24 | PARTIAL | 0.50 | 4.69% |
| SELL | retest2 | 2026-01-13 14:30:00 | 2937.10 | 2026-01-20 10:15:00 | 2781.22 | PARTIAL | 0.50 | 5.31% |
| SELL | retest2 | 2026-01-13 15:00:00 | 2929.10 | 2026-01-20 10:15:00 | 2782.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 09:30:00 | 2909.40 | 2026-01-20 14:15:00 | 2763.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 15:00:00 | 2903.30 | 2026-01-20 14:15:00 | 2758.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 10:15:00 | 2896.00 | 2026-01-20 15:15:00 | 2751.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 13:15:00 | 2927.60 | 2026-01-21 11:15:00 | 2792.10 | STOP_HIT | 0.50 | 4.63% |
| SELL | retest2 | 2026-01-13 14:30:00 | 2937.10 | 2026-01-21 11:15:00 | 2792.10 | STOP_HIT | 0.50 | 4.94% |
| SELL | retest2 | 2026-01-13 15:00:00 | 2929.10 | 2026-01-21 11:15:00 | 2792.10 | STOP_HIT | 0.50 | 4.68% |
| SELL | retest2 | 2026-01-14 09:30:00 | 2909.40 | 2026-01-21 11:15:00 | 2792.10 | STOP_HIT | 0.50 | 4.03% |
| SELL | retest2 | 2026-01-14 15:00:00 | 2903.30 | 2026-01-21 11:15:00 | 2792.10 | STOP_HIT | 0.50 | 3.83% |
| SELL | retest2 | 2026-01-16 10:15:00 | 2896.00 | 2026-01-21 11:15:00 | 2792.10 | STOP_HIT | 0.50 | 3.59% |
| SELL | retest2 | 2026-02-03 10:30:00 | 2776.20 | 2026-02-03 14:15:00 | 2769.20 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2026-02-06 09:15:00 | 2738.30 | 2026-02-09 10:15:00 | 2782.20 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2026-02-12 10:45:00 | 2814.20 | 2026-02-13 12:15:00 | 2787.40 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2026-02-12 12:00:00 | 2815.90 | 2026-02-13 12:15:00 | 2787.40 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2026-02-13 09:30:00 | 2814.00 | 2026-02-13 12:15:00 | 2787.40 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-02-23 11:00:00 | 2707.30 | 2026-03-02 09:15:00 | 2571.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 13:45:00 | 2707.90 | 2026-03-02 09:15:00 | 2572.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:30:00 | 2704.40 | 2026-03-02 09:15:00 | 2569.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 14:45:00 | 2706.60 | 2026-03-02 09:15:00 | 2571.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 09:15:00 | 2684.50 | 2026-03-02 09:15:00 | 2550.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 11:00:00 | 2707.30 | 2026-03-05 10:15:00 | 2549.10 | STOP_HIT | 0.50 | 5.84% |
| SELL | retest2 | 2026-02-23 13:45:00 | 2707.90 | 2026-03-05 10:15:00 | 2549.10 | STOP_HIT | 0.50 | 5.86% |
| SELL | retest2 | 2026-02-24 09:30:00 | 2704.40 | 2026-03-05 10:15:00 | 2549.10 | STOP_HIT | 0.50 | 5.74% |
| SELL | retest2 | 2026-02-25 14:45:00 | 2706.60 | 2026-03-05 10:15:00 | 2549.10 | STOP_HIT | 0.50 | 5.82% |
| SELL | retest2 | 2026-02-26 09:15:00 | 2684.50 | 2026-03-05 10:15:00 | 2549.10 | STOP_HIT | 0.50 | 5.04% |
| SELL | retest2 | 2026-03-18 12:30:00 | 2731.10 | 2026-03-19 13:15:00 | 2594.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 12:30:00 | 2731.10 | 2026-03-20 10:15:00 | 2646.70 | STOP_HIT | 0.50 | 3.09% |
| BUY | retest2 | 2026-04-13 10:15:00 | 2784.90 | 2026-04-17 09:15:00 | 3063.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-05-04 10:15:00 | 3447.00 | 2026-05-05 12:15:00 | 3346.00 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2026-05-04 11:15:00 | 3473.80 | 2026-05-05 12:15:00 | 3346.00 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest2 | 2026-05-06 13:30:00 | 3348.60 | 2026-05-07 10:15:00 | 3391.00 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2026-05-06 15:00:00 | 3349.50 | 2026-05-07 10:15:00 | 3391.00 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-05-07 10:00:00 | 3350.60 | 2026-05-07 10:15:00 | 3391.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-05-07 10:45:00 | 3349.40 | 2026-05-07 11:15:00 | 3427.00 | STOP_HIT | 1.00 | -2.32% |
