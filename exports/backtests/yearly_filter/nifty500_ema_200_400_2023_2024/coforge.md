# Coforge Ltd. (COFORGE)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 1365.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 10 |
| ALERT2_SKIP | 1 |
| ALERT3 | 37 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 33 |
| PARTIAL | 3 |
| TARGET_HIT | 7 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 25
- **Target hits / Stop hits / Partials:** 7 / 26 / 3
- **Avg / median % per leg:** 0.88% / -1.73%
- **Sum % (uncompounded):** 31.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 5 | 23.8% | 5 | 16 | 0 | 0.92% | 19.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 5 | 23.8% | 5 | 16 | 0 | 0.92% | 19.4% |
| SELL (all) | 15 | 6 | 40.0% | 2 | 10 | 3 | 0.82% | 12.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 15 | 6 | 40.0% | 2 | 10 | 3 | 0.82% | 12.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 36 | 11 | 30.6% | 7 | 26 | 3 | 0.88% | 31.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-21 10:15:00 | 1140.00 | 1241.47 | 1241.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-21 11:15:00 | 1132.33 | 1240.38 | 1241.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 12:15:00 | 989.80 | 989.63 | 1060.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-22 13:00:00 | 989.80 | 989.63 | 1060.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 11:15:00 | 1056.40 | 995.10 | 1056.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 11:30:00 | 1055.65 | 995.10 | 1056.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 12:15:00 | 1057.54 | 995.72 | 1056.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 13:15:00 | 1059.95 | 995.72 | 1056.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 13:15:00 | 1056.62 | 996.32 | 1056.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 13:30:00 | 1060.10 | 996.32 | 1056.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 1049.58 | 997.96 | 1056.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 12:15:00 | 1035.06 | 998.94 | 1056.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 11:15:00 | 983.31 | 1001.17 | 1048.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-04 14:15:00 | 1001.20 | 1000.99 | 1047.47 | SL hit (close>ema200) qty=0.50 sl=1000.99 alert=retest2 |

### Cycle 2 — BUY (started 2024-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 09:15:00 | 1170.89 | 1061.77 | 1061.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 10:15:00 | 1173.94 | 1062.88 | 1062.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 09:15:00 | 1157.71 | 1186.46 | 1142.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-05 10:00:00 | 1157.71 | 1186.46 | 1142.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 14:15:00 | 1740.07 | 1843.97 | 1742.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 14:30:00 | 1741.85 | 1843.97 | 1742.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 15:15:00 | 1744.15 | 1842.98 | 1742.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-14 09:30:00 | 1729.99 | 1841.94 | 1742.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 1756.34 | 1841.09 | 1742.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 09:15:00 | 1781.45 | 1827.45 | 1740.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-17 12:15:00 | 1730.61 | 1818.91 | 1741.16 | SL hit (close<static) qty=1.00 sl=1731.59 alert=retest2 |

### Cycle 3 — SELL (started 2025-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 15:15:00 | 1573.60 | 1713.52 | 1714.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 09:15:00 | 1551.34 | 1711.90 | 1713.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 10:15:00 | 1592.20 | 1591.89 | 1641.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-05 11:00:00 | 1592.20 | 1591.89 | 1641.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 1578.60 | 1546.11 | 1595.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 09:30:00 | 1593.59 | 1546.11 | 1595.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 1601.60 | 1546.98 | 1595.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:00:00 | 1601.60 | 1546.98 | 1595.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 1610.33 | 1547.61 | 1595.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:45:00 | 1613.60 | 1547.61 | 1595.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 1607.45 | 1549.65 | 1595.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 10:15:00 | 1589.80 | 1562.22 | 1597.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-03 09:15:00 | 1510.31 | 1560.30 | 1594.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-04-03 14:15:00 | 1430.82 | 1554.37 | 1590.17 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 15:15:00 | 1684.50 | 1530.34 | 1530.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 1696.10 | 1540.71 | 1535.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 15:15:00 | 1856.00 | 1856.70 | 1780.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 09:15:00 | 1858.60 | 1856.70 | 1780.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 1698.60 | 1855.79 | 1790.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:00:00 | 1698.60 | 1855.79 | 1790.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 1702.90 | 1854.27 | 1789.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 09:15:00 | 1723.00 | 1846.13 | 1787.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 10:15:00 | 1679.80 | 1843.05 | 1786.20 | SL hit (close<static) qty=1.00 sl=1683.70 alert=retest2 |

### Cycle 5 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 1637.60 | 1752.48 | 1752.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 11:15:00 | 1630.40 | 1751.26 | 1751.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 12:15:00 | 1719.80 | 1719.09 | 1734.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 13:00:00 | 1719.80 | 1719.09 | 1734.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 1728.90 | 1718.59 | 1733.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 1728.90 | 1718.59 | 1733.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1743.00 | 1718.89 | 1733.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 1719.50 | 1725.71 | 1735.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 13:15:00 | 1750.00 | 1726.09 | 1735.35 | SL hit (close>static) qty=1.00 sl=1748.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 10:15:00 | 1810.70 | 1737.87 | 1737.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 13:15:00 | 1816.00 | 1740.10 | 1738.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 09:15:00 | 1732.60 | 1745.83 | 1741.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 1732.60 | 1745.83 | 1741.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 1732.60 | 1745.83 | 1741.98 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 10:15:00 | 1636.10 | 1737.68 | 1738.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 11:15:00 | 1629.70 | 1736.61 | 1737.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 14:15:00 | 1686.00 | 1678.07 | 1703.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-07 14:45:00 | 1682.50 | 1678.07 | 1703.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1708.20 | 1678.45 | 1703.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:30:00 | 1726.40 | 1678.45 | 1703.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1695.40 | 1678.62 | 1703.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:45:00 | 1690.70 | 1688.01 | 1704.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 1750.00 | 1689.12 | 1705.01 | SL hit (close>static) qty=1.00 sl=1710.20 alert=retest2 |

### Cycle 8 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 1832.10 | 1717.88 | 1717.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 11:15:00 | 1839.70 | 1765.84 | 1747.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 11:15:00 | 1847.10 | 1852.34 | 1807.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 11:45:00 | 1850.40 | 1852.34 | 1807.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 1797.70 | 1851.57 | 1818.78 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 12:15:00 | 1661.90 | 1793.23 | 1793.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 14:15:00 | 1660.20 | 1790.61 | 1792.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 09:15:00 | 1739.20 | 1736.04 | 1758.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-19 10:15:00 | 1732.20 | 1736.04 | 1758.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1706.10 | 1700.14 | 1730.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 1607.10 | 1701.16 | 1730.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 12:15:00 | 1526.74 | 1656.88 | 1700.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-12 09:15:00 | 1446.39 | 1650.88 | 1697.00 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-08-24 11:45:00 | 1032.02 | 2023-09-06 09:15:00 | 1135.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-29 13:15:00 | 1032.80 | 2023-09-06 09:15:00 | 1136.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-29 12:15:00 | 1030.80 | 2023-10-13 10:15:00 | 1009.73 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2023-10-03 11:00:00 | 1034.80 | 2023-10-13 10:15:00 | 1009.73 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2023-10-04 15:00:00 | 1028.31 | 2023-10-13 10:15:00 | 1009.73 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2023-10-11 13:00:00 | 1029.71 | 2023-10-20 09:15:00 | 1011.40 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2023-10-13 09:15:00 | 1024.51 | 2023-11-08 13:15:00 | 1016.31 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2023-10-19 09:30:00 | 1031.01 | 2023-11-08 13:15:00 | 1016.31 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2023-11-06 09:15:00 | 1023.40 | 2023-11-08 13:15:00 | 1016.31 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2023-11-06 12:45:00 | 1021.80 | 2023-11-13 09:15:00 | 1012.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2023-11-08 09:15:00 | 1025.94 | 2023-11-13 09:15:00 | 1012.00 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2023-11-08 15:00:00 | 1022.99 | 2023-11-16 12:15:00 | 1133.88 | TARGET_HIT | 1.00 | 10.84% |
| BUY | retest2 | 2023-11-10 14:00:00 | 1024.19 | 2023-11-16 14:15:00 | 1138.28 | TARGET_HIT | 1.00 | 11.14% |
| BUY | retest2 | 2023-11-15 09:15:00 | 1040.67 | 2023-11-16 14:15:00 | 1144.74 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-05-28 12:15:00 | 1035.06 | 2024-06-04 11:15:00 | 983.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-28 12:15:00 | 1035.06 | 2024-06-04 14:15:00 | 1001.20 | STOP_HIT | 0.50 | 3.27% |
| SELL | retest2 | 2024-06-06 13:00:00 | 1037.96 | 2024-06-07 09:15:00 | 1087.08 | STOP_HIT | 1.00 | -4.73% |
| SELL | retest2 | 2024-06-06 14:30:00 | 1037.89 | 2024-06-07 09:15:00 | 1087.08 | STOP_HIT | 1.00 | -4.74% |
| SELL | retest2 | 2024-06-11 15:00:00 | 1035.71 | 2024-06-13 09:15:00 | 1054.80 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-06-12 13:45:00 | 1037.21 | 2024-06-13 09:15:00 | 1054.80 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-06-12 15:00:00 | 1036.60 | 2024-06-18 14:15:00 | 1054.52 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-06-18 10:00:00 | 1035.61 | 2024-06-19 12:15:00 | 1072.11 | STOP_HIT | 1.00 | -3.52% |
| BUY | retest2 | 2025-01-16 09:15:00 | 1781.45 | 2025-01-17 12:15:00 | 1730.61 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2025-01-23 09:15:00 | 1786.54 | 2025-01-28 09:15:00 | 1731.39 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2025-01-27 15:00:00 | 1764.00 | 2025-01-28 09:15:00 | 1731.39 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-04-01 10:15:00 | 1589.80 | 2025-04-03 09:15:00 | 1510.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 10:15:00 | 1589.80 | 2025-04-03 14:15:00 | 1430.82 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-25 09:15:00 | 1723.00 | 2025-07-25 10:15:00 | 1679.80 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-07-28 09:45:00 | 1719.00 | 2025-08-06 09:15:00 | 1667.40 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2025-07-28 11:15:00 | 1718.20 | 2025-08-06 09:15:00 | 1667.40 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2025-07-29 09:30:00 | 1720.40 | 2025-08-06 09:15:00 | 1667.40 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-08-28 09:15:00 | 1719.50 | 2025-08-28 13:15:00 | 1750.00 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-09-04 09:30:00 | 1711.90 | 2025-09-10 09:15:00 | 1754.20 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-10-14 11:45:00 | 1690.70 | 2025-10-15 09:15:00 | 1750.00 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2026-02-04 09:15:00 | 1607.10 | 2026-02-11 12:15:00 | 1526.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-04 09:15:00 | 1607.10 | 2026-02-12 09:15:00 | 1446.39 | TARGET_HIT | 0.50 | 10.00% |
