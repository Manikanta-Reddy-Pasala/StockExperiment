# BAJAJFINSV (BAJAJFINSV)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1814.00
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
| ALERT2 | 11 |
| ALERT2_SKIP | 6 |
| ALERT3 | 81 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 63 |
| PARTIAL | 8 |
| TARGET_HIT | 10 |
| STOP_HIT | 52 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 70 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 24 / 46
- **Target hits / Stop hits / Partials:** 10 / 52 / 8
- **Avg / median % per leg:** 1.25% / -0.83%
- **Sum % (uncompounded):** 87.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 8 | 25.8% | 6 | 25 | 0 | 1.02% | 31.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 31 | 8 | 25.8% | 6 | 25 | 0 | 1.02% | 31.6% |
| SELL (all) | 39 | 16 | 41.0% | 4 | 27 | 8 | 1.43% | 55.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 39 | 16 | 41.0% | 4 | 27 | 8 | 1.43% | 55.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 70 | 24 | 34.3% | 10 | 52 | 8 | 1.25% | 87.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-19 13:15:00 | 1646.80 | 1592.23 | 1592.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 13:15:00 | 1665.20 | 1597.96 | 1595.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 09:15:00 | 1594.50 | 1606.06 | 1600.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 09:15:00 | 1594.50 | 1606.06 | 1600.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 1594.50 | 1606.06 | 1600.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 09:30:00 | 1590.65 | 1606.06 | 1600.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 10:15:00 | 1575.30 | 1605.76 | 1599.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 11:00:00 | 1575.30 | 1605.76 | 1599.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-08-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 13:15:00 | 1560.85 | 1594.84 | 1594.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 1545.50 | 1590.19 | 1592.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 10:15:00 | 1581.00 | 1580.17 | 1586.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-20 11:00:00 | 1581.00 | 1580.17 | 1586.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 1590.55 | 1580.27 | 1586.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:30:00 | 1590.75 | 1580.27 | 1586.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 1591.15 | 1580.38 | 1586.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 12:30:00 | 1591.45 | 1580.38 | 1586.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 13:15:00 | 1613.55 | 1580.71 | 1587.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 14:00:00 | 1613.55 | 1580.71 | 1587.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 1602.65 | 1580.93 | 1587.13 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 12:15:00 | 1673.30 | 1592.68 | 1592.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 13:15:00 | 1679.55 | 1593.54 | 1592.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-08 12:15:00 | 1850.75 | 1860.73 | 1782.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-08 13:00:00 | 1850.75 | 1860.73 | 1782.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 1811.00 | 1859.52 | 1800.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 10:30:00 | 1823.50 | 1859.14 | 1800.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 11:15:00 | 1821.95 | 1859.14 | 1800.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 11:15:00 | 1775.35 | 1855.34 | 1800.72 | SL hit (close<static) qty=1.00 sl=1793.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 10:15:00 | 1679.60 | 1770.44 | 1770.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 11:15:00 | 1675.50 | 1769.50 | 1770.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 09:15:00 | 1661.90 | 1661.20 | 1699.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-10 09:45:00 | 1663.35 | 1661.20 | 1699.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 1657.65 | 1620.86 | 1659.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 12:00:00 | 1657.65 | 1620.86 | 1659.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 12:15:00 | 1711.90 | 1621.76 | 1659.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 12:45:00 | 1712.40 | 1621.76 | 1659.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 13:15:00 | 1700.45 | 1622.55 | 1660.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 11:30:00 | 1687.45 | 1631.69 | 1662.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 12:15:00 | 1688.55 | 1631.69 | 1662.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 13:00:00 | 1688.00 | 1632.25 | 1662.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 14:15:00 | 1690.15 | 1636.61 | 1663.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 1694.25 | 1650.89 | 1667.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 09:30:00 | 1708.20 | 1650.89 | 1667.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-01-14 15:15:00 | 1727.00 | 1654.39 | 1668.89 | SL hit (close>static) qty=1.00 sl=1714.55 alert=retest2 |

### Cycle 5 — BUY (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 10:15:00 | 1738.50 | 1679.28 | 1679.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-28 11:15:00 | 1749.05 | 1685.54 | 1682.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 1693.40 | 1704.55 | 1693.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 1693.40 | 1704.55 | 1693.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1693.40 | 1704.55 | 1693.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 1693.40 | 1704.55 | 1693.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1751.55 | 1705.01 | 1693.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:15:00 | 1772.55 | 1705.01 | 1693.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:45:00 | 1762.55 | 1705.46 | 1693.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 15:15:00 | 1769.00 | 1705.46 | 1693.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-12 11:00:00 | 1760.85 | 1737.71 | 1714.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-02-27 10:15:00 | 1936.93 | 1804.07 | 1760.01 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 15:15:00 | 1942.00 | 1992.10 | 1992.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 1928.90 | 1991.47 | 1992.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 2010.90 | 1963.58 | 1976.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 2010.90 | 1963.58 | 1976.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 2010.90 | 1963.58 | 1976.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:45:00 | 2009.00 | 1963.58 | 1976.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 2003.90 | 1963.98 | 1976.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 13:45:00 | 2000.40 | 1965.19 | 1976.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 14:30:00 | 2000.10 | 1965.44 | 1976.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 15:00:00 | 2000.90 | 1960.54 | 1969.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 13:15:00 | 2017.60 | 1963.44 | 1971.11 | SL hit (close>static) qty=1.00 sl=2017.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 11:15:00 | 2036.30 | 1978.32 | 1978.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 12:15:00 | 2038.20 | 1978.92 | 1978.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 2015.00 | 2025.48 | 2006.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 10:00:00 | 2015.00 | 2025.48 | 2006.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 2001.40 | 2025.09 | 2006.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 12:00:00 | 2001.40 | 2025.09 | 2006.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 12:15:00 | 2004.50 | 2024.89 | 2006.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 09:30:00 | 2017.40 | 2024.05 | 2006.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 10:15:00 | 2012.70 | 2024.05 | 2006.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 11:15:00 | 1999.30 | 2023.59 | 2006.07 | SL hit (close<static) qty=1.00 sl=2000.50 alert=retest2 |

### Cycle 8 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 2009.50 | 2053.54 | 2053.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 1988.60 | 2051.66 | 2052.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 2073.20 | 2046.18 | 2049.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 2073.20 | 2046.18 | 2049.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 2073.20 | 2046.18 | 2049.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:00:00 | 2073.20 | 2046.18 | 2049.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 2062.40 | 2046.34 | 2049.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:30:00 | 2074.20 | 2046.34 | 2049.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 2049.70 | 2046.40 | 2049.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 14:15:00 | 2043.80 | 2046.40 | 2049.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 15:00:00 | 2044.50 | 2046.38 | 2049.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 2023.10 | 2046.38 | 2049.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 1941.61 | 2024.09 | 2035.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 1942.27 | 2024.09 | 2035.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 13:15:00 | 1921.94 | 2010.93 | 2027.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 2010.50 | 1986.60 | 2011.45 | SL hit (close>ema200) qty=0.50 sl=1986.60 alert=retest2 |

### Cycle 9 — BUY (started 2026-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 14:15:00 | 2049.70 | 2022.50 | 2022.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 2052.80 | 2023.00 | 2022.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 2016.60 | 2024.11 | 2023.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 2016.60 | 2024.11 | 2023.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 2016.60 | 2024.11 | 2023.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:45:00 | 2016.50 | 2024.11 | 2023.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 2020.60 | 2024.07 | 2023.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:15:00 | 2016.10 | 2024.07 | 2023.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 2017.00 | 2024.00 | 2023.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:30:00 | 2013.10 | 2024.00 | 2023.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1966.20 | 2022.51 | 2022.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 1946.20 | 2021.16 | 2021.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 1805.00 | 1783.43 | 1865.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 11:00:00 | 1805.00 | 1783.43 | 1865.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 1852.40 | 1797.68 | 1852.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:45:00 | 1852.00 | 1797.68 | 1852.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 1849.40 | 1798.19 | 1852.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:00:00 | 1849.40 | 1798.19 | 1852.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 1850.40 | 1798.71 | 1852.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 15:15:00 | 1850.10 | 1798.71 | 1852.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 1850.10 | 1799.22 | 1852.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:15:00 | 1854.00 | 1799.22 | 1852.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 1840.80 | 1799.64 | 1852.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 09:15:00 | 1827.80 | 1802.48 | 1852.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 14:15:00 | 1736.41 | 1793.15 | 1838.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 1791.30 | 1790.06 | 1834.08 | SL hit (close>ema200) qty=0.50 sl=1790.06 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-24 14:45:00 | 1602.75 | 2024-05-30 14:15:00 | 1522.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-27 09:15:00 | 1600.25 | 2024-05-30 14:15:00 | 1520.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-27 11:00:00 | 1602.95 | 2024-05-30 14:15:00 | 1522.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-27 11:30:00 | 1603.00 | 2024-05-30 14:15:00 | 1522.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-24 14:45:00 | 1602.75 | 2024-06-04 11:15:00 | 1442.48 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-27 09:15:00 | 1600.25 | 2024-06-04 11:15:00 | 1440.23 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-27 11:00:00 | 1602.95 | 2024-06-04 11:15:00 | 1442.65 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-27 11:30:00 | 1603.00 | 2024-06-04 11:15:00 | 1442.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-12 13:45:00 | 1582.50 | 2024-06-18 09:15:00 | 1597.45 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-06-12 15:00:00 | 1578.95 | 2024-06-18 09:15:00 | 1597.45 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-06-13 09:30:00 | 1584.00 | 2024-06-18 09:15:00 | 1597.45 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-06-13 10:30:00 | 1584.25 | 2024-06-18 09:15:00 | 1597.45 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-06-21 15:00:00 | 1572.80 | 2024-06-25 13:15:00 | 1604.75 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-06-24 09:15:00 | 1568.45 | 2024-06-25 13:15:00 | 1604.75 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2024-06-25 11:00:00 | 1573.55 | 2024-06-25 13:15:00 | 1604.75 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2024-07-02 09:15:00 | 1573.65 | 2024-07-03 12:15:00 | 1591.45 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-07-03 11:15:00 | 1581.25 | 2024-07-03 14:15:00 | 1594.95 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-07-05 09:15:00 | 1576.30 | 2024-07-12 09:15:00 | 1593.40 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-07-09 15:15:00 | 1581.00 | 2024-07-12 09:15:00 | 1593.40 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-07-10 10:00:00 | 1579.30 | 2024-07-12 09:15:00 | 1593.40 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-10-18 10:30:00 | 1823.50 | 2024-10-21 11:15:00 | 1775.35 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2024-10-18 11:15:00 | 1821.95 | 2024-10-21 11:15:00 | 1775.35 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-01-06 11:30:00 | 1687.45 | 2025-01-14 15:15:00 | 1727.00 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-01-06 12:15:00 | 1688.55 | 2025-01-14 15:15:00 | 1727.00 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-01-06 13:00:00 | 1688.00 | 2025-01-14 15:15:00 | 1727.00 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-01-07 14:15:00 | 1690.15 | 2025-01-14 15:15:00 | 1727.00 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-01-15 13:00:00 | 1670.00 | 2025-01-16 09:15:00 | 1684.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-01-15 13:30:00 | 1665.45 | 2025-01-16 09:15:00 | 1684.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-01-15 14:15:00 | 1670.35 | 2025-01-16 09:15:00 | 1684.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-02-01 14:15:00 | 1772.55 | 2025-02-27 10:15:00 | 1936.93 | TARGET_HIT | 1.00 | 9.27% |
| BUY | retest2 | 2025-02-01 14:45:00 | 1762.55 | 2025-03-25 09:15:00 | 1938.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-01 15:15:00 | 1769.00 | 2025-03-25 09:15:00 | 1945.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-12 11:00:00 | 1760.85 | 2025-03-25 10:15:00 | 1949.81 | TARGET_HIT | 1.00 | 10.73% |
| BUY | retest2 | 2025-03-05 10:30:00 | 1789.10 | 2025-03-26 09:15:00 | 1968.01 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-05 12:15:00 | 1788.35 | 2025-03-26 09:15:00 | 1967.18 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-18 13:45:00 | 2000.40 | 2025-09-05 13:15:00 | 2017.60 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-08-18 14:30:00 | 2000.10 | 2025-09-05 13:15:00 | 2017.60 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-09-04 15:00:00 | 2000.90 | 2025-09-05 13:15:00 | 2017.60 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-09-29 09:30:00 | 2017.40 | 2025-09-29 11:15:00 | 1999.30 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-09-29 10:15:00 | 2012.70 | 2025-09-29 11:15:00 | 1999.30 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-09-29 13:30:00 | 2009.90 | 2025-10-01 09:15:00 | 1983.50 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-09-29 15:00:00 | 2025.10 | 2025-10-01 09:15:00 | 1983.50 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-10-06 09:15:00 | 2021.40 | 2025-10-10 14:15:00 | 2003.40 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-10-08 12:45:00 | 2019.10 | 2025-10-10 14:15:00 | 2003.40 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-10-09 10:30:00 | 2016.20 | 2025-10-10 14:15:00 | 2003.40 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-10-10 12:30:00 | 2013.00 | 2025-10-10 14:15:00 | 2003.40 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-10-13 11:30:00 | 2013.90 | 2025-11-11 09:15:00 | 1978.70 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-10-14 13:00:00 | 2015.30 | 2025-11-11 09:15:00 | 1978.70 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-11-12 10:30:00 | 2013.10 | 2025-11-19 09:15:00 | 2042.90 | STOP_HIT | 1.00 | 1.48% |
| BUY | retest2 | 2025-11-12 11:45:00 | 2015.90 | 2025-11-19 09:15:00 | 2042.90 | STOP_HIT | 1.00 | 1.34% |
| BUY | retest2 | 2025-11-14 12:15:00 | 2055.00 | 2025-11-19 09:15:00 | 2042.90 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-11-14 13:45:00 | 2055.20 | 2025-11-19 09:15:00 | 2042.90 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-11-18 12:30:00 | 2056.30 | 2025-11-24 11:15:00 | 2038.60 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-11-18 14:00:00 | 2058.80 | 2025-12-03 10:15:00 | 2038.90 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-11-20 10:45:00 | 2065.70 | 2025-12-03 10:15:00 | 2038.90 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-11-26 12:00:00 | 2068.50 | 2025-12-03 10:15:00 | 2038.90 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-12-02 10:15:00 | 2066.70 | 2025-12-16 09:15:00 | 2046.90 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-12-02 14:15:00 | 2065.00 | 2025-12-16 09:15:00 | 2046.90 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-12-11 14:15:00 | 2069.60 | 2025-12-16 09:15:00 | 2046.90 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-12-12 09:15:00 | 2076.70 | 2025-12-29 12:15:00 | 2009.50 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2025-12-15 11:15:00 | 2068.90 | 2025-12-29 12:15:00 | 2009.50 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2026-01-06 14:15:00 | 2043.80 | 2026-01-21 10:15:00 | 1941.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 15:00:00 | 2044.50 | 2026-01-21 10:15:00 | 1942.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 09:15:00 | 2023.10 | 2026-01-27 13:15:00 | 1921.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 14:15:00 | 2043.80 | 2026-02-03 09:15:00 | 2010.50 | STOP_HIT | 0.50 | 1.63% |
| SELL | retest2 | 2026-01-06 15:00:00 | 2044.50 | 2026-02-03 09:15:00 | 2010.50 | STOP_HIT | 0.50 | 1.66% |
| SELL | retest2 | 2026-01-07 09:15:00 | 2023.10 | 2026-02-03 09:15:00 | 2010.50 | STOP_HIT | 0.50 | 0.62% |
| SELL | retest2 | 2026-02-11 11:45:00 | 2039.00 | 2026-02-16 15:15:00 | 2053.10 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2026-04-23 09:15:00 | 1827.80 | 2026-04-30 14:15:00 | 1736.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-23 09:15:00 | 1827.80 | 2026-05-05 12:15:00 | 1791.30 | STOP_HIT | 0.50 | 2.00% |
