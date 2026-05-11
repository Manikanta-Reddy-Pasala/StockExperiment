# C.E. Info Systems Ltd. (MAPMYINDIA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 957.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 45 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 37 |
| PARTIAL | 10 |
| TARGET_HIT | 10 |
| STOP_HIT | 27 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 47 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 26
- **Target hits / Stop hits / Partials:** 10 / 27 / 10
- **Avg / median % per leg:** 2.25% / -1.36%
- **Sum % (uncompounded):** 105.90%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 1 | 7.7% | 1 | 12 | 0 | -1.26% | -16.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 1 | 7.7% | 1 | 12 | 0 | -1.26% | -16.4% |
| SELL (all) | 34 | 20 | 58.8% | 9 | 15 | 10 | 3.60% | 122.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 34 | 20 | 58.8% | 9 | 15 | 10 | 3.60% | 122.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 47 | 21 | 44.7% | 10 | 27 | 10 | 2.25% | 105.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 10:15:00 | 2079.30 | 2193.67 | 2194.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 14:15:00 | 2068.80 | 2189.05 | 2191.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 09:15:00 | 2118.55 | 2111.30 | 2142.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-23 10:00:00 | 2118.55 | 2111.30 | 2142.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 2145.90 | 2104.16 | 2134.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:00:00 | 2145.90 | 2104.16 | 2134.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 2149.20 | 2104.61 | 2134.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:45:00 | 2148.45 | 2104.61 | 2134.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 11:15:00 | 2160.55 | 2105.16 | 2134.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:45:00 | 2162.70 | 2105.16 | 2134.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 2116.10 | 2107.69 | 2135.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:30:00 | 2136.80 | 2107.69 | 2135.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 2144.25 | 2108.23 | 2135.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 13:00:00 | 2144.25 | 2108.23 | 2135.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 2171.45 | 2108.86 | 2135.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 14:00:00 | 2171.45 | 2108.86 | 2135.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 2174.65 | 2120.20 | 2138.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 10:00:00 | 2174.65 | 2120.20 | 2138.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 2197.95 | 2120.97 | 2139.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 11:00:00 | 2197.95 | 2120.97 | 2139.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 2142.15 | 2119.80 | 2136.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:30:00 | 2144.00 | 2119.80 | 2136.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 2132.55 | 2119.93 | 2136.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 12:15:00 | 2119.85 | 2119.94 | 2136.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 13:00:00 | 2115.20 | 2119.89 | 2136.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 09:15:00 | 2117.35 | 2118.42 | 2135.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 10:45:00 | 2111.60 | 2118.49 | 2135.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 2120.00 | 2118.50 | 2134.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 12:00:00 | 2120.00 | 2118.50 | 2134.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 13:15:00 | 2139.70 | 2118.84 | 2134.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 14:00:00 | 2139.70 | 2118.84 | 2134.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 2132.80 | 2118.98 | 2134.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 15:15:00 | 2125.05 | 2118.98 | 2134.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 2125.05 | 2119.04 | 2134.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:15:00 | 2141.45 | 2119.04 | 2134.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 2110.10 | 2118.95 | 2134.78 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-14 10:15:00 | 2148.75 | 2119.25 | 2134.85 | SL hit (close>static) qty=1.00 sl=2145.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 14:15:00 | 1750.70 | 1677.87 | 1677.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 15:15:00 | 1775.50 | 1678.84 | 1678.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 11:15:00 | 1910.00 | 1913.13 | 1844.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 12:00:00 | 1910.00 | 1913.13 | 1844.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1796.00 | 1914.61 | 1859.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:00:00 | 1796.00 | 1914.61 | 1859.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 09:15:00 | 1744.80 | 1822.33 | 1822.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 14:15:00 | 1736.90 | 1807.43 | 1814.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 14:15:00 | 1798.00 | 1793.76 | 1805.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-11 15:00:00 | 1798.00 | 1793.76 | 1805.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 1812.70 | 1793.42 | 1804.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 15:00:00 | 1812.70 | 1793.42 | 1804.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 1808.50 | 1793.57 | 1804.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:15:00 | 1801.00 | 1793.57 | 1804.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 1807.00 | 1793.38 | 1804.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 15:00:00 | 1807.00 | 1793.38 | 1804.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 1797.00 | 1793.42 | 1804.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 1785.70 | 1794.65 | 1804.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 11:30:00 | 1786.40 | 1794.36 | 1804.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 14:30:00 | 1787.60 | 1794.31 | 1803.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 15:15:00 | 1825.10 | 1794.62 | 1804.04 | SL hit (close>static) qty=1.00 sl=1809.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 1818.10 | 1722.19 | 1721.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 1854.00 | 1729.18 | 1725.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 15:15:00 | 1760.40 | 1763.45 | 1747.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-07 09:15:00 | 1753.20 | 1763.45 | 1747.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 1752.10 | 1763.34 | 1747.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 11:30:00 | 1764.70 | 1763.31 | 1747.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1734.00 | 1767.58 | 1750.34 | SL hit (close<static) qty=1.00 sl=1741.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 09:15:00 | 1700.20 | 1737.93 | 1738.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 14:15:00 | 1679.00 | 1735.89 | 1737.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 14:15:00 | 1729.40 | 1712.98 | 1724.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 14:15:00 | 1729.40 | 1712.98 | 1724.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 1729.40 | 1712.98 | 1724.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 09:15:00 | 1688.90 | 1713.26 | 1723.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 14:00:00 | 1688.00 | 1711.98 | 1722.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 1692.20 | 1683.81 | 1699.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 10:15:00 | 1690.40 | 1683.92 | 1699.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 1677.00 | 1684.01 | 1699.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:45:00 | 1681.50 | 1684.01 | 1699.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 1719.00 | 1682.84 | 1698.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 1719.00 | 1682.84 | 1698.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 1720.00 | 1683.21 | 1698.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:45:00 | 1721.20 | 1683.21 | 1698.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 1723.00 | 1683.94 | 1698.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:30:00 | 1714.60 | 1683.94 | 1698.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 1735.00 | 1684.45 | 1698.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:45:00 | 1734.50 | 1684.45 | 1698.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 1695.80 | 1686.50 | 1699.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:45:00 | 1696.00 | 1686.50 | 1699.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 1712.00 | 1686.86 | 1699.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:45:00 | 1712.50 | 1686.86 | 1699.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 1728.40 | 1687.28 | 1699.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 1728.40 | 1687.28 | 1699.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 1715.20 | 1687.55 | 1699.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 1711.80 | 1687.55 | 1699.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 13:15:00 | 1712.70 | 1688.46 | 1699.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 14:15:00 | 1740.00 | 1689.19 | 1700.15 | SL hit (close>static) qty=1.00 sl=1730.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-13 14:45:00 | 1928.30 | 2024-05-30 09:15:00 | 1900.05 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-05-14 09:15:00 | 1995.75 | 2024-05-30 09:15:00 | 1900.05 | STOP_HIT | 1.00 | -4.80% |
| BUY | retest2 | 2024-05-29 09:15:00 | 1922.15 | 2024-05-30 09:15:00 | 1900.05 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-05-29 10:30:00 | 1937.00 | 2024-05-30 09:15:00 | 1900.05 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2024-05-29 12:45:00 | 1936.90 | 2024-06-04 09:15:00 | 1883.60 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2024-05-29 13:15:00 | 1936.05 | 2024-06-04 09:15:00 | 1883.60 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2024-05-29 14:15:00 | 1935.00 | 2024-06-04 09:15:00 | 1883.60 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2024-05-30 11:30:00 | 1919.95 | 2024-06-04 09:15:00 | 1883.60 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-05-31 10:15:00 | 1915.35 | 2024-06-04 09:15:00 | 1883.60 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-05-31 15:00:00 | 1921.25 | 2024-06-04 09:15:00 | 1883.60 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2024-06-03 14:30:00 | 1916.75 | 2024-06-04 09:15:00 | 1883.60 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-06-06 09:15:00 | 1928.25 | 2024-06-20 09:15:00 | 2121.08 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-09 12:15:00 | 2119.85 | 2024-10-14 10:15:00 | 2148.75 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-10-09 13:00:00 | 2115.20 | 2024-10-14 10:15:00 | 2148.75 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-10-11 09:15:00 | 2117.35 | 2024-10-14 10:15:00 | 2148.75 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-10-11 10:45:00 | 2111.60 | 2024-10-14 10:15:00 | 2148.75 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-10-22 09:15:00 | 2101.95 | 2024-10-25 09:15:00 | 1996.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 09:15:00 | 2101.95 | 2024-10-29 13:15:00 | 1891.75 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-18 09:15:00 | 1785.70 | 2025-07-18 15:15:00 | 1825.10 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-07-18 11:30:00 | 1786.40 | 2025-07-18 15:15:00 | 1825.10 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-07-18 14:30:00 | 1787.60 | 2025-07-18 15:15:00 | 1825.10 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-08-01 14:15:00 | 1787.20 | 2025-08-11 14:15:00 | 1811.10 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-08-08 15:00:00 | 1767.30 | 2025-08-11 14:15:00 | 1811.10 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-08-11 13:30:00 | 1774.80 | 2025-08-11 14:15:00 | 1811.10 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-08-11 14:00:00 | 1774.80 | 2025-08-11 14:15:00 | 1811.10 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-08-13 15:00:00 | 1774.10 | 2025-08-26 09:15:00 | 1685.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-13 15:00:00 | 1774.10 | 2025-10-03 14:15:00 | 1666.90 | STOP_HIT | 0.50 | 6.04% |
| SELL | retest2 | 2025-10-08 15:15:00 | 1670.20 | 2025-10-10 09:15:00 | 1706.00 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-11-07 11:30:00 | 1764.70 | 2025-11-11 09:15:00 | 1734.00 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-12-04 09:15:00 | 1688.90 | 2026-01-01 14:15:00 | 1740.00 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2025-12-04 14:00:00 | 1688.00 | 2026-01-01 14:15:00 | 1740.00 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-12-26 09:15:00 | 1692.20 | 2026-01-09 09:15:00 | 1627.73 | PARTIAL | 0.50 | 3.81% |
| SELL | retest2 | 2025-12-26 10:15:00 | 1690.40 | 2026-01-09 09:15:00 | 1626.68 | PARTIAL | 0.50 | 3.77% |
| SELL | retest2 | 2026-01-01 09:15:00 | 1711.80 | 2026-01-09 11:15:00 | 1607.59 | PARTIAL | 0.50 | 6.09% |
| SELL | retest2 | 2026-01-01 13:15:00 | 1712.70 | 2026-01-09 11:15:00 | 1606.26 | PARTIAL | 0.50 | 6.21% |
| SELL | retest2 | 2026-01-02 09:15:00 | 1713.40 | 2026-01-09 12:15:00 | 1604.45 | PARTIAL | 0.50 | 6.36% |
| SELL | retest2 | 2026-01-02 11:00:00 | 1712.30 | 2026-01-09 12:15:00 | 1603.60 | PARTIAL | 0.50 | 6.35% |
| SELL | retest2 | 2026-01-07 09:30:00 | 1690.80 | 2026-01-09 12:15:00 | 1605.88 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2026-01-08 09:15:00 | 1682.80 | 2026-01-09 12:15:00 | 1598.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-26 09:15:00 | 1692.20 | 2026-01-14 10:15:00 | 1542.06 | TARGET_HIT | 0.50 | 8.87% |
| SELL | retest2 | 2025-12-26 10:15:00 | 1690.40 | 2026-01-14 11:15:00 | 1541.07 | TARGET_HIT | 0.50 | 8.83% |
| SELL | retest2 | 2026-01-01 09:15:00 | 1711.80 | 2026-01-14 14:15:00 | 1522.98 | TARGET_HIT | 0.50 | 11.03% |
| SELL | retest2 | 2026-01-01 13:15:00 | 1712.70 | 2026-01-14 14:15:00 | 1521.36 | TARGET_HIT | 0.50 | 11.17% |
| SELL | retest2 | 2026-01-02 09:15:00 | 1713.40 | 2026-01-14 14:15:00 | 1521.72 | TARGET_HIT | 0.50 | 11.19% |
| SELL | retest2 | 2026-01-02 11:00:00 | 1712.30 | 2026-01-16 10:15:00 | 1520.01 | TARGET_HIT | 0.50 | 11.23% |
| SELL | retest2 | 2026-01-07 09:30:00 | 1690.80 | 2026-01-16 10:15:00 | 1519.20 | TARGET_HIT | 0.50 | 10.15% |
| SELL | retest2 | 2026-01-08 09:15:00 | 1682.80 | 2026-01-16 10:15:00 | 1514.52 | TARGET_HIT | 0.50 | 10.00% |
