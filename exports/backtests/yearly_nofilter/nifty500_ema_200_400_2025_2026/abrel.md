# Aditya Birla Real Estate Ltd. (ABREL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1479.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 30 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 32 |
| PARTIAL | 17 |
| TARGET_HIT | 8 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 49 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 19
- **Target hits / Stop hits / Partials:** 8 / 24 / 17
- **Avg / median % per leg:** 2.86% / 2.45%
- **Sum % (uncompounded):** 140.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.50% | -6.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.50% | -6.0% |
| SELL (all) | 45 | 30 | 66.7% | 8 | 20 | 17 | 3.25% | 146.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 45 | 30 | 66.7% | 8 | 20 | 17 | 3.25% | 146.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 49 | 30 | 61.2% | 8 | 24 | 17 | 2.86% | 140.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 13:15:00 | 2190.90 | 2000.02 | 1999.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 14:15:00 | 2196.20 | 2001.98 | 2000.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 2315.70 | 2333.30 | 2235.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 11:45:00 | 2314.20 | 2333.30 | 2235.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 2223.00 | 2327.98 | 2239.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:45:00 | 2224.70 | 2327.98 | 2239.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 2220.40 | 2326.90 | 2239.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:00:00 | 2220.40 | 2326.90 | 2239.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 2240.00 | 2325.14 | 2239.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 2240.00 | 2325.14 | 2239.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 2242.90 | 2324.33 | 2239.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 2267.60 | 2324.33 | 2239.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 10:00:00 | 2249.80 | 2323.58 | 2239.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 11:15:00 | 2249.80 | 2322.82 | 2239.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 12:15:00 | 2217.50 | 2320.96 | 2239.55 | SL hit (close<static) qty=1.00 sl=2237.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 15:15:00 | 1953.80 | 2196.39 | 2197.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 10:15:00 | 1924.70 | 2174.70 | 2186.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 14:15:00 | 1865.90 | 1852.12 | 1940.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 14:45:00 | 1871.30 | 1852.12 | 1940.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 1913.90 | 1857.09 | 1939.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 14:00:00 | 1909.30 | 1864.55 | 1938.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 15:15:00 | 1900.00 | 1865.05 | 1938.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 14:00:00 | 1908.00 | 1867.85 | 1937.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:30:00 | 1907.40 | 1870.03 | 1936.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 10:15:00 | 1813.83 | 1866.88 | 1929.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 10:15:00 | 1812.60 | 1866.88 | 1929.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 10:15:00 | 1812.03 | 1866.88 | 1929.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 11:15:00 | 1805.00 | 1866.15 | 1928.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-09-29 11:15:00 | 1718.37 | 1851.78 | 1917.01 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 13:15:00 | 1546.40 | 1344.76 | 1344.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 14:15:00 | 1584.80 | 1396.96 | 1372.83 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-16 09:30:00 | 2042.50 | 2025-05-16 10:15:00 | 2090.40 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-07-09 09:15:00 | 2267.60 | 2025-07-09 12:15:00 | 2217.50 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-07-09 10:00:00 | 2249.80 | 2025-07-09 12:15:00 | 2217.50 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-07-09 11:15:00 | 2249.80 | 2025-07-09 12:15:00 | 2217.50 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-07-15 10:30:00 | 2250.00 | 2025-07-15 13:15:00 | 2229.30 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-09-18 14:00:00 | 1909.30 | 2025-09-25 10:15:00 | 1813.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 15:15:00 | 1900.00 | 2025-09-25 10:15:00 | 1812.60 | PARTIAL | 0.50 | 4.60% |
| SELL | retest2 | 2025-09-19 14:00:00 | 1908.00 | 2025-09-25 10:15:00 | 1812.03 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2025-09-22 13:30:00 | 1907.40 | 2025-09-25 11:15:00 | 1805.00 | PARTIAL | 0.50 | 5.37% |
| SELL | retest2 | 2025-09-18 14:00:00 | 1909.30 | 2025-09-29 11:15:00 | 1718.37 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-18 15:15:00 | 1900.00 | 2025-09-29 11:15:00 | 1710.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-19 14:00:00 | 1908.00 | 2025-09-29 11:15:00 | 1717.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-22 13:30:00 | 1907.40 | 2025-09-29 11:15:00 | 1716.66 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-07 14:45:00 | 1771.50 | 2025-11-13 09:15:00 | 1802.20 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-11-10 10:00:00 | 1766.40 | 2025-11-13 10:15:00 | 1814.30 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-11-10 10:30:00 | 1768.60 | 2025-11-13 10:15:00 | 1814.30 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2025-11-10 11:45:00 | 1767.30 | 2025-11-13 10:15:00 | 1814.30 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-11-12 13:30:00 | 1775.10 | 2025-11-13 10:15:00 | 1814.30 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-11-13 12:30:00 | 1778.60 | 2025-11-19 10:15:00 | 1808.60 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-11-19 11:15:00 | 1775.00 | 2025-11-19 12:15:00 | 1800.90 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-11-20 10:00:00 | 1780.70 | 2025-12-05 10:15:00 | 1694.23 | PARTIAL | 0.50 | 4.86% |
| SELL | retest2 | 2025-11-20 10:00:00 | 1780.70 | 2025-12-05 13:15:00 | 1758.90 | STOP_HIT | 0.50 | 1.22% |
| SELL | retest2 | 2025-11-20 12:30:00 | 1780.00 | 2025-12-08 09:15:00 | 1691.66 | PARTIAL | 0.50 | 4.96% |
| SELL | retest2 | 2025-12-02 10:30:00 | 1783.40 | 2025-12-08 09:15:00 | 1691.00 | PARTIAL | 0.50 | 5.18% |
| SELL | retest2 | 2025-12-02 12:00:00 | 1759.20 | 2025-12-08 09:15:00 | 1671.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 13:00:00 | 1780.70 | 2025-12-08 09:15:00 | 1691.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 12:30:00 | 1780.00 | 2025-12-15 11:15:00 | 1729.00 | STOP_HIT | 0.50 | 2.87% |
| SELL | retest2 | 2025-12-02 10:30:00 | 1783.40 | 2025-12-15 11:15:00 | 1729.00 | STOP_HIT | 0.50 | 3.05% |
| SELL | retest2 | 2025-12-02 12:00:00 | 1759.20 | 2025-12-15 11:15:00 | 1729.00 | STOP_HIT | 0.50 | 1.72% |
| SELL | retest2 | 2025-12-04 13:00:00 | 1780.70 | 2025-12-15 11:15:00 | 1729.00 | STOP_HIT | 0.50 | 2.90% |
| SELL | retest2 | 2025-12-08 09:30:00 | 1686.00 | 2025-12-29 14:15:00 | 1648.72 | PARTIAL | 0.50 | 2.21% |
| SELL | retest2 | 2025-12-08 12:15:00 | 1678.20 | 2025-12-29 14:15:00 | 1647.30 | PARTIAL | 0.50 | 1.84% |
| SELL | retest2 | 2025-12-16 09:15:00 | 1685.10 | 2025-12-29 14:15:00 | 1651.86 | PARTIAL | 0.50 | 1.97% |
| SELL | retest2 | 2025-12-16 11:30:00 | 1690.70 | 2025-12-29 14:15:00 | 1649.20 | PARTIAL | 0.50 | 2.45% |
| SELL | retest2 | 2025-12-08 09:30:00 | 1686.00 | 2026-01-05 14:15:00 | 1707.00 | STOP_HIT | 0.50 | -1.25% |
| SELL | retest2 | 2025-12-08 12:15:00 | 1678.20 | 2026-01-05 14:15:00 | 1707.00 | STOP_HIT | 0.50 | -1.72% |
| SELL | retest2 | 2025-12-16 09:15:00 | 1685.10 | 2026-01-05 14:15:00 | 1707.00 | STOP_HIT | 0.50 | -1.30% |
| SELL | retest2 | 2025-12-16 11:30:00 | 1690.70 | 2026-01-05 14:15:00 | 1707.00 | STOP_HIT | 0.50 | -0.96% |
| SELL | retest2 | 2025-12-17 15:15:00 | 1735.50 | 2026-01-09 09:15:00 | 1606.16 | PARTIAL | 0.50 | 7.45% |
| SELL | retest2 | 2025-12-19 09:45:00 | 1734.00 | 2026-01-12 09:15:00 | 1601.70 | PARTIAL | 0.50 | 7.63% |
| SELL | retest2 | 2025-12-23 09:15:00 | 1738.80 | 2026-01-12 09:15:00 | 1594.29 | PARTIAL | 0.50 | 8.31% |
| SELL | retest2 | 2025-12-23 10:15:00 | 1736.00 | 2026-01-12 09:15:00 | 1600.84 | PARTIAL | 0.50 | 7.79% |
| SELL | retest2 | 2025-12-17 15:15:00 | 1735.50 | 2026-01-19 09:15:00 | 1517.40 | TARGET_HIT | 0.50 | 12.57% |
| SELL | retest2 | 2025-12-19 09:45:00 | 1734.00 | 2026-01-19 09:15:00 | 1510.38 | TARGET_HIT | 0.50 | 12.90% |
| SELL | retest2 | 2025-12-23 09:15:00 | 1738.80 | 2026-01-19 09:15:00 | 1516.59 | TARGET_HIT | 0.50 | 12.78% |
| SELL | retest2 | 2025-12-23 10:15:00 | 1736.00 | 2026-01-19 09:15:00 | 1521.63 | TARGET_HIT | 0.50 | 12.35% |
| SELL | retest2 | 2026-04-08 14:00:00 | 1257.50 | 2026-04-10 09:15:00 | 1332.20 | STOP_HIT | 1.00 | -5.94% |
| SELL | retest2 | 2026-04-08 15:15:00 | 1256.00 | 2026-04-10 09:15:00 | 1332.20 | STOP_HIT | 1.00 | -6.07% |
| SELL | retest2 | 2026-04-09 09:30:00 | 1254.60 | 2026-04-10 09:15:00 | 1332.20 | STOP_HIT | 1.00 | -6.19% |
