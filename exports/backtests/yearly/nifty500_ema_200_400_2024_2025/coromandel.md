# Coromandel International Ltd. (COROMANDEL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1928.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 8 |
| ALERT2_SKIP | 4 |
| ALERT3 | 52 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 43 |
| PARTIAL | 15 |
| TARGET_HIT | 2 |
| STOP_HIT | 38 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 25 / 30
- **Target hits / Stop hits / Partials:** 2 / 38 / 15
- **Avg / median % per leg:** 0.35% / -1.09%
- **Sum % (uncompounded):** 19.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 2 | 15.4% | 2 | 11 | 0 | -1.30% | -16.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 2 | 15.4% | 2 | 11 | 0 | -1.30% | -16.9% |
| SELL (all) | 42 | 23 | 54.8% | 0 | 27 | 15 | 0.86% | 36.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 42 | 23 | 54.8% | 0 | 27 | 15 | 0.86% | 36.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 55 | 25 | 45.5% | 2 | 38 | 15 | 0.35% | 19.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 13:15:00 | 1603.90 | 1633.06 | 1633.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 09:15:00 | 1594.75 | 1632.10 | 1632.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 11:15:00 | 1631.70 | 1629.86 | 1631.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 11:15:00 | 1631.70 | 1629.86 | 1631.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 11:15:00 | 1631.70 | 1629.86 | 1631.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 11:45:00 | 1633.00 | 1629.86 | 1631.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 12:15:00 | 1635.35 | 1629.92 | 1631.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 13:00:00 | 1635.35 | 1629.92 | 1631.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 13:15:00 | 1638.35 | 1630.00 | 1631.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 13:30:00 | 1642.85 | 1630.00 | 1631.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-11-01 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 18:15:00 | 1691.00 | 1633.13 | 1633.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 11:15:00 | 1709.55 | 1637.41 | 1635.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 09:15:00 | 1733.95 | 1733.96 | 1700.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-09 10:00:00 | 1733.95 | 1733.96 | 1700.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 14:15:00 | 1810.05 | 1857.10 | 1799.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 14:30:00 | 1795.05 | 1857.10 | 1799.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 1809.00 | 1855.15 | 1809.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 10:00:00 | 1809.00 | 1855.15 | 1809.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 1795.85 | 1854.56 | 1809.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 11:00:00 | 1795.85 | 1854.56 | 1809.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 11:15:00 | 1780.05 | 1853.82 | 1809.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 11:45:00 | 1780.50 | 1853.82 | 1809.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 1813.80 | 1849.57 | 1811.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:45:00 | 1816.35 | 1849.57 | 1811.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 15:15:00 | 1812.00 | 1849.19 | 1811.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 09:15:00 | 1772.10 | 1849.19 | 1811.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 1750.75 | 1848.22 | 1810.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 10:00:00 | 1750.75 | 1848.22 | 1810.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 1812.00 | 1828.41 | 1804.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:45:00 | 1839.15 | 1825.10 | 1805.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 10:45:00 | 1836.30 | 1825.24 | 1805.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 09:30:00 | 1842.10 | 1834.60 | 1812.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-12 10:45:00 | 1844.80 | 1835.07 | 1815.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 1799.80 | 1836.91 | 1817.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 10:00:00 | 1799.80 | 1836.91 | 1817.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 10:15:00 | 1769.90 | 1836.24 | 1817.01 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-14 10:15:00 | 1769.90 | 1836.24 | 1817.01 | SL hit (close<static) qty=1.00 sl=1797.20 alert=retest2 |

### Cycle 3 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 1688.00 | 1800.70 | 1800.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 12:15:00 | 1681.15 | 1791.62 | 1796.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 11:15:00 | 1761.45 | 1745.35 | 1769.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-05 12:00:00 | 1761.45 | 1745.35 | 1769.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 10:15:00 | 1751.00 | 1742.81 | 1766.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 10:30:00 | 1756.30 | 1742.81 | 1766.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 1760.35 | 1743.47 | 1766.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-10 12:45:00 | 1747.40 | 1744.01 | 1766.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-10 13:45:00 | 1747.25 | 1744.04 | 1766.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-10 15:15:00 | 1737.00 | 1744.11 | 1766.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 13:45:00 | 1747.90 | 1743.15 | 1764.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 1759.90 | 1743.32 | 1764.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 15:00:00 | 1759.90 | 1743.32 | 1764.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 1796.60 | 1743.96 | 1764.45 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-13 09:15:00 | 1796.60 | 1743.96 | 1764.45 | SL hit (close>static) qty=1.00 sl=1777.45 alert=retest2 |

### Cycle 4 — BUY (started 2025-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 15:15:00 | 1982.15 | 1783.64 | 1782.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 09:15:00 | 1997.90 | 1785.77 | 1783.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 14:15:00 | 2309.40 | 2314.61 | 2183.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 15:00:00 | 2309.40 | 2314.61 | 2183.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 2246.80 | 2315.97 | 2251.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:30:00 | 2250.00 | 2315.97 | 2251.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 2233.00 | 2315.14 | 2251.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 11:30:00 | 2241.10 | 2315.14 | 2251.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 2303.90 | 2314.40 | 2251.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 14:15:00 | 2305.90 | 2314.40 | 2251.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-27 15:15:00 | 2536.49 | 2316.21 | 2252.90 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 11:15:00 | 2215.00 | 2346.75 | 2347.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 10:15:00 | 2204.40 | 2326.29 | 2336.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 14:15:00 | 2326.40 | 2314.14 | 2329.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 14:15:00 | 2326.40 | 2314.14 | 2329.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 2326.40 | 2314.14 | 2329.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 15:00:00 | 2326.40 | 2314.14 | 2329.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 2317.90 | 2314.18 | 2329.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 09:15:00 | 2307.00 | 2314.18 | 2329.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 09:45:00 | 2306.00 | 2314.08 | 2328.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 12:30:00 | 2306.30 | 2313.53 | 2328.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 15:00:00 | 2303.00 | 2313.35 | 2328.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 2272.50 | 2312.95 | 2327.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:30:00 | 2255.10 | 2312.19 | 2327.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 2259.50 | 2309.13 | 2325.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 12:15:00 | 2255.20 | 2308.15 | 2324.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 15:15:00 | 2262.00 | 2306.64 | 2323.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 2191.65 | 2292.37 | 2313.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 2190.70 | 2292.37 | 2313.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 2190.99 | 2292.37 | 2313.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 2187.85 | 2292.37 | 2313.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 12:15:00 | 2142.34 | 2287.17 | 2310.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 12:15:00 | 2146.53 | 2287.17 | 2310.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 12:15:00 | 2142.44 | 2287.17 | 2310.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 12:15:00 | 2148.90 | 2287.17 | 2310.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 2283.70 | 2279.57 | 2304.70 | SL hit (close>ema200) qty=0.50 sl=2279.57 alert=retest2 |

### Cycle 6 — BUY (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 09:15:00 | 2377.30 | 2253.78 | 2253.63 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 10:15:00 | 2275.40 | 2283.34 | 2283.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 2242.10 | 2282.85 | 2283.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 14:15:00 | 2290.00 | 2274.90 | 2278.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 14:15:00 | 2290.00 | 2274.90 | 2278.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 2290.00 | 2274.90 | 2278.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 14:45:00 | 2300.00 | 2274.90 | 2278.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 15:15:00 | 2280.00 | 2274.95 | 2278.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 2253.70 | 2274.95 | 2278.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 14:30:00 | 2267.90 | 2273.91 | 2278.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 12:15:00 | 2261.60 | 2273.98 | 2278.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 12:15:00 | 2266.00 | 2272.88 | 2277.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 2251.50 | 2272.27 | 2277.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 2242.70 | 2272.19 | 2276.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 10:00:00 | 2250.80 | 2270.99 | 2275.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 12:15:00 | 2292.60 | 2271.09 | 2275.79 | SL hit (close>static) qty=1.00 sl=2290.30 alert=retest2 |

### Cycle 8 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 11:15:00 | 2358.00 | 2280.54 | 2280.19 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 2173.10 | 2280.06 | 2280.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 2103.40 | 2275.53 | 2278.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2064.30 | 2024.47 | 2110.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 11:15:00 | 2114.60 | 2026.00 | 2110.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 2114.60 | 2026.00 | 2110.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 12:00:00 | 2114.60 | 2026.00 | 2110.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 12:15:00 | 2143.00 | 2027.17 | 2110.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 12:30:00 | 2149.50 | 2027.17 | 2110.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 15:15:00 | 2111.40 | 2029.78 | 2110.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 09:15:00 | 2137.60 | 2029.78 | 2110.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 2147.40 | 2032.06 | 2110.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 11:00:00 | 2147.40 | 2032.06 | 2110.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 2126.80 | 2047.28 | 2113.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 10:30:00 | 2125.20 | 2047.28 | 2113.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 2092.30 | 2049.51 | 2113.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 14:45:00 | 2113.00 | 2049.51 | 2113.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 2112.00 | 2050.63 | 2113.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 14:00:00 | 2089.80 | 2052.74 | 2112.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 09:30:00 | 2080.40 | 2053.64 | 2112.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 11:30:00 | 2089.40 | 2054.35 | 2112.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:15:00 | 1985.31 | 2052.66 | 2100.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:15:00 | 1984.93 | 2052.66 | 2100.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 09:15:00 | 1976.38 | 2042.94 | 2089.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-10-01 10:00:00 | 1693.50 | 2024-10-07 09:15:00 | 1591.55 | STOP_HIT | 1.00 | -6.02% |
| BUY | retest2 | 2025-02-04 09:45:00 | 1839.15 | 2025-02-14 10:15:00 | 1769.90 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest2 | 2025-02-04 10:45:00 | 1836.30 | 2025-02-14 10:15:00 | 1769.90 | STOP_HIT | 1.00 | -3.62% |
| BUY | retest2 | 2025-02-07 09:30:00 | 1842.10 | 2025-02-14 10:15:00 | 1769.90 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest2 | 2025-02-12 10:45:00 | 1844.80 | 2025-02-14 10:15:00 | 1769.90 | STOP_HIT | 1.00 | -4.06% |
| SELL | retest2 | 2025-03-10 12:45:00 | 1747.40 | 2025-03-13 09:15:00 | 1796.60 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2025-03-10 13:45:00 | 1747.25 | 2025-03-13 09:15:00 | 1796.60 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2025-03-10 15:15:00 | 1737.00 | 2025-03-13 09:15:00 | 1796.60 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2025-03-12 13:45:00 | 1747.90 | 2025-03-13 09:15:00 | 1796.60 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2025-06-27 14:15:00 | 2305.90 | 2025-06-27 15:15:00 | 2536.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-02 10:30:00 | 2312.70 | 2025-07-03 13:15:00 | 2238.60 | STOP_HIT | 1.00 | -3.20% |
| BUY | retest2 | 2025-07-03 09:15:00 | 2348.30 | 2025-07-03 13:15:00 | 2238.60 | STOP_HIT | 1.00 | -4.67% |
| BUY | retest2 | 2025-07-09 15:00:00 | 2307.30 | 2025-07-30 09:15:00 | 2538.03 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-14 09:15:00 | 2292.50 | 2025-08-28 14:15:00 | 2265.70 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-08-29 11:45:00 | 2304.60 | 2025-09-05 14:15:00 | 2253.60 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-08-29 13:00:00 | 2296.60 | 2025-09-05 14:15:00 | 2253.60 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-09-01 10:00:00 | 2308.00 | 2025-09-05 14:15:00 | 2253.60 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-09-18 09:15:00 | 2307.00 | 2025-09-26 14:15:00 | 2191.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 09:45:00 | 2306.00 | 2025-09-26 14:15:00 | 2190.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 12:30:00 | 2306.30 | 2025-09-26 14:15:00 | 2190.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 15:00:00 | 2303.00 | 2025-09-26 14:15:00 | 2187.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 10:30:00 | 2255.10 | 2025-09-29 12:15:00 | 2142.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 09:15:00 | 2259.50 | 2025-09-29 12:15:00 | 2146.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 12:15:00 | 2255.20 | 2025-09-29 12:15:00 | 2142.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 15:15:00 | 2262.00 | 2025-09-29 12:15:00 | 2148.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 09:15:00 | 2307.00 | 2025-10-03 09:15:00 | 2283.70 | STOP_HIT | 0.50 | 1.01% |
| SELL | retest2 | 2025-09-18 09:45:00 | 2306.00 | 2025-10-03 09:15:00 | 2283.70 | STOP_HIT | 0.50 | 0.97% |
| SELL | retest2 | 2025-09-18 12:30:00 | 2306.30 | 2025-10-03 09:15:00 | 2283.70 | STOP_HIT | 0.50 | 0.98% |
| SELL | retest2 | 2025-09-18 15:00:00 | 2303.00 | 2025-10-03 09:15:00 | 2283.70 | STOP_HIT | 0.50 | 0.84% |
| SELL | retest2 | 2025-09-19 10:30:00 | 2255.10 | 2025-10-03 09:15:00 | 2283.70 | STOP_HIT | 0.50 | -1.27% |
| SELL | retest2 | 2025-09-22 09:15:00 | 2259.50 | 2025-10-03 09:15:00 | 2283.70 | STOP_HIT | 0.50 | -1.07% |
| SELL | retest2 | 2025-09-22 12:15:00 | 2255.20 | 2025-10-03 09:15:00 | 2283.70 | STOP_HIT | 0.50 | -1.26% |
| SELL | retest2 | 2025-09-22 15:15:00 | 2262.00 | 2025-10-03 09:15:00 | 2283.70 | STOP_HIT | 0.50 | -0.96% |
| SELL | retest2 | 2025-10-03 12:15:00 | 2287.20 | 2025-10-06 14:15:00 | 2349.40 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2025-10-03 13:45:00 | 2290.30 | 2025-10-06 14:15:00 | 2349.40 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2025-10-06 10:30:00 | 2285.00 | 2025-10-06 14:15:00 | 2349.40 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2025-10-06 11:15:00 | 2288.50 | 2025-10-06 14:15:00 | 2349.40 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-10-28 12:15:00 | 2238.70 | 2025-10-31 10:15:00 | 2126.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 10:45:00 | 2240.00 | 2025-10-31 10:15:00 | 2128.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 12:30:00 | 2242.60 | 2025-10-31 10:15:00 | 2130.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 09:45:00 | 2238.00 | 2025-10-31 10:15:00 | 2126.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 12:15:00 | 2238.70 | 2025-11-13 09:15:00 | 2222.60 | STOP_HIT | 0.50 | 0.72% |
| SELL | retest2 | 2025-10-29 10:45:00 | 2240.00 | 2025-11-13 09:15:00 | 2222.60 | STOP_HIT | 0.50 | 0.78% |
| SELL | retest2 | 2025-10-29 12:30:00 | 2242.60 | 2025-11-13 09:15:00 | 2222.60 | STOP_HIT | 0.50 | 0.89% |
| SELL | retest2 | 2025-10-30 09:45:00 | 2238.00 | 2025-11-13 09:15:00 | 2222.60 | STOP_HIT | 0.50 | 0.69% |
| SELL | retest2 | 2025-11-20 11:00:00 | 2218.00 | 2025-11-20 12:15:00 | 2263.90 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2026-02-04 09:15:00 | 2253.70 | 2026-02-13 12:15:00 | 2292.60 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2026-02-04 14:30:00 | 2267.90 | 2026-02-13 12:15:00 | 2292.60 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-02-05 12:15:00 | 2261.60 | 2026-02-17 09:15:00 | 2333.00 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2026-02-06 12:15:00 | 2266.00 | 2026-02-17 09:15:00 | 2333.00 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2026-02-12 09:15:00 | 2242.70 | 2026-02-17 09:15:00 | 2333.00 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2026-02-13 10:00:00 | 2250.80 | 2026-02-17 09:15:00 | 2333.00 | STOP_HIT | 1.00 | -3.65% |
| SELL | retest2 | 2026-04-15 14:00:00 | 2089.80 | 2026-04-24 10:15:00 | 1985.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-16 09:30:00 | 2080.40 | 2026-04-24 10:15:00 | 1984.93 | PARTIAL | 0.50 | 4.59% |
| SELL | retest2 | 2026-04-16 11:30:00 | 2089.40 | 2026-04-30 09:15:00 | 1976.38 | PARTIAL | 0.50 | 5.41% |
