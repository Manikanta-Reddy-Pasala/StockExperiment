# Muthoot Finance Ltd. (MUTHOOTFIN)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 3535.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT2_SKIP | 6 |
| ALERT3 | 48 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 19 |
| PARTIAL | 0 |
| TARGET_HIT | 5 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 14
- **Target hits / Stop hits / Partials:** 5 / 14 / 0
- **Avg / median % per leg:** 0.57% / -1.24%
- **Sum % (uncompounded):** 10.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 5 | 50.0% | 5 | 5 | 0 | 2.20% | 22.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 5 | 50.0% | 5 | 5 | 0 | 2.20% | 22.0% |
| SELL (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.23% | -11.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 9 | 0 | 0.0% | 0 | 9 | 0 | -1.23% | -11.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 19 | 5 | 26.3% | 5 | 14 | 0 | 0.57% | 10.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-10 12:15:00 | 1220.05 | 1253.55 | 1253.60 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-23 15:15:00 | 1277.10 | 1252.72 | 1252.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-25 09:15:00 | 1301.00 | 1253.20 | 1252.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-26 13:15:00 | 1251.15 | 1255.06 | 1253.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-26 13:15:00 | 1251.15 | 1255.06 | 1253.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 13:15:00 | 1251.15 | 1255.06 | 1253.87 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-02-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-15 12:15:00 | 1335.95 | 1397.80 | 1398.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 13:15:00 | 1329.10 | 1386.09 | 1391.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-05 09:15:00 | 1388.60 | 1352.24 | 1370.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-05 09:15:00 | 1388.60 | 1352.24 | 1370.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 09:15:00 | 1388.60 | 1352.24 | 1370.66 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2024-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 09:15:00 | 1478.90 | 1378.08 | 1377.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 11:15:00 | 1491.95 | 1380.24 | 1378.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 09:15:00 | 1582.95 | 1617.31 | 1544.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-09 09:15:00 | 1582.95 | 1617.31 | 1544.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 09:15:00 | 1582.95 | 1617.31 | 1544.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-09 11:30:00 | 1638.05 | 1617.13 | 1545.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 09:45:00 | 1616.75 | 1616.68 | 1546.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 12:30:00 | 1646.60 | 1673.61 | 1613.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-06 09:15:00 | 1801.86 | 1677.69 | 1619.22 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 15:15:00 | 1813.00 | 1916.37 | 1916.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 10:15:00 | 1812.35 | 1914.39 | 1915.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 11:15:00 | 1894.95 | 1890.97 | 1903.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-18 12:00:00 | 1894.95 | 1890.97 | 1903.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 12:15:00 | 1912.00 | 1891.18 | 1903.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:45:00 | 1914.70 | 1891.18 | 1903.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 1904.65 | 1891.31 | 1903.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 14:15:00 | 1895.05 | 1891.31 | 1903.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 10:15:00 | 1899.60 | 1891.38 | 1902.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 10:15:00 | 1898.15 | 1891.45 | 1902.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 11:30:00 | 1902.30 | 1891.68 | 1902.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 12:15:00 | 1910.30 | 1891.86 | 1902.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 13:00:00 | 1910.30 | 1891.86 | 1902.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 13:15:00 | 1908.00 | 1892.02 | 1902.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 13:45:00 | 1911.00 | 1892.02 | 1902.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 1909.75 | 1892.30 | 1902.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 09:45:00 | 1914.65 | 1892.30 | 1902.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-22 10:15:00 | 1922.30 | 1892.60 | 1902.77 | SL hit (close>static) qty=1.00 sl=1914.30 alert=retest2 |

### Cycle 6 — BUY (started 2024-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 09:15:00 | 1936.40 | 1910.37 | 1910.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 09:15:00 | 1979.00 | 1915.38 | 1912.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 14:15:00 | 2097.00 | 2097.96 | 2037.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-13 15:00:00 | 2097.00 | 2097.96 | 2037.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 2161.65 | 2206.29 | 2153.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 11:30:00 | 2160.40 | 2206.29 | 2153.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 2144.95 | 2204.40 | 2157.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 10:00:00 | 2144.95 | 2204.40 | 2157.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 10:15:00 | 2141.90 | 2203.78 | 2156.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 11:00:00 | 2141.90 | 2203.78 | 2156.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 2145.80 | 2196.69 | 2155.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-03 15:00:00 | 2145.80 | 2196.69 | 2155.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 2148.90 | 2196.21 | 2155.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:15:00 | 2132.95 | 2196.21 | 2155.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 15:15:00 | 2145.10 | 2192.53 | 2155.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:15:00 | 2129.65 | 2192.53 | 2155.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 2132.40 | 2191.94 | 2155.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:30:00 | 2121.05 | 2191.94 | 2155.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 2137.30 | 2191.39 | 2155.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 10:00:00 | 2162.55 | 2188.12 | 2154.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-19 10:15:00 | 2378.81 | 2206.55 | 2172.28 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 13:15:00 | 2161.50 | 2200.33 | 2200.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 09:15:00 | 2136.10 | 2199.05 | 2199.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 2213.70 | 2197.49 | 2199.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 2213.70 | 2197.49 | 2199.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 2213.70 | 2197.49 | 2199.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 2213.70 | 2197.49 | 2199.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 2214.80 | 2197.66 | 2199.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:45:00 | 2212.00 | 2197.66 | 2199.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 13:15:00 | 2201.80 | 2197.79 | 2199.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:00:00 | 2201.80 | 2197.79 | 2199.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 2200.70 | 2197.82 | 2199.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:30:00 | 2204.00 | 2197.82 | 2199.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 2203.60 | 2197.87 | 2199.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:15:00 | 2246.60 | 2197.87 | 2199.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 09:15:00 | 2297.10 | 2201.32 | 2200.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 10:15:00 | 2317.70 | 2202.48 | 2201.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 2205.30 | 2212.46 | 2206.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 09:15:00 | 2205.30 | 2212.46 | 2206.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 2205.30 | 2212.46 | 2206.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 09:15:00 | 2265.00 | 2213.03 | 2207.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 2255.90 | 2216.71 | 2209.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-15 10:15:00 | 2131.00 | 2217.74 | 2210.36 | SL hit (close<static) qty=1.00 sl=2145.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 09:15:00 | 2090.40 | 2203.03 | 2203.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 15:15:00 | 2084.00 | 2196.36 | 2199.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 2150.20 | 2145.97 | 2170.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-30 10:00:00 | 2150.20 | 2145.97 | 2170.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 2204.80 | 2146.55 | 2170.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:00:00 | 2204.80 | 2146.55 | 2170.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 2221.00 | 2147.29 | 2170.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:00:00 | 2221.00 | 2147.29 | 2170.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 15:15:00 | 2454.80 | 2189.47 | 2188.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 2508.50 | 2192.64 | 2190.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-29 10:15:00 | 2597.40 | 2606.95 | 2508.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-29 11:00:00 | 2597.40 | 2606.95 | 2508.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 2546.30 | 2611.49 | 2540.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:30:00 | 2545.70 | 2611.49 | 2540.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 2539.50 | 2610.77 | 2540.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 12:00:00 | 2539.50 | 2610.77 | 2540.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 2559.70 | 2610.26 | 2540.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 09:15:00 | 2760.80 | 2601.24 | 2539.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-22 14:15:00 | 3036.88 | 2820.89 | 2717.55 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 3473.80 | 3671.04 | 3671.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 11:15:00 | 3444.40 | 3647.05 | 3659.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 3447.20 | 3329.04 | 3433.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 3447.20 | 3329.04 | 3433.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 3447.20 | 3329.04 | 3433.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:00:00 | 3447.20 | 3329.04 | 3433.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 3461.60 | 3330.36 | 3433.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:30:00 | 3465.00 | 3330.36 | 3433.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 3554.70 | 3365.34 | 3441.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 10:30:00 | 3558.80 | 3365.34 | 3441.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 3466.60 | 3455.71 | 3474.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:30:00 | 3465.20 | 3455.71 | 3474.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 3471.90 | 3455.87 | 3474.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:45:00 | 3483.10 | 3455.87 | 3474.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 3492.00 | 3456.23 | 3474.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:00:00 | 3492.00 | 3456.23 | 3474.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 3495.00 | 3456.62 | 3474.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:30:00 | 3499.80 | 3456.62 | 3474.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 3497.00 | 3460.84 | 3475.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:15:00 | 3490.20 | 3460.84 | 3475.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 09:30:00 | 3488.50 | 3462.75 | 3476.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 10:30:00 | 3489.70 | 3463.02 | 3476.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 09:45:00 | 3492.10 | 3460.95 | 3474.39 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 3478.60 | 3461.13 | 3474.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:30:00 | 3485.00 | 3461.13 | 3474.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 3497.10 | 3461.49 | 3474.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:45:00 | 3500.00 | 3461.49 | 3474.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 13:15:00 | 3480.60 | 3461.79 | 3474.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 14:00:00 | 3480.60 | 3461.79 | 3474.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 3491.00 | 3462.08 | 3474.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 15:15:00 | 3496.00 | 3462.08 | 3474.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 3467.50 | 3460.80 | 3473.42 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-06 14:15:00 | 3533.60 | 3462.35 | 3473.89 | SL hit (close>static) qty=1.00 sl=3529.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-09 11:30:00 | 1638.05 | 2024-06-06 09:15:00 | 1801.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-10 09:45:00 | 1616.75 | 2024-06-06 09:15:00 | 1778.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 12:30:00 | 1646.60 | 2024-06-27 09:15:00 | 1811.26 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-18 14:15:00 | 1895.05 | 2024-11-22 10:15:00 | 1922.30 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-11-19 10:15:00 | 1899.60 | 2024-11-22 10:15:00 | 1922.30 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-11-21 10:15:00 | 1898.15 | 2024-11-22 10:15:00 | 1922.30 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-11-21 11:30:00 | 1902.30 | 2024-11-22 10:15:00 | 1922.30 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-11-28 13:30:00 | 1898.20 | 2024-11-29 15:15:00 | 1920.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-03-06 10:00:00 | 2162.55 | 2025-03-19 10:15:00 | 2378.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-09 12:15:00 | 2191.90 | 2025-04-11 09:15:00 | 2019.60 | STOP_HIT | 1.00 | -7.86% |
| BUY | retest2 | 2025-04-09 13:15:00 | 2160.55 | 2025-04-11 09:15:00 | 2019.60 | STOP_HIT | 1.00 | -6.52% |
| BUY | retest2 | 2025-04-21 09:30:00 | 2159.80 | 2025-04-25 09:15:00 | 2112.20 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-05-12 09:15:00 | 2265.00 | 2025-05-15 10:15:00 | 2131.00 | STOP_HIT | 1.00 | -5.92% |
| BUY | retest2 | 2025-05-14 09:15:00 | 2255.90 | 2025-05-15 10:15:00 | 2131.00 | STOP_HIT | 1.00 | -5.54% |
| BUY | retest2 | 2025-08-14 09:15:00 | 2760.80 | 2025-09-22 14:15:00 | 3036.88 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-28 11:15:00 | 3490.20 | 2026-05-06 14:15:00 | 3533.60 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-04-29 09:30:00 | 3488.50 | 2026-05-06 14:15:00 | 3533.60 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2026-04-29 10:30:00 | 3489.70 | 2026-05-06 14:15:00 | 3533.60 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-05-04 09:45:00 | 3492.10 | 2026-05-06 14:15:00 | 3533.60 | STOP_HIT | 1.00 | -1.19% |
