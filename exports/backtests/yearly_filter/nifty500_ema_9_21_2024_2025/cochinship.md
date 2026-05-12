# Cochin Shipyard Ltd. (COCHINSHIP)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1769.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 109 |
| ALERT1 | 84 |
| ALERT2 | 84 |
| ALERT2_SKIP | 46 |
| ALERT3 | 213 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 108 |
| PARTIAL | 44 |
| TARGET_HIT | 21 |
| STOP_HIT | 94 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 158 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 101 / 57
- **Target hits / Stop hits / Partials:** 21 / 93 / 44
- **Avg / median % per leg:** 2.58% / 3.16%
- **Sum % (uncompounded):** 407.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 38 | 13 | 34.2% | 9 | 28 | 1 | 1.02% | 38.6% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 2.85% | 11.4% |
| BUY @ 3rd Alert (retest2) | 34 | 11 | 32.4% | 8 | 26 | 0 | 0.80% | 27.2% |
| SELL (all) | 120 | 88 | 73.3% | 12 | 65 | 43 | 3.08% | 369.0% |
| SELL @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 0 | 3 | 3 | 4.27% | 25.6% |
| SELL @ 3rd Alert (retest2) | 114 | 82 | 71.9% | 12 | 62 | 40 | 3.01% | 343.5% |
| retest1 (combined) | 10 | 8 | 80.0% | 1 | 5 | 4 | 3.70% | 37.0% |
| retest2 (combined) | 148 | 93 | 62.8% | 20 | 88 | 40 | 2.50% | 370.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 1302.15 | 1245.95 | 1239.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 13:15:00 | 1321.00 | 1270.21 | 1252.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 13:15:00 | 1329.95 | 1334.57 | 1314.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 13:45:00 | 1332.45 | 1334.57 | 1314.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 1888.85 | 1948.69 | 1897.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 10:00:00 | 1888.85 | 1948.69 | 1897.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 1919.80 | 1942.91 | 1899.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 09:30:00 | 1958.55 | 1934.71 | 1912.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 11:00:00 | 1943.25 | 1958.50 | 1956.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 11:30:00 | 1941.50 | 1959.00 | 1956.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 15:15:00 | 1943.90 | 1954.22 | 1955.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 15:15:00 | 1943.90 | 1954.22 | 1955.06 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 2048.95 | 1973.17 | 1963.60 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 1812.00 | 1956.85 | 1967.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 1630.55 | 1813.65 | 1883.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 09:15:00 | 1853.00 | 1750.69 | 1805.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 1853.00 | 1750.69 | 1805.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1853.00 | 1750.69 | 1805.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:45:00 | 1853.00 | 1750.69 | 1805.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 1853.00 | 1771.15 | 1810.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:30:00 | 1853.00 | 1771.15 | 1810.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 09:15:00 | 1887.80 | 1837.60 | 1831.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 13:15:00 | 1964.85 | 1918.27 | 1907.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 09:15:00 | 2240.00 | 2244.24 | 2150.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-19 09:30:00 | 2228.80 | 2244.24 | 2150.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 2270.30 | 2259.04 | 2205.05 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2024-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 11:15:00 | 2143.00 | 2196.11 | 2201.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 12:15:00 | 2132.70 | 2183.43 | 2195.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 09:15:00 | 2163.00 | 2162.00 | 2179.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 09:15:00 | 2163.00 | 2162.00 | 2179.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 2163.00 | 2162.00 | 2179.93 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 13:15:00 | 2193.00 | 2173.01 | 2172.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 09:15:00 | 2222.50 | 2187.16 | 2179.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 14:15:00 | 2210.95 | 2211.42 | 2196.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 14:15:00 | 2210.95 | 2211.42 | 2196.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 14:15:00 | 2210.95 | 2211.42 | 2196.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-26 14:30:00 | 2200.25 | 2211.42 | 2196.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 2211.25 | 2211.17 | 2199.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 13:00:00 | 2233.95 | 2213.25 | 2203.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 11:00:00 | 2225.70 | 2226.98 | 2215.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 15:15:00 | 2228.00 | 2216.31 | 2213.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 13:00:00 | 2237.75 | 2252.34 | 2242.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 2257.95 | 2253.46 | 2243.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 2276.00 | 2255.23 | 2246.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-03 14:15:00 | 2448.27 | 2327.41 | 2288.00 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 14:15:00 | 2723.00 | 2740.39 | 2741.36 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 09:15:00 | 2839.25 | 2760.16 | 2750.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 13:15:00 | 2863.60 | 2814.16 | 2781.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 12:15:00 | 2832.95 | 2838.18 | 2811.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 12:15:00 | 2832.95 | 2838.18 | 2811.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 12:15:00 | 2832.95 | 2838.18 | 2811.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 12:30:00 | 2820.10 | 2838.18 | 2811.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 14:15:00 | 2822.05 | 2833.85 | 2813.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 15:00:00 | 2822.05 | 2833.85 | 2813.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 2830.05 | 2833.09 | 2815.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:15:00 | 2753.40 | 2833.09 | 2815.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 2764.35 | 2819.34 | 2810.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:30:00 | 2749.05 | 2819.34 | 2810.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 2750.55 | 2805.58 | 2805.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:45:00 | 2751.80 | 2805.58 | 2805.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 11:15:00 | 2750.55 | 2794.58 | 2800.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 11:15:00 | 2739.65 | 2758.80 | 2775.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 2661.05 | 2589.07 | 2630.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 2661.05 | 2589.07 | 2630.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 2661.05 | 2589.07 | 2630.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 2661.05 | 2589.07 | 2630.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 2659.00 | 2603.06 | 2632.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:30:00 | 2670.35 | 2603.06 | 2632.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2024-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 15:15:00 | 2670.35 | 2648.30 | 2646.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 09:15:00 | 2722.20 | 2663.08 | 2653.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 12:15:00 | 2612.55 | 2666.19 | 2659.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 12:15:00 | 2612.55 | 2666.19 | 2659.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 2612.55 | 2666.19 | 2659.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 2616.05 | 2666.19 | 2659.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 2609.70 | 2654.89 | 2654.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 13:45:00 | 2610.00 | 2654.89 | 2654.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 14:15:00 | 2591.55 | 2642.22 | 2648.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-24 11:15:00 | 2580.35 | 2613.96 | 2631.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-29 09:15:00 | 2507.95 | 2488.93 | 2514.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 09:15:00 | 2507.95 | 2488.93 | 2514.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 2507.95 | 2488.93 | 2514.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 10:15:00 | 2568.90 | 2488.93 | 2514.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 10:15:00 | 2563.00 | 2503.74 | 2518.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 10:30:00 | 2563.95 | 2503.74 | 2518.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 11:15:00 | 2572.05 | 2517.41 | 2523.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 11:30:00 | 2585.00 | 2517.41 | 2523.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2024-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 13:15:00 | 2580.25 | 2537.02 | 2531.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 14:15:00 | 2590.95 | 2547.81 | 2537.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 09:15:00 | 2634.00 | 2652.38 | 2613.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 09:15:00 | 2634.00 | 2652.38 | 2613.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 2634.00 | 2652.38 | 2613.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:30:00 | 2637.95 | 2652.38 | 2613.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 2618.50 | 2638.59 | 2621.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 2618.50 | 2638.59 | 2621.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 2625.00 | 2635.87 | 2621.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:15:00 | 2651.00 | 2635.87 | 2621.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 10:15:00 | 2601.00 | 2627.08 | 2619.87 | SL hit (close<static) qty=1.00 sl=2615.20 alert=retest2 |

### Cycle 14 — SELL (started 2024-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 12:15:00 | 2585.40 | 2612.12 | 2613.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 2555.00 | 2587.83 | 2600.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 11:15:00 | 2305.90 | 2302.63 | 2368.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 11:45:00 | 2326.60 | 2302.63 | 2368.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 2341.50 | 2317.25 | 2359.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:30:00 | 2378.25 | 2317.25 | 2359.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 2351.35 | 2328.81 | 2357.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 09:30:00 | 2363.00 | 2328.81 | 2357.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 10:15:00 | 2354.75 | 2334.00 | 2357.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 10:45:00 | 2357.75 | 2334.00 | 2357.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 11:15:00 | 2372.00 | 2341.60 | 2358.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 11:30:00 | 2367.00 | 2341.60 | 2358.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 12:15:00 | 2354.95 | 2344.27 | 2358.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 12:30:00 | 2364.30 | 2344.27 | 2358.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 13:15:00 | 2331.00 | 2341.62 | 2355.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 13:30:00 | 2359.00 | 2341.62 | 2355.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 2427.80 | 2349.55 | 2355.06 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2024-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 10:15:00 | 2414.20 | 2362.48 | 2360.43 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 11:15:00 | 2331.15 | 2358.62 | 2362.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 14:15:00 | 2322.90 | 2344.69 | 2354.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 13:15:00 | 2214.95 | 2205.25 | 2247.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 14:00:00 | 2214.95 | 2205.25 | 2247.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 14:15:00 | 2214.95 | 2207.19 | 2244.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 14:30:00 | 2244.00 | 2207.19 | 2244.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 2194.05 | 2205.01 | 2236.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 12:00:00 | 2185.00 | 2198.05 | 2214.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 10:15:00 | 2075.75 | 2143.65 | 2178.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-21 09:15:00 | 2114.85 | 2100.72 | 2136.60 | SL hit (close>ema200) qty=0.50 sl=2100.72 alert=retest2 |

### Cycle 17 — BUY (started 2024-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 11:15:00 | 1939.05 | 1908.38 | 1908.11 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 1874.95 | 1910.44 | 1913.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 1861.50 | 1881.40 | 1896.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 1836.95 | 1829.26 | 1852.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 1836.95 | 1829.26 | 1852.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 1836.95 | 1829.26 | 1852.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 09:45:00 | 1820.00 | 1833.40 | 1844.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 10:45:00 | 1817.95 | 1830.02 | 1841.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 10:45:00 | 1817.50 | 1812.26 | 1817.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 11:30:00 | 1819.00 | 1814.60 | 1817.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 12:15:00 | 1824.50 | 1816.58 | 1818.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 13:00:00 | 1824.50 | 1816.58 | 1818.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 13:15:00 | 1824.00 | 1818.06 | 1818.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 13:45:00 | 1831.45 | 1818.06 | 1818.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 15:15:00 | 1818.80 | 1818.39 | 1818.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 09:15:00 | 1803.00 | 1818.39 | 1818.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 1792.20 | 1813.15 | 1816.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 10:45:00 | 1783.10 | 1806.52 | 1813.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 12:30:00 | 1786.30 | 1799.31 | 1808.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 09:45:00 | 1782.00 | 1791.70 | 1801.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 12:15:00 | 1729.00 | 1753.26 | 1772.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 12:15:00 | 1728.05 | 1753.26 | 1772.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 13:15:00 | 1727.05 | 1748.07 | 1767.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 13:15:00 | 1726.62 | 1748.07 | 1767.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 09:15:00 | 1693.94 | 1726.03 | 1752.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 09:15:00 | 1696.98 | 1726.03 | 1752.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 09:15:00 | 1692.90 | 1726.03 | 1752.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-09-19 11:15:00 | 1638.00 | 1700.97 | 1735.76 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 19 — BUY (started 2024-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 11:15:00 | 1846.05 | 1754.74 | 1742.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 10:15:00 | 1856.30 | 1822.16 | 1787.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 13:15:00 | 1829.20 | 1830.45 | 1800.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-23 14:00:00 | 1829.20 | 1830.45 | 1800.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 14:15:00 | 1782.90 | 1820.94 | 1799.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 14:45:00 | 1802.35 | 1820.94 | 1799.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 15:15:00 | 1789.00 | 1814.55 | 1798.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 09:30:00 | 1787.90 | 1806.26 | 1795.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 1788.50 | 1802.71 | 1795.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:15:00 | 1772.00 | 1802.71 | 1795.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 1784.20 | 1799.01 | 1794.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 12:15:00 | 1792.00 | 1799.01 | 1794.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 15:00:00 | 1791.85 | 1792.68 | 1792.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 09:15:00 | 1770.50 | 1789.73 | 1790.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 09:15:00 | 1770.50 | 1789.73 | 1790.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 10:15:00 | 1761.15 | 1784.02 | 1788.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 14:15:00 | 1730.00 | 1704.55 | 1722.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 14:15:00 | 1730.00 | 1704.55 | 1722.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 1730.00 | 1704.55 | 1722.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 1730.00 | 1704.55 | 1722.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 15:15:00 | 1760.00 | 1715.64 | 1725.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 09:15:00 | 1704.00 | 1715.64 | 1725.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 14:15:00 | 1701.20 | 1706.74 | 1716.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 10:00:00 | 1706.40 | 1715.86 | 1719.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 11:30:00 | 1704.65 | 1712.96 | 1717.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 1711.00 | 1708.38 | 1713.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 1688.45 | 1708.38 | 1713.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 14:15:00 | 1618.80 | 1668.20 | 1682.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 14:15:00 | 1616.14 | 1668.20 | 1682.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 14:15:00 | 1621.08 | 1668.20 | 1682.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 14:15:00 | 1619.42 | 1668.20 | 1682.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 14:15:00 | 1604.03 | 1668.20 | 1682.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-08 09:15:00 | 1533.60 | 1589.90 | 1622.99 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 21 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 1654.15 | 1631.93 | 1629.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 1676.00 | 1651.84 | 1641.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 09:15:00 | 1677.90 | 1682.59 | 1666.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 09:15:00 | 1677.90 | 1682.59 | 1666.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 1677.90 | 1682.59 | 1666.24 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 09:15:00 | 1639.65 | 1662.52 | 1662.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 11:15:00 | 1629.00 | 1650.77 | 1657.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 10:15:00 | 1675.00 | 1642.75 | 1648.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 10:15:00 | 1675.00 | 1642.75 | 1648.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 1675.00 | 1642.75 | 1648.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:00:00 | 1675.00 | 1642.75 | 1648.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 1675.50 | 1649.30 | 1650.72 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2024-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 12:15:00 | 1670.00 | 1653.44 | 1652.48 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 09:15:00 | 1611.00 | 1651.03 | 1652.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 12:15:00 | 1598.00 | 1626.11 | 1639.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 11:15:00 | 1575.00 | 1567.59 | 1587.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 12:00:00 | 1575.00 | 1567.59 | 1587.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 15:15:00 | 1369.00 | 1358.32 | 1377.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 09:15:00 | 1405.00 | 1358.32 | 1377.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 1390.00 | 1364.66 | 1378.98 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2024-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 15:15:00 | 1409.00 | 1388.43 | 1386.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 1447.00 | 1400.14 | 1391.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 1472.00 | 1496.17 | 1473.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 1472.00 | 1496.17 | 1473.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1472.00 | 1496.17 | 1473.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 09:45:00 | 1482.00 | 1496.17 | 1473.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 1471.20 | 1491.17 | 1472.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:15:00 | 1470.15 | 1491.17 | 1472.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 1481.80 | 1489.30 | 1473.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 13:00:00 | 1484.50 | 1488.34 | 1474.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 11:15:00 | 1488.40 | 1491.58 | 1482.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 12:15:00 | 1444.90 | 1479.43 | 1478.26 | SL hit (close<static) qty=1.00 sl=1463.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 1459.65 | 1503.34 | 1505.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 1400.00 | 1453.70 | 1476.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 1335.00 | 1314.14 | 1326.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 09:15:00 | 1335.00 | 1314.14 | 1326.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 1335.00 | 1314.14 | 1326.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:15:00 | 1350.00 | 1314.14 | 1326.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 1362.30 | 1323.77 | 1329.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:30:00 | 1362.30 | 1323.77 | 1329.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 1362.30 | 1337.64 | 1335.47 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 10:15:00 | 1298.30 | 1331.71 | 1334.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 12:15:00 | 1296.00 | 1319.18 | 1328.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 12:15:00 | 1300.00 | 1299.04 | 1311.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 13:00:00 | 1300.00 | 1299.04 | 1311.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 1351.00 | 1309.60 | 1312.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 10:15:00 | 1358.95 | 1309.60 | 1312.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 1364.00 | 1320.48 | 1316.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 09:15:00 | 1432.45 | 1366.42 | 1343.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 11:15:00 | 1552.95 | 1559.52 | 1520.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 11:45:00 | 1544.90 | 1559.52 | 1520.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 1675.00 | 1686.69 | 1665.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 11:00:00 | 1699.00 | 1683.05 | 1673.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 12:00:00 | 1697.15 | 1685.87 | 1676.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 13:15:00 | 1698.45 | 1687.69 | 1677.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 13:45:00 | 1696.00 | 1688.76 | 1679.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 1680.05 | 1687.88 | 1681.24 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-10 09:15:00 | 1652.00 | 1677.43 | 1679.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 09:15:00 | 1652.00 | 1677.43 | 1679.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 10:15:00 | 1647.00 | 1671.34 | 1676.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 09:15:00 | 1659.00 | 1642.21 | 1650.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 09:15:00 | 1659.00 | 1642.21 | 1650.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 1659.00 | 1642.21 | 1650.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:00:00 | 1659.00 | 1642.21 | 1650.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 1660.00 | 1645.76 | 1651.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 15:00:00 | 1653.10 | 1651.87 | 1653.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 12:15:00 | 1570.44 | 1592.61 | 1605.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-20 15:15:00 | 1487.79 | 1520.56 | 1546.20 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 31 — BUY (started 2024-12-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 12:15:00 | 1539.05 | 1483.54 | 1482.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 14:15:00 | 1584.05 | 1518.88 | 1506.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 09:15:00 | 1527.05 | 1531.90 | 1514.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 09:15:00 | 1527.05 | 1531.90 | 1514.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 1527.05 | 1531.90 | 1514.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 10:00:00 | 1606.85 | 1565.00 | 1546.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 11:15:00 | 1589.75 | 1569.30 | 1550.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 11:45:00 | 1591.05 | 1573.64 | 1554.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 15:00:00 | 1592.95 | 1582.99 | 1563.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 1544.80 | 1584.61 | 1578.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:00:00 | 1544.80 | 1584.61 | 1578.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-01-06 10:15:00 | 1518.60 | 1571.41 | 1572.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 1518.60 | 1571.41 | 1572.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 09:15:00 | 1504.00 | 1530.09 | 1548.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 12:15:00 | 1526.10 | 1525.87 | 1541.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 13:00:00 | 1526.10 | 1525.87 | 1541.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 1517.75 | 1502.39 | 1514.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:00:00 | 1517.75 | 1502.39 | 1514.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 1500.50 | 1502.01 | 1513.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 11:15:00 | 1492.50 | 1502.01 | 1513.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 1417.88 | 1477.74 | 1495.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 09:15:00 | 1401.60 | 1387.37 | 1420.14 | SL hit (close>ema200) qty=0.50 sl=1387.37 alert=retest2 |

### Cycle 33 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 1443.05 | 1421.70 | 1419.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 1505.95 | 1452.68 | 1437.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 13:15:00 | 1535.00 | 1535.66 | 1513.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 14:00:00 | 1535.00 | 1535.66 | 1513.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 1510.90 | 1530.71 | 1513.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 15:00:00 | 1510.90 | 1530.71 | 1513.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 1505.00 | 1525.57 | 1512.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 1469.10 | 1525.57 | 1512.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 1490.25 | 1518.50 | 1510.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:30:00 | 1471.60 | 1518.50 | 1510.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 1466.95 | 1508.19 | 1506.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 11:00:00 | 1466.95 | 1508.19 | 1506.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 11:15:00 | 1449.00 | 1496.35 | 1501.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 13:15:00 | 1443.00 | 1478.69 | 1492.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 1481.75 | 1474.48 | 1486.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 1481.75 | 1474.48 | 1486.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1481.75 | 1474.48 | 1486.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 1494.20 | 1474.48 | 1486.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1488.65 | 1477.32 | 1486.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 1490.95 | 1477.32 | 1486.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 1475.00 | 1476.85 | 1485.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:45:00 | 1457.30 | 1474.06 | 1480.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 11:00:00 | 1463.25 | 1471.90 | 1479.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 11:45:00 | 1459.25 | 1469.39 | 1477.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:15:00 | 1390.09 | 1442.35 | 1459.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 1384.43 | 1402.62 | 1430.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 1386.29 | 1402.62 | 1430.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 11:15:00 | 1417.05 | 1395.56 | 1421.91 | SL hit (close>ema200) qty=0.50 sl=1395.56 alert=retest2 |

### Cycle 35 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 1433.00 | 1426.30 | 1426.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 1439.00 | 1428.84 | 1427.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 1447.00 | 1451.34 | 1441.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 14:00:00 | 1447.00 | 1451.34 | 1441.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 1514.00 | 1512.28 | 1487.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:00:00 | 1514.00 | 1512.28 | 1487.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 1507.80 | 1511.38 | 1489.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 1507.25 | 1511.38 | 1489.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1469.65 | 1503.04 | 1487.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 1469.65 | 1503.04 | 1487.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1473.00 | 1497.03 | 1486.60 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 1398.75 | 1468.92 | 1475.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 1379.00 | 1408.45 | 1417.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 11:15:00 | 1371.70 | 1362.13 | 1382.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 11:15:00 | 1371.70 | 1362.13 | 1382.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 11:15:00 | 1371.70 | 1362.13 | 1382.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 11:45:00 | 1404.05 | 1362.13 | 1382.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 12:15:00 | 1374.70 | 1364.64 | 1381.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 12:45:00 | 1370.30 | 1364.64 | 1381.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 14:15:00 | 1372.30 | 1363.81 | 1377.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 15:00:00 | 1372.30 | 1363.81 | 1377.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 1332.50 | 1356.92 | 1372.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 13:00:00 | 1325.05 | 1344.13 | 1362.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 14:30:00 | 1326.90 | 1339.61 | 1356.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 15:15:00 | 1324.45 | 1339.61 | 1356.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 1258.80 | 1319.96 | 1344.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 1260.56 | 1319.96 | 1344.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 1258.23 | 1319.96 | 1344.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-12 15:15:00 | 1305.00 | 1304.52 | 1324.61 | SL hit (close>ema200) qty=0.50 sl=1304.52 alert=retest2 |

### Cycle 37 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 1323.05 | 1254.13 | 1245.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-25 10:15:00 | 1346.70 | 1317.13 | 1307.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-25 11:15:00 | 1310.05 | 1315.72 | 1307.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 11:15:00 | 1310.05 | 1315.72 | 1307.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 11:15:00 | 1310.05 | 1315.72 | 1307.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 12:00:00 | 1310.05 | 1315.72 | 1307.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 12:15:00 | 1307.10 | 1313.99 | 1307.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 12:30:00 | 1306.40 | 1313.99 | 1307.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 13:15:00 | 1305.45 | 1312.28 | 1307.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 14:00:00 | 1305.45 | 1312.28 | 1307.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 14:15:00 | 1305.00 | 1310.83 | 1307.24 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 1286.50 | 1305.08 | 1305.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 10:15:00 | 1272.70 | 1298.60 | 1302.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 1294.70 | 1292.66 | 1297.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 14:15:00 | 1294.70 | 1292.66 | 1297.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 1294.70 | 1292.66 | 1297.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 15:00:00 | 1294.70 | 1292.66 | 1297.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 1253.50 | 1285.36 | 1293.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 09:30:00 | 1240.80 | 1261.46 | 1275.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 12:15:00 | 1245.30 | 1249.34 | 1256.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 13:00:00 | 1245.60 | 1248.59 | 1255.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 13:30:00 | 1247.70 | 1249.04 | 1254.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 1261.25 | 1251.48 | 1255.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 15:00:00 | 1261.25 | 1251.48 | 1255.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 15:15:00 | 1263.30 | 1253.85 | 1256.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:15:00 | 1282.85 | 1253.85 | 1256.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 1272.75 | 1257.63 | 1257.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:30:00 | 1285.05 | 1257.63 | 1257.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-05 10:15:00 | 1270.00 | 1260.10 | 1258.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 1270.00 | 1260.10 | 1258.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 12:15:00 | 1286.70 | 1273.75 | 1267.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 1307.00 | 1317.66 | 1300.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 1307.00 | 1317.66 | 1300.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 1307.00 | 1317.66 | 1300.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 1307.00 | 1317.66 | 1300.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 1305.00 | 1315.13 | 1301.30 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 1269.40 | 1297.32 | 1298.01 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 10:15:00 | 1302.50 | 1287.99 | 1286.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 1329.95 | 1300.87 | 1294.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 1418.85 | 1419.27 | 1377.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 1418.85 | 1419.27 | 1377.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 1440.00 | 1443.49 | 1426.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:15:00 | 1475.70 | 1443.49 | 1426.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 14:15:00 | 1422.90 | 1453.64 | 1457.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 14:15:00 | 1422.90 | 1453.64 | 1457.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 13:15:00 | 1418.65 | 1435.98 | 1446.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 11:15:00 | 1400.70 | 1397.53 | 1411.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 11:30:00 | 1400.00 | 1397.53 | 1411.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 1410.00 | 1400.78 | 1409.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 1410.00 | 1400.78 | 1409.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 1413.75 | 1403.37 | 1409.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:15:00 | 1418.60 | 1403.37 | 1409.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 10:15:00 | 1438.20 | 1415.56 | 1414.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 14:15:00 | 1464.00 | 1436.84 | 1426.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 1388.45 | 1431.67 | 1425.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 1388.45 | 1431.67 | 1425.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1388.45 | 1431.67 | 1425.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 1388.45 | 1431.67 | 1425.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 1398.05 | 1424.94 | 1423.37 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 1388.40 | 1417.63 | 1420.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 1283.75 | 1377.90 | 1399.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 1344.10 | 1342.77 | 1368.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 1380.80 | 1342.77 | 1368.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1363.70 | 1346.96 | 1367.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:45:00 | 1369.15 | 1346.96 | 1367.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 1358.50 | 1349.27 | 1366.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 10:00:00 | 1338.30 | 1361.68 | 1367.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 1430.65 | 1372.46 | 1368.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 1430.65 | 1372.46 | 1368.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1460.00 | 1408.75 | 1391.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 1438.70 | 1444.52 | 1432.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 1438.70 | 1444.52 | 1432.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 1467.90 | 1485.80 | 1478.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 1467.90 | 1485.80 | 1478.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 1474.00 | 1483.44 | 1478.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 1463.20 | 1483.44 | 1478.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 1474.00 | 1480.01 | 1477.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 13:00:00 | 1474.00 | 1480.01 | 1477.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 13:15:00 | 1479.60 | 1479.93 | 1477.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 13:30:00 | 1474.50 | 1479.93 | 1477.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 1482.90 | 1485.92 | 1482.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 13:00:00 | 1482.90 | 1485.92 | 1482.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 1483.30 | 1485.39 | 1482.40 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 1428.40 | 1472.09 | 1476.94 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 11:15:00 | 1508.90 | 1469.57 | 1465.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 1531.20 | 1497.51 | 1482.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 14:15:00 | 1595.20 | 1608.16 | 1576.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-30 15:00:00 | 1595.20 | 1608.16 | 1576.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 1584.00 | 1603.33 | 1576.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:15:00 | 1560.20 | 1603.33 | 1576.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 1555.90 | 1593.85 | 1574.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:30:00 | 1566.70 | 1593.85 | 1574.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 1571.20 | 1589.32 | 1574.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:30:00 | 1562.00 | 1589.32 | 1574.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 11:15:00 | 1563.30 | 1584.11 | 1573.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 11:30:00 | 1564.00 | 1584.11 | 1573.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 12:15:00 | 1552.40 | 1577.77 | 1571.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 13:00:00 | 1552.40 | 1577.77 | 1571.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-05-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 14:15:00 | 1530.50 | 1563.12 | 1565.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-05 09:15:00 | 1524.70 | 1550.11 | 1559.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 13:15:00 | 1545.70 | 1544.70 | 1553.06 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-05 14:15:00 | 1537.20 | 1544.70 | 1553.06 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-05 14:45:00 | 1539.90 | 1544.00 | 1551.98 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-06 09:15:00 | 1531.70 | 1544.00 | 1551.26 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 1511.00 | 1537.40 | 1547.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:30:00 | 1504.60 | 1528.53 | 1541.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 13:00:00 | 1505.90 | 1524.00 | 1538.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 1460.34 | 1496.97 | 1519.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 1462.90 | 1496.97 | 1519.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 13:15:00 | 1455.12 | 1478.64 | 1502.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-05-08 09:15:00 | 1482.00 | 1473.87 | 1494.20 | SL hit (close>ema200) qty=0.50 sl=1473.87 alert=retest1 |

### Cycle 49 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 1508.30 | 1481.24 | 1480.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1570.80 | 1520.16 | 1502.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 14:15:00 | 2001.40 | 2006.64 | 1933.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 15:00:00 | 2001.40 | 2006.64 | 1933.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1858.80 | 1973.12 | 1930.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:00:00 | 1858.80 | 1973.12 | 1930.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1866.90 | 1951.87 | 1924.66 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 1833.70 | 1899.29 | 1904.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 1817.40 | 1882.91 | 1896.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 15:15:00 | 1844.90 | 1841.12 | 1861.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 09:15:00 | 1842.40 | 1841.12 | 1861.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 1896.40 | 1852.17 | 1864.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:00:00 | 1896.40 | 1852.17 | 1864.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 1880.30 | 1857.80 | 1866.27 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 13:15:00 | 1917.90 | 1879.49 | 1874.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 10:15:00 | 1937.90 | 1904.32 | 1889.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 14:15:00 | 1910.40 | 1913.36 | 1899.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-23 15:00:00 | 1910.40 | 1913.36 | 1899.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 1881.30 | 1907.05 | 1898.74 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 14:15:00 | 1884.30 | 1894.78 | 1895.13 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 09:15:00 | 1925.80 | 1899.74 | 1897.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 11:15:00 | 1968.90 | 1936.69 | 1920.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 1964.10 | 1967.82 | 1952.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 1964.10 | 1967.82 | 1952.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1964.10 | 1967.82 | 1952.99 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 09:15:00 | 1920.50 | 1945.84 | 1947.55 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 10:15:00 | 2020.80 | 1958.70 | 1950.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 11:15:00 | 2042.20 | 1975.40 | 1958.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 09:15:00 | 2062.50 | 2073.22 | 2039.33 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-05 12:30:00 | 2151.00 | 2101.26 | 2060.81 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-05 13:15:00 | 2258.55 | 2148.09 | 2085.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-06-05 15:15:00 | 2366.10 | 2221.81 | 2132.01 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 56 — SELL (started 2025-06-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 15:15:00 | 2261.80 | 2272.74 | 2273.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 09:15:00 | 2197.90 | 2257.77 | 2266.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 09:15:00 | 2223.60 | 2214.59 | 2235.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 2223.60 | 2214.59 | 2235.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 2223.60 | 2214.59 | 2235.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:30:00 | 2211.10 | 2214.59 | 2235.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 2203.00 | 2174.25 | 2198.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 11:00:00 | 2203.00 | 2174.25 | 2198.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 11:15:00 | 2169.90 | 2173.38 | 2195.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 12:30:00 | 2163.80 | 2172.68 | 2193.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 13:15:00 | 2150.80 | 2172.68 | 2193.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 14:15:00 | 2156.80 | 2174.34 | 2192.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:30:00 | 2130.00 | 2164.90 | 2183.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 2185.60 | 2169.04 | 2183.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 2185.60 | 2169.04 | 2183.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 2201.10 | 2175.45 | 2184.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:45:00 | 2202.20 | 2175.45 | 2184.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 2190.00 | 2178.36 | 2185.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 13:15:00 | 2184.00 | 2178.36 | 2185.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 14:45:00 | 2180.50 | 2182.71 | 2186.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 09:15:00 | 2208.90 | 2188.96 | 2188.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 09:15:00 | 2208.90 | 2188.96 | 2188.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 10:15:00 | 2223.80 | 2195.92 | 2191.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 12:15:00 | 2197.50 | 2198.83 | 2193.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 12:15:00 | 2197.50 | 2198.83 | 2193.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 2197.50 | 2198.83 | 2193.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 13:00:00 | 2197.50 | 2198.83 | 2193.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 2198.60 | 2199.64 | 2195.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 15:00:00 | 2198.60 | 2199.64 | 2195.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 2194.00 | 2198.51 | 2195.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 2180.00 | 2198.51 | 2195.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 2185.50 | 2195.91 | 2194.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:30:00 | 2181.40 | 2195.91 | 2194.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 10:15:00 | 2165.00 | 2189.73 | 2191.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 13:15:00 | 2154.20 | 2178.99 | 2186.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 2176.60 | 2168.43 | 2178.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 2176.60 | 2168.43 | 2178.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 2176.60 | 2168.43 | 2178.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:00:00 | 2176.60 | 2168.43 | 2178.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 2149.90 | 2164.73 | 2175.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:30:00 | 2141.30 | 2158.78 | 2172.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 2122.60 | 2118.53 | 2139.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 15:15:00 | 2204.00 | 2147.44 | 2146.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 15:15:00 | 2204.00 | 2147.44 | 2146.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 10:15:00 | 2211.60 | 2169.54 | 2157.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 2176.90 | 2206.06 | 2185.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 2176.90 | 2206.06 | 2185.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 2176.90 | 2206.06 | 2185.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:45:00 | 2181.10 | 2206.06 | 2185.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 2179.20 | 2200.69 | 2185.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:15:00 | 2171.90 | 2200.69 | 2185.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 2184.60 | 2197.47 | 2185.30 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 09:15:00 | 2135.80 | 2173.85 | 2177.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-25 11:15:00 | 2128.00 | 2158.38 | 2169.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 09:15:00 | 2109.80 | 2107.27 | 2125.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 09:15:00 | 2109.80 | 2107.27 | 2125.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 2109.80 | 2107.27 | 2125.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 11:30:00 | 2093.00 | 2103.64 | 2120.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 13:30:00 | 2093.00 | 2100.36 | 2116.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 11:15:00 | 2061.90 | 2042.39 | 2041.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 11:15:00 | 2061.90 | 2042.39 | 2041.78 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 10:15:00 | 2025.60 | 2043.37 | 2045.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 11:15:00 | 2012.40 | 2037.18 | 2042.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 2060.00 | 2033.05 | 2037.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 2060.00 | 2033.05 | 2037.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 2060.00 | 2033.05 | 2037.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:00:00 | 2060.00 | 2033.05 | 2037.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 2048.00 | 2036.04 | 2038.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 11:45:00 | 2045.40 | 2038.21 | 2038.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 12:15:00 | 2044.90 | 2038.21 | 2038.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 12:15:00 | 2045.60 | 2039.69 | 2039.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 12:15:00 | 2045.60 | 2039.69 | 2039.59 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 13:15:00 | 2035.00 | 2038.75 | 2039.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 09:15:00 | 2020.20 | 2034.60 | 2037.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 11:15:00 | 1935.40 | 1927.71 | 1943.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-16 11:45:00 | 1937.80 | 1927.71 | 1943.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 1936.00 | 1931.70 | 1941.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:45:00 | 1940.40 | 1931.70 | 1941.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1923.00 | 1931.32 | 1939.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 11:30:00 | 1915.10 | 1925.59 | 1935.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:15:00 | 1819.34 | 1842.18 | 1851.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-07-29 09:15:00 | 1723.59 | 1768.53 | 1797.00 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 65 — BUY (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 12:15:00 | 1791.00 | 1779.58 | 1779.49 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 15:15:00 | 1770.00 | 1778.50 | 1779.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 1745.80 | 1771.96 | 1776.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 1738.20 | 1738.06 | 1751.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 12:00:00 | 1738.20 | 1738.06 | 1751.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1742.20 | 1739.68 | 1747.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:15:00 | 1723.90 | 1735.18 | 1741.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 15:15:00 | 1637.70 | 1663.22 | 1683.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 09:15:00 | 1664.80 | 1663.54 | 1681.35 | SL hit (close>ema200) qty=0.50 sl=1663.54 alert=retest2 |

### Cycle 67 — BUY (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 13:15:00 | 1687.00 | 1673.80 | 1672.62 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 1660.50 | 1671.90 | 1672.42 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 14:15:00 | 1689.50 | 1674.93 | 1673.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 1707.70 | 1684.06 | 1677.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 13:15:00 | 1711.90 | 1713.16 | 1702.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 14:00:00 | 1711.90 | 1713.16 | 1702.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 1705.70 | 1711.58 | 1703.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 1705.20 | 1713.36 | 1705.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 1713.50 | 1717.71 | 1710.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:00:00 | 1713.50 | 1717.71 | 1710.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 1712.60 | 1716.69 | 1710.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 1712.60 | 1716.69 | 1710.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1744.00 | 1722.04 | 1714.05 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 1703.80 | 1716.14 | 1716.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 11:15:00 | 1698.00 | 1707.32 | 1710.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1630.80 | 1621.87 | 1641.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 10:30:00 | 1633.20 | 1621.87 | 1641.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 1636.00 | 1624.70 | 1641.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 1642.80 | 1624.70 | 1641.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1626.50 | 1619.17 | 1631.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 1641.10 | 1619.17 | 1631.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 1640.90 | 1623.33 | 1630.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 1640.90 | 1623.33 | 1630.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 1645.00 | 1627.66 | 1631.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 1645.00 | 1627.66 | 1631.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 1669.20 | 1635.97 | 1635.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 1701.50 | 1656.12 | 1644.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 15:15:00 | 1735.40 | 1736.76 | 1713.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 09:15:00 | 1718.00 | 1736.76 | 1713.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1707.40 | 1730.89 | 1712.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:00:00 | 1707.40 | 1730.89 | 1712.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 1702.90 | 1725.29 | 1711.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 1702.90 | 1725.29 | 1711.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 1692.80 | 1718.79 | 1710.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:00:00 | 1692.80 | 1718.79 | 1710.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 1678.00 | 1700.76 | 1703.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 15:15:00 | 1673.00 | 1695.21 | 1700.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 12:15:00 | 1669.10 | 1660.07 | 1671.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 12:15:00 | 1669.10 | 1660.07 | 1671.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 1669.10 | 1660.07 | 1671.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:00:00 | 1669.10 | 1660.07 | 1671.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 1656.50 | 1659.36 | 1670.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:15:00 | 1650.00 | 1658.49 | 1668.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 09:15:00 | 1693.40 | 1653.35 | 1652.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 09:15:00 | 1693.40 | 1653.35 | 1652.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 11:15:00 | 1713.90 | 1676.92 | 1665.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 09:15:00 | 1888.60 | 1896.60 | 1870.02 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 13:30:00 | 1912.00 | 1891.68 | 1875.75 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-22 09:15:00 | 1946.00 | 1894.62 | 1879.96 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 1894.00 | 1915.60 | 1903.29 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-23 09:15:00 | 1894.00 | 1915.60 | 1903.29 | SL hit (close<ema400) qty=1.00 sl=1903.29 alert=retest1 |

### Cycle 74 — SELL (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 14:15:00 | 1882.30 | 1894.74 | 1896.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 12:15:00 | 1869.00 | 1884.79 | 1890.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 14:15:00 | 1894.00 | 1885.96 | 1890.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 14:15:00 | 1894.00 | 1885.96 | 1890.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 1894.00 | 1885.96 | 1890.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 15:00:00 | 1894.00 | 1885.96 | 1890.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 1898.00 | 1888.37 | 1890.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 1952.30 | 1888.37 | 1890.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 09:15:00 | 1942.60 | 1899.22 | 1895.49 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 12:15:00 | 1886.70 | 1905.67 | 1907.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 1871.00 | 1898.74 | 1904.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 15:15:00 | 1795.00 | 1790.20 | 1817.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-01 09:15:00 | 1814.20 | 1790.20 | 1817.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1838.70 | 1799.90 | 1819.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:00:00 | 1838.70 | 1799.90 | 1819.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 1858.00 | 1811.52 | 1823.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:30:00 | 1889.60 | 1811.52 | 1823.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 1855.00 | 1832.55 | 1830.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 1862.20 | 1838.48 | 1833.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 1855.00 | 1857.77 | 1849.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 1855.00 | 1857.77 | 1849.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 1855.00 | 1857.77 | 1849.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:30:00 | 1851.90 | 1857.77 | 1849.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 1842.50 | 1854.72 | 1849.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 1840.70 | 1854.72 | 1849.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 1839.20 | 1851.61 | 1848.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:00:00 | 1839.20 | 1851.61 | 1848.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 1848.10 | 1851.63 | 1849.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 1848.10 | 1851.63 | 1849.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 1844.80 | 1850.27 | 1849.20 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 14:15:00 | 1841.30 | 1847.30 | 1847.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 1812.50 | 1839.17 | 1844.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 09:15:00 | 1783.10 | 1769.41 | 1779.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 1783.10 | 1769.41 | 1779.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 1783.10 | 1769.41 | 1779.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:30:00 | 1795.20 | 1769.41 | 1779.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 1773.20 | 1770.17 | 1778.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:15:00 | 1771.60 | 1770.17 | 1778.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 12:15:00 | 1769.90 | 1771.04 | 1778.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 13:15:00 | 1790.50 | 1775.87 | 1779.50 | SL hit (close>static) qty=1.00 sl=1788.30 alert=retest2 |

### Cycle 79 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 1793.00 | 1783.20 | 1782.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 12:15:00 | 1805.00 | 1790.15 | 1785.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 10:15:00 | 1792.10 | 1793.37 | 1789.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 10:15:00 | 1792.10 | 1793.37 | 1789.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 1792.10 | 1793.37 | 1789.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:00:00 | 1792.10 | 1793.37 | 1789.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 1788.90 | 1792.48 | 1789.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:45:00 | 1788.70 | 1792.48 | 1789.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 1785.50 | 1791.08 | 1788.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:00:00 | 1785.50 | 1791.08 | 1788.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 1792.00 | 1791.27 | 1789.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 09:30:00 | 1800.70 | 1793.25 | 1790.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 11:15:00 | 1801.00 | 1792.96 | 1790.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 15:00:00 | 1795.00 | 1793.04 | 1791.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 1797.60 | 1792.24 | 1791.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 1783.00 | 1790.39 | 1790.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 09:15:00 | 1783.00 | 1790.39 | 1790.53 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 1795.50 | 1790.74 | 1790.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 15:15:00 | 1800.80 | 1793.93 | 1792.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 1787.00 | 1807.08 | 1802.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 14:15:00 | 1787.00 | 1807.08 | 1802.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1787.00 | 1807.08 | 1802.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 1787.00 | 1807.08 | 1802.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 1792.00 | 1804.06 | 1801.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 1820.00 | 1804.06 | 1801.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 14:15:00 | 1810.20 | 1812.85 | 1813.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 14:15:00 | 1810.20 | 1812.85 | 1813.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 14:15:00 | 1803.30 | 1809.89 | 1811.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 09:15:00 | 1791.20 | 1790.43 | 1798.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-31 09:45:00 | 1789.00 | 1790.43 | 1798.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1785.60 | 1789.47 | 1797.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:45:00 | 1790.60 | 1789.47 | 1797.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 1789.00 | 1789.37 | 1796.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 1778.00 | 1789.65 | 1792.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 1689.10 | 1713.16 | 1739.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 12:15:00 | 1713.00 | 1708.77 | 1730.61 | SL hit (close>ema200) qty=0.50 sl=1708.77 alert=retest2 |

### Cycle 83 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 1765.00 | 1741.02 | 1738.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 1796.70 | 1766.80 | 1757.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 1728.00 | 1773.92 | 1768.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 1728.00 | 1773.92 | 1768.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1728.00 | 1773.92 | 1768.34 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 10:15:00 | 1721.50 | 1763.43 | 1764.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 11:15:00 | 1707.20 | 1752.19 | 1758.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 11:15:00 | 1745.80 | 1729.68 | 1740.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 11:15:00 | 1745.80 | 1729.68 | 1740.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 1745.80 | 1729.68 | 1740.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 12:00:00 | 1745.80 | 1729.68 | 1740.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 1724.60 | 1728.66 | 1739.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 13:15:00 | 1723.10 | 1728.66 | 1739.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 14:15:00 | 1721.00 | 1729.15 | 1734.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 15:00:00 | 1721.60 | 1727.64 | 1732.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 1752.80 | 1722.18 | 1718.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 11:15:00 | 1752.80 | 1722.18 | 1718.52 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 1700.00 | 1717.65 | 1718.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 09:15:00 | 1666.90 | 1697.92 | 1707.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 1689.50 | 1676.77 | 1688.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 1689.50 | 1676.77 | 1688.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 1689.50 | 1676.77 | 1688.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:45:00 | 1693.80 | 1676.77 | 1688.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 1696.00 | 1680.62 | 1689.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:00:00 | 1696.00 | 1680.62 | 1689.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 1667.30 | 1677.95 | 1687.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 12:15:00 | 1662.50 | 1677.95 | 1687.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 11:45:00 | 1664.90 | 1668.95 | 1673.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 13:00:00 | 1664.50 | 1668.06 | 1672.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 15:15:00 | 1663.90 | 1668.48 | 1671.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1655.30 | 1665.11 | 1669.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 10:30:00 | 1652.60 | 1662.89 | 1668.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 12:30:00 | 1652.50 | 1657.74 | 1664.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 1579.38 | 1617.95 | 1626.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 1581.65 | 1617.95 | 1626.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 1581.27 | 1617.95 | 1626.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 1580.70 | 1617.95 | 1626.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 1569.97 | 1617.95 | 1626.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 1569.88 | 1617.95 | 1626.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 1619.10 | 1614.01 | 1621.04 | SL hit (close>ema200) qty=0.50 sl=1614.01 alert=retest2 |

### Cycle 87 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 1551.00 | 1531.06 | 1530.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 1577.40 | 1540.33 | 1534.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 13:15:00 | 1647.70 | 1651.58 | 1632.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 13:45:00 | 1647.80 | 1651.58 | 1632.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 1652.00 | 1657.41 | 1648.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 1670.40 | 1657.41 | 1648.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 11:30:00 | 1655.50 | 1657.55 | 1650.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 1635.00 | 1653.04 | 1649.19 | SL hit (close<static) qty=1.00 sl=1647.70 alert=retest2 |

### Cycle 88 — SELL (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 14:15:00 | 1636.10 | 1646.68 | 1646.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 1617.20 | 1640.04 | 1643.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 1621.00 | 1620.36 | 1630.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:15:00 | 1630.40 | 1620.36 | 1630.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1621.10 | 1620.51 | 1629.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:45:00 | 1613.70 | 1619.43 | 1624.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:00:00 | 1612.10 | 1617.65 | 1623.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 13:00:00 | 1613.30 | 1616.78 | 1622.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 14:15:00 | 1613.00 | 1616.47 | 1621.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 1621.30 | 1617.43 | 1621.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 15:00:00 | 1621.30 | 1617.43 | 1621.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 1632.00 | 1620.35 | 1622.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:15:00 | 1634.70 | 1620.35 | 1622.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 1632.00 | 1622.68 | 1623.31 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 1635.50 | 1625.24 | 1624.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 1635.50 | 1625.24 | 1624.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 1663.00 | 1634.48 | 1629.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 1627.00 | 1636.69 | 1632.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 13:15:00 | 1627.00 | 1636.69 | 1632.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 1627.00 | 1636.69 | 1632.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 1627.00 | 1636.69 | 1632.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 1629.30 | 1635.21 | 1632.09 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 1613.40 | 1630.33 | 1630.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 1603.00 | 1617.26 | 1623.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 1608.20 | 1603.22 | 1611.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 15:00:00 | 1608.20 | 1603.22 | 1611.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 1612.50 | 1605.08 | 1611.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 1619.00 | 1605.08 | 1611.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1621.60 | 1608.38 | 1612.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:15:00 | 1607.30 | 1609.12 | 1612.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:45:00 | 1604.50 | 1607.60 | 1611.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 10:00:00 | 1602.00 | 1595.99 | 1603.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 11:15:00 | 1606.30 | 1599.05 | 1603.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 1590.80 | 1597.40 | 1602.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 12:30:00 | 1585.50 | 1596.10 | 1601.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 13:30:00 | 1583.40 | 1591.12 | 1598.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1526.93 | 1575.49 | 1589.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1524.27 | 1575.49 | 1589.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1521.90 | 1575.49 | 1589.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1525.98 | 1575.49 | 1589.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 1567.60 | 1564.36 | 1576.31 | SL hit (close>ema200) qty=0.50 sl=1564.36 alert=retest2 |

### Cycle 91 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 1505.00 | 1480.48 | 1477.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 09:15:00 | 1506.40 | 1487.98 | 1483.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-27 12:15:00 | 1488.60 | 1489.74 | 1485.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-27 12:45:00 | 1487.90 | 1489.74 | 1485.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 13:15:00 | 1483.10 | 1488.41 | 1485.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 14:00:00 | 1483.10 | 1488.41 | 1485.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 1511.00 | 1492.93 | 1487.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 15:15:00 | 1542.00 | 1492.93 | 1487.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 1553.30 | 1602.40 | 1605.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 1553.30 | 1602.40 | 1605.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 1536.00 | 1577.34 | 1592.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 10:15:00 | 1488.40 | 1486.26 | 1498.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 11:00:00 | 1488.40 | 1486.26 | 1498.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 1495.00 | 1487.75 | 1494.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 1518.60 | 1487.75 | 1494.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1509.00 | 1492.00 | 1495.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:30:00 | 1511.20 | 1492.00 | 1495.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 1514.70 | 1496.54 | 1497.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:45:00 | 1517.00 | 1496.54 | 1497.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 1514.90 | 1500.21 | 1498.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 1520.80 | 1504.33 | 1500.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 13:15:00 | 1530.90 | 1531.00 | 1519.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 14:00:00 | 1530.90 | 1531.00 | 1519.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1513.50 | 1526.15 | 1520.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 1510.40 | 1526.15 | 1520.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1512.80 | 1523.48 | 1519.55 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 14:15:00 | 1514.10 | 1516.99 | 1517.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 1507.30 | 1514.74 | 1516.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 15:15:00 | 1475.50 | 1469.31 | 1479.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 15:15:00 | 1475.50 | 1469.31 | 1479.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 1475.50 | 1469.31 | 1479.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 1541.70 | 1469.31 | 1479.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 1537.70 | 1482.99 | 1484.58 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 10:15:00 | 1529.60 | 1492.31 | 1488.67 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 09:15:00 | 1505.70 | 1516.35 | 1517.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 11:15:00 | 1500.80 | 1511.12 | 1514.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 15:15:00 | 1495.00 | 1493.60 | 1500.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 09:15:00 | 1496.40 | 1493.60 | 1500.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 1491.90 | 1493.26 | 1499.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:30:00 | 1484.30 | 1489.80 | 1496.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 1480.30 | 1487.94 | 1493.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 09:45:00 | 1483.50 | 1488.95 | 1493.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 10:15:00 | 1507.00 | 1492.56 | 1494.93 | SL hit (close>static) qty=1.00 sl=1504.60 alert=retest2 |

### Cycle 97 — BUY (started 2026-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 12:15:00 | 1503.10 | 1497.30 | 1496.83 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 1481.60 | 1495.98 | 1496.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 11:15:00 | 1477.00 | 1489.95 | 1493.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 15:15:00 | 1502.00 | 1487.31 | 1490.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 15:15:00 | 1502.00 | 1487.31 | 1490.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 1502.00 | 1487.31 | 1490.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 1521.20 | 1487.31 | 1490.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1481.20 | 1486.09 | 1489.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 1424.00 | 1462.50 | 1474.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 10:15:00 | 1424.50 | 1456.32 | 1470.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 11:30:00 | 1425.50 | 1417.91 | 1435.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 09:15:00 | 1502.40 | 1453.15 | 1447.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 1502.40 | 1453.15 | 1447.98 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2026-03-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 13:15:00 | 1445.70 | 1459.83 | 1460.96 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 1478.30 | 1462.86 | 1461.81 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 15:15:00 | 1457.10 | 1464.83 | 1465.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 1429.10 | 1457.68 | 1462.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 1358.00 | 1349.17 | 1378.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 09:45:00 | 1354.90 | 1349.17 | 1378.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1392.90 | 1360.71 | 1369.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 1398.30 | 1360.71 | 1369.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 1402.70 | 1369.11 | 1372.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:45:00 | 1403.90 | 1369.11 | 1372.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 1402.80 | 1380.18 | 1377.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 1407.80 | 1385.70 | 1380.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1383.10 | 1393.10 | 1385.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1383.10 | 1393.10 | 1385.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1383.10 | 1393.10 | 1385.55 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 1359.00 | 1383.15 | 1383.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 14:15:00 | 1335.90 | 1363.69 | 1372.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 1295.70 | 1292.51 | 1321.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 1295.70 | 1292.51 | 1321.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 1295.70 | 1292.51 | 1321.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 1287.20 | 1291.69 | 1318.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1348.90 | 1309.01 | 1315.47 | SL hit (close>static) qty=1.00 sl=1321.90 alert=retest2 |

### Cycle 105 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 1327.90 | 1321.07 | 1320.17 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 1286.60 | 1313.96 | 1317.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 1281.60 | 1307.48 | 1314.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1345.20 | 1245.53 | 1261.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1345.20 | 1245.53 | 1261.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1345.20 | 1245.53 | 1261.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 1370.40 | 1245.53 | 1261.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1343.50 | 1265.13 | 1269.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:00:00 | 1343.50 | 1265.13 | 1269.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 1352.30 | 1282.56 | 1276.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 1382.50 | 1348.64 | 1332.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 1421.50 | 1448.94 | 1427.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 1421.50 | 1448.94 | 1427.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1421.50 | 1448.94 | 1427.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:30:00 | 1442.40 | 1445.53 | 1429.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:45:00 | 1440.40 | 1442.58 | 1430.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 15:15:00 | 1440.00 | 1440.46 | 1430.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-21 09:15:00 | 1584.44 | 1550.94 | 1533.50 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 1705.20 | 1717.20 | 1718.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 13:15:00 | 1701.50 | 1711.82 | 1715.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 15:15:00 | 1718.00 | 1712.96 | 1715.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 15:15:00 | 1718.00 | 1712.96 | 1715.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 1718.00 | 1712.96 | 1715.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:15:00 | 1727.20 | 1712.96 | 1715.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 1746.90 | 1719.75 | 1718.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 10:15:00 | 1762.60 | 1728.32 | 1722.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 12:15:00 | 1782.50 | 1789.83 | 1772.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 13:00:00 | 1782.50 | 1789.83 | 1772.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 1770.20 | 1784.44 | 1772.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:00:00 | 1770.20 | 1784.44 | 1772.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 1769.40 | 1781.43 | 1772.64 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-29 09:30:00 | 1958.55 | 2024-05-31 15:15:00 | 1943.90 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-05-31 11:00:00 | 1943.25 | 2024-05-31 15:15:00 | 1943.90 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2024-05-31 11:30:00 | 1941.50 | 2024-05-31 15:15:00 | 1943.90 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2024-06-27 13:00:00 | 2233.95 | 2024-07-03 14:15:00 | 2448.27 | TARGET_HIT | 1.00 | 9.59% |
| BUY | retest2 | 2024-06-28 11:00:00 | 2225.70 | 2024-07-03 15:15:00 | 2457.34 | TARGET_HIT | 1.00 | 10.41% |
| BUY | retest2 | 2024-06-28 15:15:00 | 2228.00 | 2024-07-03 15:15:00 | 2450.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-02 13:00:00 | 2237.75 | 2024-07-03 15:15:00 | 2461.53 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-03 09:15:00 | 2276.00 | 2024-07-04 09:15:00 | 2503.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-01 09:15:00 | 2651.00 | 2024-08-01 10:15:00 | 2601.00 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-08-19 12:00:00 | 2185.00 | 2024-08-20 10:15:00 | 2075.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-19 12:00:00 | 2185.00 | 2024-08-21 09:15:00 | 2114.85 | STOP_HIT | 0.50 | 3.21% |
| SELL | retest2 | 2024-09-11 09:45:00 | 1820.00 | 2024-09-18 12:15:00 | 1729.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-11 10:45:00 | 1817.95 | 2024-09-18 12:15:00 | 1728.05 | PARTIAL | 0.50 | 4.95% |
| SELL | retest2 | 2024-09-13 10:45:00 | 1817.50 | 2024-09-18 13:15:00 | 1727.05 | PARTIAL | 0.50 | 4.98% |
| SELL | retest2 | 2024-09-13 11:30:00 | 1819.00 | 2024-09-18 13:15:00 | 1726.62 | PARTIAL | 0.50 | 5.08% |
| SELL | retest2 | 2024-09-16 10:45:00 | 1783.10 | 2024-09-19 09:15:00 | 1693.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-16 12:30:00 | 1786.30 | 2024-09-19 09:15:00 | 1696.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-17 09:45:00 | 1782.00 | 2024-09-19 09:15:00 | 1692.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-11 09:45:00 | 1820.00 | 2024-09-19 11:15:00 | 1638.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-11 10:45:00 | 1817.95 | 2024-09-19 15:15:00 | 1687.00 | STOP_HIT | 0.50 | 7.20% |
| SELL | retest2 | 2024-09-13 10:45:00 | 1817.50 | 2024-09-19 15:15:00 | 1687.00 | STOP_HIT | 0.50 | 7.18% |
| SELL | retest2 | 2024-09-13 11:30:00 | 1819.00 | 2024-09-19 15:15:00 | 1687.00 | STOP_HIT | 0.50 | 7.26% |
| SELL | retest2 | 2024-09-16 10:45:00 | 1783.10 | 2024-09-19 15:15:00 | 1687.00 | STOP_HIT | 0.50 | 5.39% |
| SELL | retest2 | 2024-09-16 12:30:00 | 1786.30 | 2024-09-19 15:15:00 | 1687.00 | STOP_HIT | 0.50 | 5.56% |
| SELL | retest2 | 2024-09-17 09:45:00 | 1782.00 | 2024-09-19 15:15:00 | 1687.00 | STOP_HIT | 0.50 | 5.33% |
| BUY | retest2 | 2024-09-24 12:15:00 | 1792.00 | 2024-09-25 09:15:00 | 1770.50 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-09-24 15:00:00 | 1791.85 | 2024-09-25 09:15:00 | 1770.50 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-09-30 09:15:00 | 1704.00 | 2024-10-04 14:15:00 | 1618.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 14:15:00 | 1701.20 | 2024-10-04 14:15:00 | 1616.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 10:00:00 | 1706.40 | 2024-10-04 14:15:00 | 1621.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 11:30:00 | 1704.65 | 2024-10-04 14:15:00 | 1619.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 09:15:00 | 1688.45 | 2024-10-04 14:15:00 | 1604.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 09:15:00 | 1704.00 | 2024-10-08 09:15:00 | 1533.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-30 14:15:00 | 1701.20 | 2024-10-08 09:15:00 | 1531.08 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-01 10:00:00 | 1706.40 | 2024-10-08 09:15:00 | 1535.76 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-01 11:30:00 | 1704.65 | 2024-10-08 09:15:00 | 1534.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-03 09:15:00 | 1688.45 | 2024-10-08 09:15:00 | 1519.61 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-11-04 13:00:00 | 1484.50 | 2024-11-05 12:15:00 | 1444.90 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2024-11-05 11:15:00 | 1488.40 | 2024-11-05 12:15:00 | 1444.90 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2024-11-05 15:00:00 | 1492.00 | 2024-11-08 10:15:00 | 1459.65 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2024-12-06 11:00:00 | 1699.00 | 2024-12-10 09:15:00 | 1652.00 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2024-12-06 12:00:00 | 1697.15 | 2024-12-10 09:15:00 | 1652.00 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2024-12-06 13:15:00 | 1698.45 | 2024-12-10 09:15:00 | 1652.00 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2024-12-06 13:45:00 | 1696.00 | 2024-12-10 09:15:00 | 1652.00 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2024-12-12 15:00:00 | 1653.10 | 2024-12-18 12:15:00 | 1570.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 15:00:00 | 1653.10 | 2024-12-20 15:15:00 | 1487.79 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-01-02 10:00:00 | 1606.85 | 2025-01-06 10:15:00 | 1518.60 | STOP_HIT | 1.00 | -5.49% |
| BUY | retest2 | 2025-01-02 11:15:00 | 1589.75 | 2025-01-06 10:15:00 | 1518.60 | STOP_HIT | 1.00 | -4.48% |
| BUY | retest2 | 2025-01-02 11:45:00 | 1591.05 | 2025-01-06 10:15:00 | 1518.60 | STOP_HIT | 1.00 | -4.55% |
| BUY | retest2 | 2025-01-02 15:00:00 | 1592.95 | 2025-01-06 10:15:00 | 1518.60 | STOP_HIT | 1.00 | -4.67% |
| SELL | retest2 | 2025-01-09 11:15:00 | 1492.50 | 2025-01-10 09:15:00 | 1417.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 11:15:00 | 1492.50 | 2025-01-14 09:15:00 | 1401.60 | STOP_HIT | 0.50 | 6.09% |
| SELL | retest2 | 2025-01-24 09:45:00 | 1457.30 | 2025-01-27 10:15:00 | 1390.09 | PARTIAL | 0.50 | 4.61% |
| SELL | retest2 | 2025-01-24 11:00:00 | 1463.25 | 2025-01-28 09:15:00 | 1384.43 | PARTIAL | 0.50 | 5.39% |
| SELL | retest2 | 2025-01-24 11:45:00 | 1459.25 | 2025-01-28 09:15:00 | 1386.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:45:00 | 1457.30 | 2025-01-28 11:15:00 | 1417.05 | STOP_HIT | 0.50 | 2.76% |
| SELL | retest2 | 2025-01-24 11:00:00 | 1463.25 | 2025-01-28 11:15:00 | 1417.05 | STOP_HIT | 0.50 | 3.16% |
| SELL | retest2 | 2025-01-24 11:45:00 | 1459.25 | 2025-01-28 11:15:00 | 1417.05 | STOP_HIT | 0.50 | 2.89% |
| SELL | retest2 | 2025-02-11 13:00:00 | 1325.05 | 2025-02-12 09:15:00 | 1258.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 14:30:00 | 1326.90 | 2025-02-12 09:15:00 | 1260.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 15:15:00 | 1324.45 | 2025-02-12 09:15:00 | 1258.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 13:00:00 | 1325.05 | 2025-02-12 15:15:00 | 1305.00 | STOP_HIT | 0.50 | 1.51% |
| SELL | retest2 | 2025-02-11 14:30:00 | 1326.90 | 2025-02-12 15:15:00 | 1305.00 | STOP_HIT | 0.50 | 1.65% |
| SELL | retest2 | 2025-02-11 15:15:00 | 1324.45 | 2025-02-12 15:15:00 | 1305.00 | STOP_HIT | 0.50 | 1.47% |
| SELL | retest2 | 2025-02-13 11:15:00 | 1318.50 | 2025-02-14 12:15:00 | 1252.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 14:00:00 | 1312.60 | 2025-02-14 12:15:00 | 1246.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 11:15:00 | 1318.50 | 2025-02-18 09:15:00 | 1186.65 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-13 14:00:00 | 1312.60 | 2025-02-18 09:15:00 | 1181.34 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-03 09:30:00 | 1240.80 | 2025-03-05 10:15:00 | 1270.00 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-03-04 12:15:00 | 1245.30 | 2025-03-05 10:15:00 | 1270.00 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-03-04 13:00:00 | 1245.60 | 2025-03-05 10:15:00 | 1270.00 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-03-04 13:30:00 | 1247.70 | 2025-03-05 10:15:00 | 1270.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-03-24 09:15:00 | 1475.70 | 2025-03-27 14:15:00 | 1422.90 | STOP_HIT | 1.00 | -3.58% |
| SELL | retest2 | 2025-04-09 10:00:00 | 1338.30 | 2025-04-11 09:15:00 | 1430.65 | STOP_HIT | 1.00 | -6.90% |
| SELL | retest1 | 2025-05-05 14:15:00 | 1537.20 | 2025-05-07 09:15:00 | 1460.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-05-05 14:45:00 | 1539.90 | 2025-05-07 09:15:00 | 1462.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-05-06 09:15:00 | 1531.70 | 2025-05-07 13:15:00 | 1455.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-05-05 14:15:00 | 1537.20 | 2025-05-08 09:15:00 | 1482.00 | STOP_HIT | 0.50 | 3.59% |
| SELL | retest1 | 2025-05-05 14:45:00 | 1539.90 | 2025-05-08 09:15:00 | 1482.00 | STOP_HIT | 0.50 | 3.76% |
| SELL | retest1 | 2025-05-06 09:15:00 | 1531.70 | 2025-05-08 09:15:00 | 1482.00 | STOP_HIT | 0.50 | 3.24% |
| SELL | retest2 | 2025-05-06 11:30:00 | 1504.60 | 2025-05-09 09:15:00 | 1429.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 11:30:00 | 1504.60 | 2025-05-09 09:15:00 | 1463.00 | STOP_HIT | 0.50 | 2.76% |
| SELL | retest2 | 2025-05-06 13:00:00 | 1505.90 | 2025-05-09 09:15:00 | 1430.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 13:00:00 | 1505.90 | 2025-05-09 09:15:00 | 1463.00 | STOP_HIT | 0.50 | 2.85% |
| BUY | retest1 | 2025-06-05 12:30:00 | 2151.00 | 2025-06-05 13:15:00 | 2258.55 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-05 12:30:00 | 2151.00 | 2025-06-05 15:15:00 | 2366.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-06-13 12:30:00 | 2163.80 | 2025-06-17 09:15:00 | 2208.90 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-06-13 13:15:00 | 2150.80 | 2025-06-17 09:15:00 | 2208.90 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-06-13 14:15:00 | 2156.80 | 2025-06-17 09:15:00 | 2208.90 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-06-16 09:30:00 | 2130.00 | 2025-06-17 09:15:00 | 2208.90 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2025-06-16 13:15:00 | 2184.00 | 2025-06-17 09:15:00 | 2208.90 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-06-16 14:45:00 | 2180.50 | 2025-06-17 09:15:00 | 2208.90 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-06-19 11:30:00 | 2141.30 | 2025-06-20 15:15:00 | 2204.00 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2025-06-20 12:15:00 | 2122.60 | 2025-06-20 15:15:00 | 2204.00 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2025-06-27 11:30:00 | 2093.00 | 2025-07-04 11:15:00 | 2061.90 | STOP_HIT | 1.00 | 1.49% |
| SELL | retest2 | 2025-06-27 13:30:00 | 2093.00 | 2025-07-04 11:15:00 | 2061.90 | STOP_HIT | 1.00 | 1.49% |
| SELL | retest2 | 2025-07-09 11:45:00 | 2045.40 | 2025-07-09 12:15:00 | 2045.60 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-07-09 12:15:00 | 2044.90 | 2025-07-09 12:15:00 | 2045.60 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-07-17 11:30:00 | 1915.10 | 2025-07-25 10:15:00 | 1819.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-17 11:30:00 | 1915.10 | 2025-07-29 09:15:00 | 1723.59 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-06 09:15:00 | 1723.90 | 2025-08-08 15:15:00 | 1637.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-06 09:15:00 | 1723.90 | 2025-08-11 09:15:00 | 1664.80 | STOP_HIT | 0.50 | 3.43% |
| SELL | retest2 | 2025-09-08 15:15:00 | 1650.00 | 2025-09-11 09:15:00 | 1693.40 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest1 | 2025-09-19 13:30:00 | 1912.00 | 2025-09-23 09:15:00 | 1894.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest1 | 2025-09-22 09:15:00 | 1946.00 | 2025-09-23 09:15:00 | 1894.00 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-10-14 11:15:00 | 1771.60 | 2025-10-14 13:15:00 | 1790.50 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-10-14 12:15:00 | 1769.90 | 2025-10-14 13:15:00 | 1790.50 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-10-17 09:30:00 | 1800.70 | 2025-10-20 09:15:00 | 1783.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-10-17 11:15:00 | 1801.00 | 2025-10-20 09:15:00 | 1783.00 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-10-17 15:00:00 | 1795.00 | 2025-10-20 09:15:00 | 1783.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-10-20 09:15:00 | 1797.60 | 2025-10-20 09:15:00 | 1783.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-10-24 09:15:00 | 1820.00 | 2025-10-28 14:15:00 | 1810.20 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-11-04 09:15:00 | 1778.00 | 2025-11-07 09:15:00 | 1689.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 09:15:00 | 1778.00 | 2025-11-07 12:15:00 | 1713.00 | STOP_HIT | 0.50 | 3.66% |
| SELL | retest2 | 2025-11-14 13:15:00 | 1723.10 | 2025-11-20 11:15:00 | 1752.80 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-11-17 14:15:00 | 1721.00 | 2025-11-20 11:15:00 | 1752.80 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-11-17 15:00:00 | 1721.60 | 2025-11-20 11:15:00 | 1752.80 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-11-25 12:15:00 | 1662.50 | 2025-12-09 09:15:00 | 1579.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 11:45:00 | 1664.90 | 2025-12-09 09:15:00 | 1581.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 13:00:00 | 1664.50 | 2025-12-09 09:15:00 | 1581.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 15:15:00 | 1663.90 | 2025-12-09 09:15:00 | 1580.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 10:30:00 | 1652.60 | 2025-12-09 09:15:00 | 1569.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 12:30:00 | 1652.50 | 2025-12-09 09:15:00 | 1569.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-25 12:15:00 | 1662.50 | 2025-12-09 14:15:00 | 1619.10 | STOP_HIT | 0.50 | 2.61% |
| SELL | retest2 | 2025-12-01 11:45:00 | 1664.90 | 2025-12-09 14:15:00 | 1619.10 | STOP_HIT | 0.50 | 2.75% |
| SELL | retest2 | 2025-12-01 13:00:00 | 1664.50 | 2025-12-09 14:15:00 | 1619.10 | STOP_HIT | 0.50 | 2.73% |
| SELL | retest2 | 2025-12-01 15:15:00 | 1663.90 | 2025-12-09 14:15:00 | 1619.10 | STOP_HIT | 0.50 | 2.69% |
| SELL | retest2 | 2025-12-02 10:30:00 | 1652.60 | 2025-12-09 14:15:00 | 1619.10 | STOP_HIT | 0.50 | 2.03% |
| SELL | retest2 | 2025-12-02 12:30:00 | 1652.50 | 2025-12-09 14:15:00 | 1619.10 | STOP_HIT | 0.50 | 2.02% |
| BUY | retest2 | 2025-12-29 09:15:00 | 1670.40 | 2025-12-29 12:15:00 | 1635.00 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-12-29 11:30:00 | 1655.50 | 2025-12-29 12:15:00 | 1635.00 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-01-01 09:45:00 | 1613.70 | 2026-01-02 10:15:00 | 1635.50 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2026-01-01 12:00:00 | 1612.10 | 2026-01-02 10:15:00 | 1635.50 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-01-01 13:00:00 | 1613.30 | 2026-01-02 10:15:00 | 1635.50 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2026-01-01 14:15:00 | 1613.00 | 2026-01-02 10:15:00 | 1635.50 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-01-08 11:15:00 | 1607.30 | 2026-01-12 09:15:00 | 1526.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:45:00 | 1604.50 | 2026-01-12 09:15:00 | 1524.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 10:00:00 | 1602.00 | 2026-01-12 09:15:00 | 1521.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 11:15:00 | 1606.30 | 2026-01-12 09:15:00 | 1525.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:15:00 | 1607.30 | 2026-01-12 15:15:00 | 1567.60 | STOP_HIT | 0.50 | 2.47% |
| SELL | retest2 | 2026-01-08 11:45:00 | 1604.50 | 2026-01-12 15:15:00 | 1567.60 | STOP_HIT | 0.50 | 2.30% |
| SELL | retest2 | 2026-01-09 10:00:00 | 1602.00 | 2026-01-12 15:15:00 | 1567.60 | STOP_HIT | 0.50 | 2.15% |
| SELL | retest2 | 2026-01-09 11:15:00 | 1606.30 | 2026-01-12 15:15:00 | 1567.60 | STOP_HIT | 0.50 | 2.41% |
| SELL | retest2 | 2026-01-09 12:30:00 | 1585.50 | 2026-01-19 09:15:00 | 1506.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 13:30:00 | 1583.40 | 2026-01-19 09:15:00 | 1504.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 12:30:00 | 1585.50 | 2026-01-20 15:15:00 | 1426.95 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-09 13:30:00 | 1583.40 | 2026-01-20 15:15:00 | 1425.06 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-01-27 15:15:00 | 1542.00 | 2026-02-01 14:15:00 | 1553.30 | STOP_HIT | 1.00 | 0.73% |
| SELL | retest2 | 2026-02-25 12:30:00 | 1484.30 | 2026-02-26 10:15:00 | 1507.00 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2026-02-26 09:15:00 | 1480.30 | 2026-02-26 10:15:00 | 1507.00 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2026-02-26 09:45:00 | 1483.50 | 2026-02-26 10:15:00 | 1507.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2026-03-04 09:15:00 | 1424.00 | 2026-03-06 09:15:00 | 1502.40 | STOP_HIT | 1.00 | -5.51% |
| SELL | retest2 | 2026-03-04 10:15:00 | 1424.50 | 2026-03-06 09:15:00 | 1502.40 | STOP_HIT | 1.00 | -5.47% |
| SELL | retest2 | 2026-03-05 11:30:00 | 1425.50 | 2026-03-06 09:15:00 | 1502.40 | STOP_HIT | 1.00 | -5.39% |
| SELL | retest2 | 2026-03-24 10:30:00 | 1287.20 | 2026-03-25 09:15:00 | 1348.90 | STOP_HIT | 1.00 | -4.79% |
| BUY | retest2 | 2026-04-13 11:30:00 | 1442.40 | 2026-04-21 09:15:00 | 1584.44 | TARGET_HIT | 1.00 | 9.85% |
| BUY | retest2 | 2026-04-13 13:45:00 | 1440.40 | 2026-04-21 09:15:00 | 1584.00 | TARGET_HIT | 1.00 | 9.97% |
| BUY | retest2 | 2026-04-13 15:15:00 | 1440.00 | 2026-04-22 14:15:00 | 1586.64 | TARGET_HIT | 1.00 | 10.18% |
