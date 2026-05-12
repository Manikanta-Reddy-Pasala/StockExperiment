# Godrej Properties Ltd. (GODREJPROP)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1874.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT2_SKIP | 2 |
| ALERT3 | 57 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 42 |
| PARTIAL | 9 |
| TARGET_HIT | 4 |
| STOP_HIT | 40 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 53 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 35
- **Target hits / Stop hits / Partials:** 4 / 40 / 9
- **Avg / median % per leg:** 0.20% / -1.58%
- **Sum % (uncompounded):** 10.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 0 | 0.0% | 0 | 16 | 0 | -2.05% | -32.8% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.02% | -6.0% |
| BUY @ 3rd Alert (retest2) | 14 | 0 | 0.0% | 0 | 14 | 0 | -1.91% | -26.8% |
| SELL (all) | 37 | 18 | 48.6% | 4 | 24 | 9 | 1.18% | 43.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 37 | 18 | 48.6% | 4 | 24 | 9 | 1.18% | 43.6% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.02% | -6.0% |
| retest2 (combined) | 51 | 18 | 35.3% | 4 | 38 | 9 | 0.33% | 16.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 14:15:00 | 2912.30 | 2972.18 | 2972.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 12:15:00 | 2878.80 | 2954.92 | 2963.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-13 09:15:00 | 2967.90 | 2928.87 | 2947.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 09:15:00 | 2967.90 | 2928.87 | 2947.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 2967.90 | 2928.87 | 2947.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 10:00:00 | 2967.90 | 2928.87 | 2947.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 10:15:00 | 2963.70 | 2929.22 | 2947.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 11:00:00 | 2963.70 | 2929.22 | 2947.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 11:15:00 | 2951.50 | 2929.44 | 2947.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 15:15:00 | 2942.00 | 2930.23 | 2947.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 10:30:00 | 2945.20 | 2930.43 | 2947.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 11:15:00 | 2797.94 | 2916.42 | 2938.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-20 11:15:00 | 2919.15 | 2915.24 | 2936.86 | SL hit (close>ema200) qty=0.50 sl=2915.24 alert=retest2 |

### Cycle 2 — BUY (started 2024-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 11:15:00 | 3284.50 | 2959.15 | 2957.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-25 12:15:00 | 3336.95 | 2962.91 | 2959.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 09:15:00 | 2983.70 | 3037.63 | 3001.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 09:15:00 | 2983.70 | 3037.63 | 3001.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 2983.70 | 3037.63 | 3001.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:30:00 | 2978.50 | 3037.63 | 3001.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 2974.90 | 3037.01 | 3001.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 11:00:00 | 2974.90 | 3037.01 | 3001.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 14:15:00 | 2979.00 | 3012.59 | 2992.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 15:00:00 | 2979.00 | 3012.59 | 2992.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 15:15:00 | 2992.00 | 3012.38 | 2992.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 09:15:00 | 3040.60 | 3012.38 | 2992.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 12:45:00 | 3002.80 | 3012.28 | 2993.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-22 11:00:00 | 2995.00 | 3045.09 | 3015.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-22 11:15:00 | 2953.45 | 3044.18 | 3015.53 | SL hit (close<static) qty=1.00 sl=2976.25 alert=retest2 |

### Cycle 3 — SELL (started 2024-11-01 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-01 18:15:00 | 2882.00 | 2994.16 | 2994.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 2808.70 | 2992.32 | 2993.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 2830.45 | 2828.74 | 2896.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 10:00:00 | 2830.45 | 2828.74 | 2896.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 2999.25 | 2830.87 | 2895.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 10:15:00 | 3008.35 | 2830.87 | 2895.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 2830.00 | 2844.83 | 2898.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 11:15:00 | 2818.00 | 2844.65 | 2897.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 11:45:00 | 2812.05 | 2844.35 | 2897.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 12:45:00 | 2822.70 | 2844.13 | 2897.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 12:15:00 | 2808.00 | 2843.40 | 2895.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 2907.45 | 2839.69 | 2888.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 15:00:00 | 2907.45 | 2839.69 | 2888.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 15:15:00 | 2905.00 | 2840.34 | 2888.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:15:00 | 2931.25 | 2840.34 | 2888.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 2904.10 | 2842.47 | 2887.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 15:00:00 | 2904.10 | 2842.47 | 2887.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 2895.00 | 2842.99 | 2887.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:15:00 | 2890.55 | 2842.99 | 2887.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 2877.65 | 2843.80 | 2882.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 11:00:00 | 2863.85 | 2844.00 | 2882.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 12:15:00 | 2911.60 | 2848.00 | 2883.14 | SL hit (close>static) qty=1.00 sl=2910.10 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 11:15:00 | 2234.00 | 2157.84 | 2157.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 12:15:00 | 2248.00 | 2158.73 | 2157.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 15:15:00 | 2339.00 | 2342.41 | 2281.99 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 09:15:00 | 2352.50 | 2342.41 | 2281.99 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 09:45:00 | 2349.60 | 2342.42 | 2282.30 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 2297.50 | 2339.15 | 2283.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 2290.00 | 2339.15 | 2283.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 2280.10 | 2338.07 | 2283.84 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-02 15:15:00 | 2280.10 | 2338.07 | 2283.84 | SL hit (close<ema400) qty=1.00 sl=2283.84 alert=retest1 |

### Cycle 5 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 2101.40 | 2270.00 | 2270.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 2083.00 | 2263.29 | 2267.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 12:15:00 | 2047.00 | 2032.86 | 2097.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 13:00:00 | 2047.00 | 2032.86 | 2097.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 2079.00 | 2037.64 | 2094.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:30:00 | 2083.00 | 2037.64 | 2094.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 2094.50 | 2038.74 | 2094.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 12:30:00 | 2079.90 | 2039.13 | 2094.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 09:45:00 | 2085.80 | 2040.94 | 2094.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 11:15:00 | 2112.80 | 2042.13 | 2094.61 | SL hit (close>static) qty=1.00 sl=2106.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 11:15:00 | 2313.50 | 2097.01 | 2096.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 2329.00 | 2118.83 | 2107.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 14:15:00 | 2190.50 | 2192.59 | 2152.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 15:00:00 | 2190.50 | 2192.59 | 2152.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 2144.50 | 2192.12 | 2152.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:30:00 | 2154.40 | 2192.12 | 2152.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 2183.80 | 2192.04 | 2152.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 09:15:00 | 2190.90 | 2185.86 | 2152.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 13:00:00 | 2188.30 | 2185.67 | 2153.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:45:00 | 2195.00 | 2187.97 | 2156.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 15:00:00 | 2200.50 | 2188.19 | 2157.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 2156.30 | 2188.01 | 2158.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 2156.30 | 2188.01 | 2158.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 2159.10 | 2187.72 | 2158.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 2154.80 | 2187.72 | 2158.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 2165.00 | 2187.50 | 2158.60 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 2137.40 | 2185.80 | 2158.46 | SL hit (close<static) qty=1.00 sl=2144.50 alert=retest2 |

### Cycle 7 — SELL (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 12:15:00 | 2073.00 | 2139.59 | 2139.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 13:15:00 | 2062.30 | 2138.82 | 2139.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 10:15:00 | 2063.00 | 2049.00 | 2080.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-02 11:00:00 | 2063.00 | 2049.00 | 2080.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 2072.50 | 2050.03 | 2079.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 2076.40 | 2050.03 | 2079.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 2098.10 | 2050.51 | 2079.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:45:00 | 2090.80 | 2050.51 | 2079.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 2100.70 | 2051.01 | 2079.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:30:00 | 2107.80 | 2051.01 | 2079.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 1849.10 | 1786.01 | 1867.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:45:00 | 1846.30 | 1793.14 | 1867.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:45:00 | 1844.00 | 1793.64 | 1867.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 09:30:00 | 1841.40 | 1795.84 | 1864.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 14:15:00 | 1753.98 | 1796.61 | 1860.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 14:15:00 | 1751.80 | 1796.61 | 1860.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 14:15:00 | 1749.33 | 1796.61 | 1860.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 1809.00 | 1796.59 | 1860.17 | SL hit (close>ema200) qty=0.50 sl=1796.59 alert=retest2 |

### Cycle 8 — BUY (started 2026-05-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 13:15:00 | 1876.50 | 1740.27 | 1740.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 1918.90 | 1743.56 | 1741.76 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-09-13 15:15:00 | 2942.00 | 2024-09-19 11:15:00 | 2797.94 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2024-09-13 15:15:00 | 2942.00 | 2024-09-20 11:15:00 | 2919.15 | STOP_HIT | 0.50 | 0.78% |
| SELL | retest2 | 2024-09-16 10:30:00 | 2945.20 | 2024-09-20 14:15:00 | 2987.50 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-09-20 14:15:00 | 2943.00 | 2024-09-20 14:15:00 | 2987.50 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-10-10 09:15:00 | 3040.60 | 2024-10-22 11:15:00 | 2953.45 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2024-10-11 12:45:00 | 3002.80 | 2024-10-22 11:15:00 | 2953.45 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2024-10-22 11:00:00 | 2995.00 | 2024-10-22 11:15:00 | 2953.45 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-10-24 09:15:00 | 3046.00 | 2024-10-24 11:15:00 | 2954.85 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2024-11-27 11:15:00 | 2818.00 | 2024-12-11 12:15:00 | 2911.60 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2024-11-27 11:45:00 | 2812.05 | 2024-12-11 15:15:00 | 2919.90 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2024-11-27 12:45:00 | 2822.70 | 2024-12-11 15:15:00 | 2919.90 | STOP_HIT | 1.00 | -3.44% |
| SELL | retest2 | 2024-11-28 12:15:00 | 2808.00 | 2024-12-11 15:15:00 | 2919.90 | STOP_HIT | 1.00 | -3.99% |
| SELL | retest2 | 2024-12-10 11:00:00 | 2863.85 | 2024-12-11 15:15:00 | 2919.90 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-12-12 10:45:00 | 2864.25 | 2024-12-13 14:15:00 | 2914.10 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-12-12 12:00:00 | 2853.20 | 2024-12-13 14:15:00 | 2914.10 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-12-13 10:00:00 | 2849.95 | 2024-12-13 14:15:00 | 2914.10 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2024-12-26 10:00:00 | 2837.65 | 2025-01-06 10:15:00 | 2695.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-26 10:45:00 | 2830.00 | 2025-01-06 10:15:00 | 2688.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-27 12:45:00 | 2833.30 | 2025-01-06 10:15:00 | 2691.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-27 14:45:00 | 2827.00 | 2025-01-06 10:15:00 | 2685.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-26 10:00:00 | 2837.65 | 2025-01-08 10:15:00 | 2553.89 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-26 10:45:00 | 2830.00 | 2025-01-08 11:15:00 | 2547.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-27 12:45:00 | 2833.30 | 2025-01-08 11:15:00 | 2549.97 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-27 14:45:00 | 2827.00 | 2025-01-08 11:15:00 | 2544.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-25 09:30:00 | 2109.00 | 2025-04-30 09:15:00 | 2196.30 | STOP_HIT | 1.00 | -4.14% |
| SELL | retest2 | 2025-04-29 09:30:00 | 2137.10 | 2025-04-30 09:15:00 | 2196.30 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2025-04-29 10:15:00 | 2133.20 | 2025-04-30 09:15:00 | 2196.30 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-04-30 15:15:00 | 2142.20 | 2025-05-02 09:15:00 | 2204.50 | STOP_HIT | 1.00 | -2.91% |
| BUY | retest1 | 2025-07-01 09:15:00 | 2352.50 | 2025-07-02 15:15:00 | 2280.10 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest1 | 2025-07-01 09:45:00 | 2349.60 | 2025-07-02 15:15:00 | 2280.10 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2025-07-03 09:45:00 | 2299.80 | 2025-07-03 12:15:00 | 2273.30 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-07-04 09:30:00 | 2301.20 | 2025-07-04 12:15:00 | 2277.10 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-07-04 14:15:00 | 2295.20 | 2025-07-07 15:15:00 | 2276.70 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-07-08 12:30:00 | 2295.50 | 2025-07-09 09:15:00 | 2271.10 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-07-15 11:15:00 | 2285.00 | 2025-07-25 10:15:00 | 2245.00 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-07-16 13:30:00 | 2287.10 | 2025-07-25 10:15:00 | 2245.00 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-09-18 12:30:00 | 2079.90 | 2025-09-19 11:15:00 | 2112.80 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-09-19 09:45:00 | 2085.80 | 2025-09-19 11:15:00 | 2112.80 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-09-24 09:30:00 | 2087.20 | 2025-09-25 11:15:00 | 1982.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 09:30:00 | 2087.20 | 2025-10-03 09:15:00 | 2039.50 | STOP_HIT | 0.50 | 2.29% |
| SELL | retest2 | 2025-10-10 13:15:00 | 2087.30 | 2025-10-15 09:15:00 | 2129.80 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-10-14 11:15:00 | 2060.70 | 2025-10-15 09:15:00 | 2129.80 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest2 | 2025-11-12 09:15:00 | 2190.90 | 2025-11-19 09:15:00 | 2137.40 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-11-12 13:00:00 | 2188.30 | 2025-11-19 09:15:00 | 2137.40 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-11-14 10:45:00 | 2195.00 | 2025-11-19 09:15:00 | 2137.40 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-11-14 15:00:00 | 2200.50 | 2025-11-19 09:15:00 | 2137.40 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2026-02-19 11:45:00 | 1846.30 | 2026-02-24 14:15:00 | 1753.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 12:45:00 | 1844.00 | 2026-02-24 14:15:00 | 1751.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 09:30:00 | 1841.40 | 2026-02-24 14:15:00 | 1749.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 11:45:00 | 1846.30 | 2026-02-25 09:15:00 | 1809.00 | STOP_HIT | 0.50 | 2.02% |
| SELL | retest2 | 2026-02-19 12:45:00 | 1844.00 | 2026-02-25 09:15:00 | 1809.00 | STOP_HIT | 0.50 | 1.90% |
| SELL | retest2 | 2026-02-23 09:30:00 | 1841.40 | 2026-02-25 09:15:00 | 1809.00 | STOP_HIT | 0.50 | 1.76% |
| SELL | retest2 | 2026-04-30 09:15:00 | 1828.20 | 2026-05-04 09:15:00 | 1889.20 | STOP_HIT | 1.00 | -3.34% |
