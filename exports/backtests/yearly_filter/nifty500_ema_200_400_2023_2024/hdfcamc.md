# HDFC Asset Management Company Ltd. (HDFCAMC)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 2843.90
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
| ALERT2_SKIP | 3 |
| ALERT3 | 28 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 19 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 16
- **Target hits / Stop hits / Partials:** 0 / 19 / 4
- **Avg / median % per leg:** -0.35% / -1.73%
- **Sum % (uncompounded):** -7.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 0 | 0.0% | 0 | 11 | 0 | -2.25% | -24.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 0 | 0.0% | 0 | 11 | 0 | -2.25% | -24.7% |
| SELL (all) | 12 | 7 | 58.3% | 0 | 8 | 4 | 1.40% | 16.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 7 | 58.3% | 0 | 8 | 4 | 1.40% | 16.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 23 | 7 | 30.4% | 0 | 19 | 4 | -0.35% | -8.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 12:15:00 | 2123.32 | 2172.34 | 2172.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 13:15:00 | 2097.98 | 2168.19 | 2170.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 2167.30 | 2158.56 | 2164.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 09:15:00 | 2167.30 | 2158.56 | 2164.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 2167.30 | 2158.56 | 2164.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:45:00 | 2172.35 | 2158.56 | 2164.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 2172.50 | 2158.70 | 2164.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:30:00 | 2174.63 | 2158.70 | 2164.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 2162.50 | 2158.74 | 2164.95 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 15:15:00 | 2229.85 | 2170.09 | 2170.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 09:15:00 | 2232.23 | 2170.71 | 2170.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 09:15:00 | 2189.60 | 2193.37 | 2182.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 09:15:00 | 2189.60 | 2193.37 | 2182.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 2189.60 | 2193.37 | 2182.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 09:30:00 | 2213.32 | 2191.56 | 2182.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 11:15:00 | 2174.40 | 2191.43 | 2182.67 | SL hit (close<static) qty=1.00 sl=2181.40 alert=retest2 |

### Cycle 3 — SELL (started 2024-12-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 15:15:00 | 2130.00 | 2174.74 | 2174.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 12:15:00 | 2120.10 | 2173.06 | 2174.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 10:15:00 | 2000.58 | 1990.95 | 2049.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-05 11:00:00 | 2000.58 | 1990.95 | 2049.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 10:15:00 | 1945.35 | 1901.16 | 1948.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 10:30:00 | 1947.33 | 1901.16 | 1948.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 11:15:00 | 1954.58 | 1901.69 | 1948.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 12:00:00 | 1954.58 | 1901.69 | 1948.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 12:15:00 | 1962.00 | 1902.29 | 1948.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 12:45:00 | 1962.63 | 1902.29 | 1948.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 12:15:00 | 1980.78 | 1954.46 | 1966.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 13:15:00 | 1969.80 | 1954.46 | 1966.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 14:15:00 | 1987.53 | 1954.99 | 1966.99 | SL hit (close>static) qty=1.00 sl=1986.38 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 15:15:00 | 2100.00 | 1973.21 | 1973.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 09:15:00 | 2188.90 | 1975.36 | 1974.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 2112.55 | 2116.29 | 2062.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-09 09:30:00 | 2115.75 | 2116.29 | 2062.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 2816.25 | 2855.49 | 2779.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 13:15:00 | 2822.50 | 2855.49 | 2779.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 15:00:00 | 2816.50 | 2854.79 | 2779.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 11:15:00 | 2765.75 | 2852.04 | 2780.07 | SL hit (close<static) qty=1.00 sl=2776.50 alert=retest2 |

### Cycle 5 — SELL (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 13:15:00 | 2740.50 | 2771.79 | 2771.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 2723.75 | 2768.07 | 2769.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 15:15:00 | 2668.00 | 2667.34 | 2705.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-12 09:15:00 | 2662.80 | 2667.34 | 2705.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 2714.80 | 2652.70 | 2692.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 2714.80 | 2652.70 | 2692.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 2718.40 | 2653.35 | 2692.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:00:00 | 2718.40 | 2653.35 | 2692.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 2696.00 | 2655.91 | 2690.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 2687.10 | 2655.91 | 2690.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 2700.00 | 2656.35 | 2690.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:00:00 | 2700.00 | 2656.35 | 2690.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 2704.50 | 2656.83 | 2690.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:00:00 | 2704.50 | 2656.83 | 2690.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 2675.50 | 2656.87 | 2682.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:30:00 | 2681.40 | 2656.87 | 2682.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 2673.20 | 2657.03 | 2682.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 13:30:00 | 2671.30 | 2656.96 | 2682.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 12:15:00 | 2537.74 | 2645.27 | 2673.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 2684.70 | 2615.80 | 2654.06 | SL hit (close>ema200) qty=0.50 sl=2615.80 alert=retest2 |

### Cycle 6 — BUY (started 2026-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 13:15:00 | 2831.00 | 2642.74 | 2642.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 14:15:00 | 2836.00 | 2644.67 | 2642.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 11:15:00 | 2694.20 | 2701.21 | 2676.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-24 11:45:00 | 2689.10 | 2701.21 | 2676.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 2682.40 | 2700.98 | 2676.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:45:00 | 2681.60 | 2700.98 | 2676.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 14:15:00 | 2679.70 | 2700.77 | 2676.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 14:45:00 | 2675.50 | 2700.77 | 2676.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 2680.00 | 2706.54 | 2682.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 10:45:00 | 2685.20 | 2706.28 | 2682.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 2580.00 | 2702.24 | 2680.72 | SL hit (close<static) qty=1.00 sl=2625.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-03-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 13:15:00 | 2437.10 | 2661.85 | 2661.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 2431.70 | 2633.85 | 2647.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2523.90 | 2445.06 | 2523.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 2523.90 | 2445.06 | 2523.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 2523.90 | 2445.06 | 2523.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:00:00 | 2523.90 | 2445.06 | 2523.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 2527.20 | 2445.87 | 2523.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:00:00 | 2527.20 | 2445.87 | 2523.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 2533.20 | 2446.74 | 2523.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:45:00 | 2527.50 | 2446.74 | 2523.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 12:15:00 | 2512.60 | 2447.40 | 2523.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:45:00 | 2496.50 | 2450.96 | 2523.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-09 11:15:00 | 2539.70 | 2452.62 | 2523.46 | SL hit (close>static) qty=1.00 sl=2537.90 alert=retest2 |

### Cycle 8 — BUY (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 12:15:00 | 2734.80 | 2572.32 | 2571.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 11:15:00 | 2749.00 | 2590.63 | 2581.00 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-12-20 09:30:00 | 2213.32 | 2024-12-20 11:15:00 | 2174.40 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-04-01 13:15:00 | 1969.80 | 2025-04-01 14:15:00 | 1987.53 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-04-04 11:45:00 | 1959.15 | 2025-04-07 09:15:00 | 1861.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 13:30:00 | 1956.53 | 2025-04-07 09:15:00 | 1858.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 11:45:00 | 1959.15 | 2025-04-11 10:15:00 | 1951.85 | STOP_HIT | 0.50 | 0.37% |
| SELL | retest2 | 2025-04-04 13:30:00 | 1956.53 | 2025-04-11 10:15:00 | 1951.85 | STOP_HIT | 0.50 | 0.24% |
| BUY | retest2 | 2025-09-29 13:15:00 | 2822.50 | 2025-09-30 11:15:00 | 2765.75 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-09-29 15:00:00 | 2816.50 | 2025-09-30 11:15:00 | 2765.75 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-10-06 09:15:00 | 2824.25 | 2025-10-07 11:15:00 | 2766.50 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-10-06 13:15:00 | 2819.75 | 2025-10-07 11:15:00 | 2766.50 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-10-07 13:30:00 | 2792.50 | 2025-10-08 09:15:00 | 2754.25 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-10-13 09:15:00 | 2819.50 | 2025-10-24 13:15:00 | 2756.00 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-10-24 10:45:00 | 2792.50 | 2025-10-24 13:15:00 | 2756.00 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-10-27 09:15:00 | 2795.50 | 2025-10-29 09:15:00 | 2710.50 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2025-10-28 11:30:00 | 2804.50 | 2025-10-29 09:15:00 | 2710.50 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2026-01-05 13:30:00 | 2671.30 | 2026-01-09 12:15:00 | 2537.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 13:30:00 | 2671.30 | 2026-01-16 09:15:00 | 2684.70 | STOP_HIT | 0.50 | -0.50% |
| SELL | retest2 | 2026-01-16 10:30:00 | 2655.00 | 2026-01-20 14:15:00 | 2522.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 10:30:00 | 2655.00 | 2026-02-02 14:15:00 | 2575.20 | STOP_HIT | 0.50 | 3.01% |
| SELL | retest2 | 2026-02-03 14:30:00 | 2668.20 | 2026-02-04 09:15:00 | 2717.10 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-02-03 15:00:00 | 2666.60 | 2026-02-04 09:15:00 | 2717.10 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2026-03-02 10:45:00 | 2685.20 | 2026-03-04 09:15:00 | 2580.00 | STOP_HIT | 1.00 | -3.92% |
| SELL | retest2 | 2026-04-09 09:45:00 | 2496.50 | 2026-04-09 11:15:00 | 2539.70 | STOP_HIT | 1.00 | -1.73% |
