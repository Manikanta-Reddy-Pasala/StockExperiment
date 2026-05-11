# Indiamart Intermesh Ltd. (INDIAMART)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 2091.00
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
| ALERT3 | 31 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 37 |
| PARTIAL | 16 |
| TARGET_HIT | 8 |
| STOP_HIT | 29 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 53 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 38 / 15
- **Target hits / Stop hits / Partials:** 8 / 29 / 16
- **Avg / median % per leg:** 2.85% / 1.72%
- **Sum % (uncompounded):** 151.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 6 | 42.9% | 6 | 8 | 0 | 3.28% | 46.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 6 | 42.9% | 6 | 8 | 0 | 3.28% | 46.0% |
| SELL (all) | 39 | 32 | 82.1% | 2 | 21 | 16 | 2.70% | 105.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 39 | 32 | 82.1% | 2 | 21 | 16 | 2.70% | 105.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 53 | 38 | 71.7% | 8 | 29 | 16 | 2.85% | 151.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 14:15:00 | 2592.05 | 2644.39 | 2644.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 09:15:00 | 2577.60 | 2643.24 | 2643.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-10 11:15:00 | 2570.95 | 2561.21 | 2594.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-10 11:45:00 | 2575.05 | 2561.21 | 2594.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 2591.40 | 2562.34 | 2593.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-12 10:00:00 | 2591.40 | 2562.34 | 2593.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 10:15:00 | 2587.55 | 2562.59 | 2593.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-12 10:30:00 | 2594.95 | 2562.59 | 2593.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 11:15:00 | 2583.00 | 2562.80 | 2593.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-13 13:30:00 | 2579.05 | 2565.22 | 2593.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 12:15:00 | 2611.80 | 2565.64 | 2590.84 | SL hit (close>static) qty=1.00 sl=2594.80 alert=retest2 |

### Cycle 2 — BUY (started 2024-06-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 14:15:00 | 2680.20 | 2609.75 | 2609.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 13:15:00 | 2688.00 | 2613.77 | 2611.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 09:15:00 | 2791.65 | 2833.05 | 2751.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-02 10:00:00 | 2791.65 | 2833.05 | 2751.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 2729.00 | 2829.58 | 2752.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-07 14:00:00 | 2770.50 | 2806.12 | 2746.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 11:30:00 | 2768.00 | 2802.71 | 2750.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 13:00:00 | 2773.05 | 2785.33 | 2746.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 14:00:00 | 2750.30 | 2784.98 | 2746.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 2751.40 | 2784.65 | 2746.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:30:00 | 2734.00 | 2784.65 | 2746.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 15:15:00 | 2750.00 | 2784.30 | 2746.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 09:15:00 | 2775.05 | 2784.30 | 2746.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 10:00:00 | 2770.60 | 2784.17 | 2747.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-26 09:15:00 | 3025.33 | 2818.85 | 2771.64 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 13:15:00 | 2490.80 | 2901.37 | 2901.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 11:15:00 | 2462.65 | 2880.71 | 2891.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 2358.25 | 2318.25 | 2417.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-09 10:00:00 | 2358.25 | 2318.25 | 2417.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 13:15:00 | 2129.50 | 2043.07 | 2124.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 14:00:00 | 2129.50 | 2043.07 | 2124.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 14:15:00 | 2127.40 | 2043.91 | 2124.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 15:00:00 | 2127.40 | 2043.91 | 2124.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 15:15:00 | 2130.00 | 2044.77 | 2124.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 09:15:00 | 2123.35 | 2044.77 | 2124.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 2118.70 | 2045.50 | 2124.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-21 12:00:00 | 2101.00 | 2046.72 | 2124.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 10:00:00 | 2101.05 | 2056.20 | 2124.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 10:45:00 | 2102.00 | 2056.60 | 2124.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 10:45:00 | 2103.50 | 2059.47 | 2123.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 2107.40 | 2061.57 | 2115.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 2107.40 | 2061.57 | 2115.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 2105.00 | 2062.00 | 2115.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:15:00 | 2123.00 | 2062.00 | 2115.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 2116.55 | 2062.54 | 2115.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 09:30:00 | 2087.55 | 2065.48 | 2115.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 1995.95 | 2064.59 | 2112.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 1996.00 | 2064.59 | 2112.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 1996.90 | 2064.59 | 2112.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 1998.32 | 2064.59 | 2112.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 1983.17 | 2064.59 | 2112.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-11 10:15:00 | 2064.90 | 2047.51 | 2098.75 | SL hit (close>ema200) qty=0.50 sl=2047.51 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 13:15:00 | 2225.00 | 2132.90 | 2132.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 15:15:00 | 2239.70 | 2134.94 | 2133.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 2557.30 | 2561.96 | 2471.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 10:00:00 | 2557.30 | 2561.96 | 2471.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 2467.50 | 2567.16 | 2501.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:00:00 | 2467.50 | 2567.16 | 2501.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 2482.90 | 2566.32 | 2501.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 12:30:00 | 2490.20 | 2564.65 | 2501.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 13:00:00 | 2488.80 | 2564.65 | 2501.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 14:30:00 | 2487.50 | 2562.97 | 2500.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 09:15:00 | 2545.00 | 2562.19 | 2500.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 2510.00 | 2559.22 | 2501.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 2507.40 | 2559.22 | 2501.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 2495.50 | 2558.58 | 2501.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 14:00:00 | 2526.20 | 2556.76 | 2501.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 14:30:00 | 2519.50 | 2555.59 | 2506.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 15:00:00 | 2523.10 | 2555.59 | 2506.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 14:45:00 | 2519.20 | 2578.93 | 2553.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 09:15:00 | 2487.00 | 2577.49 | 2552.53 | SL hit (close<static) qty=1.00 sl=2488.60 alert=retest2 |

### Cycle 5 — SELL (started 2025-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 14:15:00 | 2375.00 | 2531.32 | 2531.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 15:15:00 | 2360.00 | 2529.61 | 2530.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 10:15:00 | 2439.00 | 2411.09 | 2452.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 10:15:00 | 2439.00 | 2411.09 | 2452.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 2439.00 | 2411.09 | 2452.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:00:00 | 2439.00 | 2411.09 | 2452.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 2461.00 | 2411.58 | 2452.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:30:00 | 2456.30 | 2411.58 | 2452.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 2439.60 | 2411.86 | 2452.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 10:30:00 | 2428.00 | 2423.96 | 2454.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 11:15:00 | 2465.00 | 2426.61 | 2454.40 | SL hit (close>static) qty=1.00 sl=2461.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-06-13 13:30:00 | 2579.05 | 2024-06-19 12:15:00 | 2611.80 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-08-07 14:00:00 | 2770.50 | 2024-08-26 09:15:00 | 3025.33 | TARGET_HIT | 1.00 | 9.20% |
| BUY | retest2 | 2024-08-12 11:30:00 | 2768.00 | 2024-08-28 11:15:00 | 3047.55 | TARGET_HIT | 1.00 | 10.10% |
| BUY | retest2 | 2024-08-16 13:00:00 | 2773.05 | 2024-08-28 11:15:00 | 3044.80 | TARGET_HIT | 1.00 | 9.80% |
| BUY | retest2 | 2024-08-16 14:00:00 | 2750.30 | 2024-08-28 11:15:00 | 3050.36 | TARGET_HIT | 1.00 | 10.91% |
| BUY | retest2 | 2024-08-19 09:15:00 | 2775.05 | 2024-08-28 11:15:00 | 3052.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-19 10:00:00 | 2770.60 | 2024-08-28 11:15:00 | 3047.66 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-21 12:00:00 | 2101.00 | 2025-04-07 09:15:00 | 1995.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 10:00:00 | 2101.05 | 2025-04-07 09:15:00 | 1996.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 10:45:00 | 2102.00 | 2025-04-07 09:15:00 | 1996.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-26 10:45:00 | 2103.50 | 2025-04-07 09:15:00 | 1998.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 09:30:00 | 2087.55 | 2025-04-07 09:15:00 | 1983.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-21 12:00:00 | 2101.00 | 2025-04-11 10:15:00 | 2064.90 | STOP_HIT | 0.50 | 1.72% |
| SELL | retest2 | 2025-03-25 10:00:00 | 2101.05 | 2025-04-11 10:15:00 | 2064.90 | STOP_HIT | 0.50 | 1.72% |
| SELL | retest2 | 2025-03-25 10:45:00 | 2102.00 | 2025-04-11 10:15:00 | 2064.90 | STOP_HIT | 0.50 | 1.76% |
| SELL | retest2 | 2025-03-26 10:45:00 | 2103.50 | 2025-04-11 10:15:00 | 2064.90 | STOP_HIT | 0.50 | 1.84% |
| SELL | retest2 | 2025-04-04 09:30:00 | 2087.55 | 2025-04-11 10:15:00 | 2064.90 | STOP_HIT | 0.50 | 1.09% |
| BUY | retest2 | 2025-08-05 12:30:00 | 2490.20 | 2025-09-19 09:15:00 | 2487.00 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-08-05 13:00:00 | 2488.80 | 2025-09-19 09:15:00 | 2487.00 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-08-05 14:30:00 | 2487.50 | 2025-09-19 09:15:00 | 2487.00 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2025-08-06 09:15:00 | 2545.00 | 2025-09-19 09:15:00 | 2487.00 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2025-08-07 14:00:00 | 2526.20 | 2025-09-22 09:15:00 | 2449.30 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2025-08-12 14:30:00 | 2519.50 | 2025-09-22 09:15:00 | 2449.30 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2025-08-12 15:00:00 | 2523.10 | 2025-09-22 09:15:00 | 2449.30 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2025-09-18 14:45:00 | 2519.20 | 2025-09-22 09:15:00 | 2449.30 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2025-10-31 10:30:00 | 2428.00 | 2025-11-03 11:15:00 | 2465.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-11-07 09:15:00 | 2426.40 | 2025-11-12 09:15:00 | 2484.20 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2025-11-10 09:15:00 | 2432.80 | 2025-11-12 09:15:00 | 2484.20 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-11-10 10:45:00 | 2432.40 | 2025-11-12 09:15:00 | 2484.20 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-11-11 09:15:00 | 2437.60 | 2025-11-12 09:15:00 | 2484.20 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-11-17 09:30:00 | 2442.00 | 2025-11-17 11:15:00 | 2466.60 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-11-18 10:00:00 | 2445.00 | 2025-11-21 10:15:00 | 2322.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 11:00:00 | 2441.10 | 2025-11-21 10:15:00 | 2319.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 10:00:00 | 2445.00 | 2025-12-29 10:15:00 | 2200.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-18 11:00:00 | 2441.10 | 2025-12-29 11:15:00 | 2196.99 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-19 11:45:00 | 2207.70 | 2026-03-02 09:15:00 | 2097.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-21 10:15:00 | 2198.50 | 2026-03-02 09:15:00 | 2088.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-21 11:30:00 | 2199.10 | 2026-03-02 09:15:00 | 2089.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 12:45:00 | 2203.00 | 2026-03-02 09:15:00 | 2092.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-06 12:15:00 | 2218.50 | 2026-03-02 09:15:00 | 2107.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-06 13:00:00 | 2212.00 | 2026-03-02 09:15:00 | 2101.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 09:15:00 | 2213.10 | 2026-03-02 09:15:00 | 2102.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 09:15:00 | 2205.90 | 2026-03-02 09:15:00 | 2095.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 11:00:00 | 2211.70 | 2026-03-02 09:15:00 | 2101.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 11:45:00 | 2207.70 | 2026-03-09 14:15:00 | 2184.80 | STOP_HIT | 0.50 | 1.04% |
| SELL | retest2 | 2026-01-21 10:15:00 | 2198.50 | 2026-03-09 14:15:00 | 2184.80 | STOP_HIT | 0.50 | 0.62% |
| SELL | retest2 | 2026-01-21 11:30:00 | 2199.10 | 2026-03-09 14:15:00 | 2184.80 | STOP_HIT | 0.50 | 0.65% |
| SELL | retest2 | 2026-01-22 12:45:00 | 2203.00 | 2026-03-09 14:15:00 | 2184.80 | STOP_HIT | 0.50 | 0.83% |
| SELL | retest2 | 2026-02-06 12:15:00 | 2218.50 | 2026-03-09 14:15:00 | 2184.80 | STOP_HIT | 0.50 | 1.52% |
| SELL | retest2 | 2026-02-06 13:00:00 | 2212.00 | 2026-03-09 14:15:00 | 2184.80 | STOP_HIT | 0.50 | 1.23% |
| SELL | retest2 | 2026-02-12 09:15:00 | 2213.10 | 2026-03-09 14:15:00 | 2184.80 | STOP_HIT | 0.50 | 1.28% |
| SELL | retest2 | 2026-02-18 09:15:00 | 2205.90 | 2026-03-09 14:15:00 | 2184.80 | STOP_HIT | 0.50 | 0.96% |
| SELL | retest2 | 2026-02-25 11:00:00 | 2211.70 | 2026-03-09 14:15:00 | 2184.80 | STOP_HIT | 0.50 | 1.22% |
