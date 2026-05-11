# Hindustan Unilever Ltd. (HINDUNILVR)

## Backtest Summary

- **Window:** 2022-04-07 14:15:00 → 2026-05-08 15:15:00 (7049 bars)
- **Last close:** 2286.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 13 |
| ALERT2 | 13 |
| ALERT2_SKIP | 3 |
| ALERT3 | 67 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 52 |
| PARTIAL | 16 |
| TARGET_HIT | 5 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 68 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 35
- **Target hits / Stop hits / Partials:** 5 / 47 / 16
- **Avg / median % per leg:** 1.59% / -0.12%
- **Sum % (uncompounded):** 108.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 0 | 0.0% | 0 | 12 | 0 | -1.58% | -18.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 0 | 0.0% | 0 | 12 | 0 | -1.58% | -18.9% |
| SELL (all) | 56 | 33 | 58.9% | 5 | 35 | 16 | 2.27% | 127.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 56 | 33 | 58.9% | 5 | 35 | 16 | 2.27% | 127.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 68 | 33 | 48.5% | 5 | 47 | 16 | 1.59% | 108.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 14:15:00 | 2619.66 | 2483.02 | 2482.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-16 12:15:00 | 2626.65 | 2489.75 | 2486.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-09 12:15:00 | 2584.15 | 2586.27 | 2549.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-09 12:45:00 | 2582.14 | 2586.27 | 2549.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 09:15:00 | 2607.96 | 2631.82 | 2605.32 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2023-08-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 14:15:00 | 2529.66 | 2586.13 | 2586.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 09:15:00 | 2508.41 | 2584.78 | 2585.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-24 09:15:00 | 2541.46 | 2537.61 | 2555.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-24 09:30:00 | 2543.87 | 2537.61 | 2555.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 09:15:00 | 2509.49 | 2464.73 | 2490.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 09:45:00 | 2509.98 | 2464.73 | 2490.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 10:15:00 | 2514.76 | 2465.22 | 2490.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 11:00:00 | 2514.76 | 2465.22 | 2490.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 10:15:00 | 2505.90 | 2482.62 | 2495.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 11:00:00 | 2505.90 | 2482.62 | 2495.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 15:15:00 | 2474.97 | 2463.19 | 2478.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-09 09:15:00 | 2466.60 | 2463.19 | 2478.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-17 10:45:00 | 2471.47 | 2457.91 | 2472.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-17 12:15:00 | 2484.16 | 2458.41 | 2472.82 | SL hit (close>static) qty=1.00 sl=2483.77 alert=retest2 |

### Cycle 3 — BUY (started 2023-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-05 15:15:00 | 2521.15 | 2480.31 | 2480.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-06 09:15:00 | 2534.87 | 2480.85 | 2480.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-07 09:15:00 | 2472.90 | 2483.71 | 2481.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-07 09:15:00 | 2472.90 | 2483.71 | 2481.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 09:15:00 | 2472.90 | 2483.71 | 2481.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-07 09:30:00 | 2472.95 | 2483.71 | 2481.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 10:15:00 | 2475.75 | 2483.63 | 2481.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-07 10:30:00 | 2472.21 | 2483.63 | 2481.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 14:15:00 | 2477.72 | 2483.41 | 2481.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-07 15:00:00 | 2477.72 | 2483.41 | 2481.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 09:15:00 | 2487.41 | 2483.42 | 2481.89 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2023-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 10:15:00 | 2465.77 | 2480.42 | 2480.48 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-19 09:15:00 | 2506.93 | 2480.42 | 2480.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-19 11:15:00 | 2514.02 | 2481.06 | 2480.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 12:15:00 | 2524.74 | 2530.18 | 2510.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-08 13:00:00 | 2524.74 | 2530.18 | 2510.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 13:15:00 | 2503.20 | 2530.46 | 2512.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-11 14:00:00 | 2503.20 | 2530.46 | 2512.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 14:15:00 | 2493.61 | 2530.09 | 2512.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-11 15:00:00 | 2493.61 | 2530.09 | 2512.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 14:15:00 | 2502.75 | 2527.80 | 2511.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-12 14:45:00 | 2505.07 | 2527.80 | 2511.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 11:15:00 | 2510.23 | 2526.96 | 2511.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 11:30:00 | 2507.38 | 2526.96 | 2511.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 13:15:00 | 2521.69 | 2526.64 | 2511.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-15 14:15:00 | 2527.74 | 2526.64 | 2511.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-19 10:15:00 | 2500.10 | 2524.73 | 2512.52 | SL hit (close<static) qty=1.00 sl=2500.15 alert=retest2 |

### Cycle 6 — SELL (started 2024-01-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-24 13:15:00 | 2390.76 | 2501.33 | 2501.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 14:15:00 | 2381.52 | 2468.90 | 2483.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-01 09:15:00 | 2396.62 | 2389.58 | 2424.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-01 10:00:00 | 2396.62 | 2389.58 | 2424.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 09:15:00 | 2316.55 | 2223.62 | 2272.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 10:00:00 | 2316.55 | 2223.62 | 2272.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 10:15:00 | 2322.35 | 2224.60 | 2273.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 11:00:00 | 2322.35 | 2224.60 | 2273.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 2269.33 | 2256.23 | 2280.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-21 10:15:00 | 2266.67 | 2261.02 | 2281.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-22 09:15:00 | 2322.94 | 2262.32 | 2281.09 | SL hit (close>static) qty=1.00 sl=2298.84 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 10:15:00 | 2381.62 | 2295.45 | 2295.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-04 11:15:00 | 2398.93 | 2296.47 | 2295.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 11:15:00 | 2395.53 | 2397.07 | 2358.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-21 11:30:00 | 2401.34 | 2397.07 | 2358.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 2769.08 | 2831.73 | 2758.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:45:00 | 2755.66 | 2831.73 | 2758.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 2745.82 | 2830.29 | 2758.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 2745.82 | 2830.29 | 2758.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 2761.85 | 2829.61 | 2758.31 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2024-10-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 14:15:00 | 2485.93 | 2720.92 | 2721.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 2476.39 | 2663.02 | 2690.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 13:15:00 | 2362.38 | 2359.08 | 2436.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-03 14:00:00 | 2362.38 | 2359.08 | 2436.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 2429.86 | 2358.38 | 2426.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 11:00:00 | 2429.86 | 2358.38 | 2426.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 11:15:00 | 2408.37 | 2358.88 | 2426.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 14:00:00 | 2395.09 | 2359.74 | 2426.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 14:30:00 | 2395.09 | 2360.10 | 2426.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 2380.48 | 2360.50 | 2426.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 10:00:00 | 2388.35 | 2360.78 | 2426.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 2380.68 | 2365.95 | 2424.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 10:30:00 | 2366.37 | 2366.05 | 2424.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 11:15:00 | 2374.29 | 2366.05 | 2424.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 09:15:00 | 2275.34 | 2346.71 | 2401.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 09:15:00 | 2275.34 | 2346.71 | 2401.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 09:15:00 | 2261.46 | 2346.71 | 2401.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 09:15:00 | 2268.93 | 2346.71 | 2401.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 09:15:00 | 2248.05 | 2346.71 | 2401.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 09:15:00 | 2255.58 | 2346.71 | 2401.07 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-27 09:15:00 | 2359.38 | 2341.93 | 2394.89 | SL hit (close>ema200) qty=0.50 sl=2341.93 alert=retest2 |

### Cycle 9 — BUY (started 2025-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 14:15:00 | 2306.71 | 2276.05 | 2275.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 09:15:00 | 2328.45 | 2276.89 | 2276.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 2280.54 | 2286.19 | 2281.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 09:15:00 | 2280.54 | 2286.19 | 2281.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 2280.54 | 2286.19 | 2281.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 10:00:00 | 2280.54 | 2286.19 | 2281.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 10:15:00 | 2287.13 | 2286.20 | 2281.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 11:15:00 | 2289.99 | 2286.20 | 2281.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 13:15:00 | 2290.87 | 2286.27 | 2281.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 14:00:00 | 2290.08 | 2286.31 | 2281.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 14:30:00 | 2293.13 | 2286.39 | 2281.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 2293.43 | 2304.01 | 2292.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:45:00 | 2293.53 | 2304.01 | 2292.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 2296.97 | 2303.94 | 2292.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 11:45:00 | 2301.99 | 2303.84 | 2292.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 12:15:00 | 2287.23 | 2303.67 | 2292.89 | SL hit (close<static) qty=1.00 sl=2290.67 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 12:15:00 | 2240.31 | 2298.59 | 2298.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 13:15:00 | 2232.74 | 2297.93 | 2298.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 11:15:00 | 2283.10 | 2282.41 | 2289.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 12:00:00 | 2283.10 | 2282.41 | 2289.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 2283.20 | 2282.30 | 2289.36 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 11:15:00 | 2391.99 | 2296.04 | 2295.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 09:15:00 | 2475.01 | 2306.18 | 2301.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 11:15:00 | 2368.88 | 2375.75 | 2344.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 12:00:00 | 2368.88 | 2375.75 | 2344.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 2494.20 | 2546.03 | 2499.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 2494.20 | 2546.03 | 2499.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 2502.16 | 2545.60 | 2499.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 2497.54 | 2545.60 | 2499.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 2493.70 | 2545.08 | 2499.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:00:00 | 2493.70 | 2545.08 | 2499.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 2484.75 | 2544.48 | 2499.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:00:00 | 2484.75 | 2544.48 | 2499.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 2524.10 | 2542.07 | 2499.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 12:15:00 | 2528.03 | 2542.07 | 2499.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 12:45:00 | 2531.67 | 2541.95 | 2499.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 09:15:00 | 2473.44 | 2538.13 | 2499.75 | SL hit (close<static) qty=1.00 sl=2496.46 alert=retest2 |

### Cycle 12 — SELL (started 2025-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 13:15:00 | 2416.78 | 2487.39 | 2487.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 2409.99 | 2485.29 | 2486.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 11:15:00 | 2418.95 | 2418.52 | 2443.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-27 12:00:00 | 2418.95 | 2418.52 | 2443.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 2437.54 | 2418.77 | 2442.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:30:00 | 2440.88 | 2418.77 | 2442.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 2445.41 | 2419.03 | 2442.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:00:00 | 2445.41 | 2419.03 | 2442.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 2443.64 | 2419.28 | 2442.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 2443.64 | 2419.28 | 2442.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 2428.19 | 2419.37 | 2441.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 15:00:00 | 2421.80 | 2419.54 | 2441.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 2370.06 | 2420.45 | 2441.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:30:00 | 2421.99 | 2420.48 | 2441.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 12:15:00 | 2423.96 | 2420.53 | 2441.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 2422.29 | 2420.21 | 2439.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:45:00 | 2433.60 | 2420.21 | 2439.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 2361.60 | 2419.67 | 2439.52 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 2300.71 | 2419.67 | 2439.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 2300.89 | 2419.67 | 2439.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 2302.76 | 2419.67 | 2439.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-12-05 09:30:00 | 2462.19 | 2419.67 | 2439.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-12 09:15:00 | 2251.56 | 2388.13 | 2419.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 2335.20 | 2327.67 | 2369.08 | SL hit (close>ema200) qty=0.50 sl=2327.67 alert=retest2 |

### Cycle 13 — BUY (started 2026-02-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 13:15:00 | 2465.00 | 2379.87 | 2379.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 09:15:00 | 2466.90 | 2382.31 | 2380.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 10:15:00 | 2376.70 | 2382.25 | 2380.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 10:15:00 | 2376.70 | 2382.25 | 2380.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 2376.70 | 2382.25 | 2380.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:00:00 | 2376.70 | 2382.25 | 2380.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 2414.50 | 2382.57 | 2381.10 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 09:15:00 | 2296.30 | 2379.30 | 2379.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 2283.40 | 2364.72 | 2371.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 11:15:00 | 2360.10 | 2358.38 | 2367.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-24 12:00:00 | 2360.10 | 2358.38 | 2367.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 12:15:00 | 2367.80 | 2358.47 | 2367.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 13:00:00 | 2367.80 | 2358.47 | 2367.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 13:15:00 | 2363.00 | 2358.52 | 2367.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 14:00:00 | 2363.00 | 2358.52 | 2367.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 2367.10 | 2358.65 | 2367.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 10:45:00 | 2358.70 | 2358.66 | 2367.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 09:45:00 | 2357.30 | 2359.10 | 2367.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 14:15:00 | 2383.60 | 2359.83 | 2367.74 | SL hit (close>static) qty=1.00 sl=2379.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-11-09 09:15:00 | 2466.60 | 2023-11-17 12:15:00 | 2484.16 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2023-11-17 10:45:00 | 2471.47 | 2023-11-17 12:15:00 | 2484.16 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2023-11-20 10:00:00 | 2474.03 | 2023-11-22 14:15:00 | 2481.36 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2023-11-20 10:30:00 | 2472.56 | 2023-11-22 14:15:00 | 2481.36 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2023-11-21 14:45:00 | 2465.72 | 2023-11-23 13:15:00 | 2484.80 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2023-11-22 12:15:00 | 2463.21 | 2023-11-23 13:15:00 | 2484.80 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2023-11-28 15:15:00 | 2461.69 | 2023-11-29 11:15:00 | 2477.62 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2024-01-15 14:15:00 | 2527.74 | 2024-01-19 10:15:00 | 2500.10 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-01-19 14:30:00 | 2526.07 | 2024-01-20 09:15:00 | 2454.16 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2024-01-19 15:00:00 | 2525.72 | 2024-01-20 09:15:00 | 2454.16 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2024-05-21 10:15:00 | 2266.67 | 2024-05-22 09:15:00 | 2322.94 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-01-09 14:00:00 | 2395.09 | 2025-01-23 09:15:00 | 2275.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 14:30:00 | 2395.09 | 2025-01-23 09:15:00 | 2275.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 09:15:00 | 2380.48 | 2025-01-23 09:15:00 | 2261.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 10:00:00 | 2388.35 | 2025-01-23 09:15:00 | 2268.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-14 10:30:00 | 2366.37 | 2025-01-23 09:15:00 | 2248.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-14 11:15:00 | 2374.29 | 2025-01-23 09:15:00 | 2255.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 14:00:00 | 2395.09 | 2025-01-27 09:15:00 | 2359.38 | STOP_HIT | 0.50 | 1.49% |
| SELL | retest2 | 2025-01-09 14:30:00 | 2395.09 | 2025-01-27 09:15:00 | 2359.38 | STOP_HIT | 0.50 | 1.49% |
| SELL | retest2 | 2025-01-10 09:15:00 | 2380.48 | 2025-01-27 09:15:00 | 2359.38 | STOP_HIT | 0.50 | 0.89% |
| SELL | retest2 | 2025-01-10 10:00:00 | 2388.35 | 2025-01-27 09:15:00 | 2359.38 | STOP_HIT | 0.50 | 1.21% |
| SELL | retest2 | 2025-01-14 10:30:00 | 2366.37 | 2025-01-27 09:15:00 | 2359.38 | STOP_HIT | 0.50 | 0.30% |
| SELL | retest2 | 2025-01-14 11:15:00 | 2374.29 | 2025-01-27 09:15:00 | 2359.38 | STOP_HIT | 0.50 | 0.63% |
| SELL | retest2 | 2025-02-04 09:15:00 | 2372.66 | 2025-02-18 13:15:00 | 2254.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-04 10:15:00 | 2370.60 | 2025-02-18 13:15:00 | 2252.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-05 13:30:00 | 2369.27 | 2025-02-18 13:15:00 | 2250.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-05 14:00:00 | 2367.70 | 2025-02-19 09:15:00 | 2249.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-04 09:15:00 | 2372.66 | 2025-03-03 09:15:00 | 2135.39 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-04 10:15:00 | 2370.60 | 2025-03-03 09:15:00 | 2133.54 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-05 13:30:00 | 2369.27 | 2025-03-03 09:15:00 | 2132.34 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-05 14:00:00 | 2367.70 | 2025-03-03 09:15:00 | 2130.93 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-24 10:30:00 | 2360.81 | 2025-05-05 14:15:00 | 2306.71 | STOP_HIT | 1.00 | 2.29% |
| BUY | retest2 | 2025-05-09 11:15:00 | 2289.99 | 2025-05-22 12:15:00 | 2287.23 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-05-09 13:15:00 | 2290.87 | 2025-06-13 09:15:00 | 2272.38 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-05-09 14:00:00 | 2290.08 | 2025-06-13 09:15:00 | 2272.38 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-05-09 14:30:00 | 2293.13 | 2025-06-13 09:15:00 | 2272.38 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-05-22 11:45:00 | 2301.99 | 2025-06-13 09:15:00 | 2272.38 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-05-23 09:15:00 | 2304.25 | 2025-06-13 09:15:00 | 2272.38 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-09-24 12:15:00 | 2528.03 | 2025-09-26 09:15:00 | 2473.44 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-09-24 12:45:00 | 2531.67 | 2025-09-26 09:15:00 | 2473.44 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-10-17 09:15:00 | 2530.20 | 2025-10-24 09:15:00 | 2469.01 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-12-01 15:00:00 | 2421.80 | 2025-12-05 09:15:00 | 2300.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 09:15:00 | 2370.06 | 2025-12-05 09:15:00 | 2300.89 | PARTIAL | 0.50 | 2.92% |
| SELL | retest2 | 2025-12-03 10:30:00 | 2421.99 | 2025-12-05 09:15:00 | 2302.76 | PARTIAL | 0.50 | 4.92% |
| SELL | retest2 | 2025-12-03 12:15:00 | 2423.96 | 2025-12-12 09:15:00 | 2251.56 | PARTIAL | 0.50 | 7.11% |
| SELL | retest2 | 2025-12-01 15:00:00 | 2421.80 | 2026-01-02 09:15:00 | 2335.20 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2025-12-03 09:15:00 | 2370.06 | 2026-01-02 09:15:00 | 2335.20 | STOP_HIT | 0.50 | 1.47% |
| SELL | retest2 | 2025-12-03 10:30:00 | 2421.99 | 2026-01-02 09:15:00 | 2335.20 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2025-12-03 12:15:00 | 2423.96 | 2026-01-02 09:15:00 | 2335.20 | STOP_HIT | 0.50 | 3.66% |
| SELL | retest2 | 2026-01-09 13:30:00 | 2362.90 | 2026-01-12 09:15:00 | 2381.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2026-01-14 10:15:00 | 2365.30 | 2026-01-19 09:15:00 | 2382.90 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-01-16 10:30:00 | 2365.00 | 2026-01-19 09:15:00 | 2382.90 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-01-16 12:15:00 | 2358.00 | 2026-01-19 09:15:00 | 2382.90 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-01-28 10:30:00 | 2347.40 | 2026-02-03 11:15:00 | 2382.40 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2026-01-28 13:30:00 | 2352.40 | 2026-02-03 11:15:00 | 2382.40 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2026-01-29 09:15:00 | 2344.60 | 2026-02-05 09:15:00 | 2408.80 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2026-01-29 14:00:00 | 2352.00 | 2026-02-05 09:15:00 | 2408.80 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2026-02-01 10:00:00 | 2364.00 | 2026-02-05 09:15:00 | 2408.80 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2026-02-01 12:00:00 | 2347.30 | 2026-02-05 09:15:00 | 2408.80 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2026-02-03 15:00:00 | 2364.80 | 2026-02-05 09:15:00 | 2408.80 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2026-02-04 11:30:00 | 2362.50 | 2026-02-05 09:15:00 | 2408.80 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2026-02-05 13:00:00 | 2367.90 | 2026-02-06 12:15:00 | 2381.50 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2026-02-25 10:45:00 | 2358.70 | 2026-02-26 14:15:00 | 2383.60 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-02-26 09:45:00 | 2357.30 | 2026-02-26 14:15:00 | 2383.60 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-02-27 09:15:00 | 2356.30 | 2026-03-05 09:15:00 | 2238.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 09:15:00 | 2356.30 | 2026-03-12 12:15:00 | 2120.67 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-23 11:30:00 | 2358.00 | 2026-04-30 10:15:00 | 2240.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-23 11:30:00 | 2358.00 | 2026-04-30 10:15:00 | 2230.20 | STOP_HIT | 0.50 | 5.42% |
