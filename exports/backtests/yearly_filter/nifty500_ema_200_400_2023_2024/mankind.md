# Mankind Pharma Ltd. (MANKIND)

## Backtest Summary

- **Window:** 2023-05-09 09:15:00 → 2026-05-11 15:15:00 (5198 bars)
- **Last close:** 2460.00
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
| ALERT2_SKIP | 1 |
| ALERT3 | 48 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 41 |
| PARTIAL | 9 |
| TARGET_HIT | 8 |
| STOP_HIT | 33 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 50 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 31
- **Target hits / Stop hits / Partials:** 8 / 33 / 9
- **Avg / median % per leg:** 1.07% / -1.43%
- **Sum % (uncompounded):** 53.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 1 | 7.7% | 1 | 12 | 0 | -1.48% | -19.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 1 | 7.7% | 1 | 12 | 0 | -1.48% | -19.2% |
| SELL (all) | 37 | 18 | 48.6% | 7 | 21 | 9 | 1.96% | 72.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 37 | 18 | 48.6% | 7 | 21 | 9 | 1.96% | 72.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 50 | 19 | 38.0% | 8 | 33 | 9 | 1.07% | 53.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 14:15:00 | 2058.05 | 2213.57 | 2214.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 2013.40 | 2179.51 | 2195.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 2183.90 | 2175.97 | 2192.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 11:00:00 | 2183.90 | 2175.97 | 2192.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 10:15:00 | 2162.65 | 2172.52 | 2189.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-11 14:45:00 | 2159.00 | 2172.67 | 2188.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-12 14:15:00 | 2227.35 | 2173.80 | 2188.53 | SL hit (close>static) qty=1.00 sl=2198.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 10:15:00 | 2297.65 | 2148.43 | 2147.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 09:15:00 | 2323.05 | 2157.10 | 2152.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 11:15:00 | 2586.00 | 2589.12 | 2481.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-22 12:00:00 | 2586.00 | 2589.12 | 2481.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 14:15:00 | 2507.00 | 2580.44 | 2485.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-24 14:45:00 | 2507.35 | 2580.44 | 2485.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 2487.20 | 2578.62 | 2485.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 09:45:00 | 2481.60 | 2578.62 | 2485.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 2465.65 | 2577.50 | 2485.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:45:00 | 2464.00 | 2577.50 | 2485.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 11:15:00 | 2497.00 | 2576.69 | 2485.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-29 14:30:00 | 2516.95 | 2557.24 | 2482.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-06 09:15:00 | 2768.64 | 2588.62 | 2510.39 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 14:15:00 | 2457.55 | 2657.36 | 2657.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-31 11:15:00 | 2441.60 | 2628.09 | 2642.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 09:15:00 | 2337.50 | 2308.13 | 2409.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-21 10:00:00 | 2337.50 | 2308.13 | 2409.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 2429.00 | 2311.07 | 2408.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 15:00:00 | 2429.00 | 2311.07 | 2408.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 2403.55 | 2311.99 | 2408.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-24 09:15:00 | 2344.20 | 2311.99 | 2408.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 15:15:00 | 2395.85 | 2318.94 | 2405.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 09:30:00 | 2364.20 | 2320.51 | 2405.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 10:15:00 | 2396.00 | 2320.51 | 2405.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 2387.15 | 2321.17 | 2405.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 12:15:00 | 2384.60 | 2321.88 | 2405.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 14:15:00 | 2411.65 | 2323.93 | 2405.40 | SL hit (close>static) qty=1.00 sl=2407.70 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 11:15:00 | 2571.30 | 2435.79 | 2435.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 09:15:00 | 2591.00 | 2441.87 | 2438.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 15:15:00 | 2457.00 | 2460.42 | 2448.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 15:15:00 | 2457.00 | 2460.42 | 2448.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 2457.00 | 2460.42 | 2448.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:15:00 | 2409.60 | 2460.42 | 2448.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 2417.40 | 2459.99 | 2448.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:15:00 | 2392.10 | 2459.99 | 2448.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 2386.60 | 2459.26 | 2448.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:30:00 | 2393.70 | 2459.26 | 2448.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 2453.40 | 2444.01 | 2441.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 12:15:00 | 2472.00 | 2444.29 | 2441.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 15:15:00 | 2385.00 | 2444.01 | 2441.59 | SL hit (close<static) qty=1.00 sl=2426.40 alert=retest2 |

### Cycle 5 — SELL (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 12:15:00 | 2358.60 | 2452.60 | 2453.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 13:15:00 | 2355.00 | 2451.63 | 2452.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 11:15:00 | 2363.40 | 2362.98 | 2395.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-02 12:00:00 | 2363.40 | 2362.98 | 2395.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 2408.20 | 2363.43 | 2395.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:00:00 | 2408.20 | 2363.43 | 2395.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 2382.90 | 2363.63 | 2395.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 09:30:00 | 2376.60 | 2364.12 | 2395.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:00:00 | 2370.40 | 2364.13 | 2395.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 10:15:00 | 2424.10 | 2365.07 | 2394.92 | SL hit (close>static) qty=1.00 sl=2410.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 12:15:00 | 2667.50 | 2418.81 | 2417.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 13:15:00 | 2694.50 | 2421.55 | 2419.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 10:15:00 | 2521.80 | 2545.65 | 2502.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-07 11:00:00 | 2521.80 | 2545.65 | 2502.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 2495.20 | 2543.90 | 2503.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:45:00 | 2495.00 | 2543.90 | 2503.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 2494.30 | 2543.41 | 2503.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:30:00 | 2494.80 | 2543.41 | 2503.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 2498.00 | 2542.96 | 2503.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:30:00 | 2494.00 | 2542.96 | 2503.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 2474.50 | 2522.29 | 2497.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:00:00 | 2474.50 | 2522.29 | 2497.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 2475.80 | 2521.83 | 2497.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 12:00:00 | 2475.80 | 2521.83 | 2497.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 2480.10 | 2519.26 | 2496.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:45:00 | 2478.70 | 2519.26 | 2496.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 2492.40 | 2516.60 | 2495.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 11:15:00 | 2514.90 | 2516.60 | 2495.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 09:45:00 | 2505.00 | 2515.47 | 2495.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 10:45:00 | 2497.50 | 2527.67 | 2505.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 12:15:00 | 2483.10 | 2526.80 | 2505.68 | SL hit (close<static) qty=1.00 sl=2484.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 2465.10 | 2520.80 | 2520.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 2442.20 | 2509.00 | 2514.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 10:15:00 | 2494.30 | 2490.63 | 2503.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-20 10:30:00 | 2491.00 | 2490.63 | 2503.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 2270.00 | 2203.89 | 2259.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 2270.00 | 2203.89 | 2259.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 2285.70 | 2204.70 | 2259.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 2285.70 | 2204.70 | 2259.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 2292.60 | 2205.58 | 2259.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:00:00 | 2292.60 | 2205.58 | 2259.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 2269.30 | 2211.99 | 2260.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 15:15:00 | 2250.00 | 2213.03 | 2260.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 10:15:00 | 2137.50 | 2211.49 | 2251.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 10:15:00 | 2155.00 | 2154.04 | 2205.01 | SL hit (close>ema200) qty=0.50 sl=2154.04 alert=retest2 |

### Cycle 8 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 2306.00 | 2127.83 | 2127.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 14:15:00 | 2314.00 | 2170.53 | 2150.56 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-06-11 14:45:00 | 2159.00 | 2024-06-12 14:15:00 | 2227.35 | STOP_HIT | 1.00 | -3.17% |
| SELL | retest2 | 2024-06-25 10:30:00 | 2158.95 | 2024-07-26 11:15:00 | 2051.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-01 15:15:00 | 2151.00 | 2024-07-29 09:15:00 | 2043.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-04 09:45:00 | 2155.15 | 2024-07-29 09:15:00 | 2047.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-11 10:30:00 | 2154.85 | 2024-07-29 09:15:00 | 2047.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-15 13:30:00 | 2151.05 | 2024-07-29 09:15:00 | 2043.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-15 14:15:00 | 2152.15 | 2024-07-29 09:15:00 | 2044.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-25 10:30:00 | 2158.95 | 2024-08-05 09:15:00 | 1943.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-01 15:15:00 | 2151.00 | 2024-08-05 09:15:00 | 1935.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-04 09:45:00 | 2155.15 | 2024-08-05 09:15:00 | 1939.64 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-11 10:30:00 | 2154.85 | 2024-08-05 09:15:00 | 1939.37 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-15 13:30:00 | 2151.05 | 2024-08-05 09:15:00 | 1935.95 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-15 14:15:00 | 2152.15 | 2024-08-05 09:15:00 | 1936.94 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-08-12 14:00:00 | 2153.35 | 2024-08-14 09:15:00 | 2206.00 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2024-10-29 14:30:00 | 2516.95 | 2024-11-06 09:15:00 | 2768.64 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-22 12:45:00 | 2514.60 | 2025-01-27 09:15:00 | 2410.75 | STOP_HIT | 1.00 | -4.13% |
| BUY | retest2 | 2025-01-22 14:45:00 | 2513.55 | 2025-01-27 09:15:00 | 2410.75 | STOP_HIT | 1.00 | -4.09% |
| BUY | retest2 | 2025-01-23 09:30:00 | 2531.75 | 2025-01-27 09:15:00 | 2410.75 | STOP_HIT | 1.00 | -4.78% |
| SELL | retest2 | 2025-03-24 09:15:00 | 2344.20 | 2025-03-26 14:15:00 | 2411.65 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-03-25 15:15:00 | 2395.85 | 2025-03-27 10:15:00 | 2427.30 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-03-26 09:30:00 | 2364.20 | 2025-03-27 14:15:00 | 2473.40 | STOP_HIT | 1.00 | -4.62% |
| SELL | retest2 | 2025-03-26 10:15:00 | 2396.00 | 2025-03-27 14:15:00 | 2473.40 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2025-03-26 12:15:00 | 2384.60 | 2025-03-27 14:15:00 | 2473.40 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest2 | 2025-03-27 09:30:00 | 2382.05 | 2025-03-27 14:15:00 | 2473.40 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2025-04-07 13:15:00 | 2381.40 | 2025-04-15 11:15:00 | 2422.70 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-04-07 14:30:00 | 2383.75 | 2025-04-15 11:15:00 | 2422.70 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-04-08 11:15:00 | 2374.90 | 2025-04-16 09:15:00 | 2458.00 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2025-04-08 12:30:00 | 2367.15 | 2025-04-16 09:15:00 | 2458.00 | STOP_HIT | 1.00 | -3.84% |
| BUY | retest2 | 2025-05-08 12:15:00 | 2472.00 | 2025-05-08 15:15:00 | 2385.00 | STOP_HIT | 1.00 | -3.52% |
| BUY | retest2 | 2025-05-12 09:15:00 | 2474.40 | 2025-05-23 11:15:00 | 2412.00 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-05-28 11:00:00 | 2483.70 | 2025-06-02 14:15:00 | 2415.00 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2025-05-28 12:15:00 | 2472.10 | 2025-06-02 14:15:00 | 2415.00 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-07-03 09:30:00 | 2376.60 | 2025-07-04 10:15:00 | 2424.10 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-07-03 12:00:00 | 2370.40 | 2025-07-04 10:15:00 | 2424.10 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2025-08-19 11:15:00 | 2514.90 | 2025-08-28 12:15:00 | 2483.10 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-08-20 09:45:00 | 2505.00 | 2025-08-28 12:15:00 | 2483.10 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-08-28 10:45:00 | 2497.50 | 2025-08-28 12:15:00 | 2483.10 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-08-29 11:15:00 | 2500.00 | 2025-08-29 12:15:00 | 2477.70 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-09-01 14:45:00 | 2521.50 | 2025-09-26 09:15:00 | 2484.70 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-01-08 15:15:00 | 2250.00 | 2026-01-19 10:15:00 | 2137.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 15:15:00 | 2250.00 | 2026-02-03 10:15:00 | 2155.00 | STOP_HIT | 0.50 | 4.22% |
| SELL | retest2 | 2026-02-26 11:30:00 | 2250.30 | 2026-02-26 14:15:00 | 2282.50 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2026-02-27 09:30:00 | 2245.70 | 2026-03-09 09:15:00 | 2133.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 09:30:00 | 2245.70 | 2026-03-09 13:15:00 | 2157.20 | STOP_HIT | 0.50 | 3.94% |
| SELL | retest2 | 2026-03-02 09:15:00 | 2205.30 | 2026-03-16 10:15:00 | 2095.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 2205.30 | 2026-03-20 09:15:00 | 1984.77 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-08 10:15:00 | 2067.20 | 2026-04-15 09:15:00 | 2116.20 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2026-04-08 11:45:00 | 2069.50 | 2026-04-15 09:15:00 | 2116.20 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2026-04-09 12:30:00 | 2070.50 | 2026-04-15 09:15:00 | 2116.20 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2026-04-10 12:45:00 | 2070.80 | 2026-04-15 09:15:00 | 2116.20 | STOP_HIT | 1.00 | -2.19% |
