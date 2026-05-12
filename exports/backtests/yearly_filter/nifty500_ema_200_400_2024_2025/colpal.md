# Colgate Palmolive (India) Ltd. (COLPAL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 2193.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 32 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 16 |
| PARTIAL | 8 |
| TARGET_HIT | 2 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 10
- **Target hits / Stop hits / Partials:** 2 / 16 / 8
- **Avg / median % per leg:** 2.54% / 2.76%
- **Sum % (uncompounded):** 65.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 26 | 16 | 61.5% | 2 | 16 | 8 | 2.54% | 65.9% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| SELL @ 3rd Alert (retest2) | 22 | 12 | 54.5% | 0 | 16 | 6 | 1.63% | 35.9% |
| retest1 (combined) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| retest2 (combined) | 22 | 12 | 54.5% | 0 | 16 | 6 | 1.63% | 35.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 13:15:00 | 3081.40 | 3459.18 | 3459.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 09:15:00 | 3063.50 | 3424.75 | 3441.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 3053.90 | 3021.92 | 3176.07 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:30:00 | 3036.70 | 3021.98 | 3175.33 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 13:00:00 | 3031.45 | 3022.18 | 3173.91 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-29 15:15:00 | 2884.86 | 3014.55 | 3162.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-02 09:15:00 | 2879.88 | 3013.00 | 3161.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-12-23 09:15:00 | 2733.03 | 2893.12 | 3027.77 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 2 — BUY (started 2025-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 12:15:00 | 2615.00 | 2566.36 | 2566.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 13:15:00 | 2642.10 | 2567.11 | 2566.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 09:15:00 | 2510.20 | 2594.45 | 2581.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 2510.20 | 2594.45 | 2581.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 2510.20 | 2594.45 | 2581.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:45:00 | 2516.90 | 2594.45 | 2581.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 2524.10 | 2593.75 | 2581.30 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 11:15:00 | 2499.50 | 2570.04 | 2570.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 11:15:00 | 2487.40 | 2565.44 | 2567.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 2441.00 | 2440.75 | 2481.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 10:00:00 | 2441.00 | 2440.75 | 2481.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 2475.70 | 2441.70 | 2478.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:45:00 | 2479.50 | 2441.70 | 2478.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 2480.70 | 2442.09 | 2478.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:30:00 | 2481.00 | 2442.09 | 2478.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 2479.20 | 2442.46 | 2478.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:30:00 | 2486.80 | 2442.46 | 2478.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 2469.70 | 2442.73 | 2478.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:45:00 | 2475.50 | 2442.73 | 2478.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 2337.60 | 2276.04 | 2342.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:30:00 | 2335.50 | 2276.04 | 2342.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 2348.50 | 2276.76 | 2342.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:45:00 | 2351.50 | 2276.76 | 2342.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 2355.50 | 2277.55 | 2342.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 12:30:00 | 2351.80 | 2277.55 | 2342.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 2353.40 | 2282.56 | 2343.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:45:00 | 2355.00 | 2282.56 | 2343.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 2350.50 | 2283.24 | 2343.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:45:00 | 2352.20 | 2283.24 | 2343.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 2301.40 | 2286.07 | 2336.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 2239.50 | 2341.03 | 2350.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 14:30:00 | 2245.60 | 2270.15 | 2300.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 2213.90 | 2270.99 | 2299.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:30:00 | 2245.90 | 2261.16 | 2290.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 09:15:00 | 2127.53 | 2191.94 | 2227.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 09:15:00 | 2133.32 | 2191.94 | 2227.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 09:15:00 | 2133.61 | 2191.94 | 2227.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:15:00 | 2103.20 | 2190.34 | 2226.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 2184.00 | 2172.18 | 2206.48 | SL hit (close>ema200) qty=0.50 sl=2172.18 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 12:15:00 | 2216.00 | 2146.91 | 2146.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 13:15:00 | 2223.70 | 2147.67 | 2147.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-05 10:15:00 | 2171.70 | 2176.15 | 2163.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-05 11:00:00 | 2171.70 | 2176.15 | 2163.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 2158.00 | 2178.65 | 2165.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:45:00 | 2154.20 | 2178.65 | 2165.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 2145.00 | 2178.32 | 2165.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 11:00:00 | 2145.00 | 2178.32 | 2165.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 2112.00 | 2177.85 | 2165.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:00:00 | 2112.00 | 2177.85 | 2165.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 2113.70 | 2177.21 | 2165.53 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 1977.80 | 2153.87 | 2154.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 1966.30 | 2152.00 | 2153.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 10:15:00 | 1962.50 | 1956.38 | 2022.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-16 10:30:00 | 1963.20 | 1956.38 | 2022.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 2098.80 | 1958.71 | 2021.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 10:00:00 | 2098.80 | 1958.71 | 2021.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 10:15:00 | 2105.50 | 1960.17 | 2021.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 10:30:00 | 2110.60 | 1960.17 | 2021.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 2174.50 | 2061.03 | 2060.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 12:15:00 | 2177.70 | 2069.94 | 2065.42 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-11-28 10:30:00 | 3036.70 | 2024-11-29 15:15:00 | 2884.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-11-28 13:00:00 | 3031.45 | 2024-12-02 09:15:00 | 2879.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-11-28 10:30:00 | 3036.70 | 2024-12-23 09:15:00 | 2733.03 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2024-11-28 13:00:00 | 3031.45 | 2024-12-23 09:15:00 | 2728.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-09 13:00:00 | 2888.90 | 2025-01-14 10:15:00 | 2744.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 13:00:00 | 2888.90 | 2025-01-22 15:15:00 | 2774.50 | STOP_HIT | 0.50 | 3.96% |
| SELL | retest2 | 2025-02-01 14:15:00 | 2897.10 | 2025-02-03 12:15:00 | 2752.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-01 14:15:00 | 2897.10 | 2025-02-03 12:15:00 | 2779.80 | STOP_HIT | 0.50 | 4.05% |
| SELL | retest2 | 2025-09-26 09:15:00 | 2239.50 | 2025-12-03 09:15:00 | 2127.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-20 14:30:00 | 2245.60 | 2025-12-03 09:15:00 | 2133.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-24 09:15:00 | 2213.90 | 2025-12-03 09:15:00 | 2133.61 | PARTIAL | 0.50 | 3.63% |
| SELL | retest2 | 2025-10-30 09:30:00 | 2245.90 | 2025-12-03 11:15:00 | 2103.20 | PARTIAL | 0.50 | 6.35% |
| SELL | retest2 | 2025-09-26 09:15:00 | 2239.50 | 2025-12-16 09:15:00 | 2184.00 | STOP_HIT | 0.50 | 2.48% |
| SELL | retest2 | 2025-10-20 14:30:00 | 2245.60 | 2025-12-16 09:15:00 | 2184.00 | STOP_HIT | 0.50 | 2.74% |
| SELL | retest2 | 2025-10-24 09:15:00 | 2213.90 | 2025-12-16 09:15:00 | 2184.00 | STOP_HIT | 0.50 | 1.35% |
| SELL | retest2 | 2025-10-30 09:30:00 | 2245.90 | 2025-12-16 09:15:00 | 2184.00 | STOP_HIT | 0.50 | 2.76% |
| SELL | retest2 | 2026-01-27 12:00:00 | 2130.40 | 2026-01-27 12:15:00 | 2151.20 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2026-01-28 10:00:00 | 2127.30 | 2026-01-28 14:15:00 | 2155.50 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2026-01-28 11:30:00 | 2127.70 | 2026-01-28 14:15:00 | 2155.50 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-01-28 12:30:00 | 2129.90 | 2026-01-28 14:15:00 | 2155.50 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-01-29 09:15:00 | 2134.00 | 2026-02-04 09:15:00 | 2159.80 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-02-04 10:15:00 | 2148.00 | 2026-02-10 13:15:00 | 2165.70 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-02-09 11:00:00 | 2145.10 | 2026-02-10 13:15:00 | 2165.70 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2026-02-10 09:15:00 | 2147.00 | 2026-02-10 13:15:00 | 2165.70 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2026-02-13 09:15:00 | 2123.60 | 2026-02-17 10:15:00 | 2157.80 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-02-17 10:15:00 | 2134.00 | 2026-02-17 10:15:00 | 2157.80 | STOP_HIT | 1.00 | -1.12% |
