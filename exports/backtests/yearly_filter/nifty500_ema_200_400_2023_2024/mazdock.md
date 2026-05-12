# Mazagoan Dock Shipbuilders Ltd. (MAZDOCK)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 2656.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT2_SKIP | 4 |
| ALERT3 | 39 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 42 |
| PARTIAL | 11 |
| TARGET_HIT | 13 |
| STOP_HIT | 29 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 53 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 31 / 22
- **Target hits / Stop hits / Partials:** 13 / 29 / 11
- **Avg / median % per leg:** 2.90% / 2.10%
- **Sum % (uncompounded):** 153.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 9 | 34.6% | 9 | 17 | 0 | 2.34% | 60.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 26 | 9 | 34.6% | 9 | 17 | 0 | 2.34% | 60.8% |
| SELL (all) | 27 | 22 | 81.5% | 4 | 12 | 11 | 3.45% | 93.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 27 | 22 | 81.5% | 4 | 12 | 11 | 3.45% | 93.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 53 | 31 | 58.5% | 13 | 29 | 11 | 2.90% | 153.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 12:15:00 | 1059.93 | 1080.94 | 1081.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 14:15:00 | 1056.10 | 1080.60 | 1080.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 12:15:00 | 1004.68 | 995.13 | 1027.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-01 13:00:00 | 1004.68 | 995.13 | 1027.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 1077.50 | 996.37 | 1026.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-03 10:00:00 | 1077.50 | 996.37 | 1026.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 10:15:00 | 1106.78 | 997.47 | 1026.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-03 11:00:00 | 1106.78 | 997.47 | 1026.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-04-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 11:15:00 | 1100.35 | 1047.64 | 1047.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 09:15:00 | 1122.55 | 1055.46 | 1051.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 11:15:00 | 1101.90 | 1110.62 | 1084.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-07 11:30:00 | 1102.22 | 1110.62 | 1084.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 12:15:00 | 1083.30 | 1110.35 | 1084.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 12:45:00 | 1084.75 | 1110.35 | 1084.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 13:15:00 | 1096.38 | 1110.21 | 1084.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-08 09:45:00 | 1106.55 | 1109.75 | 1085.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 13:15:00 | 1101.30 | 1110.83 | 1087.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 15:15:00 | 1100.00 | 1110.60 | 1087.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-13 09:15:00 | 1069.80 | 1110.09 | 1087.79 | SL hit (close<static) qty=1.00 sl=1081.60 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 13:15:00 | 2102.00 | 2176.91 | 2177.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 14:15:00 | 2099.00 | 2176.13 | 2176.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-10 10:15:00 | 2192.57 | 2132.01 | 2152.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 10:15:00 | 2192.57 | 2132.01 | 2152.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 2192.57 | 2132.01 | 2152.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-10 11:00:00 | 2192.57 | 2132.01 | 2152.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 2233.00 | 2133.02 | 2152.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-10 12:00:00 | 2233.00 | 2133.02 | 2152.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 13:15:00 | 2154.10 | 2141.37 | 2155.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 13:30:00 | 2155.75 | 2141.37 | 2155.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 14:15:00 | 2141.60 | 2141.37 | 2155.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 14:30:00 | 2142.90 | 2141.37 | 2155.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 2186.50 | 2141.81 | 2155.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:00:00 | 2186.50 | 2141.81 | 2155.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 2280.73 | 2143.19 | 2156.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:00:00 | 2280.73 | 2143.19 | 2156.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 2164.88 | 2147.62 | 2158.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 10:45:00 | 2165.30 | 2147.62 | 2158.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 2161.73 | 2147.76 | 2158.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 11:15:00 | 2132.20 | 2148.32 | 2158.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 09:15:00 | 2197.75 | 2147.95 | 2157.62 | SL hit (close>static) qty=1.00 sl=2165.50 alert=retest2 |

### Cycle 4 — BUY (started 2024-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 14:15:00 | 2337.53 | 2166.67 | 2166.67 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-10-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 14:15:00 | 2065.00 | 2166.14 | 2166.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 2036.45 | 2157.50 | 2161.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 2130.45 | 2108.38 | 2133.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 2130.45 | 2108.38 | 2133.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 2130.45 | 2108.38 | 2133.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 13:45:00 | 2142.50 | 2108.38 | 2133.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 2098.93 | 2108.29 | 2133.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 14:00:00 | 2092.78 | 2112.37 | 2133.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 15:00:00 | 2088.50 | 2112.14 | 2132.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 1988.14 | 2102.27 | 2126.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 1984.07 | 2102.27 | 2126.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-25 10:15:00 | 2078.50 | 2066.79 | 2101.66 | SL hit (close>ema200) qty=0.50 sl=2066.79 alert=retest2 |

### Cycle 6 — BUY (started 2024-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 13:15:00 | 2326.60 | 2128.60 | 2128.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 09:15:00 | 2394.48 | 2135.61 | 2131.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-23 09:15:00 | 2320.00 | 2342.63 | 2258.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-23 09:45:00 | 2314.98 | 2342.63 | 2258.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 2261.25 | 2341.76 | 2269.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:30:00 | 2255.55 | 2341.76 | 2269.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 2258.30 | 2340.93 | 2269.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 15:00:00 | 2274.75 | 2337.54 | 2268.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 09:15:00 | 2279.70 | 2336.80 | 2268.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-31 09:15:00 | 2221.15 | 2335.65 | 2268.65 | SL hit (close<static) qty=1.00 sl=2250.50 alert=retest2 |

### Cycle 7 — SELL (started 2025-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 09:15:00 | 2088.00 | 2258.81 | 2259.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 09:15:00 | 2019.25 | 2246.78 | 2253.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 2227.30 | 2189.52 | 2218.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 09:15:00 | 2227.30 | 2189.52 | 2218.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 2227.30 | 2189.52 | 2218.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 2227.30 | 2189.52 | 2218.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 2197.25 | 2189.60 | 2218.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:30:00 | 2192.15 | 2189.52 | 2217.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 10:00:00 | 2182.50 | 2188.67 | 2216.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-06 10:15:00 | 2258.95 | 2192.12 | 2217.36 | SL hit (close>static) qty=1.00 sl=2257.65 alert=retest2 |

### Cycle 8 — BUY (started 2025-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 14:15:00 | 2376.05 | 2237.26 | 2236.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 2438.85 | 2240.64 | 2238.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 2282.00 | 2457.72 | 2370.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 2282.00 | 2457.72 | 2370.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 2282.00 | 2457.72 | 2370.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 09:15:00 | 2424.55 | 2449.15 | 2368.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 12:30:00 | 2375.10 | 2446.24 | 2369.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 14:30:00 | 2375.40 | 2444.67 | 2368.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 13:00:00 | 2374.00 | 2439.13 | 2368.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 13:15:00 | 2376.00 | 2438.50 | 2368.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 13:45:00 | 2364.40 | 2438.50 | 2368.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 14:15:00 | 2375.00 | 2437.87 | 2368.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 15:15:00 | 2373.90 | 2437.87 | 2368.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 15:15:00 | 2373.90 | 2437.24 | 2368.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 09:15:00 | 2427.00 | 2437.24 | 2368.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-15 09:15:00 | 2612.61 | 2439.38 | 2371.95 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 11:15:00 | 2662.90 | 3092.41 | 3094.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 2639.00 | 2942.30 | 3007.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 14:15:00 | 2758.30 | 2757.26 | 2850.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 15:00:00 | 2758.30 | 2757.26 | 2850.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 2886.10 | 2760.73 | 2847.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:00:00 | 2886.10 | 2760.73 | 2847.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 2898.20 | 2762.10 | 2848.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 2879.40 | 2860.81 | 2880.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 10:00:00 | 2863.90 | 2860.84 | 2880.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 12:15:00 | 2735.43 | 2857.07 | 2877.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 14:15:00 | 2720.70 | 2848.55 | 2872.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 2848.00 | 2846.75 | 2870.76 | SL hit (close>ema200) qty=0.50 sl=2846.75 alert=retest2 |

### Cycle 10 — BUY (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 10:15:00 | 2670.70 | 2408.86 | 2407.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 10:15:00 | 2688.00 | 2426.39 | 2416.55 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-08 09:45:00 | 1106.55 | 2024-05-13 09:15:00 | 1069.80 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2024-05-10 13:15:00 | 1101.30 | 2024-05-13 09:15:00 | 1069.80 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2024-05-10 15:15:00 | 1100.00 | 2024-05-13 09:15:00 | 1069.80 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2024-05-14 09:30:00 | 1107.72 | 2024-05-16 09:15:00 | 1218.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-05 11:15:00 | 1439.45 | 2024-06-06 09:15:00 | 1583.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-05 13:30:00 | 1394.50 | 2024-06-06 09:15:00 | 1533.95 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-17 11:15:00 | 2132.20 | 2024-10-18 09:15:00 | 2197.75 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2024-11-08 14:00:00 | 2092.78 | 2024-11-13 09:15:00 | 1988.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 15:00:00 | 2088.50 | 2024-11-13 09:15:00 | 1984.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 14:00:00 | 2092.78 | 2024-11-25 10:15:00 | 2078.50 | STOP_HIT | 0.50 | 0.68% |
| SELL | retest2 | 2024-11-08 15:00:00 | 2088.50 | 2024-11-25 10:15:00 | 2078.50 | STOP_HIT | 0.50 | 0.48% |
| BUY | retest2 | 2024-12-30 15:00:00 | 2274.75 | 2024-12-31 09:15:00 | 2221.15 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2024-12-31 09:15:00 | 2279.70 | 2024-12-31 09:15:00 | 2221.15 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2025-01-01 14:15:00 | 2271.55 | 2025-01-01 14:15:00 | 2242.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-01-02 10:00:00 | 2269.90 | 2025-01-03 12:15:00 | 2249.00 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-01-02 12:30:00 | 2262.85 | 2025-01-03 12:15:00 | 2249.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-01-03 09:15:00 | 2273.00 | 2025-01-03 12:15:00 | 2249.00 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-01-03 11:45:00 | 2263.10 | 2025-01-03 12:15:00 | 2249.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-01-15 09:30:00 | 2296.55 | 2025-01-15 13:15:00 | 2243.00 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-01-16 09:15:00 | 2257.00 | 2025-01-20 11:15:00 | 2482.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-22 12:15:00 | 2235.00 | 2025-01-27 15:15:00 | 2212.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-01-27 09:30:00 | 2232.20 | 2025-01-27 15:15:00 | 2212.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-01-27 11:15:00 | 2233.00 | 2025-01-27 15:15:00 | 2212.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-02-10 09:15:00 | 2259.40 | 2025-02-10 14:15:00 | 2207.55 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-02-10 10:30:00 | 2243.65 | 2025-02-10 14:15:00 | 2207.55 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-02-13 10:15:00 | 2244.15 | 2025-02-13 14:15:00 | 2205.00 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-03-04 11:30:00 | 2192.15 | 2025-03-06 10:15:00 | 2258.95 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-03-05 10:00:00 | 2182.50 | 2025-03-06 10:15:00 | 2258.95 | STOP_HIT | 1.00 | -3.50% |
| BUY | retest2 | 2025-04-08 09:15:00 | 2424.55 | 2025-04-15 09:15:00 | 2612.61 | TARGET_HIT | 1.00 | 7.76% |
| BUY | retest2 | 2025-04-08 12:30:00 | 2375.10 | 2025-04-15 09:15:00 | 2612.94 | TARGET_HIT | 1.00 | 10.01% |
| BUY | retest2 | 2025-04-08 14:30:00 | 2375.40 | 2025-04-15 09:15:00 | 2611.40 | TARGET_HIT | 1.00 | 9.94% |
| BUY | retest2 | 2025-04-09 13:00:00 | 2374.00 | 2025-04-15 12:15:00 | 2667.01 | TARGET_HIT | 1.00 | 12.34% |
| BUY | retest2 | 2025-04-11 09:15:00 | 2427.00 | 2025-04-15 12:15:00 | 2669.70 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-09-26 09:15:00 | 2879.40 | 2025-09-29 12:15:00 | 2735.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-26 10:00:00 | 2863.90 | 2025-09-30 14:15:00 | 2720.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-26 09:15:00 | 2879.40 | 2025-10-01 15:15:00 | 2848.00 | STOP_HIT | 0.50 | 1.09% |
| SELL | retest2 | 2025-09-26 10:00:00 | 2863.90 | 2025-10-01 15:15:00 | 2848.00 | STOP_HIT | 0.50 | 0.56% |
| SELL | retest2 | 2025-10-06 09:30:00 | 2882.00 | 2025-10-09 15:15:00 | 2898.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-10-06 14:15:00 | 2885.60 | 2025-10-09 15:15:00 | 2898.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-10-07 15:00:00 | 2875.90 | 2025-10-29 09:15:00 | 2737.90 | PARTIAL | 0.50 | 4.80% |
| SELL | retest2 | 2025-10-08 09:15:00 | 2868.00 | 2025-10-29 09:15:00 | 2741.32 | PARTIAL | 0.50 | 4.42% |
| SELL | retest2 | 2025-10-10 13:15:00 | 2875.00 | 2025-10-29 09:15:00 | 2731.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-10 14:30:00 | 2874.20 | 2025-10-29 09:15:00 | 2730.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-13 09:15:00 | 2839.50 | 2025-11-03 09:15:00 | 2711.87 | PARTIAL | 0.50 | 4.49% |
| SELL | retest2 | 2025-10-16 10:00:00 | 2854.60 | 2025-11-03 09:15:00 | 2712.25 | PARTIAL | 0.50 | 4.99% |
| SELL | retest2 | 2025-10-28 09:45:00 | 2855.00 | 2025-11-04 12:15:00 | 2697.53 | PARTIAL | 0.50 | 5.52% |
| SELL | retest2 | 2025-10-07 15:00:00 | 2875.90 | 2025-11-07 09:15:00 | 2593.80 | TARGET_HIT | 0.50 | 9.81% |
| SELL | retest2 | 2025-10-08 09:15:00 | 2868.00 | 2025-11-07 09:15:00 | 2597.04 | TARGET_HIT | 0.50 | 9.45% |
| SELL | retest2 | 2025-10-10 13:15:00 | 2875.00 | 2025-11-07 09:15:00 | 2587.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-10 14:30:00 | 2874.20 | 2025-11-07 09:15:00 | 2586.78 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-13 09:15:00 | 2839.50 | 2025-11-12 09:15:00 | 2779.80 | STOP_HIT | 0.50 | 2.10% |
| SELL | retest2 | 2025-10-16 10:00:00 | 2854.60 | 2025-11-12 09:15:00 | 2779.80 | STOP_HIT | 0.50 | 2.62% |
| SELL | retest2 | 2025-10-28 09:45:00 | 2855.00 | 2025-11-12 09:15:00 | 2779.80 | STOP_HIT | 0.50 | 2.63% |
