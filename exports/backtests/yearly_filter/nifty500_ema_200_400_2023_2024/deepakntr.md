# Deepak Nitrite Ltd. (DEEPAKNTR)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 1875.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 1 |
| ALERT3 | 43 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 34 |
| PARTIAL | 2 |
| TARGET_HIT | 3 |
| STOP_HIT | 31 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 31
- **Target hits / Stop hits / Partials:** 3 / 31 / 2
- **Avg / median % per leg:** -0.34% / -1.51%
- **Sum % (uncompounded):** -12.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 1 | 3.8% | 1 | 25 | 0 | -1.23% | -32.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 26 | 1 | 3.8% | 1 | 25 | 0 | -1.23% | -32.0% |
| SELL (all) | 10 | 4 | 40.0% | 2 | 6 | 2 | 1.96% | 19.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 4 | 40.0% | 2 | 6 | 2 | 1.96% | 19.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 36 | 5 | 13.9% | 3 | 31 | 2 | -0.34% | -12.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 15:15:00 | 1993.00 | 2027.90 | 2028.07 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 10:15:00 | 2144.80 | 2029.32 | 2028.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 11:15:00 | 2149.95 | 2030.52 | 2029.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-21 14:15:00 | 2183.65 | 2186.70 | 2129.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-21 15:00:00 | 2183.65 | 2186.70 | 2129.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 14:15:00 | 2124.95 | 2184.27 | 2130.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-22 15:00:00 | 2124.95 | 2184.27 | 2130.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 15:15:00 | 2126.75 | 2183.70 | 2130.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-25 09:15:00 | 2132.70 | 2183.70 | 2130.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 11:15:00 | 2124.90 | 2179.67 | 2131.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 12:00:00 | 2124.90 | 2179.67 | 2131.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 12:15:00 | 2121.10 | 2179.09 | 2131.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 12:45:00 | 2113.00 | 2179.09 | 2131.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 09:15:00 | 2131.70 | 2173.93 | 2131.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 10:00:00 | 2131.70 | 2173.93 | 2131.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 10:15:00 | 2126.50 | 2173.46 | 2131.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 11:00:00 | 2126.50 | 2173.46 | 2131.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 11:15:00 | 2120.35 | 2172.93 | 2130.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 12:00:00 | 2120.35 | 2172.93 | 2130.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 13:15:00 | 2126.95 | 2168.25 | 2130.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-29 14:00:00 | 2126.95 | 2168.25 | 2130.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 14:15:00 | 2120.00 | 2167.77 | 2130.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-29 15:00:00 | 2120.00 | 2167.77 | 2130.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 14:15:00 | 2119.40 | 2164.52 | 2130.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-17 09:15:00 | 2155.00 | 2129.30 | 2119.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-17 11:00:00 | 2123.75 | 2129.25 | 2119.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-17 12:00:00 | 2125.10 | 2129.21 | 2119.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-18 09:30:00 | 2126.35 | 2128.99 | 2119.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 10:15:00 | 2095.00 | 2128.65 | 2119.22 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-10-18 10:15:00 | 2095.00 | 2128.65 | 2119.22 | SL hit (close<static) qty=1.00 sl=2113.35 alert=retest2 |

### Cycle 3 — SELL (started 2023-10-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 12:15:00 | 1962.35 | 2111.22 | 2111.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-26 09:15:00 | 1943.90 | 2105.83 | 2108.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 09:15:00 | 2074.00 | 2060.41 | 2082.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-06 09:15:00 | 2074.00 | 2060.41 | 2082.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 09:15:00 | 2074.00 | 2060.41 | 2082.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 10:00:00 | 2074.00 | 2060.41 | 2082.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 10:15:00 | 2079.15 | 2060.60 | 2082.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-09 12:30:00 | 2072.15 | 2072.94 | 2086.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-09 14:15:00 | 2073.15 | 2072.97 | 2086.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-10 10:15:00 | 2104.00 | 2073.53 | 2086.50 | SL hit (close>static) qty=1.00 sl=2096.95 alert=retest2 |

### Cycle 4 — BUY (started 2023-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-22 12:15:00 | 2129.10 | 2096.36 | 2096.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-22 15:15:00 | 2136.00 | 2097.40 | 2096.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 11:15:00 | 2370.85 | 2375.96 | 2297.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-17 12:00:00 | 2370.85 | 2375.96 | 2297.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 2286.10 | 2373.85 | 2298.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-18 09:45:00 | 2271.00 | 2373.85 | 2298.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 10:15:00 | 2311.45 | 2373.23 | 2298.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-19 09:15:00 | 2340.00 | 2369.46 | 2298.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-23 09:30:00 | 2315.45 | 2365.40 | 2301.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-23 11:15:00 | 2274.75 | 2363.78 | 2301.02 | SL hit (close<static) qty=1.00 sl=2281.20 alert=retest2 |

### Cycle 5 — SELL (started 2024-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-02 11:15:00 | 2218.00 | 2275.84 | 2276.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-04 09:15:00 | 2211.50 | 2274.63 | 2275.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-02 10:15:00 | 2186.70 | 2177.31 | 2212.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-02 10:45:00 | 2191.50 | 2177.31 | 2212.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 12:15:00 | 2200.00 | 2177.72 | 2211.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-03 09:15:00 | 2180.65 | 2178.41 | 2211.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-03 10:00:00 | 2190.00 | 2178.52 | 2211.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-03 12:30:00 | 2190.95 | 2178.95 | 2211.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-04 09:30:00 | 2186.10 | 2179.23 | 2210.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 11:15:00 | 2202.05 | 2179.35 | 2210.65 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-04-04 12:15:00 | 2227.20 | 2179.83 | 2210.73 | SL hit (close>static) qty=1.00 sl=2214.05 alert=retest2 |

### Cycle 6 — BUY (started 2024-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 12:15:00 | 2316.85 | 2230.66 | 2230.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 09:15:00 | 2328.05 | 2233.76 | 2231.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 09:15:00 | 2387.00 | 2413.17 | 2351.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-23 10:00:00 | 2387.00 | 2413.17 | 2351.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 2348.90 | 2411.93 | 2351.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 10:45:00 | 2361.30 | 2407.88 | 2351.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 11:45:00 | 2360.00 | 2407.40 | 2351.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 10:00:00 | 2356.90 | 2405.38 | 2351.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 11:15:00 | 2338.15 | 2404.08 | 2351.83 | SL hit (close<static) qty=1.00 sl=2340.20 alert=retest2 |

### Cycle 7 — SELL (started 2024-10-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 13:15:00 | 2654.20 | 2830.22 | 2830.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 12:15:00 | 2632.05 | 2797.76 | 2813.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 11:15:00 | 2774.60 | 2771.48 | 2797.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-06 12:00:00 | 2774.60 | 2771.48 | 2797.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 12:15:00 | 2801.40 | 2771.78 | 2797.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 13:00:00 | 2801.40 | 2771.78 | 2797.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 13:15:00 | 2822.55 | 2772.28 | 2798.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 13:15:00 | 2801.00 | 2776.11 | 2799.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 09:15:00 | 2798.50 | 2777.18 | 2799.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:15:00 | 2660.95 | 2771.51 | 2795.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:15:00 | 2658.57 | 2771.51 | 2795.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-13 12:15:00 | 2520.90 | 2742.24 | 2778.51 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 8 — BUY (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 12:15:00 | 1682.80 | 1536.78 | 1536.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 14:15:00 | 1684.00 | 1539.68 | 1537.91 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-08-01 10:00:00 | 2022.85 | 2023-08-02 13:15:00 | 1996.30 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2023-08-01 10:45:00 | 2018.70 | 2023-08-02 13:15:00 | 1996.30 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2023-08-03 13:15:00 | 2017.65 | 2023-08-17 12:15:00 | 2007.05 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2023-08-03 14:45:00 | 2024.00 | 2023-08-17 12:15:00 | 2007.05 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2023-08-04 11:00:00 | 2032.05 | 2023-08-23 15:15:00 | 1993.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2023-08-04 12:45:00 | 2032.85 | 2023-08-23 15:15:00 | 1993.00 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2023-08-04 14:00:00 | 2033.85 | 2023-08-23 15:15:00 | 1993.00 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2023-08-07 11:00:00 | 2033.35 | 2023-08-23 15:15:00 | 1993.00 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2023-10-17 09:15:00 | 2155.00 | 2023-10-18 10:15:00 | 2095.00 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2023-10-17 11:00:00 | 2123.75 | 2023-10-18 10:15:00 | 2095.00 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2023-10-17 12:00:00 | 2125.10 | 2023-10-18 10:15:00 | 2095.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2023-10-18 09:30:00 | 2126.35 | 2023-10-18 10:15:00 | 2095.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2023-11-09 12:30:00 | 2072.15 | 2023-11-10 10:15:00 | 2104.00 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2023-11-09 14:15:00 | 2073.15 | 2023-11-10 10:15:00 | 2104.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-01-19 09:15:00 | 2340.00 | 2024-01-23 11:15:00 | 2274.75 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2024-01-23 09:30:00 | 2315.45 | 2024-01-23 11:15:00 | 2274.75 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-02-01 09:15:00 | 2323.75 | 2024-02-01 11:15:00 | 2249.00 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest2 | 2024-02-15 10:30:00 | 2320.00 | 2024-02-15 13:15:00 | 2278.95 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-02-16 10:15:00 | 2308.00 | 2024-02-27 10:15:00 | 2271.60 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-02-16 11:30:00 | 2306.40 | 2024-02-27 10:15:00 | 2271.60 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-02-20 09:15:00 | 2318.75 | 2024-02-27 10:15:00 | 2271.60 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-02-20 13:30:00 | 2340.70 | 2024-02-27 10:15:00 | 2271.60 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2024-02-27 09:15:00 | 2295.35 | 2024-02-27 10:15:00 | 2271.60 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-04-03 09:15:00 | 2180.65 | 2024-04-04 12:15:00 | 2227.20 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-04-03 10:00:00 | 2190.00 | 2024-04-04 12:15:00 | 2227.20 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-04-03 12:30:00 | 2190.95 | 2024-04-04 12:15:00 | 2227.20 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-04-04 09:30:00 | 2186.10 | 2024-04-04 12:15:00 | 2227.20 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2024-05-24 10:45:00 | 2361.30 | 2024-05-27 11:15:00 | 2338.15 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-05-24 11:45:00 | 2360.00 | 2024-05-27 11:15:00 | 2338.15 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-05-27 10:00:00 | 2356.90 | 2024-05-27 11:15:00 | 2338.15 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-05-27 14:00:00 | 2356.55 | 2024-05-28 09:15:00 | 2310.00 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-06-11 09:15:00 | 2308.00 | 2024-06-20 09:15:00 | 2538.80 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-07 13:15:00 | 2801.00 | 2024-11-11 09:15:00 | 2660.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 09:15:00 | 2798.50 | 2024-11-11 09:15:00 | 2658.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 13:15:00 | 2801.00 | 2024-11-13 12:15:00 | 2520.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-08 09:15:00 | 2798.50 | 2024-11-13 12:15:00 | 2518.65 | TARGET_HIT | 0.50 | 10.00% |
